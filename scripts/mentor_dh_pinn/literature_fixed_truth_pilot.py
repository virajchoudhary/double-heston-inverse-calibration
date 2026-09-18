"""Separate fixed-literature-truth inverse PINN pilot; frozen V2 is untouched.

prepare: teacher generation/audit only; train: quotes/PDE only; score: truth access.
No exact-pricer calls or parameter labels enter the optimization loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.stats import qmc
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual

torch.set_default_dtype(torch.float64)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feller(p):
    p = np.asarray(p).reshape(2, 5)
    return 2 * p[:, 0] * p[:, 1] - p[:, 2] ** 2


def decode(raw):
    """Shared broad domain; strict slow<fast, positivity, correlation and Feller.

    These new pilot bounds do not modify the old V2 domain. Published truths
    are interior; no truth values or truth-centered priors are used here.
    """
    u = torch.sigmoid(raw)
    ks = .02 * torch.exp(u[0] * math.log(5 / .02))
    kf = ks + .01 + (20 - ks - .01) * u[5]
    result = []
    for i, k in ((0, ks), (5, kf)):
        theta = .001 * torch.exp(u[i + 1] * math.log(1 / .001))
        sigma = (.01 + .989 * u[i + 2]) * torch.sqrt(2 * k * theta)
        rho = .999 * torch.tanh(raw[i + 3])
        v0 = .0001 * torch.exp(u[i + 4] * math.log(1 / .0001))
        result.extend([k, theta, sigma, rho, v0])
    return torch.stack(result)


def initial_raw(seed):
    # Generic starting point, fixed before recovery results. Same for all cases.
    u = np.array([.55, .60, .50, .50, .75, .20, .60, .50, .50, .75])
    raw = np.log(u / (1 - u))
    raw[[3, 8]] = np.arctanh(-.25 / .999)
    raw += np.random.default_rng(seed).normal(0, .2, 10)
    return torch.tensor(raw)


def data_coords(xt, p):
    coords = torch.stack([xt[:, 0], p[4].expand(len(xt)),
                          p[9].expand(len(xt)), xt[:, 1]], -1)
    structural = p.reshape(2, 5)[:, :4]
    return coords, structural


@torch.no_grad()
def teacher(p, xt, nodes):
    # Local import: this function is used only in prepare/score, never train.
    from src.mentor_dh_pinn.torch_pricer import price_call
    xt = torch.as_tensor(xt)
    return price_call(torch.as_tensor(p), torch.exp(xt[:, 0]),
                      torch.ones(len(xt)), xt[:, 1], torch.tensor(0.),
                      torch.tensor(0.), node_count=nodes).numpy()


def prepare(config, out):
    from src.mentor_dh_pinn.regular_pinn_data import invert_total_variance
    cfg = json.loads(config.read_text())
    pr = cfg["protocol"]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "truth_and_sources.json", cfg)
    write_json(out / "protocol.json", pr)
    # Time is in ACT/365 years, not integers labelling expiry months.
    tau = np.asarray(pr["expiry_days"]) / 365
    ratios = np.linspace(*pr["spot_strike_ratio"], pr["training_ratios"])
    holdout = (ratios[:-1] + ratios[1:]) / 2
    d = pr["collocation_domain"]
    u = qmc.LatinHypercube(4, seed=pr["collocation_seed"]).random(pr["collocation_count"])
    lo = np.array([d["x_forward"][0], *[d["each_variance"][0]] * 2, d["tau_years"][0]])
    hi = np.array([d["x_forward"][1], *[d["each_variance"][1]] * 2, d["tau_years"][1]])
    np.savez(out / "collocation.npz", coords=lo + u * (hi - lo))
    audit = {"synthetic_only": True, "no_parameter_labels_in_train_files": True,
             "cases": {}, "excluded": cfg["excluded"]}
    for case in cfg["cases"]:
        p = np.asarray(case["parameters"])
        if not np.all(feller(p) > 0):
            raise ValueError("Feller-incompatible training truth")
        folder = out / case["id"]
        folder.mkdir()
        checks = {"feller_margins": feller(p).tolist()}
        for split, grid in (("train", ratios), ("holdout", holdout)):
            rr, tt = np.meshgrid(grid, tau)
            xt = np.stack([np.log(rr.ravel()) + pr["rate"] * tt.ravel(), tt.ravel()], -1)
            price = teacher(p, xt, 128)
            coarse = teacher(p, xt, 96)
            delta = float(np.max(np.abs(price - coarse)))
            if not np.isfinite(price).all() or delta > 1e-7:
                raise ValueError(f"Teacher convergence failure: {case['id']} {split} {delta}")
            iv = np.sqrt(invert_total_variance(price, xt[:, 0]) / xt[:, 1])
            if not np.isfinite(iv).all():
                raise ValueError("Invalid IV, do not silently drop synthetic quotes")
            np.savez(folder / f"{split}.npz", xt=xt, price=price, iv=iv)
            checks[split] = {"rows": len(xt), "max_96_128_node_difference": delta,
                             "sha256": digest(folder / f"{split}.npz")}
        a = np.load(folder / "train.npz")["xt"]
        b = np.load(folder / "holdout.npz")["xt"]
        assert not set(map(tuple, a)) & set(map(tuple, b))
        audit["cases"][case["id"]] = checks
    p = cfg["cases"][0]["parameters"]
    # Published S=K=100, r=.03, T=1; normalized forward price rescaled.
    reference = float(teacher(p, np.array([[.03, 1.]]), 128)[0] * 100 * np.exp(-.03))
    assert abs(reference - 26.9504) < .00005
    audit["published_DH1_table3_check"] = {"calculated": reference, "published_rounded": 26.9504}
    audit["collocation_sha256"] = digest(out / "collocation.npz")
    write_json(out / "data_audit.json", audit)
    for path in [Path(__file__), ROOT / "src/mentor_dh_pinn/regular_pinn_torch.py",
                 ROOT / "src/mentor_dh_pinn/torch_pricer.py"]:
        audit.setdefault("code_sha256", {})[str(path.relative_to(ROOT))] = digest(path)
    write_json(out / "data_audit.json", audit)
    print(json.dumps(audit), flush=True)


def train(data, case, seed):
    """No access to truth JSON, held-out data, or exact-pricer objective."""
    pr = json.loads((data / "protocol.json").read_text())
    if seed not in pr["seeds"]:
        raise ValueError("Seed outside frozen pilot protocol")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = data / case / f"seed_{seed}"
    out.mkdir(exist_ok=False)
    with np.load(data / case / "train.npz") as z:
        if set(z.files) != {"xt", "price", "iv"}:
            raise ValueError("Unexpected columns in blind training data")
        xt, prices, iv = [torch.tensor(z[k]) for k in ("xt", "price", "iv")]
    colloc = torch.tensor(np.load(data / "collocation.npz")["coords"])
    model = TorchRegularVariancePINN(width=pr["width"], depth=pr["depth"])
    # Small initial correction; not pretrained at the target parameters.
    with torch.no_grad():
        model.head.weight.mul_(.01)
        model.head.bias.zero_()
    raw = torch.nn.Parameter(initial_raw(seed))
    write_json(out / "initial_parameters.json", decode(raw).detach().tolist())
    params = [*model.parameters(), raw]
    optimizer = torch.optim.Adam(params, lr=pr["learning_rate"])

    def loss(points):
        p = decode(raw)
        coords, structural = data_coords(xt, p)
        loss_iv = ((model.iv(coords, structural) - iv) / pr["iv_scale"]).square().mean()
        loss_price = ((model.price(coords, structural) - prices) / pr["price_scale"]).square().mean()
        res, diagnostics = residual(model, points, structural)
        loss_pde = (res / pr["pde_scale"]).square().mean()
        shape = torch.relu(-diagnostics["convexity"]).square().mean()
        value = loss_iv + loss_price + pr["pde_weight"] * loss_pde + pr["shape_weight"] * shape
        return value, {"iv_scaled_mse": loss_iv, "price_scaled_mse": loss_price,
                       "pde_scaled_mse": loss_pde, "shape": shape}

    started = time.monotonic()
    with (out / "training.jsonl").open("x") as log:
        for step in range(1, pr["adam_steps"] + 1):
            indices = rng.choice(len(colloc), pr["pde_batch"], replace=False)
            optimizer.zero_grad(set_to_none=True)
            value, terms = loss(colloc[indices])
            if not torch.isfinite(value):
                raise FloatingPointError(f"Nonfinite objective at step {step}")
            value.backward()
            torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                row = {"phase": "adam", "step": step, "loss": float(value.detach()),
                       **{k: float(v.detach()) for k, v in terms.items()},
                       "seconds": time.monotonic() - started}
                log.write(json.dumps(row) + "\n"); log.flush()
                print(case, seed, json.dumps(row), flush=True)
        # Fixed subset / deterministic closure for quasi-Newton phase.
        points = colloc[rng.choice(len(colloc), 512, replace=False)]
        optimizer = torch.optim.LBFGS(params, max_iter=pr["lbfgs_iterations"],
                                      line_search_fn="strong_wolfe", tolerance_grad=1e-8)
        calls = 0

        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            value, _ = loss(points)
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite L-BFGS objective")
            value.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("Nonfinite L-BFGS gradient")
            calls += 1
            return value

        optimizer.step(closure)
    fitted = decode(raw).detach().numpy()
    torch.save({"state_dict": model.state_dict(), "raw": raw.detach(),
                "width": pr["width"], "depth": pr["depth"]}, out / "checkpoint.pt")
    write_json(out / "fit.json", {"parameters": fitted.tolist(), "seconds": time.monotonic() - started,
                                 "lbfgs_closure_calls": calls, "feller_margins": feller(fitted).tolist(),
                                 "training_sha256": digest(data / case / "train.npz"),
                                 "truth_used_in_objective": False, "checkpoint_selection": "fixed final"})
    print("FIT COMPLETE", case, seed, flush=True)


def score(data):
    from src.mentor_dh_pinn.regular_pinn_data import invert_total_variance
    cfg = json.loads((data / "truth_and_sources.json").read_text())
    results = []
    for case in cfg["cases"]:
        truth = np.array(case["parameters"])
        for seed in cfg["protocol"]["seeds"]:
            out = data / case["id"] / f"seed_{seed}"
            fit = json.loads((out / "fit.json").read_text())
            checkpoint = torch.load(out / "checkpoint.pt", weights_only=True)
            model = TorchRegularVariancePINN(width=checkpoint["width"], depth=checkpoint["depth"])
            model.load_state_dict(checkpoint["state_dict"])
            p = torch.tensor(fit["parameters"])
            error = np.abs(p.numpy() - truth)
            tolerance = cfg["protocol"]["positive_parameter_relative_tolerance"] * np.abs(truth)
            tolerance[[3, 8]] = cfg["protocol"]["rho_absolute_tolerance"]
            row = {"case": case["id"], "seed": seed, "truth": truth.tolist(),
                   "fitted": p.tolist(), "absolute_error": error.tolist(),
                   "parameter_pass": (error <= tolerance).tolist(),
                   "passed_parameters": int((error <= tolerance).sum()),
                   "all_ten_pass": bool((error <= tolerance).all())}
            for split in ("train", "holdout"):
                with np.load(data / case["id"] / f"{split}.npz") as z:
                    xt = torch.tensor(z["xt"])
                    coords, structural = data_coords(xt, p)
                    with torch.no_grad():
                        niv = model.iv(coords, structural).numpy()
                        nprice = model.price(coords, structural).numpy()
                    ep = teacher(p.numpy(), xt.numpy(), 128)
                    ep96 = teacher(p.numpy(), xt.numpy(), 96)
                    eiv = np.sqrt(invert_total_variance(ep, xt[:, 0].numpy()) / xt[:, 1].numpy())
                    if not np.isfinite(eiv).all() or np.max(np.abs(ep - ep96)) > 1e-7:
                        raise ValueError("Fitted-parameter repricing failed numerical gate")
                    rmse = lambda x: float(np.sqrt(np.mean(x ** 2)))
                    row[split] = {"rows": len(xt), "neural_iv_rmse": rmse(niv - z["iv"]),
                                  "repriced_iv_rmse": rmse(eiv - z["iv"]),
                                  "neural_normalized_price_rmse": rmse(nprice - z["price"]),
                                  "repriced_normalized_price_rmse": rmse(ep - z["price"])}
            results.append(row)
    write_json(data / "results.json", results)
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "train", "score"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/literature_pinn_pilot.json")
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare": prepare(args.config, args.data)
    elif args.mode == "train": train(args.data, args.case, args.seed)
    else: score(args.data)


if __name__ == "__main__":
    main()
