"""Screen proposed truths, freeze one choice, confirm it without changing the PINN."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

import literature_fixed_truth_pilot as pilot
from src.mentor_dh_pinn.regular_pinn_data import invert_total_variance


def ranking(rows):
    candidates = []
    for name in sorted({row["case"] for row in rows}):
        fits = [row for row in rows if row["case"] == name]
        errors = []
        for fit in fits:
            tolerance = .05 * np.abs(fit["truth"])
            tolerance[[3, 8]] = .05
            errors.append(float(np.max(np.asarray(fit["absolute_error"]) / tolerance)))
        candidates.append({"case": name, "all_ten_pass_fits": sum(r["all_ten_pass"] for r in fits),
                           "fits": len(fits), "median_worst_normalized_error": float(np.median(errors)),
                           "worst_repriced_iv_rmse": max(r["holdout"]["repriced_iv_rmse"] for r in fits),
                           "parameter_pass_counts": [r["passed_parameters"] for r in fits]})
    return sorted(candidates, key=lambda r: (-r["all_ten_pass_fits"],
                  r["median_worst_normalized_error"], r["worst_repriced_iv_rmse"], r["case"]))


def run_stage(config, path):
    pilot.prepare(config, path)
    cfg = json.loads(config.read_text())
    jobs = [(c["id"], s) for c in cfg["cases"] for s in cfg["protocol"]["seeds"]]

    def train(job):
        name, seed = job
        with (path / name / f"seed_{seed}_process.log").open("x") as log:
            subprocess.run([sys.executable, str(Path(pilot.__file__)), "train", "--data", str(path),
                            "--case", name, "--seed", str(seed)], stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        print("FINISHED", name, seed, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(train, jobs))
    pilot.score(path)
    subprocess.run([sys.executable, str(Path(__file__).with_name("audit_literature_pilot.py")),
                    "--data", str(path)], check=True)
    return json.loads((path / "results.json").read_text())


def confirmation_quotes(case, stage, spec):
    """Only called after selection is saved. No gradient updates after this evaluation."""
    start, end, count = spec["confirmation_ratios"]
    r, t = np.meshgrid(np.linspace(start, end, int(count)),
                       np.array(spec["confirmation_expiry_days"]) / 365)
    xt = np.stack([np.log(r.ravel()) + .03 * t.ravel(), t.ravel()], -1)
    target = pilot.teacher(case["parameters"], xt, 128)
    coarse = pilot.teacher(case["parameters"], xt, 96)
    assert np.max(np.abs(target - coarse)) < 1e-7
    tiv = np.sqrt(invert_total_variance(target, xt[:, 0]) / xt[:, 1])
    assert np.isfinite(tiv).all()
    old = np.load(stage / case["id"] / "train.npz")["xt"]
    assert not set(map(tuple, old)) & set(map(tuple, xt))
    np.savez(stage / "unseen_confirmation_quotes.npz", xt=xt, price=target, iv=tiv)
    rows = []
    for seed in spec["confirmation_seeds"]:
        folder = stage / case["id"] / f"seed_{seed}"
        fit = json.loads((folder / "fit.json").read_text())
        params = torch.tensor(fit["parameters"])
        ckpt = torch.load(folder / "checkpoint.pt", weights_only=True)
        model = pilot.TorchRegularVariancePINN(width=ckpt["width"], depth=ckpt["depth"])
        model.load_state_dict(ckpt["state_dict"])
        coords, structure = pilot.data_coords(torch.tensor(xt), params)
        with torch.no_grad():
            niv = model.iv(coords, structure).numpy()
        price = pilot.teacher(fit["parameters"], xt, 128)
        coarse = pilot.teacher(fit["parameters"], xt, 96)
        assert np.max(np.abs(price - coarse)) < 1e-7
        eiv = np.sqrt(invert_total_variance(price, xt[:, 0]) / xt[:, 1])
        assert np.isfinite(eiv).all()
        rows.append({"seed": seed, "rows": len(xt),
                     "neural_iv_rmse": float(np.sqrt(np.mean((niv - tiv) ** 2))),
                     "repriced_iv_rmse": float(np.sqrt(np.mean((eiv - tiv) ** 2)))})
    pilot.write_json(stage / "unseen_confirmation_results.json", rows)
    return rows


def promotion_allowed(fits, diagnostics, unseen, gates):
    return (bool(fits) and bool(diagnostics) and bool(unseen)
            and all(r["all_ten_pass"] for r in fits)
            and all(r["holdout"]["repriced_iv_rmse"] <= gates["max_repriced_iv_rmse"] for r in fits)
            and all(r["scaled_pde_rmse"] <= gates["max_fresh_scaled_pde_rmse"]
                    and r["negative_convexity_points"] <= gates["max_sampled_negative_convexity_points"]
                    for r in diagnostics)
            and all(r["repriced_iv_rmse"] <= gates["max_repriced_iv_rmse"] for r in unseen))


def main(out):
    torch.set_num_threads(2)
    root = pilot.ROOT
    spec_path = root / "configs/literature_parameter_sweep.json"
    spec = json.loads(spec_path.read_text())
    base = json.loads((root / "configs/literature_pinn_pilot.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    pilot.write_json(out / "selection_protocol.json", spec)
    pilot.write_json(out / "runner_provenance.json", {
        "runner_sha256": pilot.digest(Path(__file__)), "spec_sha256": pilot.digest(spec_path),
        "trainer_sha256": pilot.digest(Path(pilot.__file__)),
        "budget_note": "unchanged 1000 Adam + 100 L-BFGS per fit; no retuning between cases"})
    base["experiment"] = spec["experiment"] + "_screening"
    base["cases"] = [base["cases"][0], *spec["proposals"]]
    base["protocol"]["seeds"] = spec["screening_seeds"]
    for c in base["cases"]:
        assert len(c["parameters"]) == 10 and c["parameters"][0] < c["parameters"][5]
        assert np.all(pilot.feller(c["parameters"]) > 0)
    config = out / "screening_config.json"
    pilot.write_json(config, base)
    rows = run_stage(config, out / "screening")
    ranked = ranking(rows)
    selected = next(c for c in base["cases"] if c["id"] == ranked[0]["case"])
    pilot.write_json(out / "selection_before_confirmation.json", {
        "ranking": ranked, "selected": selected,
        "disclosure": "Best observed development benchmark, not yet approved for final implementation"})
    print("SELECTED FOR CONFIRMATION", selected["id"], flush=True)
    confirm = json.loads(json.dumps(base))
    confirm["experiment"] = spec["experiment"] + "_confirmation"
    # Keep published DH1 first for the inherited published-price regression check.
    confirm["cases"] = [base["cases"][0]]
    if selected["id"] != confirm["cases"][0]["id"]:
        confirm["cases"].append(selected)
    confirm["protocol"]["seeds"] = spec["confirmation_seeds"]
    cfg = out / "confirmation_config.json"
    pilot.write_json(cfg, confirm)
    confirmation = run_stage(cfg, out / "confirmation")
    unseen = confirmation_quotes(selected, out / "confirmation", spec)
    fits = [r for r in rows + confirmation if r["case"] == selected["id"]]
    diagnostics = []
    for stage in ("screening", "confirmation"):
        audit = json.loads((out / stage / "postfit_audit.json").read_text())
        assert audit["integrity_checks_passed"]
        diagnostics += [r for r in audit["diagnostics"] if r["case"] == selected["id"]]
    assert len(fits) == len(diagnostics) == 4 and len(unseen) == 2
    approved = promotion_allowed(fits, diagnostics, unseen, spec["promotion_gates"])
    decision = {"best_development_candidate": selected, "eligible_for_final_implementation": approved,
                "final_implementation_parameters": selected["parameters"] if approved else None,
                "selection_bias_disclosure": spec["selection_disclosure"],
                "screening_ranking": ranked, "selected_candidate_fits": fits,
                "selected_candidate_physics": diagnostics, "unseen_confirmation": unseen}
    pilot.write_json(out / "decision.json", decision)
    print("SWEEP COMPLETE. FINAL ELIGIBILITY:", approved, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
