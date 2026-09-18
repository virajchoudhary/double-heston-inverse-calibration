"""Post-training audit only; never selects, updates or restarts a checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import qmc
from literature_fixed_truth_pilot import (
    ROOT, TorchRegularVariancePINN, residual, digest, write_json,
)


def main(data):
    torch.set_num_threads(2)
    audit = json.loads((data / "data_audit.json").read_text())
    cfg = json.loads((data / "truth_and_sources.json").read_text())
    protocol = json.loads((data / "protocol.json").read_text())
    assert protocol == cfg["protocol"], "Protocol changed after generation"
    for path, expected in audit["code_sha256"].items():
        assert digest(ROOT / path) == expected, f"Code changed: {path}"
    assert digest(data / "collocation.npz") == audit["collocation_sha256"]
    d = protocol["collocation_domain"]
    lo = np.array([d["x_forward"][0], *[d["each_variance"][0]] * 2, d["tau_years"][0]])
    hi = np.array([d["x_forward"][1], *[d["each_variance"][1]] * 2, d["tau_years"][1]])
    # Fresh post-fit diagnostics, never used for loss weighting or checkpoint choice.
    points = torch.tensor(lo + qmc.LatinHypercube(4, seed=915999).random(4096) * (hi - lo))
    rows = []
    starts = {}
    for case in cfg["cases"]:
        folder = data / case["id"]
        coordinates = {}
        for split in ("train", "holdout"):
            file = folder / f"{split}.npz"
            assert digest(file) == audit["cases"][case["id"]][split]["sha256"]
            with np.load(file) as z:
                assert set(z.files) == {"xt", "price", "iv"}
                assert all(np.isfinite(z[key]).all() for key in z.files)
                coordinates[split] = set(map(tuple, z["xt"]))
        assert not coordinates["train"] & coordinates["holdout"]
        for seed in protocol["seeds"]:
            out = folder / f"seed_{seed}"
            initial = json.loads((out / "initial_parameters.json").read_text())
            starts.setdefault(seed, initial)
            assert initial == starts[seed], "Case-specific initialization detected"
            fit = json.loads((out / "fit.json").read_text())
            assert fit["training_sha256"] == digest(folder / "train.npz")
            ckpt = torch.load(out / "checkpoint.pt", weights_only=True)
            model = TorchRegularVariancePINN(width=ckpt["width"], depth=ckpt["depth"])
            model.load_state_dict(ckpt["state_dict"])
            structural = torch.tensor(fit["parameters"]).reshape(2, 5)[:, :4]
            values, convexity = [], []
            for chunk in points.split(128):
                r, diag = residual(model, chunk, structural)
                values.extend(r.detach().tolist())
                convexity.extend(diag["convexity"].detach().tolist())
            values, convexity = np.asarray(values), np.asarray(convexity)
            assert np.isfinite(values).all() and np.isfinite(convexity).all()
            rows.append({"case": case["id"], "seed": seed,
                         "fresh_points": len(points),
                         "scaled_pde_rmse": float(np.sqrt(np.mean(values ** 2))),
                         "scaled_pde_max_absolute": float(np.max(np.abs(values))),
                         "negative_convexity_points": int((convexity < -1e-10).sum()),
                         "checkpoint_sha256": digest(out / "checkpoint.pt")})
    result = {"integrity_checks_passed": True, "same_seed_same_start_all_cases": True,
              "numerical_finiteness_passed": True,
              "caution": "Integrity is not parameter-recovery or PDE-accuracy success. Diagnostics were not used to tune this pilot.",
              "fresh_collocation_seed": 915999, "diagnostics": rows}
    write_json(data / "postfit_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    main(parser.parse_args().data)
