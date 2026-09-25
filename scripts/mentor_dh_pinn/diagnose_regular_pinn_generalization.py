#!/usr/bin/env python3
"""Compare existing training and validation labels without fitting or new tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint, sha256
from src.mentor_dh_pinn.regular_pinn_data import black_call, coordinates


def evaluate(model, data, indices, gradient_scale, batch=512):
    selected = {key: value[indices] for key, value in data.items()}
    g, iv, dg = [], [], []
    for start in range(0, len(indices), batch):
        q = mx.array(selected["q"][start:start + batch], dtype=mx.float32)
        def correction(query):
            state, structural = coordinates(query, model.factors, mx)
            return model.correction(state, structural)
        state, structural = coordinates(q, model.factors, mx)
        gg = correction(q)
        ii = model.iv(state, structural)
        dd = mx.grad(lambda query: mx.sum(correction(query)))(q)[:, 2:]
        mx.eval(gg, ii, dd)
        g.append(np.asarray(gg, dtype=float))
        iv.append(np.asarray(ii, dtype=float))
        dg.append(np.asarray(dd, dtype=float))
    g, iv, dg = map(np.concatenate, (g, iv, dg))
    q = selected["q"]
    tau = np.exp(q[:, 1])
    predicted_price = black_call(q[:, 0], iv**2 * tau)
    usable = selected["usable"].astype(bool)
    target_iv = np.sqrt(selected["w"] / tau)
    finite_target = (np.isfinite(selected["g"]) & np.isfinite(target_iv)
                     & np.isfinite(selected["price"]) & np.isfinite(selected["dg_du"]).all(1))
    finite_prediction = (np.isfinite(g) & np.isfinite(iv) & np.isfinite(predicted_price)
                         & np.isfinite(dg).all(1))
    good = usable & finite_target & finite_prediction
    errors = {
        "g": g - selected["g"], "iv": iv - target_iv,
        "price_per_strike": predicted_price - selected["price"],
        "price_per_spot": (predicted_price - selected["price"]) * np.exp(-q[:, 0]),
    }
    result = {
        "requested_candidates": len(indices), "usable_labels": int(usable.sum()),
        "rejected_label_candidates": int((~usable).sum()),
        "invalid_targets_marked_usable": int((usable & ~finite_target).sum()),
        "invalid_predictions_on_usable_labels": int((usable & ~finite_prediction).sum()),
        "evaluated_finite_candidates": int(good.sum()),
        "metric_scope": "finite predictions on explicitly usable reference labels; all exclusions counted above",
    }
    for key, error in errors.items():
        e = error[good]
        result[f"{key}_rmse"] = float(np.sqrt(np.mean(e**2))) if len(e) else None
        result[f"{key}_mae"] = float(np.mean(np.abs(e))) if len(e) else None
        result[f"{key}_p99_absolute_error"] = float(np.quantile(np.abs(e), .99)) if len(e) else None
    error = dg[good] - selected["dg_du"][good]
    result["dg_du_rmse_per_coordinate"] = np.sqrt(np.mean(error**2, axis=0)).tolist()
    result["dg_du_normalized_rmse_per_coordinate"] = np.sqrt(np.mean((error / gradient_scale)**2, axis=0)).tolist()
    result["dg_du_normalized_rmse"] = float(np.sqrt(np.mean((error / gradient_scale)**2)))
    return result


def row_bytes(values):
    a = np.ascontiguousarray(values)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def recover_stage1_snapshot(expected_hash):
    """Accept reconstructed source only when its complete original digest matches."""
    relative = "scripts/mentor_dh_pinn/train_regular_pinn.py"
    archive=ROOT/'outputs/regular_pinn_recovery/generalization_stage1.json'
    if archive.exists():
        saved=json.loads(archive.read_text()).get('recovered_stage1_trainer_source',{})
        source=saved.get('source')
        if source and hashlib.sha256(source.encode()).hexdigest()==expected_hash:
            return {**saved,'method':'Previously archived source, reverified by the original full SHA256 digest.'}
    source = (ROOT / relative).read_text()
    removals = [
        '    ap.add_argument("--constant-lr",action="store_true",help="Small-LR continuation without another warmup/decay cycle")\n',
        '    if args.constant_lr:schedule=args.lr\n',
        '    if args.resume:\n        provenance["initial_checkpoint_sha256"]=hashlib.sha256(args.resume.read_bytes()).hexdigest()\n        provenance["resume_policy"]="Continue network weights only; AdamW moments and learning-rate schedule restart. Not an independent initialization."\n',
        '    (args.out/"source_snapshot.json").write_text(json.dumps({str(p.relative_to(ROOT)):p.read_text() for p in sources},indent=2))\n',
    ]
    for text in removals:
        source = source.replace(text, "")
    digest = hashlib.sha256(source.encode()).hexdigest()
    return {"relative_path": relative, "expected_sha256": expected_hash,
            "candidate_sha256": digest, "exact_digest_match": digest == expected_hash,
            "source": source if digest == expected_hash else None,
            "method": "Reverse current continuation-option/provenance additions; accept only byte-for-byte SHA256 match to original training manifest."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "outputs/regular_pinn_recovery")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve previous diagnostic output")
    rows, split_evidence, models = [], {}, {}
    for family, factors in (("single", 1), ("double", 2)):
        data_dir = args.base / f"{family}_data"
        data = {split: dict(np.load(data_dir / f"{split}.npz")) for split in ("train", "validation")}
        train, validation = data["train"], data["validation"]
        count = len(validation["q"])
        seed = 927100 + factors
        indices = np.sort(np.random.default_rng(seed).choice(len(train["q"]), count, replace=False))
        hashes = {split: sha256(data_dir / f"{split}.npz") for split in data}
        overlap = len(np.intersect1d(row_bytes(train["q"][:, 2:]), row_bytes(validation["q"][:, 2:])))
        if overlap:
            raise ValueError("Exact parameter overlap found between training and validation")
        split_evidence[family] = {"full_train_candidates": len(train["q"]),
            "full_validation_candidates": count, "training_subset_seed": seed,
            "training_subset_indices": indices.tolist(), "full_parameter_overlap_count": overlap,
            "input_sha256": hashes}
        for arm in ("standard", "sobolev"):
            run = args.base / f"{family}_{arm}_s17"
            selection = json.loads((run / "selection.json").read_text())
            checkpoint = run / f"step_{selection['step']:06d}.safetensors"
            model, info = load_checkpoint(checkpoint)
            manifest = json.loads((run / "manifest.json").read_text())
            for split in data:
                path = (data_dir / f"{split}.npz").relative_to(ROOT)
                if manifest["input_sha256"].get(str(path)) != hashes[split]:
                    raise ValueError(f"Current data does not match training manifest: {path}")
            scale = np.asarray(manifest["gradient_label_scales"])
            metrics = {}
            for split in data:
                ix = indices if split == "train" else np.arange(count)
                metrics[split] = evaluate(model, data[split], ix, scale)
            ratios = {key: metrics["validation"][key] / metrics["train"][key]
                      for key in ("g_rmse", "iv_rmse", "price_per_spot_rmse", "dg_du_normalized_rmse")
                      if metrics["train"][key] not in (None, 0)}
            row = {"model": run.name, "factors": factors, "selected_step": selection["step"],
                   "train": metrics["train"], "validation": metrics["validation"],
                   "validation_to_training_error_ratio": ratios}
            rows.append(row)
            models[run.name] = {"checkpoint": info, "selection": selection,
                                "manifest_sha256": sha256(run / "manifest.json")}
            print(json.dumps({"model": run.name, "selected_step": selection["step"],
                              "ratios": ratios, "train_iv_rmse": metrics["train"]["iv_rmse"],
                              "validation_iv_rmse": metrics["validation"]["iv_rmse"]}), flush=True)
    original_hash = json.loads((args.base / "double_standard_s17/manifest.json").read_text())["source_sha256"]["scripts/mentor_dh_pinn/train_regular_pinn.py"]
    snapshot = recover_stage1_snapshot(original_hash)
    output = {"scope": "development diagnostics only, no calibration or new assessment truths",
              "model_prediction_arithmetic": "MLX float32, matching training; Black display prices evaluated in NumPy float64",
              "derivative_targets": "archived float64 implicit Fourier-to-Black d(g)/du labels",
              "script_sha256": sha256(__file__), "datasets": split_evidence,
              "models": models, "rows": rows,
              "interpretation_limit": "Comparable train/validation errors support approximation bias on this draw distribution; they do not prove absence of all overfitting or reliable parameter recovery.",
              "recovered_stage1_trainer_source": snapshot}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print("stage1 trainer exact digest recovery:", snapshot["exact_digest_match"])


if __name__ == "__main__":
    main()
