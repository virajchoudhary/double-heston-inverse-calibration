"""Deterministic data protocol for the reset regular Double Heston PINN."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy
import torch

from .regular_pinn_data import DOMAIN, decode_unit, expected_variance, teacher_labels

CONTRACT_ID = "DOUBLE_DATA_V2_REPRODUCIBLE"
BASE_COMMIT = "9c44422cf6afc46e718cb21ad561756445e49b8a"
DEFAULT_SEED = 906210
PROPOSAL_BLOCK_SIZE = 4096
FLOAT_DECIMALS = 14
REQUIRED_RUNTIME = {
    "python": "3.13.9",
    "numpy": "2.4.3",
    "scipy": "1.16.2",
    "torch": "2.11.0+cpu",
}
REJECTION_REASONS = {
    1: "structural_constraint",
    2: "nonfinite_price_128",
    4: "nonfinite_price_96",
    8: "nonfinite_implied_variance",
    16: "quadrature_difference_above_1e-9",
    32: "absolute_log_variance_correction_at_least_1.5",
    64: "nonfinite_parameter_gradient",
}


def runtime_provenance() -> dict[str, str | int | bool]:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }
    return {
        **observed,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "torch_threads": torch.get_num_threads(),
        "required_versions_match": observed == REQUIRED_RUNTIME,
    }


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _little_endian(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("Object arrays are forbidden by the V2 data contract")
    if value.dtype.byteorder not in ("|", "<"):
        value = value.astype(value.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(value)


def canonicalize(array: np.ndarray) -> np.ndarray:
    value = _little_endian(array)
    if np.issubdtype(value.dtype, np.floating):
        value = np.round(value, decimals=FLOAT_DECIMALS)
    return _little_endian(value)


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, canonicalize(array), allow_pickle=False)
    return stream.getvalue()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(npy_bytes(array)).hexdigest()


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a platform-neutral stored ZIP with fixed member metadata."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, npy_bytes(arrays[name]))


def lhs_unit_block(seed: int, block_index: int, dimensions: int) -> np.ndarray:
    """One fixed-size Latin-hypercube block from an explicit PCG64 stream."""
    sequence = np.random.SeedSequence([int(seed), int(block_index), dimensions])
    rng = np.random.Generator(np.random.PCG64(sequence))
    unit = np.empty((PROPOSAL_BLOCK_SIZE, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        permutation = rng.permutation(PROPOSAL_BLOCK_SIZE)
        unit[:, dimension] = (
            permutation.astype(np.float64) + rng.random(PROPOSAL_BLOCK_SIZE)
        ) / PROPOSAL_BLOCK_SIZE
    return unit


def proposal_block(seed: int, block_index: int, *, collocation: bool = False) -> tuple[np.ndarray, np.ndarray]:
    unit = lhs_unit_block(seed, block_index, 12)
    indices = block_index * PROPOSAL_BLOCK_SIZE + np.arange(PROPOSAL_BLOCK_SIZE, dtype=np.int64)
    q = unit.copy()
    q[:, 1] = np.log(7.0 / 365.0) + unit[:, 1] * np.log(2.0 / (7.0 / 365.0))
    tau = np.exp(q[:, 1])
    scheduled = indices % 2 == 0
    slices = np.array([30, 60, 90, 180, 365, 730], dtype=np.float64) / 365.0
    tau[scheduled] = slices[(indices[scheduled] // 2) % len(slices)]
    q[:, 1] = np.log(tau)
    variance = expected_variance(q, 2)
    q[:, 0] = (unit[:, 0] * 6.0 - 3.0) * np.sqrt(variance * tau)
    if collocation:
        wide = indices % 4 == 0
        q[wide, 0] = 3.0 * (2.0 * unit[wide, 0] - 1.0)
    return canonicalize(q), indices


def structural_validity(q: np.ndarray) -> np.ndarray:
    p = decode_unit(q[:, 2:], 2)
    positive = (p[:, [0, 1, 2, 4, 5, 6, 7, 9]] > 0.0).all(axis=1)
    ordered = p[:, 0] < p[:, 5]
    feller = (2.0 * p[:, 0] * p[:, 1] > p[:, 2] ** 2) & (
        2.0 * p[:, 5] * p[:, 6] > p[:, 7] ** 2
    )
    correlations = (np.abs(p[:, [3, 8]]) < 1.0).all(axis=1)
    disk = p[:, 3] ** 2 + p[:, 8] ** 2 < 1.0
    return positive & ordered & feller & correlations & disk


def rejection_codes(q: np.ndarray, labels: dict[str, np.ndarray]) -> np.ndarray:
    codes = np.zeros(len(q), dtype=np.uint16)
    codes[~structural_validity(q)] |= 1
    finite_price = np.isfinite(labels["price"])
    finite_quadrature = np.isfinite(labels["quadrature_difference"])
    codes[~finite_price] |= 2
    codes[finite_price & ~finite_quadrature] |= 4
    codes[~np.isfinite(labels["w"])] |= 8
    codes[finite_quadrature & (labels["quadrature_difference"] > 1e-9)] |= 16
    codes[~np.isfinite(labels["g"]) | (np.abs(labels["g"]) >= 1.5)] |= 32
    codes[~np.isfinite(labels["dg_du"]).all(axis=1)] |= 64
    return codes


def label_proposals(q: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    labels = teacher_labels(q, 2)
    return labels, rejection_codes(q, labels)


def accepted_split(count: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    accepted: dict[str, list[np.ndarray]] = {}
    rejected_q: list[np.ndarray] = []
    rejected_index: list[np.ndarray] = []
    rejected_code: list[np.ndarray] = []
    total = 0
    block_index = 0
    proposal_count = 0
    while total < count:
        q, indices = proposal_block(seed, block_index)
        labels, codes = label_proposals(q)
        good_positions = np.flatnonzero(codes == 0)
        take = min(count - total, len(good_positions))
        if take == 0:
            last_position = len(q) - 1
        else:
            last_position = int(good_positions[take - 1])
        considered = np.arange(len(q)) <= last_position
        selected = good_positions[:take]
        for name, values in labels.items():
            accepted.setdefault(name, []).append(values[selected])
        accepted.setdefault("proposal_index", []).append(indices[selected])
        rejected = considered & (codes != 0)
        rejected_q.append(q[rejected])
        rejected_index.append(indices[rejected])
        rejected_code.append(codes[rejected])
        total += take
        proposal_count = int(indices[last_position]) + 1
        block_index += 1
    output = {name: canonicalize(np.concatenate(parts)) for name, parts in accepted.items()}
    rejections = {
        "q": canonicalize(np.concatenate(rejected_q)),
        "proposal_index": canonicalize(np.concatenate(rejected_index)),
        "reason_code": canonicalize(np.concatenate(rejected_code)),
    }
    return output, rejections, proposal_count


def collocation_set(count: int, seed: int) -> dict[str, np.ndarray]:
    parts: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    block_index = 0
    while sum(len(part) for part in parts) < count:
        q, proposal_indices = proposal_block(seed, block_index, collocation=True)
        take = min(count - sum(len(part) for part in parts), len(q))
        parts.append(q[:take])
        indices.append(proposal_indices[:take])
        block_index += 1
    return {
        "q": canonicalize(np.concatenate(parts)),
        "proposal_index": canonicalize(np.concatenate(indices)),
    }


def generate_dataset(
    output: Path,
    *,
    train_count: int,
    validation_count: int,
    collocation_count: int,
    seed: int = DEFAULT_SEED,
    enforce_runtime: bool = True,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if min(train_count, validation_count, collocation_count) < 1:
        raise ValueError("All requested row counts must be positive")
    runtime = runtime_provenance()
    if enforce_runtime and not runtime["required_versions_match"]:
        raise RuntimeError(f"Runtime does not match the frozen contract: {runtime}")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    runtime = runtime_provenance()
    train, train_rejections, train_proposals = accepted_split(train_count, seed)
    validation, validation_rejections, validation_proposals = accepted_split(validation_count, seed + 1)
    collocation = collocation_set(collocation_count, seed + 2)
    rejections = {
        "split": np.concatenate(
            [np.zeros(len(train_rejections["q"]), dtype=np.uint8), np.ones(len(validation_rejections["q"]), dtype=np.uint8)]
        ),
        "q": np.concatenate([train_rejections["q"], validation_rejections["q"]]),
        "proposal_index": np.concatenate([train_rejections["proposal_index"], validation_rejections["proposal_index"]]),
        "reason_code": np.concatenate([train_rejections["reason_code"], validation_rejections["reason_code"]]),
    }
    artifacts = {
        "train.npz": train,
        "validation.npz": validation,
        "collocation.npz": collocation,
        "rejections.npz": rejections,
    }
    for name, arrays in artifacts.items():
        write_deterministic_npz(output / name, arrays)
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__),
        root / "src/mentor_dh_pinn/regular_pinn_data.py",
        root / "src/mentor_dh_pinn/torch_pricer.py",
        root / "scripts/mentor_dh_pinn/prepare_regular_pinn_v2.py",
    ]
    manifest = {
        "contract_id": CONTRACT_ID,
        "status": "complete",
        "historical_evidence_policy": "Historical double_data and checkpoints are immutable and are not inputs to this reset cohort.",
        "generator_base_commit": BASE_COMMIT,
        "generator_source_sha256": {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in sources},
        "runtime": runtime,
        "required_runtime": REQUIRED_RUNTIME,
        "dtype": "little-endian float64; indices little-endian int64; reason codes little-endian uint16",
        "device_policy": "CPU only; torch.set_num_threads(1); float64/complex128",
        "rng": {
            "algorithm": "numpy.random.PCG64 via SeedSequence([split_seed, block_index, 12])",
            "design": "independent fixed 4096-row Latin-hypercube blocks",
            "seed": seed,
            "split_seeds": {"train": seed, "validation": seed + 1, "collocation": seed + 2},
            "proposal_block_size": PROPOSAL_BLOCK_SIZE,
        },
        "parameter_bounds": DOMAIN,
        "validity_constraints": [
            "positive kappa/theta/sigma/v0",
            "kappa_slow < kappa_fast",
            "strict per-factor Feller gaps",
            "each rho strictly inside (-1,1)",
            "rho_slow^2 + rho_fast^2 < 1",
        ],
        "acceptance_rules": [
            "finite 128-node and 96-node Torch prices",
            "finite Black implied total variance",
            "96-vs-128 absolute price difference <= 1e-9",
            "absolute log variance correction < 1.5",
            "finite ten-parameter gradient",
        ],
        "ordering": "ascending proposal index; rejected proposals are recorded and the stream continues until the exact accepted count",
        "canonicalization": {"float_decimal_places": FLOAT_DECIMALS, "archive": "ZIP_STORED with sorted members and fixed 1980-01-01 metadata"},
        "rejection_reason_bits": {str(bit): reason for bit, reason in REJECTION_REASONS.items()},
        "splits": {
            "train": {"accepted": len(train["q"]), "proposals": train_proposals, "rejected": len(train_rejections["q"])},
            "validation": {"accepted": len(validation["q"]), "proposals": validation_proposals, "rejected": len(validation_rejections["q"])},
            "collocation": {"rows": len(collocation["q"])},
        },
        "artifacts": {
            name: {
                "sha256": sha256(output / name),
                "bytes": (output / name).stat().st_size,
                "arrays": {key: {"shape": list(value.shape), "dtype": str(canonicalize(value).dtype), "npy_sha256": array_sha256(value)} for key, value in arrays.items()},
            }
            for name, arrays in artifacts.items()
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii", newline="\n")
    manifest["manifest_sha256"] = sha256(manifest_path)
    return manifest
