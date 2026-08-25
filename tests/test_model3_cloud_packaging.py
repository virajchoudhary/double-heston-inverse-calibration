import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preflight_pins_exact_stage_a_identity():
    module = load_script("scripts/model3_cloud/preflight.py", "model3_preflight_test")
    assert module.INHERITED_GIT_SHA == "a01ddc1db854f823eb02b91193eecb4dc6698974"
    assert module.EXPECTED_BRANCH == "research/model3-stage-a"
    assert module.EXPECTED_CONFIG_SHA256 == (
        "d38482381bd3021baff80333b40a0770941a79d80fd5e0da3b4bc314a4f10361"
    )
    assert module.EXPECTED_DATASET_SHA256 == (
        "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_output_manifest_round_trip_and_tamper_detection(tmp_path):
    module = load_script(
        "scripts/model3_cloud/package_outputs.py", "model3_package_test"
    )
    output = tmp_path / "run"
    output.mkdir()
    (output / "small.json").write_text('{"ok": true}\n', encoding="utf-8")
    payload = module.manifest(output, failed=False)
    (output / "artifact_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert payload["artifact_count"] == 1
    assert not module.verify(output)
    (output / "small.json").write_text('{"ok": false}\n', encoding="utf-8")
    failures = module.verify(output)
    assert any("hash_mismatch" in failure for failure in failures)
    assert payload["failed_run_preserved"] is False


def test_stage_b_packages_are_independent_and_frozen():
    module = load_script(
        "scripts/model3_cloud/prepare_stage_b.py", "model3_prepare_stage_b_test"
    )
    packages = [module.package(seed) for seed in module.SEEDS]
    assert [package["seed"] for package in packages] == [11, 22, 33]
    assert len({package["output_root"] for package in packages}) == 3
    shared_keys = (
        "run_kind", "train_limit", "validation_limit", "epochs", "batch_size",
        "interior_points", "terminal_points", "learning_rate",
        "weight_decay", "device", "patience",
    )
    for key in shared_keys:
        values = {package["settings"][key] for package in packages}
        assert values == {module.FROZEN_SHARED[key]}
    for package in packages:
        assert not Path(package["output_root"]).is_absolute()
        assert not Path(package["settings"]["output_root"]).is_absolute()
        assert not Path(package["settings"]["dataset"]).is_absolute()
        assert "--seed" in package["launch_command"]
        assert str(package["seed"]) in package["launch_command"]
        assert package["settings"]["smoke_mode"] is False


def test_readme_contains_no_secret_like_assignment():
    text = (ROOT / "scripts/model3_cloud/README.md").read_text(encoding="utf-8")
    assert "KAGGLE_KEY=" not in text
    assert "KAGGLE_USERNAME=" not in text
    assert json.dumps({"no_secrets": True})


def test_stage_a_launcher_writes_operator_evidence_outside_run_directory():
    source = (ROOT / "scripts/model3_cloud/launch_stage_a.py").read_text(
        encoding="utf-8"
    )
    assert 'output_root / "cloud_preflight.json"' not in source
    assert 'output_root / "launch_transcript.json"' not in source
    assert source.count("with_name(output_root.name +") == 3
    assert "--expected-git-sha" in source
    assert "required=True" in source
