from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from vitnns import stage_f_f3c3 as f3c3
from vitnns import stage_f_f3c3_validate as validator


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "evidence/stage_f/f3c2_pilot_v1_20260907/formal/terminals/order-0416/conditional_all_blocks_fixed_clean_attention_class_logit_ig/attempts/attempt-000001"


def _copy_and_adapt(tmp_path: Path) -> Path:
    root = tmp_path / "terminal"
    shutil.copytree(SOURCE, root)
    result_path = root / "result.json"
    value = json.loads(result_path.read_text())
    value.pop("canonical_digest")
    identity = value["unit_identity"]
    identity["formal_position"] = 1
    identity["official_imagenet1k_index"] = value["target"]["official_truth_index"]
    value["unit_canonical_digest"] = f3c3.canonical_sha256(identity)
    result_path.write_bytes(f3c3.canonical_json_bytes(f3c3._signed(value)) + b"\n")
    blind_path = root / "blind.json"
    blind = json.loads(blind_path.read_text()); blind.pop("canonical_digest")
    blind["formal_position"] = 1; blind["unit_canonical_digest"] = value["unit_canonical_digest"]
    blind["result_sha256"] = f3c3._sha_file(result_path)
    blind_path.write_bytes(f3c3.canonical_json_bytes(f3c3._signed(blind)) + b"\n")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()); manifest.pop("canonical_digest")
    manifest["result"] = f3c3._file_record(root, "result.json")
    manifest_path.write_bytes(f3c3.canonical_json_bytes(f3c3._signed(manifest)) + b"\n")
    blind = json.loads(blind_path.read_text()); blind.pop("canonical_digest")
    blind["manifest_sha256"] = f3c3._sha_file(manifest_path)
    blind_path.write_bytes(f3c3.canonical_json_bytes(f3c3._signed(blind)) + b"\n")
    return root


def _replace_array(root: Path, name: str, value: np.ndarray) -> None:
    path = root / name
    with path.open("wb") as stream:
        np.save(stream, np.asarray(value, dtype=np.float32), allow_pickle=False)
    array = np.load(path, allow_pickle=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()); manifest.pop("canonical_digest")
    record = next(item for item in manifest["arrays"] if item["path"] == name)
    record.update(bytes=path.stat().st_size, sha256=f3c3._sha_file(path), array_digest=f3c3.kernels.array_digest(array), shape=list(array.shape), dtype="float32")
    manifest_path.write_bytes(f3c3.canonical_json_bytes(f3c3._signed(manifest)) + b"\n")
    blind_path = root / "blind.json"
    blind = json.loads(blind_path.read_text()); blind.pop("canonical_digest")
    blind["manifest_sha256"] = f3c3._sha_file(manifest_path)
    blind_path.write_bytes(f3c3.canonical_json_bytes(f3c3._signed(blind)) + b"\n")


def test_numeric_validator_accepts_adapted_real_terminal_without_reporting_values(tmp_path: Path) -> None:
    result = validator.validate_numeric_terminal(_copy_and_adapt(tmp_path))
    assert result["status"] == "PASS"
    assert not ({"auc", "fidelity_score", "responses"} & set(result))


@pytest.mark.parametrize("name,delta,error", [("fractions.npy", 0.01, "fraction"), ("fidelity_scores.npy", 0.1, "AUC or score"), ("target_oriented_mass.npy", 0.1, "sum to one")])
def test_numeric_mutations_are_rejected(tmp_path: Path, name: str, delta: float, error: str) -> None:
    root = _copy_and_adapt(tmp_path)
    values = np.load(root / name, allow_pickle=False); values.flat[0] += np.float32(delta)
    _replace_array(root, name, values)
    with pytest.raises(f3c3.StageFF3C3Error, match=error):
        validator.validate_numeric_terminal(root)


def test_validator_has_no_model_jpeg_render_or_statistical_surface() -> None:
    source = (REPO / "src/vitnns/stage_f_f3c3_validate.py").read_text(encoding="utf-8")
    for token in ("load_canonical_model", "Image.open", "matplotlib", "savefig", "paired_sign_flip_p_value(", "bootstrap_mean_ci(", "summarize_contrast("):
        assert token not in source
