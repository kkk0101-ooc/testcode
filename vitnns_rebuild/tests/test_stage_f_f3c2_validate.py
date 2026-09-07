from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from vitnns import stage_f_f3c2 as f3c2
from vitnns import stage_f_f3c2_validate as validator


REPO = Path(__file__).resolve().parents[1]
SOURCE_TERMINAL = REPO / f3c2.ROOT_PATH / "terminals/order-0416/conditional_all_blocks_fixed_clean_attention_class_logit_ig/attempts/attempt-000001"


def _copy_terminal(tmp_path: Path) -> Path:
    destination = tmp_path / "terminal"
    shutil.copytree(SOURCE_TERMINAL, destination)
    return destination


def _resign_json(path: Path, mutator) -> None:
    value = json.loads(path.read_text())
    value.pop("canonical_digest")
    mutator(value)
    value = f3c2._signed(value)
    path.write_bytes(f3c2.canonical_json_bytes(value) + b"\n")


def _replace_array(root: Path, name: str, value: np.ndarray) -> None:
    path = root / name
    with path.open("wb") as stream:
        np.save(stream, np.asarray(value, dtype=np.float32), allow_pickle=False)
    array = np.load(path, allow_pickle=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("canonical_digest")
    record = next(item for item in manifest["arrays"] if item["path"] == name)
    record.update(bytes=path.stat().st_size, sha256=f3c2._sha_file(path), array_digest=f3c2.kernels.array_digest(array), shape=list(array.shape), dtype="float32")
    manifest = f3c2._signed(manifest)
    manifest_path.write_bytes(f3c2.canonical_json_bytes(manifest) + b"\n")


def test_all_25_real_terminals_recompute_without_reporting_values() -> None:
    assert validator.validate_all(REPO) == {"status": "PASS", "validated_units": 25, "samples": 5}


def test_fraction_drift_rejected(tmp_path: Path) -> None:
    root = _copy_terminal(tmp_path)
    values = np.load(root / "fractions.npy", allow_pickle=False)
    values[1] += np.float32(0.01)
    _replace_array(root, "fractions.npy", values)
    with pytest.raises(f3c2.StageFF3C2Error, match="fraction contract"):
        validator.validate_numeric_terminal(root)


def test_auc_and_score_drift_rejected(tmp_path: Path) -> None:
    root = _copy_terminal(tmp_path)
    values = np.load(root / "fidelity_scores.npy", allow_pickle=False)
    values[0] += np.float32(0.1)
    _replace_array(root, "fidelity_scores.npy", values)
    with pytest.raises(f3c2.StageFF3C2Error, match="AUC or score"):
        validator.validate_numeric_terminal(root)


def test_target_identity_drift_rejected(tmp_path: Path) -> None:
    root = _copy_terminal(tmp_path)
    result_path = root / "result.json"
    _resign_json(result_path, lambda value: value["target"].__setitem__("class_index", 999))
    manifest_path = root / "manifest.json"
    _resign_json(manifest_path, lambda value: value["result"].update(f3c2._file_record(root, "result.json")))
    with pytest.raises(f3c2.StageFF3C2Error, match="target identity"):
        validator.validate_numeric_terminal(root)


def test_ig_schedule_or_first_pass_drift_rejected(tmp_path: Path) -> None:
    root = _copy_terminal(tmp_path)
    result_path = root / "result.json"
    _resign_json(result_path, lambda value: value["ig_conservation"]["attempts"][0].__setitem__("steps", 1024))
    manifest_path = root / "manifest.json"
    _resign_json(manifest_path, lambda value: value["result"].update(f3c2._file_record(root, "result.json")))
    with pytest.raises(f3c2.StageFF3C2Error, match="frozen prefix"):
        validator.validate_numeric_terminal(root)


def test_unit_identity_drift_rejected(tmp_path: Path) -> None:
    root = _copy_terminal(tmp_path)
    result_path = root / "result.json"
    _resign_json(result_path, lambda value: value["unit_identity"].__setitem__("jpeg_sha256", "0" * 64))
    manifest_path = root / "manifest.json"
    _resign_json(manifest_path, lambda value: value["result"].update(f3c2._file_record(root, "result.json")))
    with pytest.raises(f3c2.StageFF3C2Error, match="unit canonical identity"):
        validator.validate_numeric_terminal(root)


def test_target_mass_is_exactly_mass_one(tmp_path: Path) -> None:
    root = _copy_terminal(tmp_path)
    values = np.load(root / "target_oriented_mass.npy", allow_pickle=False) * np.float32(2.0)
    _replace_array(root, "target_oriented_mass.npy", values)
    with pytest.raises(f3c2.StageFF3C2Error, match="sum to one"):
        validator.validate_numeric_terminal(root)


def test_seal_source_contains_no_model_jpeg_render_or_statistical_execution() -> None:
    for relative in ("src/vitnns/stage_f_f3c2_validate.py", "src/vitnns/stage_f_f3c2_seal.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        for token in ("load_canonical_model", "Image.open", "matplotlib", "savefig", "paired_sign_flip_p_value(", "bootstrap_mean_ci(", "summarize_contrast("):
            assert token not in source
