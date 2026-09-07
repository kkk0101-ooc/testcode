from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from vitnns import stage_f_f3c1 as f3c1
from vitnns import stage_f_f3c1_kernels as kernels


REPO = Path(__file__).resolve().parents[1]


def test_f3c1_config_and_schema_freeze_exact_boundaries() -> None:
    config = f3c1.load_config(REPO)
    assert config["authorization"] == "F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY"
    assert config["output_root"] == f3c1.ROOT_PATH
    assert config["report_root"] == f3c1.REPORT_PATH
    assert config["method_id"] == kernels.METHOD_ID
    assert config["ig"]["schedule"] == [512, 1024, 2048, 4096, 8192]
    assert config["engineering_check"] == {
        "selection_order": 44, "numerical_only": True,
        "rendering_allowed": False, "manual_map_inspection_allowed": False,
        "included_in_pilot_formal_or_statistics": False,
    }
    assert config["next_authorization_on_pass"] == "F3C2_PILOT_ONLY"


def test_false_pass_attack_roster_is_exactly_24_unique() -> None:
    assert len(f3c1.FALSE_PASS_ATTACKS) == 24
    assert len(set(f3c1.FALSE_PASS_ATTACKS)) == 24


@pytest.mark.parametrize("attack", f3c1.FALSE_PASS_ATTACKS)
def test_every_false_pass_precondition_is_independently_rejected(attack: str) -> None:
    preconditions = {name: True for name in f3c1.FALSE_PASS_ATTACKS}
    preconditions[attack] = False
    assert f3c1.false_pass_reasons(preconditions) == (attack,)


def test_atomic_terminal_publish_resume_conflict_and_tamper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(f3c1, "ROOT_PATH", "formal")
    destination = tmp_path / "formal/engineering/order-44/attempts/attempt-000001"
    result = {
        "schema_version": 1, "document_type": "test", "status": "VALID",
        "pilot_sample_access": False, "formal_sample_access": False,
    }
    arrays = {"method.patch_mass.npy": np.ones((14, 14), dtype=np.float32)}
    f3c1._publish_terminal_directory(destination, result, arrays)
    validated = f3c1.validate_typed_terminal_directory(destination)
    assert validated["status"] == "PASS"
    with pytest.raises(f3c1.StageFF3C1Error, match="overwrite"):
        f3c1._publish_terminal_directory(destination, result, arrays)
    path = destination / "method.patch_mass.npy"
    changed = np.zeros((14, 14), dtype=np.float32)
    with path.open("wb") as stream:
        np.save(stream, changed, allow_pickle=False)
    with pytest.raises(f3c1.StageFF3C1Error, match="tamper"):
        f3c1.validate_typed_terminal_directory(destination)


def test_pending_cannot_validate_as_completion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(f3c1, "ROOT_PATH", "formal")
    monkeypatch.setattr(f3c1, "COMPLETION_PATH", "formal/completion/f3c1.json")
    monkeypatch.setattr(f3c1, "PENDING_PATH", "formal/completion/f3c1.pending.json")
    pending = tmp_path / f3c1.PENDING_PATH
    f3c1._write_new(pending, f3c1._signed({"status": "PENDING"}))
    with pytest.raises(FileNotFoundError):
        f3c1.validate_completion(tmp_path)


def test_statistical_freeze_is_self_signed_and_complete() -> None:
    value = kernels.statistical_method_freeze()
    unsigned = copy.deepcopy(value); observed = unsigned.pop("canonical_digest")
    assert observed == kernels.canonical_sha256(unsigned)
    assert value["primary_contrast"]["multiplicity_adjusted"] is False
    assert value["secondary_holm_family"]["members_in_frozen_order"] == list(kernels.SECONDARY_CONTRASTS)
    assert value["p_value"]["two_sided"] is True
    assert value["p_value"]["zero_differences"] == "retained"


def test_no_pilot_formal_render_or_future_stage_execution_surface() -> None:
    source = (REPO / "src/vitnns/stage_f_f3c1.py").read_text(encoding="utf-8")
    cli = (REPO / "src/vitnns/stage_f_cli.py").read_text(encoding="utf-8")
    for forbidden in ("run_f3c2", "run_f3c3", "run_f3c4", "run_f4", "render_f3c", "41625.JPEG", "40264.JPEG", "49321.JPEG"):
        assert forbidden not in source
    for command in ("run-f3c1-engineering", "validate-f3c1-completion"):
        assert command in cli
    for forbidden in ("run-f3c2-pilot", "run-f3c3-formal", "render-f3c4"):
        assert forbidden not in cli


def test_method_roster_and_labels_are_frozen() -> None:
    protocol = json.loads((REPO / f3c1.f3c0.CONFIG_PATH).read_text())
    assert tuple(item["id"] for item in protocol["methods"]) == kernels.METHODS
    labels = {item["id"]: item for item in protocol["methods"]}
    for method in kernels.CLASS_AGNOSTIC_METHODS:
        assert labels[method]["class_agnostic"] is True
        assert labels[method]["class_targeted"] is False


def test_source_contains_no_display_or_winner_producer() -> None:
    source = (REPO / "src/vitnns/stage_f_f3c1.py").read_text(encoding="utf-8")
    for token in ("matplotlib", "savefig", "Image.show", "winner =", "GO_FIDELITY"):
        assert token not in source
