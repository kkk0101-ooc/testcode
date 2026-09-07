from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from vitnns import stage_f_f3c3 as f3c3
from vitnns import stage_f_f3c1_kernels as kernels


REPO = Path(__file__).resolve().parents[1]


def test_pre_access_startup_gate_is_self_signed_and_bound() -> None:
    startup = f3c3.validate_startup_gate(REPO)
    assert startup["git"]["local_head"] == f3c3.STARTUP_HEAD
    assert startup["access_before_startup_seal"] == {
        "formal_jpeg_bytes_accessed": False,
        "formal_map_logit_curve_auc_score_or_comparison_accessed": False,
        "model_deserialized": False, "order44_recomputed": False,
        "pilot_jpeg_bytes_accessed": False, "rendered_or_manually_inspected": False,
    }


def test_config_freezes_formal_only_scope_budget_and_single_shard() -> None:
    config = f3c3.load_config(REPO)
    assert config["authorization"] == "F3C3_FORMAL_ONLY"
    assert config["formal_sample_count"] == 50
    assert tuple(config["methods"]) == kernels.METHODS
    assert config["execution"] == {
        "shard_count": 1, "worker_id": "worker-000", "device": "cuda:0",
        "mapping": "formal_position_modulo_shard_count", "image_group_atomic": True,
        "lease_protocol": "exclusive_create_claim_then_terminal_release",
    }
    assert config["budget"]["waiver_allowed"] is False
    assert config["next_authorization_on_pass"] == "F3C4_CPU_ANALYSIS_AND_SEAL_ONLY"


def test_formal_manifest_is_exact_frozen_fifty_in_order() -> None:
    rows = f3c3._formal_rows(REPO)
    assert len(rows) == 50
    assert kernels.canonical_sha256(rows) == f3c3.FORMAL_ROSTER_DIGEST
    assert [row["selection_order"] for row in rows] == [179,339,12,84,389,275,123,345,138,378,195,103,473,301,140,340,126,436,51,449,74,69,295,132,242,30,163,318,262,58,227,206,420,237,293,425,34,141,61,164,454,354,495,471,176,216,3,160,398,332]
    assert not ({row["selection_order"] for row in rows} & (f3c3.PILOT_ORDERS | {44}))


def test_false_pass_roster_is_exactly_36_unique() -> None:
    assert len(f3c3.FALSE_PASS_ATTACKS) == 36
    assert len(set(f3c3.FALSE_PASS_ATTACKS)) == 36


@pytest.mark.parametrize("attack", f3c3.FALSE_PASS_ATTACKS)
def test_every_false_pass_attack_is_independently_rejected(attack: str) -> None:
    preconditions = {name: True for name in f3c3.FALSE_PASS_ATTACKS}
    preconditions[attack] = False
    assert f3c3.false_pass_reasons(preconditions) == (attack,)


@pytest.mark.parametrize("key", sorted(f3c3.BLIND_FORBIDDEN_KEYS))
def test_blind_formal_summary_rejects_every_forbidden_field(key: str) -> None:
    with pytest.raises(f3c3.StageFF3C3Error, match="forbidden fields"):
        f3c3.validate_blind_document({"status": "VALID", "nested": {key: 1}})


def _result(status: str = "VALID", marker: str = "a") -> dict:
    return {
        "schema_version": 1, "document_type": "test", "status": status,
        "unit_identity": {"sample_id": "s", "method_id": "m", "marker": marker},
        "payload_identities": {"marker": marker},
    }


def _arrays() -> dict[str, np.ndarray]:
    return {
        "native.npy": np.ones((3, 224, 224), dtype=np.float32),
        "patch_mass.npy": np.ones((14, 14), dtype=np.float32),
        "target_oriented_mass.npy": np.ones((14, 14), dtype=np.float32),
        "rankings.npy": np.ones((5, 196), dtype=np.float32),
        "insertion_curves.npy": np.ones((5, 21), dtype=np.float32),
        "deletion_curves.npy": np.ones((5, 21), dtype=np.float32),
        "insertion_aucs.npy": np.ones((5,), dtype=np.float32),
        "deletion_aucs.npy": np.ones((5,), dtype=np.float32),
        "fidelity_scores.npy": np.ones((5,), dtype=np.float32),
        "fractions.npy": np.ones((21,), dtype=np.float32),
        "actual_fractions.npy": np.ones((21,), dtype=np.float32),
    }


def _blind(status: str = "VALID") -> dict:
    return {
        "schema_version": 1, "document_type": "blind", "status": status,
        "sample_id": "s", "formal_position": 1, "selection_order": 179,
        "method_id": "m", "unit_canonical_digest": "b" * 64,
    }


def test_atomic_terminal_resume_conflict_multiple_attempt_and_tamper(tmp_path: Path) -> None:
    destination = tmp_path / "unit/attempts/attempt-000001"
    assert f3c3._publish_terminal(destination, _result(), _arrays(), blind_context=_blind()) == "PUBLISHED"
    assert f3c3._publish_terminal(destination, _result(), _arrays(), blind_context=_blind()) == "RESUME_SKIP_VERIFIED"
    with pytest.raises(f3c3.StageFF3C3Error, match="conflicting terminal"):
        f3c3._publish_terminal(destination, _result(marker="changed"), _arrays(), blind_context=_blind())
    extra = destination.parent / "attempt-000002"
    extra.mkdir()
    with pytest.raises(f3c3.StageFF3C3Error, match="multiple VALID"):
        f3c3._publish_terminal(destination, _result(), _arrays(), blind_context=_blind())
    (destination / "patch_mass.npy").write_bytes(b"tampered")
    with pytest.raises((f3c3.StageFF3C3Error, ValueError)):
        f3c3.validate_terminal(destination)


@pytest.mark.parametrize("status", ["FAILED", "NA"])
def test_failed_na_reject_numerical_payload_on_both_surfaces(status: str, tmp_path: Path) -> None:
    destination = tmp_path / status
    with pytest.raises(f3c3.StageFF3C3Error, match="must not contain numerical payload"):
        f3c3._publish_terminal(destination, _result(status=status), {"fabricated.npy": np.zeros((14,14), dtype=np.float32)}, blind_context=_blind(status))
    f3c3._publish_terminal(destination, _result(status=status), {}, blind_context=_blind(status))
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("canonical_digest")
    manifest["arrays"] = [{"path": "fabricated.npy"}]
    manifest = f3c3._signed(manifest)
    manifest_path.write_bytes(f3c3.canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(f3c3.StageFF3C3Error, match="carries numerical payload"):
        f3c3.validate_terminal(destination)


def test_claim_is_exclusive_and_requires_closed_release_for_resume(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(f3c3, "ROOT_PATH", "formal")
    row = {"sample_id": "s", "selection_order": 179, "formal_position": 1}
    claim = f3c3._claim_image_group(tmp_path, row)
    with pytest.raises(f3c3.StageFF3C3Error, match="active"):
        f3c3._claim_image_group(tmp_path, row)
    f3c3._release_image_group(tmp_path, row, claim, "PASS")
    assert f3c3._claim_image_group(tmp_path, row)["claim_token"] == claim["claim_token"]


def test_unit_identity_changes_on_every_bound_field() -> None:
    base = {"protocol_id": "p", "sample_id": "s", "formal_position": 1, "selection_order": 179, "jpeg_sha256": "a", "official_imagenet1k_index": 1, "method_id": "m", "model_sha256": "b", "model_canonical_tensor_digest": "c", "preprocessing_identity": "d", "clean_input_sha256": "e", "frozen_target_class_index": 1, "target_identity_digest": "f", "source_config_code_digest": "g", "source_config_code_hashes": {"x": "h"}, "attempt_id": "attempt-000001"}
    original = kernels.canonical_sha256(base)
    for key in base:
        changed = copy.deepcopy(base); changed[key] = str(changed[key]) + "x"
        assert kernels.canonical_sha256(changed) != original


def test_source_has_no_render_or_statistical_execution_surface() -> None:
    for relative in ("src/vitnns/stage_f_f3c3.py", "src/vitnns/stage_f_f3c3_validate.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        for token in ("matplotlib", "savefig", "Image.show", "paired_sign_flip_p_value(", "bootstrap_mean_ci(", "summarize_contrast(", "holm_adjust_secondary(", "run_f3c4", "run_f4"):
            assert token not in source


def test_actual_budget_formula_and_stop_before_claim_are_explicit() -> None:
    source = (REPO / "src/vitnns/stage_f_f3c3.py").read_text(encoding="utf-8")
    assert "actual_total_gpu = F3C1_GPU_HOURS + F3C2_GPU_HOURS + formal_gpu" in source
    assert "actual_total_retained = EXISTING_F3C1_F3C2_RETAINED_BYTES + f3c3_retained" in source
    assert source.index("_assert_budget_before_new_claim(repo)") < source.index("claim = _claim_image_group(repo, row)")
