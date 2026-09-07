from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from vitnns import stage_f_f3c2 as f3c2
from vitnns import stage_f_f3c1_kernels as kernels


REPO = Path(__file__).resolve().parents[1]


def test_config_freezes_exact_pilot_methods_budget_and_access() -> None:
    config = f3c2.load_config(REPO)
    assert config["authorization"] == "F3C2_PILOT_ONLY"
    assert tuple(config["pilot_orders"]) == f3c2.PILOT_ORDERS
    assert tuple(config["methods"]) == kernels.METHODS
    assert config["budget"]["f3c1_engineering_gpu_hours"] == f3c2.F3C1_ENGINEERING_GPU_HOURS
    assert config["budget"]["waiver_allowed"] is False
    assert config["access"]["formal_jpeg_or_outcome_access_allowed"] is False
    assert config["next_authorization_on_pass"] == "F3C3_FORMAL_ONLY"


def test_pilot_manifest_is_exact_frozen_five_in_order() -> None:
    rows = f3c2._pilot_rows(REPO)
    assert [row["selection_order"] for row in rows] == [416, 350, 438, 411, 337]
    assert [row["image_sha256"] for row in rows] == [
        "e7e32e6a99c41c063aa8abff9335b370b28e0982906dcb16aa2f3cb6c8784798",
        "eba9e649d1d79ade172cb58fdabd1877d7349abb5e24cbbe8d40724c0f559b55",
        "9e1f12c8c0b74e2e61d0a9ce2d7848612d88b02d92972d29f46d011e04e50f05",
        "357506cc0f72e5a78b7be7399f6da677f63d9e93da30a012eb190e73687d0174",
        "7fcd65c3fa75c6ae92146aaf1b411870a98c0795d1d395d6554c99b62661bab7",
    ]
    assert [row["imagenet1k_index"] for row in rows] == [369, 498, 597, 512, 517]
    assert 44 not in {row["selection_order"] for row in rows}


def test_false_pass_roster_is_exactly_32_unique() -> None:
    assert len(f3c2.FALSE_PASS_ATTACKS) == 32
    assert len(set(f3c2.FALSE_PASS_ATTACKS)) == 32


@pytest.mark.parametrize("attack", f3c2.FALSE_PASS_ATTACKS)
def test_every_false_pass_attack_is_independently_rejected(attack: str) -> None:
    preconditions = {name: True for name in f3c2.FALSE_PASS_ATTACKS}
    preconditions[attack] = False
    assert f3c2.false_pass_reasons(preconditions) == (attack,)


@pytest.mark.parametrize("key", sorted(f3c2.BLIND_FORBIDDEN_KEYS))
def test_blind_continuation_rejects_every_forbidden_field(key: str) -> None:
    with pytest.raises(f3c2.StageFF3C2Error, match="forbidden fields"):
        f3c2.validate_blind_document({"status": "VALID", "nested": {key: 1}})


def test_blind_continuation_accepts_only_identity_status_qc_and_resources() -> None:
    value = {
        "sample_id": "sample", "selection_order": 416, "method_id": kernels.METHOD_ID,
        "status": "VALID", "endpoint_parity_status": "PASS",
        "ig_conservation_status": "PASS", "ig_selected_steps": 512,
        "manifest_hash_complete": True, "resources": {"gpu_hours": 0.01},
    }
    f3c2.validate_blind_document(value)


def _terminal_result(status: str = "VALID", marker: str = "a") -> dict:
    return {
        "schema_version": 1, "document_type": "test", "status": status,
        "unit_identity": {"sample_id": "s", "method_id": "m", "marker": marker},
        "payload_identities": {"marker": marker},
    }


def _all_arrays() -> dict[str, np.ndarray]:
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


def test_atomic_typed_terminal_publish_validate_and_verified_resume(tmp_path: Path) -> None:
    destination = tmp_path / "unit/attempts/attempt-000001"
    blind = {"schema_version": 1, "document_type": "blind", "status": "VALID", "sample_id": "s", "selection_order": 416, "method_id": "m", "unit_canonical_digest": "b" * 64}
    assert f3c2._publish_terminal(destination, _terminal_result(), _all_arrays(), blind_context=blind) == "PUBLISHED"
    assert f3c2.validate_terminal(destination)["result"]["status"] == "VALID"
    assert f3c2._publish_terminal(destination, _terminal_result(), _all_arrays(), blind_context=blind) == "RESUME_SKIP_VERIFIED"
    with pytest.raises(f3c2.StageFF3C2Error, match="conflicting terminal"):
        f3c2._publish_terminal(destination, _terminal_result(marker="changed"), _all_arrays(), blind_context=blind)


@pytest.mark.parametrize("status", ["FAILED", "NA"])
def test_failed_na_reject_numeric_payload_on_publish(status: str, tmp_path: Path) -> None:
    destination = tmp_path / status
    with pytest.raises(f3c2.StageFF3C2Error, match="must not contain numerical payload"):
        f3c2._publish_terminal(destination, _terminal_result(status=status), {"fabricated.npy": np.zeros((14, 14), dtype=np.float32)})
    assert not destination.exists()


@pytest.mark.parametrize("status", ["FAILED", "NA"])
def test_failed_na_reject_numeric_payload_on_validation(status: str, tmp_path: Path) -> None:
    destination = tmp_path / status
    f3c2._publish_terminal(destination, _terminal_result(status=status), {})
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    array_path = destination / "fabricated.npy"
    with array_path.open("wb") as stream:
        np.save(stream, np.zeros((14, 14), dtype=np.float32), allow_pickle=False)
    value = np.load(array_path, allow_pickle=False)
    manifest["arrays"] = [{"path": "fabricated.npy", "bytes": array_path.stat().st_size, "sha256": f3c2._sha_file(array_path), "array_digest": kernels.array_digest(value), "shape": [14, 14], "dtype": "float32"}]
    manifest.pop("canonical_digest")
    manifest = f3c2._signed(manifest)
    manifest_path.write_bytes(f3c2.canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(f3c2.StageFF3C2Error, match="carries numerical payload"):
        f3c2.validate_terminal(destination)


def test_valid_terminal_requires_complete_payload_roster(tmp_path: Path) -> None:
    destination = tmp_path / "valid"
    f3c2._publish_terminal(destination, _terminal_result(), {"native.npy": np.ones((14, 14), dtype=np.float32)})
    with pytest.raises(f3c2.StageFF3C2Error, match="payload roster drifted"):
        f3c2.validate_terminal(destination)


def test_terminal_tamper_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "valid"
    f3c2._publish_terminal(destination, _terminal_result(), _all_arrays())
    with (destination / "patch_mass.npy").open("wb") as stream:
        np.save(stream, np.zeros((14, 14), dtype=np.float32), allow_pickle=False)
    with pytest.raises(f3c2.StageFF3C2Error, match="payload conflict"):
        f3c2.validate_terminal(destination)


def test_source_has_no_render_statistics_formal_or_order44_execution_surface() -> None:
    source = (REPO / "src/vitnns/stage_f_f3c2.py").read_text(encoding="utf-8")
    for token in ("matplotlib", "savefig", "Image.show", "paired_sign_flip_p_value(", "bootstrap_mean_ci(", "summarize_contrast(", "run_f3c3", "run_f3c4", "run_f4", "00002287.JPEG"):
        assert token not in source


def test_budget_formula_uses_exact_frozen_conservative_equations() -> None:
    source = (REPO / "src/vitnns/stage_f_f3c2.py").read_text(encoding="utf-8")
    assert "formal_gpu = 50 * maximum_gpu" in source
    assert "total_gpu = F3C1_ENGINEERING_GPU_HOURS + pilot_gpu + formal_gpu" in source
    assert "projected_retained = existing + 50 * maximum_retained" in source


def test_unit_identity_changes_on_any_bound_identity_field() -> None:
    base = {"protocol_id": "p", "sample_id": "s", "selection_order": 416, "jpeg_sha256": "a", "method_id": "m", "model_sha256": "b", "model_canonical_tensor_digest": "c", "preprocessing_identity": "d", "clean_input_sha256": "e", "frozen_target_class_index": 1, "target_identity_digest": "f", "source_config_code_digest": "g", "attempt_id": "attempt-000001"}
    original = kernels.canonical_sha256(base)
    for key in base:
        changed = copy.deepcopy(base)
        changed[key] = str(changed[key]) + "x"
        assert kernels.canonical_sha256(changed) != original
