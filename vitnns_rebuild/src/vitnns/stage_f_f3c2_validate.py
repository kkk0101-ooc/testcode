"""Post-run source-recomputing checks for sealed F3C2 numerical terminals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .constants import REPO_ROOT
from . import stage_f_f3c2 as f3c2
from . import stage_f_f3c1_kernels as kernels


def validate_numeric_terminal(root: Path | str) -> dict[str, object]:
    root = Path(root)
    validated = f3c2.validate_terminal(root)
    result = validated["result"]
    manifest = validated["manifest"]
    if result.get("status") != "VALID":
        raise f3c2.StageFF3C2Error("numerical validator requires VALID terminal")
    loaded = {item["path"]: np.load(root / item["path"], allow_pickle=False) for item in manifest["arrays"]}
    method = result.get("method_id")
    native_shape = (3, 224, 224) if method in kernels.SIGNED_METHODS else (14, 14)
    expected_shapes = {
        "native.npy": native_shape, "patch_mass.npy": (14, 14),
        "target_oriented_mass.npy": (14, 14), "rankings.npy": (5, 196),
        "insertion_curves.npy": (5, 21), "deletion_curves.npy": (5, 21),
        "insertion_aucs.npy": (5,), "deletion_aucs.npy": (5,),
        "fidelity_scores.npy": (5,), "fractions.npy": (21,),
        "actual_fractions.npy": (21,),
    }
    if any(tuple(loaded[name].shape) != shape for name, shape in expected_shapes.items()):
        raise f3c2.StageFF3C2Error("VALID terminal numerical shape contract drifted")
    expected_fractions = np.asarray(kernels.FRACTIONS, dtype=np.float32)
    expected_actual = np.asarray(kernels._actual_fractions(), dtype=np.float32)
    if not np.array_equal(loaded["fractions.npy"], expected_fractions) or not np.array_equal(loaded["actual_fractions.npy"], expected_actual):
        raise f3c2.StageFF3C2Error("fidelity fraction contract drifted")
    actual = loaded["actual_fractions.npy"].astype(np.float64)
    insertion_auc = np.trapz(loaded["insertion_curves.npy"].astype(np.float64), actual, axis=1)
    deletion_auc = np.trapz(loaded["deletion_curves.npy"].astype(np.float64), actual, axis=1)
    scores = 0.5 * (insertion_auc - deletion_auc)
    if not np.allclose(insertion_auc, loaded["insertion_aucs.npy"], atol=2e-7, rtol=2e-6) or not np.allclose(deletion_auc, loaded["deletion_aucs.npy"], atol=2e-7, rtol=2e-6) or not np.allclose(scores, loaded["fidelity_scores.npy"], atol=2e-7, rtol=2e-6):
        raise f3c2.StageFF3C2Error("fidelity AUC or score is not source-recomputable")
    if not np.isclose(float(loaded["target_oriented_mass.npy"].sum(dtype=np.float64)), 1.0, atol=2e-6, rtol=2e-6):
        raise f3c2.StageFF3C2Error("target-oriented mass does not sum to one")
    identity = result.get("unit_identity", {})
    if result.get("unit_canonical_digest") != f3c2.canonical_sha256(identity):
        raise f3c2.StageFF3C2Error("unit canonical identity drifted")
    if identity.get("method_id") != method or identity.get("attempt_id") != f3c2.ATTEMPT_ID:
        raise f3c2.StageFF3C2Error("unit method or attempt identity drifted")
    target = result.get("target", {})
    if target.get("resolved_once_from_same_clean_forward") is not True or result.get("target_identity_digest") != f3c2.canonical_sha256(target):
        raise f3c2.StageFF3C2Error("frozen target identity drifted")
    if result.get("class_agnostic") is not (method in kernels.CLASS_AGNOSTIC_METHODS) or result.get("class_targeted") is (method in kernels.CLASS_AGNOSTIC_METHODS):
        raise f3c2.StageFF3C2Error("method class-target label drifted")
    contract = result.get("fidelity_contract", {})
    if contract.get("fractions") != list(kernels.FRACTIONS) or contract.get("actual_fractions") != list(kernels._actual_fractions()) or contract.get("target") != "unmodified_frozen_clean_predicted_class_raw_logit" or contract.get("score") != "0.5*(insertion_auc-deletion_auc)":
        raise f3c2.StageFF3C2Error("fidelity evaluator contract drifted")
    expected_seeds = [kernels.tie_seed(identity["sample_id"], repeat) for repeat in range(5)]
    if contract.get("tie_seeds") != expected_seeds or len(contract.get("tie_priority_sha256", [])) != 5:
        raise f3c2.StageFF3C2Error("fidelity tie contract drifted")
    conservation = result.get("ig_conservation")
    if method in {kernels.METHOD_ID, "full_model_class_logit_ig"}:
        if not isinstance(conservation, Mapping) or conservation.get("passed") is not True:
            raise f3c2.StageFF3C2Error("required IG conservation is not PASS")
        attempts = conservation.get("attempts", [])
        steps = [item.get("steps") for item in attempts]
        passes = [item.get("pass") == [True] for item in attempts]
        if not attempts or steps != list(kernels.IG_SCHEDULE[:len(steps)]) or conservation.get("selected_steps") != steps[-1] or passes[-1] is not True or any(passes[:-1]):
            raise f3c2.StageFF3C2Error("IG attempt journal is not frozen prefix ending at first PASS")
    elif conservation is not None:
        raise f3c2.StageFF3C2Error("non-IG method carries IG conservation")
    if method == kernels.METHOD_ID:
        endpoint = result.get("endpoint_parity", {})
        if endpoint.get("status") != "PASS" or endpoint.get("max_absolute_error", 1.0) >= 1e-6 or len(result.get("attention_hashes", [])) != 12:
            raise f3c2.StageFF3C2Error("NNS attention or endpoint parity contract failed")
    elif result.get("endpoint_parity") is not None or result.get("attention_hashes"):
        raise f3c2.StageFF3C2Error("non-NNS method carries conditional attention data")
    resources = result.get("resources", {})
    for field in ("elapsed_seconds", "cpu_seconds", "gpu_hours", "forward_calls", "backward_calls", "peak_cuda_memory_reserved_bytes", "artifact_bytes"):
        value = resources.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise f3c2.StageFF3C2Error(f"terminal resource field is invalid: {field}")
    return {"status": "PASS", "method_id": method, "unit_canonical_digest": result["unit_canonical_digest"]}


def validate_all(repo_root: Path | str = REPO_ROOT) -> dict[str, object]:
    repo = Path(repo_root).resolve()
    validated = []
    for row in f3c2._pilot_rows(repo):
        tie_sets = []
        for method in f3c2.METHODS:
            root = f3c2._terminal_root(repo, row["selection_order"], method)
            validated.append(validate_numeric_terminal(root))
            result = f3c2._read(root / "result.json")
            tie_sets.append(result["fidelity_contract"]["tie_priority_sha256"])
        if any(item != tie_sets[0] for item in tie_sets[1:]):
            raise f3c2.StageFF3C2Error("tie priority is not shared across five methods")
    if len(validated) != 25:
        raise f3c2.StageFF3C2Error("post-run validator did not cover 25 units")
    return {"status": "PASS", "validated_units": 25, "samples": 5}


__all__ = ["validate_all", "validate_numeric_terminal"]
