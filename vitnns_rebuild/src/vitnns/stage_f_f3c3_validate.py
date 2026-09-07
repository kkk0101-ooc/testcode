"""Source-recomputing numerical integrity checks for F3C3 terminals.

This validator reads numerical payloads only for integrity checks.  It returns
status and identities, never empirical values or comparisons.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .constants import REPO_ROOT
from . import stage_f_f3c3 as f3c3
from . import stage_f_f3c1_kernels as kernels


def validate_numeric_terminal(root: Path | str) -> dict[str, object]:
    root = Path(root)
    validated = f3c3.validate_terminal(root)
    result, manifest = validated["result"], validated["manifest"]
    if result.get("status") != "VALID":
        raise f3c3.StageFF3C3Error("numerical validator requires VALID terminal")
    loaded = {item["path"]: np.load(root / item["path"], allow_pickle=False) for item in manifest["arrays"]}
    method = result.get("method_id")
    expected_shapes = {
        "native.npy": (3, 224, 224) if method in kernels.SIGNED_METHODS else (14, 14),
        "patch_mass.npy": (14, 14), "target_oriented_mass.npy": (14, 14),
        "rankings.npy": (5, 196), "insertion_curves.npy": (5, 21),
        "deletion_curves.npy": (5, 21), "insertion_aucs.npy": (5,),
        "deletion_aucs.npy": (5,), "fidelity_scores.npy": (5,),
        "fractions.npy": (21,), "actual_fractions.npy": (21,),
    }
    if set(loaded) != set(expected_shapes) or any(tuple(loaded[name].shape) != shape or loaded[name].dtype != np.float32 or not np.isfinite(loaded[name]).all() for name, shape in expected_shapes.items()):
        raise f3c3.StageFF3C3Error("VALID terminal shape, dtype, finiteness, or roster drifted")
    expected_fractions = np.asarray(kernels.FRACTIONS, dtype=np.float32)
    expected_actual = np.asarray(kernels._actual_fractions(), dtype=np.float32)
    if not np.array_equal(loaded["fractions.npy"], expected_fractions) or not np.array_equal(loaded["actual_fractions.npy"], expected_actual):
        raise f3c3.StageFF3C3Error("fidelity fraction contract drifted")
    actual = loaded["actual_fractions.npy"].astype(np.float64)
    insertion = np.trapz(loaded["insertion_curves.npy"].astype(np.float64), actual, axis=1)
    deletion = np.trapz(loaded["deletion_curves.npy"].astype(np.float64), actual, axis=1)
    scores = 0.5 * (insertion - deletion)
    if not np.allclose(insertion, loaded["insertion_aucs.npy"], atol=2e-7, rtol=2e-6) or not np.allclose(deletion, loaded["deletion_aucs.npy"], atol=2e-7, rtol=2e-6) or not np.allclose(scores, loaded["fidelity_scores.npy"], atol=2e-7, rtol=2e-6):
        raise f3c3.StageFF3C3Error("fidelity AUC or score is not source-recomputable")
    if not np.isclose(float(loaded["target_oriented_mass.npy"].sum(dtype=np.float64)), 1.0, atol=2e-6, rtol=2e-6):
        raise f3c3.StageFF3C3Error("target-oriented mass does not sum to one")
    identity = result.get("unit_identity", {})
    if result.get("unit_canonical_digest") != f3c3.canonical_sha256(identity):
        raise f3c3.StageFF3C3Error("unit canonical identity drifted")
    required_identity = {
        "protocol_id", "sample_id", "formal_position", "selection_order",
        "jpeg_sha256", "official_imagenet1k_index", "method_id",
        "model_sha256", "model_canonical_tensor_digest", "preprocessing_identity",
        "clean_input_sha256", "frozen_target_class_index", "target_identity_digest",
        "source_config_code_digest", "source_config_code_hashes", "attempt_id",
    }
    if set(identity) != required_identity or identity.get("method_id") != method or identity.get("attempt_id") != f3c3.ATTEMPT_ID:
        raise f3c3.StageFF3C3Error("unit identity fields, method, or attempt drifted")
    target = result.get("target", {})
    if target.get("resolved_once_from_same_clean_forward") is not True or result.get("target_identity_digest") != f3c3.canonical_sha256(target) or identity.get("frozen_target_class_index") != target.get("class_index"):
        raise f3c3.StageFF3C3Error("frozen target identity drifted")
    if result.get("class_agnostic") is not (method in kernels.CLASS_AGNOSTIC_METHODS) or result.get("class_targeted") is (method in kernels.CLASS_AGNOSTIC_METHODS):
        raise f3c3.StageFF3C3Error("method class-target label drifted")
    contract = result.get("fidelity_contract", {})
    expected_seeds = [kernels.tie_seed(identity["sample_id"], repeat) for repeat in range(5)]
    expected_rankings = []
    sigma = int(result["fidelity_numeric_payload"]["sigma"])
    for seed in expected_seeds:
        ranking, priority = kernels.method_ranking(loaded["patch_mass.npy"], sigma=sigma, seed=seed)
        expected_rankings.append(ranking.astype(np.float32))
        if kernels.array_digest(priority) not in contract.get("tie_priority_sha256", []):
            raise f3c3.StageFF3C3Error("tie priority identity drifted")
    if not np.array_equal(np.stack(expected_rankings), loaded["rankings.npy"]):
        raise f3c3.StageFF3C3Error("ranking is not source-recomputable")
    mass = np.maximum(np.float32(sigma) * loaded["patch_mass.npy"], np.float32(0.0))
    mass = np.asarray(mass / float(mass.sum(dtype=np.float64)), dtype=np.float32)
    if not np.array_equal(mass, loaded["target_oriented_mass.npy"]):
        raise f3c3.StageFF3C3Error("target-oriented mass is not source-recomputable")
    if contract.get("fractions") != list(kernels.FRACTIONS) or contract.get("actual_fractions") != list(kernels._actual_fractions()) or contract.get("tie_seeds") != expected_seeds or contract.get("target") != "unmodified_frozen_clean_predicted_class_raw_logit" or contract.get("score") != "0.5*(insertion_auc-deletion_auc)":
        raise f3c3.StageFF3C3Error("fidelity evaluator contract drifted")
    conservation = result.get("ig_conservation")
    if method in {kernels.METHOD_ID, "full_model_class_logit_ig"}:
        if not isinstance(conservation, Mapping) or conservation.get("passed") is not True:
            raise f3c3.StageFF3C3Error("required IG conservation is not PASS")
        attempts = conservation.get("attempts", [])
        steps = [item.get("steps") for item in attempts]
        passes = [item.get("pass") == [True] for item in attempts]
        if not attempts or steps != list(kernels.IG_SCHEDULE[:len(steps)]) or conservation.get("selected_steps") != steps[-1] or passes[-1] is not True or any(passes[:-1]):
            raise f3c3.StageFF3C3Error("IG attempt journal is not frozen prefix ending at first PASS")
    elif conservation is not None:
        raise f3c3.StageFF3C3Error("non-IG method carries IG conservation")
    if method == kernels.METHOD_ID:
        endpoint = result.get("endpoint_parity", {})
        if endpoint.get("status") != "PASS" or endpoint.get("max_absolute_error", 1.0) >= 1e-6 or len(result.get("attention_hashes", [])) != 12:
            raise f3c3.StageFF3C3Error("NNS attention or endpoint parity contract failed")
    elif result.get("endpoint_parity") is not None or result.get("attention_hashes"):
        raise f3c3.StageFF3C3Error("non-NNS method carries conditional attention data")
    return {"status": "PASS", "formal_position": identity["formal_position"], "selection_order": identity["selection_order"], "method_id": method, "unit_canonical_digest": result["unit_canonical_digest"]}


def validate_all(repo_root: Path | str = REPO_ROOT) -> dict[str, object]:
    repo = Path(repo_root).resolve()
    rows = [{**row, "formal_position": index} for index, row in enumerate(f3c3._formal_rows(repo), 1)]
    validated = []
    for row in rows:
        tie_sets = []
        targets = set()
        for method in f3c3.METHODS:
            root = f3c3._terminal_root(repo, row["selection_order"], method)
            validated.append(validate_numeric_terminal(root))
            result = f3c3._read(root / "result.json")
            tie_sets.append(result["fidelity_contract"]["tie_priority_sha256"])
            targets.add(result["target_identity_digest"])
        if any(item != tie_sets[0] for item in tie_sets[1:]):
            raise f3c3.StageFF3C3Error("tie priority is not shared across five methods")
        if len(targets) != 1:
            raise f3c3.StageFF3C3Error("target identity is not shared across five methods")
    if len(validated) != 250:
        raise f3c3.StageFF3C3Error("post-run validator did not cover 250 units")
    return {"status": "PASS", "validated_units": 250, "samples": 50}


def record_validation(repo_root: Path | str = REPO_ROOT) -> dict[str, object]:
    repo = Path(repo_root).resolve()
    result = validate_all(repo)
    report = f3c3._signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_postrun_numeric_integrity_validation",
        "status": "PASS", "validated_units": result["validated_units"],
        "validated_samples": result["samples"],
        "checks": ["array_hash_shape_dtype_finiteness", "fraction_and_actual_fraction_identity", "auc_and_score_formula_recomputation", "target_oriented_mass_and_ranking_identity", "shared_tie_priority", "unit_target_method_model_source_identity", "ig_attempt_prefix_and_first_pass", "attention_hash_and_endpoint_parity"],
        "empirical_values_reported_or_displayed": False,
        "scientific_comparison_or_inference_performed": False,
        "validator_file": f3c3._file_record(repo, "src/vitnns/stage_f_f3c3_validate.py"),
    })
    f3c3._write_new(repo / f3c3.NUMERIC_VALIDATION_PATH, report)
    return report


__all__ = ["record_validation", "validate_all", "validate_numeric_terminal"]
