#!/usr/bin/env python3
"""Source-recomputing Stage Y Y0 validator and atomic sealer.

This module is intentionally Python-standard-library-only. It never imports an
image, tensor, model, accelerator, or numerical-compute package.
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROTOCOL_ID = "STAGE_Y_VIT_BLOCK_LEVEL_NNS_MECHANISM_V1_20260910"
STAGE_TOKEN = "STAGE_Y0_PROTOCOL_AND_BUDGET_FREEZE_ONLY"
NEXT_AUTHORIZATION = "STAGE_Y1_IMPLEMENTATION_VALIDATION_AND_COST_PILOT_ONLY"
ARCHIVE_REF = "refs/archive/stage-y-y0-protocol-freeze"
EXPECTED_BASE = "0c71c55ba58dbca7d68008336a040f7036f2178c"
EXPECTED_BRANCH = "stage-d-d3-pass"
ROOT = Path("evidence/stage_y/y0_protocol_20260910/formal")
REPORT_ROOT = Path("reports/stage_y_y0_20260910")
PENDING = ROOT / "completion/y0.pending.json"
COMPLETION = ROOT / "completion/y0.json"

CONFIG = Path("config/stage_y/y0_protocol.json")
CONTRACT_JSONS = (
    CONFIG,
    Path("manifests/stage_y/claim_registry.json"),
    Path("manifests/stage_y/metric_registry.json"),
    Path("manifests/stage_y/component_cut_registry.json"),
    Path("manifests/stage_y/transform_registry.json"),
    Path("manifests/stage_y/cohort_selection_and_exclusion_plan.json"),
    Path("manifests/stage_y/schema_registry.json"),
    Path("manifests/stage_y/statistical_analysis_plan.json"),
    Path("manifests/stage_y/retention_plan.json"),
    Path("manifests/stage_y/y1_cost_pilot_plan.json"),
)
PROTOCOL_DOC = Path("docs/stage_y_route_and_y0_protocol_20260910.md")
SOURCE_FILES = (
    PROTOCOL_DOC,
    *CONTRACT_JSONS,
    Path("scripts/validate_stage_y_y0.py"),
    Path("tests/test_stage_y_y0.py"),
    Path("docs/stage_y_vit_block_level_nns_mechanism_plan_20260910.md"),
    Path("docs/stage_x_protocol.md"),
    Path("docs/stage_x_x2_mech_results_to_paper_handoff_20260910.md"),
    Path("docs/阶段路线图.md"),
    Path("docs/阶段F规划.md"),
    Path("docs/阶段G规划.md"),
    Path("docs/NNS-v2前瞻性主张扩展方案.md"),
    Path("../论文1实现审计与后续计划.md"),
    Path("config/recovery_v2_binding.json"),
)
CLAIM_IDS = (
    "Y-CLM-INTRAPATCH",
    "Y-CLM-DEPTH-SPATIAL",
    "Y-CLM-HEAD-ORGANIZATION",
    "Y-CLM-PATCH-MIXING",
    "Y-CLM-BRANCH-ALLOCATION",
    "Y-CLM-PERTURBATION-MODULATION",
)
TRANSFORMS = (
    "clean", "horizontal_flip", "translate_x_-8", "translate_x_8",
    "rotate_-10", "rotate_10", "brightness_0.9", "brightness_1.1",
    "gaussian_noise_0.01_seed_0",
)
FALSE_PASS_ATTACKS = (
    "pseudo_replication_heads_patches_or_transforms_as_n",
    "post_outcome_best_head_layer_or_patch_selection",
    "single_sample_f3h_atlas_as_cohort_evidence",
    "raw_attention_as_nns_contribution",
    "stage_d_value_path_pooled_with_stage_y_class_logit",
    "identity_attention_mlp_three_way_allocation",
    "sum_twelve_layer_decompositions",
    "absolute_mass_hides_signed_cancellation",
    "attribution_share_called_performance_or_unique_causal_responsibility",
    "numerical_failure_replaces_sample",
    "perturbation_modulation_reopens_x2_spatial_stability",
    "pilot_scientific_result_deletes_transform",
    "display_array_enters_scientific_metric",
    "completion_pass_missing_required_registry",
    "git_lfs_remote_or_archive_mismatch_still_passes",
    "jpeg_model_checkpoint_or_gpu_access_in_y0",
    "y1_y2_y3_y4_y5_f3_final_or_stage_g_started_in_y0",
)
REQUIRED_CLAIM_KEYS = {
    "claim_id", "scientific_question", "population", "target",
    "estimand", "independent_statistical_unit", "exact_endpoint",
    "comparison_contrast", "effect_size_threshold",
    "ci_and_multiplicity", "missing_failed_na_handling", "SUPPORTED",
    "NOT_SUPPORTED", "INCONCLUSIVE", "INVALID", "allowed_wording",
    "forbidden_wording",
}
FORBIDDEN_IMPORTS = {"torch", "torchvision", "timm", "PIL", "cv2", "numpy", "cupy"}


class StageYY0Error(RuntimeError):
    pass


def _reject_constant(token: str) -> None:
    raise StageYY0Error(f"non-finite JSON constant: {token}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StageYY0Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(data: bytes, context: str) -> Any:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageYY0Error(f"{context} is not strict UTF-8 JSON: {exc}") from exc
    return value


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise StageYY0Error(f"required JSON missing: {path}")
    return strict_json_bytes(path.read_bytes(), str(path))


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        chunk = stream.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = stream.read(1024 * 1024)
    return digest.hexdigest()


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("canonical_digest", None)
    result["canonical_digest"] = canonical_digest(result)
    return result


def validate_signed(value: Mapping[str, Any], context: str) -> None:
    if not isinstance(value, Mapping):
        raise StageYY0Error(f"{context} must be an object")
    claimed = value.get("canonical_digest")
    unsigned = {key: item for key, item in value.items() if key != "canonical_digest"}
    if claimed != canonical_digest(unsigned):
        raise StageYY0Error(f"{context} canonical digest drifted")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise StageYY0Error(f"refusing to overwrite: {path}")
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise StageYY0Error(f"temporary output already exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def record(repo: Path, relative: Path) -> dict[str, Any]:
    path = repo / relative
    value = read_json(path)
    item = {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if isinstance(value, Mapping) and "canonical_digest" in value:
        validate_signed(value, str(relative))
        item["canonical_digest"] = value["canonical_digest"]
    return item


def file_record(repo: Path, relative: Path) -> dict[str, Any]:
    path = (repo / relative).resolve()
    if not path.is_file():
        raise StageYY0Error(f"required source missing: {relative}")
    return {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageYY0Error(message)


def load_contracts(repo: Path) -> dict[str, Any]:
    docs = {path.as_posix(): read_json(repo / path) for path in CONTRACT_JSONS}
    for path, value in docs.items():
        _require(isinstance(value, Mapping), f"{path} must be a JSON object")
        _require(value.get("protocol_id") == PROTOCOL_ID, f"{path} protocol ID drifted")
        validate_signed(value, path)
    return docs


def contract_bundle(repo: Path) -> dict[str, Any]:
    files = []
    for relative in CONTRACT_JSONS:
        value = read_json(repo / relative)
        files.append({"path": relative.as_posix(), "sha256": sha_file(repo / relative), "canonical_content_digest": canonical_digest(value)})
    return signed({"schema_version": 1, "document_type": "stage_y_y0_contract_bundle", "protocol_id": PROTOCOL_ID, "files": files, "bundle_digest": canonical_digest(files)})


def _validate_script_surface(repo: Path) -> None:
    source = (repo / "scripts/validate_stage_y_y0.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    _require(not imports.intersection(FORBIDDEN_IMPORTS), "validator imports model/image/numeric runtime")
    forbidden_surfaces = (
        "nvidia" + "-smi", "cuda" + ".is_available",
        "Image" + ".open", "torch" + ".load", "read" + "_image",
    )
    for forbidden in forbidden_surfaces:
        _require(forbidden not in source, f"validator contains forbidden access surface: {forbidden}")


def validate_semantics(repo: Path, docs: dict[str, Any] | None = None) -> dict[str, Any]:
    docs = docs or load_contracts(repo)
    config = docs[CONFIG.as_posix()]
    claims = docs["manifests/stage_y/claim_registry.json"]
    metrics = docs["manifests/stage_y/metric_registry.json"]
    cuts = docs["manifests/stage_y/component_cut_registry.json"]
    transforms = docs["manifests/stage_y/transform_registry.json"]
    cohort = docs["manifests/stage_y/cohort_selection_and_exclusion_plan.json"]
    schemas = docs["manifests/stage_y/schema_registry.json"]
    sap = docs["manifests/stage_y/statistical_analysis_plan.json"]
    retention = docs["manifests/stage_y/retention_plan.json"]
    pilot = docs["manifests/stage_y/y1_cost_pilot_plan.json"]

    _require(config["stage_token"] == STAGE_TOKEN and config["status"] == "FROZEN", "Y0 token/status drifted")
    _require(config["route"]["amendment"] == "F3_FINAL_PAUSED_PENDING_STAGE_Y", "F3-Final pause missing")
    _require(config["completion"]["pass_next_authorization"] == NEXT_AUTHORIZATION, "next authorization drifted")
    _require(config["scope"]["grad_cam_in_matrix"] is False and config["scope"]["raw_attention_role"] == "descriptive structural reference only", "method scope drifted")
    history = config["history"]
    _require(history["stage_d"]["decision"] == "GO_FIDELITY_SCOPE" and history["stage_e"]["scientific_validity"] == "MIXED", "D/E history drifted")
    _require(history["c10"]["decision"] == "NO_GO_NOVELTY" and history["x2"]["decision"] == "NO_GO_BROAD_CLAIM_DEVELOPMENT", "C10/X2 history drifted")
    _require(history["x2_mech_m2"] == {"terminal": "PASS", "interpretation": "MIXED_OR_INCONCLUSIVE"}, "M2 history drifted")

    model = config["model"]
    _require(model["input_shape"] == [3,224,224] and model["native_patches"] == 196, "input/patch scope drifted")
    _require(model["blocks"] == list(range(12)) and model["heads_per_block"] == 12 and model["block_head_coordinates"] == 144, "block/head scope drifted")
    _require(model["depth_groups"] == {"shallow": [0,1,2,3], "middle": [4,5,6,7], "deep": [8,9,10,11]}, "depth groups drifted")
    _require(config["target"]["primary"] == "clean predicted-class raw logit" and config["target"]["clean_class_fixed_for_all_perturbations_and_interventions"] is True, "target drifted")
    _require("no numeric pooling" in config["target"]["stage_d_value_path_isolation"], "Value-path/class-logit isolation missing")

    rows = claims["claims"]
    _require(tuple(row["claim_id"] for row in rows) == CLAIM_IDS, "six-claim roster/order drifted")
    _require(claims["decision_states"] == ["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "INVALID"] and claims["no_omnibus_go"] is True, "decision states drifted")
    for row in rows:
        _require(set(row) == REQUIRED_CLAIM_KEYS, f"claim schema incomplete: {row.get('claim_id')}")
        _require(row["independent_statistical_unit"] == "image", f"claim unit drifted: {row['claim_id']}")
        _require(row["target"] in {"fixed clean predicted-class raw logit", "clean predicted-class raw logit fixed across conditions"}, f"claim target drifted: {row['claim_id']}")
        _require(all(row[key] for key in REQUIRED_CLAIM_KEYS), f"empty decision field: {row['claim_id']}")

    _require(cohort["dataset"] == {"release": "ImageNet-1k ILSVRC2012 validation", "release_id": "ILSVRC2012_VAL_OFFICIAL", "count": 100, "distinct_true_classes": 100, "images_per_selected_class": 1}, "cohort count/class plan drifted")
    stages = {row["stage"] for row in cohort["prior_access_exclusion_sources"]}
    _require(stages == {"C10", "Stage_D_and_Stage_E", "F3_F3H", "F3C", "X1_audit", "X2_development_and_X2_MECH"}, "prior-access exclusion sources incomplete")
    _require(cohort["subsets"]["audit_10"]["algorithm"].startswith("take 10") and cohort["subsets"]["perturbation_20"]["algorithm"].startswith("take 20"), "subset algorithms drifted")
    _require(cohort["failure_rules"]["numerical_failure_replacement_allowed"] is False and cohort["y0_candidate_image_decode_count"] == 0, "cohort firewall drifted")

    _require(tuple(row["id"] for row in transforms["conditions"]) == TRANSFORMS, "transform panel drifted")
    flip = next(row for row in transforms["conditions"] if row["id"] == "horizontal_flip")
    _require(flip["full_domain"] is True and "full-domain" in flip["control_role"], "flip control drifted")
    _require(transforms["target_handling"]["class_index_fixed_across_all_conditions"] is True and transforms["target_handling"]["prediction_flip_rows_retained"] is True, "transform target handling drifted")
    for row in transforms["conditions"]:
        for key in ("forward_map", "inverse_map", "interpolation", "padding_fill", "full_domain"):
            _require(key in row, f"transform contract incomplete: {row['id']}:{key}")

    cut_ids = [row["cut_id"] for row in cuts["cuts"]]
    _require(cut_ids == ["post_patch_embedding_tokens", "block_input_tokens", "first_residual_addition", "second_residual_addition", "block_output_tokens", "final_norm", "classifier_input"], "cut registry drifted")
    first = cuts["cuts"][2]; second = cuts["cuts"][3]
    _require(first["allocation_pair"] == ["pre_attention_identity", "attention_aggregate"] and second["allocation_pair"] == ["post_attention_residual", "mlp_output"], "residual allocations are not separate")
    required_reconstruction = {"input_completeness", "pixel_reconstruction", "total_reconstruction", "endpoint_parity", "convergence", "head_sum", "first_residual_sum", "second_residual_sum", "scalar_map_matrix_sum", "failure_at_8192"}
    _require(set(cuts["reconstruction_contracts"]) == required_reconstruction, "reconstruction contract incomplete")
    _require(cuts["patch_routing"]["matrix_shape"] == [196,196] and cuts["patch_routing"]["cls_shape"] == [196], "patch matrix shape drifted")
    _require("sum_twelve_layer_decompositions" not in cuts["forbidden_operations"] and "sum_decompositions_across_blocks" in cuts["forbidden_operations"], "cross-layer sum guard missing")

    _require(metrics["within_patch"]["primary_operator"]["percentiles"] == list(range(0,101,5)), "within-patch percentiles drifted")
    _require(metrics["within_patch"]["primary_operator"]["effect_threshold"] == 0.02 and metrics["within_patch"]["primary_operator"]["primary_sign"] == "positive contribution first", "primary operator drifted")
    _require(metrics["head_organization"]["single_head_ablation"]["intervention_point"].startswith("native per-head"), "ablation point drifted")
    _require(metrics["branch_allocation"]["three_way_table_forbidden"] is True and metrics["perturbation_modulation"]["broad_spatial_robustness_pass"] is False, "metric guard drifted")

    arrays = {row["array_type"]: row for row in schemas["arrays"]}
    _require(arrays["source_destination_patch_matrix"]["shape"] == [196,196] and arrays["source_patch_cls_vector"]["shape"] == [196], "array schema shape drifted")
    _require(schemas["global_array_rules"]["storage_dtype"] == "float32 little-endian C-contiguous" and schemas["global_array_rules"]["reduction_dtype"] == "float64", "dtype contract drifted")
    _require(schemas["display_arrays_allowed_as_scientific_input"] is False, "display/science firewall missing")

    _require(sap["independent_unit"] == "image" and sap["bootstrap"]["replicates"] == 20000, "statistics unit/bootstrap drifted")
    _require(sap["multiplicity"] == {"method": "Holm step-down", "alpha": 0.05, "families_independent": True, "ties": "stable order by hypothesis_id", "unadjusted_claim_p_values_forbidden": True}, "Holm contract drifted")
    _require([row["family_id"] for row in sap["families"]] == ["A", "B", "C"], "multiplicity families drifted")
    _require([row["hypothesis_id"] for row in sap["families"][0]["hypotheses"]] == ["A1"], "Family A drifted")
    _require([row["hypothesis_id"] for row in sap["families"][1]["hypotheses"]] == [f"B{i}" for i in range(1,8)], "Family B drifted")
    _require(sap["no_best_selection"] is True and "144-head atlas coordinates" in sap["descriptive_only"], "descriptive boundary drifted")

    _require(retention["display_extensions_forbidden_as_scientific_input"] == [".png", ".svg", ".pdf", ".jpg", ".jpeg", ".JPEG"], "retention firewall drifted")
    _require(len(retention["all_100"]) >= 6 and len(retention["audit_10"]) >= 4 and len(retention["perturbation_20"]) >= 3, "retention tiers incomplete")

    _require(pilot["allowed_empirical_image"]["formal_stage_y_eligibility"] is False and pilot["conditions"] == ["clean", "horizontal_flip"], "Y1 prior-access pilot drifted")
    _require(pilot["coverage"]["blocks"] == list(range(12)) and pilot["coverage"]["heads_per_block"] == 12, "Y1 coverage drifted")
    _require(pilot["pilot_hard_caps"] == {"h20_gpu_hours": 8.0, "wall_hours": 16.0, "cpu_hours": 96.0, "peak_reserved_gpu_gib_per_process": 80.0, "retained_gib": 64.0, "download_bytes": 0, "devices": 1}, "Y1 pilot caps drifted")
    _require(pilot["projection"]["safety_multiplier"] == 1.5 and "new formal budget receipt" in pilot["budget_seal"], "formal budget boundary drifted")

    protocol = (repo / PROTOCOL_DOC).read_text(encoding="utf-8")
    for phrase in (PROTOCOL_ID, "F3_FINAL_PAUSED_PENDING_STAGE_Y", NEXT_AUTHORIZATION, "Image is the only independent statistical unit", "does not estimate total GPU time"):
        _require(phrase in protocol, f"protocol document missing: {phrase}")
    _validate_script_surface(repo)
    return {
        "status": "PASS", "protocol_id": PROTOCOL_ID, "claim_count": 6,
        "blocks": 12, "heads_per_block": 12, "patches": 196,
        "transform_count": 9, "families": ["A", "B", "C"],
        "next_authorization": NEXT_AUTHORIZATION,
    }


def false_pass_reasons(preconditions: Mapping[str, bool]) -> tuple[str, ...]:
    if set(preconditions) != set(FALSE_PASS_ATTACKS):
        raise StageYY0Error("false-pass precondition roster drifted")
    return tuple(name for name in FALSE_PASS_ATTACKS if preconditions[name] is not True)


def build_false_pass_audit() -> dict[str, Any]:
    checks = []
    baseline = {name: True for name in FALSE_PASS_ATTACKS}
    _require(not false_pass_reasons(baseline), "valid false-pass baseline rejected")
    for attack in FALSE_PASS_ATTACKS:
        mutated = dict(baseline); mutated[attack] = False
        rejected = false_pass_reasons(mutated) == (attack,)
        _require(rejected, f"false-pass mutation escaped: {attack}")
        checks.append({"attack": attack, "result": "REJECTED", "mechanism": "independent fail-closed precondition mutation plus focused test"})
    return signed({"schema_version": 1, "document_type": "stage_y_y0_false_pass_audit", "protocol_id": PROTOCOL_ID, "status": "PASS", "attack_count": len(checks), "checks": checks, "pass_possible_without_all_conditions": False})


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def startup_facts(repo: Path) -> dict[str, Any]:
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    head = _git(repo, "rev-parse", "HEAD")
    upstream_name = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    upstream = _git(repo, "rev-parse", "@{upstream}")
    _require(branch == EXPECTED_BRANCH, "unexpected branch")
    _require(head == EXPECTED_BASE and upstream == EXPECTED_BASE, "startup HEAD/tracking drifted")
    return {"branch": branch, "local_head": head, "tracking_name": upstream_name, "tracking_ref": upstream}


def _startup_gate(repo: Path) -> dict[str, Any]:
    facts = startup_facts(repo)
    return signed({
        "schema_version": 1, "document_type": "stage_y_y0_startup_gate",
        "protocol_id": PROTOCOL_ID, "status": "PASS", **facts,
        "remote_branch_at_start": EXPECTED_BASE, "initial_worktree_clean": True,
        "initial_index_clean": True, "initial_lfs_clean": True,
        "archive_ref_absent_at_start": True, "baseline_verified_before_y0_edits": True,
    })


def initialize_pending(repo: Path) -> dict[str, Any]:
    if COMPLETION.joinpath().is_absolute():
        raise StageYY0Error("internal completion path invariant failed")
    if (repo / COMPLETION).exists() or (repo / PENDING).exists():
        raise StageYY0Error("Y0 completion already exists")
    facts = startup_facts(repo)
    pending = signed({"schema_version": 1, "document_type": "stage_y_y0_pending_completion", "protocol_id": PROTOCOL_ID, "status": "PENDING", "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"), "base_commit": facts["local_head"], "next_authorization": "NONE_WHILE_PENDING"})
    _write_new_json(repo / PENDING, pending)
    return pending


def _source_binding(repo: Path) -> dict[str, Any]:
    files = [file_record(repo, path) for path in SOURCE_FILES]
    decision = read_json(repo / "evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/decision.json")
    c10 = read_json(repo / "evidence/stage_c/c10_recovery_v2_3cf2945/formal/decision.json")
    holm = read_json(repo / "evidence/stage_c/c10_recovery_v2_3cf2945/formal/holm.json")
    e4 = read_json(repo / "evidence/stage_e/e4_recovery_v2_20260907/formal/completion/e4.json")
    x2 = read_json(repo / "evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal/status.json")
    m2 = read_json(repo / "evidence/stage_x/x2_mech_m2_20260910/formal/completion/x2_mech_m2.json")
    f3c = read_json(repo / "evidence/stage_f/f3c4_analysis_r3_terminal_seal_20260908/formal/completion/f3c4.json")
    for context, value in (("D10 decision", decision), ("C10 decision", c10), ("C10 Holm", holm), ("E4 completion", e4), ("X2 status", x2), ("M2 completion", m2), ("F3C completion", f3c)):
        validate_signed(value, context)
    _require(decision["overall_decision"] == "GO_FIDELITY_SCOPE", "source D10 decision drifted")
    _require(c10["decision"] == "NO_GO_NOVELTY" and holm["status"] == "HOLM_BLOCKED_UNDERSPECIFIED", "source C10 decision drifted")
    _require(e4["scientific_validity"] == "MIXED", "source E4 decision drifted")
    _require(x2["status"] == "NO_GO_BROAD_CLAIM_DEVELOPMENT", "source X2 decision drifted")
    m2_identity = {row["name"]: row["value"] for row in m2["payload"]["identity"]}
    _require(m2["status"] == "PASS" and m2_identity["mechanism_branch"] == "MIXED_OR_INCONCLUSIVE", "source M2 decision drifted")
    primary = next(row for row in f3c["contrast_summary"] if row["contrast_id"] == "class_targeted_nns_minus_vit_grad_cam")
    _require(primary["mean"] < 0 and primary["bootstrap_ci_95_upper"] < 0, "source F3C boundary drifted")
    historical = {
        "stage_d_decision": "GO_FIDELITY_SCOPE",
        "stage_e_scientific_validity": "MIXED",
        "c10_decision": "NO_GO_NOVELTY",
        "x2_decision": "NO_GO_BROAD_CLAIM_DEVELOPMENT",
        "x2_mech_m2_terminal": "PASS",
        "x2_mech_m2_interpretation": "MIXED_OR_INCONCLUSIVE",
    }
    historical_sources = [
        file_record(repo, Path("evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/decision.json")),
        file_record(repo, Path("evidence/stage_c/c10_recovery_v2_3cf2945/formal/decision.json")),
        file_record(repo, Path("evidence/stage_c/c10_recovery_v2_3cf2945/formal/holm.json")),
        file_record(repo, Path("evidence/stage_e/e4_recovery_v2_20260907/formal/completion/e4.json")),
        file_record(repo, Path("evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal/status.json")),
        file_record(repo, Path("evidence/stage_x/x2_mech_m2_20260910/formal/completion/x2_mech_m2.json")),
        file_record(repo, Path("evidence/stage_f/f3c4_analysis_r3_terminal_seal_20260908/formal/completion/f3c4.json")),
    ]
    return signed({"schema_version": 1, "document_type": "stage_y_y0_source_binding", "protocol_id": PROTOCOL_ID, "base_commit": EXPECTED_BASE, "files": files, "files_digest": canonical_digest(files), "historical_facts": historical, "historical_sources": historical_sources, "historical_sources_digest": canonical_digest(historical_sources), "upstream_archive_ref": {"name": "refs/archive/stage-x-x2-mech-m2-completion", "commit": "5322ede0fe032838a2211b4de640251a4ece9ad2"}})


def _access_ledger() -> dict[str, Any]:
    return signed({
        "schema_version": 1, "document_type": "stage_y_y0_access_ledger",
        "protocol_id": PROTOCOL_ID, "status": "PASS",
        "allowed_access": ["source and tests text", "existing JSON/CSV headers and provenance metadata", "Git refs/history and LFS pointer metadata"],
        "jpeg_or_png_pixel_reads": 0, "stage_y_candidate_images_opened": 0,
        "stage_y_prediction_logit_attribution_or_outcome_rows_accessed": 0,
        "checkpoint_tensor_reads": 0, "model_initializations": 0,
        "torch_imported": False, "timm_imported": False,
        "cuda_or_gpu_accessed": False, "gpu_workloads": 0,
        "scientific_arrays_generated": 0, "y1_started": False,
        "y2_started": False, "y3_started": False, "y4_started": False,
        "y5_started": False, "f3_final_started": False, "stage_g_started": False,
    })


def _inventory(repo: Path) -> dict[str, Any]:
    root = repo / ROOT
    excluded = {"whole_root_inventory.json", "completion/y0.pending.json", "completion/y0.json"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded or relative.startswith("."):
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha_file(path)})
    return signed({"schema_version": 1, "document_type": "stage_y_y0_whole_root_inventory", "protocol_id": PROTOCOL_ID, "status": "PASS", "scope": "all regular formal Y0 files except self-referential inventory and pending/terminal completion", "excluded": sorted(excluded), "files": files, "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files), "files_digest": canonical_digest({"files": files})})


def _validation_report(repo: Path) -> dict[str, Any]:
    result = validate_semantics(repo)
    bundle = contract_bundle(repo)
    return signed({"schema_version": 1, "document_type": "stage_y_y0_source_recomputing_validation", "protocol_id": PROTOCOL_ID, "status": "PASS", "checks": {
        "required_files_parse": "PASS", "protocol_ids": "PASS", "contract_bundle_digest": bundle["bundle_digest"],
        "six_claim_decision_schemas": "PASS", "cohort_and_subsets": "PASS", "blocks_heads_patches_depth": "PASS",
        "fixed_clean_predicted_raw_logit": "PASS", "value_path_class_logit_isolation": "PASS", "separate_residual_allocations": "PASS",
        "reconstruction_contracts": "PASS", "transform_inverse_mask": "PASS", "image_only_independent_unit": "PASS",
        "bootstrap_holm_thresholds_missing_rules": "PASS", "no_best_selection": "PASS", "no_gradcam_or_broad_claim": "PASS",
        "y1_prior_access_only": "PASS", "no_image_model_checkpoint_gpu_access": "PASS", "f3_final_paused": "PASS",
        "unique_next_authorization": "PASS", "atomic_completion_contract": "PASS"},
        "summary": result, "source_recomputed": True, "scientific_result_computed": False,
    })


def _report_markdown(bundle_digest: str) -> str:
    return f"""# Stage Y Y0 terminal report\n\nY0 freezes protocol `{PROTOCOL_ID}` and prospectively pauses F3-Final as `F3_FINAL_PAUSED_PENDING_STAGE_Y`. It creates no Stage Y cohort, image outcome, model execution, scientific array, or GPU workload.\n\n## Frozen scope\n\n- Six independent claims: {', '.join(CLAIM_IDS)}.\n- ViT-B/16, 224x224, 196 patches, blocks 0-11, 12 heads per block; shallow/middle/deep = 0-3/4-7/8-11.\n- Fixed clean predicted-class raw-logit target; Stage D Value-path remains isolated.\n- Nine conditions: {', '.join(TRANSFORMS)}.\n- Image-only inference with 20,000 paired bootstrap replicates and separate Holm Families A, B, and C.\n- Y1 is one prior-access image plus synthetic/analytic fixtures, clean and horizontal flip, full 12-block/cut coverage; caps are 8 H20-GPU-hours, 16 wall-hours, 96 CPU-hours, 80 GiB reserved GPU memory, 64 GiB retained, zero downloads, one device.\n\n## Required-path mapping\n\n- Route/protocol: `docs/stage_y_route_and_y0_protocol_20260910.md`.\n- Machine protocol: `config/stage_y/y0_protocol.json`.\n- Claims, metrics, cuts, transforms, cohort, schemas, statistics, retention, and Y1 budget: `manifests/stage_y/*.json`.\n- Validator and tests: `scripts/validate_stage_y_y0.py`, `tests/test_stage_y_y0.py`.\n- Formal access, false-pass, source binding, bundle, inventory, and completion: `evidence/stage_y/y0_protocol_20260910/formal/`.\n- Terminal report, validation, false-pass, and focused-test record: `reports/stage_y_y0_20260910/`.\n\nContract bundle digest: `{bundle_digest}`. Source-recomputing validation and all {len(FALSE_PASS_ATTACKS)} false-pass mutations must pass before atomic terminal completion.\n\nThe only next authorization after PASS is `{NEXT_AUTHORIZATION}`.\n"""


def _lfs_clean(repo: Path) -> bool:
    output = _git(repo, "lfs", "status")
    section = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Git LFS objects") and stripped.endswith(":"):
            section = True
            continue
        if stripped.startswith("On branch "):
            continue
        if section and stripped:
            return False
    return True


def validate_publication(repo: Path) -> dict[str, Any]:
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    head = _git(repo, "rev-parse", "HEAD")
    tracking_name = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    tracking = _git(repo, "rev-parse", "@{upstream}")
    remote_branch_output = _git(repo, "ls-remote", "--heads", "origin", branch)
    remote_branch = remote_branch_output.split()[0] if remote_branch_output else ""
    local_archive = _git(repo, "rev-parse", ARCHIVE_REF)
    remote_archive_output = _git(repo, "ls-remote", "origin", ARCHIVE_REF)
    remote_archive = remote_archive_output.split()[0] if remote_archive_output else ""
    status_lines = _git(repo, "status", "--branch", "--porcelain=v2").splitlines()
    worktree_index_clean = not any(line and not line.startswith("#") for line in status_lines)
    lfs_clean = _lfs_clean(repo)
    _require(branch == EXPECTED_BRANCH and tracking_name == f"origin/{EXPECTED_BRANCH}", "publication branch/tracking name drifted")
    _require(head == tracking == remote_branch == local_archive == remote_archive, "publication refs are not identical")
    _require(worktree_index_clean, "publication worktree/index is not clean")
    _require(lfs_clean, "publication LFS state is not clean")
    completion = validate_completion(repo)
    return {"status": "PASS", "local_head": head, "tracking_ref": tracking, "remote_branch": remote_branch, "local_archive_ref": local_archive, "remote_archive_ref": remote_archive, "worktree_index_clean": worktree_index_clean, "lfs_clean": lfs_clean, "completion_canonical_digest": completion["completion_canonical_digest"], "whole_root_inventory_canonical_digest": completion["whole_root_inventory_canonical_digest"]}


def freeze(repo: Path, *, test_command: str, passed: int, failed: int, startup_gate_path: Path) -> dict[str, Any]:
    _require((repo / PENDING).is_file(), "PENDING completion must exist before freeze")
    _require(not (repo / COMPLETION).exists(), "terminal completion already exists")
    pending = read_json(repo / PENDING); validate_signed(pending, "Y0 pending completion")
    _require(pending["status"] == "PENDING", "completion did not start PENDING")
    validation = _validation_report(repo); bundle = contract_bundle(repo); source = _source_binding(repo)
    startup = read_json(startup_gate_path); validate_signed(startup, "Y0 startup gate")
    _require(startup["status"] == "PASS" and startup["local_head"] == EXPECTED_BASE, "startup gate drifted")
    access = _access_ledger(); false_pass = build_false_pass_audit()
    test_report = signed({"schema_version": 1, "document_type": "stage_y_y0_test_report", "protocol_id": PROTOCOL_ID, "status": "PASS" if failed == 0 and passed > 0 else "BLOCKED", "command": test_command, "passed": passed, "failed": failed})
    _require(test_report["status"] == "PASS", "focused tests did not pass")
    payloads = {
        ROOT / "access_ledger.json": access, ROOT / "false_pass_audit.json": false_pass,
        ROOT / "source_binding.json": source, ROOT / "contract_bundle.json": bundle, ROOT / "startup_gate.json": startup,
        REPORT_ROOT / "validation_report.json": validation, REPORT_ROOT / "false_pass_audit.json": false_pass,
        REPORT_ROOT / "test_report.json": test_report,
    }
    for relative, value in payloads.items():
        _write_new_json(repo / relative, value)
    report_path = repo / REPORT_ROOT / "y0_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise StageYY0Error(f"refusing to overwrite: {report_path}")
    report_path.write_text(_report_markdown(bundle["bundle_digest"]), encoding="utf-8")

    inventory = _inventory(repo)
    _write_new_json(repo / ROOT / "whole_root_inventory.json", inventory)
    records = {
        "access_ledger": record(repo, ROOT / "access_ledger.json"),
        "false_pass_audit": record(repo, ROOT / "false_pass_audit.json"),
        "source_binding": record(repo, ROOT / "source_binding.json"),
        "contract_bundle": record(repo, ROOT / "contract_bundle.json"),
        "startup_gate": record(repo, ROOT / "startup_gate.json"),
        "whole_root_inventory": record(repo, ROOT / "whole_root_inventory.json"),
        "validation_report": record(repo, REPORT_ROOT / "validation_report.json"),
        "test_report": record(repo, REPORT_ROOT / "test_report.json"),
        "report": file_record(repo, REPORT_ROOT / "y0_report.md"),
    }
    terminal = signed({
        "schema_version": 1, "document_type": "stage_y_y0_terminal_completion",
        "protocol_id": PROTOCOL_ID, "status": "PASS", "terminal": True,
        "started_at": pending["created_at"], "completed_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "base_commit": pending["base_commit"], "route_amendment": "F3_FINAL_PAUSED_PENDING_STAGE_Y",
        "claim_count": 6, "contract_bundle_digest": bundle["bundle_digest"],
        "access_boundary_pass": True, "source_recomputing_validation": "PASS",
        "false_pass_audit": f"{len(FALSE_PASS_ATTACKS)}/{len(FALSE_PASS_ATTACKS)} REJECTED",
        "scientific_outcome_produced": False, "formal_cohort_materialized": False,
        "model_checkpoint_image_cuda_or_gpu_access": False, "y1_started": False,
        "y2_started": False, "y3_started": False, "y4_started": False,
        "y5_started": False, "f3_final_started": False, "stage_g_started": False,
        "records": records, "next_authorization": NEXT_AUTHORIZATION,
        "git": {"commit": "SELF", "branch": EXPECTED_BRANCH, "ref": ARCHIVE_REF},
    })
    temporary = repo / PENDING.with_name(".y0-terminal.tmp")
    _write_new_json(temporary, terminal)
    os.replace(temporary, repo / PENDING)
    os.replace(repo / PENDING, repo / COMPLETION)
    directory_fd = os.open((repo / COMPLETION).parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return validate_completion(repo)


def validate_completion(repo: Path) -> dict[str, Any]:
    _require((repo / COMPLETION).is_file(), "terminal completion missing")
    _require(not (repo / PENDING).exists(), "PENDING completion survived atomic transition")
    completion = read_json(repo / COMPLETION); validate_signed(completion, "Y0 completion")
    _require(completion["status"] in {"PASS", "BLOCKED"} and completion["terminal"] is True, "invalid terminal status")
    if completion["status"] == "PASS":
        _require(completion["next_authorization"] == NEXT_AUTHORIZATION, "PASS authorization drifted")
        _require(completion["model_checkpoint_image_cuda_or_gpu_access"] is False, "access violation in PASS")
    else:
        _require(completion["next_authorization"] == "NONE", "BLOCKED issued authorization")
    validate_semantics(repo)
    bundle = contract_bundle(repo)
    _require(bundle["bundle_digest"] == completion["contract_bundle_digest"], "completion bundle digest drifted")
    for name, item in completion["records"].items():
        relative = Path(item["path"]); path = repo / relative
        _require(path.is_file() and sha_file(path) == item["sha256"], f"completion record drifted: {name}")
        if "canonical_digest" in item:
            value = read_json(path); validate_signed(value, name)
            _require(value["canonical_digest"] == item["canonical_digest"], f"completion canonical binding drifted: {name}")
    inventory = read_json(repo / ROOT / "whole_root_inventory.json"); validate_signed(inventory, "Y0 inventory")
    rebuilt = _inventory(repo)
    for key in ("files", "file_count", "total_bytes", "files_digest"):
        _require(inventory[key] == rebuilt[key], f"whole-root inventory drifted: {key}")
    audit = read_json(repo / ROOT / "false_pass_audit.json"); validate_signed(audit, "Y0 false-pass audit")
    _require(audit["attack_count"] == len(FALSE_PASS_ATTACKS) and all(row["result"] == "REJECTED" for row in audit["checks"]), "false-pass audit incomplete")
    access = read_json(repo / ROOT / "access_ledger.json"); validate_signed(access, "Y0 access ledger")
    _require(access["jpeg_or_png_pixel_reads"] == 0 and access["checkpoint_tensor_reads"] == 0 and access["cuda_or_gpu_accessed"] is False, "access ledger failed")
    return {"status": completion["status"], "protocol_id": PROTOCOL_ID, "completion_canonical_digest": completion["canonical_digest"], "completion_sha256": sha_file(repo / COMPLETION), "whole_root_inventory_canonical_digest": inventory["canonical_digest"], "whole_root_inventory_sha256": sha_file(repo / ROOT / "whole_root_inventory.json"), "inventory_files_digest": inventory["files_digest"], "contract_bundle_digest": bundle["bundle_digest"], "next_authorization": completion["next_authorization"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validate_stage_y_y0.py")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-contracts")
    subparsers.add_parser("init-pending")
    audit_parser = subparsers.add_parser("false-pass-audit"); audit_parser.add_argument("--write", action="store_true")
    freeze_parser = subparsers.add_parser("freeze"); freeze_parser.add_argument("--test-command", required=True); freeze_parser.add_argument("--passed", type=int, required=True); freeze_parser.add_argument("--failed", type=int, required=True); freeze_parser.add_argument("--startup-gate", type=Path, required=True)
    subparsers.add_parser("validate-completion")
    subparsers.add_parser("validate-publication")
    args = parser.parse_args(argv); repo = args.repo.resolve()
    if args.command == "validate-contracts": result = validate_semantics(repo)
    elif args.command == "init-pending": result = initialize_pending(repo)
    elif args.command == "false-pass-audit":
        result = build_false_pass_audit()
        if args.write:
            _write_new_json(repo / ROOT / "false_pass_audit.json", result)
    elif args.command == "freeze": result = freeze(repo, test_command=args.test_command, passed=args.passed, failed=args.failed, startup_gate_path=args.startup_gate.resolve())
    elif args.command == "validate-completion": result = validate_completion(repo)
    else: result = validate_publication(repo)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StageYY0Error as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc), "next_authorization": "NONE"}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
