#!/usr/bin/env python3
"""Freeze NNS-CST N1 contracts before executing analytic fixtures."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVIDENCE = HERE / "evidence/n1_r2_20260922"
R2_CONFIG = HERE / "config/n1_r2"
R2_MANIFESTS = HERE / "manifests/n1_r2"
GRAPH_REVISION = "timm_unfused_vit_n1_contract_v1"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_or_identical(path: Path, value: Any) -> None:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError("frozen artifact drift: " + str(path))
    if not path.exists():
        path.write_bytes(raw)


def port(model: Dict[str, Any], block: int, branch: str, producer: str,
         consumer: str, slot: int, role: str, token_set: str) -> Dict[str, Any]:
    invocation = 0
    identity = {
        "model_id": model["model_id"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "graph_revision": GRAPH_REVISION,
        "block_index": block,
        "branch": branch,
        "producer_tensor": producer,
        "consumer_operator": consumer,
        "input_slot": slot,
        "invocation_index": invocation,
    }
    identity_text = "|".join(str(identity[key]) for key in (
        "model_id", "checkpoint_sha256", "graph_revision", "block_index",
        "branch", "producer_tensor", "consumer_operator", "input_slot", "invocation_index"))
    return {
        "port_id": model["model_id"] + ":" + hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:20],
        **identity,
        "tensor_role": role,
        "token_set": token_set,
    }


def build_port_registry() -> Dict[str, Any]:
    models = [
        {
            "model_id": "vit_b16",
            "timm_name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "checkpoint_sha256": "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d",
            "blocks": 12, "heads": 12, "embedding_dim": 768,
        },
        {
            "model_id": "deit_s16",
            "timm_name": "deit_small_patch16_224.fb_in1k",
            "checkpoint_sha256": "cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f",
            "blocks": 12, "heads": 6, "embedding_dim": 384,
        },
    ]
    records: List[Dict[str, Any]] = []
    for model in models:
        records.append(port(model, -1, "embedding", "patch_embed.proj#0", "positional_add", 0,
                            "patch_embedding_residual_stream", "CLS_PLUS_PATCH_14X14"))
        for block in range(model["blocks"]):
            prefix = "blocks.%d" % block
            records.extend([
                port(model, block, "attention", prefix + ".input#0", prefix + ".norm1", 0,
                     "pre_layernorm1_residual", "CLS_PLUS_PATCH_14X14"),
                port(model, block, "attention", prefix + ".norm1#0", prefix + ".attn.qkv", 0,
                     "post_layernorm1", "CLS_PLUS_PATCH_14X14"),
            ])
            for head in range(model["heads"]):
                hp = prefix + ".attn.head%d" % head
                records.extend([
                    port(model, block, "attention_head_%d" % head, prefix + ".norm1#0", hp + ".query_dot", 0,
                         "attention_head_query_input", "CLS_PLUS_PATCH_14X14"),
                    port(model, block, "attention_head_%d" % head, prefix + ".norm1#0", hp + ".key_dot", 1,
                         "attention_head_key_input", "CLS_PLUS_PATCH_14X14"),
                    port(model, block, "attention_head_%d" % head, prefix + ".norm1#0", hp + ".value_projection", 2,
                         "attention_head_value_input", "CLS_PLUS_PATCH_14X14"),
                    port(model, block, "attention_head_%d" % head, hp + ".value#0", hp + ".weighted_sum", 1,
                         "attention_head_value", "CLS_PLUS_PATCH_14X14"),
                    port(model, block, "attention_head_%d" % head, hp + ".pre_projection#0", prefix + ".attn.proj", 0,
                         "attention_head_pre_projection_output", "CLS_PLUS_PATCH_14X14"),
                ])
            records.extend([
                port(model, block, "attention", prefix + ".attn.proj#0", prefix + ".attn_residual_add", 1,
                     "attention_merged_output", "CLS_PLUS_PATCH_14X14"),
                port(model, block, "attention_identity", prefix + ".input#0", prefix + ".attn_residual_add", 0,
                     "attention_residual_use_edge", "CLS_PLUS_PATCH_14X14"),
                port(model, block, "mlp", prefix + ".attn_residual_add#0", prefix + ".norm2", 0,
                     "pre_layernorm2_residual", "CLS_PLUS_PATCH_14X14"),
                port(model, block, "mlp", prefix + ".norm2#0", prefix + ".mlp.fc1", 0,
                     "post_layernorm2", "CLS_PLUS_PATCH_14X14"),
                port(model, block, "mlp", prefix + ".mlp.fc1#0", prefix + ".mlp.act", 0,
                     "mlp_expand", "CLS_PLUS_PATCH_14X14"),
                port(model, block, "mlp", prefix + ".mlp.act#0", prefix + ".mlp.fc2", 0,
                     "mlp_activation", "CLS_PLUS_PATCH_14X14"),
                port(model, block, "mlp", prefix + ".mlp.fc2#0", prefix + ".mlp_residual_add", 1,
                     "mlp_contract_output", "CLS_PLUS_PATCH_14X14"),
                port(model, block, "mlp_identity", prefix + ".attn_residual_add#0", prefix + ".mlp_residual_add", 0,
                     "mlp_residual_use_edge", "CLS_PLUS_PATCH_14X14"),
            ])
        final = model["blocks"] - 1
        records.extend([
            port(model, final, "readout", "blocks.%d.mlp_residual_add#0" % final, "cls_select", 0,
                 "cls_token", "CLS_ONLY"),
            port(model, final, "readout", "cls_select#0", "final_norm", 0,
                 "final_norm_input", "CLS_ONLY"),
            port(model, final, "readout", "final_norm#0", "classifier", 0,
                 "final_norm_output", "CLS_ONLY"),
            port(model, final, "readout", "classifier.logits#0", "classifier_margin", 0,
                 "classifier_margin", "SCALAR_TARGET_MINUS_REFERENCE"),
        ])
    return {
        "schema_version": 1,
        "document_type": "nns_cst_port_registry",
        "stage_id": "N1",
        "status": "FROZEN_BEFORE_N1_TEST_EXECUTION",
        "graph_revision": GRAPH_REVISION,
        "identity_fields": [
            "model_id", "checkpoint_sha256", "graph_revision", "block_index", "branch",
            "producer_tensor", "consumer_operator", "input_slot", "invocation_index",
        ],
        "models": models,
        "record_count": len(records),
        "records": records,
    }


def build_subspace_contract() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "document_type": "nns_cst_subspace_contract",
        "stage_id": "N1",
        "status": "FROZEN_BEFORE_N1_TEST_EXECUTION",
        "activation_layout": {
            "default": "BATCH_TOKEN_CHANNEL",
            "shape_symbol": ["B", "T_l", "D_l"],
            "dtype": "float32_production_float64_reduction",
            "token_sets": ["CLS_ONLY", "PATCH_ONLY", "CLS_PLUS_PATCH_14X14"],
            "missing_token_policy": "MISSING_TOKEN",
            "cross_topology_policy": "EXPLICIT_BIJECTIVE_TOKEN_MAP_OR_FAIL",
        },
        "paired_difference": {
            "formula": "delta_a[i,t]=a_semantic_plus[i,t]-a_semantic_minus[i,t]",
            "pairing_unit": "source_group_id",
            "centering": "WITHIN_PAIR_SUBTRACTION_NO_ACROSS_PAIR_MEAN_REMOVAL",
            "token_weight_formula": "sum_t w[i,t]*delta_a[i,t] with sum_t w[i,t]=1 per source",
            "source_weight_formula": "equal weight per source after token aggregation",
        },
        "control_covariance": {
            "source": "intensity_matched_null_pair_differences",
            "operation": "project semantic differences onto orthogonal complement of nuisance span",
            "span_tolerance": 1e-10,
            "singular_policy": "NEAR_SINGULAR_CONTROL_COVARIANCE",
            "semantic_removed_policy": "NO_SEMANTIC_AFTER_CONTROL",
        },
        "estimator": {
            "name": "RESIDUALIZED_PAIRED_SECOND_MOMENT_EIGENSPACE",
            "formula": "top-r eigenspace of sum_i w_i r_i r_i^T / sum_i w_i after nuisance projection",
            "rank_candidates": [1, 2, 4, 8, 16],
            "ridge_candidates": [0.0, 1e-08, 1e-06],
            "selection_stage": "N2_PILOT_ONLY",
            "production_reduction": "float64_math_fsum",
        },
        "basis_invariants": {
            "projection": "P=U U^T for row-orthonormal U",
            "similarity": "trace(P_A P_B)/r",
            "distance": "frobenius(P_A-P_B)/sqrt(max(r_A,r_B))",
            "individual_basis_vector_claims_allowed": False,
        },
        "frozen_readout": {
            "fit_split": "probe_train_or_N2_pilot_discovery_only",
            "method": "projected_class_mean_difference",
            "validation_role": "selection_only_no_refit",
            "serialization": "closed_json_float64_decimal_with_contract_hash",
            "confirmation_refit_allowed": False,
        },
        "dimension_and_topology": {
            "different_D_l": "independent native-dimensional projectors plus registered transport map",
            "same_topology": "token_id_exact_match",
            "different_topology": "frozen_bijective_token_map_required",
            "implicit_resize_or_token_drop_allowed": False,
        },
        "typed_failures": [
            "RANK_EXCEEDS_DIMENSION", "RANK_DEFICIENT", "ZERO_PAIR_DIFFERENCE",
            "NO_SEMANTIC_AFTER_CONTROL", "NEAR_SINGULAR_CONTROL_COVARIANCE",
            "NONFINITE_VALUE", "MISSING_TOKEN", "TOKEN_ALIGNMENT_ERROR",
            "PORT_IDENTITY_MISMATCH", "DUPLICATE_PORT", "CUT_MISSING_EDGE",
            "OFF_MANIFOLD_INTERVENTION",
        ],
    }


def build_fixture_registry() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "document_type": "nns_cst_n1_fixture_registry",
        "stage_id": "N1",
        "status": "FROZEN_BEFORE_N1_TEST_EXECUTION",
        "numeric_contract": {"device": "CPU", "dtype": "float64", "atol": 1e-09, "rtol": 1e-07},
        "fixtures": [
            {"fixture_id": "ROTATED_TWO_DIMENSIONAL_SUBSPACE", "required_assertions": ["PROJECTOR_INVARIANCE", "PROJECTION_INVARIANCE", "SIMILARITY_ONE"]},
            {"fixture_id": "DISTRIBUTED_MULTI_HEAD_SIGNAL", "required_assertions": ["EACH_SINGLE_HEAD_INCOMPLETE", "JOINT_SUBSPACE_COMPLETE"]},
            {"fixture_id": "BRIGHTNESS_CONTROL_RESIDUALIZATION", "required_assertions": ["NUISANCE_REMOVED", "SEMANTIC_RETAINED"]},
            {"fixture_id": "ANALYTIC_TRANSPORT", "required_assertions": ["IDENTITY", "LINEAR", "RESIDUAL", "TWO_BRANCH"]},
            {"fixture_id": "SEMANTIC_COMPLEMENT_COMPLETE_CUT", "required_assertions": ["COORDINATE_CLOSURE", "SCALAR_CLOSURE"]},
            {"fixture_id": "INTERVENTION_ANALYTICS", "required_assertions": ["INJECTION", "ERASURE", "RANDOM_SUBSPACE", "ORTHOGONAL_COMPLEMENT"]},
            {"fixture_id": "TYPED_FAILURES", "required_assertions": ["RANK_TOO_LARGE", "RANK_DEFICIENT", "ZERO_DIFFERENCE", "NONFINITE", "NEAR_SINGULAR_CONTROL", "NO_SEMANTIC_AFTER_CONTROL", "MISSING_TOKEN", "OFF_MANIFOLD"]},
        ],
        "mutations": [
            {"mutation_id": "OMIT_EDGE", "expected_failure": "CUT_MISSING_EDGE"},
            {"mutation_id": "DUPLICATE_EDGE", "expected_failure": "DUPLICATE_PORT"},
            {"mutation_id": "SWAP_Q_K_V_SLOT", "expected_failure": "PORT_IDENTITY_MISMATCH"},
            {"mutation_id": "WRONG_TOKEN_ALIGNMENT", "expected_failure": "TOKEN_ALIGNMENT_ERROR"},
        ],
        "determinism_probe": {
            "process_count": 2,
            "pythonhashseed_values": ["0", "99173"],
            "comparison": "canonical_sha256_of_scientific_payload",
        },
        "microbenchmark": {
            "iterations": 200,
            "operations": ["rank2_subspace_estimation", "analytic_transport", "semantic_complement_attribution"],
            "report": ["wall_seconds", "cpu_seconds", "peak_rss_kib", "iterations_per_second"],
        },
    }


def main() -> None:
    port_registry = build_port_registry()
    subspace_contract = build_subspace_contract()
    fixture_registry = build_fixture_registry()
    write_new_or_identical(R2_MANIFESTS / "model_port_registry.json", port_registry)
    write_new_or_identical(R2_CONFIG / "subspace_contract.json", subspace_contract)
    write_new_or_identical(R2_MANIFESTS / "n1_fixture_registry.json", fixture_registry)
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=str(REPO)).decode().strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
    receipt = {
        "schema_version": 1,
        "document_type": "nns_cst_n1_r2_authorization_and_freeze_receipt",
        "stage_id": "N1",
        "status": "FROZEN_BEFORE_N1_TEST_EXECUTION",
        "authorization": {
            "source": "explicit_current_thread_user_request",
            "request": "执行testcode/vit_nns_cst/执行规划.md的N1",
            "authorized_stage": "N1",
            "n2_authorized": False,
        },
        "predecessor": {
            "n0_decision_sha256": sha256(HERE / "n0_decision.json"),
            "n0_decision": "GO_REQUEST_N1",
            "n0_validator_reexecuted": True,
            "n0_validator_status": "PASS",
            "legacy_n1_authorized_field_preserved": False,
            "n1_r1_terminal_path": "evidence/n1_r1_20260922/terminal_receipt.json",
            "n1_r1_terminal_sha256": sha256(HERE / "evidence/n1_r1_20260922/terminal_receipt.json"),
            "n1_r1_status": "FAIL_N1_R1_SCHEMA_FREEZE_INCOMPLETE",
        },
        "authority": {
            "execution_plan_sha256": sha256(HERE / "执行规划.md"),
            "research_goal_sha256": sha256(HERE / "研究目标.md"),
            "mathematical_contract_sha256": sha256(HERE / "N1数学合同.md"),
        },
        "frozen_artifacts": [],
        "repository": {"branch": branch, "head_commit": head},
        "execution_boundary": {
            "formal_image_decodes": 0, "model_loads": 0, "checkpoint_deserializations": 0,
            "gpu_or_cuda_uses": 0, "training_runs": 0, "inference_runs": 0,
            "formal_outcome_reads": 0, "n2_started": False,
        },
    }
    frozen_paths = [
        "config/n1_r2/subspace_contract.json",
        "manifests/n1_r2/model_port_registry.json",
        "manifests/n1_r2/n1_fixture_registry.json",
        "schemas/n1_port_registry.schema.json",
        "schemas/n1_subspace_contract.schema.json",
        "schemas/n1_fixture_registry.schema.json",
        "schemas/n1_analytic_results.schema.json",
        "schemas/n1_terminal_receipt.schema.json",
        "N1数学合同.md", "N1-R2恢复修订.md",
        "src/nns_cst/core.py", "src/nns_cst/interfaces.py",
        "run_n1.py", "n1_determinism_probe.py", "validate_n1.py",
        "tests/test_n1.py",
    ]
    receipt["frozen_artifacts"] = [
        {"path": path, "sha256": sha256(HERE / path)} for path in frozen_paths]
    write_new_or_identical(EVIDENCE / "freeze_receipt.json", receipt)
    print(json.dumps({
        "status": "PASS_N1_CONTRACTS_FROZEN",
        "port_count": port_registry["record_count"],
        "fixture_count": len(fixture_registry["fixtures"]),
        "mutation_count": len(fixture_registry["mutations"]),
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
