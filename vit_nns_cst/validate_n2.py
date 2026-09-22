#!/usr/bin/env python3
"""Independent N2 terminal validator; does not import producer code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import jsonschema
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence/n2_r1c_20260922"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(); digest.update(str(value.dtype).encode()); digest.update(b"\0")
    digest.update(str(tuple(value.shape)).encode()); digest.update(b"\0"); digest.update(value.tobytes())
    return digest.hexdigest()


def metrics(base: np.ndarray, other: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    delta = other.astype(np.float64) - base.astype(np.float64)
    grey = delta.mean(2); ring = ndimage.binary_dilation(mask, iterations=3) ^ ndimage.binary_erosion(mask, iterations=3)
    return {"L1": float(np.mean(np.abs(delta))), "L2": float(np.sqrt(np.mean(delta ** 2))),
            "RGB_MEAN_SHIFT": float(np.linalg.norm(delta.mean((0, 1)))),
            "HIGHPASS_L1": float(np.mean(np.abs(grey - ndimage.gaussian_filter(grey, 2.0)))),
            "BOUNDARY_L1": float(np.mean(np.abs(delta[ring]))) if ring.any() else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--finalize-validation", action="store_true")
    args = parser.parse_args()
    contract = json.loads((ROOT / "config/n2_r1/pilot_contract.json").read_text())
    pair = json.loads((EVIDENCE / "pair_manifest.json").read_text())
    qc = json.loads((EVIDENCE / "blind_qc.json").read_text())
    terminal_path = EVIDENCE / "terminal_receipt.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    jsonschema.validate(contract, json.loads((ROOT / "schemas/n2_pilot_contract.schema.json").read_text()))
    jsonschema.validate(pair, json.loads((ROOT / "schemas/n2_pair_manifest.schema.json").read_text()))
    jsonschema.validate(qc, json.loads((ROOT / "schemas/n2_blind_qc.schema.json").read_text()))
    source_path = ROOT.parent / "vitnns_rebuild/evidence/nns_vit_patch_depth/s0_r5_20260914/split_manifest.json"
    source = json.loads(source_path.read_text()); source_rows = source["payload"]["records"]
    by_name = {Path(row["image_local_path"]).name: row for row in source_rows if row["image_local_path"] is not None}
    selected = contract["recipients"] + contract["aquatic_donors"] + contract["non_aquatic_donors"]
    checks = []
    def check(identifier: str, condition: bool, evidence: str) -> None:
        if not condition: raise AssertionError(identifier)
        checks.append({"check_id": identifier, "status": "PASS", "evidence": evidence})
    check("N1_PREDECESSOR", json.loads((ROOT / "evidence/n1_r3_20260922/terminal_receipt.json").read_text())["status"] == "PASS_N1_ANALYTIC_CONTRACT_AND_SKELETON", "N1 R3 terminal receipt")
    check("CONTRACT_CLOSED_AND_BOUND", pair["contract_sha256"] == sha(ROOT / "config/n2_r1/pilot_contract.json"), "schema validation and recomputed SHA-256")
    check("SOURCE_MANIFEST_BOUND", pair["source_manifest_sha256"] == sha(source_path), "recomputed source manifest SHA-256")
    check("PILOT_ONLY_ROSTER", len(set(selected)) == 24 and all(by_name[name]["final_split"] == "pilot" and by_name[name]["use_policy"] == "S1_PILOT_ONLY" for name in selected), "24 unique source records")
    forbidden_groups = {row["source_group_id"] for row in source_rows if row["final_split"] != "pilot"}
    check("NO_FORBIDDEN_SPLIT_INTERSECTION", not forbidden_groups.intersection(by_name[name]["source_group_id"] for name in selected), "source-group set intersection recomputed")
    archive = Path(pair["arrays"]["path"]); archive = ROOT / archive
    check("PAIR_ARRAY_SHA", sha(archive) == pair["arrays"]["sha256"], "pair_inputs.npz SHA-256")
    data = np.load(archive, allow_pickle=False); images = data["images"]; masks = data["masks"]
    check("ARRAY_CONTRACT", images.shape == (8, 4, 224, 224, 3) and images.dtype == np.uint8 and masks.shape == (8, 224, 224) and masks.dtype == np.bool_, "raw NPZ shapes and dtypes")
    max_metric_error = 0.0; max_foreground_difference = 0
    for index, row in enumerate(pair["records"]):
        check(f"ARM_DIGEST_{index}", [array_digest(x) for x in images[index]] == row["arm_array_digests"], "independent raw-array digest")
        base, semantic, null, render = images[index].astype(np.float64) / 255.0
        for observed, recomputed in ((row["semantic_metrics"], metrics(base, semantic, masks[index])), (row["within_null_metrics"], metrics(base, null, masks[index])), (row["render_null_metrics"], metrics(base, render, masks[index]))):
            max_metric_error = max(max_metric_error, max(abs(observed[k] - recomputed[k]) for k in observed))
        interior = ndimage.binary_erosion(masks[index], iterations=4)
        if interior.any(): max_foreground_difference = max(max_foreground_difference, int(np.abs(images[index, 1:].astype(int) - images[index, 0].astype(int))[:, interior].max()))
        logs = [math.log(max(row["semantic_metrics"][k], 1e-8) / max(row["within_null_metrics"][k], 1e-8)) for k in row["semantic_metrics"]]
        check(f"MATCH_SCORE_{index}", abs(math.sqrt(sum(x*x for x in logs)) - row["aggregate_log_ratio_distance"]) < 1e-10, "independent score recomputation")
    check("LOW_LEVEL_METRICS_RECOMPUTED", max_metric_error <= 0.002, f"max quantized-input metric difference={max_metric_error}")
    check("RECIPIENT_FOREGROUND_PRESERVED", max_foreground_difference == 0, f"max eroded-mask uint8 difference={max_foreground_difference}")
    check("NUMERIC_MATCHING_LIMIT", pair["identifiability"]["unmatched_rate"] == 0.25 and pair["identifiability"]["matched_count"] == 6, "six of eight match; exact maximum permitted rate")
    failed = [row for row in qc["pair_reviews"] if not row["artifact_free"]]
    check("BLIND_QC_RECOMPUTED", len(failed) == qc["quality_gate"]["artifact_failed_pair_count"] == 7 and abs(qc["quality_gate"]["artifact_failed_rate"] - 7/8) < 1e-12, "blind QC rows recounted")
    check("SEMANTIC_RELIABILITY_FAILS_CLOSED", qc["semantic_label_reliability"]["completed_independent_reviewers"] < qc["semantic_label_reliability"]["required_independent_reviewers"], "one completed reviewer versus two required")
    check("NO_MODEL_OUTCOME_ACCESS", qc["review_boundary"]["model_checkpoint_deserializations_before_review"] == 0 and not (EVIDENCE / "behavior_results.json").exists(), "QC boundary and absence of behavior artifact")
    check("SEQUENTIAL_STOP_ENFORCED", set(qc["downstream_execution"].values()) == {"NOT_RUN_PRECEDING_IDENTIFIABILITY_FAILURE"}, "all downstream work marked NOT_RUN")
    check("TERMINAL_DECISION", qc["decision"] == "NO_GO_NOT_IDENTIFIABLE" and not qc["quality_gate"]["overall_identifiability_passed"], "failed artifact parity and label reliability")
    if terminal is not None:
        jsonschema.validate(terminal, json.loads((ROOT / "schemas/n2_terminal_receipt.schema.json").read_text()))
        bindings = terminal["artifact_bindings"]
        bound_paths = {
            "execution_plan_sha256": ROOT / "执行规划.md", "research_goal_sha256": ROOT / "研究目标.md",
            "n1_terminal_sha256": ROOT / "evidence/n1_r3_20260922/terminal_receipt.json",
            "protocol_sha256": ROOT / "N2隔离Pilot协议.md", "r1c_amendment_sha256": ROOT / "N2-R1C最终生产器恢复修订.md",
            "contract_sha256": ROOT / "config/n2_r1/pilot_contract.json", "pair_manifest_sha256": EVIDENCE / "pair_manifest.json",
            "pair_inputs_sha256": EVIDENCE / "pair_inputs.npz", "blind_qc_sha256": EVIDENCE / "blind_qc.json",
            "validation_sha256": EVIDENCE / "validation.json",
            "human_completion_audit_sha256": ROOT / "N2完成审计.md",
            "completion_audit_sha256": EVIDENCE / "completion_audit.json",
            "completion_audit_schema_sha256": ROOT / "schemas/n2_completion_audit.schema.json",
            "terminal_schema_sha256": ROOT / "schemas/n2_terminal_receipt.schema.json",
        }
        check("TERMINAL_BINDINGS", all(bindings[key] == sha(path) for key, path in bound_paths.items()), "recomputed all fourteen terminal SHA-256 bindings")
        check("N3_NOT_AUTHORIZED", contract["n3_authorized"] is False and terminal["next_gate"].startswith("STOP_N2"), "contract and terminal next gate")
    result = {"schema_version": 1, "stage_id": "N2", "attempt_id": "N2_R1C_20260922", "status": "PASS_N2_TERMINAL_VALIDATION", "validated_terminal_decision": "NO_GO_NOT_IDENTIFIABLE", "checks": checks, "summary": {"check_count": len(checks), "pair_count": 8, "numeric_matched_count": 6, "artifact_failed_count": 7, "max_metric_recompute_error": max_metric_error, "max_foreground_uint8_difference": max_foreground_difference, "model_outcome_reads": 0}}
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write or args.finalize_validation:
        path = EVIDENCE / "validation.json"
        if path.exists() and not args.finalize_validation:
            raise RuntimeError("VALIDATION_ALREADY_EXISTS")
        if args.finalize_validation:
            if terminal is None or not path.exists():
                raise RuntimeError("FINALIZATION_REQUIRES_BOUND_PRETERMINAL_VALIDATION")
            if terminal["artifact_bindings"]["validation_sha256"] != sha(path):
                raise RuntimeError("PRETERMINAL_VALIDATION_BINDING_DRIFT")
        path.write_text(text)
    print(text, end=""); return 0

if __name__ == "__main__": raise SystemExit(main())
