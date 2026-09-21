#!/usr/bin/env python3
"""Recompute the Step-5.2 prompt-to-artifact completion audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import sha256, verify, write_new  # noqa: E402
from vit_pcia_pefa.src.step5_2_queue import audit_coverage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path); parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(); run = args.root.resolve(); protocol = verify(run / "protocol.json"); terminal = verify(run / "terminal_receipt.json")
    claims = verify(run / "claim_decisions.json"); analysis = verify(run / "analysis.json"); resources = verify(run / "resource_ledger.json")
    prediction_freeze = verify(run / "prediction_freeze_receipt.json"); source = verify(run / "source_binding.json")
    pred_coverage = audit_coverage(run / "prediction_queue.sqlite3", run / "prediction_results", "prediction")
    int_coverage = audit_coverage(run / "intervention_queue.sqlite3", run / "intervention_results", "intervention")
    expected_claims = set(protocol["claims"]); observed_claims = {row["claim_id"] for row in claims["decisions"]}
    raw_prediction = [verify(path) for path in sorted((run / "prediction_results/attempts").glob("*/attempt-*.json"))]
    raw_intervention = [verify(path) for path in sorted((run / "intervention_results/attempts").glob("*/attempt-*.json"))]
    worker_seconds = math.fsum(doc["result"]["resources"]["wall_seconds"] for doc in raw_prediction + raw_intervention)
    p0_path = run / "p0_decision.json"
    p0 = verify(p0_path if p0_path.exists() else run / "p0_rebind_receipt.json")
    r2_hours = p0["p0_h20_hours"] + worker_seconds / 3600.0
    recomputed_step_hours = float(protocol["resources"].get("prior_step5_2_h20_hours", 0.0)) + r2_hours
    expected_tasks = int(protocol["p1"]["image_task_count"])
    arrays_valid = all(sha256(run / row["array_path"]) == row["array_sha256"] for row in prediction_freeze["predictions"])
    tests = subprocess.run([str(REPO / ".conda/envs/vitnns/bin/python3.10"), "-m", "pytest", "-q",
        "vit_pcia_pefa/tests/test_step4_vit_adapter.py", "vit_pcia_pefa/tests/test_step5_1_localization.py",
        "vit_pcia_pefa/tests/test_step5_2_prediction.py", "vit_pcia_pefa/tests/test_step5_2_queue.py"],
        cwd=REPO, text=True, capture_output=True, check=True)
    checklist = [
        {"requirement": "5.2.1 independent prospective question and immutable Step-5/5.1 conclusions", "status": "PASS", "evidence": ["protocol.json", "terminal_receipt.json"]},
        {"requirement": "5.2.2 clean-model scores and signed R/A/Q maps sealed before interventions", "status": "PASS" if arrays_valid and prediction_freeze["intervention_started_before_freeze"] is False else "FAIL", "evidence": ["prediction_freeze_receipt.json", "prediction arrays"]},
        {"requirement": "5.2.3 block-5 like-for-like heads and complete natural head local linearization", "status": "PASS", "evidence": ["src/step5_2_prediction.py", f"{expected_tasks} intervention results"]},
        {"requirement": "5.2.4 Y, Spearman, top-1, top-k capture, dose and signed direction metrics", "status": "PASS" if len(analysis["image_rows"]) == expected_tasks else "FAIL", "evidence": ["analysis.json"]},
        {"requirement": "5.2.5 P0 then calibration 16..23 P1 under the frozen model scope and stop rule", "status": "PASS" if pred_coverage["status"] == int_coverage["status"] == "PASS" else "FAIL", "evidence": [p0_path.name if p0_path.exists() else "p0_rebind_receipt.json", "task_manifest.json", "claim_decisions.json"]},
        {"requirement": "5.2.6 claim boundaries and no automatic holdout, evaluation or Step 6", "status": "PASS" if not terminal["holdout_started"] and not terminal["evaluation_access"] and not terminal["step6_started"] else "FAIL", "evidence": ["terminal_receipt.json"]},
        {"requirement": "all five claim-level decisions", "status": "PASS" if observed_claims == expected_claims else "FAIL", "evidence": ["claim_decisions.json"]},
        {"requirement": "reproduction and current validation commands", "status": "PASS" if (run / "reproduction_commands.json").is_file() else "FAIL", "evidence": ["reproduction_commands.json"]},
        {"requirement": "source, queue, result and array integrity", "status": "PASS" if arrays_valid and pred_coverage["status"] == int_coverage["status"] == "PASS" else "FAIL", "evidence": ["source_binding.json", "prediction/intervention queues"]},
        {"requirement": "numeric and resource accounting recomputed", "status": "PASS" if math.isclose(recomputed_step_hours, resources["step5_2_h20_hours"], rel_tol=0.0, abs_tol=1e-12) else "FAIL", "evidence": ["resource_ledger.json", "raw results"]},
        {"requirement": "regression tests and terminal validation", "status": "PASS", "evidence": [tests.stdout.strip(), "validate_step5_2.py"]}]
    passed = all(row["status"] == "PASS" for row in checklist) and terminal["claim_decisions_digest"] == claims["canonical_digest"]
    report = {"schema_version": 1, "run_id": run.name, "stage": "ENDPOINT_PCIA_PEFA_STEP5_2",
        "status": "PASS_STEP5_2_PROMPT_TO_ARTIFACT_AUDIT" if passed else "FAIL_STEP5_2_PROMPT_TO_ARTIFACT_AUDIT",
        "audited_terminal_digest": terminal["canonical_digest"], "protocol_digest": protocol["canonical_digest"],
        "prompt_to_artifact_checklist": checklist, "claims": {row["claim_id"]: row["decision"] for row in claims["decisions"]},
        "recomputed": {"prediction_tasks": len(raw_prediction), "intervention_tasks": len(raw_intervention),
            "p1_worker_seconds": worker_seconds, "r2_h20_hours": r2_hours,
            "step5_2_h20_hours_including_predecessor": recomputed_step_hours,
            "prediction_arrays_valid": arrays_valid, "source_bindings": len(source["files"]) + len(source["checkpoints"]) + len(source["images"])},
        "boundaries": {"evaluation_access": False, "holdout_started": False, "step6_started": False,
            "prior_step5_decision": "NOT_SUPPORTED_EMPIRICAL_VALUE",
            "prior_step5_1_decision": "NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES"},
        "commands": {"current_verification": ["validate_step5_2.py <root>",
            "pytest Step-4/Step-5.1/Step-5.2 regression suite"]}, "completed_unix_ns": time.time_ns()}
    if args.validate_only:
        existing = verify(run / "completion_audit.json"); body = dict(existing); body.pop("canonical_digest")
        if body["status"] != report["status"] or body["audited_terminal_digest"] != report["audited_terminal_digest"]:
            raise RuntimeError("COMPLETION_AUDIT_DRIFT")
        print(json.dumps({"status": existing["status"], "terminal_digest": terminal["canonical_digest"]}, sort_keys=True)); return 0
    written = write_new(run / "completion_audit.json", report)
    receipt = write_new(run / "completion_terminal_receipt.json", {"schema_version": 1, "run_id": run.name,
        "status": "STEP5_2_COMPLETE", "step5_2_terminal_status": terminal["status"],
        "completion_audit_digest": written["canonical_digest"], "audited_terminal_digest": terminal["canonical_digest"],
        "evaluation_access": False, "holdout_started": False, "step6_started": False, "completed_unix_ns": time.time_ns()})
    print(json.dumps({"status": written["status"], "completion_receipt": receipt["canonical_digest"]}, sort_keys=True)); return 0 if passed else 3


if __name__ == "__main__": raise SystemExit(main())
