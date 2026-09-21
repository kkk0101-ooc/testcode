#!/usr/bin/env python3
"""Validate Step-5.2 freeze, phase separation, coverage, and terminal links."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import sha256, verify, verify_binding  # noqa: E402
from vit_pcia_pefa.src.step5_2_queue import audit_coverage  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: validate_step5_2.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json"); manifest = verify(run / "task_manifest.json")
    source = verify(run / "source_binding.json"); freeze = verify(run / "freeze_receipt.json"); verify_binding(source, REPO)
    if protocol["stage"] != "ENDPOINT_PCIA_PEFA_STEP5_2" or protocol["evaluation_access"] or protocol["step6_authorized"]:
        raise RuntimeError("STAGE_OR_DATA_BOUNDARY")
    if freeze["intervention_started"] or source["outcome_reads_before_freeze"] != 0 or source["evaluation_member_reads"] != 0:
        raise RuntimeError("PROSPECTIVE_BOUNDARY")
    models = list(protocol["models"]); p1 = protocol["p1"]
    expected_p0 = len(models); expected_p1 = int(p1["image_task_count"])
    if len(manifest["p0_tasks"]) != expected_p0 or len(manifest["prediction_tasks"]) != expected_p1 or manifest["prediction_tasks"] != manifest["intervention_tasks"]:
        raise RuntimeError("TASK_COVERAGE")
    expected_orders = list(range(*p1["split_order_range"]))
    observed_pairs = {(row["model_id"], int(row["split_order"])) for row in manifest["prediction_tasks"]}
    expected_pairs = {(model, order) for model in models for order in expected_orders}
    if ({row["model_id"] for row in manifest["p0_tasks"]} != set(models)
            or observed_pairs != expected_pairs
            or any(row["split"] != p1["split"] for row in manifest["prediction_tasks"])):
        raise RuntimeError("P1_SELECTION")
    if protocol.get("model_scope") == "DEIT_S16_ONLY":
        if (models != ["deit_s16"] or protocol.get("cross_architecture_replication_claim")
                or expected_p0 != 1 or expected_p1 != 8
                or protocol["integration"]["real_coarse_nodes"] != 1024
                or protocol["integration"]["real_fine_nodes"] != 2048
                or protocol["numeric"]["real"].get("convergence_rule")
                != "global_relative_l2_or_absolute_near_zero"):
            raise RuntimeError("MODEL_SCOPE_BOUNDARY")
    status = "PASS_STEP5_2_FREEZE"
    if (run / "p0_rebind_receipt.json").exists():
        p0_rebind = verify(run / "p0_rebind_receipt.json")
        if p0_rebind["status"] != "PASS_P0_REBOUND_FROM_R2" or not p0_rebind["prediction_authorized_by_gate"]:
            raise RuntimeError("P0_REBIND")
        status = "PASS_STEP5_2_P0_REBIND"
    if (run / "p0_decision.json").exists():
        p0 = verify(run / "p0_decision.json"); status = "PASS_STEP5_2_P0" if p0["status"] == "PASS_P0_NUMERICAL_FEASIBILITY" else "INVALID_STEP5_2_P0"
    if (run / "prediction_freeze_receipt.json").exists():
        prediction_freeze = verify(run / "prediction_freeze_receipt.json")
        prediction_coverage = audit_coverage(run / "prediction_queue.sqlite3", run / "prediction_results", "prediction")
        if prediction_freeze["status"] != "PASS_PREDICTIONS_FROZEN_BEFORE_INTERVENTION" or prediction_coverage["status"] != "PASS":
            raise RuntimeError("PREDICTION_FREEZE_COVERAGE")
        for row in prediction_freeze["predictions"]:
            if sha256(run / row["result_path"]) != row["result_sha256"] or sha256(run / row["array_path"]) != row["array_sha256"]:
                raise RuntimeError("PREDICTION_ARTIFACT_DRIFT")
        status = "PASS_STEP5_2_PREDICTIONS_FROZEN"
    if (run / "terminal_receipt.json").exists():
        terminal = verify(run / "terminal_receipt.json"); claims = verify(run / "claim_decisions.json")
        analysis = verify(run / "analysis.json"); resources = verify(run / "resource_ledger.json"); evidence = verify(run / "evidence_index.json")
        intervention_coverage = audit_coverage(run / "intervention_queue.sqlite3", run / "intervention_results", "intervention")
        expected = {"NATURAL_GAP_PREDICTION", "PREDICTIVE_INCREMENT_OVER_SIMPLE_BASELINES", "DOSE_ORDER",
                    "DIRECTIONAL_ALIGNMENT", "NUMERICAL_VALIDITY_AND_COST"}
        if {row["claim_id"] for row in claims["decisions"]} != expected: raise RuntimeError("CLAIM_COVERAGE")
        if (intervention_coverage["status"] != "PASS"
                or terminal["claim_decisions_digest"] != claims["canonical_digest"]
                or terminal["analysis_digest"] != analysis["canonical_digest"]
                or terminal["resource_ledger_digest"] != resources["canonical_digest"]
                or terminal["evidence_index_digest"] != evidence["canonical_digest"]):
            raise RuntimeError("TERMINAL_LINKAGE")
        if terminal["evaluation_access"] or terminal["step6_started"] or terminal["holdout_started"]:
            raise RuntimeError("TERMINAL_BOUNDARY")
        status = "PASS_STEP5_2_TERMINAL_COVERAGE"
    print(json.dumps({"status": status, "protocol_digest": protocol["canonical_digest"],
        "models": models, "p0_task_count": expected_p0, "p1_image_task_count": expected_p1,
        "cross_architecture_replication_claim": protocol.get("cross_architecture_replication_claim"),
        "evaluation_access": False, "step6_started": False}, sort_keys=True))
    return 0 if status.startswith("PASS") else 3


if __name__ == "__main__": raise SystemExit(main())
