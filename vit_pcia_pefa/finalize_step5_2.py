#!/usr/bin/env python3
"""Aggregate Step-5.2 P1, apply frozen gates, and seal the terminal state."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import sha256, verify, write_new  # noqa: E402
from vit_pcia_pefa.src.step5_2_prediction import bootstrap_mean_interval, prediction_metrics  # noqa: E402
from vit_pcia_pefa.src.step5_2_queue import audit_coverage  # noqa: E402


def seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")


def load_results(run: Path, phase: str):
    documents = []
    for path in sorted((run / f"{phase}_results/attempts").glob("*/attempt-*.json")):
        document = verify(path); documents.append((path, document))
    return documents


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: finalize_step5_2.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json"); prediction_freeze = verify(run / "prediction_freeze_receipt.json")
    prediction_launch = verify(run / "prediction_launch_terminal.json"); intervention_launch = verify(run / "intervention_launch_terminal.json")
    pred_coverage = audit_coverage(run / "prediction_queue.sqlite3", run / "prediction_results", "prediction")
    int_coverage = audit_coverage(run / "intervention_queue.sqlite3", run / "intervention_results", "intervention")
    if prediction_launch["status"] != "PASS_PREDICTION_TASK_EXECUTION" or intervention_launch["status"] != "PASS_INTERVENTION_TASK_EXECUTION":
        raise RuntimeError("P1_LAUNCH_STATUS")
    if pred_coverage["status"] != "PASS" or int_coverage["status"] != "PASS": raise RuntimeError("P1_COVERAGE")
    predictions = {doc["task_id"]: doc["result"] for _, doc in load_results(run, "prediction")}
    interventions = {doc["task_id"]: doc["result"] for _, doc in load_results(run, "intervention")}
    if set(predictions) != set(interventions) or len(predictions) != protocol["p1"]["image_task_count"]: raise RuntimeError("P1_TASK_MATCHING")
    methods = protocol["p1"]["methods"]; image_rows = []
    for task_id in sorted(predictions):
        prediction = predictions[task_id]; intervention = interventions[task_id]
        if intervention["prediction_freeze_digest"] != prediction_freeze["canonical_digest"]: raise RuntimeError("PREDICTION_FREEZE_LINKAGE")
        lambda0 = sorted((row for row in intervention["rows"] if row["lambda"] == 0.0), key=lambda row: row["head_index"])
        lambda05 = sorted((row for row in intervention["rows"] if row["lambda"] == 0.5), key=lambda row: row["head_index"])
        head_count = int(prediction["head_count"])
        if len(lambda0) != head_count or len(lambda05) != head_count: raise RuntimeError("HEAD_OUTCOME_COVERAGE")
        outcomes = [row["outcome"]["Y"] for row in lambda0]
        metrics = {method: prediction_metrics(prediction["scores"][method], outcomes, int(prediction["top_k"])) for method in methods}
        for value in metrics.values():
            value["spearman_undefined"] = value["spearman"] is None
            value["spearman_for_gate"] = protocol["p1"]["undefined_spearman_for_gate"] if value["spearman"] is None else value["spearman"]
        cosines = [protocol["p1"]["undefined_direction_for_summary"] if row["outcome"]["negative_R_delta_G_cosine"] is None
                   else row["outcome"]["negative_R_delta_G_cosine"] for row in lambda0]
        signs = [protocol["p1"]["undefined_direction_for_summary"] if row["outcome"]["negative_R_delta_G_sign_agreement"] is None
                 else row["outcome"]["negative_R_delta_G_sign_agreement"] for row in lambda0]
        dose_flags = [lambda05[i]["outcome"]["Y"] <= lambda0[i]["outcome"]["Y"] + protocol["numeric"]["dose_absolute_tolerance"] for i in range(head_count)]
        image_rows.append({"task_id": task_id, "model_id": prediction["model_id"], "sample_id": prediction["sample_id"],
            "split_order": prediction["split_order"], "head_count": head_count, "metrics": metrics,
            "outcomes_lambda0": outcomes, "dose_order_rate": math.fsum(float(x) for x in dose_flags) / head_count,
            "mean_direction_cosine": math.fsum(cosines) / head_count, "mean_sign_agreement": math.fsum(signs) / head_count})
    summaries = {}; prediction_pass = {}; increment_pass = {}; dose_pass = {}; direction_pass = {}
    resamples = protocol["p1"]["bootstrap_resamples"]; confidence = protocol["p1"]["bootstrap_confidence"]
    for model in protocol["models"]:
        selected = [row for row in image_rows if row["model_id"] == model]
        if len(selected) != protocol["p1"]["image_count"]: raise RuntimeError("MODEL_IMAGE_COVERAGE")
        method_intervals = {}
        for method in methods:
            values = [row["metrics"][method]["spearman_for_gate"] for row in selected]
            method_intervals[method] = bootstrap_mean_interval(values, resamples=resamples, confidence=confidence,
                seed=seed(f"{protocol['p1']['bootstrap_seed_namespace']}|{model}|{method}"))
        baseline_order = protocol["p1"]["simple_baseline_tie_order"]
        strongest = max(baseline_order, key=lambda method: (method_intervals[method]["mean"], -baseline_order.index(method)))
        paired = [row["metrics"]["pefa_R"]["spearman_for_gate"] - row["metrics"][strongest]["spearman_for_gate"] for row in selected]
        paired_interval = bootstrap_mean_interval(paired, resamples=resamples, confidence=confidence,
            seed=seed(f"{protocol['p1']['bootstrap_seed_namespace']}|{model}|pefa_minus_{strongest}"))
        dose_interval = bootstrap_mean_interval([row["dose_order_rate"] for row in selected], resamples=resamples, confidence=confidence,
            seed=seed(f"{protocol['p1']['bootstrap_seed_namespace']}|{model}|dose"))
        cosine_interval = bootstrap_mean_interval([row["mean_direction_cosine"] for row in selected], resamples=resamples, confidence=confidence,
            seed=seed(f"{protocol['p1']['bootstrap_seed_namespace']}|{model}|cosine"))
        sign_interval = bootstrap_mean_interval([row["mean_sign_agreement"] for row in selected], resamples=resamples, confidence=confidence,
            seed=seed(f"{protocol['p1']['bootstrap_seed_namespace']}|{model}|sign"))
        top1 = {method: math.fsum(float(row["metrics"][method]["top1_hit"]) for row in selected) / len(selected) for method in methods}
        capture = {method: math.fsum(row["metrics"][method]["top_k_effect_capture"] for row in selected) / len(selected) for method in methods}
        prediction_gate = protocol["p1"]["positive_prediction"]; increment_gate = protocol["p1"]["increment_over_strongest_baseline"]
        prediction_pass[model] = method_intervals["pefa_R"]["mean"] >= prediction_gate["mean_spearman_min"] and method_intervals["pefa_R"]["lower"] > prediction_gate["lower_ci_strictly_above"]
        increment_pass[model] = paired_interval["mean"] >= increment_gate["paired_mean_spearman_min"] and paired_interval["lower"] > increment_gate["lower_ci_strictly_above"]
        dose_pass[model] = dose_interval["mean"] >= protocol["p1"]["dose_order_rate_min"]
        direction_pass[model] = cosine_interval["lower"] > 0.0 and sign_interval["lower"] > 0.5
        summaries[model] = {"method_spearman": method_intervals, "strongest_simple_baseline": strongest,
            "paired_pefa_minus_strongest": paired_interval, "top1_hit_rate": top1, "mean_top_k_effect_capture": capture,
            "dose_order": dose_interval, "direction_cosine": cosine_interval, "direction_sign_agreement": sign_interval,
            "gates": {"natural_gap_prediction": prediction_pass[model], "predictive_increment": increment_pass[model],
                      "dose_order": dose_pass[model], "directional_alignment": direction_pass[model]}}
    all_rows = list(predictions.values()) + list(interventions.values())
    p0_path = run / "p0_decision.json"
    p0 = verify(p0_path if p0_path.exists() else run / "p0_rebind_receipt.json")
    worker_seconds = math.fsum(float(row["resources"]["wall_seconds"]) for row in all_rows)
    r2_hours = float(p0["p0_h20_hours"]) + worker_seconds / 3600.0
    prior_step5_2 = float(protocol["resources"].get("prior_step5_2_h20_hours", 0.0))
    step_hours = prior_step5_2 + r2_hours
    cumulative = protocol["resources"]["predecessor_h20_hours"] + step_hours
    retained = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    resources_valid = step_hours <= protocol["resources"]["step5_2_h20_hours_cap"] and cumulative <= protocol["resources"]["cumulative_step5_h20_hours_cap"]
    resources_valid = resources_valid and retained <= protocol["resources"]["retained_bytes_cap"]
    resources_valid = resources_valid and max(row["resources"]["peak_reserved_gpu_bytes"] for row in all_rows) <= protocol["resources"]["gpu_memory_cap_bytes"]
    resources_valid = resources_valid and max(row["resources"]["peak_cpu_rss_bytes"] for row in all_rows) <= protocol["resources"]["cpu_rss_cap_bytes"]
    numerical_valid = all(row["clean_closure"]["pass"] for row in all_rows) and all(row["quadrature_convergence"]["pass"] for row in predictions.values())
    numerical_valid = numerical_valid and all(item["clean_parity"]["pass"] and item["closure"]["pass"] for row in interventions.values() for item in row["rows"])
    natural = numerical_valid and all(prediction_pass.values()); incremental = natural and all(increment_pass.values())
    analysis = write_new(run / "analysis.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "VALID" if numerical_valid else "INVALID", "image_rows": image_rows, "model_summaries": summaries,
        "prediction_coverage": pred_coverage, "intervention_coverage": int_coverage})
    resources = write_new(run / "resource_ledger.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "SUPPORTED" if resources_valid else "INVALID", "p0_h20_hours": p0["p0_h20_hours"],
        "p1_worker_seconds": worker_seconds, "p1_h20_hours": worker_seconds / 3600.0,
        "r2_h20_hours": r2_hours, "prior_step5_2_h20_hours": prior_step5_2,
        "step5_2_h20_hours": step_hours,
        "predecessor_h20_hours": protocol["resources"]["predecessor_h20_hours"],
        "cumulative_step5_h20_hours": cumulative, "retained_bytes_before_final_artifacts": retained, "caps": protocol["resources"]})
    decisions = [
        {"claim_id": "NATURAL_GAP_PREDICTION", "decision": "SUPPORTED" if natural else ("NOT_SUPPORTED" if numerical_valid else "INVALID")},
        {"claim_id": "PREDICTIVE_INCREMENT_OVER_SIMPLE_BASELINES", "decision": "SUPPORTED" if incremental else ("NOT_SUPPORTED" if numerical_valid else "INVALID")},
        {"claim_id": "DOSE_ORDER", "decision": "SUPPORTED" if numerical_valid and all(dose_pass.values()) else ("NOT_SUPPORTED" if numerical_valid else "INVALID")},
        {"claim_id": "DIRECTIONAL_ALIGNMENT", "decision": "SUPPORTED" if numerical_valid and all(direction_pass.values()) else ("NOT_SUPPORTED" if numerical_valid else "INVALID")},
        {"claim_id": "NUMERICAL_VALIDITY_AND_COST", "decision": "SUPPORTED" if numerical_valid and resources_valid else "INVALID"}]
    terminal_status = ("INVALID_NUMERICAL_OR_INTERVENTION" if not numerical_valid or not resources_valid else
        "NOT_SUPPORTED_NATURAL_GAP_PREDICTION" if not natural else
        "NOT_SUPPORTED_PREDICTIVE_INCREMENT" if not incremental else "SUPPORTED_PREDICTIVE_INCREMENT")
    claims = write_new(run / "claim_decisions.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": terminal_status, "decisions": decisions, "holdout_authorized": False,
        "model_scope": protocol.get("model_scope", "MULTI_MODEL"),
        "cross_architecture_replication_claim": protocol.get("cross_architecture_replication_claim", True),
        "mainline_position": "ENDPOINT_EXTENSION_ONLY", "prior_step5_decision_unchanged": "NOT_SUPPORTED_EMPIRICAL_VALUE",
        "prior_step5_1_decision_unchanged": "NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES", "step6_authorized": False})
    reproduction = write_new(run / "reproduction_commands.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "note": "Historical mutation commands must target a fresh root; validation commands are safe for the immutable completed root.",
        "historical_execution": [f"STEP5_2_RUN_ID={protocol['run_id']} .conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/freeze_step5_2.py",
            f".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/launch_step5_2.py prediction vit_pcia_pefa/{protocol['run_id']}",
            f".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/freeze_step5_2_predictions.py vit_pcia_pefa/{protocol['run_id']}",
            f".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/launch_step5_2.py intervention vit_pcia_pefa/{protocol['run_id']}",
            f".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/finalize_step5_2.py vit_pcia_pefa/{protocol['run_id']}"],
        "current_validation": [f".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/validate_step5_2.py vit_pcia_pefa/{protocol['run_id']}",
            f".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/audit_step5_2_completion.py vit_pcia_pefa/{protocol['run_id']} --validate-only"]})
    p0_evidence_name = "p0_decision.json" if (run / "p0_decision.json").exists() else "p0_rebind_receipt.json"
    evidence_paths = [run / name for name in ("protocol.json", "task_manifest.json", "source_binding.json", "freeze_receipt.json",
        p0_evidence_name, "prediction_startup_snapshot.json", "prediction_launch_terminal.json", "prediction_freeze_receipt.json",
        "intervention_startup_snapshot.json", "intervention_launch_terminal.json", "analysis.json", "resource_ledger.json",
        "claim_decisions.json", "reproduction_commands.json")]
    evidence = write_new(run / "evidence_index.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "files": [{"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in evidence_paths],
        "prediction_result_count": len(predictions), "intervention_result_count": len(interventions),
        "prediction_result_digests": [doc["canonical_digest"] for _, doc in load_results(run, "prediction")],
        "intervention_result_digests": [doc["canonical_digest"] for _, doc in load_results(run, "intervention")]})
    terminal = write_new(run / "terminal_receipt.json", {"schema_version": 1, "run_id": protocol["run_id"], "status": terminal_status,
        "claim_decisions_digest": claims["canonical_digest"], "analysis_digest": analysis["canonical_digest"],
        "resource_ledger_digest": resources["canonical_digest"], "evidence_index_digest": evidence["canonical_digest"],
        "prediction_freeze_digest": prediction_freeze["canonical_digest"], "holdout_started": False,
        "evaluation_access": False, "step6_started": False, "completed_unix_ns": time.time_ns()})
    print(json.dumps({"status": terminal["status"], "natural": natural, "incremental": incremental,
        "step5_2_h20_hours": step_hours, "strongest_baselines": {m: summaries[m]["strongest_simple_baseline"] for m in summaries}}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
