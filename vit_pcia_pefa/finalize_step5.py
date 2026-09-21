#!/usr/bin/env python3
"""Outcome-locked aggregation and claim decisions for the frozen pilot."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_queue import audit_coverage  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value); value["canonical_digest"] = digest(value); return value


def write_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    claimed = value.pop("canonical_digest")
    if digest(value) != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed
    return value


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summary(values: Iterable[float | None], namespace: str) -> dict[str, Any]:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not kept:
        return {"n": 0, "mean": None, "median": None, "bootstrap_95": None}
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")
    generator = random.Random(seed)
    boot = [math.fsum(generator.choice(kept) for _ in kept) / len(kept) for _ in range(10000)]
    return {
        "n": len(kept), "mean": math.fsum(kept) / len(kept),
        "median": statistics.median(kept),
        "bootstrap_95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "bootstrap_replicates": 10000, "bootstrap_seed": seed,
    }


def all_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def category_contains(category: str, name: str) -> bool:
    return {
        "residual": name in {"identity", "carry"},
        "attention_branch": name == "attention",
        "query": name.startswith("query_head_"),
        "key": name.startswith("key_head_"),
        "value": name.startswith("value_head_"),
        "head": name.startswith("head_"),
        "projection": name == "projection",
        "mlp": name in {"mlp", "mlp_fc1", "mlp_gelu", "mlp_fc2"},
    }[category]


def representation_report(documents: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    """Aggregate every preregistered D/Q/R/Omega statistic by model and depth."""
    quantities = {"D": "D", "Q": "T", "R": "R", "Omega": "Omega"}
    categories = tuple(protocol["representation"]["families"])

    def view(row: dict[str, Any], block: str, source: str) -> dict[str, Any]:
        current = row["block_results"][block]
        return (current["omega_directional"]["representation"] if source == "Omega"
                else current["representation"][source])

    def aggregate_views(rows: list[dict[str, Any]], block: str, source: str, namespace: str) -> dict[str, Any]:
        views = [view(row, block, source) for row in rows]
        overall = {
            statistic: summary(
                [item["overall"][statistic] for item in views], f"{namespace}|overall|{statistic}",
            ) for statistic in ("signed_mass", "absolute_mass", "cancellation_rate")
        }
        overall["routing_concentration_hhi"] = summary(
            [item["routing_concentration_hhi"] for item in views], f"{namespace}|overall|hhi",
        )
        category_rows = {}
        category_key = "categories" if source == "Omega" else "registered_categories"
        for category in categories:
            table = {
                statistic: summary(
                    [item[category_key][category][statistic] for item in views],
                    f"{namespace}|{category}|{statistic}",
                ) for statistic in ("signed_mass", "absolute_mass", "cancellation_rate")
            }
            hhi_values = []
            for item in views:
                per_port = item["per_port"] if source == "Omega" else item["registered_per_port"]
                masses = [port["absolute_mass"] for name, port in per_port.items()
                          if category_contains(category, name)]
                if masses:
                    total = math.fsum(masses)
                    hhi_values.append(0.0 if total == 0.0 else math.fsum((mass / total) ** 2 for mass in masses))
            table["routing_concentration_hhi"] = summary(hhi_values, f"{namespace}|{category}|hhi")
            table["routing_concentration_available"] = len(hhi_values) == len(views)
            category_rows[category] = table
        return {"overall": overall, "registered_categories": category_rows}

    return {
        "status": "DESCRIPTIVE",
        "quantity_semantics": {
            "D": "terminal local port field",
            "Q": "terminal port Endpoint contribution T=(x-b)*D",
            "R": "formation residual Q-A",
            "Omega": "directional port-routing curl (J_e^T-J_e)v",
        },
        "statistics": ["signed_mass", "absolute_mass", "cancellation_rate",
                       "routing_concentration_hhi"],
        "models": {
            model: {
                block: {
                    output_name: aggregate_views(
                        [row for row in documents if row["model_id"] == model],
                        block, source, f"representation|{model}|{block}|{output_name}",
                    ) for output_name, source in quantities.items()
                } for block in ("0", "5", "11")
            } for model in protocol["models"]
        },
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: finalize_step5.py PILOT_ROOT")
    pilot = Path(sys.argv[1]).resolve()
    protocol = verify_signed(pilot / "protocol.json")
    manifest = verify_signed(pilot / "task_manifest.json")
    binding = verify_signed(pilot / "source_binding.json")
    freeze = verify_signed(pilot / "freeze_receipt.json")
    if any((pilot / name).exists() for name in ("terminal_receipt.json", "claim_decisions.json")):
        raise RuntimeError("TERMINAL_ALREADY_EXISTS")
    if freeze["protocol_digest"] != protocol["canonical_digest"] or manifest["task_count"] != 16:
        raise RuntimeError("FROZEN_IDENTITY_DRIFT")
    for row in (*binding["files"], *binding["checkpoints"], *binding["pilot_images"]):
        path = REPO / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError("FAIL_SOURCE_BINDING:" + row["path"])
    coverage = audit_coverage(pilot / "queue.sqlite3", pilot / "results")
    if coverage["status"] != "PASS":
        raise RuntimeError("TASK_COVERAGE_FAIL:" + json.dumps(coverage, sort_keys=True))
    documents = []
    for task_id, attempt in coverage["accepted_attempts"]:
        path = pilot / "results/attempts" / task_id / f"attempt-{attempt:02d}.json"
        documents.append(verify_signed(path)["result"])
    documents.sort(key=lambda row: (row["model_id"], row["split_order"]))
    if any(row["status"] != "PASS_TASK" or row["split"] != "calibration" for row in documents):
        raise RuntimeError("INVALID_TASK_RESULT")
    omega_coverage = audit_coverage(pilot / "omega_audit_queue.sqlite3", pilot / "omega_audit_results")
    if omega_coverage["status"] != "PASS":
        raise RuntimeError("OMEGA_AUDIT_COVERAGE_FAIL:" + json.dumps(omega_coverage, sort_keys=True))
    omega_audits = {}
    for task_id, attempt in omega_coverage["accepted_attempts"]:
        path = pilot / "omega_audit_results/attempts" / task_id / f"attempt-{attempt:02d}.json"
        audit = verify_signed(path)["result"]
        if audit["status"] != "PASS_OMEGA_FLOAT64_AUDIT":
            raise RuntimeError("INVALID_OMEGA_AUDIT:" + task_id)
        omega_audits[task_id] = audit

    methods = protocol["contribution"]["methods"]
    input_effects = [name for name in protocol["interventions"]["effects"]
                     if name in documents[0]["input_intervention"]["effects"]]
    head_effects = list(documents[0]["head_ablation"]["effects"])
    aggregate: dict[str, Any] = {"input": {}, "head": {}, "stability": {}}
    for model_id in protocol["models"]:
        rows = [row for row in documents if row["model_id"] == model_id]
        aggregate["input"][model_id] = {}
        for effect in input_effects:
            aggregate["input"][model_id][effect] = {
                method: summary(
                    [row["input_intervention"]["correlations"][method][effect] for row in rows],
                    f"{protocol['run_id']}|input|{model_id}|{effect}|{method}",
                ) for method in methods
            }
        aggregate["head"][model_id] = {}
        for effect in head_effects:
            aggregate["head"][model_id][effect] = {
                method: summary(
                    [row["head_ablation"]["correlations"][method][effect] for row in rows],
                    f"{protocol['run_id']}|head|{model_id}|{effect}|{method}",
                ) for method in rows[0]["head_ablation"]["methods"]
            }
        aggregate["stability"][model_id] = {
            check: {
                "R_rank_spearman": summary(
                    [row["stability_checks"][check]["spearman_vs_primary"] for row in rows],
                    f"{protocol['run_id']}|stability|R|{model_id}|{check}",
                ),
                "endpoint_rank_spearman": summary(
                    [row["stability_checks"][check]["endpoint_map_spearman_vs_primary"] for row in rows],
                    f"{protocol['run_id']}|stability|E|{model_id}|{check}",
                ),
                "reference_condition": rows[0]["stability_checks"][check]["reference_condition"],
            } for check in rows[0]["stability_checks"]
        }

    incremental_effects = []
    contribution_metric_coverage = True
    for effect in input_effects:
        per_model = {}
        for model_id in protocol["models"]:
            table = aggregate["input"][model_id][effect]
            pefa = table["pefa_R"]["mean"]
            comparators = ["endpoint", "ig", "conductance_pcia", "integrated_hessians",
                           "random", "input_norm"]
            complete = pefa is not None and all(table[name]["mean"] is not None for name in comparators)
            contribution_metric_coverage = contribution_metric_coverage and complete
            baseline = max(table[name]["mean"] for name in comparators) if complete else None
            margin = pefa - baseline if complete else None
            per_model[model_id] = {"pefa_R_mean": pefa, "comparator_methods": comparators,
                                   "best_simple_mean": baseline, "margin": margin,
                                   "pass": complete and margin >= 0.05}
        incremental_effects.append({"effect": effect, "models": per_model,
                                    "pass_both_models": all(row["pass"] for row in per_model.values())})
    contribution_supported = any(row["pass_both_models"] for row in incremental_effects)

    predictive_candidates = []
    prediction_metric_coverage = True
    prediction_comparators = {
        "input": ["endpoint", "ig", "conductance_pcia", "integrated_hessians",
                  "random", "input_norm"],
        "head": ["endpoint_port_Q", "conductance_pcia", "pefa_T", "pefa_R",
                 "activation_norm", "random"],
    }
    for family, method, effects in (("input", "pefa_R", input_effects), ("head", "omega", head_effects)):
        for effect in effects:
            values = {model: aggregate[family][model][effect][method]["mean"] for model in protocol["models"]}
            per_model = {}
            for model in protocol["models"]:
                table = aggregate[family][model][effect]
                comparators = prediction_comparators[family]
                complete = values[model] is not None and all(table[name]["mean"] is not None for name in comparators)
                baseline = max(table[name]["mean"] for name in comparators) if complete else None
                margin = values[model] - baseline if complete else None
                per_model[model] = {
                    "candidate_mean": values[model], "comparator_methods": comparators,
                    "best_comparator_mean": baseline, "margin": margin,
                    "pass": complete and margin >= 0.05,
                }
                prediction_metric_coverage = prediction_metric_coverage and complete
            predictive_candidates.append({
                "family": family, "method": method, "effect": effect,
                "models": per_model,
                "pass_both_models": all(row["pass"] for row in per_model.values()),
            })
    intervention_supported = any(row["pass_both_models"] for row in predictive_candidates)

    scientific_closure_pass = all(
        all(block["closure"]["pass"] for block in row["block_results"].values())
        and all(check["closure"]["pass"] for check in row["stability_checks"].values())
        for row in documents
    )
    omega_audit_pass = len(omega_audits) == 16 and all(
        all(block["pass"] for block in row["blocks"].values()) for row in omega_audits.values()
    )
    closure_pass = scientific_closure_pass and omega_audit_pass
    negative_pass = all(all(control["pass"] for control in row["negative_controls"]) for row in documents)
    inheritance = verify_signed(pilot / "inheritance_receipt.json")
    inherited_task_ids = set(inheritance["inherited_task_ids"])
    all_result_worker_seconds = math.fsum(row["resources"]["wall_seconds"] for row in documents)
    fresh_worker_seconds = math.fsum(row["resources"]["wall_seconds"] for row in documents
                                     if row["task_id"] not in inherited_task_ids)
    predecessor_failure = json.loads((ROOT / "step5_pilot_r7_20260921/failure_receipt.json").read_text())
    diagnosis = json.loads((ROOT / "step5_r7_omega_diagnosis_receipt.json").read_text())
    if (predecessor_failure["status"] != "TERMINAL_MATHEMATICAL_CLOSURE_FAIL"
            or diagnosis["status"] != "PASS_ROOT_CAUSE_FLOAT32_ANTISYMMETRIC_CANCELLATION"):
        raise RuntimeError("PREDECESSOR_STATUS_DRIFT")
    predecessor_resource = {
        "through_run_id": "step5_r7_omega_diagnosis_20260921",
        "r1_through_r7_h20_hours_upper_bound": predecessor_failure["cumulative_h20_hours_upper_bound"],
        "r7_diagnosis_h20_hours": diagnosis["diagnosis_h20_hours"],
        "h20_hours_upper_bound": diagnosis["cumulative_h20_hours_after_diagnosis"],
        "sources": ["step5_pilot_r7_20260921/failure_receipt.json#cumulative_h20_hours_upper_bound",
                    "step5_r7_omega_diagnosis_receipt.json#diagnosis_h20_hours"],
    }
    launch = verify_signed(pilot / "launch_terminal.json")
    omega_launch = verify_signed(pilot / "omega_audit_terminal.json")
    if launch["status"] != "PASS_TASK_EXECUTION" or omega_launch["status"] != "PASS_OMEGA_AUDIT_EXECUTION":
        raise RuntimeError("LAUNCH_TERMINAL_FAIL")
    current_worker_seconds_upper_bound = launch["elapsed_seconds"] * launch["selected_gpu_count"]
    omega_worker_seconds_upper_bound = omega_launch["elapsed_seconds"] * omega_launch["selected_gpu_count"]
    gpu_hours = ((current_worker_seconds_upper_bound + omega_worker_seconds_upper_bound) / 3600.0
                 + predecessor_resource["h20_hours_upper_bound"])
    current_bytes = sum(path.stat().st_size for path in all_paths(pilot))
    cost_pass = (
        gpu_hours <= protocol["resources"]["h20_hours_cap"]
        and launch["elapsed_seconds"] / 3600 <= protocol["resources"]["wall_hours_cap"]
        and current_bytes <= protocol["resources"]["retained_bytes_cap"]
        and max(max(row["resources"]["peak_reserved_gpu_bytes"] for row in documents),
                max(row["resources"]["peak_reserved_gpu_bytes"] for row in omega_audits.values()))
            <= protocol["resources"]["gpu_memory_cap_bytes"]
        and max(max(row["resources"]["peak_cpu_rss_bytes"] for row in documents),
                max(row["resources"]["peak_cpu_rss_bytes"] for row in omega_audits.values()))
            <= protocol["resources"]["cpu_rss_cap_bytes"]
    )
    decisions = [
        {"claim_id": "MATHEMATICAL_CLOSURE", "decision": "SUPPORTED" if closure_pass else "INVALID",
         "evidence": "all float32 D/PCIA/PEFA/product/R checks and independent float64 complete-cut Omega audits"},
        {"claim_id": "NEGATIVE_CONTROL_DETECTION", "decision": "SUPPORTED" if negative_pass else "INVALID",
         "evidence": "all frozen structural negative controls on every task"},
        {"claim_id": "COST_AFFORDABLE", "decision": "SUPPORTED" if cost_pass else "INVALID",
         "evidence": "resource_ledger.json against frozen caps"},
        {"claim_id": "INCREMENTAL_OVER_SIMPLE_BASELINES",
         "decision": "INVALID" if not contribution_metric_coverage else "SUPPORTED" if contribution_supported else "NOT_SUPPORTED",
         "evidence": "input intervention within-image correlations aggregated by image"},
        {"claim_id": "PREREGISTERED_INTERVENTION_PREDICTION",
         "decision": "INVALID" if not prediction_metric_coverage else "SUPPORTED" if intervention_supported else "NOT_SUPPORTED",
         "evidence": "input perturbation and head ablation causal outcomes"},
    ]
    promoted = all(row["decision"] == "SUPPORTED" for row in decisions)
    valid = (closure_pass and negative_pass and cost_pass and contribution_metric_coverage
             and prediction_metric_coverage)
    terminal_status = (
        "PASS_PEFA_PILOT_PROMOTION_GATE" if promoted else
        "NOT_SUPPORTED_EMPIRICAL_VALUE" if valid else "INVALID_STEP5_PILOT"
    )

    worst = []
    for row in documents:
        for block_id, block in row["block_results"].items():
            for check, values in block["closure"]["checks"].items():
                worst.append({"model_id": row["model_id"], "sample_id": row["sample_id"],
                              "block": int(block_id), "check": check, **values})
            audit = omega_audits[row["task_id"]]["blocks"][block_id]
            worst.append({"model_id": row["model_id"], "sample_id": row["sample_id"],
                          "block": int(block_id), "check": "omega_complete_cut_zero_sum",
                          "audit_dtype": "float64", **audit["complete_cut_zero_sum"]})
    worst.sort(key=lambda row: row["max_abs"], reverse=True)
    representation = representation_report(documents, protocol)
    representation["worst_closure_rows"] = worst[:10]
    representation["worst_closure_by_model_block_check"] = [
        max((row for row in worst if row["model_id"] == model and row["block"] == block
             and row["check"] == check), key=lambda row: row["max_abs"])
        for model in protocol["models"] for block in protocol["blocks"]
        for check in ("port_D_to_full_gradient", "pcia_to_ig", "pefa_to_endpoint", "T_equals_A_plus_R",
                      "endpoint_minus_ig_equals_R", "omega_complete_cut_zero_sum")
    ]
    resource = signed({
        "schema_version": 1, "run_id": protocol["run_id"],
        "status": "SUPPORTED" if cost_pass else "INVALID",
        "task_count": len(documents), "all_result_worker_seconds": all_result_worker_seconds,
        "fresh_worker_wall_seconds_sum": fresh_worker_seconds,
        "current_launch_worker_seconds_upper_bound": current_worker_seconds_upper_bound,
        "omega_audit_launch_worker_seconds_upper_bound": omega_worker_seconds_upper_bound,
        "predecessor_resource": predecessor_resource,
        "aggregate_h20_hours": gpu_hours, "launch_elapsed_seconds": launch["elapsed_seconds"],
        "omega_audit_launch_elapsed_seconds": omega_launch["elapsed_seconds"],
        "selected_gpu_count": launch["selected_gpu_count"],
        "omega_audit_selected_gpu_count": omega_launch["selected_gpu_count"],
        "peak_reserved_gpu_bytes": max(
            max(row["resources"]["peak_reserved_gpu_bytes"] for row in documents),
            max(row["resources"]["peak_reserved_gpu_bytes"] for row in omega_audits.values()),
        ),
        "peak_cpu_rss_bytes": max(
            max(row["resources"]["peak_cpu_rss_bytes"] for row in documents),
            max(row["resources"]["peak_cpu_rss_bytes"] for row in omega_audits.values()),
        ),
        "retained_bytes_before_finalize": current_bytes, "caps": protocol["resources"],
        "gpu_hours_are_predecessor_upper_bound_plus_scientific_and_audit_launch_wall_times_selected_gpus": True,
        "wall_time_not_divided_by_worker_count": True,
    })
    validation = signed({
        "schema_version": 1, "run_id": protocol["run_id"], "status": terminal_status,
        "coverage": coverage, "omega_audit_coverage": omega_coverage, "task_count": len(documents),
        "models": {model: sum(row["model_id"] == model for row in documents) for model in protocol["models"]},
        "closure_pass": closure_pass, "scientific_closure_pass": scientific_closure_pass,
        "omega_float64_audit_pass": omega_audit_pass, "negative_controls_pass": negative_pass,
        "contribution_metric_coverage": contribution_metric_coverage,
        "prediction_metric_coverage": prediction_metric_coverage,
        "cost_pass": cost_pass, "incremental_effects": incremental_effects,
        "predictive_candidates": predictive_candidates, "aggregates": aggregate,
        "representation": representation, "raw_hessian_materialized": False,
        "full_port_input_input_tensor_materialized": False,
        "reference_conditions_reported_separately": True, "statistics_unit": "image",
        "correlation_and_causal_intervention_separated": True, "evaluation_access": False,
    })
    claim_table = signed({
        "schema_version": 1, "run_id": protocol["run_id"], "status": terminal_status,
        "decisions": decisions, "promotion_gate_pass": promoted,
        "mainline_position": "COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE" if promoted else "ENDPOINT_EXTENSION_ONLY",
        "step6_production_authorized": False,
    })
    write_new(pilot / "resource_ledger.json", resource)
    write_new(pilot / "validation_report.json", validation)
    write_new(pilot / "claim_decisions.json", claim_table)
    evidence_index = signed({
        "schema_version": 1, "run_id": protocol["run_id"],
        "claims": [
            {"claim_id": "MATHEMATICAL_CLOSURE", "artifacts": ["validation_report.json#/closure_pass", "validation_report.json#/representation/worst_closure_rows", "results/attempts/", "omega_audit_results/attempts/"]},
            {"claim_id": "NEGATIVE_CONTROL_DETECTION", "artifacts": ["validation_report.json#/negative_controls_pass", "results/attempts/"]},
            {"claim_id": "COST_AFFORDABLE", "artifacts": ["resource_ledger.json", "startup_snapshot.json", "launch_terminal.json"]},
            {"claim_id": "INCREMENTAL_OVER_SIMPLE_BASELINES", "artifacts": ["validation_report.json#/incremental_effects", "validation_report.json#/aggregates/input"]},
            {"claim_id": "PREREGISTERED_INTERVENTION_PREDICTION", "artifacts": ["validation_report.json#/predictive_candidates", "validation_report.json#/aggregates/head"]},
        ],
        "recovery_evidence": [
            "../step5_pilot_r1_20260921/failure_receipt.json",
            "../step5_pilot_r2_20260921/failure_receipt.json",
            "../step5_pilot_r3_20260921/failure_receipt.json",
            "../step5_pilot_r4_20260921/failure_receipt.json",
            "../step5_pilot_r5_20260921/failure_receipt.json",
            "../step5_r6_contract_audit.md",
            "../step5_r7_discovery_probe_receipt.json",
            "../step5_pilot_r7_20260921/failure_receipt.json",
            "../step5_r7_omega_diagnosis_receipt.json",
            "inheritance_receipt.json", "omega_audit_results/attempts/",
        ],
        "representation_evidence": "validation_report.json#/representation",
        "scientific_raw_results": "results/attempts/",
        "step6_started": False,
    })
    write_new(pilot / "claim_evidence_index.json", evidence_index)
    reproduction = signed({
        "schema_version": 1, "run_id": protocol["run_id"],
        "environment": ".conda/envs/vitnns/bin/python with PYTHONPATH=.:vitnns_rebuild/src",
        "commands": [
            ".conda/envs/vitnns/bin/python vit_pcia_pefa/freeze_step5.py",
            ".conda/envs/vitnns/bin/python vit_pcia_pefa/inherit_step5_r8.py",
            ".conda/envs/vitnns/bin/python vit_pcia_pefa/launch_step5.py vit_pcia_pefa/step5_pilot_r8_20260921",
            ".conda/envs/vitnns/bin/python vit_pcia_pefa/launch_step5_omega_audit.py vit_pcia_pefa/step5_pilot_r8_20260921",
            ".conda/envs/vitnns/bin/python vit_pcia_pefa/finalize_step5.py vit_pcia_pefa/step5_pilot_r8_20260921",
            ".conda/envs/vitnns/bin/python vit_pcia_pefa/validate_step5.py",
        ],
        "note": "freeze and immutable-write commands require a fresh absent run root",
    })
    write_new(pilot / "reproduction_commands.json", reproduction)
    terminal = signed({
        "schema_version": 1, "run_id": protocol["run_id"], "status": terminal_status,
        "task_count": len(documents), "claim_decisions_digest": claim_table["canonical_digest"],
        "validation_report_digest": validation["canonical_digest"],
        "resource_ledger_digest": resource["canonical_digest"],
        "promotion_gate_pass": promoted, "step6_started": False,
        "next_authorization": "STEP6_REQUIRES_EXPLICIT_USER_REQUEST" if promoted else "NONE",
        "completed_unix_ns": time.time_ns(),
    })
    write_new(pilot / "terminal_receipt.json", terminal)
    artifact_rows = [{"path": path.relative_to(pilot).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256(path)} for path in all_paths(pilot)
                     if path.name != "artifact_manifest.json"]
    write_new(pilot / "artifact_manifest.json", signed({
        "schema_version": 1, "run_id": protocol["run_id"],
        "file_count": len(artifact_rows), "files": artifact_rows,
    }))
    print(json.dumps({"status": terminal_status, "tasks": len(documents),
                      "promotion_gate_pass": promoted, "gpu_hours": gpu_hours}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
