#!/usr/bin/env python3
"""Finalize A1, apply the frozen gate, and seal Step-5.1 if B1 is closed."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_1_queue import audit_coverage  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed: raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def signed_write(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value); body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest()
    payload = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return body


def aggregate(rows: list[dict[str, Any]], model: str, role: str, strength: float, method: str) -> dict[str, float]:
    selected = [condition["rank_metrics"][method] for row in rows if row["model_id"] == model
                for condition in row["conditions"] if condition["role"] == role and condition["strength"] == strength]
    if len(selected) != 8: raise RuntimeError("AGGREGATE_COVERAGE")
    return {"n": len(selected), "top1_rate": math.fsum(float(row["top1"]) for row in selected) / len(selected),
            "top_k_rate": math.fsum(float(row["top_k"]) for row in selected) / len(selected),
            "mean_rank": math.fsum(float(row["rank"]) for row in selected) / len(selected),
            "mean_reciprocal_rank": math.fsum(row["reciprocal_rank"] for row in selected) / len(selected),
            "mean_percentile_rank": math.fsum(row["percentile_rank"] for row in selected) / len(selected)}


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: finalize_step5_1.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json")
    launch = verify(run / "a1_launch_terminal.json"); c1 = verify(run / "c1_result.json")
    coverage = audit_coverage(run / "a1_queue.sqlite3", run / "a1_results")
    if launch["status"] != "PASS_A1_TASK_EXECUTION" or coverage["status"] != "PASS":
        raise RuntimeError("A1_COVERAGE")
    documents = []
    for path in sorted((run / "a1_results/attempts").glob("*/attempt-*.json")):
        document = verify(path)
        if document["result"]["status"] != "PASS_A1_TASK": raise RuntimeError("A1_TASK_STATUS")
        documents.append(document)
    rows = [document["result"] for document in documents]
    methods = protocol["a1"]["methods"]; summaries = {}
    controlled_checks, incremental_checks = [], []
    controlled_gate = protocol["a1"]["controlled_gate_each_model_each_strength"]
    incremental_gate = protocol["a1"]["incremental_gate_each_model_each_strength"]
    for model in protocol["models"]:
        for role in ("primary", "decoy"):
            for strength in protocol["wrapper"]["a1_strengths"]:
                key = f"{model}|{role}|{strength}"; summaries[key] = {}
                for method in methods: summaries[key][method] = aggregate(rows, model, role, strength, method)
                pefa = summaries[key]["pefa_R_delta"]
                controlled_checks.append({"model_id": model, "role": role, "strength": strength,
                    "pass": pefa["top1_rate"] >= controlled_gate["pefa_top1_min"]
                            and pefa["top_k_rate"] >= controlled_gate["pefa_top_k_min"]
                            and pefa["mean_percentile_rank"] >= controlled_gate["pefa_mean_percentile_min"],
                    "observed": pefa})
                advantages = {method: {"mean_percentile": pefa["mean_percentile_rank"] - summaries[key][method]["mean_percentile_rank"],
                                       "top1_rate": pefa["top1_rate"] - summaries[key][method]["top1_rate"]}
                              for method in methods if method != "pefa_R_delta"}
                incremental_checks.append({"model_id": model, "role": role, "strength": strength,
                    "pass": all(value["mean_percentile"] >= incremental_gate["mean_percentile_advantage_over_every_baseline"]
                                    and value["top1_rate"] >= incremental_gate["top1_rate_advantage_over_every_baseline"]
                                    for value in advantages.values()), "advantages": advantages})
    controls_valid = all(control["maximum_change_score"] <= protocol["numeric"]["real"]["atol"]
                         and control["clean_parity"]["pass"] and control["closure"]["pass"]
                         for row in rows for control in row["controls"])
    numerical_valid = controls_valid and all(row["clean_closure"]["pass"] for row in rows)
    a1_worker_seconds = math.fsum(row["resources"]["wall_seconds"] for row in rows)
    prior_step5 = verify(ROOT / "step5_completion_20260921/resource_ledger.json")
    prior_step5_upper_bound = float(prior_step5["cumulative_step5_h20_hours_upper_bound"])
    pre_a1_h20_hours = math.fsum((prior_step5_upper_bound, 0.003181305090905258,
                                  0.09303328139810926, 0.25000801771111114))
    if not math.isclose(pre_a1_h20_hours, protocol["resources"]["predecessor_h20_hours"],
                        rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("PRE_A1_RESOURCE_LINEAGE")
    step5_1_h20_hours = (a1_worker_seconds / 3600.0 + 0.09303328139810926
                         + 0.003181305090905258 + 0.25000801771111114)
    cumulative_step5_h20_hours = math.fsum((prior_step5_upper_bound, step5_1_h20_hours))
    resources_valid = (step5_1_h20_hours <= protocol["resources"]["step5_1_h20_hours_cap"]
        and cumulative_step5_h20_hours <= protocol["resources"]["cumulative_step5_h20_hours_cap"]
        and max(row["resources"]["peak_reserved_gpu_bytes"] for row in rows) <= protocol["resources"]["gpu_memory_cap_bytes"]
        and max(row["resources"]["peak_cpu_rss_bytes"] for row in rows) <= protocol["resources"]["cpu_rss_cap_bytes"])
    controlled = numerical_valid and all(row["pass"] for row in controlled_checks)
    incremental = numerical_valid and all(row["pass"] for row in incremental_checks)
    b1_authorized = controlled and incremental and resources_valid
    analysis = signed_write(run / "a1_analysis.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "VALID" if numerical_valid else "INVALID", "summaries": summaries,
        "controlled_checks": controlled_checks, "incremental_checks": incremental_checks,
        "controls_valid": controls_valid, "coverage": coverage})
    resources = signed_write(run / "resource_ledger.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "SUPPORTED" if resources_valid else "INVALID", "r1_failed_launch_h20_hours": 0.003181305090905258,
        "r2_a0_h20_hours": 0.09303328139810926, "r5_a1_worker_seconds": a1_worker_seconds,
        "r4_failed_launch_h20_hours": 0.25000801771111114,
        "r5_a1_h20_hours": a1_worker_seconds / 3600.0, "step5_1_h20_hours": step5_1_h20_hours,
        "prior_step5_h20_hours_upper_bound": prior_step5_upper_bound,
        "pre_a1_h20_hours_upper_bound": pre_a1_h20_hours,
        "cumulative_step5_h20_hours": cumulative_step5_h20_hours,
        "cumulative_step5_h20_hours_upper_bound": cumulative_step5_h20_hours,
        "caps": protocol["resources"]})
    decisions = [
        {"claim_id": "CONTROLLED_CURVATURE_LOCALIZATION", "decision": "SUPPORTED" if controlled else ("NOT_SUPPORTED" if numerical_valid else "INVALID")},
        {"claim_id": "NATURAL_NONLINEARITY_LOCALIZATION", "decision": "NOT_RUN_A1_GATE_CLOSED" if not b1_authorized else "PENDING_B1"},
        {"claim_id": "ATTRIBUTION_ERROR_LOCALIZATION", "decision": c1["status"]},
        {"claim_id": "INCREMENTAL_OVER_SIMPLE_BASELINES", "decision": "SUPPORTED" if incremental else ("NOT_SUPPORTED" if numerical_valid else "INVALID")},
        {"claim_id": "NUMERICAL_VALIDITY_AND_COST", "decision": "SUPPORTED" if numerical_valid and resources_valid else "INVALID"},
    ]
    claim_file = signed_write(run / "claim_decisions.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "A1_GATE_PASS_B1_REQUIRED" if b1_authorized else "NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES",
        "decisions": decisions, "b1_authorized_by_gate": b1_authorized,
        "confirmation_experiment_authorized": False, "mainline_position": "ENDPOINT_EXTENSION_ONLY",
        "prior_step5_decision_unchanged": "NOT_SUPPORTED_EMPIRICAL_VALUE", "step6_authorized": False})
    evidence_paths = [run / name for name in ("protocol.json", "task_manifest.json", "source_binding.json",
        "freeze_receipt.json", "inheritance_receipt.json", "a0_inheritance_receipt.json",
        "a0_decision.json", "a1_startup_snapshot.json",
        "a1_launch_terminal.json", "a1_analysis.json", "c1_result.json", "resource_ledger.json", "claim_decisions.json")]
    evidence_paths.append(ROOT / "step5_completion_20260921/resource_ledger.json")
    evidence = signed_write(run / "evidence_index.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "files": [{"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in evidence_paths],
        "raw_result_count": len(documents), "raw_result_digests": [doc["canonical_digest"] for doc in documents]})
    if not b1_authorized:
        terminal = signed_write(run / "terminal_receipt.json", {"schema_version": 1, "run_id": protocol["run_id"],
            "status": "NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES",
            "reason": "A1_CONTROLLED_LOCALIZATION_SUPPORTED_BUT_PCIA_DELTA_TIED_PEFA_R_DELTA",
            "claim_decisions_digest": claim_file["canonical_digest"], "analysis_digest": analysis["canonical_digest"],
            "resource_ledger_digest": resources["canonical_digest"], "evidence_index_digest": evidence["canonical_digest"],
            "b1_started": False, "evaluation_access": False, "step6_started": False,
            "completed_unix_ns": time.time_ns()})
    print(json.dumps({"status": claim_file["status"], "controlled": controlled, "incremental": incremental,
                      "b1_authorized": b1_authorized, "step5_1_h20_hours": step5_1_h20_hours}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
