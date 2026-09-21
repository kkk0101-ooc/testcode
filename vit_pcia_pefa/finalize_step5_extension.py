#!/usr/bin/env python3
"""Aggregate one frozen Step-5 mechanism-extension phase."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_extension_queue import audit_coverage  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value); value["canonical_digest"] = digest(value); return value


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if digest(value) != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def write_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def result_documents(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "results/attempts").glob("*/attempt-*.json")):
        envelope = verify_signed(path)
        if envelope["result"]["status"] != "PASS_EXTENSION_TASK":
            raise RuntimeError("RESULT_STATUS:" + str(path))
        rows.append(envelope["result"])
    return rows


def median(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not valid else float(statistics.median(valid))


def geometric_ratio(numerator: list[float], denominator: list[float]) -> float | None:
    ratios = [left / right for left, right in zip(numerator, denominator) if left > 0 and right > 0]
    return median(ratios)


def finalize_e1_e2(protocol: dict[str, Any], documents: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary = float(protocol["loop"]["primary_epsilon_l2"]); loop_thresholds = protocol["loop"]["thresholds"]
    e1_models, e2_models = {}, {}
    for model_id in protocol["models"]:
        rows = [row for row in documents if row["model_id"] == model_id]
        clean_primary, boundary_primary = [], []
        valid = True
        for row in rows:
            clean = next(item for item in row["clean_loop"]["scale_rows"] if item["epsilon"] == primary)
            boundary = next(item for item in row["boundary_loop"]["scale_rows"] if item["epsilon"] == primary)
            clean_primary.append((row, clean)); boundary_primary.append((row, boundary))
            for loop_name in ("clean_loop", "boundary_loop"):
                for item in row[loop_name]["scale_rows"]:
                    validity = protocol["loop"]["validity"]
                    valid &= item["identical_endpoint_max_abs"] <= validity["identical_endpoint_max_abs"]
                    valid &= item["complete_cut_vs_full_abs_error"] <= validity["complete_cut_vs_full_abs_error"]
                    valid &= abs(item["full_gradient_circulation"]) <= validity["full_gradient_circulation_abs"]
        e1_summary = {
            "valid": valid,
            "median_absolute_rank_spearman": median([row[1]["absolute_rank_spearman"] for row in clean_primary]),
            "median_signed_cosine": median([row[1]["signed_cosine"] for row in clean_primary]),
            "median_sign_agreement": median([row[1]["sign_agreement"] for row in clean_primary]),
            "median_relative_slope_error": median([row[1]["relative_slope_error"] for row in clean_primary]),
            "median_scaling_exponent": median([row["clean_loop"]["circulation_scaling_exponent"] for row in rows]),
            "median_rank_advantage_over_random": median([row[1]["rank_advantage_over_random"] for row in clean_primary]),
        }
        metrics_present = all(value is not None for key, value in e1_summary.items() if key != "valid")
        e1_summary["supported"] = bool(valid and metrics_present
            and e1_summary["median_absolute_rank_spearman"] >= loop_thresholds["absolute_rank_spearman"]
            and e1_summary["median_signed_cosine"] >= loop_thresholds["signed_cosine"]
            and e1_summary["median_sign_agreement"] >= loop_thresholds["sign_agreement"]
            and e1_summary["median_relative_slope_error"] <= loop_thresholds["relative_slope_error_max"]
            and loop_thresholds["scaling_exponent_min"] <= e1_summary["median_scaling_exponent"] <= loop_thresholds["scaling_exponent_max"]
            and e1_summary["median_rank_advantage_over_random"] >= loop_thresholds["rank_advantage_over_random"])
        e1_models[model_id] = e1_summary

        clean_curl = [math.sqrt(sum(float(value) ** 2 for key, value in row["clean_loop"]["local_curl_kappa"].items() if key.startswith("head_"))) for row in rows]
        boundary_curl = [math.sqrt(sum(float(value) ** 2 for key, value in row["boundary_loop"]["local_curl_kappa"].items() if key.startswith("head_"))) for row in rows]
        clean_density = [item[1]["circulation_density_l2"] for item in clean_primary]
        boundary_density = [item[1]["circulation_density_l2"] for item in boundary_primary]
        ordering = sorted(range(len(rows)), key=lambda index: (abs(rows[index]["boundary_probe"]["clean_dynamic_margin"]), rows[index]["split_order"]))
        low = [clean_density[index] for index in ordering[:4]]; high = [clean_density[index] for index in ordering[4:]]
        e2_summary = {
            "valid": valid,
            "margin_reduced_or_crossed_fraction": sum(bool(row["boundary_probe"]["absolute_margin_reduced_or_crossed"]) for row in rows) / len(rows),
            "median_boundary_to_clean_curl_norm_ratio": geometric_ratio(boundary_curl, clean_curl),
            "median_boundary_to_clean_circulation_density_ratio": geometric_ratio(boundary_density, clean_density),
            "median_boundary_signed_cosine": median([item[1]["signed_cosine"] for item in boundary_primary]),
            "low_margin_median_clean_circulation_density": median(low),
            "high_margin_median_clean_circulation_density": median(high),
        }
        threshold = protocol["nonlinearity"]["thresholds"]
        present = all(value is not None for key, value in e2_summary.items() if key != "valid")
        e2_summary["supported"] = bool(valid and present
            and e2_summary["margin_reduced_or_crossed_fraction"] >= threshold["margin_reduced_or_crossed_fraction"]
            and e2_summary["median_boundary_to_clean_curl_norm_ratio"] >= threshold["boundary_to_clean_curl_norm_ratio"]
            and e2_summary["median_boundary_to_clean_circulation_density_ratio"] >= threshold["boundary_to_clean_circulation_density_ratio"]
            and e2_summary["median_boundary_signed_cosine"] >= threshold["boundary_signed_cosine"]
            and e2_summary["low_margin_median_clean_circulation_density"] > e2_summary["high_margin_median_clean_circulation_density"])
        e2_models[model_id] = e2_summary
    valid = all(row["valid"] for row in (*e1_models.values(), *e2_models.values()))
    claims = [
        {"claim_id": "E1_LOCAL_CURL_PREDICTS_LOOP_CIRCULATION",
         "decision": "INVALID" if not valid else "SUPPORTED" if all(row["supported"] for row in e1_models.values()) else "NOT_SUPPORTED"},
        {"claim_id": "E2_NONLINEAR_CONDITION_AMPLIFIES_VALID_CURVATURE_EFFECT",
         "decision": "INVALID" if not valid else "SUPPORTED" if all(row["supported"] for row in e2_models.values()) else "NOT_SUPPORTED"},
    ]
    return {"E1_by_model": e1_models, "E2_by_model": e2_models}, claims


def finalize_e3a(protocol: dict[str, Any], documents: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary = protocol["source_resolution"]["primary_effect"]
    by_model = {}
    all_valid = True
    for model_id in protocol["models"]:
        rows = [row for row in documents if row["model_id"] == model_id]
        grid_summary = {}
        for grid in protocol["source_resolution"]["grids"]:
            key = str(grid); per_image = []
            for row in rows:
                current = row["source_resolution"][key]
                closure = current["source_sum_closure"]
                valid = max(closure.values()) <= 1e-10
                all_valid &= valid
                correlations = current["correlations"]
                pefa = correlations["pefa_R"][primary]
                baseline_values = [correlations[name][primary]
                                   for name in protocol["source_resolution"]["comparators"]
                                   if correlations[name][primary] is not None]
                baseline = None if not baseline_values else max(baseline_values)
                valid &= pefa is not None and baseline is not None
                per_image.append({"sample_id": row["sample_id"], "valid": valid,
                                  "pefa_R": pefa, "best_baseline": baseline,
                                  "advantage": None if pefa is None or baseline is None else pefa - baseline,
                                  "terminal_T_convergence": current["terminal_T_convergence"]})
            grid_summary[key] = {
                "per_image": per_image, "mean_pefa_R": (None if any(item["pefa_R"] is None for item in per_image) else float(statistics.fmean(item["pefa_R"] for item in per_image))),
                "mean_best_baseline": (None if any(item["best_baseline"] is None for item in per_image) else float(statistics.fmean(item["best_baseline"] for item in per_image))),
                "mean_advantage": (None if any(item["advantage"] is None for item in per_image) else float(statistics.fmean(item["advantage"] for item in per_image))),
            }
        delta = (None if grid_summary["7"]["mean_pefa_R"] is None or grid_summary["4"]["mean_pefa_R"] is None
                 else grid_summary["7"]["mean_pefa_R"] - grid_summary["4"]["mean_pefa_R"])
        every = all(four["pefa_R"] is not None and seven["pefa_R"] is not None
                    and seven["pefa_R"] - four["pefa_R"] >= protocol["source_resolution"]["advance_to_14x14"]["minimum_7x7_over_4x4"]
                    for four, seven in zip(grid_summary["4"]["per_image"], grid_summary["7"]["per_image"]))
        gate = (grid_summary["7"]["mean_advantage"] is not None and delta is not None
                and grid_summary["7"]["mean_advantage"] >= protocol["source_resolution"]["advance_to_14x14"]["minimum_baseline_advantage"]
                and delta >= protocol["source_resolution"]["advance_to_14x14"]["minimum_7x7_over_4x4"] and every)
        by_model[model_id] = {"grids": grid_summary, "mean_7x7_over_4x4": delta,
                              "every_image_improves": every, "advance_gate": gate}
    advance = all_valid and all(row["advance_gate"] for row in by_model.values())
    claim = {"claim_id": "E3A_7X7_PEFA_INCREMENTAL_RESOLUTION_VALUE",
             "decision": "INVALID" if not all_valid else "SUPPORTED" if advance else "NOT_SUPPORTED"}
    return {"E3A_by_model": by_model, "advance_to_14x14": advance,
            "pixel_level_authorized": False}, [claim]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: finalize_step5_extension.py EXTENSION_ROOT")
    root = Path(sys.argv[1]).resolve()
    if any((root / name).exists() for name in ("analysis.json", "claim_decisions.json",
                                                "resource_ledger.json", "terminal_receipt.json")):
        raise RuntimeError("EXTENSION_ALREADY_FINALIZED")
    protocol = verify_signed(root / "protocol.json"); launch = verify_signed(root / "launch_terminal.json")
    coverage = audit_coverage(root / "queue.sqlite3", root / "results")
    if launch["status"] != "PASS_EXTENSION_EXECUTION" or coverage["status"] != "PASS":
        raise RuntimeError("EXTENSION_EXECUTION_INCOMPLETE")
    documents = result_documents(root)
    if len(documents) != protocol["task_count"]:
        raise RuntimeError("EXTENSION_RESULT_COUNT")
    if protocol["phase"] == "E1_E2":
        analysis_body, decisions = finalize_e1_e2(protocol, documents)
    else:
        analysis_body, decisions = finalize_e3a(protocol, documents)
    analysis = signed({"schema_version": 1, "run_id": protocol["run_id"],
                       "phase": protocol["phase"], **analysis_body, "step6_started": False})
    claims = signed({"schema_version": 1, "run_id": protocol["run_id"],
                     "phase": protocol["phase"], "decisions": decisions,
                     "original_r8_decision_unchanged": "NOT_SUPPORTED_EMPIRICAL_VALUE",
                     "confirmatory_claim_allowed": False, "step6_production_authorized": False})
    predecessor_hours = float(protocol["predecessor_h20_hours_including_diagnosis"])
    current_upper = float(launch["elapsed_seconds"]) * int(launch["selected_gpu_count"]) / 3600
    peak_gpu = max(row["resources"]["peak_reserved_gpu_bytes"] for row in documents)
    peak_cpu = max(row["resources"]["peak_cpu_rss_bytes"] for row in documents)
    ledger = signed({
        "schema_version": 1, "run_id": protocol["run_id"], "phase": protocol["phase"],
        "predecessor_run_id": protocol["predecessor_run_id"],
        "predecessor_h20_hours": predecessor_hours,
        "current_launch_h20_hours_upper_bound": current_upper,
        "cumulative_step5_h20_hours": predecessor_hours + current_upper,
        "peak_reserved_gpu_bytes": peak_gpu, "peak_cpu_rss_bytes": peak_cpu,
        "caps": protocol["resources"],
        "status": "SUPPORTED" if (current_upper <= protocol["resources"]["extension_h20_hours_cap"]
            and predecessor_hours + current_upper <= protocol["resources"]["cumulative_step5_h20_hours_cap"]
            and peak_gpu <= protocol["resources"]["gpu_memory_cap_bytes"]
            and peak_cpu <= protocol["resources"]["cpu_rss_cap_bytes"]) else "INVALID",
    })
    decision_values = [row["decision"] for row in decisions]
    terminal_status = "INVALID" if "INVALID" in decision_values or ledger["status"] == "INVALID" else (
        "SUPPORTED" if all(value == "SUPPORTED" for value in decision_values) else "NOT_SUPPORTED")
    terminal = signed({
        "schema_version": 1, "run_id": protocol["run_id"], "phase": protocol["phase"],
        "status": terminal_status, "completed_unix_ns": time.time_ns(),
        "analysis_digest": analysis["canonical_digest"],
        "claim_decisions_digest": claims["canonical_digest"],
        "resource_ledger_digest": ledger["canonical_digest"],
        "original_r8_decision_unchanged": "NOT_SUPPORTED_EMPIRICAL_VALUE",
        "next_authorization": ("FRESH_E3B_14X14_ROOT_ONLY" if analysis.get("advance_to_14x14") else "NONE"),
        "step6_started": False,
    })
    write_new(root / "analysis.json", analysis); write_new(root / "claim_decisions.json", claims)
    write_new(root / "resource_ledger.json", ledger); write_new(root / "terminal_receipt.json", terminal)
    artifacts = []
    for path in sorted(file for file in root.rglob("*") if file.is_file()
                       and file.name != "artifact_manifest.json"):
        relative = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_hash})
    write_new(root / "artifact_manifest.json", signed({
        "schema_version": 1, "run_id": protocol["run_id"],
        "artifact_count": len(artifacts), "artifacts": artifacts,
    }))
    print(json.dumps({"status": terminal_status, "phase": protocol["phase"],
                      "decisions": decisions, "cumulative_h20_hours": ledger["cumulative_step5_h20_hours"],
                      "step6_started": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
