#!/usr/bin/env python3
"""Freeze and execute the revision-v7 CPU-only reanalysis.

The freeze command records the analysis contract and source identities before
any newly stratified result is computed.  The analyze command is deliberately
separate and refuses to run when the frozen script or source bindings drift.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
PREVIOUS_ROOT = REPO / "reports/nns_vit_endpoint_revision_v7_r3_20260923"
ROOT = REPO / "reports/nns_vit_endpoint_revision_v7_r4_20260923"
PUBLICATION = REPO / "papers/nns_vit_endpoint/publication_r2_20260920"
E4_INDEX = REPO / "reports/nns_vit_endpoint_e4_intervention_r3_20260918/combined_result_index.json"
E6_ROOT = REPO / "reports/nns_vit_endpoint_e6_release_r3_20260918"
E5_ROOT = REPO / "reports/nns_vit_revision_v5_r1_20260921/e5"
E5_ANALYSIS = REPO / "reports/nns_vit_revision_v5_r1_20260921/analysis"
M3_ROOT = REPO / "reports/nns_vit_endpoint_m3_r1_20260919"
V6_ROOT = REPO / "reports/nns_vit_endpoint_revision_v6_r1_20260923"
WORKFLOW = REPO / "修改意见/修改意见v7_两轮评审核验与修改补实验复审工作规程.md"
SCRIPT = Path(__file__).resolve()
VALIDATOR = REPO / "scripts/validate_revision_v7_analysis.py"
TEST_FILE = REPO / "tests/test_revision_v7.py"
BLOCKS = (0, 3, 7, 11)
REFERENCES = ("normalized_zero", "blur_rgb")
MODELS = ("deit_s16", "vit_b16")
E5_MODELS = ("deit_s16", "vit_b16", "vit_b16_in1k", "vit_s16_augreg")
EPS = 1e-12
N_BOOTSTRAP = 10_000
STAGE1_STATUS = "REVISION_V7_STAGE1_PASS_AND_REANALYSIS_FROZEN"
STAGE2_STATUS = "REVISION_V7_REANALYSIS_PASS_WITH_ALL_STRATA_REPORTED"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("RANK_INPUT")
    return rankdata(array, method="average").astype(np.float64)


def spearman(left: Sequence[float], right: Sequence[float], minimum: int = 4) -> tuple[float | None, str]:
    x = np.asarray(left, dtype=np.float64); y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        return None, "NA_SHAPE"
    if x.size < minimum:
        return None, "NA_FEWER_THAN_FOUR"
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, "NA_NONFINITE"
    rx = average_ranks(x); ry = average_ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator == 0.0:
        return None, "NA_CONSTANT_VECTOR"
    return float(np.dot(rx, ry) / denominator), ""


def exact_two_sided_sign(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("SIGN_INPUT")
    positive = int(np.sum(array > EPS)); negative = int(np.sum(array < -EPS))
    ties = int(array.size - positive - negative); n = positive + negative
    if n == 0:
        p_value = 1.0
    else:
        tail = math.fsum(math.comb(n, j) for j in range(min(positive, negative) + 1))
        p_value = min(1.0, float(2.0 * tail / (2 ** n)))
    return {"positive": positive, "negative": negative, "ties": ties, "raw_p": p_value}


def holm(rows: list[dict[str, Any]], *, p_key: str = "raw_p", output_key: str = "holm_p") -> None:
    order = sorted(range(len(rows)), key=lambda index: (float(rows[index][p_key]), str(rows[index].get("metric", rows[index].get("claim_id", index)))))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * float(rows[index][p_key])))
        rows[index][output_key] = running


def bootstrap_median(values: Sequence[float], key: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, math.nan
    seed = int.from_bytes(hashlib.sha256(("NNS_VIT_REVISION_V7_R2_20260923|bootstrap|" + key).encode()).digest()[:8], "big")
    generator = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for start in range(0, N_BOOTSTRAP, 250):
        stop = min(N_BOOTSTRAP, start + 250)
        indices = generator.integers(0, array.size, size=(stop - start, array.size), endpoint=False)
        estimates[start:stop] = np.median(array[indices], axis=1)
    return float(np.median(array)), float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))


def residualized_rank_correlation(rows: Sequence[Mapping[str, Any]], controls: Sequence[str]) -> tuple[float | None, int, float | None, str]:
    x = average_ranks([float(row["m"]) for row in rows]); y = average_ranks([float(row["absolute_effect"]) for row in rows])
    columns = [np.ones(len(rows), dtype=np.float64)]
    for control in controls:
        levels = sorted({row[control] for row in rows}, key=str)
        for level in levels[1:]:
            columns.append(np.asarray([float(row[control] == level) for row in rows]))
    design = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(design)); df = len(rows) - rank
    condition = float(np.linalg.cond(design))
    if df < 3 or not np.isfinite(condition) or condition > 1e12:
        return None, df, condition, "NA_ILL_CONDITIONED_DESIGN"
    x_res = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_res = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denominator = float(np.linalg.norm(x_res) * np.linalg.norm(y_res))
    if denominator <= EPS:
        return None, df, condition, "NA_CONSTANT_RESIDUAL"
    return float(np.dot(x_res, y_res) / denominator), df, condition, ""


def tie_count(values: Sequence[float]) -> int:
    array = np.asarray(values)
    return int(array.size - np.unique(array).size)


def source_inventory() -> list[dict[str, Any]]:
    paths = [
        E4_INDEX, E6_ROOT / "per_image_statistics.json", E6_ROOT / "family_decisions.json",
        E6_ROOT / "sign_test_report.json", E6_ROOT / "holm_report.json", E6_ROOT / "bootstrap_report.json",
        E5_ANALYSIS / "e5_image_values.csv", E5_ANALYSIS / "e5_summary.csv",
        M3_ROOT / "task_manifest.json", V6_ROOT / "source_bindings.json",
        PUBLICATION / "evidence/revision_v6/original_inference_recomputed.json",
        PREVIOUS_ROOT / "analysis_run.json", PREVIOUS_ROOT / "validation_report.json", PREVIOUS_ROOT / "terminal_receipt.json",
    ]
    index = json.loads(E4_INDEX.read_text())
    paths.extend(REPO / row["source_result_path"] for row in index["records"])
    for metadata in sorted(E5_ROOT.glob("*.json")):
        paths.extend((metadata, metadata.with_suffix(".npz")))
    manifest = json.loads((M3_ROOT / "task_manifest.json").read_text())
    paths.extend(M3_ROOT / "task_payloads" / row["task_id"] / "metrics.npz" for row in manifest["tasks"])
    paths.extend(sorted((V6_ROOT / "tasks").glob("structure__*.npz")))
    unique = sorted(set(paths))
    if not unique or any(not path.is_file() for path in unique):
        raise ValueError("SOURCE_FILE_MISSING")
    return [file_record(path) for path in unique]


def closed_object(properties: Mapping[str, Any], required: Sequence[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": dict(properties), "required": list(required or properties), "additionalProperties": False}


CSV_HEADERS = {
    "table6_stratified_image_values.csv": ["model", "reference", "image", "metric", "block", "value", "n_components", "residual_df", "condition_number", "na_reason", "analysis_identity"],
    "table6_stratified_summary.csv": ["model", "reference", "metric", "block", "n_total", "n_valid", "n_na", "median", "ci95_low", "ci95_high", "q10", "q25", "q75", "q90", "positive", "negative", "ties", "raw_p", "holm_p", "analysis_identity"],
    "table6_scale_diagnostics.csv": ["model", "reference", "family", "operation", "block", "quantity", "n", "median", "ci95_low", "ci95_high", "q10", "q90"],
    "h4_recomputed_image_values.csv": ["model", "image", "block", "path_map_mass_rho", "activation_gradient_rho", "difference", "h4_image_mean", "valid_heads", "path_ties", "activation_gradient_ties", "effect_ties", "na_reason"],
    "h4_recomputed_summary.csv": ["model", "hypothesis", "n_total", "n_valid", "n_na", "median", "ci95_low", "ci95_high", "positive", "negative", "ties", "raw_p", "holm_p", "decision"],
    "pvalue_audit.csv": ["claim_id", "source_rows", "statistic", "null", "alternative", "unit", "ties", "raw_p", "family", "holm_p", "display", "recalculation"],
    "locality_radius_sensitivity.csv": ["model", "radius", "contrast", "n", "median", "ci95_low", "ci95_high", "q10", "q90", "analysis_identity"],
    "residual_tv_quality_diagnostics.csv": ["model", "residual_type", "mass_quartile", "n", "median_tv", "ci95_low", "ci95_high", "q10", "q90", "analysis_identity"],
}


def analysis_schema() -> dict[str, Any]:
    string = {"type": "string"}; number = {"type": ["number", "null"]}
    json_schemas = {
        "cut_definition_counterexample.json": closed_object({"schema_version": {"type": "integer"}, "status": string, "graph": closed_object({"nodes": {"type": "array", "items": string}, "edges": {"type": "array", "items": string}, "target": string}), "input": {"type": "number"}, "omitted_edge": string, "structural_cut_valid": {"type": "boolean"}, "pointwise_derivative_on_omitted_branch": {"type": "number"}, "pointwise_closure_error_after_omission": {"type": "number"}, "interpretation": string}),
        "stage1_receipt.json": closed_object({"schema_version": {"type": "integer"}, "status": string, "cpu_only": {"type": "boolean"}, "protocol_sha256": string, "source_bindings_sha256": string, "analysis_schema_sha256": string, "counterexample_sha256": string, "script_sha256": string, "validator_sha256": string, "test_sha256": string, "manuscripts": {"type": "array", "items": closed_object({"path": string, "bytes": {"type": "integer"}, "sha256": string})}, "old_families_unchanged": {"type": "boolean"}, "new_results_accessed": {"type": "boolean"}}),
        "validation_report.json": closed_object({"schema_version": {"type": "integer"}, "status": string, "checks": {"type": "array", "items": closed_object({"id": string, "status": string, "detail": string})}, "errors": {"type": "array", "items": string}, "cpu_only": {"type": "boolean"}, "gpu_model_imports": {"type": "integer"}, "old_family_max_abs_error": {"type": "number"}, "h4_max_abs_error": {"type": "number"}, "output_sha256": {"type": "array", "items": closed_object({"path": string, "bytes": {"type": "integer"}, "sha256": string})}}),
        "terminal_receipt.json": closed_object({"schema_version": {"type": "integer"}, "status": string, "cpu_only": {"type": "boolean"}, "gpu_inference_runs": {"type": "integer"}, "table6_images": {"type": "integer"}, "h4_images": {"type": "integer"}, "artifact_manifest_sha256": string, "validation_report_sha256": string, "old_sealed_roots_modified": {"type": "boolean"}, "next": string}),
    }
    return {"schema_version": 1, "csv": {name: {"header": header, "additional_columns": False} for name, header in CSV_HEADERS.items()}, "json": json_schemas, "all_json_objects_closed": True}


def freeze() -> None:
    if ROOT.exists():
        raise ValueError("V7_ROOT_ALREADY_EXISTS")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU_ONLY_BOUNDARY")
    if not VALIDATOR.is_file() or not TEST_FILE.is_file():
        raise ValueError("V7_CODE_INCOMPLETE")
    ROOT.mkdir(parents=True)
    bindings = {"schema_version": 1, "status": "BOUND_BEFORE_STRATIFIED_RESULTS", "files": source_inventory()}
    bindings["files_digest"] = canonical_digest(bindings["files"])
    write_json(ROOT / "source_bindings.json", bindings)
    write_json(ROOT / "analysis_schema.json", analysis_schema())
    counterexample = {
        "schema_version": 1, "status": "PASS_DISTINGUISHES_STRUCTURE_FROM_POINTWISE_DERIVATIVE",
        "graph": {"nodes": ["x", "linear_branch", "square_branch", "sum_target"], "edges": ["x->linear_branch", "x->square_branch", "linear_branch->sum_target", "square_branch->sum_target"], "target": "s(x)=x+x^2"},
        "input": 0.0, "omitted_edge": "square_branch->sum_target", "structural_cut_valid": False,
        "pointwise_derivative_on_omitted_branch": 0.0, "pointwise_closure_error_after_omission": 0.0,
        "interpretation": "The omitted branch exists structurally but has zero derivative at x=0, so numerical closure cannot certify structural completeness.",
    }
    write_json(ROOT / "cut_definition_counterexample.json", counterexample)
    protocol = {
        "schema_version": 1, "namespace": "NNS_VIT_REVISION_V7_R4_20260923",
        "status": "FROZEN_BEFORE_NEW_STRATIFIED_RESULTS", "cpu_only": True, "cuda_visible_devices": "",
        "old_results_read_only": True, "image_decode_allowed": False, "model_load_allowed": False, "gpu_inference_allowed": False,
        "source_bindings_sha256": sha256(ROOT / "source_bindings.json"),
        "analysis_schema_sha256": sha256(ROOT / "analysis_schema.json"),
        "scripts": {"analysis": file_record(SCRIPT), "validator": file_record(VALIDATOR), "tests": file_record(TEST_FILE)},
        "table6": {
            "unit": "image", "models": list(MODELS), "references": list(REFERENCES), "blocks": list(range(12)),
            "effect": "abs(clean_ground_truth_logit_minus_single_component_reference_replacement)", "predictor": "component_map_absolute_mass_m",
            "rank_ties": "average", "minimum_components": 4,
            "historical_families": {"heads": ["head"], "routing_value": ["routing", "value"], "residual_branches": ["attention", "mlp"]},
            "new_strata": ["A_only_across_blocks", "V_only_across_blocks", "attention_branch_only_across_blocks", "mlp_branch_only_across_blocks", "heads_within_block_fisher_mean"],
            "residualization": {"response": "average_rank(abs_effect)", "predictor": "average_rank(m)", "models": ["operation", "block", "operation_and_block"], "fit": "OLS with intercept and reference-coded fixed effects", "diagnostic": "report residual df and design condition number; NA if df<3 or condition>1e12"},
            "head_aggregation": "clip each block rho to [-1+1e-12,1-1e-12], Fisher z, arithmetic mean over 12 blocks, tanh",
            "av_concordance": "mean over 12 blocks of indicator sign((m_A-m_V)*(abs_effect_A-abs_effect_V)) positive; zero products are ties and excluded",
            "scale_diagnostics": "model-reference-operation-block summaries of log1p(m) and log1p(abs_effect)",
            "identity": "post-hoc diagnostic; historical pooled families and original 28-test family remain immutable",
        },
        "h4": {"blocks": list(BLOCKS), "reference": "normalized_zero", "target": "ground_truth_logit", "path_score": "sum_abs(path_local_map)", "activation_gradient_score": "abs(dot(attn.proj_input_head_slice, gradient))", "effect": "abs(clean_score-normalized_zero_head_replacement_score)", "correlation": "Spearman average ranks within heads; fewer than4, nonfinite, or constant vector is NA", "aggregation": "within block path rho minus activation-gradient rho; arithmetic mean over blocks 0,3,7,11", "inference": "10000 image bootstrap percentile median CI; two-sided exact sign; Holm H1-H4 within each checkpoint"},
        "statistics": {"bootstrap_replicates": N_BOOTSTRAP, "bootstrap_level": 0.95, "bootstrap_quantile": "linear", "sign_test": "two-sided exact binomial on nonzero signs", "zero_rule": "abs(value-null)<=1e-12 is tie excluded from binomial but retained in N and interval", "posthoc_holm": "all nonhistorical Table6 image-level summary tests in one separate family"},
        "locality": {"status": "RUN_EXISTING_ARRAYS", "radii": [0, 1, 2], "grid": [14, 14], "metric": "Chebyshev-neighborhood mass minus equal-area uniform baseline; CLS excluded", "contrasts": ["late_minus_early", "block11_minus_block0"], "identity": "post-hoc descriptive"},
        "residual_quality": {"status": "RUN_EXISTING_ARRAYS", "quality": "identity component total absolute mass", "groups": "within-model and residual-type empirical quartiles frozen as Q1-Q4 rule", "outcome": "normalized-input output-identity patch total variation pooled across blocks", "identity": "post-hoc descriptive"},
        "amendment": {"attempt": "r4", "reason": "R3 coupled the fresh-root name to the bootstrap RNG namespace, causing avoidable Monte Carlo endpoint drift despite unchanged data and estimands. R4 restores the frozen R2 bootstrap namespace and adds explicit stage1 code-hash verification. R1-R3 remain immutable.", "predecessor_terminal_sha256": sha256(PREVIOUS_ROOT / "terminal_receipt.json"), "other_estimands_unchanged": True},
        "failure": {"table6_reproduction": "BLOCKED_TABLE6_REPRODUCTION", "h4_reproduction": "BLOCKED_H4_REPRODUCTION", "source_drift": "BLOCKED_SOURCE_BINDING_DRIFT"},
    }
    write_json(ROOT / "protocol.json", protocol)
    manuscripts = [PUBLICATION / lang / name for lang in ("zh", "en") for name in ("manuscript.md", "supplement.md", "supplement_experiments_v1.md")]
    receipt = {"schema_version": 1, "status": STAGE1_STATUS, "cpu_only": True, "protocol_sha256": sha256(ROOT / "protocol.json"), "source_bindings_sha256": sha256(ROOT / "source_bindings.json"), "analysis_schema_sha256": sha256(ROOT / "analysis_schema.json"), "counterexample_sha256": sha256(ROOT / "cut_definition_counterexample.json"), "script_sha256": sha256(SCRIPT), "validator_sha256": sha256(VALIDATOR), "test_sha256": sha256(TEST_FILE), "manuscripts": [file_record(p) for p in manuscripts], "old_families_unchanged": True, "new_results_accessed": False}
    write_json(ROOT / "stage1_receipt.json", receipt)
    print(json.dumps({"status": STAGE1_STATUS, "root": ROOT.relative_to(REPO).as_posix(), "source_files": len(bindings["files"])}, ensure_ascii=False))


def verify_frozen() -> dict[str, Any]:
    protocol = json.loads((ROOT / "protocol.json").read_text()); receipt = json.loads((ROOT / "stage1_receipt.json").read_text())
    if protocol["status"] != "FROZEN_BEFORE_NEW_STRATIFIED_RESULTS" or receipt["status"] != STAGE1_STATUS:
        raise ValueError("STAGE1_NOT_FROZEN")
    for key, path in (("script_sha256", SCRIPT), ("validator_sha256", VALIDATOR), ("test_sha256", TEST_FILE)):
        if receipt[key] != sha256(path): raise ValueError("FROZEN_CODE_DRIFT:" + key)
    bindings = json.loads((ROOT / "source_bindings.json").read_text())
    for row in bindings["files"]:
        path = REPO / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("BLOCKED_SOURCE_BINDING_DRIFT:" + row["path"])
    return protocol


def load_e4() -> list[dict[str, Any]]:
    index = json.loads(E4_INDEX.read_text()); out = []
    for record in index["records"]:
        document = json.loads((REPO / record["source_result_path"]).read_text())
        if document["canonical_digest"] != record["source_result_digest"] or document["task_id"] != record["task_id"]:
            raise ValueError("E4_BINDING")
        out.append(document["result"])
    if len(out) != 800: raise ValueError("E4_COVERAGE")
    return out


def append_metric(rows: list[dict[str, Any]], *, model: str, reference: str, image: str, metric: str, block: int | str, value: float | None, n: int, df: int | str = "", condition: float | str = "", reason: str = "", identity: str = "post_hoc_diagnostic") -> None:
    rows.append({"model": model, "reference": reference, "image": image, "metric": metric, "block": block, "value": "" if value is None else value, "n_components": n, "residual_df": df, "condition_number": condition, "na_reason": reason, "analysis_identity": identity})


def table6_analysis(e4: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float]:
    image_rows: list[dict[str, Any]] = []; scale_values: dict[tuple[str, str, str, int, str], list[float]] = defaultdict(list)
    legacy_lookup = {(row["model_id"], row["reference_id"], row["sample_id"]): row for row in json.loads((E6_ROOT / "per_image_statistics.json").read_text())["rows"]}
    legacy_families = {row["family_id"]: row["test_results"][0] for row in json.loads((E6_ROOT / "family_decisions.json").read_text())["families"] if row["claim"] == "C4a"}
    legacy_max_error = 0.0
    for result in e4:
        model = result["model_id"]; image = result["sample_id"]
        for reference in REFERENCES:
            components = result["finite_utility"][reference]["components"]
            for family, operations in (("heads", {"head"}), ("routing_value", {"routing", "value"}), ("residual_branches", {"attention", "mlp"})):
                part = [row for row in components if row["operation"] in operations]
                value, reason = spearman([row["m"] for row in part], [row["absolute_effect"] for row in part])
                expected = legacy_lookup[model, reference, image]["c4a"][family]["primary"]["value"]
                if value is None or expected is None: raise ValueError("BLOCKED_TABLE6_REPRODUCTION:NA")
                legacy_max_error = max(legacy_max_error, abs(value - expected))
                append_metric(image_rows, model=model, reference=reference, image=image, metric="historical_pooled_" + family, block="", value=value, n=len(part), reason=reason, identity="historical_frozen")
            for metric, operation in (("A_only_across_blocks", "routing"), ("V_only_across_blocks", "value"), ("attention_branch_only_across_blocks", "attention"), ("mlp_branch_only_across_blocks", "mlp")):
                part = sorted((row for row in components if row["operation"] == operation), key=lambda row: row["block"])
                value, reason = spearman([row["m"] for row in part], [row["absolute_effect"] for row in part])
                append_metric(image_rows, model=model, reference=reference, image=image, metric=metric, block="", value=value, n=len(part), reason=reason)
            head_rhos = []
            for block in range(12):
                part = sorted((row for row in components if row["operation"] == "head" and row["block"] == block), key=lambda row: row["component_id"])
                value, reason = spearman([row["m"] for row in part], [row["absolute_effect"] for row in part])
                append_metric(image_rows, model=model, reference=reference, image=image, metric="heads_within_block", block=block, value=value, n=len(part), reason=reason)
                if value is not None: head_rhos.append(float(np.clip(value, -1 + EPS, 1 - EPS)))
            fisher = float(np.tanh(np.mean(np.arctanh(head_rhos)))) if len(head_rhos) == 12 else None
            append_metric(image_rows, model=model, reference=reference, image=image, metric="heads_within_block_fisher_mean", block="", value=fisher, n=len(head_rhos), reason="" if fisher is not None else "NA_INCOMPLETE_BLOCKS")
            for family, operations, prefix in (("routing_value", {"routing", "value"}, "A_V"), ("residual_branches", {"attention", "mlp"}, "residual")):
                part = [row for row in components if row["operation"] in operations]
                for controls, suffix in ((("operation",), "operation"), (("block",), "block"), (("operation", "block"), "operation_and_block")):
                    value, df, condition, reason = residualized_rank_correlation(part, controls)
                    append_metric(image_rows, model=model, reference=reference, image=image, metric=f"{prefix}_adjusted_{suffix}_rank_correlation", block="", value=value, n=len(part), df=df, condition=condition, reason=reason)
            by = {(row["block"], row["operation"]): row for row in components if row["operation"] in {"routing", "value"}}
            signs = []
            for block in range(12):
                a = by[block, "routing"]; v = by[block, "value"]
                product = (a["m"] - v["m"]) * (a["absolute_effect"] - v["absolute_effect"]); signs.append(np.sign(product))
            nonzero = [x for x in signs if x != 0]
            concordance = None if not nonzero else float(np.mean(np.asarray(nonzero) > 0))
            append_metric(image_rows, model=model, reference=reference, image=image, metric="A_V_paired_concordance", block="", value=concordance, n=len(nonzero), reason="" if nonzero else "NA_ALL_TIES")
            for row in components:
                if row["operation"] in {"routing", "value", "attention", "mlp"}:
                    family = "A_V" if row["operation"] in {"routing", "value"} else "residual"
                    scale_values[model, reference, family, int(row["block"]), "log1p_m|" + row["operation"]].append(math.log1p(float(row["m"])))
                    scale_values[model, reference, family, int(row["block"]), "log1p_abs_effect|" + row["operation"]].append(math.log1p(float(row["absolute_effect"])))
    if legacy_max_error > 1e-12: raise ValueError("BLOCKED_TABLE6_REPRODUCTION")
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    totals: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in image_rows:
        key = (row["model"], row["reference"], row["metric"], str(row["block"]), row["analysis_identity"]); totals[key] += 1
        if row["value"] != "": grouped[key].append(float(row["value"]))
    summary = []
    for key in sorted(totals):
        values = grouped[key]; model, reference, metric, block, identity = key
        estimate, low, high = bootstrap_median(values, "table6|" + "|".join(key))
        signs = exact_two_sided_sign(values) if values else {"positive": 0, "negative": 0, "ties": 0, "raw_p": 1.0}
        if metric == "A_V_paired_concordance":
            signs = {"positive": "", "negative": "", "ties": "", "raw_p": ""}
        historical_holm = ""
        if identity == "historical_frozen":
            family = metric.removeprefix("historical_pooled_")
            historical_holm = legacy_families[f"C4a.{model}.{reference}.{family}"]["holm_adjusted_p_value"]
        summary.append({"model": model, "reference": reference, "metric": metric, "block": block, "n_total": totals[key], "n_valid": len(values), "n_na": totals[key] - len(values), "median": estimate, "ci95_low": low, "ci95_high": high, "q10": float(np.quantile(values, .10)) if values else math.nan, "q25": float(np.quantile(values, .25)) if values else math.nan, "q75": float(np.quantile(values, .75)) if values else math.nan, "q90": float(np.quantile(values, .90)) if values else math.nan, **signs, "holm_p": historical_holm, "analysis_identity": identity})
    tested = [row for row in summary if row["analysis_identity"] == "post_hoc_diagnostic" and row["block"] in ("", "None") and row["n_valid"] > 0 and row["metric"] != "A_V_paired_concordance"]
    holm(tested)
    scale_rows = []
    for key, values in sorted(scale_values.items()):
        model, reference, family, block, quantity_operation = key; quantity, operation = quantity_operation.split("|")
        estimate, low, high = bootstrap_median(values, "scale|" + "|".join(map(str, key)))
        scale_rows.append({"model": model, "reference": reference, "family": family, "operation": operation, "block": block, "quantity": quantity, "n": len(values), "median": estimate, "ci95_low": low, "ci95_high": high, "q10": float(np.quantile(values, .1)), "q90": float(np.quantile(values, .9))})
    return image_rows, summary, scale_rows, legacy_max_error


def h4_analysis() -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    old_values: dict[tuple[str, str, str], float] = {}
    with (E5_ANALYSIS / "e5_image_values.csv").open() as stream:
        for row in csv.DictReader(stream): old_values[row["model"], row["image"], row["hypothesis"]] = float(row["value"])
    image_rows = []; all_h: dict[tuple[str, str], list[float]] = defaultdict(list); max_error = 0.0
    for metadata in sorted(E5_ROOT.glob("*.json")):
        record = json.loads(metadata.read_text()); model = record["model_id"]; image = record["image"]
        output = REPO / record["output"]
        if record["status"] != "VALID" or sha256(output) != record["output_sha256"]: raise ValueError("E5_BINDING")
        values: dict[str, dict[int, float]] = defaultdict(dict); blocks = []
        with np.load(output, allow_pickle=False) as arrays:
            for block in BLOCKS:
                path = arrays[f"b{block:02d}_heads_absolute_mass"].astype(np.float64)
                ag = np.abs(arrays[f"b{block:02d}_head_activation_gradient"].astype(np.float64))
                effect = np.abs(arrays[f"b{block:02d}_normalized_zero_head_effect"].astype(np.float64))
                path_rho, r1 = spearman(path, effect); ag_rho, r2 = spearman(ag, effect)
                difference = None if path_rho is None or ag_rho is None else path_rho - ag_rho
                blocks.append((block, path_rho, ag_rho, difference, len(path), tie_count(path), tie_count(ag), tie_count(effect), r1 or r2))
                values["entropy"][block] = float(np.median(arrays[f"b{block:02d}_post_entropy"]))
                get = lambda key: float(arrays[f"b{block:02d}_{key}_absolute_mass"][0])
                a, v = get("routing"), get("V"); values["routing"][block] = a / (a + v) if a + v else math.nan
                a, v = get("carry"), get("mlp_branch"); values["mlp_identity"][block] = a / (a + v) if a + v else math.nan
                a, v = get("identity"), get("attention"); values["attention_identity"][block] = a / (a + v) if a + v else math.nan
        diffs = [row[3] for row in blocks if row[3] is not None]
        h = {"H1": values["entropy"][11] - values["entropy"][0], "H2": values["routing"][11] - values["routing"][0], "H3": (values["mlp_identity"][11] - values["mlp_identity"][0]) - (values["attention_identity"][11] - values["attention_identity"][0]), "H4": float(np.mean(diffs)) if len(diffs) == 4 else math.nan}
        for hypothesis, value in h.items():
            expected = old_values[model, image, hypothesis]; max_error = max(max_error, abs(value - expected)); all_h[model, hypothesis].append(value)
        for block, path_rho, ag_rho, difference, n, pt, at, et, reason in blocks:
            image_rows.append({"model": model, "image": image, "block": block, "path_map_mass_rho": "" if path_rho is None else path_rho, "activation_gradient_rho": "" if ag_rho is None else ag_rho, "difference": "" if difference is None else difference, "h4_image_mean": h["H4"], "valid_heads": n, "path_ties": pt, "activation_gradient_ties": at, "effect_ties": et, "na_reason": reason})
    if max_error > 1e-12: raise ValueError("BLOCKED_H4_REPRODUCTION")
    summary = []
    for (model, hypothesis), values in sorted(all_h.items()):
        finite = np.asarray(values, float); finite = finite[np.isfinite(finite)]
        estimate, low, high = bootstrap_median(finite, f"h4|{model}|{hypothesis}"); signs = exact_two_sided_sign(finite)
        summary.append({"model": model, "hypothesis": hypothesis, "n_total": len(values), "n_valid": len(finite), "n_na": len(values) - len(finite), "median": estimate, "ci95_low": low, "ci95_high": high, **signs, "holm_p": "", "decision": ""})
    for model in E5_MODELS:
        part = [row for row in summary if row["model"] == model]; holm(part)
        for row in part: row["decision"] = "SUPPORTED_DIRECTION" if row["holm_p"] <= .05 and row["median"] > 0 else "OPPOSITE_DIRECTION" if row["holm_p"] <= .05 and row["median"] < 0 else "NOT_SUPPORTED"
    return image_rows, summary, max_error


def locality_analysis() -> list[dict[str, Any]]:
    manifest = json.loads((M3_ROOT / "task_manifest.json").read_text()); grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for task in manifest["tasks"]:
        model = task["payload"]["model_id"]; path = M3_ROOT / "task_payloads" / task["task_id"] / "metrics.npz"
        with np.load(path, allow_pickle=False) as arrays:
            block = arrays["block"]; token = arrays["token"]; mass = arrays["patch_absolute_mass_196"].astype(np.float64)
            per = {radius: {} for radius in (0, 1, 2)}
            for b in range(12):
                keep = np.where((block == b) & (token > 0))[0]
                target = token[keep] - 1; total = mass[keep].sum(1); valid = total > 0
                for radius in (0, 1, 2):
                    values = []
                    for index, patch in zip(keep[valid], target[valid]):
                        row, col = divmod(int(patch), 14); cells = [r * 14 + c for r in range(max(0, row-radius), min(14, row+radius+1)) for c in range(max(0, col-radius), min(14, col+radius+1))]
                        values.append(float(mass[index, cells].sum() / mass[index].sum() - len(cells) / 196.0))
                    per[radius][b] = values
            for radius in (0, 1, 2):
                grouped[model, radius, "late_minus_early"].append(float(np.median(sum((per[radius][b] for b in range(8,12)), [])) - np.median(sum((per[radius][b] for b in range(4)), []))))
                grouped[model, radius, "block11_minus_block0"].append(float(np.median(per[radius][11]) - np.median(per[radius][0])))
    rows = []
    for key, values in sorted(grouped.items()):
        model, radius, contrast = key; estimate, low, high = bootstrap_median(values, "locality|" + "|".join(map(str,key)))
        rows.append({"model": model, "radius": radius, "contrast": contrast, "n": len(values), "median": estimate, "ci95_low": low, "ci95_high": high, "q10": float(np.quantile(values,.1)), "q90": float(np.quantile(values,.9)), "analysis_identity": "post_hoc_descriptive"})
    return rows


def residual_quality_analysis() -> list[dict[str, Any]]:
    raw: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
    for path in sorted((V6_ROOT / "tasks").glob("structure__*.npz")):
        model = path.name.split("__")[1]
        with np.load(path, allow_pickle=False) as arrays:
            for residual, identity, output in (("attention", "attention_identity", "attention_output"), ("mlp", "mlp_identity", "mlp_output")):
                qi = (arrays["coordinate"] == "normalized_endpoint") & (arrays["object"] == identity)
                qo = (arrays["coordinate"] == "normalized_endpoint") & (arrays["object"] == output)
                im = arrays["patch_mass"][qi].astype(np.float64); om = arrays["patch_mass"][qo].astype(np.float64)
                ib = arrays["block"][qi]; ob = arrays["block"][qo]
                for block in range(12):
                    a=im[ib==block][0]; b=om[ob==block][0]; mass=float(a.sum())
                    tv=float(.5*np.abs(a/a.sum()-b/b.sum()).sum()) if a.sum()>0 and b.sum()>0 else math.nan
                    image = path.stem.split("__", 2)[2]
                    raw[model,residual].append((image,mass,tv))
    rows=[]
    for (model,residual), values in sorted(raw.items()):
        masses=np.asarray([x[1] for x in values]); edges=np.quantile(masses,[.25,.5,.75]); by_image: dict[tuple[str,int],list[float]]=defaultdict(list)
        bins=np.digitize(masses,edges,right=True)
        for (image,_mass,tv),quartile in zip(values,bins):
            if np.isfinite(tv): by_image[image,int(quartile)].append(tv)
        for q in range(4):
            selected=np.asarray([float(np.median(v)) for (image,quartile),v in by_image.items() if quartile==q],dtype=np.float64); estimate,low,high=bootstrap_median(selected,f"tvq|{model}|{residual}|{q+1}")
            rows.append({"model":model,"residual_type":residual,"mass_quartile":f"Q{q+1}","n":len(selected),"median_tv":estimate,"ci95_low":low,"ci95_high":high,"q10":float(np.quantile(selected,.1)),"q90":float(np.quantile(selected,.9)),"analysis_identity":"post_hoc_descriptive"})
    return rows


def pvalue_audit(table6: Sequence[Mapping[str, Any]], h4: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows=[]
    for row in table6:
        if row["analysis_identity"] == "historical_frozen":
            claim=f"C4a.{row['model']}.{row['reference']}.{row['metric'].removeprefix('historical_pooled_')}"
            family="original_28_joint_Holm"
        elif str(row["block"]) in ("", "None") and row["metric"] != "A_V_paired_concordance":
            claim=f"V7.{row['model']}.{row['reference']}.{row['metric']}";family="v7_table6_posthoc_joint_Holm"
        else: continue
        hp=row["holm_p"] if row["holm_p"] != "" else row["raw_p"]
        rows.append({"claim_id":claim,"source_rows":f"table6_stratified_image_values.csv:{row['model']}|{row['reference']}|{row['metric']}","statistic":"image-level value","null":"equal positive/negative sign probability among non-ties","alternative":"two-sided","unit":"image","ties":row["ties"],"raw_p":row["raw_p"],"family":family,"holm_p":hp,"display":"p<10^{-6}" if float(hp)<1e-6 else f"p={float(hp):.3g}","recalculation":"v7_from_bound_per_image_values"})
    for row in h4:
        claim=f"{row['model']}.{row['hypothesis']}";hp=float(row["holm_p"])
        rows.append({"claim_id":claim,"source_rows":f"bound_E5_raw:{row['model']}|{row['hypothesis']}","statistic":"image-level H value","null":"equal positive/negative sign probability among non-ties","alternative":"two-sided","unit":"image","ties":row["ties"],"raw_p":row["raw_p"],"family":f"H1-H4_within_{row['model']}","holm_p":hp,"display":"p<10^{-6}" if hp<1e-6 else f"p={hp:.3g}","recalculation":"v7_from_bound_E5_JSON_NPZ"})
    return rows


def analyze() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or set(sys.modules).intersection({"torch", "timm", "PIL"}): raise ValueError("CPU_ONLY_BOUNDARY")
    verify_frozen()
    outputs = [ROOT / name for name in CSV_HEADERS]
    if any(path.exists() for path in outputs + [ROOT / "artifact_manifest.json", ROOT / "validation_report.json", ROOT / "terminal_receipt.json"]): raise ValueError("ANALYSIS_OUTPUT_EXISTS")
    t6image,t6summary,scale,legacy_error=table6_analysis(load_e4())
    h4image,h4summary,h4error=h4_analysis(); locality=locality_analysis(); quality=residual_quality_analysis(); pvalues=pvalue_audit(t6summary,h4summary)
    write_csv(ROOT/"table6_stratified_image_values.csv",t6image,CSV_HEADERS["table6_stratified_image_values.csv"]);write_csv(ROOT/"table6_stratified_summary.csv",t6summary,CSV_HEADERS["table6_stratified_summary.csv"]);write_csv(ROOT/"table6_scale_diagnostics.csv",scale,CSV_HEADERS["table6_scale_diagnostics.csv"]);write_csv(ROOT/"h4_recomputed_image_values.csv",h4image,CSV_HEADERS["h4_recomputed_image_values.csv"]);write_csv(ROOT/"h4_recomputed_summary.csv",h4summary,CSV_HEADERS["h4_recomputed_summary.csv"]);write_csv(ROOT/"pvalue_audit.csv",pvalues,CSV_HEADERS["pvalue_audit.csv"]);write_csv(ROOT/"locality_radius_sensitivity.csv",locality,CSV_HEADERS["locality_radius_sensitivity.csv"]);write_csv(ROOT/"residual_tv_quality_diagnostics.csv",quality,CSV_HEADERS["residual_tv_quality_diagnostics.csv"])
    state={"schema_version":1,"status":"ANALYSIS_COMPLETE_PENDING_INDEPENDENT_VALIDATION","cpu_only":True,"gpu_inference_runs":0,"legacy_table6_max_abs_error":legacy_error,"h4_max_abs_error":h4error,"row_counts":{p.name:sum(1 for _ in p.open())-1 for p in outputs},"outputs":[file_record(p) for p in outputs]}
    write_json(ROOT/"analysis_run.json",state)
    manifest_files=[p for p in sorted(ROOT.iterdir()) if p.is_file() and p.name not in {"artifact_manifest.json","validation_report.json","terminal_receipt.json"}]
    write_json(ROOT/"artifact_manifest.json",{"schema_version":1,"files":[file_record(p) for p in manifest_files]})
    print(json.dumps({"status":state["status"],"row_counts":state["row_counts"],"legacy_error":legacy_error,"h4_error":h4error},ensure_ascii=False))


def main() -> int:
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument("--freeze",action="store_true");group.add_argument("--analyze",action="store_true");args=parser.parse_args()
    if args.freeze: freeze()
    else: analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
