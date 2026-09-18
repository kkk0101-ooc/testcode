#!/usr/bin/env python3
"""Independent source-recomputing validator for ENDPOINT-E5."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_e5_spatial_r1_20260918"
SOURCE_E4 = REPO / "reports/nns_vit_endpoint_e4_intervention_r3_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e5_spatial_run_spec.json"
PASS_STATUS = "ENDPOINT_E5_PASS_SPATIAL_INTERVENTION_COMPLETE_NO_CLAIM_DECISION"
NEXT = "ENDPOINT_E6_REQUIRES_EXPLICIT_USER_REQUEST"
QUEUE_NAMESPACE = "NNS_VIT_ENDPOINT_QUEUE_V1"
RANDOM_NAMESPACE = "NNS_VIT_ENDPOINT_E5_RANDOM_REGIONS_V1"
METHODS = ("endpoint_component", "total_gxi", "raw_gradient", "regional_taylor")
BASELINES = ("total_gxi", "raw_gradient", "regional_taylor", "random_regions")
REFERENCES = ("normalized_zero", "blur_rgb")
HEADS = {"vit_b16": (0, 6, 11), "deit_s16": (0, 3, 5)}
COUNTS = {"candidate_region_count": 16, "reference_count": 2, "selected_head_count": 3,
          "curve_count": 144, "random_curve_count": 120,
          "candidate_taylor_forward_evaluations": 32, "finite_union_full_forward_curve_points": 432,
          "finite_union_isolated_head_evaluations": 432}
SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_e5_spatial_run_spec.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE5执行合同_20260918.md",
    "src/vitnns/endpoint_e5_spatial.py", "scripts/run_endpoint_e5_spatial_worker.py",
    "scripts/run_endpoint_e5_spatial.py", "scripts/validate_endpoint_e5_spatial.py",
    "scripts/validate_endpoint_unified_plan.py", "tests/test_endpoint_e5_spatial.py",
]
VALIDATOR_AMENDMENT = "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE5封存验证器修订R1_20260918.md"
BINDINGS = {
    "cohort_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
    "endpoint_e4_r3_terminal_sha256": "reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json",
    "endpoint_e4_r3_validation_sha256": "reports/nns_vit_endpoint_e4_intervention_r3_20260918/validation_report.json",
    "intervention_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/intervention_registry.json",
    "materialization_receipt_sha256": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
    "model_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
    "statistical_analysis_plan_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json",
    "target_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
    "unified_plan_sha256": "docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md",
}
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import digest, file_sha256  # noqa: E402
from vitnns.endpoint_e5_spatial import validate_run_spec  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def close(left: float, right: float, code: str, tolerance: float = 1e-12) -> None:
    require(math.isfinite(float(left)) and math.isfinite(float(right)) and
            math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance), code)


def git_blob_sha(commit: str, relative: str) -> str:
    data = subprocess.check_output(["git", "show", f"{commit}:vitnns_rebuild/{relative}"], cwd=REPO)
    return hashlib.sha256(data).hexdigest()


def queue_seed(task_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{QUEUE_NAMESPACE}|{task_id}".encode()).digest()[:8], "big")


def random_order(task_seed: int, model_id: str, reference: str, head: int, repeat: int) -> tuple[int, list[int]]:
    payload = f"{RANDOM_NAMESPACE}|{task_seed}|{model_id}|{reference}|{head}|{repeat}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return seed, [int(value) for value in np.random.Generator(np.random.PCG64(seed)).permutation(16)]


def ranking(scores: list[float]) -> list[int]:
    require(len(scores) == 16 and all(math.isfinite(float(value)) for value in scores), "RANK_SCORE")
    return sorted(range(16), key=lambda index: (-float(scores[index]), index))


def auc(values: list[float]) -> float:
    require(len(values) == 4 and all(math.isfinite(float(value)) and float(value) >= 0 for value in values), "AUC_INPUT")
    areas = (0.0, .0625, .125, .25)
    return sum((areas[index + 1] - areas[index]) * (values[index] + values[index + 1]) / 2
               for index in range(3)) / .25


def masks() -> list[str]:
    rows = []
    for region in range(16):
        mask = np.zeros((224, 224), dtype=np.uint8); row, column = divmod(region, 4)
        mask[row * 56:(row + 1) * 56, column * 56:(column + 1) * 56] = 1
        rows.append(hashlib.sha256(mask.tobytes()).hexdigest())
    return rows


def region_scores(value: np.ndarray) -> list[float]:
    require(value.shape == (1, 3, 224, 224) and value.dtype == np.float32 and np.isfinite(value).all(), "RAW_MAP_ARRAY")
    output = []
    for region in range(16):
        row, column = divmod(region, 4)
        output.append(float(np.abs(value.astype(np.float64)[:, :, row * 56:(row + 1) * 56,
                                                        column * 56:(column + 1) * 56]).sum()))
    return output


def validate_reference(result: dict, reference: str, task_seed: int) -> None:
    value = result["references"][reference]
    require(set(value) == {"ranking_scores", "candidate_single_region_full_forward_scores",
                           "candidate_single_region_full_forward_effects", "unique_multi_region_full_forward_input_count", "curves", "random_aggregates",
                           "image_method_scores", "paired_endpoint_minus_baseline"}, "REFERENCE_SCHEMA")
    scores = value["ranking_scores"]
    require(set(scores) == set(METHODS), "RANKER_COVERAGE")
    require(set(scores["endpoint_component"]) == {f"head_{head}" for head in result["heads"]} and
            set(scores["regional_taylor"]) == {f"head_{head}" for head in result["heads"]} and
            isinstance(scores["total_gxi"], list) and isinstance(scores["raw_gradient"], list), "RANKER_SCHEMA")
    for method in METHODS:
        rows = scores[method].values() if isinstance(scores[method], dict) else (scores[method],)
        for row in rows:
            ranking(list(row))
    candidates = value["candidate_single_region_full_forward_scores"]
    effects = value["candidate_single_region_full_forward_effects"]
    require(len(candidates) == len(effects) == 16, "CANDIDATE_COVERAGE")
    require(1 <= value["unique_multi_region_full_forward_input_count"] <= 144, "UNIQUE_MULTI_REGION_COUNT")
    for score, effect in zip(candidates, effects): close(effect, result["clean_score"] - score, "CANDIDATE_EFFECT", 1e-10)
    curves = value["curves"]
    require(len(curves) == len({row["curve_id"] for row in curves}) == 72, "CURVE_COVERAGE")
    expected_counts = {"endpoint_component": 3, "total_gxi": 3, "raw_gradient": 3,
                       "regional_taylor": 3, "random_regions": 60}
    require({method: sum(row["method"] == method for row in curves) for method in (*METHODS, "random_regions")} == expected_counts,
            "CURVE_METHOD_COUNTS")
    by_method_head: dict[tuple[str, int], list[dict]] = {}
    for curve in curves:
        require(set(curve) == {"curve_id", "method", "head", "repeat", "random_seed", "ranking", "points",
                               "absolute_isolated_effect_auc_div_0_25"}, "CURVE_SCHEMA")
        method, head = curve["method"], int(curve["head"])
        require(head in result["heads"] and method in (*METHODS, "random_regions"), "CURVE_IDENTITY")
        if method == "random_regions":
            repeat = int(curve["repeat"]); seed, order = random_order(task_seed, result["model_id"], reference, head, repeat)
            require(curve["random_seed"] == seed and curve["ranking"] == order and
                    curve["curve_id"] == f"random_regions.head_{head}.repeat_{repeat:02d}", "RANDOM_CURVE")
        else:
            expected_ranking = (ranking(scores[method][f"head_{head}"])
                                if method in {"endpoint_component", "regional_taylor"}
                                else ranking(scores[method]))
            require(curve["repeat"] is None and curve["random_seed"] is None and
                    curve["ranking"] == expected_ranking and
                    curve["curve_id"] == f"{method}.head_{head}", "DETERMINISTIC_CURVE")
        require(len(curve["points"]) == 4, "POINT_COVERAGE")
        for point, count, area in zip(curve["points"], (0, 1, 2, 4), (0.0, .0625, .125, .25)):
            require(set(point) == {"budget_count", "actual_area_fraction", "selected_regions", "isolated_score",
                                   "isolated_effect", "full_forward_score", "full_forward_effect"}, "POINT_SCHEMA")
            require(point["budget_count"] == count and point["actual_area_fraction"] == area and
                    point["selected_regions"] == curve["ranking"][:count], "POINT_SELECTION")
            close(point["isolated_effect"], result["clean_score"] - point["isolated_score"], "ISOLATED_EFFECT", 1e-10)
            close(point["full_forward_effect"], result["clean_score"] - point["full_forward_score"], "FULL_EFFECT", 1e-10)
            if count == 0:
                close(point["isolated_effect"], 0.0, "ZERO_ISOLATED"); close(point["full_forward_effect"], 0.0, "ZERO_FULL")
        close(curve["absolute_isolated_effect_auc_div_0_25"], auc([abs(row["isolated_effect"]) for row in curve["points"]]), "CURVE_AUC")
        by_method_head.setdefault((method, head), []).append(curve)
    method_scores = value["image_method_scores"]
    require(set(method_scores) == {"endpoint_component", *BASELINES}, "METHOD_SCORE_SCHEMA")
    for method in METHODS:
        expected = np.mean([by_method_head[(method, head)][0]["absolute_isolated_effect_auc_div_0_25"] for head in result["heads"]])
        close(method_scores[method], expected, "METHOD_SCORE")
    aggregates = value["random_aggregates"]
    require(len(aggregates) == 3 and {row["head"] for row in aggregates} == set(result["heads"]), "RANDOM_AGGREGATE_COVERAGE")
    random_head_scores = []
    for aggregate in aggregates:
        head = aggregate["head"]; selected = sorted(by_method_head[("random_regions", head)], key=lambda row: row["repeat"])
        means = [float(np.mean([row["points"][index]["isolated_effect"] for row in selected])) for index in range(4)]
        require(aggregate["repeat_count"] == 20, "RANDOM_REPEAT_COUNT")
        for observed, expected in zip(aggregate["mean_signed_isolated_effect_curve"], means): close(observed, expected, "RANDOM_MEAN")
        for observed, expected in zip(aggregate["absolute_of_mean_isolated_effect_curve"], map(abs, means)): close(observed, expected, "RANDOM_ABSOLUTE")
        expected_auc = auc(list(map(abs, means))); close(aggregate["absolute_isolated_effect_auc_div_0_25"], expected_auc, "RANDOM_AUC")
        random_head_scores.append(expected_auc)
    close(method_scores["random_regions"], float(np.mean(random_head_scores)), "RANDOM_METHOD_SCORE")
    require(set(value["paired_endpoint_minus_baseline"]) == set(BASELINES), "PAIRED_SCHEMA")
    for baseline in BASELINES:
        close(value["paired_endpoint_minus_baseline"][baseline], method_scores["endpoint_component"] - method_scores[baseline], "PAIRED_VALUE")


def validate_result(result: dict, frozen: dict, task_seed: int, e4_result: dict, root: Path) -> None:
    required = {"stage", "claim_scope", "model_id", "sample_id", "split_order", "class_id", "wnid", "image_sha256",
        "image_bytes", "original_size_wh", "input_shape", "dtype", "reference_identity", "block", "heads", "clean_score",
        "clean_predicted_class", "candidate_mask_sha256", "budgets", "actual_area_fractions", "references", "counts",
        "raw_maps_record", "scan_seconds", "task_seconds", "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes",
        "peak_cpu_rss_bytes", "model_binding", "worker_identity"}
    require(set(result) == required, "RESULT_SCHEMA")
    require(result["stage"] == "ENDPOINT-E5" and result["claim_scope"] == "SPATIAL_PER_IMAGE_INPUTS_NO_GLOBAL_INFERENCE_OR_CLAIM_DECISION", "RESULT_SCOPE")
    require((result["sample_id"], result["split_order"], result["class_id"], result["wnid"], result["image_sha256"], result["image_bytes"]) ==
            (frozen["sample_id"], frozen["split_order"], frozen["class_id"], frozen["wnid"], frozen["sha256"], frozen["bytes"]), "RESULT_COHORT")
    require(result["input_shape"] == [1, 3, 224, 224] and result["dtype"] == "torch.float32", "INPUT_CONTRACT")
    require(result["block"] == 5 and tuple(result["heads"]) == HEADS[result["model_id"]], "BLOCK_HEADS")
    require(result["candidate_mask_sha256"] == masks() and result["budgets"] == [0, 1, 2, 4] and
            result["actual_area_fractions"] == [0.0, .0625, .125, .25] and result["counts"] == COUNTS, "SPATIAL_IDENTITY")
    require(result["reference_identity"] == e4_result["reference_identity"] and result["model_binding"] == e4_result["model_binding"], "E4_IDENTITY_REBIND")
    require(set(result["references"]) == set(REFERENCES), "REFERENCE_COVERAGE")
    for reference in REFERENCES: validate_reference(result, reference, task_seed)
    identity = result["worker_identity"]
    require(identity["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and identity["startup_matmul_backward"] == "PASS" and
            identity["deterministic_algorithms"] and not identity["cuda_matmul_allow_tf32"] and
            not identity["cudnn_allow_tf32"] and not identity["autocast"], "WORKER_IDENTITY")
    require(result["scan_seconds"] > 0 and result["task_seconds"] >= result["scan_seconds"] and
            result["task_seconds"] <= 1800 and result["peak_reserved_gpu_bytes"] <= 24 * 1024 ** 3 and
            result["peak_cpu_rss_bytes"] <= 32 * 1024 ** 3, "TASK_RESOURCE")
    raw = result["raw_maps_record"]
    require((raw is not None) == (result["split_order"] < 16), "RAW_MAP_SCOPE")
    if raw is not None:
        path = (REPO / raw["path"]).resolve(); require(path.is_relative_to(root) and path.is_file(), "RAW_MAP_PATH")
        require(path.stat().st_size == raw["bytes"] and file_sha256(path) == raw["sha256"], "RAW_MAP_FILE")
        with np.load(path, allow_pickle=False) as arrays:
            require(set(arrays.files) == {"total_gxi", "raw_gradient", *{f"head_{head}" for head in result["heads"]}}, "RAW_MAP_KEYS")
            records = {row["name"]: row for row in raw["arrays"]}; require(set(records) == set(arrays.files), "RAW_MAP_RECORDS")
            for name in arrays.files:
                value = arrays[name]; computed = region_scores(value); row = records[name]
                require(row["shape"] == [1, 3, 224, 224] and row["dtype"] == "float32" and
                        row["array_sha256"] == hashlib.sha256(value.tobytes()).hexdigest(), "RAW_MAP_DIGEST")
                source = (result["references"]["normalized_zero"]["ranking_scores"]["endpoint_component"][name]
                          if name.startswith("head_") else result["references"]["normalized_zero"]["ranking_scores"][name])
                for observed, expected in zip(source, computed): close(observed, expected, "RAW_MAP_REGION_SCORE", 1e-8)
                for reference in REFERENCES:
                    other = (result["references"][reference]["ranking_scores"]["endpoint_component"][name]
                             if name.startswith("head_") else result["references"][reference]["ranking_scores"][name])
                    require(other == source, "REFERENCE_INDEPENDENT_RANKER_DRIFT")


def e4_index() -> dict[str, dict]:
    combined = load(SOURCE_E4 / "combined_result_index.json"); output = {}
    for row in combined["records"]:
        output[row["task_id"]] = load(REPO / row["source_result_path"])["result"]
    require(len(output) == 800, "E4_INDEX")
    return output


def validate(root: Path) -> dict:
    root = root.resolve(); spec = load(SPEC_PATH); validate_run_spec(spec)
    require(spec["outputs"]["formal_root"] == root.relative_to(REPO).as_posix(), "SPEC_ROOT")
    report = load(root / "validation_report.json"); terminal = load(root / "terminal_receipt.json"); manifest = load(root / "artifact_manifest.json")
    require(report["status"] == terminal["status"] == PASS_STATUS and report["stage"] == terminal["stage"] == "ENDPOINT-E5", "STATUS")
    require(terminal["validation_digest"] == digest(report) and terminal["manifest_digest"] == manifest["canonical_digest"] == digest(manifest["files"]), "TERMINAL_DIGEST")
    paths = [row["path"] for row in manifest["files"]]; require(len(paths) == len(set(paths)), "MANIFEST_DUPLICATE")
    for row in manifest["files"]:
        path = (root / row["path"]).resolve(); require(path.is_relative_to(root) and path.is_file(), "MANIFEST_PATH")
        require(path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"], "MANIFEST_FILE")
    producer = report["producer_commit"]
    require(subprocess.run(["git", "merge-base", "--is-ancestor", producer, "HEAD"], cwd=REPO).returncode == 0, "PRODUCER_ANCESTRY")
    require({row["path"] for row in report["implementation_bindings"]} == set(SOURCE_FILES), "IMPLEMENTATION_COVERAGE")
    for row in report["implementation_bindings"]:
        require(row["commit"] == producer and git_blob_sha(producer, row["path"]) == row["sha256"], "IMPLEMENTATION_BINDING")
    validator_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--",
        "vitnns_rebuild/scripts/validate_endpoint_e5_spatial.py"], cwd=REPO, text=True).strip()
    require(git_blob_sha(validator_commit, "scripts/validate_endpoint_e5_spatial.py") == file_sha256(Path(__file__)) and
            git_blob_sha(validator_commit, VALIDATOR_AMENDMENT) == file_sha256(REPO / VALIDATOR_AMENDMENT),
            "VALIDATOR_AMENDMENT_BINDING")
    for key, relative in BINDINGS.items():
        observed = git_blob_sha(producer, relative) if key == "unified_plan_sha256" else file_sha256(REPO / relative)
        require(observed == spec["predecessors"][key], "PREDECESSOR_BINDING:" + key)
    material = load(REPO / spec["outputs"]["source_materialization"])
    require(material["records_digest"] == report["materialization_records_digest"] == digest(material["records"]), "MATERIALIZATION")
    frozen = {row["split_order"]: row for row in material["records"]}; require(len(frozen) == 400, "COHORT")
    for row in frozen.values():
        path = REPO / row["cache_path"]; require(path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"], "IMAGE_SOURCE_HASH")
    tasks = load(root / "task_manifest.json")
    require(tasks["task_count"] == 800 and tasks["manifest_digest"] == digest(tasks["tasks"]) and
            tasks["queue_identity"]["manifest_digest"] == tasks["manifest_digest"], "TASK_MANIFEST")
    with sqlite3.connect(root / "formal_queue.sqlite3") as connection:
        queue_rows = connection.execute("SELECT task_id,payload_json,payload_digest,seed,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id").fetchall()
        attempts = connection.execute("SELECT status FROM attempts").fetchall()
    require(len(queue_rows) == len(attempts) == 800 and all(row[4] == "completed" for row in queue_rows) and
            all(row[0] == "COMPLETED" for row in attempts), "QUEUE_COVERAGE")
    e4 = e4_index(); queue_manifest = []; formal = {}
    for task_id, payload_json, payload_digest, seed, state, attempt, result_path, result_digest in queue_rows:
        require(hashlib.sha256(payload_json.encode()).hexdigest() == payload_digest and int(seed) == queue_seed(task_id), "QUEUE_IDENTITY")
        payload = json.loads(payload_json); queue_manifest.append({"task_id": task_id, "payload": payload})
        path = root / "formal_results" / result_path; document = load(path); signed = dict(document); observed = signed.pop("canonical_digest")
        require(observed == result_digest == digest(signed) and document["task_id"] == task_id and document["attempt"] == attempt, "RESULT_DIGEST")
        model_id, order_text = task_id.rsplit("__", 1); order = int(order_text)
        require(payload["stage"] == "ENDPOINT-E5" and payload["split_order"] == order and payload["model_id"] == model_id and
                payload["retain_raw_maps"] == (order < 16), "TASK_PAYLOAD")
        result = document["result"]; require(result["model_id"] == model_id, "RESULT_MODEL")
        validate_result(result, frozen[order], int(seed), e4[task_id], root); formal[task_id] = result
    require(queue_manifest == tasks["tasks"], "QUEUE_MANIFEST_BINDING")
    require(report["image_model_tasks"] == terminal["image_model_task_count"] == 800 and report["evaluation_images"] == terminal["evaluation_image_count"] == 400, "REPORT_COHORT")
    require(report["curves"] == terminal["curve_count"] == 115200 and report["random_curves"] == terminal["random_curve_count"] == 96000, "REPORT_CURVES")
    require(report["candidate_taylor_forward_evaluations"] == 25600 and report["finite_union_full_forward_curve_points"] == 345600 and
            report["finite_union_isolated_head_evaluations"] == 345600 and report["raw_map_task_count"] == 32, "REPORT_COUNTS")
    require(report["unique_multi_region_full_forward_inputs"] == sum(reference["unique_multi_region_full_forward_input_count"] for row in formal.values() for reference in row["references"].values()), "REPORT_UNIQUE_INPUTS")
    expected_summary = []
    for model_id in ("vit_b16", "deit_s16"):
        model_rows = [row for row in formal.values() if row["model_id"] == model_id]
        for reference in REFERENCES:
            method_values = {method: [row["references"][reference]["image_method_scores"][method] for row in model_rows]
                             for method in ("endpoint_component", *BASELINES)}
            paired = {baseline: [row["references"][reference]["paired_endpoint_minus_baseline"][baseline] for row in model_rows]
                      for baseline in BASELINES}
            expected_summary.append({"model_id": model_id, "reference_id": reference, "n": 400,
                "median_image_method_scores": {key: float(np.median(value)) for key, value in method_values.items()},
                "median_paired_endpoint_minus_baseline": {key: float(np.median(value)) for key, value in paired.items()},
                "inference_performed": False})
    require(report["descriptive_per_image_score_summary"] == expected_summary, "DESCRIPTIVE_SUMMARY")
    e4_report = load(SOURCE_E4 / "validation_report.json")
    close(report["combined_e4_e5_gpu_hours_observed"], e4_report["combined_gpu_hours_observed"] + report["gpu_hours_upper_observed"], "GPU_ACCOUNT")
    close(report["combined_e4_e5_wall_seconds_observed"], e4_report["combined_wall_seconds_observed"] + report["formal_wall_seconds"], "WALL_ACCOUNT")
    require(report["combined_e4_e5_gpu_hours_observed"] <= 136 and report["combined_e4_e5_wall_seconds_observed"] <= 132 * 3600 and
            report["retained_bytes_before_report"] + e4_report["retained_bytes_before_report"] <= 56 * 1024 ** 3, "RESOURCE_CAP")
    require(report["deferred"] == {"bootstrap": "ENDPOINT-E6", "holm_28_tests": "ENDPOINT-E6",
        "c4b_claim_decisions": "ENDPOINT-E6", "path_ig_input_fidelity_stability": "NOT_RUN_NOT_ALLOWED"}, "DEFERRED")
    require(report["next_authorization"] == terminal["next_authorization"] == NEXT and
            report["claim_scope"] == terminal["claim_scope"] == "EXECUTION_PASS_PER_IMAGE_C4B_INPUTS_NO_BOOTSTRAP_HOLM_OR_CLAIM_DECISION", "BOUNDARY")
    return {"status": PASS_STATUS, "image_model_tasks": 800, "curves": 115200,
            "random_curves": 96000, "raw_map_tasks": 32, "next_authorization": NEXT}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
