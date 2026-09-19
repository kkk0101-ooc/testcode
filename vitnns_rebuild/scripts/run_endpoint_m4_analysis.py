#!/usr/bin/env python3
"""CPU-only source-recomputing analysis for Endpoint M4."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
PAPER = REPO / "papers/nns_vit_endpoint"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_ep1_queue import audit_coverage  # noqa: E402
from vitnns.endpoint_m4_statistics import (  # noqa: E402
    ATTENTION_METRICS, RESIDUAL_METRICS, bootstrap_median,
    exact_two_sided_sign, holm, image_values, median, summarize,
)


def load(path: Path):
    return json.loads(path.read_text())


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def accepted_results() -> list[dict]:
    coverage = audit_coverage(ROOT / "queue.sqlite3", ROOT / "results")
    if coverage["status"] != "PASS" or coverage["completed_count"] != 800:
        raise ValueError("QUEUE_COVERAGE")
    output = []
    for task_id, attempt in coverage["accepted_attempts"]:
        path = ROOT / "results/attempts" / task_id / f"attempt-{attempt:02d}.json"
        output.append(load(path)["result"])
    return sorted(output, key=lambda row: (row["model_id"], row["split_order"]))


def validate_result(row: dict) -> dict:
    heads = 12 if row["model_id"] == "vit_b16" else 6
    if len(row["attention"]) != 12 * heads or len(row["residual"]) != 24 or len(row["block_validation"]) != 12:
        raise ValueError("RESULT_COVERAGE:" + row["sample_id"])
    if len(row["attention_detach_validation"]) != (12 if row["split_order"] < 16 else 0):
        raise ValueError("DETACH_COVERAGE")
    if row["e3_crosscheck"]["status"] != "PASS" or row["e3_crosscheck"]["scalar_checks"] != 288:
        raise ValueError("E3_CROSSCHECK")
    if any(not item["qkv_native_closure"]["pass"] or not item["frozen_attention_identity"]["pass"]
           or item["directions"] != 8 for item in row["attention"]):
        raise ValueError("ATTENTION_VALIDATION")
    if any(not item["closure"]["pass"] or not item["negative_control"]["passed"] for item in row["residual"]):
        raise ValueError("RESIDUAL_VALIDATION")
    if any(item["explicit_jvp_relative_error"] is not None and item["explicit_jvp_relative_error"] > 1e-4
           for item in row["attention"]):
        raise ValueError("JVP_TOLERANCE")
    if any(not item["attention_detach_value_gradient"]["pass"] for item in row["attention_detach_validation"]):
        raise ValueError("DETACH_TOLERANCE")
    identity = row["worker_identity"]
    if identity["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8" or not identity["deterministic_algorithms"] or identity["cuda_matmul_allow_tf32"] or identity["cudnn_allow_tf32"] or identity["autocast"]:
        raise ValueError("WORKER_NUMERIC_IDENTITY")
    return image_values(row)


def flatten_values(results: list[dict], derived: list[dict]) -> list[dict]:
    rows = []
    for result, value in zip(results, derived):
        base = {"model_id": result["model_id"], "sample_id": result["sample_id"],
                "split_order": result["split_order"]}
        for (block, metric), observed in value["attention"].items():
            rows.append({**base, "domain": "attention", "group": "heads", "block": block, "metric": metric, "value": observed})
        for (block, metric), observed in value["attention_relations"].items():
            rows.append({**base, "domain": "attention_relation", "group": "heads", "block": block, "metric": metric, "value": observed})
        for (kind, block, metric), observed in value["residual"].items():
            rows.append({**base, "domain": "residual", "group": kind, "block": block, "metric": metric, "value": observed})
        for metric, observed in value["contrasts"].items():
            rows.append({**base, "domain": "contrast", "group": "image", "block": "", "metric": metric, "value": observed})
    return rows


def block_summaries(results: list[dict], derived: list[dict]):
    attention, residual, heads = [], [], []
    for model in ("vit_b16", "deit_s16"):
        indexes = [index for index, row in enumerate(results) if row["model_id"] == model]
        for block in range(12):
            for metric in ATTENTION_METRICS + ("saturation_routing", "saturation_error"):
                source = "attention_relations" if metric.startswith("saturation_") else "attention"
                values = [derived[index][source][(block, metric)] for index in indexes]
                attention.append({"model_id": model, "block": block, "metric": metric,
                                  **summarize(values, f"M4|attention|{model}|{block}|{metric}")})
            for kind in ("attention", "mlp"):
                for metric in RESIDUAL_METRICS:
                    values = [derived[index]["residual"][(kind, block, metric)] for index in indexes]
                    residual.append({"model_id": model, "residual_type": kind, "block": block,
                                     "metric": metric, **summarize(values, f"M4|residual|{model}|{kind}|{block}|{metric}")})
        head_count = 12 if model == "vit_b16" else 6
        for block in range(12):
            for head in range(head_count):
                for metric in ("routing_share", "frozen_attention_endpoint_error",
                               "frozen_attention_jvp_error", "mean_query_saturation",
                               "routing_value_path_cancellation"):
                    values = [next(item for item in results[index]["attention"]
                                   if item["block"] == block and item["head"] == head)[metric] for index in indexes]
                    heads.append({"model_id": model, "block": block, "head": head, "metric": metric,
                                  **summarize(values, f"M4|head|{model}|{block}|{head}|{metric}")})
    return attention, residual, heads


def infer(results: list[dict], derived: list[dict]):
    tests = []
    for model in ("vit_b16", "deit_s16"):
        indexes = [index for index, row in enumerate(results) if row["model_id"] == model]
        family = []
        mapping = {
            "late_minus_early_routing_share": "attention_routing_share_late_minus_early",
            "late_minus_early_frozen_attention_endpoint_error": "attention_frozen_attention_endpoint_error_late_minus_early",
            "saturation_routing_relation": "saturation_routing",
            "saturation_error_relation": "saturation_error",
        }
        for test_id, metric in mapping.items():
            values = [derived[index]["contrasts"][metric] for index in indexes]
            family.append({"test_id": test_id, "model_id": model, "domain": "attention",
                           "metric": metric, "median": median(values),
                           "ci95": bootstrap_median(values, f"M4|test|{model}|{test_id}"),
                           **exact_two_sided_sign(values)})
        tests.extend(holm(family))
        for kind in ("attention", "mlp"):
            family = []
            for short in ("identity_share", "path_cancellation"):
                test_id = f"late_minus_early_{short}"
                key = f"{kind}_{short}_late_minus_early"
                values = [derived[index]["contrasts"][key] for index in indexes]
                family.append({"test_id": test_id, "model_id": model, "domain": "residual",
                               "residual_type": kind, "metric": key, "median": median(values),
                               "ci95": bootstrap_median(values, f"M4|test|{model}|{kind}|{test_id}"),
                               **exact_two_sided_sign(values)})
            tests.extend(holm(family))
    for row in tests:
        row["decision"] = "SUPPORTED_DIFFERENCE" if row["reject"] else "NOT_SUPPORTED"
        row["direction"] = "POSITIVE" if row["median"] is not None and row["median"] > 0 else "NEGATIVE" if row["median"] is not None and row["median"] < 0 else "ZERO_OR_NA"
    return tests


def lookup(rows: list[dict], model: str, block: int, metric: str, kind=None):
    return next(row for row in rows if row["model_id"] == model and row["block"] == block
                and row["metric"] == metric and (kind is None or row["residual_type"] == kind))


def render(attention: list[dict], residual: list[dict]) -> list[dict]:
    out = ROOT / "figures"; paper = PAPER / "figures/m4"
    out.mkdir(); paper.mkdir(parents=True, exist_ok=False)
    colors = {"vit_b16": "#355c9a", "deit_s16": "#d07a2d"}
    figures = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for model in colors:
        for axis, metric, title in zip(axes, ("routing_share", "frozen_attention_endpoint_error", "frozen_attention_jvp_error"), ("Routing share", "Frozen-attention Endpoint error", "Frozen-attention JVP error")):
            values = [lookup(attention, model, block, metric) for block in range(12)]
            y = [row["median"] for row in values]; low = [row["ci95"][0] for row in values]; high = [row["ci95"][1] for row in values]
            axis.plot(range(12), y, marker="o", color=colors[model], label=model)
            axis.fill_between(range(12), low, high, color=colors[model], alpha=.15)
            axis.set_title(title); axis.set_xlabel("Block"); axis.grid(alpha=.2)
    axes[0].set_ylabel("Image-level median across heads"); axes[0].legend(frameon=False)
    fig.tight_layout(); figures.append(("m4_attention_depth", fig))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for model in colors:
        x = [lookup(attention, model, block, "mean_query_saturation")["median"] for block in range(12)]
        for axis, metric, title in zip(axes, ("routing_share", "frozen_attention_endpoint_error"), ("Saturation and routing", "Saturation and FA error")):
            y = [lookup(attention, model, block, metric)["median"] for block in range(12)]
            axis.scatter(x, y, color=colors[model], label=model)
            axis.set_xlabel("Mean query saturation"); axis.set_ylabel(metric); axis.set_title(title); axis.grid(alpha=.2)
    axes[0].legend(frameon=False); fig.tight_layout(); figures.append(("m4_attention_saturation", fig))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    styles = {"attention": "-", "mlp": "--"}
    for model in colors:
        for kind in styles:
            label = f"{model} {kind}"
            for axis, metric, title in zip(axes, ("identity_share", "path_cancellation", "output_identity_patch_tv"), ("Identity share", "Path cancellation", "Output-identity spatial TV")):
                values = [lookup(residual, model, block, metric, kind)["median"] for block in range(12)]
                axis.plot(range(12), values, styles[kind], marker="o", markersize=3, color=colors[model], label=label)
                axis.set_title(title); axis.set_xlabel("Block"); axis.grid(alpha=.2)
    axes[0].set_ylabel("Image-level metric"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); figures.append(("m4_residual_depth", fig))
    manifest = []
    for figure_id, figure in figures:
        for extension in ("png", "pdf"):
            path = out / f"{figure_id}.{extension}"
            figure.savefig(path, dpi=180 if extension == "png" else None, bbox_inches="tight")
            destination = paper / path.name; shutil.copyfile(path, destination)
            manifest.append({"figure_id": figure_id, "format": extension,
                             "report_path": path.relative_to(REPO).as_posix(),
                             "paper_path": destination.relative_to(REPO).as_posix()})
        plt.close(figure)
    return manifest


def main() -> None:
    if any(name in sys.modules for name in ("torch", "timm", "PIL")):
        raise ValueError("CPU_ONLY_IMPORT_BOUNDARY")
    results = accepted_results(); derived = [validate_result(row) for row in results]
    if len(results) != 800 or {row["model_id"] for row in results} != {"vit_b16", "deit_s16"}:
        raise ValueError("RESULT_SCOPE")
    identities = defaultdict(set)
    for row in results:
        identities[row["model_id"]].add((row["sample_id"], row["image_sha256"]))
    if len(identities["vit_b16"]) != 400 or identities["vit_b16"] != identities["deit_s16"]:
        raise ValueError("SHARED_COHORT_IDENTITY")
    flat = flatten_values(results, derived)
    attention, residual, heads = block_summaries(results, derived)
    tests = infer(results, derived)
    fields = ["model_id", "sample_id", "split_order", "domain", "group", "block", "metric", "value"]
    write_csv(ROOT / "image_values.csv", fields, flat)
    write_csv(ROOT / "attention_block_summary.csv", list(attention[0]), attention)
    write_csv(ROOT / "residual_block_summary.csv", list(residual[0]), residual)
    write_csv(ROOT / "attention_head_summary.csv", list(heads[0]), heads)
    write_new(ROOT / "m4_tests.json", {"tests": tests, "test_count": len(tests)})
    figure_manifest = render(attention, residual)
    write_new(ROOT / "figure_manifest.json", {"figures": figure_manifest})
    write_new(ROOT / "analysis_receipt.json", {
        "status": "PASS", "mode": "CPU_ONLY", "image_model_tasks": 800,
        "shared_images": 400, "attention_head_records": sum(len(row["attention"]) for row in results),
        "residual_records": sum(len(row["residual"]) for row in results),
        "jvp_direction_head_records": sum(len(item["direction_records"]) for row in results for item in row["attention"]),
        "e3_scalar_crosschecks": sum(row["e3_crosscheck"]["scalar_checks"] for row in results),
        "detach_block_checks": sum(len(row["attention_detach_validation"]) for row in results),
        "primary_tests": len(tests), "image_value_rows": len(flat),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "prohibited_imports_absent": True,
    })
    print(json.dumps(load(ROOT / "analysis_receipt.json")))


if __name__ == "__main__":
    main()
