#!/usr/bin/env python3
"""CPU-only image-level analysis for revision-v3 Step-2 experiments."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_revision_v3_step2 import bootstrap_median, spearman

SPEC = REPO / "config/nns_vit_endpoint/revision_v3_step2.json"
ROOT = REPO / "reports/nns_vit_endpoint_revision_v3_step2_20260921"


def load(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def dump_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def result(task_id: str) -> dict:
    return load(ROOT / "results" / "attempts" / task_id / "attempt-01.json")["result"]


def summarize(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...], prefix: str) -> list[dict]:
    output = []
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    for identity, part in sorted(groups.items()):
        base = dict(zip(keys, identity))
        for metric in metrics:
            values = [float(row[metric]) for row in part if row[metric] is not None and math.isfinite(float(row[metric]))]
            output.append({
                **base, "metric": metric, "n_images": len(values),
                "median": None if not values else float(np.median(values)),
                "ci95_low": None if not values else bootstrap_median(values, prefix + "|" + "|".join(map(str, identity)) + "|" + metric)[0],
                "ci95_high": None if not values else bootstrap_median(values, prefix + "|" + "|".join(map(str, identity)) + "|" + metric)[1],
            })
    return output


def analyze_amplitude(tasks: list[dict]) -> tuple[list[dict], list[dict], dict]:
    image_rows = []
    for task in (task for task in tasks if task["payload"]["kind"] == "primary"):
        record = result(task["task_id"]); data = np.load(REPO / record["amplitude_path"])
        for reference in sorted(set(data["reference_id"].tolist())):
            for alpha in sorted(set(float(value) for value in data["alpha"])):
                mask = (data["reference_id"] == reference) & np.isclose(data["alpha"], alpha, atol=1e-7, rtol=0)
                effects = data["finite_effect"][mask].astype(np.float64)
                predictions = data["taylor_prediction"][mask].astype(np.float64)
                absolute = data["absolute_error"][mask].astype(np.float64)
                relative = data["relative_error"][mask].astype(np.float64)
                block_values = []
                for block in range(12):
                    local = mask & (data["block"] == block)
                    value = spearman(np.abs(data["taylor_prediction"][local]), np.abs(data["finite_effect"][local]))
                    if value is not None:
                        block_values.append(value)
                image_rows.append({
                    "model_id": record["model_id"], "sample_id": record["sample_id"],
                    "image_sha256": record["image_sha256"], "reference_id": reference, "alpha": alpha,
                    "median_absolute_finite_effect": float(np.median(np.abs(effects))),
                    "median_absolute_taylor_prediction": float(np.median(np.abs(predictions))),
                    "median_absolute_error": float(np.median(absolute)),
                    "median_relative_error": None if not np.isfinite(relative).any() else float(np.median(relative[np.isfinite(relative)])),
                    "mean_within_block_spearman": None if not block_values else float(np.mean(block_values)),
                    "valid_relative_components": int(np.isfinite(relative).sum()),
                    "valid_rank_blocks": len(block_values),
                })
    metrics = ("median_absolute_finite_effect", "median_absolute_taylor_prediction",
               "median_absolute_error", "median_relative_error", "mean_within_block_spearman")
    summary = summarize(image_rows, ("model_id", "reference_id", "alpha"), metrics, "amplitude")
    trend_rows = []
    for model in sorted({row["model_id"] for row in image_rows}):
        for reference in sorted({row["reference_id"] for row in image_rows}):
            part = [row for row in image_rows if row["model_id"] == model and row["reference_id"] == reference]
            by_image = defaultdict(dict)
            for row in part:
                by_image[row["sample_id"]][row["alpha"]] = row["median_absolute_error"]
            alphas = sorted({row["alpha"] for row in part})
            cohort = [float(np.median([row["median_absolute_error"] for row in part if row["alpha"] == alpha])) for alpha in alphas]
            monotonic = [all(values[b] >= values[a] for a, b in zip(alphas, alphas[1:])) for values in by_image.values()]
            trend_rows.append({
                "model_id": model, "reference_id": reference, "alphas": alphas,
                "cohort_median_absolute_errors": cohort,
                "cohort_median_monotonic_non_decreasing": all(b >= a for a, b in zip(cohort, cohort[1:])),
                "image_monotonic_count": int(sum(monotonic)), "image_count": len(monotonic),
                "image_monotonic_rate": float(np.mean(monotonic)),
            })
    decision = {
        "strong_boundary_supported": all(row["cohort_median_monotonic_non_decreasing"] for row in trend_rows),
        "rule": "all four model-reference cohort median absolute-error curves must be non-decreasing",
        "families": trend_rows,
    }
    return image_rows, summary, decision


def analyze_qkv(tasks: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    metrics = ("absolute_mass", "normalized_spatial_entropy", "normalized_pairwise_distance",
               "cancellation", "relative_scale")
    for task in (task for task in tasks if task["payload"]["kind"] == "primary"):
        record = result(task["task_id"]); data = np.load(REPO / record["qkv_path"])
        for index in range(len(data["block"])):
            row = {
                "model_id": record["model_id"], "sample_id": record["sample_id"],
                "image_sha256": record["image_sha256"], "block": int(data["block"][index]),
                "path": str(data["path"][index]),
            }
            row.update({metric: None if not np.isfinite(data[metric][index]) else float(data[metric][index]) for metric in metrics})
            rows.append(row)
    summary = summarize(rows, ("model_id", "block", "path"), metrics, "qkv-block")
    contrasts = []
    for model in sorted({row["model_id"] for row in rows}):
        for path in ("Q", "K", "V"):
            by_image = defaultdict(dict)
            for row in rows:
                if row["model_id"] == model and row["path"] == path:
                    by_image[row["sample_id"]][row["block"]] = row
            for metric in metrics:
                values = []
                for blocks in by_image.values():
                    if blocks[0][metric] is not None and blocks[11][metric] is not None:
                        values.append(blocks[11][metric] - blocks[0][metric])
                interval = bootstrap_median(values, f"qkv-contrast|{model}|{path}|{metric}")
                contrasts.append({
                    "model_id": model, "path": path, "metric": metric, "contrast": "block11_minus_block0",
                    "n_images": len(values), "median_difference": float(np.median(values)),
                    "ci95_low": interval[0], "ci95_high": interval[1],
                })
    for path in ("Q", "K", "V"):
        for metric in metrics:
            part = [row for row in contrasts if row["path"] == path and row["metric"] == metric]
            signs = [int(np.sign(row["median_difference"])) for row in part]
            for row in part:
                row["cross_model_direction_consistent"] = len(set(signs)) == 1 and signs[0] != 0
    return rows, summary, contrasts


def analyze_sampling(tasks: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    metrics = ("normalized_spatial_entropy", "own_token_locality_above_uniform", "input_dependency_distance")
    image_rows = []
    for task in (task for task in tasks if task["payload"]["kind"] == "sampling"):
        record = result(task["task_id"]); data = np.load(REPO / record["sampling_path"])
        for group_index in range(5):
            group = data["group_index"] == group_index
            for metric in metrics:
                values = data[metric].astype(np.float64)
                def median(mask):
                    valid = values[mask & np.isfinite(values)]
                    return None if not len(valid) else float(np.median(valid))
                early = median(group & (data["block"] <= 3)); late = median(group & (data["block"] >= 8))
                block0 = median(group & (data["block"] == 0)); block11 = median(group & (data["block"] == 11))
                image_rows.append({
                    "model_id": record["model_id"], "sample_id": record["sample_id"],
                    "image_sha256": record["image_sha256"], "group_id": f"sample_{group_index + 1}",
                    "metric": metric, "early": early, "late": late, "block0": block0, "block11": block11,
                    "late_minus_early": None if early is None or late is None else late - early,
                    "block11_minus_block0": None if block0 is None or block11 is None else block11 - block0,
                })
    summaries = []
    for model in sorted({row["model_id"] for row in image_rows}):
        for group_id in sorted({row["group_id"] for row in image_rows}):
            for source_metric in metrics:
                part = [row for row in image_rows if row["model_id"] == model and row["group_id"] == group_id and row["metric"] == source_metric]
                for contrast in ("late_minus_early", "block11_minus_block0"):
                    values = [row[contrast] for row in part if row[contrast] is not None]
                    interval = bootstrap_median(values, f"sampling|{model}|{group_id}|{source_metric}|{contrast}")
                    summaries.append({
                        "model_id": model, "group_id": group_id, "source_metric": source_metric,
                        "contrast": contrast, "n_images": len(values),
                        "median": float(np.median(values)), "ci95_low": interval[0], "ci95_high": interval[1],
                    })
    expected = load(SPEC)["independent_sampling"]["expected_directions"]
    variation = []
    for model in sorted({row["model_id"] for row in image_rows}):
        for metric in metrics:
            for contrast in ("late_minus_early", "block11_minus_block0"):
                medians = [
                    row["median"] for row in summaries
                    if row["model_id"] == model and row["source_metric"] == metric and row["contrast"] == contrast
                ]
                sign = 1 if expected[metric] == "positive" else -1
                consistent = [value * sign > 0 for value in medians]
                variation.append({
                    "model_id": model, "metric": metric, "contrast": contrast,
                    "group_medians": medians, "between_group_sd": float(np.std(medians, ddof=1)),
                    "between_group_min": min(medians), "between_group_max": max(medians),
                    "expected_direction": expected[metric], "direction_consistent_groups": int(sum(consistent)),
                    "group_count": len(medians), "direction_consistency_rate": float(np.mean(consistent)),
                })
    return image_rows, summaries, variation


def plot_results(amplitude_summary, qkv_summary, sampling_variation) -> None:
    figure_root = ROOT / "figures"; figure_root.mkdir(exist_ok=True)
    colors = {"vit_b16": "#2166ac", "deit_s16": "#b2182b"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model in colors:
        for reference, style in (("normalized_zero", "-"), ("blur_rgb", "--")):
            rows = [row for row in amplitude_summary if row["model_id"] == model and row["reference_id"] == reference]
            for metric, axis in (("median_absolute_error", axes[0]), ("mean_within_block_spearman", axes[1])):
                part = sorted((row for row in rows if row["metric"] == metric), key=lambda row: row["alpha"])
                axis.plot([row["alpha"] for row in part], [row["median"] for row in part], style, color=colors[model], marker="o", label=f"{model}, {reference}")
    axes[0].set(xlabel=r"amplitude $\alpha$", ylabel="median absolute Taylor error")
    axes[1].set(xlabel=r"amplitude $\alpha$", ylabel="mean within-block Spearman")
    axes[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(figure_root / "multi_amplitude.png", dpi=180); fig.savefig(figure_root / "multi_amplitude.pdf"); plt.close(fig)

    metrics = ("absolute_mass", "normalized_spatial_entropy", "normalized_pairwise_distance", "cancellation", "relative_scale")
    fig, axes = plt.subplots(len(metrics), 2, figsize=(11, 14), sharex=True)
    for column, model in enumerate(colors):
        for row_index, metric in enumerate(metrics):
            for path, color in (("Q", "#2166ac"), ("K", "#67a9cf"), ("V", "#b2182b")):
                part = sorted((row for row in qkv_summary if row["model_id"] == model and row["path"] == path and row["metric"] == metric), key=lambda row: row["block"])
                axes[row_index, column].plot([row["block"] for row in part], [row["median"] for row in part], marker="o", label=path, color=color)
            axes[row_index, column].set_ylabel(metric); axes[row_index, column].set_title(model if row_index == 0 else "")
    axes[0, 1].legend(); axes[-1, 0].set_xlabel("block"); axes[-1, 1].set_xlabel("block"); fig.tight_layout()
    fig.savefig(figure_root / "qkv_population.png", dpi=180); fig.savefig(figure_root / "qkv_population.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric in zip(axes, ("normalized_spatial_entropy", "own_token_locality_above_uniform", "input_dependency_distance")):
        for model, marker in (("vit_b16", "o"), ("deit_s16", "s")):
            row = next(row for row in sampling_variation if row["model_id"] == model and row["metric"] == metric and row["contrast"] == "late_minus_early")
            axis.plot(range(1, 6), row["group_medians"], marker=marker, color=colors[model], label=model)
        axis.axhline(0, color="#777777", linewidth=.8); axis.set(xlabel="independent sample group", ylabel="late minus early", title=metric)
    axes[-1].legend(); fig.tight_layout()
    fig.savefig(figure_root / "independent_sampling.png", dpi=180); fig.savefig(figure_root / "independent_sampling.pdf"); plt.close(fig)


def main() -> None:
    tasks = load(ROOT / "task_manifest.json")["tasks"]
    amplitude_images, amplitude_summary, amplitude_decision = analyze_amplitude(tasks)
    qkv_images, qkv_summary, qkv_contrasts = analyze_qkv(tasks)
    sampling_images, sampling_summary, sampling_variation = analyze_sampling(tasks)
    dump_csv(ROOT / "multi_amplitude_image_values.csv", amplitude_images)
    dump_csv(ROOT / "multi_amplitude_summary.csv", amplitude_summary)
    write(ROOT / "multi_amplitude_decision.json", amplitude_decision)
    dump_csv(ROOT / "qkv_image_values.csv", qkv_images); dump_csv(ROOT / "qkv_block_summary.csv", qkv_summary)
    dump_csv(ROOT / "qkv_depth_contrasts.csv", qkv_contrasts)
    dump_csv(ROOT / "independent_sampling_image_values.csv", sampling_images)
    dump_csv(ROOT / "independent_sampling_summary.csv", sampling_summary)
    write(ROOT / "independent_sampling_variation.json", {"rows": sampling_variation})
    plot_results(amplitude_summary, qkv_summary, sampling_variation)
    write(ROOT / "analysis_receipt.json", {
        "status": "PASS", "statistical_unit": "image",
        "amplitude_image_rows": len(amplitude_images), "amplitude_summary_rows": len(amplitude_summary),
        "qkv_image_rows": len(qkv_images), "qkv_summary_rows": len(qkv_summary),
        "qkv_contrast_rows": len(qkv_contrasts), "sampling_image_rows": len(sampling_images),
        "sampling_summary_rows": len(sampling_summary), "sampling_variation_rows": len(sampling_variation),
    })
    print(json.dumps(load(ROOT / "analysis_receipt.json")))


if __name__ == "__main__":
    main()
