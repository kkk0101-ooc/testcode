#!/usr/bin/env python3
"""Execute the published CPU-only Endpoint M4 external-review analysis."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_m4_review_r1.json"
M4 = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
M1 = REPO / "reports/nns_vit_endpoint_m1_r1_20260918/per_image"
E3 = REPO / "reports/nns_vit_endpoint_e3_closure_r1_20260918"
M3 = REPO / "reports/nns_vit_endpoint_m3_sup_r2_20260919"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_m4_review import (  # noqa: E402
    bootstrap_median, entropy, exact_sign, holm, partial_spearman,
    patch_distribution, quantiles, spearman, total_variation,
)

SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_m4_review_r1.json",
    "docs/endpoint_m4_review_protocol_20260919.md",
    "src/vitnns/endpoint_m4_review.py",
    "scripts/run_endpoint_m4_review.py",
    "tests/test_endpoint_m4_review.py",
]
PHASES = {"early": range(4), "middle": range(4, 8), "late": range(8, 12), "all": range(12)}


def load(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def remote(ref):
    values = subprocess.check_output(["git", "ls-remote", "origin", ref], cwd=REPO, text=True).split()
    return values[0] if len(values) == 2 else ""


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("EMPTY_CSV:" + path.name)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def summarize(values, namespace, bootstrap=True):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    result = {"n": len(values), **quantiles(values)}
    result["ci95"] = bootstrap_median(values, namespace) if bootstrap else None
    return result


def startup(spec):
    if any(name in sys.modules for name in ("torch", "timm")):
        raise ValueError("CPU_ONLY_IMPORTS")
    if os.environ["CUDA_VISIBLE_DEVICES"] != "":
        raise ValueError("CUDA_VISIBLE_DEVICES")
    for relative, expected in spec["source_bindings"].items():
        if sha(REPO / relative) != expected:
            raise ValueError("SOURCE_BINDING:" + relative)
    completed = subprocess.run(
        [sys.executable, "scripts/seal_endpoint_m4.py", "--verify"], cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1"),
        text=True, capture_output=True,
    )
    if completed.returncode:
        raise ValueError("M4_REVALIDATION:" + completed.stderr)
    head = git("rev-parse", "HEAD"); branch = git("symbolic-ref", "--short", "HEAD")
    remote_head = remote("refs/heads/" + branch)
    if head != git("rev-parse", "@{upstream}") or head != remote_head:
        raise ValueError("PUBLISHED_PRODUCER_REQUIRED")
    for relative in SOURCE_FILES:
        local = sha(REPO / relative)
        published = hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO)).hexdigest()
        if local != published:
            raise ValueError("PUBLISHED_SOURCE:" + relative)
    return head, completed.stdout.strip()


def m4_results():
    tasks = load(M4 / "task_manifest.json")["tasks"]
    output = []
    for task in tasks:
        task_id = task["task_id"]
        paths = sorted((M4 / "results/attempts" / task_id).glob("attempt-*.json"))
        if len(paths) != 1:
            raise ValueError("RESULT_COUNT:" + task_id)
        envelope = load(paths[0])
        unsigned = dict(envelope); observed_digest = unsigned.pop("canonical_digest", None)
        if canonical_digest(unsigned) != observed_digest:
            raise ValueError("RESULT_DIGEST:" + task_id)
        if envelope["task_id"] != task_id or envelope["result"]["sample_id"] != task["payload"]["sample_id"]:
            raise ValueError("RESULT_IDENTITY:" + task_id)
        output.append((task_id, envelope["result"]))
    return output


def m1_head_controls(task_id, model_id):
    value = load(M1 / f"{task_id}.json.gz")
    rows = [row for row in value["component_rows"]["normalized_zero"] if row["operation"] == "head"]
    expected = 144 if model_id == "vit_b16" else 72
    if len(rows) != expected:
        raise ValueError("M1_HEAD_SCOPE:" + task_id)
    return {(row["block"], row["head"]): (row["output_norm"], row["gradient_norm"]) for row in rows}


def jvp_analysis(results):
    image_rows = []
    for task_id, result in results:
        for block in range(12):
            heads = [row for row in result["attention"] if row["block"] == block]
            metrics = {"relative_error": [], "absolute_error": [], "full_response_denominator": [], "routing_numerator": []}
            for row in heads:
                full = float(np.sqrt(sum(direction["full_l2"] ** 2 for direction in row["direction_records"])))
                routing = float(np.sqrt(sum(direction["routing_l2"] ** 2 for direction in row["direction_records"])))
                metrics["relative_error"].append(row["frozen_attention_jvp_error"])
                metrics["absolute_error"].append(row["frozen_attention_jvp_absolute_error"])
                metrics["full_response_denominator"].append(full)
                metrics["routing_numerator"].append(routing)
            for metric, values in metrics.items():
                image_rows.append({"task_id": task_id, "model_id": result["model_id"],
                                   "sample_id": result["sample_id"], "block": block,
                                   "metric": metric, "image_head_median": float(np.median(values))})
    summary = []
    for model in ("vit_b16", "deit_s16"):
        for block in range(12):
            for metric in ("relative_error", "absolute_error", "full_response_denominator", "routing_numerator"):
                values = [row["image_head_median"] for row in image_rows
                          if row["model_id"] == model and row["block"] == block and row["metric"] == metric]
                summary.append({"model_id": model, "block": block, "metric": metric,
                                **summarize(values, f"M4R1|JVP|{model}|{block}|{metric}")})
    return image_rows, summary


def saturation_analysis(results):
    image_rows = []
    for task_id, result in results:
        controls = m1_head_controls(task_id, result["model_id"])
        for block in range(12):
            heads = sorted((row for row in result["attention"] if row["block"] == block), key=lambda row: row["head"])
            output_norm = [controls[(block, row["head"])][0] for row in heads]
            cotangent_norm = [controls[(block, row["head"])][1] for row in heads]
            for saturation in ("mean_query_saturation", "cls_query_saturation"):
                x = [row[saturation] for row in heads]
                for outcome in ("routing_share", "frozen_attention_endpoint_error", "frozen_attention_jvp_error"):
                    y = [row[outcome] for row in heads]
                    estimates = {
                        "unadjusted": spearman(x, y),
                        "adjust_output_norm": partial_spearman(x, y, [output_norm]),
                        "adjust_output_and_cotangent_norm": partial_spearman(x, y, [output_norm, cotangent_norm]),
                    }
                    for adjustment, estimate in estimates.items():
                        image_rows.append({"task_id": task_id, "model_id": result["model_id"],
                                           "sample_id": result["sample_id"], "block": block,
                                           "saturation": saturation, "outcome": outcome,
                                           "adjustment": adjustment, "rho": estimate})
    phase_rows = []
    identities = sorted({(row["task_id"], row["model_id"], row["sample_id"]) for row in image_rows})
    for task_id, model, sample in identities:
        task = [row for row in image_rows if row["task_id"] == task_id]
        for phase, blocks in PHASES.items():
            for saturation in ("mean_query_saturation", "cls_query_saturation"):
                for outcome in ("routing_share", "frozen_attention_endpoint_error", "frozen_attention_jvp_error"):
                    for adjustment in ("unadjusted", "adjust_output_norm", "adjust_output_and_cotangent_norm"):
                        values = [row["rho"] for row in task if row["block"] in blocks and row["saturation"] == saturation
                                  and row["outcome"] == outcome and row["adjustment"] == adjustment]
                        finite = [value for value in values if value is not None]
                        phase_rows.append({"task_id": task_id, "model_id": model, "sample_id": sample,
                                           "phase": phase, "saturation": saturation, "outcome": outcome,
                                           "adjustment": adjustment,
                                           "rho": None if not finite else float(np.median(finite))})
    summary = []
    for key in sorted({(row["model_id"], row["phase"], row["saturation"], row["outcome"], row["adjustment"])
                       for row in phase_rows}):
        values = [row["rho"] for row in phase_rows if (row["model_id"], row["phase"], row["saturation"], row["outcome"], row["adjustment"]) == key]
        summary.append({"model_id": key[0], "phase": key[1], "saturation": key[2],
                        "outcome": key[3], "adjustment": key[4],
                        **summarize(values, "M4R1|SAT|" + "|".join(key), bootstrap=key[1] == "all")})
    tests = []
    for model in ("vit_b16", "deit_s16"):
        family = []
        for saturation in ("mean_query_saturation", "cls_query_saturation"):
            for outcome in ("routing_share", "frozen_attention_endpoint_error", "frozen_attention_jvp_error"):
                for adjustment in ("unadjusted", "adjust_output_norm", "adjust_output_and_cotangent_norm"):
                    values = [row["rho"] for row in phase_rows if row["model_id"] == model and row["phase"] == "all"
                              and row["saturation"] == saturation and row["outcome"] == outcome and row["adjustment"] == adjustment]
                    family.append({"test_id": f"{saturation}|{outcome}|{adjustment}", "model_id": model,
                                   "median": float(np.median([value for value in values if value is not None])),
                                   **exact_sign(values)})
        tests.extend(holm(family))
    return image_rows, phase_rows, summary, tests


def residual_spatial_analysis(results):
    image_rows = []; tests = []
    for task_id, result in results:
        for residual_type in ("attention", "mlp"):
            rows = sorted((row for row in result["residual"] if row["residual_type"] == residual_type), key=lambda row: row["block"])
            early = float(np.median([row["output_identity_patch_tv"] for row in rows[:4]]))
            late = float(np.median([row["output_identity_patch_tv"] for row in rows[8:]]))
            image_rows.append({"task_id": task_id, "model_id": result["model_id"], "sample_id": result["sample_id"],
                               "residual_type": residual_type, "late_minus_early_spatial_tv": late - early,
                               "branch_share_spatial_tv_spearman": spearman(
                                   [row["branch_share"] for row in rows], [row["output_identity_patch_tv"] for row in rows])})
    family = []
    for model in ("vit_b16", "deit_s16"):
        for residual_type in ("attention", "mlp"):
            values = [row["late_minus_early_spatial_tv"] for row in image_rows
                      if row["model_id"] == model and row["residual_type"] == residual_type]
            family.append({"test_id": f"{model}|{residual_type}|late_minus_early_spatial_tv",
                           "model_id": model, "residual_type": residual_type,
                           "median": float(np.median(values)),
                           "ci95": bootstrap_median(values, f"M4R1|TV|{model}|{residual_type}"),
                           **exact_sign(values)})
    tests = holm(family)
    summary = []
    for model in ("vit_b16", "deit_s16"):
        for residual_type in ("attention", "mlp"):
            part = [row for row in image_rows if row["model_id"] == model and row["residual_type"] == residual_type]
            for metric in ("late_minus_early_spatial_tv", "branch_share_spatial_tv_spearman"):
                summary.append({"model_id": model, "residual_type": residual_type, "metric": metric,
                                **summarize([row[metric] for row in part], f"M4R1|RES|{model}|{residual_type}|{metric}")})
    return image_rows, summary, tests


def bridge_analysis():
    rows = []
    for model in ("vit_b16", "deit_s16"):
        for order in range(16):
            task_id = f"{model}__{order:03d}"
            envelope = load(next((E3 / "formal_results/attempts" / task_id).glob("attempt-*.json")))
            unsigned = dict(envelope); observed_digest = unsigned.pop("canonical_digest", None)
            if canonical_digest(unsigned) != observed_digest:
                raise ValueError("E3_RESULT_DIGEST:" + task_id)
            result = envelope["result"]
            for artifact in result["artifacts"]:
                array = np.load(REPO / artifact["path"], mmap_mode="r")
                indexes = {name: artifact["component_order"].index(name) for name in ("attention", "routing", "value")}
                distributions = {name: patch_distribution(array[index]) for name, index in indexes.items()}
                full_entropy = entropy(distributions["attention"]); value_entropy = entropy(distributions["value"])
                rows.append({"model_id": model, "task_id": task_id, "sample_id": result["sample_id"],
                             "block": artifact["block"], "full_entropy": full_entropy,
                             "value_entropy": value_entropy,
                             "routing_entropy": entropy(distributions["routing"]),
                             "full_minus_value_entropy": None if full_entropy is None or value_entropy is None else full_entropy - value_entropy,
                             "full_value_patch_tv": total_variation(distributions["attention"], distributions["value"])})
    summary = []
    for model in ("vit_b16", "deit_s16"):
        for block in range(12):
            part = [row for row in rows if row["model_id"] == model and row["block"] == block]
            for metric in ("full_entropy", "value_entropy", "routing_entropy", "full_minus_value_entropy", "full_value_patch_tv"):
                summary.append({"model_id": model, "block": block, "metric": metric,
                                **summarize([row[metric] for row in part], f"M4R1|BRIDGE|{model}|{block}|{metric}")})
    m3_rows = []
    with (M3 / "block_summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["metric"] in ("entropy", "locality", "distance"):
                m3_rows.append({"model_id": row["model"], "block": int(row["block"]),
                                "metric": row["metric"], "median": float(row["median"])})
    trajectory = []
    for model in ("vit_b16", "deit_s16"):
        for m3_metric in ("entropy", "locality", "distance"):
            left = [next(row["median"] for row in m3_rows if row["model_id"] == model and row["block"] == block and row["metric"] == m3_metric) for block in range(12)]
            for bridge_metric in ("full_minus_value_entropy", "full_value_patch_tv"):
                right = [next(row["median"] for row in summary if row["model_id"] == model and row["block"] == block and row["metric"] == bridge_metric) for block in range(12)]
                trajectory.append({"model_id": model, "m3_metric": m3_metric, "m4_bridge_metric": bridge_metric,
                                   "block_trajectory_spearman": spearman(left, right),
                                   "unit": "12_model_level_block_medians", "causal_interpretation": false})
    return rows, summary, m3_rows, trajectory


def numeric_locations(results):
    records = []
    for task_id, result in results:
        base = {"task_id": task_id, "model_id": result["model_id"], "sample_id": result["sample_id"]}
        for row in result["attention"]:
            records.extend((
                {**base, "metric": "qkv_closure_max_abs", "block": row["block"], "head": row["head"], "residual_type": "", "value": row["qkv_native_closure"]["max_abs"]},
                {**base, "metric": "frozen_identity_max_abs", "block": row["block"], "head": row["head"], "residual_type": "", "value": row["frozen_attention_identity"]["max_abs"]},
                {**base, "metric": "explicit_jvp_relative_error", "block": row["block"], "head": row["head"], "residual_type": "", "value": row["explicit_jvp_relative_error"] or 0},
            ))
        for row in result["residual"]:
            records.append({**base, "metric": "residual_closure_max_abs", "block": row["block"], "head": "",
                            "residual_type": row["residual_type"], "value": row["closure"]["max_abs"]})
    return [max((row for row in records if row["metric"] == metric), key=lambda row: row["value"])
            for metric in ("qkv_closure_max_abs", "frozen_identity_max_abs", "explicit_jvp_relative_error", "residual_closure_max_abs")]


def render(root, jvp_summary, sat_summary, residual_summary, bridge_summary, m3_rows):
    figure_root = root / "figures"; paper_root = REPO / "papers/nns_vit_endpoint/figures/m4_review_r1"
    figure_root.mkdir(); paper_root.mkdir(parents=True, exist_ok=False)
    colors = {"vit_b16": "#355c9a", "deit_s16": "#d07a2d"}; figures = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for model in colors:
        for axis, metric, title in zip(axes, ("relative_error", "absolute_error", "full_response_denominator"),
                                       ("Normalized RMS discrepancy", "Absolute routing discrepancy", "Full-Jacobian response norm")):
            part = [next(row for row in jvp_summary if row["model_id"] == model and row["block"] == block and row["metric"] == metric) for block in range(12)]
            axis.plot(range(12), [row["median"] for row in part], marker="o", color=colors[model], label=model)
            axis.fill_between(range(12), [row["q25"] for row in part], [row["q75"] for row in part], color=colors[model], alpha=.14)
            axis.set_title(title); axis.set_xlabel("Block"); axis.grid(alpha=.2)
    axes[0].legend(frameon=False); axes[0].set_ylabel("Image-level median across heads"); fig.tight_layout(); figures.append(("m4_review_jvp_diagnostics", fig))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for model in colors:
        for axis, saturation, title in zip(axes, ("mean_query_saturation", "cls_query_saturation"), ("Mean-query saturation", "CLS-query saturation")):
            for adjustment, style in (("unadjusted", "-"), ("adjust_output_norm", "--"), ("adjust_output_and_cotangent_norm", ":")):
                part = [next(row for row in sat_summary if row["model_id"] == model and row["phase"] == phase and row["saturation"] == saturation and row["outcome"] == "frozen_attention_endpoint_error" and row["adjustment"] == adjustment) for phase in ("early", "middle", "late")]
                axis.plot(range(3), [row["median"] for row in part], style, marker="o", color=colors[model], label=f"{model} {adjustment}")
            axis.set_xticks(range(3), ("early", "middle", "late")); axis.set_title(title); axis.set_ylabel("Within-layer head rank correlation"); axis.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=6); fig.tight_layout(); figures.append(("m4_review_saturation_robustness", fig))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for model in colors:
        tests = [row for row in residual_summary if row["model_id"] == model and row["metric"] == "late_minus_early_spatial_tv"]
        axes[0].bar([f"{model}\n{row['residual_type']}" for row in tests], [row["median"] for row in tests], alpha=.75, color=colors[model])
        relation = [row for row in residual_summary if row["model_id"] == model and row["metric"] == "branch_share_spatial_tv_spearman"]
        axes[1].bar([f"{model}\n{row['residual_type']}" for row in relation], [row["median"] for row in relation], alpha=.75, color=colors[model])
    axes[0].axhline(0, color="black", lw=.8); axes[0].set_title("Late minus early spatial TV")
    axes[1].axhline(0, color="black", lw=.8); axes[1].set_title("Branch-share vs spatial-TV rho")
    for axis in axes: axis.tick_params(axis="x", labelsize=7); axis.grid(axis="y", alpha=.2)
    fig.tight_layout(); figures.append(("m4_review_residual_spatial", fig))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for model in colors:
        entropy_values = [next(row["median"] for row in bridge_summary if row["model_id"] == model and row["block"] == block and row["metric"] == "full_minus_value_entropy") for block in range(12)]
        tv_values = [next(row["median"] for row in bridge_summary if row["model_id"] == model and row["block"] == block and row["metric"] == "full_value_patch_tv") for block in range(12)]
        axes[0].plot(range(12), entropy_values, marker="o", color=colors[model], label=model)
        axes[1].plot(range(12), tv_values, marker="o", color=colors[model], label=model)
    axes[0].axhline(0, color="black", lw=.8); axes[0].set_title("H(full) - H(value)")
    axes[1].set_title("Full-value patch TV")
    for axis in axes: axis.set_xlabel("Block"); axis.grid(alpha=.2)
    axes[0].legend(frameon=False); fig.tight_layout(); figures.append(("m4_review_m3_m4_bridge", fig))
    manifest = []
    for figure_id, figure in figures:
        for extension in ("png", "pdf"):
            path = figure_root / f"{figure_id}.{extension}"; figure.savefig(path, dpi=180 if extension == "png" else None, bbox_inches="tight")
            destination = paper_root / path.name; shutil.copyfile(path, destination)
            manifest.append({"figure_id": figure_id, "format": extension,
                             "report_path": path.relative_to(REPO).as_posix(), "paper_path": destination.relative_to(REPO).as_posix(),
                             "sha256": sha(path), "bytes": path.stat().st_size})
        plt.close(figure)
    return manifest


def main():
    spec = load(SPEC_PATH); root = REPO / spec["output_root"]
    if root.exists():
        raise ValueError("FRESH_ROOT_REQUIRED")
    head, predecessor = startup(spec); root.mkdir()
    write(root / "freeze.json", {"stage": spec["stage"], "status": spec["status"], "time_ns": time.time_ns(),
                                  "producer_commit": head, "sources": {path: sha(REPO / path) for path in SOURCE_FILES},
                                  "source_bindings": spec["source_bindings"], "predecessor_revalidation": predecessor})
    results = m4_results()
    jvp_rows, jvp_summary = jvp_analysis(results)
    sat_rows, sat_phase, sat_summary, sat_tests = saturation_analysis(results)
    residual_rows, residual_summary, residual_tests = residual_spatial_analysis(results)
    bridge_rows, bridge_summary, m3_rows, trajectory = bridge_analysis()
    numeric = numeric_locations(results)
    write_csv(root / "jvp_image_block_values.csv", jvp_rows); write_csv(root / "jvp_block_summary.csv", jvp_summary)
    write_csv(root / "saturation_image_block_values.csv", sat_rows); write_csv(root / "saturation_image_phase_values.csv", sat_phase)
    write_csv(root / "saturation_sensitivity_summary.csv", sat_summary); write(root / "saturation_sensitivity_tests.json", {"tests": sat_tests})
    write_csv(root / "residual_spatial_image_values.csv", residual_rows); write_csv(root / "residual_spatial_summary.csv", residual_summary)
    write(root / "residual_spatial_tests.json", {"tests": residual_tests})
    write_csv(root / "bridge_raw_map_values.csv", bridge_rows); write_csv(root / "bridge_block_summary.csv", bridge_summary)
    write_csv(root / "m3_block_source.csv", m3_rows); write_csv(root / "m3_m4_trajectory_relations.csv", trajectory)
    write(root / "numeric_error_locations.json", {"records": numeric})
    manifest = render(root, jvp_summary, sat_summary, residual_summary, bridge_summary, m3_rows)
    write(root / "figure_manifest.json", {"figures": manifest})
    receipt = {"status": "PASS_POST_HOC_REVIEW_ANALYSIS", "stage": spec["stage"], "mode": "CPU_ONLY",
               "m4_tasks": len(results), "jvp_image_block_rows": len(jvp_rows), "saturation_block_rows": len(sat_rows),
               "saturation_phase_rows": len(sat_phase), "residual_image_rows": len(residual_rows),
               "raw_map_tasks": len({row["task_id"] for row in bridge_rows}), "raw_map_records": len(bridge_rows),
               "new_primary_M4_tests": 0, "post_hoc_sensitivity_tests": len(sat_tests) + len(residual_tests),
               "torch_or_timm_imported": any(name in sys.modules for name in ("torch", "timm")),
               "jpeg_or_model_reads": 0, "next_authorization": spec["next_authorization"]}
    write(root / "analysis_receipt.json", receipt); print(json.dumps(receipt))


if __name__ == "__main__":
    main()
