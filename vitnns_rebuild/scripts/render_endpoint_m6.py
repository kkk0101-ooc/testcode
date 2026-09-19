#!/usr/bin/env python3
"""Render the frozen Endpoint M6 cases from raw selected arrays."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m6_r1_20260919"
PAPER = REPO / "papers/nns_vit_endpoint/figures/m6"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_m6 import patch_absolute, patch_signed  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_key(row):
    return "image__" + row["task_id"]


def neuron_key(row):
    return "m3__" + "__".join(map(str, (row["task_id"], row["block"], row["token"], row["channel"])))


def signed(axis, value, title):
    data = np.asarray(value).sum(axis=0) if np.asarray(value).ndim == 3 else np.asarray(value)
    maximum = float(np.max(np.abs(data))) or 1.0
    axis.imshow(data, cmap="RdBu_r", norm=Normalize(-maximum, maximum), interpolation="none")
    axis.set_title(title, fontsize=8); axis.axis("off")


def absolute(axis, value, title):
    data = np.abs(value).sum(axis=0) if np.asarray(value).ndim == 3 else np.abs(np.asarray(value))
    axis.imshow(data, cmap="magma", interpolation="none")
    axis.set_title(title, fontsize=8); axis.axis("off")


def original(axis, value, title="Input"):
    axis.imshow(np.asarray(value, dtype=np.uint8), interpolation="none"); axis.set_title(title, fontsize=8); axis.axis("off")


def metadata(axis, row, fields):
    lines = [f"{name}: {row.get(name)}" for name in fields]
    axis.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7)
    axis.axis("off")


def save(fig, figure_id, description, case_ids):
    records = []; PAPER.mkdir(parents=True, exist_ok=True); (ROOT / "figures").mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        report = ROOT / "figures" / (figure_id + "." + suffix); paper = PAPER / report.name
        fig.savefig(report, dpi=180 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        shutil.copyfile(report, paper)
        records.append({"format": suffix, "report_path": report.relative_to(REPO).as_posix(), "paper_path": paper.relative_to(REPO).as_posix(),
                        "sha256": sha(report), "bytes": report.stat().st_size})
    plt.close(fig)
    return {"figure_id": figure_id, "description": description, "case_ids": case_ids, "files": records}


def neuron_figure(cases, arrays):
    selected = [cases[key] for key in ("m3_early_local_patch", "m3_late_dispersed_patch", "m3_cls_patch_contrast", "m3_high_signed_cancellation")]
    rows = []
    for case in selected:
        for unit in case["selected_units"]: rows.append((case["case_id"], unit))
    fig, axes = plt.subplots(len(rows), 5, figsize=(13, 2.35 * len(rows)))
    for index, (case_id, row) in enumerate(rows):
        value = arrays[neuron_key(row)]; original(axes[index, 0], arrays[image_key(row)], case_id.replace("m3_", ""))
        signed(axes[index, 1], value, "Signed Endpoint"); absolute(axes[index, 2], value, "Absolute Endpoint")
        absolute(axes[index, 3], patch_absolute(value), "Patch mass")
        metadata(axes[index, 4], row, ("model_id", "sample_id", "block", "token", "channel", "normalized_spatial_entropy", "own_token_locality", "input_dependency_distance", "cancellation"))
    fig.suptitle("Automatically selected token-channel local input dependence", fontsize=13); fig.tight_layout()
    return save(fig, "m6_neuron_atlas", "Four frozen M3 case categories; the CLS-patch category includes the paired units.", [x["case_id"] for x in selected])


def model_figure(case, arrays):
    rows = case["selected_units"]; fig, axes = plt.subplots(2, 5, figsize=(13, 5.2))
    for model_index, model in enumerate(("vit_b16", "deit_s16")):
        units = sorted((row for row in rows if row["model_id"] == model), key=lambda row: row["block"])
        original(axes[model_index, 0], arrays[image_key(units[0])], model)
        for offset, row in enumerate(units):
            value = arrays[neuron_key(row)]; signed(axes[model_index, 1 + 2 * offset], value, f"Block {row['block']} signed")
            absolute(axes[model_index, 2 + 2 * offset], patch_absolute(value), f"Patch mass\nH={row['normalized_spatial_entropy']:.3f}")
    fig.suptitle("Representative shared-image depth contrast in ViT-B/16 and DeiT-S/16", fontsize=13); fig.tight_layout()
    return save(fig, "m6_model_layer_comparison", "Median-nearest cross-model depth-difference case with median-nearest token-30 channels.", [case["case_id"]])


def attention_figure(cases, arrays):
    selected = [cases[key] for key in ("m4_routing_dominant", "m4_value_dominant", "m4_high_frozen_attention_error")]
    fig, axes = plt.subplots(3, 9, figsize=(20, 7.2))
    for index, case in enumerate(selected):
        row = case["selected_heads"][0]; prefix = case["case_id"] + "__"
        original(axes[index, 0], arrays[image_key(row)], case["case_id"].replace("m4_", ""))
        for column, component in enumerate(("full", "routing", "value", "q", "k"), 1): signed(axes[index, column], arrays[prefix + component], component.upper())
        absolute(axes[index, 6], patch_absolute(arrays[prefix + "full"]), "Full patch mass")
        absolute(axes[index, 7], arrays[prefix + "attention_cls"], "CLS attention")
        metadata(axes[index, 8], row, ("model_id", "sample_id", "block", "head", "routing_share", "frozen_attention_endpoint_error", "mean_query_saturation"))
    fig.suptitle("Automatically selected target-conditioned attention paths", fontsize=13); fig.tight_layout()
    return save(fig, "m6_attention_cases", "Full, routing, Value, Q and K Endpoint maps plus CLS-query attention for three frozen head cases.", [x["case_id"] for x in selected])


def residual_figure(cases, arrays):
    selected = [cases[key] for key in ("m4_identity_dominant", "m4_branch_dominant", "m4_strong_path_cancellation")]
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.2))
    for index, case in enumerate(selected):
        row = case["selected_residuals"][0]; prefix = case["case_id"] + "__"
        original(axes[index, 0], arrays[image_key(row)], case["case_id"].replace("m4_", ""))
        for column, component in enumerate(("output", "identity", "branch"), 1): signed(axes[index, column], arrays[prefix + component], component.title())
        absolute(axes[index, 4], patch_absolute(arrays[prefix + "output"]), "Output patch mass")
        metadata(axes[index, 5], row, ("model_id", "sample_id", "block", "residual_type", "identity_share", "branch_share", "path_cancellation", "output_identity_patch_tv"))
    fig.suptitle("Automatically selected target-conditioned residual paths", fontsize=13); fig.tight_layout()
    return save(fig, "m6_residual_cases", "Output, identity and branch Endpoint maps for three frozen residual cases.", [x["case_id"] for x in selected])


def boundary_figure(case, arrays):
    row = case["selected_boundary"]; fig, axes = plt.subplots(1, 5, figsize=(15, 3.1))
    original(axes[0], arrays[image_key(row)], "Input")
    absolute(axes[1], arrays["m2_semantic_boundary__endpoint_absolute"], "Endpoint absolute")
    absolute(axes[2], arrays["m2_semantic_boundary__regional_taylor"], "Regional Taylor")
    by_method = {item["method"]: item for item in row["curves"]}
    for axis, outcome in zip(axes[3:], ("full", "isolated")):
        for method, color in (("endpoint_absolute", "#355c9a"), ("regional_taylor", "#d07a2d")):
            points = by_method[method]["points"]; axis.plot([p["area"] for p in points], [p["absolute"][outcome] for p in points], marker="o", label=method, color=color)
        axis.set_title(outcome + " finite effect", fontsize=8); axis.set_xlabel("selected area"); axis.grid(alpha=.2)
    axes[3].set_ylabel("absolute logit effect"); axes[3].legend(fontsize=6, frameon=False)
    fig.suptitle(f"M2 semantic boundary: fixed-reference displacement, block 5 head {row['head']}", fontsize=12); fig.tight_layout()
    return save(fig, "m6_m2_semantic_boundary", "Endpoint dependence and reference-matched Taylor answer different questions; selection did not use intervention outcomes.", [case["case_id"]])


def main():
    selection = load(ROOT / "selection_manifest.json"); receipt = load(ROOT / "map_recompute_receipt.json")
    if sha(ROOT / "selection_manifest.json") != receipt["selection_manifest_sha256"]: raise ValueError("SELECTION_DRIFT")
    if sha(ROOT / "raw_arrays.npz") != receipt["raw_arrays_sha256"]: raise ValueError("ARRAY_DRIFT")
    cases = {row["case_id"]: row for row in selection["cases"]}
    with np.load(ROOT / "raw_arrays.npz") as source: arrays = {key: source[key] for key in source.files}
    figures = [neuron_figure(cases, arrays), model_figure(cases["m3_model_layer_difference"], arrays), attention_figure(cases, arrays),
               residual_figure(cases, arrays), boundary_figure(cases["m2_semantic_boundary"], arrays)]
    manifest = {"stage": "ENDPOINT-M6", "selection_manifest_sha256": sha(ROOT / "selection_manifest.json"), "raw_arrays_sha256": sha(ROOT / "raw_arrays.npz"),
                "display_policy": "per-panel scaling; signed symmetric; no smoothing; numerical claims use raw arrays", "figures": figures}
    (ROOT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (REPO / "papers/nns_vit_endpoint/figure_manifest_m6.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "M6_FIGURES_RENDERED", "figures": len(figures), "files": sum(len(x["files"]) for x in figures)}))


if __name__ == "__main__":
    main()
