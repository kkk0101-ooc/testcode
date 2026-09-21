#!/usr/bin/env python3
"""Render compact v5 result figures from frozen class/image-level summaries."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
ANALYSIS = REPO / "reports/nns_vit_revision_v5_r1_20260921/analysis"
E1_R2 = REPO / "reports/nns_vit_revision_v5_e1_r2_r1_20260921/analysis"
OUT = PACKAGE / "figures/revision_v5"
OUT.mkdir(parents=True, exist_ok=True)


def rows(name):
    with (ANALYSIS / name).open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errorbar(axis, x, records, label, color, offset=0.0):
    center = np.array([float(r["median"]) for r in records])
    low = np.array([float(r["ci95_low"]) for r in records])
    high = np.array([float(r["ci95_high"]) for r in records])
    axis.errorbar(np.asarray(x) + offset, center, [center-low, high-center],
                  marker="o", lw=1.4, capsize=2.5, label=label, color=color)


plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))

# E1-R2: category consistency against WordNet-matched different classes.
with (E1_R2 / "summary.csv").open() as stream:
    e1 = list(csv.DictReader(stream))
ax = axes[0, 0]
blocks = [0, 3, 7, 11]
for model, color, marker_offset in [("vit_b16", "#2166ac", -0.08), ("deit_s16", "#b85c38", 0.08)]:
    selected = [next(r for r in e1 if r["model"] == model and r["kind"] == "mlp"
                     and int(r["block"]) == b) for b in blocks]
    errorbar(ax, range(4), selected, model, color, marker_offset)
ax.axhline(0, color="#555", lw=.8)
ax.set_xticks(range(4), blocks); ax.set_xlabel("Block (zero-based)")
ax.set_ylabel("Same-class minus matched-class JS similarity")
ax.set_title("(a) E1-R2 category consistency: MLP units", loc="left", weight="bold")
ax.legend(frameon=False)

# E2: neuron and head spatial entropy.
e2 = rows("e2_summary.csv")
ax = axes[0, 1]
for model, base in [("vit_b16", "#2166ac"), ("deit_s16", "#b85c38")]:
    for kind, style in [("mlp_scalar", "-"), ("heads", "--")]:
        selected = [next(r for r in e2 if r["model"] == model and r["kind"] == kind
                         and r["metric"] == "entropy" and int(r["block"]) == b) for b in range(12)]
        ax.plot(range(12), [float(r["median"]) for r in selected], style, color=base,
                label=f"{model} | {kind}")
ax.set_xticks(range(12)); ax.set_xlabel("Block (zero-based)"); ax.set_ylabel("Normalized entropy")
ax.set_title("(b) E2 MLP-neuron and attention-head profiles", loc="left", weight="bold")
ax.legend(frameon=False, fontsize=7)

# E4: held-out prediction.
e4 = rows("e4_models.csv")
ax = axes[1, 0]
models = ["vit_b16", "deit_s16", "vit_b16_in1k", "vit_s16_augreg"]
x = np.arange(4); width = .24
for i, (method, color) in enumerate([("total", "#7f8c8d"), ("structure", "#2166ac"), ("shuffled", "#d98f39")]):
    vals = [float(next(r for r in e4 if r["model"] == m and r["features"] == method)["test_mae"]) for m in models]
    ax.bar(x + (i-1)*width, vals, width, label=method, color=color)
ax.set_xticks(x, ["ViT-B\n21k→1k", "DeiT-S", "ViT-B\n1k", "ViT-S"], rotation=0)
ax.set_ylabel("Held-out MAE (lower is better)")
ax.set_title("(c) E4 effect prediction on independent test images", loc="left", weight="bold")
ax.legend(frameon=False)

# E5: four frozen hypotheses across checkpoints.
e5 = rows("e5_summary.csv")
ax = axes[1, 1]
hypotheses = ["H1", "H2", "H3", "H4"]
colors = ["#2166ac", "#b85c38", "#469b76", "#8e6caa"]
for i, (model, color) in enumerate(zip(models, colors)):
    selected = [next(r for r in e5 if r["model"] == model and r["hypothesis"] == h) for h in hypotheses]
    errorbar(ax, range(4), selected, model, color, (i-1.5)*.06)
ax.axhline(0, color="#555", lw=.8)
ax.set_xticks(range(4), ["H1 entropy", "H2 routing", "H3 residual", "H4 path−A×G"])
ax.set_ylabel("Image-level contrast")
ax.set_title("(d) E5 independent images and checkpoints", loc="left", weight="bold")
ax.legend(frameon=False, fontsize=7)

for ax in axes.flat:
    ax.grid(axis="y", alpha=.16)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"revision_v5_results.{suffix}", dpi=190)
plt.close(fig)

receipt = {
    "status": "RENDERED_FROM_ANALYSIS_SUMMARIES",
    "sources": {name: sha(ANALYSIS/name) for name in (
        "e2_summary.csv", "e4_models.csv", "e5_summary.csv")},
    "e1_r2_source": {"summary.csv": sha(E1_R2/"summary.csv")},
    "outputs": {f"revision_v5_results.{suffix}": sha(OUT/f"revision_v5_results.{suffix}") for suffix in ("png", "pdf")},
    "display_only": True,
}
(OUT/"results_figure_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
manifest_path = PACKAGE / "figure_manifest.json"
manifest = json.loads(manifest_path.read_text())
record = next(row for row in manifest["figures"] if row.get("id") == "revision_v5_results")
record["sha256"] = receipt["outputs"]["revision_v5_results.png"]
record["role"] = "prospective E1-R2 and E2-E5 summary"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
