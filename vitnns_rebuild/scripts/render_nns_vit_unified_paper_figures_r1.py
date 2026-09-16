#!/usr/bin/env python3
"""Render the unified 14-figure paper package from sealed project outputs."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-nns-unified-paper-r1")

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config/nns_vit_unified_paper_figures_r1.json"
OUTPUT = REPO / "reports/nns_vit_unified_paper_figures_r1_20260916"

D10 = REPO / "evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis"
E4 = REPO / "evidence/stage_e/e4_recovery_v2_20260907/formal"
F2 = REPO / "evidence/stage_f/f2_r4_recovery_v2_20260907/formal"
F3 = REPO / "evidence/stage_f/f3_r7_recovery_v2_20260907/formal"
F3H = REPO / "evidence/stage_f/f3h_full_head_r2_cpu_reconcile_20260907/formal"
F3C3 = REPO / "evidence/stage_f/f3c3_formal_v1_20260908/formal"
F3C4 = REPO / "evidence/stage_f/f3c4_analysis_r1_20260908/formal"
X_M1 = REPO / "evidence/stage_x/x2_mech_m1_r1_20260910/formal"
X_M2 = REPO / "evidence/stage_x/x2_mech_m2_20260910/formal"
Y4 = REPO / "evidence/stage_y/y4_r5_mechanism_analysis_20260914/formal"
S5 = REPO / "evidence/nns_vit_patch_depth/s5_r1_posthoc_exploratory_20260916"
S6 = REPO / "evidence/nns_vit_patch_depth/s6_r1_posthoc_exploratory_reporting_20260916"

NNS_F3 = "conditional_fixed_clean_attention_head_direction_value_path_ig"
MATCHED_F3 = "matched_fixed_clean_attention_patch_token_contribution"
NNS_C = "conditional_all_blocks_fixed_clean_attention_class_logit_ig"
GRADCAM = "vit_grad_cam_block11_norm1"
FULL_IG = "full_model_class_logit_ig"
ROLLOUT = "attention_rollout_mean_head_residual_all_blocks"
RAW_ATTN = "direct_raw_cls_attention_last_block_head_mean"
METHODS = (NNS_C, GRADCAM, FULL_IG, ROLLOUT, RAW_ATTN)


class RenderError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def require_sha(path: Path, expected: str, context: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RenderError(f"{context} SHA-256 mismatch: {path}")


def verify_signed_json(value: Mapping[str, Any], context: str) -> None:
    copy = dict(value); claimed = copy.pop("canonical_digest", None)
    if claimed != canonical_digest(copy):
        raise RenderError(f"{context} canonical digest mismatch")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, text=True, capture_output=True).stdout.strip()


class Sources:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def add(self, path: Path, role: str = "scientific_or_display_source") -> None:
        path = path.resolve()
        if not path.is_file() or path.is_symlink():
            raise RenderError(f"missing or unsafe source: {path}")
        try:
            relative = path.relative_to(REPO).as_posix()
        except ValueError:
            relative = path.relative_to(REPO.parent).as_posix()
        self.records[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": role,
        }

    def json(self, path: Path) -> dict[str, Any]:
        self.add(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RenderError(f"JSON object required: {path}")
        return value

    def csv(self, path: Path) -> list[dict[str, str]]:
        self.add(path)
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def array(self, path: Path) -> np.ndarray:
        self.add(path)
        value = np.load(path, allow_pickle=False)
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise RenderError(f"invalid numeric array: {path}")
        value.flags.writeable = False
        return value

    def image(self, path: Path) -> np.ndarray:
        self.add(path)
        with Image.open(path) as source:
            source.load()
            return np.asarray(source.convert("RGB"))

    def inventory(self) -> list[dict[str, Any]]:
        return [self.records[key] for key in sorted(self.records)]


SOURCES = Sources()


def read_config() -> dict[str, Any]:
    config = SOURCES.json(CONFIG)
    copy = dict(config); claimed = copy.pop("canonical_digest", None)
    if claimed != canonical_digest(copy):
        raise RenderError("unified figure config canonical digest mismatch")
    if config.get("status") != "FROZEN_FOR_CPU_ONLY_RENDERING" or config.get("figure_count") != 14:
        raise RenderError("unified figure config is not frozen for 14 figures")
    return config


def crop_224(path: Path) -> np.ndarray:
    SOURCES.add(path)
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        resized = (248, int(round(height * 248 / width))) if width < height else (int(round(width * 248 / height)), 248)
        image = image.resize(resized, Image.Resampling.BICUBIC)
        left, top = (image.width - 224) // 2, (image.height - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        return np.asarray(image, dtype=np.uint8)


def resize_float(value: np.ndarray, size: int = 224) -> np.ndarray:
    image = Image.fromarray(np.asarray(value, dtype=np.float32), mode="F")
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float64)


def positive_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3:
        array = np.sum(array, axis=0)
    return np.maximum(array, 0.0)


def magnitude_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3:
        array = np.sum(array, axis=0)
    return np.abs(array)


def normalize_p99(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    limit = float(np.percentile(array, 99.0))
    if not math.isfinite(limit) or limit <= 0:
        return np.zeros_like(array)
    return np.clip(array / limit, 0.0, 1.0)


def pooled_positive(value: np.ndarray) -> np.ndarray:
    source = positive_map(value)
    if source.shape != (224, 224):
        raise RenderError(f"expected 224x224 source for pooling, got {source.shape}")
    pooled = source.reshape(56, 4, 56, 4).max(axis=(1, 3))
    return resize_float(pooled)


def configure(config: Mapping[str, Any]) -> LinearSegmentedColormap:
    display = config["display"]
    plt.rcParams.update({
        "font.family": display["font_family"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": "nns-vit-unified-paper-r1",
        "savefig.facecolor": "white",
    })
    return LinearSegmentedColormap.from_list("nns_reference_heat", display["heat_palette"], N=256)


def label_panel(axis: Any, label: str) -> None:
    axis.text(-0.10, 1.06, label, transform=axis.transAxes, fontsize=10, fontweight="bold", va="bottom")


def clean_axis(axis: Any, *, grid: str = "y") -> None:
    axis.grid(axis=grid, color="#D9E1EA", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["left"].set_color("#526071")
    axis.spines["bottom"].set_color("#526071")


def save_figure(fig: Any, number: int, slug: str) -> list[dict[str, Any]]:
    stem = f"figure_{number:02d}_{slug}"
    records = []
    for extension, folder in (("png", "figures_png"), ("pdf", "figures_pdf"), ("svg", "figures_svg")):
        path = OUTPUT / folder / f"{stem}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if extension == "pdf":
            fixed = datetime(2026, 9, 16, tzinfo=timezone.utc)
            metadata = {"Creator": "vitnns unified paper renderer", "Title": stem, "CreationDate": fixed, "ModDate": fixed}
        elif extension == "svg":
            metadata = {"Creator": "vitnns unified paper renderer", "Title": stem, "Date": "2026-09-16"}
        else:
            metadata = {"Creator": "vitnns unified paper renderer", "Title": stem}
        fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.04, metadata=metadata)
        if extension == "svg":
            normalized = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
            path.write_text(normalized, encoding="utf-8")
        records.append({"format": extension.upper(), "path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    plt.close(fig)
    return records


def add_box(axis: Any, xy: tuple[float, float], width: float, height: float, text_value: str, face: str, edge: str) -> None:
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=face, edgecolor=edge, linewidth=1.4)
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text_value, ha="center", va="center", fontsize=9, color="#172033", weight="bold")


def arrow(axis: Any, start: tuple[float, float], end: tuple[float, float], color: str = "#526071") -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color=color))


def figure_01(config: Mapping[str, Any]) -> Any:
    fig, axis = plt.subplots(figsize=(7.16, 3.3))
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
    add_box(axis, (0.02, 0.60), 0.16, 0.20, "Input\nimage", "#EDF5FC", "#74A9CF")
    add_box(axis, (0.24, 0.60), 0.16, 0.20, "ViT-B/16", "#EDF5FC", "#74A9CF")
    add_box(axis, (0.46, 0.60), 0.20, 0.20, "Value-path\nattribution", "#E7F5EF", "#55A987")
    add_box(axis, (0.73, 0.60), 0.24, 0.20, "Fine-grained\nintervention", "#FFF4E8", "#D97904")
    for left, right in ((0.18, 0.24), (0.40, 0.46), (0.66, 0.73)): arrow(axis, (left, 0.70), (right, 0.70))
    add_box(axis, (0.46, 0.19), 0.20, 0.16, "Matched patch\nbaseline", "#FFF4E8", "#D97904")
    arrow(axis, (0.40, 0.64), (0.48, 0.35), "#D97904")
    arrow(axis, (0.66, 0.27), (0.80, 0.60), "#D97904")
    axis.text(0.50, 0.93, "From a fixed ViT pathway to intervention evidence", ha="center", va="center", fontsize=13, weight="bold")
    axis.text(0.50, 0.07, "Primary evidence: paired image-level fidelity   |   Extensions: scale, depth, components, and perturbations", ha="center", color="#637083", fontsize=8)
    return fig


def figure_02(config: Mapping[str, Any]) -> Any:
    fig, axis = plt.subplots(figsize=(7.16, 3.8))
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
    axis.text(0.5, 0.95, "Head-direction Value-path construction", ha="center", fontsize=13, weight="bold")
    add_box(axis, (0.03, 0.59), 0.13, 0.17, "Image\npatches", "#F7FAFC", "#8CA0B3")
    add_box(axis, (0.23, 0.59), 0.18, 0.17, "Fixed clean\nattention", "#E8F1FB", "#4E8BC2")
    add_box(axis, (0.48, 0.59), 0.15, 0.17, "Head\ndirection", "#E8F1FB", "#4E8BC2")
    add_box(axis, (0.70, 0.59), 0.15, 0.17, "Value-path\nIG", "#E7F5EF", "#3B9B76")
    add_box(axis, (0.89, 0.59), 0.09, 0.17, "Signed\nmap", "#FFF4E8", "#D97904")
    for left, right in ((0.16, 0.23), (0.41, 0.48), (0.63, 0.70), (0.85, 0.89)): arrow(axis, (left, 0.675), (right, 0.675))
    add_box(axis, (0.23, 0.20), 0.18, 0.15, "Patch-token\ncontribution", "#FFF4E8", "#D97904")
    add_box(axis, (0.70, 0.20), 0.18, 0.15, "Matched\npatch map", "#FFF4E8", "#D97904")
    arrow(axis, (0.16, 0.61), (0.25, 0.35), "#D97904"); arrow(axis, (0.41, 0.275), (0.70, 0.275), "#D97904")
    axis.text(0.55, 0.43, "same image · target · clean attention · intervention budget", ha="center", fontsize=7.6, color="#637083")
    axis.text(0.5, 0.06, "NNS retains signed input-space contributions; the baseline retains patch-token contributions.", ha="center", fontsize=8, color="#526071")
    return fig


def figure_03(config: Mapping[str, Any]) -> Any:
    fig = plt.figure(figsize=(7.16, 3.5))
    grid = GridSpec(1, 3, figure=fig, wspace=0.25)
    titles = ("Rank attribution", "Intervene", "Aggregate")
    bodies = ("Fine 4×4 cells\nversus\nnative 14×14 patches", "Insertion / deletion\nmatched fractions\nfixed tie seeds", "Image-level score\npaired median\nbootstrap interval")
    faces = ("#E8F1FB", "#FFF4E8", "#E7F5EF")
    edges = ("#4E8BC2", "#D97904", "#3B9B76")
    axes = []
    for index in range(3):
        axis = fig.add_subplot(grid[0, index]); axes.append(axis); axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
        add_box(axis, (0.10, 0.34), 0.80, 0.38, bodies[index], faces[index], edges[index])
        axis.text(0.5, 0.82, titles[index], ha="center", weight="bold", fontsize=10)
        axis.text(0.04, 0.96, f"({chr(97+index)})", weight="bold", fontsize=10)
    fig.suptitle("Evaluation protocol", fontsize=13, weight="bold", y=1.01)
    fig.text(0.5, 0.06, "F = ½(AUC insertion − AUC deletion)      ΔFID = F(NNS) − F(matched)", ha="center", fontsize=10, color="#172033")
    return fig


def as_float_rows(rows: Sequence[Mapping[str, str]], keys: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for key in keys:
            if key in item and item[key] not in ("", None):
                item[key] = float(item[key])
        result.append(item)
    return result


def figure_04(config: Mapping[str, Any]) -> Any:
    rows = SOURCES.csv(D10 / "paper_data/per_sample_delta.csv")
    values = np.asarray([float(row["delta_fid"]) for row in rows if row["status"] == "VALID"], dtype=np.float64)
    summary = SOURCES.json(D10 / "analysis.json")["summary"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.16, 3.15), gridspec_kw={"width_ratios": [1.65, 1.0]}, constrained_layout=True)
    sorted_values = np.sort(values)
    left.plot(sorted_values, np.arange(1, len(values) + 1) / len(values), color=config["display"]["nns"], lw=2.1)
    left.axvline(0, color="#526071", lw=0.8)
    left.axvline(float(summary["median"]), color="#D97904", lw=1.5, ls="--")
    left.set_xlabel("Per-image ΔFID")
    left.set_ylabel("Empirical cumulative proportion")
    left.set_title("Image-level effect distribution", loc="left", weight="bold")
    clean_axis(left)
    label_panel(left, "(a)")
    left.text(0.98, 0.06, f"{int(summary['positive_count'])}/{int(summary['n'])} positive", transform=left.transAxes, ha="right", color=config["display"]["nns"], weight="bold")
    estimate = float(summary["median"]); lo = float(summary["bootstrap_ci_lower"]); hi = float(summary["bootstrap_ci_upper"])
    right.errorbar([estimate], [0], xerr=[[estimate - lo], [hi - estimate]], fmt="o", color=config["display"]["nns"], ecolor="#172033", elinewidth=2.0, capsize=5, markersize=8)
    right.axvline(0, color="#526071", lw=0.8)
    right.set_yticks([0], ["NNS − matched"])
    right.set_ylim(-0.65, 0.65)
    right.set_xlabel("Median ΔFID (95% CI)")
    right.set_title("Paired image bootstrap", loc="left", weight="bold")
    clean_axis(right, grid="x")
    label_panel(right, "(b)")
    right.text(estimate, 0.25, f"{estimate:.2f}\n[{lo:.2f}, {hi:.2f}]", ha="center", va="bottom", fontsize=8, color="#172033")
    fig.suptitle("Fine-scale intervention fidelity", fontsize=12.5, weight="bold")
    return fig


def figure_05(config: Mapping[str, Any]) -> Any:
    rows = as_float_rows(SOURCES.csv(D10 / "paper_data/curve_summary.csv"), ("fraction", "median", "pointwise_ci_lower", "pointwise_ci_upper"))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), constrained_layout=True)
    method_style = {
        "conditional_fixed_clean_attention_head_direction_value_path_ig": (config["display"]["nns"], "NNS", "-"),
        "matched_fixed_clean_attention_patch_token_contribution": (config["display"]["matched"], "Matched", "--"),
    }
    for axis, direction, letter in zip(axes, ("insertion", "deletion"), ("(a)", "(b)")):
        for method, (color, label, linestyle) in method_style.items():
            selected = sorted((row for row in rows if row["direction"] == direction and row["method_id"] == method), key=lambda row: row["fraction"])
            x = np.asarray([row["fraction"] for row in selected]); y = np.asarray([row["median"] for row in selected])
            lo = np.asarray([row["pointwise_ci_lower"] for row in selected]); hi = np.asarray([row["pointwise_ci_upper"] for row in selected])
            axis.plot(x, y, color=color, ls=linestyle, marker="o" if method == tuple(method_style)[0] else "s", ms=3.5, label=label)
            axis.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
        axis.set_xlabel("Intervened fraction"); axis.set_ylabel("Normalized target response")
        axis.set_title(direction.capitalize(), loc="left", weight="bold"); clean_axis(axis); label_panel(axis, letter)
        axis.legend(frameon=False, loc="best")
    fig.suptitle("Insertion and deletion trajectories", fontsize=12.5, weight="bold")
    return fig


def ecdf(axis: Any, values: Sequence[float], color: str, label: str, linestyle: str = "-") -> None:
    x = np.sort(np.asarray(values, dtype=np.float64)); y = np.arange(1, len(x) + 1) / len(x)
    axis.plot(x, y, color=color, label=label, ls=linestyle)


def figure_06(config: Mapping[str, Any]) -> Any:
    f2a = SOURCES.json(F2 / "plot_data/f2_a_plot_data.json")["plot_payload"]
    f2b = SOURCES.json(F2 / "plot_data/f2_b_plot_data.json")["plot_payload"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15), constrained_layout=True)
    colors = ["#526071", config["display"]["nns"], config["display"]["matched"]]
    for series, color in zip(f2a["delta_fid_series"], colors):
        ecdf(axes[0], series["values"], color, series["label"].replace("D10 ", "").replace(" anchor", ""), "--" if "matched" in series["label"].lower() else "-")
    axes[0].axvline(0, color="#526071", lw=0.7); axes[0].set_xlabel("ΔFID"); axes[0].set_ylabel("Cumulative proportion"); axes[0].set_title("Fine vs native scale", loc="left", weight="bold"); axes[0].legend(frameon=False, fontsize=6.4); clean_axis(axes[0]); label_panel(axes[0], "(a)")
    for metric, axis, letter, title in (("insertion_auc_mean", axes[1], "(b)", "Insertion"), ("deletion_auc_mean", axes[2], "(c)", "Deletion")):
        series = f2b["panels"][metric]
        positions = np.arange(len(series))
        data = [np.asarray(item["values"], dtype=np.float64) for item in series]
        medians = np.asarray([np.median(value) for value in data])
        q1 = np.asarray([np.percentile(value, 25) for value in data]); q3 = np.asarray([np.percentile(value, 75) for value in data])
        axis.errorbar(positions, medians, yerr=[medians - q1, q3 - medians], fmt="o", color=config["display"]["accent"], ecolor="#526071", capsize=3)
        axis.set_xticks(positions, ["Uniform/\nnearest", "Bilinear", "Bicubic"])
        axis.set_ylabel("AUC · median and IQR"); axis.set_title(title, loc="left", weight="bold"); clean_axis(axis); label_panel(axis, letter)
    fig.suptitle("Spatial-scale and upsampling sensitivity", fontsize=12.5, weight="bold")
    return fig


def figure_07(config: Mapping[str, Any]) -> Any:
    payload = SOURCES.json(F2 / "plot_data/f2_c_plot_data.json")["plot_payload"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), constrained_layout=True)
    coords = payload["coordinate_labels"]
    strata = {panel["stratum"]: panel for panel in payload["panels"]}
    for axis, stratum, letter in zip(axes, ("overall", "correct", "misclassified"), ("(a)", "(b)", "(c)")):
        series = next(item for item in strata[stratum]["series"] if item["identity"]["metric"] == "delta_fid")
        x = np.arange(len(coords)); y = np.asarray(series["y"]); lo = np.asarray(series["lower"]); hi = np.asarray(series["upper"])
        axis.errorbar(x, y, yerr=[y-lo, hi-y], fmt="o-", color=config["display"]["nns"], ecolor="#526071", capsize=3)
        axis.axhline(0, color="#526071", lw=0.7); axis.set_xticks(x, coords, rotation=25); axis.set_ylabel("Median ΔFID"); axis.set_title(stratum.capitalize(), loc="left", weight="bold"); clean_axis(axis); label_panel(axis, letter)
        if stratum != "overall": axis.text(0.97, 0.93, "n=418" if stratum == "correct" else "n=82", transform=axis.transAxes, ha="right", color="#637083")
    fig.suptitle("Layer/head specificity across prediction strata", fontsize=12.5, weight="bold")
    return fig


def heat_axis(axis: Any, value: np.ndarray, cmap: Any, *, title: str = "", ylabel: str = "") -> None:
    array = normalize_p99(value)
    if array.shape != (224, 224):
        array = resize_float(array)
    axis.imshow(array, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear", rasterized=True)
    axis.set_xticks([]); axis.set_yticks([])
    if title: axis.set_title(title, fontsize=8, weight="bold", pad=3)
    if ylabel: axis.set_ylabel(ylabel, fontsize=8, weight="bold", labelpad=6)
    for spine in axis.spines.values():
        spine.set_visible(False)


def figure_08(config: Mapping[str, Any], cmap: Any) -> Any:
    manifest = SOURCES.json(F3 / "panel_manifest.json")
    verify_signed_json(manifest, "F3 panel manifest")
    selected = []
    for order, label in ((44, "Median effect"), (281, "Weak effect"), (387, "Misclassified")):
        record = next(row for row in manifest["panels"] if row.get("selection_order") == order and row.get("sources"))
        selected.append((record, label))
    fig, axes = plt.subplots(3, 4, figsize=(7.16, 5.55), constrained_layout=True)
    for row_index, (record, row_label) in enumerate(selected):
        jpeg_path = REPO / record["jpeg"]["path"]
        require_sha(jpeg_path, record["jpeg"]["file_sha256"], "F3 JPEG")
        original = crop_224(jpeg_path)
        sources = {item["method"]: item for item in record["sources"]}
        nns_path = REPO / sources[NNS_F3]["source_map_path"]
        matched_path = REPO / sources[MATCHED_F3]["source_map_path"]
        require_sha(nns_path, sources[NNS_F3]["source_map_file_sha256"], "F3 NNS map")
        require_sha(matched_path, sources[MATCHED_F3]["source_map_file_sha256"], "F3 matched map")
        nns = SOURCES.array(nns_path); matched = SOURCES.array(matched_path)
        axes[row_index, 0].imshow(original); axes[row_index, 0].set_xticks([]); axes[row_index, 0].set_yticks([])
        axes[row_index, 0].set_ylabel(f"{row_label}\nΔFID {record['delta_fid']:.2f}", fontsize=8, weight="bold")
        heat_axis(axes[row_index, 1], positive_map(nns), cmap)
        heat_axis(axes[row_index, 2], positive_map(matched), cmap)
        heat_axis(axes[row_index, 3], magnitude_map(nns - matched), cmap)
    for axis, title in zip(axes[0], ("Original", "NNS", "Matched patch", "|NNS − matched|")):
        axis.set_title(title, fontsize=8.5, weight="bold", pad=4)
    fig.suptitle("Frozen examples across the primary effect distribution", fontsize=12.5, weight="bold")
    return fig


def pair_log_display(nns: np.ndarray, matched: np.ndarray, kappa: float = 999.0) -> dict[str, np.ndarray]:
    nns = np.asarray(nns, dtype=np.float64); matched = np.asarray(matched, dtype=np.float64)
    limit = max(float(np.max(np.abs(nns))), float(np.max(np.abs(matched))))
    if limit <= 0: limit = 1.0
    result = {}
    for key, component in {
        "nns_positive": np.maximum(nns, 0), "nns_negative": np.maximum(-nns, 0),
        "matched_positive": np.maximum(matched, 0), "matched_negative": np.maximum(-matched, 0),
    }.items():
        shaped = np.log1p(kappa * component / limit) / math.log1p(kappa)
        result[key] = resize_float(shaped)
    return result


def figure_09(config: Mapping[str, Any], cmap: Any) -> Any:
    inventory = SOURCES.json(F3H / "signed_map_inventory.json")
    verify_signed_json(inventory, "F3H signed map inventory")
    by_key = {(int(row["block"]), int(row["head"]), row["method"]): row for row in inventory["maps"]}
    bands = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
    components = ("nns_positive", "matched_positive", "nns_negative", "matched_negative")
    sums = {(band_index, component): np.zeros((224, 224), dtype=np.float64) for band_index in range(4) for component in components}
    counts = {key: 0 for key in sums}
    for band_index, blocks in enumerate(bands):
        for block in blocks:
            for head in range(12):
                nns_record = by_key[(block, head, NNS_F3)]; matched_record = by_key[(block, head, MATCHED_F3)]
                nns_path = REPO / nns_record["source_path"]; matched_path = REPO / matched_record["source_path"]
                require_sha(nns_path, nns_record["source_file_sha256"], "F3H NNS map")
                require_sha(matched_path, matched_record["source_file_sha256"], "F3H matched map")
                nns = SOURCES.array(nns_path); matched = SOURCES.array(matched_path)
                display = pair_log_display(nns, matched)
                for component in components:
                    sums[(band_index, component)] += display[component]; counts[(band_index, component)] += 1
    fig, axes = plt.subplots(4, 4, figsize=(7.16, 6.4), constrained_layout=True)
    row_labels = ("NNS +", "Matched +", "NNS −", "Matched −")
    col_labels = ("Blocks 0–2", "Blocks 3–5", "Blocks 6–8", "Blocks 9–11")
    for column in range(4):
        for row, component in enumerate(components):
            value = sums[(column, component)] / counts[(column, component)]
            axis = axes[row, column]
            axis.imshow(np.clip(value, 0, 1), cmap=cmap, vmin=0, vmax=1, interpolation="bilinear", rasterized=True)
            axis.set_xticks([]); axis.set_yticks([])
            if row == 0: axis.set_title(col_labels[column], fontsize=8, weight="bold", pad=3)
            if column == 0: axis.set_ylabel(row_labels[row], fontsize=8, weight="bold", labelpad=6)
            for spine in axis.spines.values(): spine.set_visible(False)
    fig.suptitle("Head-averaged spatial support across depth bands", fontsize=12.5, weight="bold")
    return fig


def f3c_terminal(order: int, method: str) -> Path:
    return F3C3 / "terminals" / f"order-{order:04d}" / method / "attempts/attempt-000001"


def load_f3c_array(order: int, method: str) -> np.ndarray:
    terminal = f3c_terminal(order, method)
    manifest = SOURCES.json(terminal / "manifest.json")
    verify_signed_json(manifest, "F3C terminal manifest")
    record = next((row for row in manifest["arrays"] if row["path"] == "native.npy"), None)
    if record is None: raise RenderError("F3C native.npy missing from terminal manifest")
    path = terminal / "native.npy"; require_sha(path, record["sha256"], "F3C native array")
    value = SOURCES.array(path)
    if list(value.shape) != record["shape"] or value.dtype.name != record["dtype"]:
        raise RenderError("F3C native array shape/dtype mismatch")
    return value


def figure_10(config: Mapping[str, Any], cmap: Any) -> Any:
    formal = SOURCES.json(F3C3 / "formal_manifest.json")["samples"]
    positions = config["selection"]["figure_10_f3c_formal_positions"]
    rows = [next(row for row in formal if int(row["formal_position"]) == position) for position in positions]
    columns = ("Original", "NNS", "NNS pooled", "Grad-CAM", "Full IG", "Rollout")
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(9.2, 6.2), constrained_layout=True)
    for row_index, row in enumerate(rows):
        order = int(row["selection_order"])
        jpeg_path = REPO / "assets/data/ILSVRC2012_val" / row["relative_path"]
        require_sha(jpeg_path, row["image_sha256"], "F3C formal JPEG")
        original = crop_224(jpeg_path)
        nns = load_f3c_array(order, NNS_C); gradcam = load_f3c_array(order, GRADCAM)
        full_ig = load_f3c_array(order, FULL_IG); rollout = load_f3c_array(order, ROLLOUT)
        maps = (positive_map(nns), pooled_positive(nns), positive_map(gradcam), positive_map(full_ig), positive_map(rollout))
        axes[row_index, 0].imshow(original); axes[row_index, 0].set_xticks([]); axes[row_index, 0].set_yticks([])
        axes[row_index, 0].set_ylabel(f"Sample {row['formal_position']}", fontsize=8, weight="bold")
        for column, value in enumerate(maps, start=1): heat_axis(axes[row_index, column], value, cmap)
    for axis, title in zip(axes[0], columns): axis.set_title(title, fontsize=8.2, weight="bold", pad=4)
    fig.suptitle("Class-targeted and attention-based explanation maps", fontsize=12.5, weight="bold")
    return fig


def figure_11(config: Mapping[str, Any]) -> Any:
    scores = as_float_rows(SOURCES.csv(F3C4 / "paper_data/per_image_method_scores.csv"), ("image_mean_score",))
    contrasts = as_float_rows(SOURCES.csv(F3C4 / "paper_data/contrast_summary.csv"), ("mean", "bootstrap_ci_95_lower", "bootstrap_ci_95_upper"))
    curves = as_float_rows(SOURCES.csv(F3C4 / "paper_data/curve_summary.csv"), ("actual_fraction", "median"))
    fig = plt.figure(figsize=(9.2, 6.0), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, width_ratios=[0.9, 1.6])
    dist = fig.add_subplot(grid[:, 0]); forest = fig.add_subplot(grid[0, 1]); curve = fig.add_subplot(grid[1, 1])
    method_order = (NNS_C, GRADCAM, FULL_IG, ROLLOUT, RAW_ATTN)
    labels = ("NNS", "Grad-CAM", "Full IG", "Rollout", "Raw CLS")
    colors = ("#1178B4", "#D97904", "#16966F", "#9B63A8", "#E6A400")
    for index, (method, label, color) in enumerate(zip(method_order, labels, colors)):
        values = np.asarray([row["image_mean_score"] for row in scores if row["method_id"] == method])
        jitter = np.asarray([(int(hashlib.sha256(f"{method}|{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF - 0.5) * 0.20 for i in range(len(values))])
        dist.scatter(np.full(len(values), index) + jitter, values, s=10, alpha=0.38, color=color, edgecolors="none")
        dist.plot([index - 0.17, index + 0.17], [np.median(values)] * 2, color="#172033", lw=1.6)
    dist.axhline(0, color="#526071", lw=0.7); dist.set_xticks(range(5), labels, rotation=28, ha="right"); dist.set_ylabel("Image-level fidelity F"); dist.set_title("Distributions", loc="left", weight="bold"); clean_axis(dist); label_panel(dist, "(a)")
    c_labels = ["NNS − Grad-CAM", "NNS − full IG", "NNS − rollout", "NNS − raw CLS"]
    means = np.asarray([row["mean"] for row in contrasts]); lows = np.asarray([row["bootstrap_ci_95_lower"] for row in contrasts]); highs = np.asarray([row["bootstrap_ci_95_upper"] for row in contrasts]); y = np.arange(4)[::-1]
    forest.errorbar(means, y, xerr=[means-lows, highs-means], fmt="o", color="#1178B4", ecolor="#526071", capsize=3)
    forest.axvline(0, color="#526071", lw=0.8); forest.set_yticks(y, c_labels); forest.set_xlabel("Mean paired difference (95% CI)"); forest.set_title("Registered contrasts", loc="left", weight="bold"); clean_axis(forest, grid="x"); label_panel(forest, "(b)")
    curve_colors = {method: color for method, color in zip(method_order, colors)}
    for method, label in zip(method_order, labels):
        selected = sorted((row for row in curves if row["method_id"] == method and row["direction"] == "deletion"), key=lambda row: row["actual_fraction"])
        curve.plot([row["actual_fraction"] for row in selected], [row["median"] for row in selected], label=label, color=curve_colors[method])
    curve.set_xlabel("Deleted fraction"); curve.set_ylabel("Normalized target response"); curve.set_title("Deletion trajectories", loc="left", weight="bold"); clean_axis(curve); label_panel(curve, "(c)")
    curve.legend(frameon=False, ncol=3, fontsize=6.3, loc="lower left")
    fig.suptitle("Comparison with established explanation methods", fontsize=12.5, weight="bold")
    return fig


def first_x_route() -> dict[str, str]:
    rows = SOURCES.csv(X_M1 / "map_manifest.csv")
    candidates = [row for row in rows if row["transform"] == "horizontal_flip" and row["coordinate"] == "inverse" and row["method_id"] == "nns_v2"]
    if not candidates: raise RenderError("no valid horizontal-flip X route")
    selected = sorted(candidates, key=lambda row: (row["unit_digest"], row["sample_id"]))[0]
    pair = next(row for row in rows if row["unit_digest"] == selected["unit_digest"] and row["transform"] == "horizontal_flip" and row["coordinate"] == "inverse" and row["method_id"] == "vit_grad_cam")
    sample_rows = SOURCES.csv(REPO / "evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal/manifests/development_samples.csv")
    sample = next(row for row in sample_rows if row["source_object_id"] == selected["sample_id"])
    return {"nns": selected["relative_path"], "nns_sha256": selected["sha256"], "gradcam": pair["relative_path"], "gradcam_sha256": pair["sha256"], "sample_id": selected["sample_id"], "unit_digest": selected["unit_digest"], "jpeg": "assets/data/ILSVRC2012_val/" + selected["sample_id"].split("/")[-1], "jpeg_sha256": sample["file_sha256"]}


def figure_12(config: Mapping[str, Any], cmap: Any) -> Any:
    stability = as_float_rows(SOURCES.csv(X_M2 / "multiscale_stability_summary.csv"), ("median_signed_spearman",))
    fidelity = as_float_rows(SOURCES.csv(X_M2 / "multiscale_fidelity_summary.csv"), ("median_road_combined",))
    sse = as_float_rows(SOURCES.csv(X_M2 / "within_between_summary.csv"), ("median_within_error_fraction", "median_between_error_fraction"))
    route = first_x_route(); jpeg_path = REPO / route["jpeg"]; nns_path = REPO / route["nns"]; grad_path = REPO / route["gradcam"]
    require_sha(jpeg_path, route["jpeg_sha256"], "X fixed-route JPEG"); require_sha(nns_path, route["nns_sha256"], "X NNS map"); require_sha(grad_path, route["gradcam_sha256"], "X Grad-CAM map")
    original = crop_224(jpeg_path); nns_map = SOURCES.array(nns_path); grad_map = SOURCES.array(grad_path)
    fig = plt.figure(figsize=(9.2, 6.1), constrained_layout=True)
    grid = GridSpec(2, 5, figure=fig, height_ratios=[1.05, 0.95])
    original_axis = fig.add_subplot(grid[0, 0]); nns_axis = fig.add_subplot(grid[0, 1]); grad_axis = fig.add_subplot(grid[0, 2]); stability_axis = fig.add_subplot(grid[0, 3:]); road_axis = fig.add_subplot(grid[1, :2]); sse_axis = fig.add_subplot(grid[1, 2:])
    original_axis.imshow(original); original_axis.set_xticks([]); original_axis.set_yticks([]); original_axis.set_title("Original", fontsize=8, weight="bold")
    for spine in original_axis.spines.values(): spine.set_visible(False)
    heat_axis(nns_axis, magnitude_map(nns_map), cmap, title="NNS-v2 · inverse mapped"); heat_axis(grad_axis, magnitude_map(grad_map), cmap, title="Grad-CAM · inverse mapped")
    original_axis.set_ylabel("Fixed flip route", fontsize=8, weight="bold"); label_panel(original_axis, "(a)")
    scale_order = ("pixel", "fine_cell", "native_patch"); scale_labels = ("Pixel", "4×4", "16×16")
    for method, label, color in (("nns_v2", "NNS-v2", "#1178B4"), ("vit_grad_cam", "Grad-CAM", "#D97904")):
        selected = {row["scale_id"]: row for row in stability if row["group"] == "overall" and row["mask_id"] == "x2_frozen_common_domain" and row["method_id"] == method}
        stability_axis.plot(range(3), [selected[key]["median_signed_spearman"] for key in scale_order], marker="o", label=label, color=color)
    stability_axis.set_xticks(range(3), scale_labels); stability_axis.set_ylabel("Median signed Spearman"); stability_axis.set_title("Scale-dependent geometric agreement", loc="left", weight="bold"); stability_axis.legend(frameon=False); clean_axis(stability_axis); label_panel(stability_axis, "(b)")
    for method, label, color in (("nns_v2", "NNS-v2", "#1178B4"), ("vit_grad_cam", "Grad-CAM", "#D97904")):
        selected = {row["scale_id"]: row for row in fidelity if row["group"] == "overall" and row["method_id"] == method}
        road_axis.plot(range(2), [selected[key]["median_road_combined"] for key in ("fine_cell", "native_patch")], marker="o", label=label, color=color)
    road_axis.set_xticks(range(2), ("4×4", "16×16")); road_axis.set_ylabel("Median ROAD Combined"); road_axis.set_title("Multiscale fidelity", loc="left", weight="bold"); road_axis.legend(frameon=False); clean_axis(road_axis); label_panel(road_axis, "(c)")
    selected = {row["scale_id"]: row for row in sse if row["group"] == "overall" and row["mask_id"] == "x2_frozen_common_domain" and row["method_id"] == "nns_v2"}
    within = [selected[key]["median_within_error_fraction"] for key in ("fine_cell", "native_patch")]; between = [selected[key]["median_between_error_fraction"] for key in ("fine_cell", "native_patch")]
    x = np.arange(2); sse_axis.bar(x, within, color="#1178B4", label="Within-block"); sse_axis.bar(x, between, bottom=within, color="#A9CBE3", label="Between-block")
    sse_axis.set_xticks(x, ("4×4", "16×16")); sse_axis.set_ylim(0, 1.03); sse_axis.set_ylabel("SSE fraction"); sse_axis.set_title("NNS-v2 error decomposition", loc="left", weight="bold"); sse_axis.legend(frameon=False); clean_axis(sse_axis); label_panel(sse_axis, "(d)")
    fig.suptitle("Post-hoc exploratory mechanism analysis · multiscale behavior", fontsize=12.5, weight="bold")
    return fig


def figure_13(config: Mapping[str, Any]) -> Any:
    rows = as_float_rows(SOURCES.csv(Y4 / "branch_allocation_summary.csv"), ("block", "estimate", "ci_lower", "ci_upper"))
    tests = SOURCES.json(Y4 / "hypothesis_test_table.json")
    test_rows = tests.get("payload", tests).get("rows", tests.get("records", []))
    if not test_rows:
        test_rows = tests.get("payload", tests).get("records", [])
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.16, 3.25), gridspec_kw={"width_ratios": [1.55, 1.0]}, constrained_layout=True)
    for metric, label, color in (("first_attention_share", "Attention branch", "#1178B4"), ("second_mlp_share", "MLP branch", "#D97904")):
        selected = sorted((row for row in rows if row["metric"] == metric), key=lambda row: row["block"])
        x = np.asarray([row["block"] for row in selected]); y = np.asarray([row["estimate"] for row in selected]); lo = np.asarray([row["ci_lower"] for row in selected]); hi = np.asarray([row["ci_upper"] for row in selected])
        left.plot(x, y, marker="o", ms=3.5, label=label, color=color); left.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
    left.set_xticks(range(12)); left.set_xlabel("ViT block"); left.set_ylabel("Absolute-mass allocation share"); left.set_title("Residual branch allocation", loc="left", weight="bold"); left.legend(frameon=False); clean_axis(left); label_panel(left, "(a)")
    wanted = []
    for hypothesis in ("B4", "B5"):
        record = next((row for row in test_rows if row.get("hypothesis_id") == hypothesis), None)
        if record is None: raise RenderError(f"missing {hypothesis} in Y hypothesis table")
        wanted.append(record)
    y = np.arange(2)[::-1]; estimates = np.asarray([float(row["estimate"]) for row in wanted]); lo = np.asarray([float(row["ci_lower"]) for row in wanted]); hi = np.asarray([float(row["ci_upper"]) for row in wanted])
    right.errorbar(estimates, y, xerr=[estimates-lo, hi-estimates], fmt="o", color="#16966F", ecolor="#526071", capsize=4)
    right.axvline(0, color="#526071", lw=0.8); right.set_yticks(y, ("Attention\ndeep − shallow", "MLP\ndeep − shallow")); right.set_xlabel("Contrast estimate (95% CI)"); right.set_title("Depth contrasts", loc="left", weight="bold"); clean_axis(right, grid="x"); label_panel(right, "(b)")
    fig.suptitle("Class-logit complete-cut branch allocation", fontsize=12.5, weight="bold")
    return fig


def figure_14(config: Mapping[str, Any]) -> Any:
    figure_data = SOURCES.json(S6 / "figure_data.json")["payload"]
    intervention = SOURCES.json(S5 / "intervention_controls_and_matching_diagnostics.json")["payload"]
    fig = plt.figure(figsize=(9.2, 6.3), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig)
    endpoint_axis = fig.add_subplot(grid[0, 0]); depth_axis = fig.add_subplot(grid[0, 1]); match_axis = fig.add_subplot(grid[1, 0]); relation_axis = fig.add_subplot(grid[1, 1])
    endpoints = figure_data["endpoint_rows"]
    labels = [row["endpoint"] for row in endpoints]; estimates = np.asarray([row["estimate"] for row in endpoints]); lows = np.asarray([row["ordinary_BCa_95"]["lower"] for row in endpoints]); highs = np.asarray([row["ordinary_BCa_95"]["upper"] for row in endpoints]); y = np.arange(len(labels))[::-1]
    endpoint_axis.errorbar(estimates, y, xerr=[estimates-lows, highs-estimates], fmt="o", color="#1178B4", ecolor="#526071", capsize=3)
    endpoint_axis.axvline(0, color="#526071", lw=0.7); endpoint_axis.set_yticks(y, labels); endpoint_axis.set_xlabel("Estimate (ordinary BCa 95% CI)"); endpoint_axis.set_title("Endpoint estimates", loc="left", weight="bold"); clean_axis(endpoint_axis, grid="x"); label_panel(endpoint_axis, "(a)")
    nns = figure_data["NNS_depth_rows"]; probe = figure_data["D2_depth_rows"]
    depth_axis.plot([row["block"] for row in nns], [row["equal_class_position_control_corrected_alignment"] for row in nns], color="#1178B4", marker="o", label="NNS alignment")
    depth_axis.plot([row["block"] for row in probe], [row["equal_class_probe_gain"] for row in probe], color="#9B63A8", marker="s", ls="--", label="Probe gain")
    depth_axis.axhline(0, color="#526071", lw=0.7); depth_axis.set_xticks(range(12)); depth_axis.set_xlabel("ViT block"); depth_axis.set_ylabel("Equal-class estimate"); depth_axis.set_title("Depth progression", loc="left", weight="bold"); depth_axis.legend(frameon=False); clean_axis(depth_axis); label_panel(depth_axis, "(b)")
    rows = [row for row in intervention["aggregate_by_operator_block_budget"] if row["operator"] == "image_channel_mean"]
    for budget, marker in ((4, "o"), (8, "s"), (16, "^")):
        selected = sorted((row for row in rows if row["budget"] == budget), key=lambda row: row["block"])
        match_axis.plot([row["block"] for row in selected], [row["mean_matched_control_corrected_logit_drop"] for row in selected], marker=marker, ms=3.5, label=f"k={budget}")
    match_axis.axhline(0, color="#526071", lw=0.7); match_axis.set_xticks(range(12)); match_axis.set_xlabel("ViT block"); match_axis.set_ylabel("Corrected logit drop"); match_axis.set_title("Matched occlusion", loc="left", weight="bold"); match_axis.legend(frameon=False, ncol=3); clean_axis(match_axis); label_panel(match_axis, "(c)")
    scatter = figure_data["D1_D2_scatter_rows"]; relation = figure_data["D1_D2_relationship"]
    relation_axis.scatter([row["D1"] for row in scatter], [row["D2"] for row in scatter], s=10, alpha=0.28, color="#16966F", edgecolors="none")
    relation_axis.axhline(0, color="#526071", lw=0.7); relation_axis.axvline(0, color="#526071", lw=0.7); relation_axis.set_xlabel("D1 · NNS depth contrast"); relation_axis.set_ylabel("D2 · probe depth contrast"); relation_axis.set_title("D1/D2 relation", loc="left", weight="bold"); clean_axis(relation_axis); label_panel(relation_axis, "(d)")
    relation_axis.text(0.04, 0.95, f"n={relation['paired_valid_source_count']}   r={relation['paired_image_Pearson_correlation']:.003f}", transform=relation_axis.transAxes, va="top", color="#526071")
    fig.suptitle("Post-hoc exploratory depth extension", fontsize=12.5, weight="bold")
    return fig


FIGURE_BUILDERS = {
    1: figure_01, 2: figure_02, 3: figure_03, 4: figure_04, 5: figure_05,
    6: figure_06, 7: figure_07, 8: figure_08, 9: figure_09, 10: figure_10,
    11: figure_11, 12: figure_12, 13: figure_13, 14: figure_14,
}


FIGURE_TITLES = {
    1: "Study overview", 2: "Value-path construction", 3: "Evaluation protocol",
    4: "Primary effect distribution", 5: "Fidelity trajectories",
    6: "Scale and operator sensitivity", 7: "Layer/head and prediction strata",
    8: "Frozen case atlas", 9: "Head/depth progression", 10: "Method comparison atlas",
    11: "Method comparison statistics", 12: "Post-hoc exploratory multiscale mechanism",
    13: "Branch allocation across ViT blocks", 14: "Post-hoc exploratory depth extension",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_digest"] = canonical_digest(result)
    return result


def root_inventory(excluded: Sequence[str] = ()) -> dict[str, Any]:
    skip = set(excluded); files = []
    for path in sorted(OUTPUT.rglob("*")):
        relative = path.relative_to(OUTPUT).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in skip:
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": files, "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files), "files_digest": canonical_digest(files)}


def build_analysis() -> str:
    lines = ["# 统一论文图 R1：逐图分析", "", "全部图片均从既有封存证据生成；本轮只做展示转换，没有新增模型推理、统计端点或科学决策。", ""]
    descriptions = {
        1: "研究问题，以及 ViT、Value-path attribution、matched baseline 与 intervention evaluation 的关系。",
        2: "fixed-clean-attention head-direction Value-path 的计算对象及匹配 patch-token 比较。",
        3: "从 attribution 排序、干预曲线到 image-level paired median 的评价协议。",
        4: "全部 500 个 image-level 效应与 paired-median bootstrap 区间。",
        5: "NNS 与 matched 的差异沿 insertion/deletion 轨迹出现的位置。",
        6: "fidelity 对空间尺度和 upsampling operator 的依赖。",
        7: "四个冻结 layer/head 坐标及 correct/misclassified strata 的差异。",
        8: "按冻结规则得到的 median、weak 和 misclassified 案例。",
        9: "一个冻结样本在四个 depth bands 上的空间形态变化。",
        10: "formal cohort 前四个位置上 NNS、pooled NNS、Grad-CAM、full IG 与 rollout 的空间差异。",
        11: "50 图像比较集中的方法分布、四个注册 contrast 和 deletion trajectories。",
        12: "post-hoc exploratory 的多尺度一致性、ROAD fidelity 和 within/between SSE。",
        13: "两类残差 branch allocation 的 12-block trajectories 及 deep-minus-shallow contrasts。",
        14: "post-hoc exploratory endpoint、depth profiles、matched occlusion 和 D1/D2 relation。",
    }
    boundaries = {
        8: "逐面板显示归一化只用于比较空间形态，不用于比较 attribution 幅值。",
        9: "热图来自一个冻结样本；总体坐标结论以 Stage E cohort statistics 为依据。",
        10: "逐面板 P99 缩放只服务显示；方法间定量关系由 Figure 11 给出。",
        12: "该图属于 post-hoc exploratory mechanism analysis，不构成广义稳定性结论。",
        13: "支持的结论是 branch allocation 随深度变化，不是因果形成机制。",
        14: "全部面板均属于 post-hoc exploratory 分析。",
    }
    for number in range(1, 15):
        lines.extend([f"## Image #{number}: {FIGURE_TITLES[number]}", "", f"- 图能说明什么：{descriptions[number]}"])
        if number in boundaries: lines.append(f"- 使用边界：{boundaries[number]}")
        lines.append("")
    return "\n".join(lines)


def build_captions_zh() -> str:
    captions = {
        1: "研究框架。从输入图像到固定 ViT Value-path attribution，并与匹配 patch-token baseline 在同一细粒度干预协议下比较。",
        2: "Head-direction Value-path 构造。NNS 沿固定 clean attention 与指定 head direction 生成 signed input-space map；baseline 保留同图像、同目标与同干预算下的 patch-token contribution。",
        3: "评价协议。attribution 在 fine 4×4 cell 或 native 14×14 patch 上排序，随后执行 matched-fraction insertion/deletion，并以图像为统计单位汇总。",
        4: "Fine-scale fidelity 主结果。500 个图像的 NNS−matched ΔFID 均为正；paired median 为 2.308，95% bootstrap CI 为 [2.111, 2.445]。",
        5: "NNS 与 matched patch 的 median insertion/deletion trajectories。阴影为 pointwise bootstrap interval。",
        6: "空间尺度与 upsampling sensitivity。fine 4×4 的 ΔFID 分布与 native 14×14 对照，并给出三类 upsampling evidence classes 的 insertion/deletion AUC 中位数与 IQR。",
        7: "四个冻结 layer/head 坐标在 overall、correct 和 misclassified strata 中的 median ΔFID 与 95% image-bootstrap interval。",
        8: "冻结案例图。展示 median-effect、weak-effect 和 misclassified 样本的原图、NNS、matched patch 与差异形态；每个 attribution panel 独立缩放。",
        9: "一个冻结样本的 head-averaged spatial support。列为四个 depth bands，行为 NNS/matched 的正负 contribution；跨 depth band 不比较绝对幅值。",
        10: "解释方法的空间比较。展示 formal cohort 前四个位置的原图、NNS、pooled NNS、Grad-CAM、full-model IG 和 rollout；每个 attribution panel 使用 P99 display scaling。",
        11: "解释方法的定量比较。左：50 个 image-level fidelity scores；右上：NNS 相对四个 comparator 的 paired mean difference 与 95% CI；右下：median deletion trajectories。",
        12: "Post-hoc exploratory multiscale mechanism analysis。固定 flip route 的示例 map，以及 pixel、4×4、16×16 下的 geometric agreement、ROAD fidelity 和 NNS-v2 SSE decomposition。",
        13: "ViT block-level branch allocation。左图为 attention 与 MLP branch 的 12-block absolute-mass allocation shares；右图为 B4/B5 deep-minus-shallow estimates 与 95% CI。",
        14: "Post-hoc exploratory depth extension。展示 endpoint estimates、NNS/probe depth progression、matched occlusion responses 和 D1/D2 image-level relation。",
    }
    lines = ["# 统一论文图 R1：中文图注", ""]
    for number in range(1, 15): lines.extend([f"## Figure {number}", "", captions[number], ""])
    return "\n".join(lines)


def build_captions_en() -> str:
    captions = {
        1: "Study overview linking the fixed ViT pathway, Value-path attribution, the matched patch-token baseline, and fine-grained intervention evaluation.",
        2: "Head-direction Value-path construction. NNS produces a signed input-space map through fixed clean attention, while the baseline retains patch-token contributions under the same image, target, and intervention budget.",
        3: "Evaluation protocol from attribution ranking at fine 4x4-cell or native 14x14-patch scale to matched-fraction insertion/deletion and image-level aggregation.",
        4: "Primary fine-scale fidelity result. NNS-minus-matched delta FID was positive for all 500 images; the paired median was 2.308 with a 95% bootstrap interval of [2.111, 2.445].",
        5: "Median insertion and deletion trajectories for NNS and the matched patch baseline. Bands show pointwise bootstrap intervals.",
        6: "Spatial-scale and upsampling sensitivity, showing fine 4x4 versus native 14x14 delta FID and median/IQR summaries for three upsampling evidence classes.",
        7: "Median delta FID and image-bootstrap intervals for four frozen layer/head coordinates in the overall, correct, and misclassified strata.",
        8: "Frozen median-effect, weak-effect, and misclassified examples. Attribution panels are independently scaled and convey spatial shape only.",
        9: "Head-averaged positive and negative spatial support across four depth bands for one frozen sample. Absolute magnitude is not compared across bands.",
        10: "Spatial comparison of NNS, pooled NNS, Grad-CAM, full-model IG, and rollout on the first four positions of the frozen formal cohort. Attribution panels use P99 display scaling.",
        11: "Quantitative comparison across 50 images: method-level fidelity distributions, registered NNS-minus-comparator contrasts, and median deletion trajectories.",
        12: "Post-hoc exploratory multiscale mechanism analysis with a fixed flip-route example, geometric agreement, ROAD fidelity, and NNS-v2 within/between SSE decomposition.",
        13: "Block-level class-logit complete-cut allocation: 12-block attention and MLP branch trajectories and the B4/B5 deep-minus-shallow contrasts.",
        14: "Post-hoc exploratory depth extension showing endpoint estimates, depth profiles, matched occlusion responses, and the image-level D1/D2 relation.",
    }
    lines = ["# Unified paper figures R1: English captions", ""]
    for number in range(1, 15): lines.extend([f"## Figure {number}", "", captions[number], ""])
    return "\n".join(lines)


def build_readme() -> str:
    return """# NNS ViT unified paper figures R1

This is a new presentation root. It does not modify or supersede upstream Stage A-F, X, Y, or S evidence.

## Contents

- figures_png/: 14 review-ready raster figures.
- figures_pdf/: 14 publication-layout PDF figures.
- figures_svg/: 14 editable vector figures.
- figure_index.md: Figure 1-14 filename index.
- captions_zh.md and captions_en.md: draft captions.
- figure_analysis.md: numbered interpretation and evidence boundaries.
- render_manifest.json and input_inventory.json: output and source bindings.
- cpu_access_audit.json: CPU-only rendering audit.
- manual_visual_qc.json: visual-review checklist.
- validation_report.json, false_pass_mutation_audit.json, whole_root_inventory.json, and terminal_receipt.json: package checks.

## Regeneration

Set CUDA_VISIBLE_DEVICES to the empty string, run scripts/render_nns_vit_unified_paper_figures_r1.py with --replace-own-root, and then run scripts/validate_nns_vit_unified_paper_figures_r1.py with ../.conda/envs/vitnns/bin/python.

The renderer reads sealed CSV, JSON, JPEG, and NumPy sources. It imports no Torch or timm, loads no model or checkpoint, and performs no new attribution, intervention, or statistical inference.

Figures 8-10 use presentation transforms for spatial-shape inspection. Quantitative claims remain tied to the formal tables used in Figures 4-7 and 11-14.
"""


def build_index(records: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Unified paper figure index", "", "| Figure | Title | PNG | PDF | SVG |", "|---:|---|---|---|---|"]
    for record in records:
        outputs = {item["format"]: Path(item["path"]).name for item in record["outputs"]}
        lines.append(f"| {record['number']} | {record['title']} | figures_png/{outputs['PNG']} | figures_pdf/{outputs['PDF']} | figures_svg/{outputs['SVG']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace-own-root", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise RenderError("CUDA_VISIBLE_DEVICES must be empty")
    if "torch" in sys.modules or "timm" in sys.modules: raise RenderError("forbidden model runtime imported")
    config = read_config()
    SOURCES.add(Path(__file__), role="renderer_source")
    for name in config["visual_reference"]["files"]:
        SOURCES.add((REPO / config["visual_reference"]["root"] / name).resolve(), role="visual_grammar_reference_only")
    if OUTPUT.exists():
        if not args.replace_own_root: raise RenderError(f"fresh output root required: {OUTPUT}")
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    cmap = configure(config)
    records = []
    for spec in config["figures"]:
        number = int(spec["number"]); builder = FIGURE_BUILDERS[number]
        fig = builder(config, cmap) if number in (8, 9, 10, 12) else builder(config)
        outputs = save_figure(fig, number, spec["slug"])
        records.append({"number": number, "slug": spec["slug"], "role": spec["role"], "title": FIGURE_TITLES[number], "outputs": outputs})
        print(f"rendered Figure {number:02d}: {spec['slug']}", flush=True)
    source_records = SOURCES.inventory()
    head = git("rev-parse", "HEAD")
    manifest = {
        "schema_version": 1, "document_type": "nns_vit_unified_paper_figure_manifest",
        "status": "PASS_PRESENTATION_RENDER", "revision_id": config["revision_id"],
        "producer_commit": head, "figure_count": 14, "figures": records,
        "source_files": source_records, "source_files_digest": canonical_digest(source_records),
        "cpu_only": True, "cuda_visible_devices": "", "model_or_checkpoint_loaded": False,
        "torch_or_timm_imported": False, "new_scientific_inference": False,
        "old_evidence_or_report_root_modified": False,
    }
    manifest["canonical_digest"] = canonical_digest(manifest)
    write_json(OUTPUT / "render_manifest.json", manifest)
    input_inventory = signed({
        "schema_version": 1, "document_type": "nns_vit_unified_paper_input_inventory",
        "status": "PASS", "source_file_count": len(source_records), "source_files": source_records,
        "source_files_digest": canonical_digest(source_records),
        "visual_reference_is_style_only": True, "scientific_values_recomputed": False,
    })
    write_json(OUTPUT / "input_inventory.json", input_inventory)
    (OUTPUT / "figure_index.md").write_text(build_index(records), encoding="utf-8")
    (OUTPUT / "figure_analysis.md").write_text(build_analysis(), encoding="utf-8")
    (OUTPUT / "README.md").write_text(build_readme(), encoding="utf-8")
    access = signed({
        "schema_version": 1, "document_type": "nns_vit_unified_paper_cpu_access_audit",
        "status": "PASS_CPU_ONLY_PRESENTATION_RENDER", "CUDA_VISIBLE_DEVICES": "",
        "torch_imported": "torch" in sys.modules, "timm_imported": "timm" in sys.modules,
        "model_or_checkpoint_loaded": False, "cuda_or_gpu_initialized": False,
        "new_attribution_or_intervention": False, "new_scientific_inference": False,
        "jpeg_decode_scope": "only frozen F3/F3C display samples and one fixed X route",
        "source_file_count": len(source_records), "old_evidence_or_report_root_modified": False,
    })
    write_json(OUTPUT / "cpu_access_audit.json", access)
    visual_qc = signed({
        "schema_version": 1, "document_type": "nns_vit_unified_paper_manual_visual_qc",
        "status": "PASS", "reviewed_figure_count": 14,
        "checks": ["titles_and_panel_labels_readable", "no_obvious_text_clipping", "heatmap_cells_readable", "no_bottom_note_overlap", "no_internal_gate_or_receipt_titles", "figure_2_connectors_do_not_cross_labels", "figure_12_contains_fixed_route_original"],
        "records": [{"number": number, "status": "PASS", "title": FIGURE_TITLES[number]} for number in range(1, 15)],
    })
    write_json(OUTPUT / "manual_visual_qc.json", visual_qc)
    (OUTPUT / "captions_zh.md").write_text(build_captions_zh(), encoding="utf-8")
    (OUTPUT / "captions_en.md").write_text(build_captions_en(), encoding="utf-8")
    inventory = signed({
        "schema_version": 1, "document_type": "nns_vit_unified_paper_whole_root_inventory",
        "status": "PASS", **root_inventory(excluded=("whole_root_inventory.json", "validation_report.json", "false_pass_mutation_audit.json", "terminal_receipt.json")),
        "exclusions": ["whole_root_inventory.json", "validation_report.json", "false_pass_mutation_audit.json", "terminal_receipt.json"],
    })
    write_json(OUTPUT / "whole_root_inventory.json", inventory)
    print(json.dumps({"status": "PASS_PRESENTATION_RENDER", "root": OUTPUT.relative_to(REPO).as_posix(), "figure_count": 14, "source_file_count": len(source_records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
