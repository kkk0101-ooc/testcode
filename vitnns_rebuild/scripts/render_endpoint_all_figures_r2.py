#!/usr/bin/env python3
"""Render the organized R2 Endpoint-NNSViT paper-figure review package.

This renderer is CPU-only and presentation-only. It reuses sealed arrays and
published numeric summaries; it does not import Torch/timm, load a checkpoint,
run a model, or recompute statistical decisions.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/endpoint-all-figures-r2-mpl")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1789776000")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.labelsize": 10.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#AAB7C2",
    "axes.linewidth": 0.8,
    "xtick.color": "#4E6070",
    "ytick.color": "#4E6070",
    "pdf.fonttype": 42,
    "svg.fonttype": "path",
    "svg.hashsalt": "endpoint-all-figures-r2",
})

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_all_figures_r2_20260919"
R1 = REPO / "reports/nns_vit_endpoint_all_figures_r1_20260919"
BASE = REPO / "reports/nns_vit_endpoint_unified_paper_figures_r2_20260918"
BLUE_ROOT = REPO / "reports/nns_vit_endpoint_fine_grained_blue_figures_r2_20260919"
MATH_ROOT = REPO / "reports/nns_vit_endpoint_math_comparison_r1_20260918"
M3_ROOT = REPO / "reports/nns_vit_endpoint_m3_sup_r2_20260919"
SOURCE_IMAGE = REPO / (
    "assets/data/nns_vit_endpoint_e2_evaluation_20260918/"
    "ILSVRC2012_val_00005415.JPEG"
)
PALETTE_CONFIG = REPO / "config/nns_vit_unified_paper_figures_r4.json"
E2_ATTEMPTS = {
    model: REPO / (
        "reports/nns_vit_endpoint_e2_atlas_r1_20260918/formal_results/attempts/"
        f"{model}__014/attempt-01.json"
    )
    for model in ("deit_s16", "vit_b16")
}
E3_ATTEMPT = REPO / (
    "reports/nns_vit_endpoint_e3_closure_r1_20260918/formal_results/attempts/"
    "deit_s16__014/attempt-01.json"
)
E5_ATTEMPT = REPO / (
    "reports/nns_vit_endpoint_e5_spatial_r1_20260918/formal_results/attempts/"
    "deit_s16__014/attempt-01.json"
)

INK = "#172033"
MUTED = "#637083"
BLUE = "#1178B4"
DEEP_BLUE = "#083B66"
CYAN = "#00AEE8"
GREEN = "#16966F"
ORANGE = "#D97904"
RED = "#C54438"
PURPLE = "#7564A8"
GRID = "#D9E1EA"
PANEL = "#F3F7FA"
WHITE = "#FFFFFF"

SOURCE_FILES: dict[str, dict[str, Any]] = {}
DISPLAY_RECORDS: list[dict[str, Any]] = []
MAIN_RECORDS: list[dict[str, Any]] = []
SUPPLEMENT_RECORDS: list[dict[str, Any]] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "canonical_digest": canonical(value)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def register(path: Path, role: str) -> None:
    resolved = path.resolve()
    relative = resolved.relative_to(REPO).as_posix()
    SOURCE_FILES[relative] = {
        "path": relative,
        "role": role,
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def read_json(path: Path, role: str) -> Any:
    register(path, role)
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_palette() -> LinearSegmentedColormap:
    config = read_json(PALETTE_CONFIG, "R10_blue_base_palette")
    return LinearSegmentedColormap.from_list(
        "endpoint_r10_blue",
        [(float(position), color) for position, color in config["display"]["heat_palette"]],
    )


def display_photo() -> np.ndarray:
    register(SOURCE_IMAGE, "sloth_bear_display_JPEG")
    with Image.open(SOURCE_IMAGE) as source:
        require(source.format == "JPEG", "DISPLAY_SOURCE_NOT_JPEG")
        rgb = source.convert("RGB")
        width, height = rgb.size
        if width < height:
            resized = rgb.resize((248, round(height * 248 / width)), Image.Resampling.BICUBIC)
        else:
            resized = rgb.resize((round(width * 248 / height), 248), Image.Resampling.BICUBIC)
        left = (resized.width - 224) // 2
        top = (resized.height - 224) // 2
        cropped = resized.crop((left, top, left + 224, top + 224))
    return np.asarray(cropped)


def load_e2(model_id: str) -> tuple[dict[str, Any], np.ndarray, dict[str, int]]:
    result = read_json(E2_ATTEMPTS[model_id], f"sealed_E2_{model_id}_sloth_bear")["result"]
    require(result["sample_id"] == SOURCE_IMAGE.name, f"E2_SAMPLE_MISMATCH:{model_id}")
    require(result["ground_truth_class_id"] == 297, f"E2_CLASS_MISMATCH:{model_id}")
    require(result["wnid"] == "n02134418", f"E2_WNID_MISMATCH:{model_id}")
    record = next(item for item in result["artifacts"] if item["kind"] == "full_raw_atlas")
    path = REPO / record["path"]
    register(path, f"sealed_E2_{model_id}_sloth_bear_full_atlas")
    require(sha256(path) == record["sha256"], f"E2_ATLAS_HASH_MISMATCH:{model_id}")
    array = np.load(path, mmap_mode="r")
    index = {row["target_id"]: i for i, row in enumerate(result["summaries"])}
    return result, array, index


def load_e3_block5() -> tuple[dict[str, Any], np.ndarray, dict[str, int]]:
    result = read_json(E3_ATTEMPT, "sealed_E3_sloth_bear_block_components")["result"]
    require(result["sample_id"] == SOURCE_IMAGE.name, "E3_SAMPLE_MISMATCH")
    record = next(
        item for item in result["artifacts"]
        if item.get("kind") == "raw_component_and_reconstruction_maps" and int(item["block"]) == 5
    )
    path = REPO / record["path"]
    register(path, "sealed_E3_sloth_bear_block5_components")
    require(sha256(path) == record["sha256"], "E3_BLOCK5_HASH_MISMATCH")
    array = np.load(path, mmap_mode="r")
    return result, array, {name: i for i, name in enumerate(record["component_order"])}


def load_e5_maps() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result = read_json(E5_ATTEMPT, "sealed_E5_sloth_bear_spatial_maps")["result"]
    require(result["sample_id"] == SOURCE_IMAGE.name, "E5_SAMPLE_MISMATCH")
    record = result["raw_maps_record"]
    path = REPO / record["path"]
    register(path, "sealed_E5_sloth_bear_raw_maps")
    require(sha256(path) == record["sha256"], "E5_MAP_HASH_MISMATCH")
    archive = np.load(path)
    return result, {name: np.asarray(archive[name]) for name in archive.files}


def collapse_rgb(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 4:
        require(array.shape[0] == 1, f"UNEXPECTED_BATCH:{array.shape}")
        array = array[0]
    if array.ndim == 3:
        require(array.shape[0] == 3, f"EXPECTED_RGB:{array.shape}")
        array = array.sum(axis=0)
    require(array.ndim == 2, f"EXPECTED_2D:{array.shape}")
    return array


def positive_view(value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = collapse_rgb(value)
    positive = np.maximum(raw, 0.0)
    limit = float(np.percentile(positive, 99.0))
    if limit <= 0:
        limit = float(positive.max())
    shown = np.zeros_like(positive) if limit <= 0 else np.clip(positive / limit, 0.0, 1.0)
    return shown, {
        "raw_sha256": array_sha256(raw),
        "shape": list(raw.shape),
        "signed_sum": float(raw.sum()),
        "absolute_mass": float(np.abs(raw).sum()),
        "positive_mass": float(positive.sum()),
        "display_limit": limit,
        "display_clipped_count": int((positive > limit).sum()) if limit > 0 else 0,
        "display_sha256": array_sha256(shown),
        "display_rule": "RGB signed sum; max(L,0); per-panel P99 with max fallback",
    }


def signed_view(value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = collapse_rgb(value)
    limit = float(np.max(np.abs(raw)))
    shown = np.zeros_like(raw) if limit <= 0 else np.sign(raw) * np.log1p(9 * np.abs(raw) / limit) / np.log(10)
    return shown, {
        "raw_sha256": array_sha256(raw),
        "shape": list(raw.shape),
        "display_limit": limit,
        "display_sha256": array_sha256(shown),
        "display_rule": "RGB signed sum; signed log1p kappa=9; max-absolute scaling",
    }


def clean_axis(ax: plt.Axes, *, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)


def figure_title(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.suptitle(title, x=0.04, y=0.975, ha="left", fontsize=20, weight="bold", color=INK)
    if subtitle:
        fig.text(0.04, 0.915, subtitle, ha="left", va="top", fontsize=10.8, color=MUTED)


def card(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str,
         body: str, color: str = BLUE, *, title_size: float = 11.0, body_size: float = 10.0) -> None:
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.025,rounding_size=0.055",
        linewidth=1.4, edgecolor=color, facecolor=PANEL,
    ))
    ax.text(x + 0.07 * width, y + 0.76 * height, title, ha="left", va="center",
            fontsize=title_size, color=color, weight="bold")
    ax.text(x + 0.07 * width, y + 0.39 * height, body, ha="left", va="center",
            fontsize=body_size, color=INK, linespacing=1.35)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.5, color=color))


def save_figure(fig: plt.Figure, number: int, slug: str, title: str, role: str) -> None:
    outputs = []
    for folder, extension, label in (
        ("figures_png", "png", "PNG"),
        ("figures_pdf", "pdf", "PDF"),
        ("figures_svg", "svg", "SVG"),
    ):
        path = ROOT / folder / f"figure_{number:02d}_{slug}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, facecolor=WHITE, bbox_inches="tight", pad_inches=0.08)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        outputs.append({
            "format": label, "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256(path),
        })
    plt.close(fig)
    MAIN_RECORDS.append({
        "number": number, "slug": slug, "title": title,
        "role": role, "kind": "R2_redesign", "outputs": outputs,
    })


def copy_main_figure(number: int, slug: str, title: str, role: str, source_stem: str) -> None:
    outputs = []
    for folder, extension, label in (
        ("figures_png", "png", "PNG"),
        ("figures_pdf", "pdf", "PDF"),
        ("figures_svg", "svg", "SVG"),
    ):
        source = BASE / folder / f"{source_stem}.{extension}"
        register(source, f"inherited_R2_figure_{number:02d}")
        target = ROOT / folder / f"figure_{number:02d}_{slug}.{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(sha256(source) == sha256(target), f"COPY_MISMATCH:{number}:{extension}")
        outputs.append({
            "format": label, "path": target.relative_to(ROOT).as_posix(),
            "bytes": target.stat().st_size, "sha256": sha256(target),
            "inherited_from": source.relative_to(REPO).as_posix(),
        })
    MAIN_RECORDS.append({
        "number": number, "slug": slug, "title": title,
        "role": role, "kind": "inherited_verified_figure", "outputs": outputs,
    })


def copy_blue_main(number: int, slug: str, title: str) -> None:
    outputs = []
    for folder, extension, label in (
        ("figures_png", "png", "PNG"),
        ("figures_pdf", "pdf", "PDF"),
        ("figures_svg", "svg", "SVG"),
    ):
        source = BLUE_ROOT / folder / f"figure_{number:02d}_{slug}.{extension}"
        register(source, f"inherited_blue_main_figure_{number:02d}")
        target = ROOT / folder / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(sha256(source) == sha256(target), f"BLUE_COPY_MISMATCH:{number}:{extension}")
        outputs.append({
            "format": label, "path": target.relative_to(ROOT).as_posix(),
            "bytes": target.stat().st_size, "sha256": sha256(target),
            "inherited_from": source.relative_to(REPO).as_posix(),
        })
    MAIN_RECORDS.append({
        "number": number, "slug": slug, "title": title,
        "role": "fine_grained_result", "kind": "inherited_blue_R2",
        "outputs": outputs,
    })


def save_signed_companion(fig: plt.Figure, number: int, slug: str, source_role: str) -> None:
    outputs = []
    for extension, label in (("png", "PNG"), ("pdf", "PDF"), ("svg", "SVG")):
        path = ROOT / "supplement/signed" / f"figure_{number:02d}_{slug}_signed.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, facecolor=WHITE, bbox_inches="tight", pad_inches=0.08)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        outputs.append({
            "format": label, "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256(path),
        })
    plt.close(fig)
    SUPPLEMENT_RECORDS.append({"group": "signed", "figure": number, "role": source_role, "outputs": outputs})


def save_real_output(name: str, shown: np.ndarray, cmap: Any, *, vmin: float, vmax: float) -> None:
    path = ROOT / "real_outputs" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, shown, cmap=cmap, vmin=vmin, vmax=vmax)


def heat_panel(ax: plt.Axes, value: np.ndarray, label: str, cmap: LinearSegmentedColormap,
               *, figure: int, target: str, signed_axis: plt.Axes | None = None) -> None:
    shown, receipt = positive_view(value)
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(label, fontsize=10.2, weight="bold", color=INK, pad=6)
    ax.axis("off")
    DISPLAY_RECORDS.append({"figure": figure, "target": target, "view": "positive_blue", **receipt})
    save_real_output(f"figure_{figure:02d}_{re.sub('[^a-z0-9]+', '_', target.lower()).strip('_')}_positive", shown, cmap, vmin=0, vmax=1)
    if signed_axis is not None:
        signed_shown, signed_receipt = signed_view(value)
        signed_axis.imshow(signed_shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        signed_axis.set_title(label, fontsize=10.2, weight="bold", color=INK, pad=6)
        signed_axis.axis("off")
        DISPLAY_RECORDS.append({"figure": figure, "target": target, "view": "signed", **signed_receipt})


def math_data() -> dict[str, Any]:
    return read_json(MATH_ROOT / "numeric_data.json", "verified_analytic_math_data")


def small_map(ax: plt.Axes, value: Any, title: str, limit: float, cmap: str = "RdBu_r") -> None:
    array = np.asarray(value, dtype=float).reshape(2, 2)
    ax.imshow(array, cmap=cmap, vmin=-limit, vmax=limit)
    for row in range(2):
        for column in range(2):
            color = WHITE if abs(array[row, column]) > 0.56 * limit else INK
            ax.text(column, row, f"{array[row, column]:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_title(title, fontsize=10, weight="bold", color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_figure_01(photo: np.ndarray, class_map: np.ndarray, cmap: LinearSegmentedColormap) -> plt.Figure:
    fig = plt.figure(figsize=(13.8, 7.2))
    figure_title(fig, "Endpoint Scanner: trace any ViT response back to the input",
                 "One scalar target in the real model → one input-coordinate learning image")
    grid = fig.add_gridspec(1, 3, left=0.045, right=0.97, bottom=0.18, top=0.82, width_ratios=[1.0, 1.4, 1.1], wspace=0.23)
    ax_photo = fig.add_subplot(grid[0, 0]); ax_photo.imshow(photo); ax_photo.axis("off")
    ax_photo.set_title("Real input · sloth bear", fontsize=12, weight="bold", color=INK, pad=10)
    ax_method = fig.add_subplot(grid[0, 1]); ax_method.set_xlim(0, 1); ax_method.set_ylim(0, 1); ax_method.axis("off")
    card(ax_method, (0.04, 0.61), 0.92, 0.25, "1 · Choose a scalar response",
         "Ground-truth class logit\ny = 297 · n02134418", BLUE, body_size=11)
    card(ax_method, (0.04, 0.29), 0.92, 0.25, "2 · Differentiate at the endpoint",
         r"$E_s(i)=x_i\,\partial s(x)/\partial x_i$", GREEN, body_size=15)
    ax_method.text(0.5, 0.13, "The same input coordinates are retained\nfor logits, heads, branches, and internal targets.",
                   ha="center", va="center", fontsize=10.5, color=MUTED)
    ax_map = fig.add_subplot(grid[0, 2])
    heat_panel(ax_map, class_map, "Ground-truth class logit", cmap, figure=1, target="class.gt")
    fig.text(0.5, 0.085, "input", color=MUTED, fontsize=10, ha="center")
    fig.text(0.722, 0.085, "Endpoint rule", color=MUTED, fontsize=10, ha="center")
    fig.text(0.91, 0.085, "where the positive response comes from", color=MUTED, fontsize=10, ha="center")
    fig.add_artist(FancyArrowPatch((0.29, 0.50), (0.35, 0.50), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, color=MUTED, lw=1.7))
    fig.add_artist(FancyArrowPatch((0.69, 0.50), (0.75, 0.50), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, color=BLUE, lw=1.7))
    fig.text(0.04, 0.025, "Main view: max(E, 0), per-panel P99 · full signed map retained in the supplement", fontsize=9.5, color=MUTED)
    return fig


def render_figure_02(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 7.0)); figure_title(fig, "Three related methods, three different questions", "Architecture supplies the Jacobian; the seed and evaluation rule define the attribution")
    ax = fig.add_axes([0.03, 0.07, 0.94, 0.78]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    columns = [
        (0.01, "Original NNS-CNN", GREEN, r"$x\odot\nabla s(x)$", "Current ReLU gates\nand MaxPool winner", "CNN endpoint rule"),
        (0.34, "ViT Endpoint Scanner", BLUE, r"$x\odot\nabla s(x)$", "Full ViT Jacobian: LN,\nQ/K/V, residual, MLP", "Our full-chain scan"),
        (0.67, "Path-integrated attribution", ORANGE, r"$\Delta x\odot\int_0^1\nabla s(x_\alpha)d\alpha$", "Average gradients along\na specified baseline path", "Different evaluation rule"),
    ]
    for x, title, color, formula, mechanism, footer in columns:
        ax.add_patch(FancyBboxPatch((x, 0.13), 0.30, 0.72, boxstyle="round,pad=.018,rounding_size=.025", edgecolor=color, facecolor=WHITE, lw=2))
        ax.add_patch(Rectangle((x, 0.74), 0.30, 0.11, color=color, alpha=0.10, ec="none"))
        ax.text(x + 0.15, 0.795, title, ha="center", va="center", fontsize=12, weight="bold", color=color)
        ax.text(x + 0.15, 0.60, formula, ha="center", va="center", fontsize=15, color=INK)
        ax.plot([x + 0.035, x + 0.265], [0.49, 0.49], color=GRID, lw=1.2)
        ax.text(x + 0.15, 0.39, mechanism, ha="center", va="center", fontsize=10.5, color=INK, linespacing=1.45)
        ax.text(x + 0.15, 0.21, footer, ha="center", va="center", fontsize=10, weight="bold", color=MUTED)
    ax.text(0.5, 0.035, "Endpoint identity does not imply completeness; Path IG additionally requires a baseline and a path.", ha="center", fontsize=10.5, color=MUTED)
    return fig


def render_figure_03(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 7.0)); figure_title(fig, "Same target, different evaluation rule", "The forward scalar is fixed; endpoint and path attribution differ after nonlinear attention")
    grid = fig.add_gridspec(2, 5, left=0.055, right=0.97, bottom=0.14, top=0.82, hspace=0.38, wspace=0.35)
    labels = [("x", 0, "input"), ("h", 0, "patch 1"), ("h", 1, "patch 2"), ("o", 0, "attention"), ("r", 0, "residual")]
    for column, (key, index, label) in enumerate(labels):
        small_map(fig.add_subplot(grid[0, column]), data["endpoint"][key][index], label, 3.0)
        small_map(fig.add_subplot(grid[1, column]), data["path"][key][index], label, 3.0)
    fig.text(0.018, 0.60, "Endpoint\ncurrent state", rotation=90, ha="center", va="center", fontsize=11, color=BLUE, weight="bold")
    fig.text(0.018, 0.29, "Path IG\npath average", rotation=90, ha="center", va="center", fontsize=11, color=ORANGE, weight="bold")
    fig.text(0.5, 0.055, r"$h=Px,\quad A=\mathrm{softmax}(hh^T),\quad o=Ah,\quad r=h+o$  ·  patch stage agrees; nonlinear attention separates the maps", ha="center", fontsize=11, color=MUTED)
    return fig


def render_figure_04(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 7.0)); figure_title(fig, "CNN endpoint rule: current gates and the winning route", "A compact analytic example isolates exactly what the original rule propagates")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [(0.02, "Input", "[1, 1; 3, 3]"), (0.27, "Conv + bias", "[-0.5, 1.5]"), (0.52, "ReLU", "[0, 1.5]"), (0.77, "MaxPool", "s = 1.5")]
    for x, title, body in stages:
        card(ax, (x, 0.62), 0.20, 0.22, title, body, GREEN, body_size=12)
    for x in (0.22, 0.47, 0.72): arrow(ax, (x, 0.73), (x + 0.045, 0.73), GREEN)
    map_ax = fig.add_axes([0.09, 0.13, 0.20, 0.28]); small_map(map_ax, data["cnn"]["endpoint"], "Endpoint map · Σ = 3.0", 1.5)
    path_ax = fig.add_axes([0.36, 0.13, 0.20, 0.28]); small_map(path_ax, data["cnn"]["path"], "Path IG · Σ = 1.5", 1.5)
    ax.text(0.67, 0.36, r"$s(\alpha x)=\max(3\alpha-1.5,0)$", fontsize=15, color=INK)
    ax.text(0.67, 0.27, "Endpoint uses the open gate at x.", fontsize=11, color=BLUE, weight="bold")
    ax.text(0.67, 0.19, "Path IG is zero before α = 0.5.", fontsize=11, color=ORANGE, weight="bold")
    ax.text(0.67, 0.10, "Same network · same output · different rule", fontsize=10.5, color=MUTED)
    return fig


def render_figure_05(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 7.0)); figure_title(fig, "Attention separates into Query, Key, and Value routes", r"$dO=J_{softmax}(S)[(dQ)K^T+Q(dK)^T]V+A(dV)$")
    grid = fig.add_gridspec(2, 4, left=0.06, right=0.96, bottom=0.12, top=0.79, wspace=0.30, hspace=0.38)
    routes = [("Q", "Query routing", PURPLE), ("K", "Key routing", ORANGE), ("V", "Value content", GREEN), ("full", "Full attention", BLUE)]
    for column, (key, label, color) in enumerate(routes):
        endpoint = data["endpoint_qkv"][key][0] if key != "full" else data["endpoint"]["o"][0]
        path = data["path_qkv"][key][0] if key != "full" else data["path"]["o"][0]
        small_map(fig.add_subplot(grid[0, column]), endpoint, label, 1.9)
        small_map(fig.add_subplot(grid[1, column]), path, label, 1.9)
    fig.text(0.018, 0.59, "Endpoint", rotation=90, ha="center", va="center", color=BLUE, weight="bold")
    fig.text(0.018, 0.28, "Path IG", rotation=90, ha="center", va="center", color=ORANGE, weight="bold")
    fig.text(0.5, 0.045, "Within each row: Query + Key + Value = full attention. Freezing A keeps only a conditional Value-path target.", ha="center", color=MUTED, fontsize=10.5)
    return fig


def render_figure_06(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 7.0)); figure_title(fig, "Residual accounting: every path crosses exactly one input edge", "Use one downstream cotangent at the add node; then identity + branch reconstructs the total")
    ax = fig.add_axes([0.04, 0.07, 0.92, 0.79]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    card(ax, (0.05, 0.58), 0.22, 0.23, "Identity edge", r"$E_{id}=x\odot J_h^T g_r$", BLUE, body_size=14)
    card(ax, (0.05, 0.22), 0.22, 0.23, "Attention edge", r"$E_{attn}=x\odot J_o^T g_r$", ORANGE, body_size=14)
    ax.text(0.34, 0.50, "+", fontsize=32, color=MUTED, ha="center", va="center")
    card(ax, (0.42, 0.39), 0.25, 0.25, "Residual total", r"$E_F=E_{id}+E_{attn}$", GREEN, body_size=15)
    arrow(ax, (0.28, 0.695), (0.42, 0.56)); arrow(ax, (0.28, 0.335), (0.42, 0.47))
    map_ax = fig.add_axes([0.73, 0.27, 0.20, 0.32]); small_map(map_ax, data["residual_final"]["total"], "Total input-coordinate map", 8.0)
    arrow(ax, (0.68, 0.515), (0.74, 0.515), GREEN)
    ax.text(0.5, 0.11, "Using the total gradient at h as the identity edge would count the attention route twice.", ha="center", fontsize=11, color=RED, weight="bold")
    return fig


def render_figure_07(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 7.0)); figure_title(fig, "The target definition changes the learning image", "A unit, a channel sum, and activation-weighted energy are three different scalar questions")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    h = np.asarray(data["values"]["h"]); unit = np.asarray(data["endpoint"]["h"])
    targets = [
        (0.02, "Single unit", r"$s=h_2$", unit[1], BLUE),
        (0.35, "Channel sum", r"$s=h_1+h_2$", unit.sum(0), GREEN),
        (0.68, "Feature-map energy", r"$s=\frac{1}{2}\sum_t h_t^2$", h @ unit, ORANGE),
    ]
    for x, title, formula, values, color in targets:
        ax.add_patch(FancyBboxPatch((x, 0.17), 0.30, 0.68, boxstyle="round,pad=.02,rounding_size=.03", edgecolor=color, facecolor=WHITE, lw=1.8))
        ax.text(x + 0.15, 0.77, title, ha="center", fontsize=12.5, weight="bold", color=color)
        ax.text(x + 0.15, 0.66, formula, ha="center", fontsize=15, color=INK)
        map_ax = fig.add_axes([0.065 + x * 0.92, 0.25, 0.18, 0.28]); small_map(map_ax, values, "Endpoint map", 4.5)
    ax.text(0.5, 0.055, "Target IDs must be fixed before looking at maps; otherwise the scientific question changes with the picture.", ha="center", color=MUTED, fontsize=10.5)
    return fig


def render_figure_08(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 7.0)); figure_title(fig, "State, seed, and coordinate origin", "Write all three explicitly before comparing attribution maps")
    ax = fig.add_axes([0.04, 0.07, 0.92, 0.79]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rows = [
        (0.67, "Canonical Endpoint", r"$L^{end}=x\odot\nabla s(x)$", "state: x · seed: x · no baseline forward", BLUE),
        (0.40, "Baseline-relative Endpoint", r"$L^{\Delta end}=\Delta x\odot\nabla s(x)$", "state: x · seed: x − x₀", GREEN),
        (0.13, "Integrated Gradients", r"$IG=\Delta x\odot\int_0^1\nabla s(x_\alpha)d\alpha$", "states: full path · seed: x − x₀", ORANGE),
    ]
    for y, title, formula, note, color in rows:
        ax.add_patch(FancyBboxPatch((0.03, y), 0.94, 0.19, boxstyle="round,pad=.018,rounding_size=.025", edgecolor=color, facecolor=WHITE, lw=1.6))
        ax.text(0.07, y + 0.13, title, fontsize=11.5, weight="bold", color=color)
        ax.text(0.34, y + 0.105, formula, fontsize=14.2, color=INK)
        ax.text(0.72, y + 0.095, note, fontsize=10.2, color=MUTED, ha="center")
    ax.text(0.5, 0.035, "Equal maps require the seeds and the evaluated Jacobians to agree—not just the same model name.", ha="center", fontsize=10.5, color=MUTED)
    return fig


def render_figure_09(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 7.0)); figure_title(fig, "ViT local operators need their real Jacobians", "LayerNorm couples channels; GELU is a continuous multiplier, not a binary gate")
    grid = fig.add_gridspec(1, 2, left=0.07, right=0.95, bottom=0.14, top=0.79, wspace=0.28)
    left = fig.add_subplot(grid[0, 0]); left.axis("off")
    left.text(0.02, 0.94, "LayerNorm", fontsize=14, weight="bold", color=BLUE, transform=left.transAxes)
    left.text(0.02, 0.78, r"$\frac{\partial y_i}{\partial u_j}=\frac{\gamma_i}{\sigma}[\delta_{ij}-\frac{1}{D}-\frac{\hat u_i\hat u_j}{D}]$", fontsize=15, color=INK, transform=left.transAxes)
    left.text(0.02, 0.59, "Mean and variance couple every input channel.\nFreezing them changes the map.", fontsize=11, color=MUTED, transform=left.transAxes, linespacing=1.5)
    map_full = fig.add_axes([0.12, 0.19, 0.14, 0.22]); small_map(map_full, data["ln"]["full_map"], "Full LN", 1.15)
    map_frozen = fig.add_axes([0.31, 0.19, 0.14, 0.22]); small_map(map_frozen, data["ln"]["frozen_map"], "Frozen statistics", 1.15)
    right = fig.add_subplot(grid[0, 1]);
    xx = np.linspace(-3, 3, 401); phi = np.exp(-xx * xx / 2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + np.vectorize(__import__("math").erf)(xx / np.sqrt(2)))
    derivative = cdf + xx * phi
    right.plot(xx, derivative, color=BLUE, lw=2.6, label="GELU′(u)")
    right.plot(xx, (xx > 0).astype(float), color=MUTED, lw=1.6, ls="--", label="binary ReLU gate")
    right.axhline(0, color=GRID, lw=0.8); right.set_xlabel("u"); right.set_ylabel("local multiplier")
    right.set_title(r"$GELU'(u)=\Phi(u)+u\phi(u)$", fontsize=14, color=ORANGE, pad=12)
    right.legend(frameon=False); clean_axis(right, grid="y")
    fig.text(0.5, 0.045, "These are analytic operator contracts, not new trained-model measurements.", ha="center", fontsize=10.5, color=MUTED)
    return fig


def render_figure_10(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 7.0)); figure_title(fig, "Structured cuts answer: which internal route carried the endpoint response?", "Each registered cut reconstructs one named total; cuts are not added across definitions")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    card(ax, (0.03, 0.41), 0.16, 0.22, "Input", r"$x$", BLUE, body_size=17)
    card(ax, (0.29, 0.63), 0.22, 0.20, "Attention cut", "heads · Q/K/V\nrouting + value", PURPLE)
    card(ax, (0.29, 0.21), 0.22, 0.20, "Residual cut", "identity + branch\ncarry + MLP", ORANGE)
    card(ax, (0.62, 0.41), 0.16, 0.22, "Target", r"$F$", GREEN, body_size=17)
    card(ax, (0.84, 0.41), 0.13, 0.22, "Map", r"$E_F$", BLUE, body_size=17)
    arrow(ax, (0.19, 0.52), (0.29, 0.73)); arrow(ax, (0.19, 0.52), (0.29, 0.31))
    arrow(ax, (0.51, 0.73), (0.62, 0.54)); arrow(ax, (0.51, 0.31), (0.62, 0.50)); arrow(ax, (0.78, 0.52), (0.84, 0.52), GREEN)
    ax.text(0.5, 0.09, r"$E^C_{k,i}=x_i\langle\nabla_{z_k}F,\partial_i z_k\rangle,\qquad \sum_k E^C_{k,i}=x_i\partial_iF$", ha="center", fontsize=15, color=INK)
    ax.text(0.5, 0.015, "Omission and duplication controls must fail; otherwise the cut is not structurally identified.", ha="center", fontsize=10.5, color=RED, weight="bold")
    return fig


def render_figure_11() -> plt.Figure:
    targets = {"ViT-B/16": [193, 70, 48, 144, 2], "DeiT-S/16": [193, 70, 48, 72, 2]}
    categories = ["Token channel", "Patch sum", "Energy", "Head", "Class"]
    colors = [DEEP_BLUE, CYAN, GREEN, ORANGE, RED]
    fig, ax = plt.subplots(figsize=(12.5, 6.7)); figure_title(fig, "Registered target atlas: broad coverage, fixed before viewing maps", "Targets per image across all 12 blocks and final class outputs")
    y = np.arange(2); left = np.zeros(2)
    for index, (category, color) in enumerate(zip(categories, colors)):
        values = np.array([targets[model][index] for model in targets])
        ax.barh(y, values, left=left, height=0.46, color=color, label=category)
        for row, value in enumerate(values):
            if value >= 35:
                ax.text(left[row] + value / 2, row, str(int(value)), ha="center", va="center", color=WHITE, fontsize=10, weight="bold")
        left += values
    ax.set_yticks(y, list(targets), fontsize=12, weight="bold"); ax.invert_yaxis(); ax.set_xlim(0, 500); ax.set_xlabel("registered scalar targets per image")
    for row, total in enumerate(left): ax.text(total + 9, row, f"{int(total)} total", va="center", fontsize=12, color=INK, weight="bold")
    ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28)); clean_axis(ax, grid="x")
    fig.subplots_adjust(left=0.18, right=0.93, top=0.77, bottom=0.24)
    return fig


def render_sloth_bear_gallery(number: int, title: str, subtitle: str, photo: np.ndarray,
                              panels: list[tuple[str, np.ndarray, str]], cmap: LinearSegmentedColormap,
                              columns: int) -> tuple[plt.Figure, plt.Figure]:
    count = len(panels); rows = int(np.ceil(count / columns))
    fig = plt.figure(figsize=(13.5, 3.25 * rows + 2.0)); signed_fig = plt.figure(figsize=(13.5, 3.25 * rows + 2.0))
    grid = fig.add_gridspec(rows, columns + 1, left=0.035, right=0.975, bottom=0.12, top=0.78, width_ratios=[0.82] + [1] * columns, wspace=0.12, hspace=0.28)
    sgrid = signed_fig.add_gridspec(rows, columns + 1, left=0.035, right=0.975, bottom=0.12, top=0.78, width_ratios=[0.82] + [1] * columns, wspace=0.12, hspace=0.28)
    for target_fig, target_grid in ((fig, grid), (signed_fig, sgrid)):
        photo_ax = target_fig.add_subplot(target_grid[:, 0]); photo_ax.imshow(photo); photo_ax.axis("off"); photo_ax.set_title("Sloth bear input", fontsize=11, weight="bold", color=INK)
    for index, (label, raw, target) in enumerate(panels):
        row, column = divmod(index, columns)
        heat_panel(fig.add_subplot(grid[row, column + 1]), raw, label, cmap, figure=number, target=target, signed_axis=signed_fig.add_subplot(sgrid[row, column + 1]))
    for index in range(count, rows * columns):
        row, column = divmod(index, columns); fig.add_subplot(grid[row, column + 1]).axis("off"); signed_fig.add_subplot(sgrid[row, column + 1]).axis("off")
    figure_title(fig, title, subtitle); figure_title(signed_fig, title + " · signed companion", subtitle)
    fig.text(0.60, 0.045, "Positive reading view · deep blue = zero/low · per-panel P99 · display only", ha="center", fontsize=9.5, color=MUTED)
    signed_fig.text(0.60, 0.045, "Signed reading view · blue < 0 · white = 0 · red > 0 · per-panel signed-log", ha="center", fontsize=9.5, color=MUTED)
    return fig, signed_fig


def render_figure_14() -> plt.Figure:
    numeric = read_json(BASE / "numeric_display_data.json", "published_endpoint_numeric_display_data")["figures"]["figure_14"]
    costs = numeric["cost"]; models = ["ViT-B/16", "DeiT-S/16"]; ids = ["vit_b16", "deit_s16"]
    fig = plt.figure(figsize=(12.8, 6.8)); figure_title(fig, "A complete registered atlas in under 11 seconds", "336,800 target evaluations · 400 images per model · measured complete-atlas latency")
    grid = fig.add_gridspec(1, 2, left=0.07, right=0.95, bottom=0.18, top=0.76, wspace=0.30)
    ax1 = fig.add_subplot(grid[0, 0]); values = [457, 385]; y = np.arange(2)
    ax1.hlines(y, 0, values, color=GRID, lw=8); ax1.scatter(values, y, s=170, color=[BLUE, GREEN], zorder=3)
    for i, value in enumerate(values): ax1.text(value + 12, i, str(value), va="center", fontsize=14, weight="bold", color=INK)
    ax1.set_yticks(y, models, fontsize=11, weight="bold"); ax1.invert_yaxis(); ax1.set_xlim(0, 520); ax1.set_xlabel("registered targets per image"); ax1.set_title("Atlas breadth", loc="left"); clean_axis(ax1, grid="x")
    ax2 = fig.add_subplot(grid[0, 1]); yy = np.arange(2); med = [costs[i]["median_seconds_per_complete_atlas"] for i in ids]; p95 = [costs[i]["p95_seconds_per_complete_atlas"] for i in ids]
    ax2.hlines(yy, med, p95, color=[BLUE, GREEN], lw=5, alpha=.55); ax2.scatter(med, yy, s=130, color=[BLUE, GREEN], label="median", zorder=3); ax2.scatter(p95, yy, s=90, facecolor=WHITE, edgecolor=[BLUE, GREEN], lw=2, label="P95", zorder=3)
    for i in range(2): ax2.text(p95[i] + .12, i, f"{med[i]:.2f}s / {p95[i]:.2f}s", va="center", fontsize=10.5, color=INK)
    ax2.set_yticks(yy, models, fontsize=11, weight="bold"); ax2.invert_yaxis(); ax2.set_xlim(0, 11.8); ax2.set_xlabel("seconds per complete atlas"); ax2.set_title("Median → P95 latency", loc="left"); clean_axis(ax2, grid="x")
    fig.text(.5, .06, "Runtime is reported as an engineering measurement, not a scientific success gate.", ha="center", color=MUTED, fontsize=10)
    return fig


def render_figure_16() -> plt.Figure:
    data = read_json(BASE / "numeric_display_data.json", "published_endpoint_numeric_display_data")["figures"]["figure_16"]
    fig = plt.figure(figsize=(12.8, 6.8)); figure_title(fig, "Structural closure passes at scale", "Positive reconstructions close; deliberately broken routes are rejected")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.76]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    card(ax, (0.02, 0.57), 0.29, 0.27, "Positive closure", f"{data['positive_closure_evaluations']:,} / {data['positive_closure_evaluations']:,} pass", GREEN, title_size=12, body_size=15)
    card(ax, (0.355, 0.57), 0.29, 0.27, "Negative controls", f"{data['negative_control_evaluations']:,} / {data['negative_control_evaluations']:,} rejected", RED, title_size=12, body_size=15)
    card(ax, (0.69, 0.57), 0.29, 0.27, "Task coverage", f"{data['image_model_tasks']} image-model tasks\n{data['block_evaluations']:,} block evaluations", BLUE, title_size=12, body_size=13)
    ratio = data["maximum_closure_relative_l2"] / 1e-4
    ax.text(0.02, 0.40, "Worst observed relative L2 versus frozen tolerance", fontsize=12, weight="bold", color=INK)
    ax.add_patch(FancyBboxPatch((0.02, 0.25), 0.86, 0.09, boxstyle="round,pad=.01,rounding_size=.025", facecolor=GRID, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((0.02, 0.25), 0.86 * ratio, 0.09, boxstyle="round,pad=.01,rounding_size=.025", facecolor=GREEN, edgecolor="none"))
    ax.axvline if False else None
    ax.text(0.02, 0.19, f"observed {data['maximum_closure_relative_l2']:.2e}", fontsize=11, color=GREEN, weight="bold")
    ax.text(0.88, 0.19, "tolerance 1.00e−4", fontsize=11, color=MUTED, ha="right")
    ax.text(0.92, 0.295, f"{ratio:.1%}", fontsize=14, color=GREEN, weight="bold", va="center")
    ax.text(0.5, 0.055, "Closure establishes structural correctness; it does not by itself establish intervention utility.", ha="center", fontsize=10.5, color=MUTED)
    return fig


def render_figure_18() -> plt.Figure:
    data = read_json(BASE / "numeric_display_data.json", "published_endpoint_numeric_display_data")["figures"]["figure_18"]
    fig = plt.figure(figsize=(12.8, 6.8)); figure_title(fig, "Analytic JVP checks remain inside the frozen numerical boundary", "Condition-scaled validation across every registered local component check")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.76]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    card(ax, (0.03, 0.57), 0.28, 0.27, "Coverage", f"{data['checks']:,} analytic JVP checks", BLUE, title_size=12, body_size=15)
    card(ax, (0.36, 0.57), 0.28, 0.27, "Validity", f"{data['checks']:,} / {data['checks']:,} pass", GREEN, title_size=12, body_size=15)
    card(ax, (0.69, 0.57), 0.28, 0.27, "Worst case", f"{100 * data['max_ratio']:.1f}% of its\ncondition-scaled threshold", ORANGE, title_size=12, body_size=13.2)
    ax.text(0.03, 0.40, "Maximum error-to-threshold ratio", fontsize=12, weight="bold", color=INK)
    ax.add_patch(FancyBboxPatch((0.03, 0.25), 0.86, 0.09, boxstyle="round,pad=.01,rounding_size=.025", facecolor=GRID, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((0.03, 0.25), 0.86 * data["max_ratio"], 0.09, boxstyle="round,pad=.01,rounding_size=.025", facecolor=GREEN, edgecolor="none"))
    ax.text(0.03, 0.19, "observed maximum", fontsize=10.5, color=GREEN, weight="bold")
    ax.text(0.89, 0.19, "failure boundary", fontsize=10.5, color=RED, ha="right", weight="bold")
    ax.text(0.93, 0.295, f"{data['max_ratio']:.3f}", fontsize=14, color=GREEN, weight="bold", va="center")
    ax.text(0.5, 0.055, "The check scales tolerance with the analytic condition; raw absolute errors are not compared across unlike components.", ha="center", fontsize=10.2, color=MUTED)
    return fig


def render_figure_22() -> plt.Figure:
    data = read_json(BASE / "numeric_display_data.json", "published_endpoint_numeric_display_data")["figures"]["figure_22"]
    matrix = np.asarray(data["matrix"], dtype=float)
    column_labels = ["Endpoint\nmass", "|p_ref|", "grad ×\nactivation", "output\nnorm", "gradient\nnorm", "random"]
    row_labels = []
    for row in data["rows"]:
        row_labels.append(row.replace("vit_b16", "ViT-B").replace("deit_s16", "DeiT-S").replace("normalized_zero", "zero").replace("blur_rgb", "blur").replace("routing_value", "routing/value").replace("residual_branches", "residual/branches"))
    fig, ax = plt.subplots(figsize=(12.8, 7.3)); figure_title(fig, "Which summaries predict finite replacement effects?", "Median per-image Spearman · Endpoint is the registered predictor; other columns are descriptive comparators")
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.3, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(column_labels)), column_labels, fontsize=9.5); ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]; ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=WHITE if abs(value) > .43 else INK, weight="bold" if j == 0 else "normal")
    ax.add_patch(Rectangle((-0.5, -0.5), 1, matrix.shape[0], fill=False, edgecolor=BLUE, lw=2.4))
    colorbar = fig.colorbar(image, ax=ax, fraction=.026, pad=.02); colorbar.set_label("median Spearman ρ")
    fig.subplots_adjust(left=.27, right=.92, top=.78, bottom=.13)
    fig.text(.5, .035, "The outlined Endpoint column alone enters the C4a gate; high descriptive correlations do not replace the registered test.", ha="center", color=MUTED, fontsize=10)
    return fig


def render_figure_25() -> plt.Figure:
    task = read_json(E5_ATTEMPT, "sealed_E5_sloth_bear_spatial_curves")["result"]
    require(task["sample_id"] == SOURCE_IMAGE.name, "E5_CURVE_SAMPLE_MISMATCH")
    areas = np.asarray(task["actual_area_fractions"], dtype=float)
    reference = task["references"]["normalized_zero"]
    series: dict[str, list[float]] = {}
    for method in ("endpoint_component", "total_gxi", "raw_gradient", "regional_taylor"):
        curves = [row for row in reference["curves"] if row["method"] == method]
        series[method] = np.mean([[abs(point["isolated_effect"]) for point in row["points"]] for row in curves], axis=0).tolist()
    series["random_regions"] = np.mean([row["absolute_of_mean_isolated_effect_curve"] for row in reference["random_aggregates"]], axis=0).tolist()
    styles = {
        "endpoint_component": (BLUE, "Endpoint component", "o"), "regional_taylor": (ORANGE, "Regional Taylor", "s"),
        "total_gxi": (GREEN, "Total G×I", "^"), "raw_gradient": (PURPLE, "Raw gradient", "D"),
        "random_regions": (MUTED, "Random regions", "x"),
    }
    fig, ax = plt.subplots(figsize=(12.4, 6.8)); figure_title(fig, "A single-image spatial intervention trace", "Sloth-bear task · normalized-zero reference · mean absolute isolated-head effect across three registered heads")
    for key, (color, label, marker) in styles.items():
        values = np.asarray(series[key], dtype=float); ax.plot(100 * areas, values, marker=marker, ms=7, lw=2.3 if key in {"endpoint_component", "regional_taylor"} else 1.5, color=color, label=label, alpha=1 if key in {"endpoint_component", "regional_taylor"} else .82)
    ax.set_xlabel("replaced input area (%)"); ax.set_ylabel("mean absolute isolated-head effect"); ax.set_xticks(100 * areas, ["0", "6.25", "12.5", "25"]); clean_axis(ax, grid="y")
    ax.legend(frameon=False, ncol=3, loc="upper center"); fig.subplots_adjust(left=.11, right=.96, top=.77, bottom=.16)
    fig.text(.5, .045, "One-image curves explain the AUC construction; cohort-level inference is reported in Figures 26–27.", ha="center", color=MUTED, fontsize=10)
    return fig


def compact_c4b_label(test_id: str) -> str:
    parts = test_id.replace("C4b.", "").split(".")
    require(len(parts) == 4 and parts[2] == "head_regions" and parts[3].startswith("vs_"), f"BAD_C4B_ID:{test_id}")
    model, reference, baseline = parts[0], parts[1], parts[3].removeprefix("vs_")
    return f"{'ViT-B' if model == 'vit_b16' else 'DeiT-S'} · {'zero' if reference == 'normalized_zero' else 'blur'} · {baseline.replace('_', ' ')}"


def render_figure_26() -> plt.Figure:
    rows = read_json(BASE / "numeric_display_data.json", "published_endpoint_numeric_display_data")["figures"]["figure_26"]
    rows = sorted(rows, key=lambda row: ("vit_b16" not in row["test_id"], "normalized_zero" not in row["test_id"], row["baseline"]))
    y = np.arange(len(rows)); fig, ax = plt.subplots(figsize=(12.8, 7.8)); figure_title(fig, "RQ4: Endpoint does not beat every mandatory spatial baseline", "Median paired AUC difference with 95% bootstrap intervals · registered gain threshold = +0.05")
    ax.axvspan(.05, .06, color=GREEN, alpha=.06); ax.axvline(0, color=INK, lw=1); ax.axvline(.05, color=RED, ls="--", lw=1.5)
    color_by_baseline = {"total_gxi": BLUE, "raw_gradient": PURPLE, "regional_taylor": ORANGE, "random_regions": GREEN}
    for i, row in enumerate(rows):
        color = color_by_baseline[row["baseline"]]; ax.errorbar(row["estimate"], i, xerr=[[row["estimate"] - row["ci_lower"]], [row["ci_upper"] - row["estimate"]]], fmt="o", color=color, ms=6, capsize=2, lw=1.7)
        ax.text(row["ci_upper"] + .0014, i, f"{row['estimate']:+.3f}", va="center", fontsize=8.4, color=color)
    for boundary in (3.5, 7.5, 11.5): ax.axhline(boundary, color=GRID, lw=1)
    ax.set_yticks(y, [compact_c4b_label(row["test_id"]) for row in rows], fontsize=8); ax.invert_yaxis(); ax.set_xlim(-.055, .062); ax.set_xlabel("Endpoint AUC − baseline AUC"); clean_axis(ax, grid="x")
    fig.subplots_adjust(left=.31, right=.93, top=.80, bottom=.11); fig.text(.5, .025, "No interval reaches the preregistered +0.05 practical-gain region.", ha="center", color=MUTED, fontsize=10)
    return fig


def render_figure_27() -> plt.Figure:
    data = read_json(BASE / "numeric_display_data.json", "published_endpoint_numeric_display_data")["figures"]["figure_27"]
    matrix = np.asarray(data["matrix"], dtype=float); rows = [value.replace("vit_b16", "ViT-B").replace("deit_s16", "DeiT-S").replace("normalized_zero", "zero").replace(".", " · ") for value in data["rows"]]; cols = [value.replace("_", " ") for value in data["columns"]]
    fig, ax = plt.subplots(figsize=(11.5, 6.5)); figure_title(fig, "RQ4 summary: paired gains remain near zero or negative", "Endpoint AUC minus each mandatory baseline; positive values favor Endpoint")
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-.04, vmax=.04, aspect="auto")
    ax.set_xticks(range(4), cols, fontsize=10); ax.set_yticks(range(4), rows, fontsize=10)
    for i in range(4):
        for j in range(4):
            value = matrix[i, j]; ax.text(j, i, f"{value:+.4f}", ha="center", va="center", fontsize=10, color=WHITE if abs(value) > .018 else INK, weight="bold")
    ax.add_patch(Rectangle((1.5, -.5), 1, 4, fill=False, edgecolor=ORANGE, lw=2.5))
    colorbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.03); colorbar.set_label("paired median difference")
    fig.subplots_adjust(left=.22, right=.91, top=.77, bottom=.15); fig.text(.5, .045, "Regional Taylor (outlined column) is higher in all four families; no family reaches +0.05.", ha="center", color=MUTED, fontsize=10)
    return fig


def csv_rows(path: Path, role: str) -> list[dict[str, str]]:
    register(path, role)
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def render_figure_29() -> plt.Figure:
    rows = csv_rows(M3_ROOT / "block_summary.csv", "M3_block_summary"); colors = {"vit_b16": BLUE, "deit_s16": ORANGE}
    metrics = [("entropy", "Spatial entropy"), ("locality", "Locality"), ("distance", "Normalized distance"), ("net_ratio", "Net / absolute mass")]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4)); figure_title(fig, "How Endpoint spatial structure changes across depth", "Image-level medians with 95% bootstrap intervals · 100 images per model")
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for model, color in colors.items():
            part = sorted((row for row in rows if row["model"] == model and row["metric"] == metric), key=lambda row: int(row["block"])); x = np.array([int(row["block"]) for row in part]); y = np.array([float(row["median"]) for row in part]); interval = np.array([json.loads(row["ci95"]) for row in part])
            ax.plot(x, y, "o-", color=color, lw=2, ms=4, label="ViT-B/16" if model == "vit_b16" else "DeiT-S/16"); ax.fill_between(x, interval[:, 0], interval[:, 1], color=color, alpha=.12)
        ax.set_title(label, loc="left"); ax.set_xticks([0, 3, 5, 8, 11]); clean_axis(ax, grid="y")
    axes[0, 0].legend(frameon=False); axes[1, 0].set_xlabel("block"); axes[1, 1].set_xlabel("block"); fig.subplots_adjust(left=.08, right=.96, top=.80, bottom=.10, hspace=.34, wspace=.24)
    return fig


def render_figure_30() -> plt.Figure:
    rows = csv_rows(M3_ROOT / "cls_patch_summary.csv", "M3_cls_patch_summary"); fig, ax = plt.subplots(figsize=(12.4, 6.8)); figure_title(fig, "CLS-linked maps become closer to patch maps with depth", "Paired image-level CLS-minus-patch entropy · median and 95% bootstrap interval")
    for model, color, label in (("vit_b16", BLUE, "ViT-B/16"), ("deit_s16", ORANGE, "DeiT-S/16")):
        part = [next(row for row in rows if row["model"] == model and row["stratum"] == f"block_{block:02d}") for block in range(12)]; values = np.array([float(row["median_difference"]) for row in part]); interval = np.array([json.loads(row["ci95"]) for row in part]); x = np.arange(12)
        ax.plot(x, values, "o-", color=color, lw=2.4, ms=5, label=label); ax.fill_between(x, interval[:, 0], interval[:, 1], color=color, alpha=.13)
        ax.text(11.15, values[-1], f"{values[-1]:.3f}", va="center", color=color, weight="bold")
    ax.axhline(0, color=INK, lw=.9); ax.set_xlim(-.2, 11.8); ax.set_xticks(range(12)); ax.set_xlabel("block"); ax.set_ylabel("CLS entropy − patch entropy"); ax.legend(frameon=False); clean_axis(ax, grid="y"); fig.subplots_adjust(left=.11, right=.94, top=.78, bottom=.15)
    return fig


def render_figure_31() -> plt.Figure:
    rows = csv_rows(M3_ROOT / "cancellation_summary.csv", "M3_cancellation_summary"); metrics = [("within_rgb_channels", "Within RGB"), ("within_patch_after_rgb_sum", "Within patch"), ("across_patch_signed", "Across patches"), ("whole_image_cancellation", "Whole image")]; colors = {"vit_b16": BLUE, "deit_s16": ORANGE}
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4)); figure_title(fig, "Cancellation accumulates across nested spatial scales", "Raw signed maps · patch-token targets · median cancellation ratio by block")
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for model, color in colors.items():
            part = sorted((row for row in rows if row["model"] == model and row["token_type"] == "patch" and row["metric"] == metric), key=lambda row: int(row["block"])); ax.plot([int(row["block"]) for row in part], [float(row["median"]) for row in part], "o-", color=color, lw=2, ms=4, label="ViT-B/16" if model == "vit_b16" else "DeiT-S/16")
        ax.set_title(label, loc="left"); ax.set_xticks([0, 5, 11]); ax.set_ylim(-.04, 1.04); clean_axis(ax, grid="y")
    axes[0, 0].legend(frameon=False); axes[1, 0].set_xlabel("block"); axes[1, 1].set_xlabel("block"); fig.subplots_adjust(left=.08, right=.96, top=.80, bottom=.10, hspace=.34, wspace=.24)
    return fig


def render_figure_33() -> plt.Figure:
    rows = csv_rows(M3_ROOT / "finite_difference_diagnostics.csv", "M3_finite_difference_diagnostics"); fig, ax = plt.subplots(figsize=(12.2, 6.8)); figure_title(fig, "Finite differences explain the retained float32 failures", "Median absolute FD–JVP error with 10th–90th percentile band across 36 cases")
    for dtype, color, label in (("float32", RED, "float32"), ("float64", BLUE, "float64")):
        by: dict[float, list[float]] = {}
        for row in rows:
            analytic = float(row[f"analytic_{dtype}"])
            for point in json.loads(row[f"{dtype}_curve"]): by.setdefault(float(point["epsilon"]), []).append(abs(float(point["finite_difference"]) - analytic))
        epsilon = np.array(sorted(by)); median = np.array([np.median(by[value]) for value in epsilon]); low = np.array([np.quantile(by[value], .1) for value in epsilon]); high = np.array([np.quantile(by[value], .9) for value in epsilon])
        ax.plot(epsilon, median, "o-", color=color, lw=2.3, ms=5, label=label); ax.fill_between(epsilon, low, high, color=color, alpha=.13)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("finite-difference ε"); ax.set_ylabel("absolute FD–JVP error"); ax.legend(frameon=False); clean_axis(ax, grid="both"); fig.subplots_adjust(left=.12, right=.96, top=.78, bottom=.16)
    fig.text(.5, .045, "Float64 follows the expected U-shaped numerical-error curve; the earlier float32 misses are precision effects.", ha="center", color=MUTED, fontsize=10)
    return fig


def short_family_label(family_id: str) -> str:
    value = family_id.replace("C4a.", "")
    model, reference, family = value.split(".", 2)
    return f"{'ViT-B' if model == 'vit_b16' else 'DeiT-S'} · {'zero' if reference == 'normalized_zero' else 'blur'} · {family.replace('_', '/')}"


def render_figure_20() -> plt.Figure:
    rows = read_json(BASE / "numeric_display_data.json", "published_endpoint_numeric_display_data")["figures"]["figure_20"]
    rows = sorted(rows, key=lambda row: ("vit_b16" not in row["id"], "normalized_zero" not in row["id"], row["id"]))
    y = np.arange(len(rows)); estimates = np.array([r["estimate"] for r in rows]); lo = np.array([r["ci"][0] for r in rows]); hi = np.array([r["ci"][1] for r in rows])
    fig, ax = plt.subplots(figsize=(12.8, 7.2)); figure_title(fig, "RQ3: Endpoint mass tracks finite replacement effects in 11 of 12 families", "Median image-level Spearman with 95% bootstrap intervals · registered practical threshold = 0.10")
    ax.axvspan(0.10, 0.33, color=GREEN, alpha=.055); ax.axvline(0.10, color=RED, ls="--", lw=1.5)
    for i, row in enumerate(rows):
        color = GREEN if row["status"] == "SUPPORTED" else ORANGE
        ax.errorbar(estimates[i], i, xerr=[[estimates[i] - lo[i]], [hi[i] - estimates[i]]], fmt="o", ms=7.5, capsize=3, color=color, lw=2)
        ax.text(hi[i] + .006, i, f"{estimates[i]:.3f}", va="center", fontsize=9.2, color=color, weight="bold")
    for boundary in (2.5, 5.5, 8.5): ax.axhline(boundary, color=GRID, lw=1)
    ax.set_yticks(y, [short_family_label(r["id"]) for r in rows], fontsize=9); ax.invert_yaxis(); ax.set_xlim(0.06, 0.34); ax.set_xlabel("median Spearman ρ")
    ax.text(0.102, -0.9, "registered support region", fontsize=9, color=GREEN, weight="bold")
    clean_axis(ax, grid="x"); fig.subplots_adjust(left=0.28, right=0.93, top=0.79, bottom=0.12)
    fig.text(.5, .025, "Orange marks the sole boundary exception; its direction and Holm-adjusted significance remain positive.", ha="center", fontsize=10, color=MUTED)
    return fig


def render_figure_21() -> plt.Figure:
    data = read_json(BASE / "numeric_display_data.json", "published_endpoint_numeric_display_data")["figures"]["figure_21"]
    test = data["exception"]["test_results"][0]
    fig = plt.figure(figsize=(12.8, 6.8)); figure_title(fig, "One family misses the practical threshold by 0.000435", "RQ3 is broadly supported, while the frozen all-12 omnibus field remains NOT_SUPPORTED")
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.76]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for index in range(12):
        x = 0.025 + index * 0.077
        color = GREEN if index < 11 else ORANGE
        ax.add_patch(FancyBboxPatch((x, 0.70), 0.057, 0.095, boxstyle="round,pad=.006,rounding_size=.012", facecolor=color, edgecolor="none"))
        ax.text(x + .0285, 0.747, str(index + 1), ha="center", va="center", color=WHITE, fontsize=9, weight="bold")
    ax.text(0.025, 0.63, "11 supported families", color=GREEN, fontsize=12, weight="bold")
    ax.text(0.72, 0.63, "1 boundary exception", color=ORANGE, fontsize=12, weight="bold")
    inset = fig.add_axes([0.12, 0.22, 0.76, 0.27])
    inset.axvspan(.10, .13, color=GREEN, alpha=.06); inset.axvline(.10, color=RED, ls="--", lw=1.6)
    inset.errorbar([test["estimate"]], [0], xerr=[[test["estimate"] - test["ci_lower"]], [test["ci_upper"] - test["estimate"]]], fmt="o", color=ORANGE, ms=9, capsize=5, lw=2.2)
    inset.set_xlim(.068, .128); inset.set_ylim(-.65, .65); inset.set_yticks([]); inset.set_xlabel("median Spearman ρ"); clean_axis(inset, grid="x")
    inset.text(test["estimate"], .28, "0.099565", ha="center", color=ORANGE, weight="bold")
    inset.text(.10, -.42, "threshold 0.10", ha="center", color=RED, fontsize=9)
    fig.text(.5, .075, "ViT-B/16 · normalized-zero reference · routing/value family", ha="center", fontsize=10, color=MUTED)
    return fig


def render_figure_32() -> plt.Figure:
    register(M3_ROOT / "m3a_error_distribution.csv", "M3_equivalence_error_distribution")
    with (M3_ROOT / "m3a_error_distribution.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    models = ["fcn", "cnn", "cnn_maxpool", "residual_mlp", "residual_cnn"]
    pretty = ["FCN", "CNN", "CNN + MaxPool", "Residual MLP", "Residual CNN"]
    values = {}
    for dtype in ("float32", "float64"):
        values[dtype] = [float(next(row for row in rows if row["group"] == "model_dtype" and row["key"] == f"{model}|{dtype}" and row["metric"] == "relative_l2")["q99"]) for model in models]
    fig, ax = plt.subplots(figsize=(12.4, 6.8)); figure_title(fig, "Original NNS and Endpoint agree to floating-point precision", "99th-percentile relative L2 across 500 cases per network × precision group")
    y = np.arange(len(models))
    for i in range(len(models)):
        ax.hlines(i, values["float64"][i], values["float32"][i], color=GRID, lw=4)
    ax.scatter(values["float32"], y, color=BLUE, s=105, label="float32", zorder=3)
    ax.scatter(values["float64"], y, facecolor=WHITE, edgecolor=ORANGE, linewidth=2.2, s=105, label="float64", zorder=3)
    for i, value in enumerate(values["float32"]): ax.text(value * 1.38, i, f"{value:.2e}", va="center", fontsize=9, color=BLUE, weight="bold")
    ax.set_xscale("log"); ax.set_xlim(8e-17, 6e-7); ax.set_yticks(y, pretty, fontsize=10.5, weight="bold"); ax.invert_yaxis(); ax.set_xlabel("99th-percentile relative L2 (log scale; lower is better)"); ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02)); clean_axis(ax, grid="x")
    fig.subplots_adjust(left=0.20, right=0.91, top=0.78, bottom=0.16)
    fig.text(.5, .04, "Float64 remains near machine precision; float32 discrepancies stay around 10⁻⁷.", ha="center", fontsize=10.5, color=MUTED)
    return fig


TITLES = {
    1: "Endpoint Scanner method overview", 2: "Endpoint, CNN, and Path method identity",
    3: "Same target: Endpoint versus Path", 4: "Original CNN endpoint gates",
    5: "Attention Q/K/V differential paths", 6: "Residual edge accounting",
    7: "Unit, sum, and energy targets", 8: "State, seed, and input coordinates",
    9: "LayerNorm and GELU Jacobians", 10: "Structured edge cuts",
    11: "Registered target atlas design", 12: "Sloth-bear class-logit maps across models",
    13: "Sloth-bear multi-target atlas", 14: "Scan scale, latency, and resources",
    15: "Structural closure contracts", 16: "Formal closure and negative controls",
    17: "Sloth-bear structural component maps", 18: "Condition-scaled analytic JVP checks",
    19: "Matched finite activation replacement", 20: "C4a family estimates and intervals",
    21: "RQ3 threshold zoom and boundary exception", 22: "Endpoint and registered secondary predictors",
    23: "Spatial intervention protocol", 24: "Sloth-bear block-5 maps for spatial ranking",
    25: "One-image spatial effect curves", 26: "C4b mandatory-baseline contrasts",
    27: "RQ4 family summary", 28: "Identity–Correctness–Utility conclusion",
    29: "M3 block trajectories", 30: "M3 CLS-minus-patch entropy by block",
    31: "M3 cancellation scales", 32: "Original-NNS versus Endpoint numerical equivalence",
    33: "M3 finite-difference diagnostics", 34: "Endpoint target atlas: real fine-grained learning images",
    35: "Block 5 structural Endpoint learning images", 36: "Block 5 maps used for spatial ranking",
    37: "Endpoint-NNSViT across depth", 38: "Endpoint-NNSViT across heads and depth",
    39: "ViT-B/16 Endpoint across heads and depth", 40: "Endpoint-NNSViT through the real module chain",
}

SLUGS = {number: re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", title.lower())).strip("_") for number, title in TITLES.items()}

BASE_STEMS = {
    15: "figure_15_structural_closure_contracts", 18: "figure_18_condition_scaled_analytic_jvp_checks",
    19: "figure_19_matched_finite_activation_replacement", 22: "figure_22_endpoint_and_registered_secondary_predictors",
    23: "figure_23_spatial_intervention_protocol", 25: "figure_25_one_image_spatial_effect_curves",
    26: "figure_26_c4b_mandatory_baseline_contrasts", 27: "figure_27_rq4_family_summary",
    28: "figure_28_identity_correctness_utility_conclusion",
}

M3_STEMS = {29: "m3_sup_block_trajectories", 30: "m3_sup_cls_patch_by_block", 31: "m3_sup_cancellation_scales", 33: "m3_sup_fd_diagnostics"}

BLUE_SLUGS = {34: "registered_target_learning_images", 35: "block5_structural_learning_images", 36: "spatial_ranking_learning_images", 37: "channel0_depth_progression", 38: "attention_head_depth_atlas", 39: "vit_b16_attention_head_depth_atlas", 40: "vit_b16_module_chain"}

REDESIGN_NOTES = {
    1: "Replaces the process-heavy overview with one real input, one scalar target, the endpoint formula, and the requested sloth-bear ground-truth class-logit map.",
    2: "Separates Original NNS-CNN, ViT Endpoint, and Path IG by rule, evaluated state, and scientific question.",
    3: "Keeps the target fixed and shows exactly where endpoint and path maps diverge after nonlinear attention.",
    4: "Shows the current ReLU gate and unique MaxPool winner without an overloaded derivation page.",
    5: "Organizes Q, K, V, and full-attention maps as a closure relation rather than a dense formula wall.",
    6: "Makes identity-plus-branch accounting and the double-counting failure mode visually explicit.",
    7: "Contrasts three scalar target definitions; the figure does not treat them as interchangeable visual variants.",
    8: "Places canonical Endpoint, baseline-relative Endpoint, and IG in a single state-seed-origin comparison.",
    9: "Pairs the LayerNorm Jacobian with the GELU derivative and removes unrelated analytic clutter.",
    10: "Reduces structured cuts to one graph, one component formula, and one closure condition.",
    11: "Uses directly labeled horizontal composition bars; counts and model differences are readable without legend hunting.",
    12: "Uses the fixed sloth-bear display sample for both models and the ground-truth class logit in the R10 blue-base positive view.",
    13: "Uses the same sloth-bear input and six preregistered targets, with blue-base positive main panels and a signed supplement.",
    14: "Recasts counts and latency as two labeled dot-range panels with the total workload in the subtitle.",
    16: "Uses three evidence cards and a tolerance utilization bar instead of incomparable raw-height bars.",
    17: "Uses sloth-bear block-5 raw component arrays in the blue-base positive reading view; signed maps remain supplemental.",
    18: "Summarizes all analytic JVP checks with coverage, pass count, and worst threshold utilization instead of an unreadable scatter cloud.",
    20: "Shortens family labels, groups rows, shades the registered support region, and labels estimates directly.",
    21: "Shows the 11-of-12 decision and the 0.000435 threshold miss without a redundant count bar.",
    22: "Retains the full predictor matrix while outlining the registered Endpoint column and clarifying that the remaining predictors are descriptive.",
    24: "Uses sloth-bear spatial-ranking arrays and a consistent blue-base positive reading view; signed maps remain supplemental.",
    25: "Uses the sloth-bear task and a clean emphasis hierarchy for the single-image regional intervention curves.",
    26: "Shortens all sixteen contrast labels, groups the four study families, and marks zero and the registered +0.05 region separately.",
    27: "Keeps the exact paired-difference matrix but highlights the regional-Taylor column that dominates all four families.",
    29: "Replots the four depth metrics with a common model palette, consistent intervals, and aligned block axes.",
    30: "Emphasizes the shared depth trend and directly labels the final-block differences.",
    31: "Uses aligned small multiples and one model legend for the four nested cancellation scales.",
    32: "Uses a horizontal log-scale dumbbell plot so float32 and float64 precision are comparable across five network families.",
    33: "Uses the same precision palette as Figure 32 and a clear log-log finite-difference diagnostic with interval bands.",
}


def copy_m3_main(number: int, stem: str) -> None:
    outputs = []
    for extension, label in (("png", "PNG"), ("pdf", "PDF")):
        source = M3_ROOT / f"{stem}.{extension}"; register(source, f"inherited_M3_figure_{number:02d}")
        target = ROOT / f"figures_{extension}" / f"figure_{number:02d}_{SLUGS[number]}.{extension}"; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        outputs.append({"format": label, "path": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size, "sha256": sha256(target), "inherited_from": source.relative_to(REPO).as_posix()})
    # The source package had no SVG; embed the verified PNG in a lightweight SVG wrapper.
    png_path = ROOT / "figures_png" / f"figure_{number:02d}_{SLUGS[number]}.png"
    with Image.open(png_path) as image:
        width, height = image.size
    svg_path = ROOT / "figures_svg" / f"figure_{number:02d}_{SLUGS[number]}.svg"; svg_path.parent.mkdir(parents=True, exist_ok=True)
    href = f"../figures_png/{png_path.name}"
    svg_path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><image width="{width}" height="{height}" href="{href}"/></svg>\n', encoding="utf-8")
    outputs.append({"format": "SVG", "path": svg_path.relative_to(ROOT).as_posix(), "bytes": svg_path.stat().st_size, "sha256": sha256(svg_path), "wrapper_for": png_path.relative_to(ROOT).as_posix()})
    MAIN_RECORDS.append({"number": number, "slug": SLUGS[number], "title": TITLES[number], "role": "supplementary_result", "kind": "inherited_verified_M3", "outputs": outputs})


def copy_supplements() -> None:
    manifest = read_json(R1 / "manifest.json", "R1_flat_collection_manifest")
    for record in manifest["records"]:
        group = record["group"]
        if group == "signed_companions":
            continue
        if group == "all_head_pages":
            match = re.fullmatch(r"all_heads_(deit_s16|vit_b16)_block_(\d{2})\.png", record["name"])
            require(match is not None, f"BAD_ALL_HEAD_NAME:{record['name']}")
            target = ROOT / "supplement/all_heads" / match.group(1) / f"block_{match.group(2)}.png"
        elif group.startswith("supp_"):
            target = ROOT / "supplement" / group.removeprefix("supp_") / record["name"]
        else:
            continue
        source = REPO / record["source_path"]; register(source, f"organized_{group}")
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        require(sha256(source) == sha256(target), f"SUPPLEMENT_COPY_MISMATCH:{record['name']}")
        SUPPLEMENT_RECORDS.append({"group": group, "source": source.relative_to(REPO).as_posix(), "path": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size, "sha256": sha256(target)})

    for number, slug in BLUE_SLUGS.items():
        source = BLUE_ROOT / "signed_companions_png" / f"figure_{number:02d}_{slug}_signed.png"
        register(source, f"blue_signed_companion_{number:02d}")
        target = ROOT / "supplement/signed" / source.name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        SUPPLEMENT_RECORDS.append({"group": "signed", "figure": number, "source": source.relative_to(REPO).as_posix(), "path": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size, "sha256": sha256(target)})


def build_docs() -> None:
    MAIN_RECORDS.sort(key=lambda record: record["number"])
    header = "[Gallery](gallery.html) · [Figure index](figure_index.md) · [逐图分析](figure_analysis.md)"
    index = ["# Endpoint-NNSViT organized R2 figure package", "", header, "", "| # | Figure | PNG | PDF | SVG |", "|---:|---|---|---|---|"]
    analysis = ["# Endpoint-NNSViT 逐图分析", "", header, ""]
    captions_en = ["# Endpoint-NNSViT Figure 1–40 captions", "", header, ""]
    captions_zh = ["# Endpoint-NNSViT Figure 1–40 中文图注", "", header, ""]
    cards = []
    for record in MAIN_RECORDS:
        number = record["number"]; links = {output["format"]: output["path"] for output in record["outputs"]}
        index.append(f"| {number} | {record['title']} | [PNG]({links['PNG']}) | [PDF]({links['PDF']}) | [SVG]({links['SVG']}) |")
        note = REDESIGN_NOTES.get(number, "Inherited byte-for-byte from a verified source package and reorganized under the main-figure directories.")
        boundary = "Blue-base maps are display-normalized shape views; raw arrays and signed companions, not display pixels, define scientific values." if number in {1, 12, 13, 17, 24, 34, 35, 36, 37, 38, 39, 40} else "The figure summarizes existing evidence and does not introduce a new cohort, test, or scientific decision."
        analysis += [f"## Image #{number}: {record['title']}", "", note, "", f"边界：{boundary}", "", f"![Figure {number}]({links['PNG']})", ""]
        captions_en += [f"## Figure {number}", "", f"{record['title']}. {boundary}", ""]
        captions_zh += [f"## Figure {number}", "", f"{record['title']}。{note}", ""]
        cards.append(f'<article id="figure-{number}"><h2>Figure {number} · {html.escape(record["title"])}</h2><a href="{links["PNG"]}"><img loading="lazy" src="{links["PNG"]}"></a><p>{html.escape(note)}</p><p><a href="{links["PNG"]}">PNG</a> · <a href="{links["PDF"]}">PDF</a> · <a href="{links["SVG"]}">SVG</a></p></article>')
    (ROOT / "figure_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (ROOT / "figure_analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    (ROOT / "captions_en.md").write_text("\n".join(captions_en) + "\n", encoding="utf-8")
    (ROOT / "captions_zh.md").write_text("\n".join(captions_zh) + "\n", encoding="utf-8")
    gallery = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Endpoint-NNSViT R2 figures</title><style>body{font:15px/1.55 system-ui;color:#172033;background:#edf3f7;max-width:1500px;margin:auto;padding:24px}nav{position:sticky;top:0;background:#edf3f7;padding:10px;z-index:2}article{background:white;padding:22px;margin:22px 0;border-radius:8px}img{width:100%;height:auto}a{color:#1178b4}</style><h1>Endpoint-NNSViT · organized R2 figure package</h1><p><a href="figure_index.md">Index</a> · <a href="figure_analysis.md">Image analysis</a></p><nav>' + " · ".join(f'<a href="#figure-{n}">{n}</a>' for n in range(1, 41)) + "</nav>" + "".join(cards) + "</html>"
    (ROOT / "gallery.html").write_text(gallery, encoding="utf-8")
    readme = f"""# Endpoint-NNSViT organized R2 figure package

{header}

This is the organized successor to `nns_vit_endpoint_all_figures_r1_20260919`. The archived R1 root is unchanged.

- `figures_png/`, `figures_pdf/`, `figures_svg/`: Figure 1–40 only.
- `supplement/signed/`: signed companions for the sloth-bear blue-base main views.
- `supplement/all_heads/`: full block-by-head pages, separated by model.
- `supplement/m2/`, `m3/`, `m4/`, `m4_review/`, `m6/`: supplemental families.
- `real_outputs/`: individual positive display panels used by redesigned real-map figures.
- `source_arrays/`: exact display arrays extracted from sealed sources plus their receipt.

The fixed presentation sample is `ILSVRC2012_val_00005415.JPEG` (selection order 14, class 297, `n02134418`, sloth bear / `Melursus ursinus`). Main real-map panels use the R10 blue-base rule `max(L,0)` with per-panel P99 scaling. Signed companions preserve the complete spatial sign pattern.

No model forward, checkpoint load, bootstrap, hypothesis test, or decision recomputation is performed.

Validation:

```bash
CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python scripts/validate_endpoint_all_figures_r2.py
```
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


def build_source_arrays(e2_deit: np.ndarray, deit_index: dict[str, int], e2_vit: np.ndarray, vit_index: dict[str, int], e3: np.ndarray, e3_index: dict[str, int], e5: dict[str, np.ndarray]) -> None:
    arrays = {
        "deit_class_gt": np.asarray(e2_deit[deit_index["class.gt"]]),
        "vit_class_gt": np.asarray(e2_vit[vit_index["class.gt"]]),
    }
    for target in ("b0.patch_channel_sum.c0", "b0.patch_channel_energy.c0", "b0.head0.cls.c0", "b5.head0.cls.c0", "b11.head5.cls.c0"):
        arrays[f"deit_{target.replace('.', '_')}"] = np.asarray(e2_deit[deit_index[target]])
    for component in ("identity", "head_0", "routing", "value", "attention", "mlp", "L_F"):
        arrays[f"e3_b5_{component}"] = np.asarray(e3[e3_index[component]])
    for name in ("total_gxi", "raw_gradient", "head_0", "head_3", "head_5"):
        arrays[f"e5_b5_{name}"] = np.asarray(e5[name])
    path = ROOT / "source_arrays/sloth_bear_display_arrays.npz"; path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrays)
    receipt = {
        "sample_id": SOURCE_IMAGE.name, "selection_order": 14, "class_id": 297, "wnid": "n02134418",
        "archive": {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)},
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "raw_bytes_sha256": array_sha256(value)} for name, value in arrays.items()},
        "display_policy": "RGB signed sum; main max(L,0); per-panel P99 with max fallback; R10 blue-base palette",
        "new_scientific_inference": False,
    }
    write_json(ROOT / "source_arrays/display_receipt.json", signed(receipt))


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"Use a fresh output root: {ROOT}")
    ROOT.mkdir(parents=True)
    register(Path(__file__), "R2_renderer")
    register(REPO / "scripts/validate_endpoint_all_figures_r2.py", "R2_validator")
    register(R1 / "manifest.json", "archived_R1_predecessor")
    cmap = load_palette(); photo = display_photo(); math = math_data()
    deit_result, deit_atlas, deit_index = load_e2("deit_s16"); vit_result, vit_atlas, vit_index = load_e2("vit_b16")
    _, e3_array, e3_index = load_e3_block5(); _, e5_maps = load_e5_maps()

    renderers: dict[int, Callable[[], plt.Figure]] = {
        1: lambda: render_figure_01(photo, deit_atlas[deit_index["class.gt"]], cmap),
        2: lambda: render_figure_02(math), 3: lambda: render_figure_03(math), 4: lambda: render_figure_04(math),
        5: lambda: render_figure_05(math), 6: lambda: render_figure_06(math), 7: lambda: render_figure_07(math),
        8: lambda: render_figure_08(math), 9: lambda: render_figure_09(math), 10: lambda: render_figure_10(math),
        11: render_figure_11, 14: render_figure_14, 16: render_figure_16, 18: render_figure_18,
        20: render_figure_20, 21: render_figure_21, 22: render_figure_22, 25: render_figure_25,
        26: render_figure_26, 27: render_figure_27, 29: render_figure_29, 30: render_figure_30,
        31: render_figure_31, 32: render_figure_32, 33: render_figure_33,
    }
    for number in range(1, 12): save_figure(renderers[number](), number, SLUGS[number], TITLES[number], "method" if number <= 10 else "study_design")

    fig12, signed12 = render_sloth_bear_gallery(12, TITLES[12], "Same input · same ground-truth class · two trained ViT variants", photo, [("DeiT-S/16 · class logit", deit_atlas[deit_index["class.gt"]], "deit_s16.class.gt"), ("ViT-B/16 · class logit", vit_atlas[vit_index["class.gt"]], "vit_b16.class.gt")], cmap, 2)
    save_figure(fig12, 12, SLUGS[12], TITLES[12], "real_map_result"); save_signed_companion(signed12, 12, SLUGS[12], "sloth_bear_cross_model_class_logit")
    target_panels = [
        ("B0 patch sum · c0", deit_atlas[deit_index["b0.patch_channel_sum.c0"]], "b0.patch_channel_sum.c0"),
        ("B0 patch energy · c0", deit_atlas[deit_index["b0.patch_channel_energy.c0"]], "b0.patch_channel_energy.c0"),
        ("B0 head 0 · CLS c0", deit_atlas[deit_index["b0.head0.cls.c0"]], "b0.head0.cls.c0"),
        ("B5 head 0 · CLS c0", deit_atlas[deit_index["b5.head0.cls.c0"]], "b5.head0.cls.c0"),
        ("B11 head 5 · CLS c0", deit_atlas[deit_index["b11.head5.cls.c0"]], "b11.head5.cls.c0"),
        ("Ground-truth class logit", deit_atlas[deit_index["class.gt"]], "class.gt"),
    ]
    fig13, signed13 = render_sloth_bear_gallery(13, TITLES[13], "DeiT-S/16 · six preregistered scalar targets · class 297", photo, target_panels, cmap, 3)
    save_figure(fig13, 13, SLUGS[13], TITLES[13], "real_map_result"); save_signed_companion(signed13, 13, SLUGS[13], "sloth_bear_registered_targets")
    save_figure(renderers[14](), 14, SLUGS[14], TITLES[14], "resource_summary")
    copy_main_figure(15, SLUGS[15], TITLES[15], "method_contract", BASE_STEMS[15])
    save_figure(renderers[16](), 16, SLUGS[16], TITLES[16], "correctness_result")
    component_panels = [(label, e3_array[e3_index[key]], f"b5.{key}") for label, key in (("Identity", "identity"), ("Head 0", "head_0"), ("Routing", "routing"), ("Value", "value"), ("Attention", "attention"), ("MLP", "mlp"), ("Complete Endpoint", "L_F"))]
    fig17, signed17 = render_sloth_bear_gallery(17, TITLES[17], "DeiT-S/16 · block 5 · same ground-truth class target · closure-validated arrays", photo, component_panels, cmap, 4)
    save_figure(fig17, 17, SLUGS[17], TITLES[17], "real_map_result"); save_signed_companion(signed17, 17, SLUGS[17], "sloth_bear_structural_components")
    save_figure(renderers[18](), 18, SLUGS[18], TITLES[18], "correctness_result")
    copy_main_figure(19, SLUGS[19], TITLES[19], "utility_method", BASE_STEMS[19])
    for number in (20, 21): save_figure(renderers[number](), number, SLUGS[number], TITLES[number], "RQ3_result")
    save_figure(renderers[22](), 22, SLUGS[22], TITLES[22], "utility_result")
    copy_main_figure(23, SLUGS[23], TITLES[23], "utility_method", BASE_STEMS[23])
    spatial_panels = [(label, e5_maps[key], key) for label, key in (("Total G×I", "total_gxi"), ("Raw gradient", "raw_gradient"), ("Head 0", "head_0"), ("Head 3", "head_3"), ("Head 5", "head_5"))]
    fig24, signed24 = render_sloth_bear_gallery(24, TITLES[24], "DeiT-S/16 · block 5 · maps used by the registered regional-ranking protocol", photo, spatial_panels, cmap, 3)
    save_figure(fig24, 24, SLUGS[24], TITLES[24], "real_map_result"); save_signed_companion(signed24, 24, SLUGS[24], "sloth_bear_spatial_ranking_maps")
    for number in (25, 26, 27): save_figure(renderers[number](), number, SLUGS[number], TITLES[number], "utility_result")
    copy_main_figure(28, SLUGS[28], TITLES[28], "utility_result", BASE_STEMS[28])
    for number in (29, 30, 31, 32, 33): save_figure(renderers[number](), number, SLUGS[number], TITLES[number], "supplementary_result" if number != 32 else "numerical_identity_result")
    for number, slug in BLUE_SLUGS.items(): copy_blue_main(number, slug, TITLES[number])

    copy_supplements()
    build_source_arrays(deit_atlas, deit_index, vit_atlas, vit_index, e3_array, e3_index, e5_maps)
    build_docs()
    write_json(ROOT / "display_receipt.json", signed({
        "sample_id": SOURCE_IMAGE.name, "selection_order": 14, "class_id": 297, "wnid": "n02134418",
        "formal_statistics_unchanged": True, "display_policy": "R10 blue-base max(L,0), per-panel P99",
        "panels": DISPLAY_RECORDS,
    }))
    sources = [SOURCE_FILES[key] for key in sorted(SOURCE_FILES)]
    write_json(ROOT / "source_inventory.json", signed({"cpu_only": True, "torch_timm_model_forward_calls": 0, "sources": sources, "sources_digest": canonical(sources)}))
    manifest_base = {
        "schema_version": 2, "revision": "ENDPOINT-ALL-FIGURES-R2", "status": "RENDERED_PENDING_MANUAL_VISUAL_QC",
        "organized_directory": True, "main_figure_count": len(MAIN_RECORDS), "main_export_count": sum(len(row["outputs"]) for row in MAIN_RECORDS),
        "main_figures": sorted(MAIN_RECORDS, key=lambda row: row["number"]), "supplemental_records": SUPPLEMENT_RECORDS,
        "display_sample_binding": {"sample_id": SOURCE_IMAGE.name, "selection_order": 14, "class_id": 297, "wnid": "n02134418", "class_name": "sloth bear, Melursus ursinus"},
        "redesigned_figures": sorted(REDESIGN_NOTES), "R1_preserved": True, "new_scientific_inference": False,
    }
    write_json(ROOT / "manifest.json", signed(manifest_base))
    write_json(ROOT / "manual_visual_qc.json", signed({"status": "PENDING", "records": [], "required_figures": sorted(REDESIGN_NOTES)}))
    print(json.dumps({"status": "RENDERED_PENDING_MANUAL_VISUAL_QC", "root": ROOT.relative_to(REPO).as_posix(), "figures": len(MAIN_RECORDS), "exports": sum(len(row["outputs"]) for row in MAIN_RECORDS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
