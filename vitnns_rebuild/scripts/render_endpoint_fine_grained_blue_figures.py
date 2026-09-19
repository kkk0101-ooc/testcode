#!/usr/bin/env python3
"""Render paper-facing fine-grained Endpoint maps from sealed raw arrays.

This is a CPU-only, presentation-only renderer. It does not import Torch or timm,
load a model, run a forward pass, recompute an attribution, or make a new
scientific inference. Main panels use the R10 blue-base positive reading view;
signed companions and raw-source hashes preserve the scientific boundary.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/endpoint-fine-grained-blue-mpl")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1789776000")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

matplotlib.rcParams["svg.hashsalt"] = "endpoint-fine-grained-blue-r1"


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_fine_grained_blue_figures_r1_20260919"
PAPER_MANIFEST = REPO / "papers/nns_vit_endpoint/figure_manifest_fine_grained_blue_r1.json"
E2_ATTEMPTS = {
    model_id: REPO / (
        "reports/nns_vit_endpoint_e2_atlas_r1_20260918/formal_results/attempts/"
        f"{model_id}__000/attempt-01.json"
    )
    for model_id in ("deit_s16", "vit_b16")
}
E3_ATTEMPT = REPO / (
    "reports/nns_vit_endpoint_e3_closure_r1_20260918/formal_results/attempts/"
    "deit_s16__000/attempt-01.json"
)
E5_ATTEMPT = REPO / (
    "reports/nns_vit_endpoint_e5_spatial_r1_20260918/formal_results/attempts/"
    "deit_s16__000/attempt-01.json"
)
SOURCE_IMAGE = REPO / (
    "assets/data/nns_vit_endpoint_e2_evaluation_20260918/"
    "ILSVRC2012_val_00024828.JPEG"
)
PALETTE_CONFIG = REPO / "config/nns_vit_unified_paper_figures_r4.json"

INK = "#172033"
MUTED = "#637083"
BLUE = "#1178B4"
SOURCE_FILES: dict[str, dict[str, Any]] = {}
DISPLAY_PANELS: list[dict[str, Any]] = []
FIGURES: list[dict[str, Any]] = []
SUPPLEMENTAL_HEAD_PAGES: list[dict[str, Any]] = []


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
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
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
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
        "role": role,
    }


def load_json(path: Path, role: str) -> dict[str, Any]:
    register(path, role)
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_palette() -> LinearSegmentedColormap:
    config = load_json(PALETTE_CONFIG, "frozen_R10_display_palette")
    stops = [(float(position), color) for position, color in config["display"]["heat_palette"]]
    return LinearSegmentedColormap.from_list("r10_blue_base", stops)


def display_photo() -> np.ndarray:
    register(SOURCE_IMAGE, "fixed_first_endpoint_evaluation_JPEG")
    with Image.open(SOURCE_IMAGE) as source:
        require(source.format == "JPEG", "SOURCE_IMAGE_NOT_JPEG")
        rgb = source.convert("RGB")
        width, height = rgb.size
        if width < height:
            resized_size = (248, int(round(height * 248 / width)))
        else:
            resized_size = (int(round(width * 248 / height)), 248)
        resized = rgb.resize(resized_size, resample=Image.Resampling.BICUBIC)
        left = (resized.width - 224) // 2
        top = (resized.height - 224) // 2
        cropped = resized.crop((left, top, left + 224, top + 224))
    return np.asarray(cropped)


def load_e2_atlas(model_id: str) -> tuple[dict[str, Any], np.ndarray, dict[str, int]]:
    attempt_path = E2_ATTEMPTS[model_id]
    result = load_json(attempt_path, f"sealed_E2_{model_id}_fixed_first_task")["result"]
    require(result["model_id"] == model_id, f"E2_MODEL_ID_MISMATCH:{model_id}")
    require(result["sample_id"] == SOURCE_IMAGE.name, f"E2_SAMPLE_ID_MISMATCH:{model_id}")
    atlas_record = next(item for item in result["artifacts"] if item["kind"] == "full_raw_atlas")
    atlas_path = REPO / atlas_record["path"]
    register(atlas_path, f"sealed_E2_{model_id}_full_raw_atlas")
    require(
        SOURCE_FILES[atlas_record["path"]]["sha256"] == atlas_record["sha256"],
        f"E2_ATLAS_SHA256_MISMATCH:{model_id}",
    )
    atlas = np.load(atlas_path, mmap_mode="r")
    require(list(atlas.shape) == atlas_record["shape"], f"E2_ATLAS_SHAPE_MISMATCH:{model_id}")
    target_index = {row["target_id"]: index for index, row in enumerate(result["summaries"])}
    return result, atlas, target_index


def load_e3_head_maps(model_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    attempt_path = REPO / (
        "reports/nns_vit_endpoint_e3_closure_r1_20260918/formal_results/attempts/"
        f"{model_id}__000/attempt-01.json"
    )
    result = load_json(attempt_path, f"sealed_E3_{model_id}_fixed_first_task")["result"]
    require(result["model_id"] == model_id, f"E3_MODEL_ID_MISMATCH:{model_id}")
    require(result["sample_id"] == SOURCE_IMAGE.name, f"E3_SAMPLE_ID_MISMATCH:{model_id}")
    maps: dict[str, np.ndarray] = {}
    for record in result["artifacts"]:
        if record.get("kind") != "raw_component_and_reconstruction_maps":
            continue
        block = int(record["block"])
        path = REPO / record["path"]
        register(path, f"sealed_E3_{model_id}_block_{block:02d}_component_maps")
        require(
            SOURCE_FILES[record["path"]]["sha256"] == record["sha256"],
            f"E3_COMPONENT_SHA256_MISMATCH:{model_id}:{block}",
        )
        array = np.load(path, mmap_mode="r")
        require(list(array.shape) == record["shape"], f"E3_COMPONENT_SHAPE_MISMATCH:{model_id}:{block}")
        component_index = {name: index for index, name in enumerate(record["component_order"])}
        head_count = 12 if model_id == "vit_b16" else 6
        for head in range(head_count):
            maps[f"b{block}.head{head}.class_gt"] = np.array(array[component_index[f"head_{head}"]])
    expected = 144 if model_id == "vit_b16" else 72
    require(len(maps) == expected, f"E3_HEAD_MAP_COVERAGE:{model_id}:{len(maps)}")
    return result, maps


def collapse_rgb(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 4:
        require(array.shape[0] == 1, f"UNEXPECTED_BATCH_SHAPE:{array.shape}")
        array = array[0]
    require(array.ndim == 3 and array.shape[0] == 3, f"EXPECTED_RGB_MAP:{array.shape}")
    return array.sum(axis=0)


def positive_view(value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = collapse_rgb(value)
    positive = np.maximum(raw, 0.0)
    limit = float(np.percentile(positive, 99.0))
    if limit <= 0:
        limit = float(positive.max())
    if limit <= 0:
        shown = np.zeros_like(positive)
        limit = 0.0
    else:
        shown = np.clip(positive / limit, 0.0, 1.0)
    return shown, {
        "raw_signed_2d_sha256": array_sha256(raw),
        "raw_rgb_sha256": array_sha256(np.asarray(value)),
        "raw_shape": list(np.asarray(value).shape),
        "signed_sum": float(raw.sum()),
        "absolute_mass": float(np.abs(raw).sum()),
        "positive_mass": float(positive.sum()),
        "display_limit": limit,
        "display_clipped_count": int((positive > limit).sum()) if limit else 0,
        "display_sha256": array_sha256(shown),
        "display_rule": "RGB signed sum, max(L,0), per-panel P99 with max fallback, clip to [0,1]",
    }


def signed_view(value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = collapse_rgb(value)
    limit = float(np.max(np.abs(raw)))
    if limit <= 0:
        shown = np.zeros_like(raw)
        limit = 0.0
    else:
        shown = np.sign(raw) * np.log1p(9.0 * np.abs(raw) / limit) / np.log(10.0)
    return shown, {
        "raw_signed_2d_sha256": array_sha256(raw),
        "display_limit": limit,
        "display_sha256": array_sha256(shown),
        "display_rule": "RGB signed sum, signed log1p kappa=9, max-absolute scaling, no clipping",
    }


def render_grid(
    *,
    number: int,
    slug: str,
    title: str,
    subtitle: str,
    photo: np.ndarray,
    panels: list[tuple[str, np.ndarray, str]],
    cmap: LinearSegmentedColormap,
) -> tuple[plt.Figure, plt.Figure]:
    count = len(panels)
    columns = 4 if count >= 6 else 3
    rows = int(np.ceil(count / columns))
    figure = plt.figure(figsize=(12, 3.55 * rows + 1.6))
    grid = figure.add_gridspec(
        rows, columns + 1, left=0.035, right=0.985, bottom=0.11, top=0.80,
        width_ratios=[0.86] + [1.0] * columns, wspace=0.13, hspace=0.28,
    )
    photo_axis = figure.add_subplot(grid[:, 0])
    photo_axis.imshow(photo)
    photo_axis.set_title("Fixed input", fontsize=11, weight="bold", pad=8)
    photo_axis.axis("off")
    signed_figure = plt.figure(figsize=(12, 3.55 * rows + 1.6))
    signed_grid = signed_figure.add_gridspec(
        rows, columns + 1, left=0.035, right=0.985, bottom=0.11, top=0.80,
        width_ratios=[0.86] + [1.0] * columns, wspace=0.13, hspace=0.28,
    )
    signed_photo_axis = signed_figure.add_subplot(signed_grid[:, 0])
    signed_photo_axis.imshow(photo)
    signed_photo_axis.set_title("Fixed input", fontsize=11, weight="bold", pad=8)
    signed_photo_axis.axis("off")

    for index, (label, raw, target_id) in enumerate(panels):
        row, column = divmod(index, columns)
        axis = figure.add_subplot(grid[row, column + 1])
        shown, receipt = positive_view(raw)
        axis.imshow(shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(label, fontsize=10.3, weight="bold", pad=6)
        axis.axis("off")
        DISPLAY_PANELS.append({
            "figure": number, "target_id": target_id, "view": "positive_blue", **receipt
        })

        signed_axis = signed_figure.add_subplot(signed_grid[row, column + 1])
        signed_shown, signed_receipt = signed_view(raw)
        signed_axis.imshow(
            signed_shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
        )
        signed_axis.set_title(label, fontsize=10.3, weight="bold", pad=6)
        signed_axis.axis("off")
        DISPLAY_PANELS.append({
            "figure": number, "target_id": target_id, "view": "signed_companion",
            **signed_receipt,
        })

    total_slots = rows * columns
    for index in range(count, total_slots):
        row, column = divmod(index, columns)
        figure.add_subplot(grid[row, column + 1]).axis("off")
        signed_figure.add_subplot(signed_grid[row, column + 1]).axis("off")

    figure.suptitle(title, fontsize=19, weight="bold", y=0.965, color=INK)
    figure.text(0.59, 0.865, subtitle, ha="center", fontsize=10.5, color=MUTED)
    figure.text(
        0.59, 0.035,
        "Positive reading view · deep blue = zero/low · hot colors = higher positive value · per-panel P99",
        ha="center", fontsize=9.5, color=MUTED,
    )
    signed_figure.suptitle(title + " · signed companion", fontsize=19, weight="bold", y=0.965, color=INK)
    signed_figure.text(0.59, 0.865, subtitle, ha="center", fontsize=10.5, color=MUTED)
    signed_figure.text(
        0.59, 0.035,
        "Signed reading view · blue < 0, white = 0, red > 0 · per-panel signed-log display",
        ha="center", fontsize=9.5, color=MUTED,
    )
    return figure, signed_figure


def patch_absolute(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 4:
        require(array.shape[0] == 1, f"UNEXPECTED_BATCH_SHAPE:{array.shape}")
        array = array[0]
    require(array.shape == (3, 224, 224), f"EXPECTED_RGB_MAP:{array.shape}")
    return np.abs(array).sum(axis=0).reshape(14, 16, 14, 16).sum(axis=(1, 3))


def block_progression_figure(
    photo: np.ndarray, atlas: np.ndarray, target_index: dict[str, int], cmap: LinearSegmentedColormap
) -> tuple[plt.Figure, plt.Figure]:
    panels = [("Patch embed", "patch.sum.c0")] + [
        (f"Block {block}", f"b{block}.patch_channel_sum.c0") for block in range(12)
    ] + [("Final CLS c0", "final.cls.c0"), ("Class logit", "class.gt")]
    figure, axes = plt.subplots(2, len(panels), figsize=(19.0, 4.9))
    signed_figure, signed_axes = plt.subplots(1, len(panels), figsize=(19.0, 2.75))
    for column, (label, target_id) in enumerate(panels):
        raw = np.asarray(atlas[target_index[target_id]])
        shown, receipt = positive_view(raw)
        axes[0, column].imshow(shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[0, column].set_title(label, fontsize=8.2, weight="bold", pad=5)
        axes[0, column].axis("off")
        patch = patch_absolute(raw)
        patch_limit = float(np.percentile(patch, 99.0)) or float(patch.max()) or 1.0
        patch_shown = np.clip(patch / patch_limit, 0.0, 1.0)
        axes[1, column].imshow(patch_shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[1, column].axis("off")
        DISPLAY_PANELS.append({
            "figure": 37, "target_id": target_id, "view": "positive_blue", **receipt
        })
        DISPLAY_PANELS.append({
            "figure": 37, "target_id": target_id, "view": "absolute_patch_mass",
            "display_limit": patch_limit, "display_clipped_count": int((patch > patch_limit).sum()),
            "display_sha256": array_sha256(patch_shown),
            "display_rule": "sum absolute RGB mass within each native 16x16 patch; per-panel P99",
        })
        signed_shown, signed_receipt = signed_view(raw)
        signed_axes[column].imshow(
            signed_shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
        )
        signed_axes[column].set_title(label, fontsize=8.2, weight="bold", pad=5)
        signed_axes[column].axis("off")
        DISPLAY_PANELS.append({
            "figure": 37, "target_id": target_id, "view": "signed_companion",
            **signed_receipt,
        })
    axes[0, 0].text(
        -0.18, 0.5, "Learning image\n224 × 224 input coordinates", transform=axes[0, 0].transAxes,
        ha="right", va="center", fontsize=9, rotation=90, color=INK,
    )
    axes[1, 0].text(
        -0.18, 0.5, "Endpoint patch mass\n14 × 14 input regions", transform=axes[1, 0].transAxes,
        ha="right", va="center", fontsize=9, rotation=90, color=INK,
    )
    figure.suptitle(
        "Endpoint-NNSViT across depth: channel-0 learning images", fontsize=18, weight="bold", y=0.98, color=INK
    )
    figure.text(
        0.5, 0.91,
        "Fixed DeiT-S/16 input · patch-channel-sum target through all 12 block outputs · final targets appended",
        ha="center", fontsize=10, color=MUTED,
    )
    figure.text(
        0.5, 0.025,
        "Top: positive Endpoint reading view · Bottom: absolute Endpoint mass pooled over native input patches (not a forward feature map)",
        ha="center", fontsize=9.3, color=MUTED,
    )
    signed_figure.suptitle(
        "Endpoint-NNSViT across depth: channel-0 signed companions", fontsize=15, weight="bold", y=0.98, color=INK
    )
    signed_figure.text(
        0.5, 0.03, "Blue < 0, white = 0, red > 0 · per-panel signed-log display",
        ha="center", fontsize=9, color=MUTED,
    )
    return figure, signed_figure


def head_atlas_figure(
    photo: np.ndarray, head_maps: dict[str, np.ndarray], _unused: dict[str, int], cmap: LinearSegmentedColormap
) -> tuple[plt.Figure, plt.Figure]:
    blocks = (0, 1, 5, 11)
    heads = range(6)
    figure = plt.figure(figsize=(13.7, 9.0))
    grid = figure.add_gridspec(
        len(blocks), len(heads) + 1, left=0.035, right=0.985, bottom=0.075, top=0.86,
        width_ratios=[0.90] + [1.0] * len(heads), wspace=0.055, hspace=0.19,
    )
    signed_figure = plt.figure(figsize=(13.7, 9.0))
    signed_grid = signed_figure.add_gridspec(
        len(blocks), len(heads) + 1, left=0.035, right=0.985, bottom=0.075, top=0.86,
        width_ratios=[0.90] + [1.0] * len(heads), wspace=0.055, hspace=0.19,
    )
    for row, block in enumerate(blocks):
        image_axis = figure.add_subplot(grid[row, 0])
        image_axis.imshow(photo)
        image_axis.axis("off")
        image_axis.set_ylabel(f"Block {block}", fontsize=11, weight="bold", rotation=90, labelpad=5)
        signed_image_axis = signed_figure.add_subplot(signed_grid[row, 0])
        signed_image_axis.imshow(photo)
        signed_image_axis.axis("off")
        signed_image_axis.set_ylabel(f"Block {block}", fontsize=11, weight="bold", rotation=90, labelpad=5)
        for head in heads:
            target_id = f"b{block}.head{head}.class_gt"
            raw = head_maps[target_id]
            axis = figure.add_subplot(grid[row, head + 1])
            shown, receipt = positive_view(raw)
            axis.imshow(shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            axis.axis("off")
            if row == 0:
                axis.set_title(f"Head {head}", fontsize=10.5, weight="bold", pad=6)
            DISPLAY_PANELS.append({
                "figure": 38, "target_id": target_id, "view": "positive_blue", **receipt
            })
            signed_axis = signed_figure.add_subplot(signed_grid[row, head + 1])
            signed_shown, signed_receipt = signed_view(raw)
            signed_axis.imshow(
                signed_shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
            )
            signed_axis.axis("off")
            if row == 0:
                signed_axis.set_title(f"Head {head}", fontsize=10.5, weight="bold", pad=6)
            DISPLAY_PANELS.append({
                "figure": 38, "target_id": target_id, "view": "signed_companion",
                **signed_receipt,
            })
    figure.suptitle(
        "Endpoint-NNSViT across heads and depth", fontsize=19, weight="bold", y=0.965, color=INK
    )
    figure.text(
        0.55, 0.91, "Fixed DeiT-S/16 input · all head-edge contributions to the same ground-truth class logit",
        ha="center", fontsize=10.5, color=MUTED,
    )
    figure.text(
        0.55, 0.025, "Positive Endpoint reading view · rows are fixed blocks · columns are all six heads",
        ha="center", fontsize=9.3, color=MUTED,
    )
    signed_figure.suptitle(
        "Endpoint-NNSViT across heads and depth · signed companions",
        fontsize=19, weight="bold", y=0.965, color=INK,
    )
    signed_figure.text(
        0.55, 0.025, "Blue < 0, white = 0, red > 0 · per-panel signed-log display",
        ha="center", fontsize=9.3, color=MUTED,
    )
    return figure, signed_figure


def vit_head_depth_figure(
    photo: np.ndarray, head_maps: dict[str, np.ndarray], _unused: dict[str, int], cmap: LinearSegmentedColormap
) -> tuple[plt.Figure, plt.Figure]:
    blocks = (0, 1, 5, 11)
    heads = range(12)
    figure = plt.figure(figsize=(20.0, 7.4))
    grid = figure.add_gridspec(
        len(blocks), len(heads) + 1, left=0.025, right=0.99, bottom=0.08, top=0.84,
        width_ratios=[0.90] + [1.0] * len(heads), wspace=0.035, hspace=0.18,
    )
    signed_figure = plt.figure(figsize=(20.0, 7.4))
    signed_grid = signed_figure.add_gridspec(
        len(blocks), len(heads) + 1, left=0.025, right=0.99, bottom=0.08, top=0.84,
        width_ratios=[0.90] + [1.0] * len(heads), wspace=0.035, hspace=0.18,
    )
    for row, block in enumerate(blocks):
        for active_figure, active_grid in ((figure, grid), (signed_figure, signed_grid)):
            image_axis = active_figure.add_subplot(active_grid[row, 0])
            image_axis.imshow(photo)
            image_axis.axis("off")
            image_axis.set_ylabel(f"Block {block}", fontsize=10, weight="bold", rotation=90, labelpad=4)
        for head in heads:
            target_id = f"b{block}.head{head}.class_gt"
            raw = head_maps[target_id]
            axis = figure.add_subplot(grid[row, head + 1])
            shown, receipt = positive_view(raw)
            axis.imshow(shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            axis.axis("off")
            if row == 0:
                axis.set_title(str(head), fontsize=8.5, weight="bold", pad=4)
            DISPLAY_PANELS.append({
                "figure": 39, "target_id": target_id, "view": "positive_blue", **receipt
            })
            signed_axis = signed_figure.add_subplot(signed_grid[row, head + 1])
            signed_shown, signed_receipt = signed_view(raw)
            signed_axis.imshow(signed_shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            signed_axis.axis("off")
            if row == 0:
                signed_axis.set_title(str(head), fontsize=8.5, weight="bold", pad=4)
            DISPLAY_PANELS.append({
                "figure": 39, "target_id": target_id, "view": "signed_companion",
                **signed_receipt,
            })
    figure.suptitle(
        "ViT-B/16 Endpoint learning images across heads and depth", fontsize=18, weight="bold", y=0.965, color=INK
    )
    figure.text(
        0.55, 0.90, "Fixed input · all head-edge contributions to the same ground-truth class logit",
        ha="center", fontsize=10, color=MUTED,
    )
    figure.text(
        0.55, 0.025, "Positive reading view · fixed head roster · per-panel P99 display scale",
        ha="center", fontsize=9.2, color=MUTED,
    )
    signed_figure.suptitle(
        "ViT-B/16 Endpoint learning images across heads and depth · signed companions",
        fontsize=18, weight="bold", y=0.965, color=INK,
    )
    signed_figure.text(
        0.55, 0.025, "Blue < 0, white = 0, red > 0 · per-panel signed-log display",
        ha="center", fontsize=9.2, color=MUTED,
    )
    return figure, signed_figure


def module_chain_figure(
    photo: np.ndarray, atlas: np.ndarray, target_index: dict[str, int], cmap: LinearSegmentedColormap
) -> tuple[plt.Figure, plt.Figure]:
    panels = [("Patch embed", "patch.sum.c0")]
    for block in (0, 5, 11):
        panels.extend([
            (f"B{block} Norm1", f"b{block}.norm1.sum.c0"),
            (f"B{block} Attn proj", f"b{block}.attn.proj.sum.c0"),
            (f"B{block} Norm2", f"b{block}.norm2.sum.c0"),
            (f"B{block} MLP fc1", f"b{block}.mlp.fc1.sum.c0"),
            (f"B{block} GELU", f"b{block}.mlp.act.sum.c0"),
            (f"B{block} MLP fc2", f"b{block}.mlp.fc2.sum.c0"),
        ])
    panels.extend([("Final CLS c0", "final.cls.c0"), ("Class logit", "class.gt")])
    columns = 7
    rows = 3
    figure = plt.figure(figsize=(16.5, 8.0))
    grid = figure.add_gridspec(
        rows, columns + 1, left=0.03, right=0.985, bottom=0.07, top=0.84,
        width_ratios=[0.86] + [1.0] * columns, wspace=0.055, hspace=0.22,
    )
    signed_figure = plt.figure(figsize=(16.5, 8.0))
    signed_grid = signed_figure.add_gridspec(
        rows, columns + 1, left=0.03, right=0.985, bottom=0.07, top=0.84,
        width_ratios=[0.86] + [1.0] * columns, wspace=0.055, hspace=0.22,
    )
    for active_figure, active_grid in ((figure, grid), (signed_figure, signed_grid)):
        input_axis = active_figure.add_subplot(active_grid[:, 0])
        input_axis.imshow(photo)
        input_axis.axis("off")
        input_axis.set_title("Fixed input", fontsize=10.5, weight="bold", pad=6)
    for index, (label, target_id) in enumerate(panels):
        row, column = divmod(index, columns)
        raw = np.asarray(atlas[target_index[target_id]])
        axis = figure.add_subplot(grid[row, column + 1])
        shown, receipt = positive_view(raw)
        axis.imshow(shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(label, fontsize=8.8, weight="bold", pad=5)
        axis.axis("off")
        DISPLAY_PANELS.append({
            "figure": 40, "target_id": target_id, "view": "positive_blue", **receipt
        })
        signed_axis = signed_figure.add_subplot(signed_grid[row, column + 1])
        signed_shown, signed_receipt = signed_view(raw)
        signed_axis.imshow(signed_shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        signed_axis.set_title(label, fontsize=8.8, weight="bold", pad=5)
        signed_axis.axis("off")
        DISPLAY_PANELS.append({
            "figure": 40, "target_id": target_id, "view": "signed_companion",
            **signed_receipt,
        })
    figure.suptitle(
        "Endpoint-NNSViT through the real module chain", fontsize=19, weight="bold", y=0.965, color=INK
    )
    figure.text(
        0.57, 0.90, "Fixed ViT-B/16 input and channel 0 · early, middle, and late operator targets",
        ha="center", fontsize=10.3, color=MUTED,
    )
    figure.text(
        0.57, 0.025, "Positive Endpoint reading view · each panel is a registered scalar target in input coordinates",
        ha="center", fontsize=9.2, color=MUTED,
    )
    signed_figure.suptitle(
        "Endpoint-NNSViT through the real module chain · signed companions",
        fontsize=19, weight="bold", y=0.965, color=INK,
    )
    signed_figure.text(
        0.57, 0.025, "Blue < 0, white = 0, red > 0 · per-panel signed-log display",
        ha="center", fontsize=9.2, color=MUTED,
    )
    return figure, signed_figure


def render_all_head_supplements(
    *, model_id: str, model_label: str, head_count: int, photo: np.ndarray,
    head_maps: dict[str, np.ndarray], cmap: LinearSegmentedColormap
) -> None:
    for block in range(12):
        columns = 6
        rows = int(np.ceil(head_count / columns))
        figure = plt.figure(figsize=(13.6, 3.1 * rows + 1.0))
        grid = figure.add_gridspec(
            rows, columns + 1, left=0.035, right=0.985, bottom=0.08, top=0.83,
            width_ratios=[0.85] + [1.0] * 6, wspace=0.055, hspace=0.19,
        )
        input_axis = figure.add_subplot(grid[:, 0])
        input_axis.imshow(photo)
        input_axis.axis("off")
        input_axis.set_title("Fixed input", fontsize=10.5, weight="bold", pad=6)
        for head in range(head_count):
            row, column = divmod(head, 6)
            target_id = f"b{block}.head{head}.class_gt"
            raw = head_maps[target_id]
            axis = figure.add_subplot(grid[row, column + 1])
            shown, receipt = positive_view(raw)
            axis.imshow(shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(f"Head {head}", fontsize=9.2, weight="bold", pad=5)
            axis.axis("off")
            DISPLAY_PANELS.append({
                "figure": f"S{block + 1:02d}", "target_id": target_id,
                "view": "positive_blue", **receipt,
            })
        figure.suptitle(
            f"{model_label} Block {block}: all-head Endpoint learning images",
            fontsize=18, weight="bold", y=0.965, color=INK,
        )
        figure.text(
            0.57, 0.885, f"Same ground-truth class target · fixed first evaluation image · all {head_count} heads",
            ha="center", fontsize=10, color=MUTED,
        )
        figure.text(
            0.57, 0.025, "Positive reading view · per-panel P99 · fixed roster, no visual selection",
            ha="center", fontsize=9.2, color=MUTED,
        )
        path = ROOT / f"all_heads_{model_id}" / f"block_{block:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(figure)
        SUPPLEMENTAL_HEAD_PAGES.append({
            "model_id": model_id, "block": block, "head_count": head_count,
            "target_kind": "head_edge_contribution_to_ground_truth_class_logit",
            "path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })


def save_figure(figure: plt.Figure, number: int, slug: str) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for folder, extension, file_format in (
        ("figures_png", "png", "PNG"),
        ("figures_pdf", "pdf", "PDF"),
        ("figures_svg", "svg", "SVG"),
    ):
        path = ROOT / folder / f"figure_{number:02d}_{slug}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=400, facecolor="white", bbox_inches="tight")
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
        outputs.append({
            "format": file_format,
            "path": path.relative_to(REPO).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    plt.close(figure)
    return outputs


def save_signed_companion(figure: plt.Figure, number: int, slug: str) -> dict[str, Any]:
    path = ROOT / "signed_companions_png" / f"figure_{number:02d}_{slug}_signed.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return {
        "format": "PNG",
        "path": path.relative_to(REPO).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_sources() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, int],
    list[tuple[str, np.ndarray, str]],
    list[tuple[str, np.ndarray, str]],
    list[tuple[str, np.ndarray, str]],
]:
    e2, atlas, target_index = load_e2_atlas("deit_s16")
    e3 = load_json(E3_ATTEMPT, "sealed_E3_fixed_first_task")["result"]
    e5 = load_json(E5_ATTEMPT, "sealed_E5_fixed_first_task")["result"]
    require(
        e2["sample_id"] == e3["sample_id"] == e5["sample_id"] == SOURCE_IMAGE.name,
        "CROSS_STAGE_SAMPLE_ID_MISMATCH",
    )
    require(sha256(SOURCE_IMAGE) == e2["image_sha256"], "SOURCE_IMAGE_SHA256_MISMATCH")

    target_specs = [
        ("B0 patch sum · c0", "b0.patch_channel_sum.c0"),
        ("B0 patch energy · c0", "b0.patch_channel_energy.c0"),
        ("B0 head 0 · CLS c0", "b0.head0.cls.c0"),
        ("B5 head 0 · CLS c0", "b5.head0.cls.c0"),
        ("B11 head 5 · CLS c0", "b11.head5.cls.c0"),
        ("Ground-truth class logit", "class.gt"),
    ]
    atlas_panels = [(label, np.asarray(atlas[target_index[target_id]]), target_id) for label, target_id in target_specs]

    block_record = next(
        item for item in e3["artifacts"]
        if item["kind"] == "raw_component_and_reconstruction_maps" and item["block"] == 5
    )
    block_path = REPO / block_record["path"]
    register(block_path, "sealed_E3_block5_component_maps")
    require(SOURCE_FILES[block_record["path"]]["sha256"] == block_record["sha256"], "E3_BLOCK5_SHA256_MISMATCH")
    block_array = np.load(block_path, mmap_mode="r")
    require(list(block_array.shape) == block_record["shape"], "E3_BLOCK5_SHAPE_MISMATCH")
    block_index = {name: index for index, name in enumerate(block_record["component_order"])}
    component_specs = [
        ("Identity path", "identity"),
        ("Attention head 0", "head_0"),
        ("Routing · Q+K", "routing"),
        ("Value path", "value"),
        ("Attention branch", "attention"),
        ("MLP branch", "mlp"),
        ("Complete class Endpoint", "L_F"),
    ]
    component_panels = [
        (label, np.asarray(block_array[block_index[target_id]]), f"block5.{target_id}")
        for label, target_id in component_specs
    ]

    maps_record = e5["raw_maps_record"]
    maps_path = REPO / maps_record["path"]
    register(maps_path, "sealed_E5_fixed_first_ranking_maps")
    require(SOURCE_FILES[maps_record["path"]]["sha256"] == maps_record["sha256"], "E5_MAPS_SHA256_MISMATCH")
    with np.load(maps_path) as mapping:
        ranking_specs = [
            ("Total G×I", "total_gxi"),
            ("Raw input gradient", "raw_gradient"),
            ("Block 5 · head 0", "head_0"),
            ("Block 5 · head 3", "head_3"),
            ("Block 5 · head 5", "head_5"),
        ]
        ranking_panels = [(label, np.asarray(mapping[target_id]), target_id) for label, target_id in ranking_specs]
    return display_photo(), atlas, target_index, atlas_panels, component_panels, ranking_panels


def build_docs() -> None:
    captions_zh = {
        34: "固定第一张 DeiT-S/16 evaluation 图像的六个注册标量目标。蓝底主图只显示正部，逐面板 P99 缩放；它是空间阅读视图，不用于跨目标比较强度。",
        35: "同一图像、同一 ground-truth 类别目标、同一 block 5 的 identity、head、routing、value、attention、MLP 与完整 Endpoint。各图直接来自通过闭合验证的 raw component arrays。",
        36: "RQ4 固定任务使用的 total G×I、raw gradient 与三个注册 head component map。图只解释排序输入的细粒度空间形态，行为结论仍由有限干预曲线决定。",
        37: "参考 NNS 论文 Figure 13 的跨层结构，展示固定 DeiT-S/16 输入从 patch embedding、Block 0–11 到 final CLS 与类别 logit 的 Endpoint learning image。下排是由同一 raw Endpoint map 在原生 16×16 input patches 内汇总得到的 14×14 absolute mass，不是 forward feature map。",
        38: "固定 DeiT-S/16 输入在 Block 0、1、5、11 的全部六个 attention heads。每格是相应 head edge 对同一个 ground-truth class logit 的 224×224 Endpoint contribution map；行比较深度、列比较 head index，显示尺度逐面板独立。",
        39: "固定 ViT-B/16 输入在 Block 0、1、5、11 的全部十二个 attention heads。每格解释同一个 ground-truth class logit，因此可在共同目标下比较各 head 的空间形态；完整 12-block × 12-head 页面保留在 all_heads_vit_b16/。",
        40: "固定 ViT-B/16 输入和 channel 0，展示 patch embedding、Block 0/5/11 的 Norm1、attention projection、Norm2、MLP fc1、GELU、fc2，以及 final CLS 和类别 logit。它对应 NNSViT-IG Figure 28–34 的模块链阅读方式，但使用 Endpoint 标量目标。",
    }
    captions_en = {
        34: "Six registered scalar targets for the fixed first DeiT-S/16 evaluation image. The blue-base main panels show only positive values with per-panel P99 scaling; they are spatial reading views, not cross-target magnitude comparisons.",
        35: "Identity, head, routing, value, attention, MLP, and complete Endpoint maps for the same image, ground-truth class target, and block 5. Every panel comes directly from raw component arrays that passed closure validation.",
        36: "Total G×I, raw gradient, and three registered head-component maps used by the fixed RQ4 task. These panels show the fine-grained spatial form of the ranking inputs; finite-intervention curves determine the behavioral result.",
        37: "Following the cross-layer organization of NNS Figure 13, this figure tracks a fixed DeiT-S/16 input from patch embedding through Blocks 0–11 to final CLS and the class logit. The bottom row is 14×14 absolute mass pooled from the same raw Endpoint maps over native 16×16 input patches; it is not a forward feature map.",
        38: "All six attention-head edge contributions to the same ground-truth class logit for a fixed DeiT-S/16 input at Blocks 0, 1, 5, and 11. Rows compare depth and columns compare head index; display scaling is independent per panel.",
        39: "All twelve attention-head edge contributions to the same ground-truth class logit for a fixed ViT-B/16 input at Blocks 0, 1, 5, and 11. Complete 12-block by 12-head pages are retained under all_heads_vit_b16/.",
        40: "For a fixed ViT-B/16 input and channel 0, this figure shows patch embedding; Norm1, attention projection, Norm2, MLP fc1, GELU, and fc2 at Blocks 0, 5, and 11; final CLS; and the class logit. It follows the module-chain reading style of NNSViT-IG Figures 28–34 while using Endpoint scalar targets.",
    }
    lines = [
        "# Endpoint fine-grained blue-base paper figures", "",
        "This is a CPU-only display package derived from sealed E2, E3, and E5 raw arrays.",
        "It adds no model execution, attribution recomputation, statistical test, or scientific decision.", "",
        "The main panels follow the R10 reading style: RGB channels are summed with sign,",
        "only `max(L, 0)` is shown, each panel uses P99 scaling with a max fallback, and deep",
        "blue denotes zero or low positive value. Signed companions are retained beside the main",
        "figures. Numerical claims continue to use the sealed raw arrays.", "",
        "The Endpoint targets here are scalar internal or class targets with direct `3×224×224`",
        "input-coordinate maps. Unlike the path-paper patch-embedding figure, they do not have a",
        "corresponding `14×14` output-token response row for the same scalar target.", "",
        "## Figures", "",
    ]
    analysis = ["# Endpoint 细粒度蓝底图逐图分析", ""]
    for figure in FIGURES:
        number = figure["paper_number"]
        png = next(item["path"] for item in figure["files"] if item["format"] == "PNG")
        relative_png = Path(png).relative_to(ROOT.relative_to(REPO)).as_posix()
        signed_png = Path(figure["signed_companion"]["path"]).relative_to(ROOT.relative_to(REPO)).as_posix()
        lines.extend([
            f"### Figure {number}: {figure['title']}", "", captions_en[number], "",
            f"[Main PNG]({relative_png}) · [Signed companion]({signed_png})", "",
        ])
        analysis.extend([
            f"## Image #{number - 33}: {figure['title']}", "", captions_zh[number], "",
            f"![Figure {number}]({relative_png})", "",
            f"[Signed companion]({signed_png})", "",
        ])
    (ROOT / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    (ROOT / "figure_analysis.md").write_text("\n".join(analysis).rstrip() + "\n", encoding="utf-8")
    (ROOT / "captions_zh.md").write_text(
        "# 图注\n\n" + "\n\n".join(f"## Figure {n}\n\n{captions_zh[n]}" for n in sorted(captions_zh)) + "\n",
        encoding="utf-8",
    )
    (ROOT / "captions_en.md").write_text(
        "# Captions\n\n" + "\n\n".join(f"## Figure {n}\n\n{captions_en[n]}" for n in sorted(captions_en)) + "\n",
        encoding="utf-8",
    )
    cards = []
    for figure in FIGURES:
        number = figure["paper_number"]
        png = next(item["path"] for item in figure["files"] if item["format"] == "PNG")
        relative_png = Path(png).relative_to(ROOT.relative_to(REPO)).as_posix()
        cards.append(
            f'<article><h2>Figure {number} · {html.escape(figure["title"])}</h2>'
            f'<img src="{relative_png}"><p>{html.escape(captions_zh[number])}</p></article>'
        )
    gallery = (
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Endpoint fine-grained figures</title><style>'
        'body{font:16px/1.6 system-ui;color:#172033;background:#eef3f6;max-width:1500px;margin:auto;padding:24px}'
        'article{background:white;padding:20px;margin:24px 0}img{width:100%;height:auto}</style>'
        '<h1>Endpoint fine-grained blue-base figures</h1>' + "".join(cards) + '</html>'
    )
    (ROOT / "gallery.html").write_text(gallery, encoding="utf-8")


def validate_outputs(manifest: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(condition), "detail": detail})

    check("cpu_only_cuda_hidden", os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    check("figure_count", manifest["figure_count"] == 7)
    check("paper_numbers", [item["paper_number"] for item in manifest["figures"]] == [34, 35, 36, 37, 38, 39, 40])
    check("source_files", len(SOURCE_FILES) == 34, str(len(SOURCE_FILES)))
    check("display_panel_count", len(DISPLAY_PANELS) == 483, str(len(DISPLAY_PANELS)))
    check("supplemental_head_page_count", len(SUPPLEMENTAL_HEAD_PAGES) == 24, str(len(SUPPLEMENTAL_HEAD_PAGES)))
    check("all_display_arrays_finite", all(np.isfinite(item["display_limit"]) for item in DISPLAY_PANELS))
    for figure in manifest["figures"]:
        for record in [*figure["files"], figure["signed_companion"]]:
            path = REPO / record["path"]
            check(f"exists:{record['path']}", path.is_file())
            if path.is_file():
                check(f"sha256:{record['path']}", sha256(path) == record["sha256"])
                check(f"nonempty:{record['path']}", path.stat().st_size > 10_000, str(path.stat().st_size))
    for record in manifest["supplemental_all_head_pages"]:
        path = REPO / record["path"]
        check(f"exists:{record['path']}", path.is_file())
        if path.is_file():
            check(f"sha256:{record['path']}", sha256(path) == record["sha256"])
    errors = [item for item in checks if not item["pass"]]
    return signed({
        "status": "PASS_PRESENTATION_RENDER" if not errors else "FAIL",
        "check_count": len(checks),
        "error_count": len(errors),
        "errors": errors,
        "scientific_inference": "NONE",
    })


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    cmap = load_palette()
    photo, atlas, target_index, atlas_panels, component_panels, ranking_panels = load_sources()
    vit_result, vit_atlas, vit_target_index = load_e2_atlas("vit_b16")
    require(vit_result["image_sha256"] == sha256(SOURCE_IMAGE), "VIT_SOURCE_IMAGE_SHA256_MISMATCH")
    deit_e3_result, deit_head_maps = load_e3_head_maps("deit_s16")
    vit_e3_result, vit_head_maps = load_e3_head_maps("vit_b16")
    require(deit_e3_result["image_sha256"] == sha256(SOURCE_IMAGE), "DEIT_E3_SOURCE_IMAGE_SHA256_MISMATCH")
    require(vit_e3_result["image_sha256"] == sha256(SOURCE_IMAGE), "VIT_E3_SOURCE_IMAGE_SHA256_MISMATCH")
    specs = [
        (34, "registered_target_learning_images", "Endpoint target atlas: real fine-grained learning images",
         "DeiT-S/16 · fixed first evaluation image · six preregistered scalar targets", atlas_panels),
        (35, "block5_structural_learning_images", "Block 5: real structural Endpoint learning images",
         "Same image · same ground-truth class target · closure-validated component arrays", component_panels),
        (36, "spatial_ranking_learning_images", "Block 5: real maps used for spatial ranking",
         "Same fixed RQ4 task · baselines and preregistered head components", ranking_panels),
    ]
    for number, slug, title, subtitle, panels in specs:
        main_figure, signed_figure = render_grid(
            number=number, slug=slug, title=title, subtitle=subtitle, photo=photo,
            panels=panels, cmap=cmap,
        )
        outputs = save_figure(main_figure, number, slug)
        companion = save_signed_companion(signed_figure, number, slug)
        FIGURES.append({
            "paper_number": number,
            "slug": slug,
            "title": title,
            "role": "display_only_spatial_reading_view",
            "files": outputs,
            "signed_companion": companion,
        })
    for number, slug, title, builder in (
        (37, "channel0_depth_progression", "Endpoint-NNSViT across depth", block_progression_figure),
        (38, "attention_head_depth_atlas", "Endpoint-NNSViT across heads and depth", head_atlas_figure),
    ):
        source_maps, source_index = (deit_head_maps, {}) if number == 38 else (atlas, target_index)
        main_figure, signed_figure = builder(photo, source_maps, source_index, cmap)
        outputs = save_figure(main_figure, number, slug)
        companion = save_signed_companion(signed_figure, number, slug)
        FIGURES.append({
            "paper_number": number,
            "slug": slug,
            "title": title,
            "role": "display_only_spatial_reading_view",
            "files": outputs,
            "signed_companion": companion,
        })
    for number, slug, title, builder in (
        (39, "vit_b16_attention_head_depth_atlas", "ViT-B/16 Endpoint across heads and depth", vit_head_depth_figure),
        (40, "vit_b16_module_chain", "Endpoint-NNSViT through the real module chain", module_chain_figure),
    ):
        source_maps, source_index = (vit_head_maps, {}) if number == 39 else (vit_atlas, vit_target_index)
        main_figure, signed_figure = builder(photo, source_maps, source_index, cmap)
        outputs = save_figure(main_figure, number, slug)
        companion = save_signed_companion(signed_figure, number, slug)
        FIGURES.append({
            "paper_number": number,
            "slug": slug,
            "title": title,
            "role": "display_only_spatial_reading_view",
            "files": outputs,
            "signed_companion": companion,
        })
    render_all_head_supplements(
        model_id="deit_s16", model_label="DeiT-S/16", head_count=6, photo=photo,
        head_maps=deit_head_maps, cmap=cmap,
    )
    render_all_head_supplements(
        model_id="vit_b16", model_label="ViT-B/16", head_count=12, photo=photo,
        head_maps=vit_head_maps, cmap=cmap,
    )

    source_inventory = signed({
        "schema_version": 1,
        "sample_id": SOURCE_IMAGE.name,
        "sources": sorted(SOURCE_FILES.values(), key=lambda item: item["path"]),
    })
    write_json(ROOT / "source_inventory.json", source_inventory)
    display_receipt = signed({
        "schema_version": 1,
        "display_only": True,
        "new_scientific_inference": False,
        "main_view": "positive blue-base R10-style reading view",
        "signed_companions_retained": True,
        "cross_panel_magnitude_comparison_allowed": False,
        "raw_arrays_remain_authoritative": True,
        "panels": DISPLAY_PANELS,
    })
    write_json(ROOT / "display_receipt.json", display_receipt)
    manifest_base = {
        "schema_version": 1,
        "revision": "ENDPOINT-FINE-GRAINED-BLUE-R1",
        "status": "PENDING_VALIDATION",
        "figure_count": len(FIGURES),
        "figures": FIGURES,
        "supplemental_all_head_pages": SUPPLEMENTAL_HEAD_PAGES,
        "source_inventory_sha256": sha256(ROOT / "source_inventory.json"),
        "display_receipt_sha256": sha256(ROOT / "display_receipt.json"),
        "scientific_boundaries": {
            "cpu_only": True,
            "model_or_checkpoint_load": False,
            "attribution_recomputation": False,
            "new_statistics_or_inference": False,
            "positive_main_panels_are_display_only": True,
            "numerical_claims_use_sealed_raw_arrays": True,
        },
    }
    manifest = signed(manifest_base)
    validation = validate_outputs(manifest)
    require(validation["status"] == "PASS_PRESENTATION_RENDER", f"VALIDATION_FAILED:{validation['errors']}")
    manifest = signed({**manifest_base, "status": "PASS_PRESENTATION_RENDER"})
    write_json(ROOT / "figure_manifest.json", manifest)
    write_json(ROOT / "validation_report.json", validation)
    paper_figures = []
    for figure in FIGURES:
        exports = {record["format"].lower(): record for record in figure["files"]}
        paper_figures.append({
            "paper_number": figure["paper_number"],
            "title": figure["title"],
            "png": {key: exports["png"][key] for key in ("path", "sha256")},
            "pdf": {key: exports["pdf"][key] for key in ("path", "sha256")},
            "svg": {key: exports["svg"][key] for key in ("path", "sha256")},
            "signed_companion": {
                key: figure["signed_companion"][key] for key in ("path", "sha256")
            },
        })
    paper_binding = {
        "schema_version": 1,
        "source_root": ROOT.relative_to(REPO).as_posix(),
        "source_manifest": {
            "path": (ROOT / "figure_manifest.json").relative_to(REPO).as_posix(),
            "sha256": sha256(ROOT / "figure_manifest.json"),
        },
        "figure_count": len(FIGURES),
        "display_policy": "RGB signed sum; main view max(L,0); per-panel P99 with max fallback; R10 blue-base palette; display-only; no cross-panel magnitude comparison",
        "figures": paper_figures,
        "supplemental_all_head_pages": SUPPLEMENTAL_HEAD_PAGES,
    }
    write_json(PAPER_MANIFEST, paper_binding)
    build_docs()
    print(json.dumps({
        "status": validation["status"],
        "root": ROOT.relative_to(REPO).as_posix(),
        "figure_count": len(FIGURES),
        "display_panels": len(DISPLAY_PANELS),
        "supplemental_all_head_pages": len(SUPPLEMENTAL_HEAD_PAGES),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
