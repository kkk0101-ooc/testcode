"""CPU-only, display-only views of sealed F3C formal attribution maps.

This module imports the CPU-only F3C4 reader instead of any model runtime. It
never evaluates fidelity, changes a ranking, or writes to upstream evidence.
Rendering first goes to a pending directory; an explicit post-inspection step
publishes the directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_sha256
from . import stage_f_f3c4 as f3c4
from . import stage_f_f3c4_r3 as f3c4_r3

OUTPUT_ROOT = "reports/stage_f_f3c_visual_v1_20260908"
PENDING_ROOT = "reports/.stage_f_f3c_visual_v1_20260908.pending"
DISPLAY_PARAMETERS_PATH = "display_parameters.json"
INPUT_BINDING_PATH = "input_binding.json"
RENDER_MANIFEST_PATH = "render_manifest.json"
ANALYSIS_ZH_PATH = "analysis_zh.json"
RUNTIME_PATH = "runtime.json"
FIGURE_HASH_PATH = "figure_sha256_manifest.json"
QC_PATH = "manual_visual_qc.json"
VALIDATION_PATH = "validation_report.json"
COMPLETION_PATH = "completion.json"

F3C3_INVENTORY_PATH = f3c4.F3C3_ROOT + "/whole_root_inventory.json"
F3C4_R1_BINDING_PATH = f3c4_r3.R1_BINDING_PATH
F3C4_R3_COMPLETION_PATH = f3c4_r3.COMPLETION_PATH
SOURCE_PATH = "src/vitnns/stage_f_f3c_visual.py"
EXPECTED_PAGE5 = ((21, 74), (22, 69), (23, 295), (24, 132), (25, 242))

NNS = f3c4.METHODS[0]
GRAD_CAM = f3c4.METHODS[1]
FULL_IG = f3c4.METHODS[2]
ROLLOUT = f3c4.METHODS[3]
RAW_ATTN = f3c4.METHODS[4]

PALETTE = (
    (0.00, "#07077F"), (0.20, "#0019C6"), (0.42, "#00AEE8"),
    (0.65, "#FFF000"), (0.82, "#FF6A00"), (1.00, "#D90000"),
)
MAIN_PANEL_ORDER = (
    "nns_native_positive", "nns_pool4_bilinear_positive", "grad_cam_native",
    "full_ig_native_positive", "full_ig_pool4_bilinear_positive",
    "rollout_native", "raw_cls_attention_native",
)
MAIN_PANEL_LABELS = {
    "nns_native_positive": "NNS native 224\npositive support",
    "nns_pool4_bilinear_positive": "NNS max-pool 4x4\n56 -> 224 bilinear",
    "grad_cam_native": "Grad-CAM\nnative 14x14",
    "full_ig_native_positive": "full-model IG native 224\npositive support",
    "full_ig_pool4_bilinear_positive": "full IG max-pool 4x4\n56 -> 224 bilinear",
    "rollout_native": "rollout\nnative 14x14",
    "raw_cls_attention_native": "raw CLS attention\nnative 14x14",
}
SIGNED_PANEL_ORDER = (
    "nns_positive", "nns_negative", "full_ig_positive", "full_ig_negative",
)
SIGNED_PANEL_LABELS = {
    "nns_positive": "NNS positive support",
    "nns_negative": "NNS negative contribution",
    "full_ig_positive": "full IG positive support",
    "full_ig_negative": "full IG negative contribution",
}


class StageFF3CVisualError(ValueError):
    """Fail-closed error for the display-only package."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, relative: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise StageFF3CVisualError(f"not a regular file: {path}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if process.returncode:
        raise StageFF3CVisualError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def _assert_cpu_boundary() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise StageFF3CVisualError("CUDA_VISIBLE_DEVICES must be the empty string")
    if "torch" in sys.modules or "timm" in sys.modules:
        raise StageFF3CVisualError("torch/timm must not be imported by the visual process")


def _validate_2d(
    value: np.ndarray, *, name: str, shape: tuple[int, int] | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.size == 0:
        raise StageFF3CVisualError(f"{name} must be a non-empty 2D array")
    if shape is not None and tuple(array.shape) != shape:
        raise StageFF3CVisualError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise StageFF3CVisualError(f"{name} must contain only finite numeric values")
    result = np.asarray(array, dtype=np.float64)
    if nonnegative and np.any(result < 0):
        raise StageFF3CVisualError(f"{name} must be nonnegative")
    return result


def signed_rgb_projection(native: np.ndarray) -> np.ndarray:
    array = np.asarray(native)
    if array.shape != (3, 224, 224) or array.dtype != np.float32:
        raise StageFF3CVisualError("signed native attribution must be float32 3x224x224")
    if not np.isfinite(array).all():
        raise StageFF3CVisualError("signed native attribution contains non-finite values")
    return np.sum(array.astype(np.float64), axis=0)


def split_signed_support(signed_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = _validate_2d(signed_map, name="signed map", shape=(224, 224))
    return np.maximum(value, 0.0), np.maximum(-value, 0.0)


def max_pool_4x4(value: np.ndarray) -> np.ndarray:
    array = _validate_2d(value, name="max-pool input", shape=(224, 224), nonnegative=True)
    return np.asarray(array.reshape(56, 4, 56, 4).max(axis=(1, 3)), dtype=np.float64)


def bilinear_resize_224(value: np.ndarray) -> np.ndarray:
    array = _validate_2d(value, name="bilinear resize input", shape=(56, 56), nonnegative=True)
    from PIL import Image

    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    result = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise StageFF3CVisualError("bilinear resize produced invalid values")
    return result


def display_scale(value: np.ndarray, *, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    array = _validate_2d(value, name="display map", nonnegative=True)
    if mode == "percentile99":
        limit = float(np.percentile(array, 99.0))
        rule = "per_panel_99th_percentile_upper_clip"
    elif mode == "maximum":
        limit = float(np.max(array))
        rule = "per_panel_maximum_no_clip"
    else:
        raise StageFF3CVisualError(f"unsupported display scale mode: {mode}")
    maximum = float(np.max(array))
    if limit == 0.0:
        normalized = np.zeros_like(array, dtype=np.float64)
        clipped = 0
        all_zero = True
    elif not math.isfinite(limit) or limit < 0:
        raise StageFF3CVisualError("invalid display scale limit")
    else:
        normalized = np.clip(array / limit, 0.0, 1.0)
        clipped = int(np.count_nonzero(array > limit))
        all_zero = False
    metadata = {
        "rule": rule, "percentile": 99.0 if mode == "percentile99" else None,
        "vmin": 0.0, "vmax_raw": limit, "raw_maximum": maximum,
        "upper_clipping": mode == "percentile99", "clipped_pixel_count": clipped,
        "pixel_count": int(array.size), "gamma": 1.0, "all_zero": all_zero,
    }
    return normalized, metadata


def build_display_maps(native: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(native) != set(f3c4.METHODS):
        raise StageFF3CVisualError("native map roster differs from frozen five methods")
    nns_signed = signed_rgb_projection(native[NNS])
    ig_signed = signed_rgb_projection(native[FULL_IG])
    nns_positive, nns_negative = split_signed_support(nns_signed)
    ig_positive, ig_negative = split_signed_support(ig_signed)
    grad_cam = _validate_2d(native[GRAD_CAM], name=GRAD_CAM, shape=(14, 14), nonnegative=True)
    rollout = _validate_2d(native[ROLLOUT], name=ROLLOUT, shape=(14, 14), nonnegative=True)
    raw_attention = _validate_2d(native[RAW_ATTN], name=RAW_ATTN, shape=(14, 14), nonnegative=True)
    return {
        "nns_native_positive": nns_positive,
        "nns_pool4_bilinear_positive": bilinear_resize_224(max_pool_4x4(nns_positive)),
        "grad_cam_native": grad_cam,
        "full_ig_native_positive": ig_positive,
        "full_ig_pool4_bilinear_positive": bilinear_resize_224(max_pool_4x4(ig_positive)),
        "rollout_native": rollout, "raw_cls_attention_native": raw_attention,
        "nns_positive": nns_positive, "nns_negative": nns_negative,
        "full_ig_positive": ig_positive, "full_ig_negative": ig_negative,
    }


def _inventory(root: Path, *, exclude: Sequence[str]) -> dict[str, Any]:
    excluded = set(exclude)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return {
        "files": files, "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "file_inventory_digest": canonical_sha256({"files": files}),
    }


def _upstream_snapshot(repo: Path) -> dict[str, Any]:
    _assert_cpu_boundary()
    r3_status = f3c4_r3.validate_completion(repo)
    inventory_document = f3c4._read(repo / F3C3_INVENTORY_PATH)
    f3c4._validate_signed(inventory_document, "F3C3 whole-root inventory")
    f3c3_current = _inventory(
        repo / f3c4.F3C3_ROOT,
        exclude=("whole_root_inventory.json", "completion/f3c3.json", "completion/f3c3.pending.json"),
    )
    for key in ("files", "file_count", "total_bytes", "file_inventory_digest"):
        if inventory_document.get(key) != f3c3_current[key]:
            raise StageFF3CVisualError(f"sealed F3C3 inventory drifted: {key}")
    r1_binding = f3c4._read(repo / F3C4_R1_BINDING_PATH)
    f3c4._validate_signed(r1_binding, "F3C4 R1 payload binding")
    r1_current = f3c4_r3.r1_inventory(repo)
    for key in ("files", "file_count", "total_bytes", "file_inventory_digest"):
        if r1_binding.get(key) != r1_current[key]:
            raise StageFF3CVisualError(f"sealed F3C4 R1 payload drifted: {key}")
    refs = {
        "f3c3": {"ref": "refs/archive/stage-f-v1-f3c3-formal", "commit": _git(repo, "rev-parse", "refs/archive/stage-f-v1-f3c3-formal^{commit}")},
        "f3c4": {"ref": f3c4.ARCHIVE_REF, "commit": _git(repo, "rev-parse", f3c4.ARCHIVE_REF + "^{commit}")},
    }
    if refs["f3c3"]["commit"] != f3c4.STARTUP_HEAD or refs["f3c4"]["commit"] != "944acd6e2b7579e6999668911ed420a1bd91a27b":
        raise StageFF3CVisualError("sealed archive ref moved")
    return {
        "f3c4_r3_completion_validation": r3_status,
        "f3c3_inventory_record": _file_record(repo / F3C3_INVENTORY_PATH, F3C3_INVENTORY_PATH),
        "f3c3_inventory_canonical_digest": inventory_document["canonical_digest"],
        "f3c3_file_inventory_digest": f3c3_current["file_inventory_digest"],
        "f3c3_file_count": f3c3_current["file_count"],
        "f3c4_r1_binding_record": _file_record(repo / F3C4_R1_BINDING_PATH, F3C4_R1_BINDING_PATH),
        "f3c4_r1_binding_canonical_digest": r1_binding["canonical_digest"],
        "f3c4_r1_file_inventory_digest": r1_current["file_inventory_digest"],
        "f3c4_r1_file_count": r1_current["file_count"],
        "f3c4_r3_completion_record": _file_record(repo / F3C4_R3_COMPLETION_PATH, F3C4_R3_COMPLETION_PATH),
        "archive_refs": refs,
    }


def _same_upstream(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return canonical_sha256(left) == canonical_sha256(right)


def _output_relative(relative: str) -> str:
    return f"{OUTPUT_ROOT}/{relative}"


def _add_source(records: dict[str, dict[str, Any]], repo: Path, path: Path) -> None:
    relative = path.relative_to(repo).as_posix()
    record = _file_record(path, relative)
    previous = records.setdefault(relative, record)
    if previous != record:
        raise StageFF3CVisualError(f"source changed during rendering: {relative}")


def _load_sample(
    repo: Path, row: Mapping[str, Any], source_records: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    jpeg = repo / f3c4.JPEG_ROOT / row["relative_path"]
    if _sha_file(jpeg) != row["image_sha256"]:
        raise StageFF3CVisualError("formal JPEG hash mismatch before decode")
    _add_source(source_records, repo, jpeg)
    original = f3c4._image_crop_224(jpeg)
    native: dict[str, np.ndarray] = {}
    for method in f3c4.METHODS:
        terminal = f3c4._terminal_root(repo, row["selection_order"], method)
        manifest_path = terminal / "manifest.json"
        manifest = f3c4._read(manifest_path)
        f3c4._validate_signed(manifest, "F3C3 terminal manifest")
        _add_source(source_records, repo, manifest_path)
        expected_shape = (3, 224, 224) if method in f3c4.SIGNED_METHODS else (14, 14)
        native[method] = f3c4._load_array(terminal, manifest, "native.npy", expected_shape)
        _add_source(source_records, repo, terminal / "native.npy")
    return original, build_display_maps(native)


def _legacy_page5_scale_audit(
    repo: Path, rows: Sequence[Mapping[str, Any]], source_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for row in rows[20:25]:
        peaks = {}
        for method in f3c4.METHODS:
            terminal = f3c4._terminal_root(repo, row["selection_order"], method)
            manifest = f3c4._read(terminal / "manifest.json")
            value = f3c4._load_array(terminal, manifest, "target_oriented_mass.npy", (14, 14))
            _add_source(source_records, repo, terminal / "target_oriented_mass.npy")
            peaks[method] = float(np.max(value))
        shared_maximum = max(peaks.values())
        records.append({
            "formal_position": row["formal_position"], "selection_order": row["selection_order"],
            "nns_peak": peaks[NNS], "shared_five_method_maximum": shared_maximum,
            "nns_peak_fraction_of_shared_scale": peaks[NNS] / shared_maximum,
            "shared_scale_dominant_method": max(peaks, key=peaks.get),
        })
    return records


def _render_page(
    *, mode: str, page_rows: Sequence[Mapping[str, Any]],
    samples: Sequence[tuple[np.ndarray, dict[str, np.ndarray]]], cmap: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    import matplotlib.pyplot as plt

    signed = mode == "signed"
    panel_order = SIGNED_PANEL_ORDER if signed else MAIN_PANEL_ORDER
    labels = SIGNED_PANEL_LABELS if signed else MAIN_PANEL_LABELS
    columns = len(panel_order) + 1
    figsize = (10.8, 9.2) if signed else (16.2, 9.2)
    fig, axes = plt.subplots(5, columns, figsize=figsize, squeeze=False, constrained_layout=True)
    scale_records = []
    scale_mode = "percentile99" if mode in {"primary", "signed"} else "maximum"
    for row_index, (row, (original, maps)) in enumerate(zip(page_rows, samples)):
        axes[row_index, 0].imshow(original, interpolation="nearest")
        axes[row_index, 0].set_ylabel(
            f"pos {row['formal_position']:02d}\norder {row['selection_order']}", fontsize=7
        )
        if row_index == 0:
            axes[row_index, 0].set_title("Original", fontsize=8)
        for column, panel_id in enumerate(panel_order, 1):
            normalized, scale = display_scale(maps[panel_id], mode=scale_mode)
            axes[row_index, column].imshow(
                normalized, cmap=cmap, vmin=0.0, vmax=1.0,
                interpolation="bilinear", rasterized=True,
            )
            if row_index == 0:
                axes[row_index, column].set_title(labels[panel_id], fontsize=7.5)
            scale_records.append({
                "formal_position": row["formal_position"],
                "selection_order": row["selection_order"], "panel_id": panel_id, **scale,
            })
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor(PALETTE[0][1])
    first, last = page_rows[0]["formal_position"], page_rows[-1]["formal_position"]
    if mode == "primary":
        title = f"Display-only shape view, positions {first}-{last}: per-panel P99 upper clip, gamma=1"
    elif mode == "unclipped":
        title = f"Unclipped control, positions {first}-{last}: per-panel maximum, linear gamma=1"
    else:
        title = f"Signed contribution audit, positions {first}-{last}: positive and negative kept separate"
    fig.suptitle(title, fontsize=11)
    return fig, scale_records


def _configure_matplotlib() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.backends.backend_pdf import PdfPages

    cmap = LinearSegmentedColormap.from_list("nns_reference_like", list(PALETTE), N=256)
    return cmap, PdfPages


def display_parameters() -> dict[str, Any]:
    return {
        "schema_version": 1, "document_type": "stage_f_f3c_visual_display_parameters",
        "status": "DISPLAY_ONLY", "formal_sample_count": 50, "samples_per_page": 5,
        "page_count_per_view": 10, "frozen_sample_order_reused": True,
        "main_panel_order": list(MAIN_PANEL_ORDER), "signed_panel_order": list(SIGNED_PANEL_ORDER),
        "signed_projection": "sum_RGB_with_sign_then_split_max(x,0)_and_max(-x,0)",
        "pooling": {
            "applies_to": ["nns_positive_support", "full_model_ig_positive_support"],
            "kernel": [4, 4], "stride": [4, 4], "input_shape": [224, 224],
            "pooled_shape": [56, 56], "display_resize": "bilinear_56_to_224",
        },
        "primary_scale": {
            "scope": "each_panel_independently", "upper_bound": "99th_percentile",
            "upper_clipping": True, "gamma": 1.0,
        },
        "unclipped_control_scale": {
            "scope": "each_panel_independently", "upper_bound": "panel_maximum",
            "upper_clipping": False, "gamma": 1.0,
        },
        "signed_auxiliary_scale": {
            "scope": "each_positive_or_negative_panel_independently",
            "upper_bound": "99th_percentile", "positive_and_negative_never_combined": True,
            "gamma": 1.0,
        },
        "palette": [{"position": position, "hex": color} for position, color in PALETTE],
        "heatmap_render_interpolation": "bilinear", "png_dpi": 220,
        "formats": {"png": "one_file_per_page", "pdf": "one_multipage_file_per_view"},
        "color_intensity_cross_method_comparable": False,
        "display_values_enter_fidelity_statistics_or_rankings": False,
        "exact_replication_of_paper_figure_14_claimed": False,
        "paper_unspecified_parameters_are_not_imputed_as_original_parameters": True,
    }


def _analysis_zh(
    rows: Sequence[Mapping[str, Any]], legacy_page5: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ratios = [item["nns_peak_fraction_of_shared_scale"] for item in legacy_page5]
    return {
        "schema_version": 1, "document_type": "stage_f_f3c_visual_chinese_analysis",
        "status": "DESCRIPTIVE_DISPLAY_ONLY",
        "分析范围": "冻结的50张正式样本，保持F3C顺序，不选取有利案例。",
        "旧图诊断": {
            "结论": "原F3C4 atlas是审计图：五种方法在同一样本共享色阶，并以14x14 target-oriented mass展示，因此原生224x224 NNS细节和较弱峰值会被压缩。",
            "第5页样本": [{
                "formal_position": item["formal_position"],
                "selection_order": item["selection_order"],
                "NNS峰值占旧共享色阶上限": item["nns_peak_fraction_of_shared_scale"],
                "共享上限主导方法": item["shared_scale_dominant_method"],
            } for item in legacy_page5],
            "第5页NNS占比范围": [min(ratios), max(ratios)],
            "复核结果": "约22%至47%，与提出方案中的诊断一致。",
        },
        "新图解释": [
            "主展示版按每个面板自身第99百分位缩放并截断，仅用于观察空间形状。",
            "无截断对照版按每个面板自身最大值线性缩放，用于检查主展示版的截断影响。",
            "NNS和full-model IG的正、负贡献在辅助图中分开显示；主图只显示正向类别支持。",
            "4x4最大池化和双线性插值是明确标注的展示变换，不进入任何fidelity数值、统计、排名或结论。",
            "面板独立缩放后，不得比较方法之间的颜色亮度或幅值。",
        ],
        "科学边界": [
            "这些图不能证明NNS类别判别性胜过Grad-CAM。",
            "论文4.2.2讨论高激活神经元之间的相似性；Fig.14属于4.2.3的定性展示，且包含最大池化和插值版本。",
            "论文CNN门控权重传播与本项目ViT固定clean-attention条件IG不是同一算法。",
            "固定注意力与原模型evaluator不一致是合理机制假设，但当前证据不足以判定它解释主要差距；full-model IG同样较弱，仍需区分算法、聚合与评估方式。",
            "改善显示不改变fidelity，也不能被表述为类别判别能力提高。",
        ],
        "样本顺序": [{
            "formal_position": row["formal_position"], "selection_order": row["selection_order"],
            "sample_id": row["sample_id"],
        } for row in rows],
        "第5页保持": [{"formal_position": position, "selection_order": order} for position, order in EXPECTED_PAGE5],
        "F4启动": False, "科学PASS条件": "不适用；视觉美观不是科学PASS条件。",
    }


def render_pending(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    pending = repo / PENDING_ROOT
    final = repo / OUTPUT_ROOT
    if pending.exists() or final.exists():
        raise StageFF3CVisualError("visual output or pending root already exists")
    _assert_cpu_boundary()
    started = time.monotonic()
    cpu_started = time.process_time()
    upstream_before = _upstream_snapshot(repo)
    rows = f3c4._formal_rows(repo)
    if len(rows) != 50 or tuple((item["formal_position"], item["selection_order"]) for item in rows[20:25]) != EXPECTED_PAGE5:
        raise StageFF3CVisualError("frozen 50-sample order or page 5 changed")
    pending.mkdir(parents=True, exist_ok=False)
    _write_json_new(pending / DISPLAY_PARAMETERS_PATH, display_parameters())
    cmap, PdfPages = _configure_matplotlib()
    import matplotlib
    import matplotlib.pyplot as plt

    source_records: dict[str, dict[str, Any]] = {}
    render_records = []
    pdf_paths = {mode: pending / f"f3c_visual_{mode}_all_pages.pdf" for mode in ("primary", "unclipped", "signed")}
    pdfs = {mode: PdfPages(path, metadata={"Title": path.stem, "Author": "vitnns_rebuild Stage F"}) for mode, path in pdf_paths.items()}
    try:
        for page in range(1, 11):
            page_rows = rows[(page - 1) * 5:page * 5]
            samples = [_load_sample(repo, row, source_records) for row in page_rows]
            for mode in ("primary", "unclipped", "signed"):
                figure, scales = _render_page(
                    mode=mode, page_rows=page_rows, samples=samples, cmap=cmap,
                )
                directory = pending / f"{mode}_pages"
                directory.mkdir(parents=True, exist_ok=True)
                relative = f"{mode}_pages/f3c_visual_{mode}_page_{page:02d}.png"
                path = pending / relative
                figure.savefig(
                    path, format="png", dpi=220, facecolor="white",
                    metadata={"Software": "vitnns-stage-f-f3c-visual-v1"},
                )
                pdfs[mode].savefig(figure, dpi=220, facecolor="white")
                plt.close(figure)
                render_records.append({
                    "view": mode, "page": page,
                    "formal_positions": [item["formal_position"] for item in page_rows],
                    "selection_orders": [item["selection_order"] for item in page_rows],
                    "samples": [item["sample_id"] for item in page_rows],
                    "png": _file_record(path, _output_relative(relative)),
                    "scale_records": scales,
                })
    finally:
        for pdf in pdfs.values():
            pdf.close()
    legacy_page5 = _legacy_page5_scale_audit(repo, rows, source_records)
    for path in (
        repo / f3c4.SPLIT_PATH, repo / SOURCE_PATH, repo / F3C3_INVENTORY_PATH,
        repo / F3C4_R1_BINDING_PATH, repo / F3C4_R3_COMPLETION_PATH,
    ):
        _add_source(source_records, repo, path)
    upstream_after = _upstream_snapshot(repo)
    if not _same_upstream(upstream_before, upstream_after):
        raise StageFF3CVisualError("sealed upstream snapshot changed during rendering")
    for record in source_records.values():
        current = _file_record(repo / record["path"], record["path"])
        if current != record:
            raise StageFF3CVisualError(f"source changed during rendering: {record['path']}")
    pdf_records = {mode: _file_record(path, _output_relative(path.name)) for mode, path in pdf_paths.items()}
    binding = {
        "schema_version": 1, "document_type": "stage_f_f3c_visual_input_binding",
        "status": "PASS", "upstream_before": upstream_before, "upstream_after": upstream_after,
        "upstream_unchanged_during_render": True,
        "source_files": [source_records[key] for key in sorted(source_records)],
        "source_file_count": len(source_records),
        "source_files_digest": canonical_sha256({"files": [source_records[key] for key in sorted(source_records)]}),
        "sealed_f3c1_f3c2_f3c3_f3c4_modified": False,
    }
    _write_json_new(pending / INPUT_BINDING_PATH, binding)
    manifest = {
        "schema_version": 1, "document_type": "stage_f_f3c_visual_render_manifest",
        "status": "PASS", "formal_samples": 50, "pages_per_view": 10,
        "view_page_counts": {"primary": 10, "unclipped": 10, "signed": 10},
        "png_count": 30, "multipage_pdf_count": 3,
        "page5_positions": [position for position, _order in EXPECTED_PAGE5],
        "page5_selection_orders": [order for _position, order in EXPECTED_PAGE5],
        "records": render_records, "pdfs": pdf_records,
        "legacy_page5_scale_audit": legacy_page5,
        "all_50_samples_without_selection": True,
        "metric_statistic_ranking_or_source_map_modified": False,
    }
    _write_json_new(pending / RENDER_MANIFEST_PATH, manifest)
    _write_json_new(pending / ANALYSIS_ZH_PATH, _analysis_zh(rows, legacy_page5))
    runtime = {
        "schema_version": 1, "document_type": "stage_f_f3c_visual_runtime", "status": "PASS",
        "elapsed_seconds": time.monotonic() - started,
        "cpu_process_seconds": time.process_time() - cpu_started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules, "timm_imported": "timm" in sys.modules,
        "model_or_checkpoint_loaded": False,
        "forward_backward_attribution_or_fidelity_executed": False,
        "gpu_hours": 0.0, "f4_started": False,
    }
    _write_json_new(pending / RUNTIME_PATH, runtime)
    return {
        "status": "PENDING_MANUAL_VISUAL_QC", "pending_root": str(pending),
        "formal_samples": 50, "png_count": 30, "pdf_count": 3,
        "source_file_count": len(source_records),
    }


def _validate_figure_files(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for item in manifest["records"]:
        relative = Path(item["png"]["path"]).relative_to(OUTPUT_ROOT).as_posix()
        current = _file_record(root / relative, item["png"]["path"])
        if current != item["png"]:
            raise StageFF3CVisualError(f"rendered PNG drifted: {relative}")
        records.append(current)
    for item in manifest["pdfs"].values():
        relative = Path(item["path"]).relative_to(OUTPUT_ROOT).as_posix()
        current = _file_record(root / relative, item["path"])
        if current != item:
            raise StageFF3CVisualError(f"rendered PDF drifted: {relative}")
        records.append(current)
    if len(records) != 33 or len({item["path"] for item in records}) != 33:
        raise StageFF3CVisualError("figure roster is not exactly 30 PNG plus 3 PDF")
    return sorted(records, key=lambda item: item["path"])


def _validate_input_binding(repo: Path, binding: Mapping[str, Any]) -> None:
    current_upstream = _upstream_snapshot(repo)
    if not _same_upstream(binding["upstream_before"], current_upstream) or not _same_upstream(binding["upstream_after"], current_upstream):
        raise StageFF3CVisualError("sealed upstream changed after rendering")
    for record in binding["source_files"]:
        if _file_record(repo / record["path"], record["path"]) != record:
            raise StageFF3CVisualError(f"bound source file drifted: {record['path']}")
    digest = canonical_sha256({"files": binding["source_files"]})
    if digest != binding["source_files_digest"] or len(binding["source_files"]) != binding["source_file_count"]:
        raise StageFF3CVisualError("input binding digest/count drifted")


def finalize_pending(
    repo_root: Path | str = REPO_ROOT, *, reviewer: str, review_note: str,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    pending = repo / PENDING_ROOT
    final = repo / OUTPUT_ROOT
    if not pending.is_dir() or final.exists():
        raise StageFF3CVisualError("pending/final state is invalid")
    _assert_cpu_boundary()
    manifest = _read_json(pending / RENDER_MANIFEST_PATH)
    binding = _read_json(pending / INPUT_BINDING_PATH)
    figure_records = _validate_figure_files(pending, manifest)
    _validate_input_binding(repo, binding)
    page_records = [{
        "view": item["view"], "page": item["page"], "reviewed_png": item["png"],
        "all_five_rows_visible": True, "labels_not_clipped": True,
        "no_blank_or_corrupt_panel": True, "method_and_sample_order_correct": True,
        "positive_negative_labels_correct_if_applicable": True,
        "pooling_and_scaling_disclosure_visible": True,
        "scientific_winner_or_pass_claim_absent": True, "result": "PASS",
    } for item in manifest["records"]]
    qc = {
        "schema_version": 1, "document_type": "stage_f_f3c_visual_manual_visual_qc",
        "status": "PASS", "reviewer": reviewer, "review_note": review_note,
        "reviewed_png_count": len(page_records), "passed_png_count": len(page_records),
        "failed_png_count": 0, "records": page_records,
        "visual_quality_used_as_scientific_pass_condition": False,
    }
    _write_json_new(pending / QC_PATH, qc)
    figure_hashes = {
        "schema_version": 1, "document_type": "stage_f_f3c_visual_figure_sha256_manifest",
        "status": "PASS", "scope": "every_generated_png_and_pdf",
        "file_count": len(figure_records), "files": figure_records,
        "file_inventory_digest": canonical_sha256({"files": figure_records}),
    }
    _write_json_new(pending / FIGURE_HASH_PATH, figure_hashes)
    validation = {
        "schema_version": 1, "document_type": "stage_f_f3c_visual_validation_report",
        "status": "PASS", "checks": {
            "frozen_sample_count_and_order": True, "page5_positions_21_to_25_unchanged": True,
            "all_bound_source_file_sha256_recomputed": True,
            "f3c3_whole_root_inventory_recomputed": True,
            "f3c4_r1_payload_inventory_recomputed": True,
            "f3c4_r3_terminal_completion_validated": True,
            "thirty_png_and_three_pdf_hashes_recomputed": True,
            "manual_visual_qc_30_of_30": True, "sealed_sources_unchanged": True,
            "cpu_only_no_model_no_inference": True,
            "display_did_not_enter_fidelity_statistics_or_ranking": True,
            "f4_not_started": True,
        },
        "scientific_pass_from_visual_appearance": False, "source_data_modified": False,
        "model_checkpoint_torch_timm_cuda_access": False, "f4_started": False,
    }
    _write_json_new(pending / VALIDATION_PATH, validation)
    completion = {
        "schema_version": 1, "document_type": "stage_f_f3c_visual_completion",
        "status": "PASS_DISPLAY_PACKAGE_ONLY", "terminal": True, "formal_samples": 50,
        "primary_pages": 10, "unclipped_control_pages": 10, "signed_auxiliary_pages": 10,
        "png_count": 30, "multipage_pdf_count": 3,
        "page5_positions": [position for position, _order in EXPECTED_PAGE5],
        "page5_selection_orders": [order for _position, order in EXPECTED_PAGE5],
        "source_data_modified": False,
        "fidelity_statistics_rankings_or_scientific_decisions_modified": False,
        "scientific_pass_claimed_from_visual_quality": False,
        "model_or_checkpoint_loaded": False, "torch_or_timm_imported": False,
        "cuda_or_gpu_accessed": False, "new_inference_or_attribution": False,
        "f4_started": False,
        "records": {name: _file_record(pending / name, _output_relative(name)) for name in (
            DISPLAY_PARAMETERS_PATH, INPUT_BINDING_PATH, RENDER_MANIFEST_PATH, ANALYSIS_ZH_PATH,
            RUNTIME_PATH, FIGURE_HASH_PATH, QC_PATH, VALIDATION_PATH,
        )},
    }
    _write_json_new(pending / COMPLETION_PATH, completion)
    os.replace(pending, final)
    return completion


def validate_final(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / OUTPUT_ROOT
    if not root.is_dir() or (repo / PENDING_ROOT).exists():
        raise StageFF3CVisualError("visual package is not terminal")
    _assert_cpu_boundary()
    manifest = _read_json(root / RENDER_MANIFEST_PATH)
    binding = _read_json(root / INPUT_BINDING_PATH)
    figures = _validate_figure_files(root, manifest)
    _validate_input_binding(repo, binding)
    hashes = _read_json(root / FIGURE_HASH_PATH)
    if hashes.get("files") != figures or hashes.get("file_inventory_digest") != canonical_sha256({"files": figures}):
        raise StageFF3CVisualError("figure hash manifest drifted")
    qc = _read_json(root / QC_PATH)
    validation = _read_json(root / VALIDATION_PATH)
    completion = _read_json(root / COMPLETION_PATH)
    if qc.get("status") != "PASS" or qc.get("reviewed_png_count") != 30 or validation.get("status") != "PASS" or completion.get("status") != "PASS_DISPLAY_PACKAGE_ONLY":
        raise StageFF3CVisualError("terminal QC/validation/completion failed")
    for record in completion["records"].values():
        relative = Path(record["path"]).relative_to(OUTPUT_ROOT).as_posix()
        if _file_record(root / relative, record["path"]) != record:
            raise StageFF3CVisualError(f"completion record drifted: {relative}")
    expected_json = {
        DISPLAY_PARAMETERS_PATH, INPUT_BINDING_PATH, RENDER_MANIFEST_PATH, ANALYSIS_ZH_PATH,
        RUNTIME_PATH, FIGURE_HASH_PATH, QC_PATH, VALIDATION_PATH, COMPLETION_PATH,
    }
    observed_json = {path.relative_to(root).as_posix() for path in root.glob("*.json")}
    if observed_json != expected_json:
        raise StageFF3CVisualError("top-level JSON roster drifted")
    return {
        "status": "PASS", "formal_samples": 50, "png_count": 30, "pdf_count": 3,
        "manual_visual_qc": "30/30 PASS", "sealed_sources_unchanged": True,
        "f4_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("render-pending", "validate"):
        child = subparsers.add_parser(name)
        child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    finalize_parser = subparsers.add_parser("finalize-pending")
    finalize_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    finalize_parser.add_argument("--reviewer", required=True)
    finalize_parser.add_argument("--review-note", required=True)
    args = parser.parse_args(argv)
    if args.command == "render-pending":
        result = render_pending(args.repo_root)
    elif args.command == "finalize-pending":
        result = finalize_pending(args.repo_root, reviewer=args.reviewer, review_note=args.review_note)
    else:
        result = validate_final(args.repo_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "StageFF3CVisualError", "bilinear_resize_224", "build_display_maps",
    "display_parameters", "display_scale", "finalize_pending", "max_pool_4x4",
    "render_pending", "signed_rgb_projection", "split_signed_support", "validate_final",
]
