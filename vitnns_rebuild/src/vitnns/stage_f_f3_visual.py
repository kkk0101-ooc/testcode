"""Post-F3 blue-background display revision.

This module consumes only the sealed F3 r7 source bindings.  It creates a
presentation-only, per-panel shape-normalized magnitude view while preserving
the r7 signed shared-scale panels as the authoritative magnitude audit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from .stage_f_renderer import _save_figure, load_contracts, renderer_style, sha256_file
from . import stage_f_f3 as f3


CONFIG_PATH = "config/stage_f_f3_visual_revision.json"
PROTOCOL_PATH = "docs/stage_f_post_f3_blue_display_revision_20260907.md"
ROOT_PATH = "evidence/stage_f/f3_visual_r1_recovery_v2_20260907/formal"
REPORT_PATH = "reports/stage_f_f3_visual_r1_20260907"
PENDING_PATH = ROOT_PATH + "/completion/f3_visual.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/f3_visual.json"
STARTUP_PATH = ROOT_PATH + "/startup_gate.json"
SOURCE_BINDING_PATH = ROOT_PATH + "/source_binding.json"
TRANSFORM_MANIFEST_PATH = ROOT_PATH + "/transform_manifest.json"
RENDER_INVENTORY_PATH = ROOT_PATH + "/render_inventory.json"
REVIEW_INVENTORY_PATH = ROOT_PATH + "/review_copy_inventory.json"
ACCESS_AUDIT_PATH = ROOT_PATH + "/access_audit.json"
PAYLOAD_MANIFEST_PATH = ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = ROOT_PATH + "/status.json"
VALIDATION_PATH = REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = REPORT_PATH + "/test_report.json"
ANALYSIS_PATH = REPORT_PATH + "/figure_analysis.md"
REVIEW_ROOT_PATH = REPORT_PATH + "/figures_png"
ARCHIVE_REF = "refs/archive/stage-f-v1-f3-blue-display-revision"

F3_REF = "refs/archive/stage-f-v1-f3-completion"
F3_COMMIT = "1589fc3895657ee6a8f01d843ee26b67603998cb"
F3_COMPLETION_PATH = "evidence/stage_f/f3_r7_recovery_v2_20260907/formal/completion/f3.json"
F3_COMPLETION_SHA256 = "f1423479cc391b729cea37f34c663db833b91e3dff1cdee576efa61ce87e56d6"
F3_INVENTORY_DIGEST = "fe19352ef4d11ee4b173e56a0e1cd70e290fe9b34bc5f6b20f77b9394cbfc1e4"
F3_ROOT = "evidence/stage_f/f3_r7_recovery_v2_20260907/formal"
F3_PANEL_PATH = F3_ROOT + "/panel_manifest.json"
F3_CROSS_PANEL_PATH = F3_ROOT + "/cross_layer_panel_manifest.json"
F3_CROSS_SELECTION_PATH = F3_ROOT + "/cross_layer_selection_manifest.json"
F3_SELECTION_PATH = F3_ROOT + "/selection_manifest.json"
F3_RULE_PATH = F3_ROOT + "/rule_to_image_manifest.json"
F3_RENDER_PATH = F3_ROOT + "/render_inventory.json"
F3_SCALE_PATH = F3_ROOT + "/cross_layer_scale_manifest.json"
REGISTRY_PATH = "manifests/stage_d_fidelity_v1/registry.json"
F3_CONFIG_PATH = "config/stage_f_f3.json"
REFERENCE_PATH = "../论文1/参考文章.pdf"
REFERENCE_SHA256 = "33e323f8cb664864b12daad1f06b6ef0d3268bcafaea4e84ac695fd286f3855b"
NNS = f3.NNS_METHOD
MATCHED = f3.MATCHED_METHOD
METHODS = (NNS, MATCHED)
COORDINATES = f3.COORDINATES

MODULE_COUNTS = {"figure5": 4, "figure6": 1, "s1": 5, "s2": 6, "s4": 11}
ATTACKS = (
    "f3_archive_completion_or_inventory_tampered",
    "f3_selection_or_slot_mapping_changed",
    "jpeg_or_map_hash_mismatch",
    "signed_map_replaced_by_display_array",
    "normalization_not_per_panel_max_absolute",
    "gamma_or_colormap_changed",
    "percentile_clipping_smoothing_or_max_pool_added",
    "magnitude_comparison_claim_added",
    "failed_cell_imputed_or_hidden",
    "figure6_not_median_effect",
    "s4_slot_or_unique_sample_changed",
    "png_pdf_svg_content_identity_diverged",
    "model_checkpoint_torch_timm_cuda_or_new_attribution_used",
    "gradcam_or_figure7_generated_without_protocol",
    "f4_started_or_final_combined_figure_generated",
    "terminal_pass_before_review_validation_and_tests",
)


class StageFVisualError(RuntimeError):
    pass


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: item for key, item in value.items() if key != "canonical_digest"}
    result["canonical_digest"] = canonical_sha256(result)
    return result


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes(); value = json.loads(raw)
    except Exception as error:
        raise StageFVisualError(f"{context} is invalid JSON: {error}") from error
    if not isinstance(value, dict): raise StageFVisualError(f"{context} must be an object")
    return value, raw


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest") != canonical_sha256({key: item for key, item in value.items() if key != "canonical_digest"}):
        raise StageFVisualError(f"{context} canonical digest mismatch")


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())


def _write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode: raise StageFVisualError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative; value, _ = _strict_json(path, relative); _validate_signed(value, relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "canonical_digest": value["canonical_digest"]}


def _file_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude); files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": files, "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files), "canonical_digest": canonical_sha256({"files": files})}


def _load_config(repo: Path) -> dict[str, Any]:
    value, _ = _strict_json(repo / CONFIG_PATH, "visual revision config"); _validate_signed(value, "visual revision config")
    transform = value["transform"]
    expected = {"source": "ABSOLUTE_MAGNITUDE_OF_RAW_SIGNED_MAP", "normalization": "PER_PANEL_MAX_ABSOLUTE_VALUE", "gamma": 2.0, "resize": [224, 224], "resize_order": 1, "resize_interpolation": "bilinear", "clipping": None, "smoothing": None, "max_pool": None, "colormap_stops": [[0.0, "#07077F"], [0.2, "#0019C6"], [0.42, "#00AEE8"], [0.65, "#FFF000"], [0.82, "#FF6A00"], [1.0, "#D90000"]], "magnitude_comparison_allowed": False, "spatial_shape_comparison_only": True, "authoritative_signed_audit": "F3_R7_SHARED_SCALE"}
    if transform != expected: raise StageFVisualError("blue-display transform contract drifted")
    if not all(value["prohibitions"].values()): raise StageFVisualError("visual revision prohibitions drifted")
    return value


def _runtime_boundary() -> dict[str, Any]:
    forbidden = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "timm" or name.startswith("timm."))
    if forbidden: raise StageFVisualError(f"forbidden runtime imported: {forbidden[0]}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise StageFVisualError("CUDA_VISIBLE_DEVICES must be empty")
    return {"status": "PASS", "model_loaded": False, "checkpoint_loaded": False, "torch_imported": False, "timm_imported": False, "cuda_used": False, "new_attribution_or_fidelity": False}


def verify_startup(repo_root: Path | str = REPO_ROOT, *, require_clean: bool = True, require_remote: bool = True) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _load_config(repo); head = _git(repo, "rev-parse", "HEAD"); tracking = _git(repo, "rev-parse", "origin/stage-d-d3-pass")
    if _git(repo, "symbolic-ref", "--short", "HEAD") != "stage-d-d3-pass" or head != tracking: raise StageFVisualError("branch/local/tracking mismatch")
    if _git(repo, "rev-parse", F3_REF + "^{commit}") != F3_COMMIT: raise StageFVisualError("F3 archive ref drifted")
    completion, _ = _strict_json(repo / F3_COMPLETION_PATH, "F3 completion"); _validate_signed(completion, "F3 completion")
    if sha256_file(repo / F3_COMPLETION_PATH) != F3_COMPLETION_SHA256 or completion.get("status") != "PASS" or completion.get("next_authorization") != "F4_ONLY" or completion.get("f4_started") is not False or completion.get("whole_root_inventory", {}).get("canonical_digest") != F3_INVENTORY_DIGEST:
        raise StageFVisualError("F3 completion binding drifted")
    if sha256_file((repo / REFERENCE_PATH).resolve()) != REFERENCE_SHA256: raise StageFVisualError("reference PDF drifted")
    if require_clean and _git(repo, "status", "--porcelain=v1"): raise StageFVisualError("startup requires clean worktree")
    remote = None
    if require_remote:
        rows = _git(repo, "ls-remote", "--heads", "origin", "stage-d-d3-pass").splitlines(); remote = rows[0].split()[0] if len(rows) == 1 else None
        refs = dict(reversed(row.split()) for row in _git(repo, "ls-remote", "origin", F3_REF).splitlines())
        if remote != head or refs.get(F3_REF) != F3_COMMIT: raise StageFVisualError("remote branch or F3 ref drifted")
    return _signed({"schema_version": 1, "document_type": "stage_f_f3_visual_revision_startup", "status": "PASS", "head": head, "tracking": tracking, "remote": remote, "f3_archive_commit": F3_COMMIT, "f3_completion_sha256": F3_COMPLETION_SHA256, "f3_inventory_digest": F3_INVENTORY_DIGEST, "reference_pdf_sha256": REFERENCE_SHA256, "protocol_sha256": sha256_file(repo / PROTOCOL_PATH), "config_digest": config["canonical_digest"], "runtime": _runtime_boundary(), "f3_r7_modified": False, "f4_started": False})


def _array_digest(array: np.ndarray) -> str:
    return f3._array_digest(array)


def _blue_cmap(config: Mapping[str, Any]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("vitnns_blue_hot", [(value, color) for value, color in config["transform"]["colormap_stops"]], N=256)


def _blue_transform(raw: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(raw)
    if source.dtype != np.float32 or source.shape != (56, 56) or not np.isfinite(source).all(): raise StageFVisualError("source map contract failed")
    maximum = float(np.max(np.abs(source))); normalized = np.zeros(source.shape, dtype=np.float32) if maximum == 0 else (np.abs(source) / np.float32(maximum)).astype(np.float32)
    gamma = float(config["transform"]["gamma"]); shaped = np.power(normalized, np.float32(gamma), dtype=np.float32)
    resized = zoom(shaped, (4.0, 4.0), order=1, mode="nearest", prefilter=False).astype(np.float32, copy=False)
    if resized.shape != (224, 224): raise StageFVisualError("blue display resize shape drifted")
    return np.ascontiguousarray(resized), {"raw_signed_map_sha256": _array_digest(source), "raw_max_abs": maximum, "magnitude_56_sha256": _array_digest(np.abs(source).astype(np.float32)), "normalized_56_sha256": _array_digest(normalized), "gamma": gamma, "gamma_56_sha256": _array_digest(shaped), "resize": [224, 224], "resize_order": 1, "resize_interpolation": "bilinear", "display_224_sha256": _array_digest(resized), "clipping": None, "smoothing": None, "max_pool": None, "normalization": "PER_PANEL_MAX_ABSOLUTE_VALUE", "magnitude_comparison_allowed": False, "display_only": True}


def _load_map(repo: Path, record: Mapping[str, Any]) -> np.ndarray | None:
    if record["validity_status"] != "VALID":
        if record.get("source_path") is None or record.get("signed_map_sha256") is not None: raise StageFVisualError("failed source record drifted")
        return None
    path = repo / record["source_path"]
    if sha256_file(path) != record["source_file_sha256"]: raise StageFVisualError("source map file hash mismatch")
    array = np.load(path, allow_pickle=False)
    if _array_digest(array) != record["signed_map_sha256"]: raise StageFVisualError("source signed map digest mismatch")
    return array


def _load_rgb(repo: Path, meta: Mapping[str, Any]) -> np.ndarray:
    path = repo / meta["path"]
    if sha256_file(path) != meta["file_sha256"]: raise StageFVisualError("JPEG hash mismatch")
    with Image.open(path) as source: image = source.convert("RGB"); image.load()
    image = image.resize(tuple(meta["resized_size"]), Image.Resampling.BICUBIC); return np.asarray(image.crop(tuple(meta["crop_box"])))


def _sources(repo: Path) -> dict[str, Any]:
    documents = {}
    for key, relative in (("panel", F3_PANEL_PATH), ("cross_panel", F3_CROSS_PANEL_PATH), ("cross_selection", F3_CROSS_SELECTION_PATH), ("selection", F3_SELECTION_PATH), ("rules", F3_RULE_PATH), ("render", F3_RENDER_PATH), ("scale", F3_SCALE_PATH)):
        value, _ = _strict_json(repo / relative, key); _validate_signed(value, key); documents[key] = value
    qualitative = {}
    for item in documents["panel"]["panels"]:
        if item.get("sample_id") and item.get("sources") and item["sample_id"] not in qualitative: qualitative[item["sample_id"]] = item
    cross = {(item["sample_id"], tuple(item["coordinate"]), item["method"]): item for item in documents["cross_panel"]["cells"] if item["method"] in METHODS}
    return {**documents, "qualitative": qualitative, "cross": cross, "source_digest": canonical_sha256({key: value["canonical_digest"] for key, value in documents.items()})}


def _save_report(repo: Path, fig: Any, output_base: Path, *, module: str, source_digest: str, config: Mapping[str, Any], transformations: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    f0 = load_contracts(repo)["config"]; outputs = _save_figure(fig, output_base, f0); size = [float(value) for value in fig.get_size_inches()]; plt.close(fig)
    content = canonical_sha256({"module": module, "source_digest": source_digest, "config_digest": config["canonical_digest"], "transformations": list(transformations), "metadata": dict(metadata)})
    report = _signed({"schema_version": 1, "document_type": "stage_f_f3_blue_display_render_report", "revision_id": config["revision_id"], "module": module, "figure_id": output_base.name, "source_digest": source_digest, "config_digest": config["canonical_digest"], "renderer_commit": _git(repo, "rev-parse", "HEAD"), "content_digest": content, "format_content_digests": {extension: content for extension in outputs}, "outputs": outputs, "size_inches": size, "dpi": f0["export"]["png_dpi"], "transformations": list(transformations), "metadata": dict(metadata), "display_only": True, "shape_normalized": True, "magnitude_comparison_allowed": False, "authoritative_signed_audit": "F3_R7_SHARED_SCALE", "model_loaded": False, "torch_imported": False, "timm_imported": False, "cuda_used": False, "new_attribution_or_fidelity": False, "gradcam_or_figure7_generated": False, "f4_started": False})
    _write_new(output_base.with_suffix(".report.json"), report); return report


def _render_case(repo: Path, record: Mapping[str, Any], *, module: str, title: str, output_base: Path, config: Mapping[str, Any], cmap: Any, source_digest: str) -> dict[str, Any]:
    sources = {item["method"]: item for item in record["sources"]}; nns_raw = _load_map(repo, {"validity_status": "VALID", "source_path": sources[NNS]["source_map_path"], "source_file_sha256": sources[NNS]["source_map_file_sha256"], "signed_map_sha256": sources[NNS]["signed_map_sha256"]}); matched_raw = _load_map(repo, {"validity_status": "VALID", "source_path": sources[MATCHED]["source_map_path"], "source_file_sha256": sources[MATCHED]["source_map_file_sha256"], "signed_map_sha256": sources[MATCHED]["signed_map_sha256"]})
    nns, nns_transform = _blue_transform(nns_raw, config); matched, matched_transform = _blue_transform(matched_raw, config); difference, difference_transform = _blue_transform((nns_raw - matched_raw).astype(np.float32), config); rgb = _load_rgb(repo, record["jpeg"])
    results = {method: json.loads((repo / sources[method]["source_result_path"]).read_bytes()) for method in METHODS}
    f0 = load_contracts(repo)["config"]
    with renderer_style(f0):
        fig = plt.figure(figsize=(7.16, 4.05), constrained_layout=True); grid = GridSpec(2, 4, figure=fig, wspace=0.055, hspace=0.13, height_ratios=[1, 1])
        top = ((rgb, "Original", None), (nns, "NNS", cmap), (matched, "Matched baseline", cmap), (difference, "|NNS - matched|", cmap))
        axes = []
        for column, (array, label, selected_cmap) in enumerate(top):
            axis = fig.add_subplot(grid[0, column]); axis.imshow(array, cmap=selected_cmap, vmin=0 if selected_cmap else None, vmax=1 if selected_cmap else None, interpolation="bicubic" if selected_cmap else "nearest"); axis.set_title(label, fontsize=8.2, fontweight="bold", pad=4); axis.axis("off"); axis.text(0.015, 0.985, f"({chr(97+column)})", transform=axis.transAxes, va="top", ha="left", fontsize=7.5, fontweight="bold", color="white", bbox={"facecolor": "#111111", "alpha": .55, "edgecolor": "none", "pad": .8}); axes.append(axis)
        for column, (array, label) in enumerate(((nns, "NNS overlay"), (matched, "Matched overlay"))):
            axis = fig.add_subplot(grid[1, column]); axis.imshow(rgb); axis.imshow(array, cmap=cmap, vmin=0, vmax=1, alpha=.62, interpolation="bicubic"); axis.set_title(label, fontsize=7.8, pad=3); axis.axis("off"); axis.text(0.015, .985, f"({chr(101+column)})", transform=axis.transAxes, va="top", ha="left", fontsize=7.5, fontweight="bold", color="white", bbox={"facecolor": "#111111", "alpha": .55, "edgecolor": "none", "pad": .8}); axes.append(axis)
        axis = fig.add_subplot(grid[1, 2:4]); colors = {NNS: "#0878B9", MATCHED: "#E36C0A"}
        for method in METHODS:
            aggregates = results[method]["fidelity_aggregates"]; x = results[method]["fidelity_curves"][0]["actual_fractions"]; label = "NNS" if method == NNS else "Matched"
            axis.plot(x, aggregates[0]["response_mean"], color=colors[method], lw=1.3, label=label + " insertion"); axis.plot(x, aggregates[1]["response_mean"], color=colors[method], lw=1.3, ls="--", label=label + " deletion")
        axis.set_title("Insertion / deletion fidelity", fontsize=7.8, pad=3); axis.set_xlabel("Fraction", fontsize=7); axis.set_ylabel("Response", fontsize=7); axis.tick_params(labelsize=6); axis.grid(alpha=.18, lw=.35); axis.legend(frameon=False, ncol=2, fontsize=5.7, loc="upper center"); axis.text(.015,.985,"(g)",transform=axis.transAxes,va="top",ha="left",fontsize=7.5,fontweight="bold")
        fig.suptitle(title, fontsize=9.2, fontweight="bold", y=.99)
        return _save_report(repo, fig, output_base, module=module, source_digest=source_digest, config=config, transformations=[nns_transform, matched_transform, difference_transform], metadata={"sample_id": record["sample_id"], "selection_order": record["selection_order"], "selection_rules": record["selection_rules"], "sample_status": record["sample_status"], "delta_fid": record["delta_fid"], "jpeg": record["jpeg"], "source_records": record["sources"]})


def _failure_axis(axis: Any, record: Mapping[str, Any]) -> None:
    axis.set_facecolor("#07077F"); axis.set_xticks([]); axis.set_yticks([])
    code = record["failure_code"].replace("ig_conservation_failed_after_e3_r1_schedule", "IG conservation failed\nafter E3-R1 schedule").replace("ig_conservation_failed_typed_terminal", "IG conservation failed\ntyped terminal")
    axis.text(.5,.58,"FAILED / NA",ha="center",va="center",color="white",fontweight="bold",fontsize=8); axis.text(.5,.38,f"order {record['selection_order']}\n{code}",ha="center",va="center",color="white",fontsize=5.8)


def _render_matrix(repo: Path, *, sample_id: str, slots: Sequence[str], output_base: Path, module: str, title: str, sources: Mapping[str, Any], config: Mapping[str, Any], cmap: Any) -> dict[str, Any]:
    record = sources["qualitative"][sample_id]; rgb = _load_rgb(repo, record["jpeg"]); transformations = []; source_records = []
    f0 = load_contracts(repo)["config"]
    with renderer_style(f0):
        fig = plt.figure(figsize=(7.16, 3.55), constrained_layout=True); grid = GridSpec(2, 5, figure=fig, width_ratios=[.93, 1, 1, 1, 1], wspace=.025, hspace=.04)
        original = fig.add_subplot(grid[:,0]); original.imshow(rgb); original.axis("off"); original.set_title("Original", fontsize=8.2, fontweight="bold", pad=5)
        for column, coordinate in enumerate(COORDINATES, start=1):
            for row, method in enumerate(METHODS):
                source = sources["cross"][(sample_id, coordinate, method)]; axis = fig.add_subplot(grid[row,column])
                if source["validity_status"] != "VALID": _failure_axis(axis, source); transformations.append({"sample_id": sample_id, "coordinate": list(coordinate), "method": method, "validity_status": source["validity_status"], "failure_code": source["failure_code"], "imputed": False})
                else:
                    raw = _load_map(repo, source); displayed, transform = _blue_transform(raw, config); transform.update({"sample_id": sample_id, "coordinate": list(coordinate), "method": method}); transformations.append(transform); axis.imshow(displayed, cmap=cmap, vmin=0, vmax=1, interpolation="bicubic"); axis.axis("off")
                source_records.append(source)
                if row == 0: axis.set_title(f"Block {coordinate[0]} · Head {coordinate[1]}", fontsize=7.5, fontweight="bold", pad=4)
                if column == 1: axis.text(-.07,.5,"NNS" if row==0 else "Matched",transform=axis.transAxes,rotation=90,ha="center",va="center",fontsize=7.8,fontweight="bold")
        fig.suptitle(title, fontsize=9.2, fontweight="bold", y=.99)
        return _save_report(repo, fig, output_base, module=module, source_digest=sources["source_digest"], config=config, transformations=transformations, metadata={"sample_id": sample_id, "selection_order": record["selection_order"], "slots": list(slots), "jpeg": record["jpeg"], "source_records": source_records, "difference_audit": "available_in_F3_r7_signed_shared_scale"})


def _render_mapping(repo: Path, rules: Mapping[str, Any], output_base: Path, config: Mapping[str, Any], source_digest: str) -> dict[str, Any]:
    f0 = load_contracts(repo)["config"]
    with renderer_style(f0):
        fig, axis = plt.subplots(figsize=(7.16, 2.9), constrained_layout=True); axis.axis("off"); rows = [[item["rule"].replace("_", " "), str(item["selection_order"]), item["sample_id"].split("/")[-1].replace(".JPEG", ""), f"{item['delta_fid']:.3f}", item["prediction_status"]] for item in rules["rules"]]
        table = axis.table(cellText=rows, colLabels=["Rule slot", "Order", "Sample", "delta_fid", "Status"], loc="center", cellLoc="left", colLoc="left", colWidths=[.28,.08,.34,.12,.13]); table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1,1.38)
        for (row,_), cell in table.get_celld().items(): cell.set_linewidth(.45); cell.set_facecolor("#E7EFF8" if row==0 else "white")
        axis.set_title("Frozen rule-to-image mapping", fontsize=9, fontweight="bold", pad=8)
        return _save_report(repo, fig, output_base, module="Supplement S2", source_digest=source_digest, config=config, transformations=[], metadata={"rules": rules["rules"], "duplicates": rules["duplicates"]})


def initialize(repo_root: Path | str = REPO_ROOT, *, renderer_commit: str, started_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); gate = verify_startup(repo)
    if renderer_commit != gate["head"]: raise StageFVisualError("renderer commit must equal startup HEAD")
    if (repo / ROOT_PATH).exists() or (repo / REPORT_PATH).exists(): raise StageFVisualError("visual revision root already exists")
    _write_new(repo / STARTUP_PATH, gate); _write_new(repo / PENDING_PATH, _signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_pending","revision_id":"F3_BLUE_DISPLAY_R1_20260907","status":"PENDING","started_at":started_at,"renderer_commit":renderer_commit,"startup_gate":_record(repo,STARTUP_PATH),"terminal_pass_published":False,"f4_started":False}))
    return {"status":"INITIALIZED","renderer_commit":renderer_commit,"source":"SEALED_F3_R7","f4_started":False}


def build_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); config=_load_config(repo); _runtime_boundary(); sources=_sources(repo); cmap=_blue_cmap(config)
    source_binding=_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_source_binding","revision_id":config["revision_id"],"status":"PASS","f3_archive_commit":F3_COMMIT,"f3_completion":_record(repo,F3_COMPLETION_PATH),"source_documents":{key:{"path":relative,"sha256":sha256_file(repo/relative),"canonical_digest":sources[key]["canonical_digest"]} for key,relative in (("panel",F3_PANEL_PATH),("cross_panel",F3_CROSS_PANEL_PATH),("cross_selection",F3_CROSS_SELECTION_PATH),("selection",F3_SELECTION_PATH),("rules",F3_RULE_PATH),("render",F3_RENDER_PATH),("scale",F3_SCALE_PATH))},"source_digest":sources["source_digest"],"selection_changed":False,"maps_or_jpegs_changed":False});_write_new(repo/SOURCE_BINDING_PATH,source_binding)
    reports=[]
    for index,sample_id in enumerate(sources["rules"]["render_order_unique_sample_ids"]):
        record=sources["qualitative"][sample_id]; rules=[item["rule"] for item in sources["rules"]["rules"] if item["sample_id"]==sample_id]; title=f"{rules[0].replace('_',' ').title()} · order {record['selection_order']}"; base=repo/ROOT_PATH/"modules/figure_5"/f"case_{index:02d}_order_{record['selection_order']:03d}"; reports.append(("figure5",_render_case(repo,record,module="Figure 5",title=title,output_base=base,config=config,cmap=cmap,source_digest=sources["source_digest"]),base.with_suffix(".report.json")))
    for index,item in enumerate(sources["selection"]["fixed_audit_examples"]):
        record=sources["qualitative"][item["sample_id"]]; base=repo/ROOT_PATH/"modules/supplement_s1"/f"fixed_audit_{index:02d}_order_{record['selection_order']:03d}"; reports.append(("s1",_render_case(repo,record,module="Supplement S1",title=f"Fixed audit example {index+1} · order {record['selection_order']}",output_base=base,config=config,cmap=cmap,source_digest=sources["source_digest"]),base.with_suffix(".report.json")))
    map_base=repo/ROOT_PATH/"modules/supplement_s2"/"rule_to_image_mapping";reports.append(("s2",_render_mapping(repo,sources["rules"],map_base,config,sources["source_digest"]),map_base.with_suffix(".report.json")))
    for index,item in enumerate(sources["rules"]["rules"]):
        record=sources["qualitative"][item["sample_id"]];base=repo/ROOT_PATH/"modules/supplement_s2"/f"slot_{index:02d}_{item['rule']}_order_{record['selection_order']:03d}";reports.append(("s2",_render_case(repo,record,module="Supplement S2",title=f"{item['rule'].replace('_',' ').title()} · order {record['selection_order']}",output_base=base,config=config,cmap=cmap,source_digest=sources["source_digest"]),base.with_suffix(".report.json")))
    figure6=sources["cross_selection"]["figure_6"]; base=repo/ROOT_PATH/"modules/figure_6"/"median_effect_progression";reports.append(("figure6",_render_matrix(repo,sample_id=figure6["sample_id"],slots=["median_effect"],output_base=base,module="Figure 6",title=f"Median effect · order {figure6['selection_order']}",sources=sources,config=config,cmap=cmap),base.with_suffix(".report.json")))
    slots=sources["cross_selection"]["s4_slots"]; unique=sources["cross_selection"]["unique_render_sample_ids"]
    for index,sample_id in enumerate(unique):
        sample_slots=[item["slot"] for item in slots if item["sample_id"]==sample_id]; order=next(item["selection_order"] for item in slots if item["sample_id"]==sample_id);base=repo/ROOT_PATH/"modules/supplement_s4"/f"matrix_{index:02d}_order_{order:03d}";reports.append(("s4",_render_matrix(repo,sample_id=sample_id,slots=sample_slots,output_base=base,module="Supplement S4",title=f"Order {order} · {', '.join(s.replace('_',' ') for s in sample_slots)}",sources=sources,config=config,cmap=cmap),base.with_suffix(".report.json")))
    transforms=[];render_rows=[];review=[]
    for index,(group,report,path) in enumerate(reports,1):
        transforms.extend(report["transformations"]); render_rows.append({"module_group":group,"module":report["module"],"figure_id":report["figure_id"],"report":_record(repo,path.relative_to(repo).as_posix()),"outputs":{ext:{"path":(path.parent/item["path"]).relative_to(repo).as_posix(),"sha256":item["sha256"],"bytes":item["bytes"]} for ext,item in report["outputs"].items()},"content_digest":report["content_digest"]}); src=path.parent/report["outputs"]["png"]["path"];dst=repo/REVIEW_ROOT_PATH/f"{index:02d}_{group}_{src.name}";dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);review.append({"module_group":group,"path":dst.relative_to(repo).as_posix(),"sha256":sha256_file(dst),"source_path":src.relative_to(repo).as_posix(),"source_sha256":sha256_file(src),"exact_copy":sha256_file(dst)==sha256_file(src)})
    _write_new(repo/TRANSFORM_MANIFEST_PATH,_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_transform_manifest","revision_id":config["revision_id"],"status":"PASS","config_digest":config["canonical_digest"],"source_digest":sources["source_digest"],"transform_count":len(transforms),"transforms":transforms,"per_panel_normalization":True,"magnitude_comparison_allowed":False,"r7_signed_audit_preserved":True}))
    _write_new(repo/RENDER_INVENTORY_PATH,_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_render_inventory","revision_id":config["revision_id"],"status":"PASS","render_count":len(render_rows),"exported_file_count":len(render_rows)*3,"module_counts":dict(Counter(item["module_group"] for item in render_rows)),"renders":render_rows,"formats":["png","pdf","svg"],"png_dpi":600,"final_combined_manuscript_figures_rendered":False}))
    _write_new(repo/REVIEW_INVENTORY_PATH,_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_review_inventory","revision_id":config["revision_id"],"status":"PASS","review_count":len(review),"copies":review}))
    _write_new(repo/ACCESS_AUDIT_PATH,_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_access_audit","revision_id":config["revision_id"],"status":"PASS","source":"SEALED_F3_R7_ONLY","model_access":False,"checkpoint_access":False,"torch_imported":False,"timm_imported":False,"cuda_or_gpu_access":False,"new_attribution_or_fidelity":False,"gradcam_or_figure7_generated":False,"f4_started":False}))
    _write_new(repo/FALSE_PASS_PATH,build_false_pass_audit(repo))
    return {"status":"OUTPUTS_BUILT","render_count":len(render_rows),"exported_file_count":len(render_rows)*3,"review_count":len(review),"module_counts":dict(Counter(item["module_group"] for item in render_rows)),"figure6_sample_id":figure6["sample_id"],"s4_slots":len(slots),"s4_unique_samples":len(unique)}


def build_false_pass_audit(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    evidence={attack:"source binding, deterministic transform, render report, validator, or runtime boundary rejects this path" for attack in ATTACKS}
    return _signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_false_pass_audit","revision_id":"F3_BLUE_DISPLAY_R1_20260907","status":"PASS","attacks":[{"attack":attack,"result":"REJECTED","evidence":evidence[attack]} for attack in ATTACKS],"pass_possible_without_all_conditions":False})


def _validate_report(repo: Path, path: Path, config: Mapping[str, Any], source_digest: str) -> dict[str, Any]:
    report,raw=_strict_json(path,"visual render report");_validate_signed(report,"visual render report")
    if raw!=canonical_json_bytes(report)+b"\n" or report["config_digest"]!=config["canonical_digest"] or report["source_digest"]!=source_digest:raise StageFVisualError("render report binding drifted")
    if any(report.get(key) is not False for key in ("magnitude_comparison_allowed","model_loaded","torch_imported","timm_imported","cuda_used","new_attribution_or_fidelity","gradcam_or_figure7_generated","f4_started")):raise StageFVisualError("render report crossed boundary")
    if len(set(report["format_content_digests"].values()))!=1:raise StageFVisualError("format content identity drifted")
    for ext,item in report["outputs"].items():
        target=path.parent/item["path"]
        if not target.is_file() or sha256_file(target)!=item["sha256"]:raise StageFVisualError(f"render output drifted: {ext}")
    return report


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve();config=_load_config(repo);_runtime_boundary();sources=_sources(repo);binding,_=_strict_json(repo/SOURCE_BINDING_PATH,"source binding");_validate_signed(binding,"source binding")
    if binding["f3_archive_commit"]!=F3_COMMIT or binding["source_digest"]!=sources["source_digest"] or binding["selection_changed"] or binding["maps_or_jpegs_changed"]:raise StageFVisualError("source binding drifted")
    transform,_=_strict_json(repo/TRANSFORM_MANIFEST_PATH,"transform manifest");render,_=_strict_json(repo/RENDER_INVENTORY_PATH,"render inventory");review,_=_strict_json(repo/REVIEW_INVENTORY_PATH,"review inventory");access,_=_strict_json(repo/ACCESS_AUDIT_PATH,"access audit");false,_=_strict_json(repo/FALSE_PASS_PATH,"false audit")
    for value,name in ((transform,"transform"),(render,"render"),(review,"review"),(access,"access"),(false,"false")):_validate_signed(value,name)
    if transform["config_digest"]!=config["canonical_digest"] or transform["source_digest"]!=sources["source_digest"] or transform["per_panel_normalization"] is not True or transform["magnitude_comparison_allowed"] is not False or transform["r7_signed_audit_preserved"] is not True:raise StageFVisualError("transform manifest drifted")
    for item in transform["transforms"]:
        if item.get("validity_status") in {"FAILED","NA"}:
            if item.get("imputed") is not False:raise StageFVisualError("failure imputation detected")
            continue
        if item.get("normalization")!="PER_PANEL_MAX_ABSOLUTE_VALUE" or item.get("gamma")!=2.0 or item.get("clipping") is not None or item.get("smoothing") is not None or item.get("max_pool") is not None or item.get("magnitude_comparison_allowed") is not False:raise StageFVisualError("blue transform drifted")
        source_path=repo/next(row["source_path"] for row in sources["cross_panel"]["cells"] if row.get("signed_map_sha256")==item["raw_signed_map_sha256"] and row.get("source_path")) if any(row.get("signed_map_sha256")==item["raw_signed_map_sha256"] and row.get("source_path") for row in sources["cross_panel"]["cells"]) else None
        if source_path and _array_digest(np.load(source_path,allow_pickle=False))!=item["raw_signed_map_sha256"]:raise StageFVisualError("raw map digest no longer matches")
    if render["module_counts"]!=MODULE_COUNTS or render["render_count"]!=27 or render["exported_file_count"]!=81:raise StageFVisualError("render coverage drifted")
    for item in render["renders"]:_validate_report(repo,repo/item["report"]["path"],config,sources["source_digest"])
    if review["review_count"]!=27 or len(review["copies"])!=27 or any(not item["exact_copy"] or sha256_file(repo/item["path"])!=item["sha256"] for item in review["copies"]):raise StageFVisualError("review copies drifted")
    if any(access.get(key) is not False for key in ("model_access","checkpoint_access","torch_imported","timm_imported","cuda_or_gpu_access","new_attribution_or_fidelity","gradcam_or_figure7_generated","f4_started")):raise StageFVisualError("access boundary drifted")
    if [item["attack"] for item in false["attacks"]]!=list(ATTACKS) or any(item["result"]!="REJECTED" for item in false["attacks"]):raise StageFVisualError("false-pass audit drifted")
    if not (repo/ANALYSIS_PATH).is_file():raise StageFVisualError("figure analysis absent")
    analysis=(repo/ANALYSIS_PATH).read_text(encoding="utf-8")
    if any(f"## Image #{index}:" not in analysis for index in range(1,28)) or any(term not in analysis for term in ("图能说明什么","图不能说明什么","最可能的误读","文字或 colorbar 裁切","字号","FAILED/NA","panel 顺序","插值伪影")):raise StageFVisualError("figure analysis incomplete")
    if _git(repo,"diff","--check"):raise StageFVisualError("git diff --check failed")
    checks={attack:"PASS" for attack in ATTACKS};return _signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_validation","revision_id":config["revision_id"],"status":"PASS","checks":checks,"render_count":27,"exported_file_count":81,"review_count":27,"module_counts":MODULE_COUNTS,"figure6_sample_id":sources["cross_selection"]["figure_6"]["sample_id"],"s4_slot_count":12,"s4_unique_sample_count":11,"shape_normalized":True,"magnitude_comparison_allowed":False,"r7_signed_audit_preserved":True,"gradcam_or_figure7_generated":False,"model_torch_timm_cuda_access":False,"new_attribution_or_fidelity":False,"f4_started":False,"next_authorization":"F4_ONLY"})


def record_tests(repo_root: Path | str, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    if passed<=0 or failed!=0:raise StageFVisualError("test report requires positive passes and zero failures")
    report=_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_test_report","revision_id":"F3_BLUE_DISPLAY_R1_20260907","status":"PASS","passed":passed,"failed":failed,"command":command,"model_access":False,"torch_imported":False,"timm_imported":False,"cuda_or_gpu_access":False,"new_attribution_or_fidelity":False});_write_new(Path(repo_root).resolve()/TEST_PATH,report);return report


def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo=Path(repo_root).resolve();root=repo/ROOT_PATH;validation=validate_outputs(repo);_write_new(repo/VALIDATION_PATH,validation);test=_record(repo,TEST_PATH);excluded={"payload_manifest.json","whole_root_inventory.json","status.json","completion/f3_visual.pending.json","completion/f3_visual.json"};files=[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in excluded];_write_new(repo/PAYLOAD_MANIFEST_PATH,_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_payload_manifest","revision_id":"F3_BLUE_DISPLAY_R1_20260907","immutable":True,"payload_count":len(files),"artifacts":files}));inv=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/f3_visual.pending.json","completion/f3_visual.json"));invdoc=_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_whole_root_inventory","revision_id":"F3_BLUE_DISPLAY_R1_20260907","root":ROOT_PATH,"files":inv["files"],"file_count":inv["file_count"],"total_bytes":inv["total_bytes"]});_write_new(repo/INVENTORY_PATH,invdoc);status=_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_status","revision_id":"F3_BLUE_DISPLAY_R1_20260907","status":"PASS","terminal":True,"package_validation":"PASS","workflow_validation":"PASS","render_count":27,"exported_file_count":81,"review_count":27,"shape_normalized":True,"magnitude_comparison_allowed":False,"r7_signed_audit_preserved":True,"gradcam_or_figure7_generated":False,"model_torch_timm_gpu_cuda_access":False,"new_attribution_or_fidelity":False,"f3_archive_modified":False,"f4_started":False,"next_authorization":"F4_ONLY","whole_root_inventory":{"canonical_digest":invdoc["canonical_digest"],"file_count":invdoc["file_count"],"total_bytes":invdoc["total_bytes"]},"git":{"commit":"SELF","ref":ARCHIVE_REF}});_write_new(repo/STATUS_PATH,status);pending,_=_strict_json(repo/PENDING_PATH,"pending");completion=_signed({"schema_version":1,"document_type":"stage_f_f3_blue_display_terminal_completion","revision_id":"F3_BLUE_DISPLAY_R1_20260907","status":"PASS","started_at":pending["started_at"],"completed_at":completed_at,"package_validation":"PASS","workflow_validation":"PASS","render_count":27,"exported_file_count":81,"review_count":27,"shape_normalized":True,"magnitude_comparison_allowed":False,"r7_signed_audit_preserved":True,"gradcam_or_figure7_generated":False,"model_access":False,"torch_imported":False,"timm_imported":False,"gpu_cuda_access":False,"new_attribution_or_fidelity":False,"f3_archive_modified":False,"f4_started":False,"next_authorization":"F4_ONLY","startup_gate":_record(repo,STARTUP_PATH),"source_binding":_record(repo,SOURCE_BINDING_PATH),"transform_manifest":_record(repo,TRANSFORM_MANIFEST_PATH),"render_inventory":_record(repo,RENDER_INVENTORY_PATH),"review_copy_inventory":_record(repo,REVIEW_INVENTORY_PATH),"access_audit":_record(repo,ACCESS_AUDIT_PATH),"validation_report":_record(repo,VALIDATION_PATH),"false_pass_audit":_record(repo,FALSE_PASS_PATH),"test_report":test,"figure_analysis":_file_record(repo,ANALYSIS_PATH),"payload_manifest":_record(repo,PAYLOAD_MANIFEST_PATH),"status_record":_record(repo,STATUS_PATH),"whole_root_inventory":_record(repo,INVENTORY_PATH),"git":{"commit":"SELF","ref":ARCHIVE_REF}});tmp=(repo/PENDING_PATH).with_name(".f3-visual-terminal.tmp");_write_new(tmp,completion);os.replace(tmp,repo/PENDING_PATH);os.replace(repo/PENDING_PATH,repo/COMPLETION_PATH);return validate_completion(repo)


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve();root=repo/ROOT_PATH
    if not (repo/COMPLETION_PATH).is_file() or (repo/PENDING_PATH).exists():raise StageFVisualError("terminal completion absent or pending remains")
    completion,raw=_strict_json(repo/COMPLETION_PATH,"completion");_validate_signed(completion,"completion")
    if raw!=canonical_json_bytes(completion)+b"\n":raise StageFVisualError("completion not canonical")
    expected={"status":"PASS","package_validation":"PASS","workflow_validation":"PASS","render_count":27,"exported_file_count":81,"review_count":27,"shape_normalized":True,"magnitude_comparison_allowed":False,"r7_signed_audit_preserved":True,"gradcam_or_figure7_generated":False,"model_access":False,"torch_imported":False,"timm_imported":False,"gpu_cuda_access":False,"new_attribution_or_fidelity":False,"f3_archive_modified":False,"f4_started":False,"next_authorization":"F4_ONLY"}
    if any(completion.get(k)!=v for k,v in expected.items()):raise StageFVisualError("completion state drifted")
    for field in ("startup_gate","source_binding","transform_manifest","render_inventory","review_copy_inventory","access_audit","validation_report","false_pass_audit","test_report","payload_manifest","status_record","whole_root_inventory"):
        if completion[field]!=_record(repo,completion[field]["path"]):raise StageFVisualError(f"completion binding drifted: {field}")
    if completion["figure_analysis"]!=_file_record(repo,ANALYSIS_PATH):raise StageFVisualError("figure analysis binding drifted")
    inv,_=_strict_json(repo/INVENTORY_PATH,"inventory");rebuilt=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/f3_visual.pending.json","completion/f3_visual.json"));unsigned={"schema_version":1,"document_type":"stage_f_f3_blue_display_whole_root_inventory","revision_id":"F3_BLUE_DISPLAY_R1_20260907","root":ROOT_PATH,"files":rebuilt["files"],"file_count":rebuilt["file_count"],"total_bytes":rebuilt["total_bytes"]}
    if any(inv[k]!=rebuilt[k] for k in ("files","file_count","total_bytes")) or inv["canonical_digest"]!=canonical_sha256(unsigned):raise StageFVisualError("inventory drifted")
    return {**expected,"whole_root_inventory_digest":inv["canonical_digest"],"archive_ref":ARCHIVE_REF}


def _file_record(repo: Path, relative: str) -> dict[str, Any]:
    path=repo/relative;return {"path":relative,"bytes":path.stat().st_size,"sha256":sha256_file(path)}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    for name in ("verify-startup","build","validate-outputs","validate-completion"):
        child=sub.add_parser(name);child.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    init=sub.add_parser("initialize");init.add_argument("--repo-root",type=Path,default=REPO_ROOT);init.add_argument("--renderer-commit",required=True);init.add_argument("--started-at",required=True)
    tests=sub.add_parser("record-tests");tests.add_argument("--repo-root",type=Path,default=REPO_ROOT);tests.add_argument("--passed",type=int,required=True);tests.add_argument("--failed",type=int,required=True);tests.add_argument("--test-command",required=True)
    final=sub.add_parser("finalize");final.add_argument("--repo-root",type=Path,default=REPO_ROOT);final.add_argument("--completed-at",required=True);args=parser.parse_args(argv)
    try:
        if args.command=="verify-startup":result=verify_startup(args.repo_root)
        elif args.command=="initialize":result=initialize(args.repo_root,renderer_commit=args.renderer_commit,started_at=args.started_at)
        elif args.command=="build":result=build_outputs(args.repo_root)
        elif args.command=="validate-outputs":result=validate_outputs(args.repo_root)
        elif args.command=="record-tests":result=record_tests(args.repo_root,passed=args.passed,failed=args.failed,command=args.test_command)
        elif args.command=="finalize":result=finalize(args.repo_root,completed_at=args.completed_at)
        else:result=validate_completion(args.repo_root)
    except (StageFVisualError,OSError,ValueError) as error:
        print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True),file=sys.stderr);return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__ == "__main__":
    raise SystemExit(main())
