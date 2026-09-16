"""Presentation-only S6-R2 figures inspired by the NNS reference-paper layout."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-nns-s6-r2")

import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np

STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
STAGE_ID = "S6-R2-PRESENTATION-REVISION"
RUN_ID = "s6_r2_presentation_revision_20260916"
LABEL = "post-hoc exploratory"
ROOT_REL = Path("evidence/nns_vit_patch_depth/s6_r2_presentation_revision_20260916")
REPORT_REL = Path("reports/nns_vit_patch_depth_s6_r2_presentation_revision_20260916")
S5_ROOT = Path("evidence/nns_vit_patch_depth/s5_r1_posthoc_exploratory_20260916")
S6_ROOT = Path("evidence/nns_vit_patch_depth/s6_r1_posthoc_exploratory_reporting_20260916")
REFERENCE_ROOT = Path("../相关文章/Understanding neural network through neuron level visualization/assets/figures")
BASE_SOURCE_FILES = (
    S5_ROOT / "image_level_endpoint_values.json",
    S5_ROOT / "endpoint_decisions.json",
    S5_ROOT / "original_vs_posthoc_P2_comparison.json",
    S6_ROOT / "figure_data.json",
    S6_ROOT / "terminal_receipt.json",
)
INTERVENTION_SOURCE = S5_ROOT / "intervention_controls_and_matching_diagnostics.json"
SOURCE_FILES = BASE_SOURCE_FILES[:3] + (INTERVENTION_SOURCE,) + BASE_SOURCE_FILES[3:]
REFERENCE_FILES = ("fig-08.jpg", "fig-10.jpg", "fig-12.jpg", "fig-13.jpg", "fig-14.jpg")


def revision_paths(revision: str) -> tuple[Path, Path]:
    if revision == "r2": return ROOT_REL, REPORT_REL
    if revision == "r3": return Path("evidence/nns_vit_patch_depth/s6_r3_paper_figure_revision_20260916"), Path("reports/nns_vit_patch_depth_s6_r3_paper_figure_revision_20260916")
    raise VisualError(f"unknown presentation revision: {revision}")


class VisualError(RuntimeError): pass


def require(condition: bool, message: str) -> None:
    if not condition: raise VisualError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def canonical_digest(value: Any) -> str: return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def verify_signed(value: Mapping[str, Any]) -> bool:
    copy = dict(value); claimed = copy.pop("canonical_digest", None)
    return isinstance(claimed, str) and claimed == canonical_digest(copy)


def signed(kind: str, status: str, payload: Mapping[str, Any], commit: str) -> dict[str, Any]:
    value = {"schema_version": 1, "study_id": STUDY_ID, "stage_id": STAGE_ID, "run_id": RUN_ID, "document_type": kind, "status": status, "payload": dict(payload), "producer_commit": commit}
    value["canonical_digest"] = canonical_digest(value); return value


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"); stream.flush(); os.fsync(stream.fileno())


def write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream: stream.write(value); stream.flush(); os.fsync(stream.fileno())


def artifact(repo: Path, path: Path) -> dict[str, Any]:
    value = {"path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if isinstance(document, dict) and isinstance(document.get("canonical_digest"), str): value["canonical_digest"] = document["canonical_digest"]
    return value


def load_inputs(repo: Path, *, paper_safe: bool = False) -> tuple[dict[str, dict], list[dict]]:
    inputs = {}; records = []
    source_files = SOURCE_FILES if paper_safe else BASE_SOURCE_FILES
    for relative in source_files:
        path = repo / relative; require(path.is_file(), f"missing source: {relative}")
        value = json.loads(path.read_text()); require(verify_signed(value), f"source canonical drift: {relative}")
        inputs[path.name] = value; records.append(artifact(repo, path))
    for name in REFERENCE_FILES:
        path = (repo / REFERENCE_ROOT / name).resolve(); require(path.is_file(), f"missing reference figure: {name}")
        records.append({"path": path.relative_to(repo.parent).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path), "role": "visual_style_reference_only"})
    terminal = inputs["terminal_receipt.json"]
    require(terminal["status"] == "PASS_POST_HOC_EXPLORATORY_REPORTING" and terminal["payload"]["next_authorization"] == "NONE", "S6-R1 terminal drift")
    return inputs, records


def configure():
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "figure.dpi": 120, "savefig.dpi": 600, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


COLORS = {"P1": "#1f77b4", "P2": "#ff8c00", "D1": "#15803d", "D2": "#c026d3"}
TITLES = {"P1": "NNS–foreground alignment", "P2": "Matched occlusion response", "D1": "NNS depth contrast", "D2": "Probe depth contrast"}


def deterministic_jitter(ids: list[str], scale: float = 0.12) -> np.ndarray:
    values = [int.from_bytes(hashlib.sha256(item.encode()).digest()[:4], "big") / (2**32 - 1) for item in ids]
    return (np.asarray(values) - 0.5) * 2 * scale


def save(fig: Any, base: Path) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = base.with_suffix("." + suffix); fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white"); paths.append(path)
    return paths


def render(root: Path, inputs: Mapping[str, Mapping], *, paper_safe: bool = False) -> list[Path]:
    plt = configure(); root.mkdir(parents=True, exist_ok=True)
    endpoint_records = inputs["image_level_endpoint_values.json"]["payload"]["endpoint_records"]
    decisions = inputs["endpoint_decisions.json"]["payload"]["endpoints"]
    figure_data = inputs["figure_data.json"]["payload"]
    outputs = []

    # Image 01: compact 2x2 distribution atlas, following the reference paper's small-multiple grammar.
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), constrained_layout=True)
    for label, axis, endpoint in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat, ("P1", "P2", "D1", "D2")):
        rows = endpoint_records[endpoint]; values = np.asarray([row["value"] for row in rows], dtype=float)
        ids = [row["source_image_id"] for row in rows]; color = COLORS[endpoint]; summary = decisions[endpoint]
        violin = axis.violinplot([values], positions=[0.48], vert=False, widths=0.55, showmeans=False, showmedians=False, showextrema=False, points=180)
        for body in violin["bodies"]: body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.22); body.set_linewidth(0.8)
        if len(values) > 600:
            order = np.argsort([hashlib.sha256(item.encode()).hexdigest() for item in ids])[:600]
        else: order = np.arange(len(values))
        axis.scatter(values[order], 0.48 + deterministic_jitter([ids[index] for index in order]), s=5 if len(order) > 300 else 8, color=color, alpha=0.18, edgecolors="none", rasterized=True)
        estimate = summary["estimate"]; lo = summary["ordinary_BCa_95"]["lower"]; hi = summary["ordinary_BCa_95"]["upper"]
        axis.plot([lo, hi], [0.92, 0.92], color="#111827", lw=2.2, solid_capstyle="round")
        axis.scatter([estimate], [0.92], s=42, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        axis.axvline(0, color="#6b7280", lw=0.7)
        axis.axvline(summary["posthoc_threshold"], color=color, lw=1.1, ls="--")
        axis.set_ylim(0.12, 1.08); axis.set_yticks([]); axis.grid(axis="x", color="#d1d5db", alpha=0.45, lw=0.6)
        axis.set_title(f"{label} {endpoint} · {TITLES[endpoint]}", loc="left", weight="bold")
        axis.text(0.02, 0.97, f"equal-class estimate {estimate:.3f}\nBCa 95% CI [{lo:.3f}, {hi:.3f}]\nn={summary['n_valid']} valid", transform=axis.transAxes, va="top", ha="left", fontsize=7.3)
        axis.set_xlabel("Image-level value")
    fig.suptitle("Post-hoc exploratory evidence atlas", fontsize=12, weight="bold")
    outputs += save(fig, root / "image_01_evidence_atlas")
    plt.close(fig)

    # Image 02: simple layer-wise curves using the reference paper's two-panel line grammar.
    nns = figure_data["NNS_depth_rows"]; probe = figure_data["D2_depth_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), constrained_layout=True)
    axes[0].plot([row["block"] for row in nns], [row["equal_class_position_control_corrected_alignment"] for row in nns], color="#1f77b4", marker="o", ms=4.2, lw=1.8)
    axes[1].plot([row["block"] for row in probe], [row["equal_class_probe_gain"] for row in probe], color="#c026d3", marker="*", ms=7, lw=1.8, ls="--")
    for label, axis, title, ylabel in (("(a)", axes[0], "NNS alignment across blocks", "Corrected alignment"), ("(b)", axes[1], "Probe gain across blocks", "Control MSE − probe MSE")):
        axis.axhline(0, color="#6b7280", lw=0.7); axis.grid(color="#d1d5db", alpha=0.45, lw=0.6); axis.set_xticks(range(12)); axis.set_xlabel("ViT block"); axis.set_ylabel(ylabel); axis.set_title(f"{label} {title}", loc="left", weight="bold")
    fig.suptitle("Post-hoc exploratory depth progression", fontsize=11.5, weight="bold")
    outputs += save(fig, root / "image_02_depth_progression")
    plt.close(fig)

    # Image 03: relationship plot retained, visually harmonized.
    scatter = figure_data["D1_D2_scatter_rows"]; relation = figure_data["D1_D2_relationship"]
    fig, axis = plt.subplots(figsize=(3.65, 3.5), constrained_layout=True)
    axis.scatter([row["D1"] for row in scatter], [row["D2"] for row in scatter], s=14, alpha=0.32, color="#0ea5e9", edgecolors="none")
    axis.axhline(0, color="#6b7280", lw=0.8); axis.axvline(0, color="#6b7280", lw=0.8); axis.grid(color="#d1d5db", alpha=0.45, lw=0.6)
    axis.set_xlabel("D1 · NNS deep − shallow"); axis.set_ylabel("D2 · probe-gain deep − shallow")
    axis.set_title("Post-hoc exploratory D1/D2 relation", loc="left", weight="bold")
    axis.text(0.03, 0.97, f"n={relation['paired_valid_source_count']}\nr={relation['paired_image_Pearson_correlation']:.3f}", transform=axis.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.35"})
    outputs += save(fig, root / "image_03_D1_D2_relation")
    plt.close(fig)

    if paper_safe:
        intervention = inputs["intervention_controls_and_matching_diagnostics.json"]["payload"]
        rows = intervention["aggregate_by_operator_block_budget"]; matching = intervention["matching_diagnostics"]
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), constrained_layout=True)
        styles = {"image_channel_mean": ("#ff8c00", "-", "Primary operator"), "normalized_zero": ("#6366f1", "--", "Sensitivity operator")}
        markers = {4: "o", 8: "s", 16: "^"}
        for operator, (color, linestyle, label_text) in styles.items():
            for budget in (4, 8, 16):
                selected = sorted((row for row in rows if row["operator"] == operator and row["budget"] == budget), key=lambda row: row["block"])
                axes[0].plot([row["block"] for row in selected], [row["mean_matched_control_corrected_logit_drop"] for row in selected], color=color, linestyle=linestyle, marker=markers[budget], ms=3.8, lw=1.5, label=f"{label_text}, k={budget}")
        axes[0].axhline(0, color="#6b7280", lw=0.7); axes[0].grid(color="#d1d5db", alpha=0.45, lw=0.6); axes[0].set_xticks(range(12)); axes[0].set_xlabel("ViT block"); axes[0].set_ylabel("Matched-control corrected logit drop"); axes[0].set_title("(a) Occlusion response by block and budget", loc="left", weight="bold"); axes[0].legend(frameon=False, ncol=2, fontsize=6.4)
        fractions = [matching["matched_image_fraction"], matching["matched_coordinate_fraction"], matching["matched_control_fraction"]]
        labels = ["Images", "Coordinates", "Controls"]; bars = axes[1].barh(labels, fractions, color=["#1f77b4", "#15803d", "#7c3aed"], alpha=0.82)
        axes[1].set_xlim(0.90, 1.005); axes[1].axvline(1, color="#6b7280", lw=0.7); axes[1].grid(axis="x", color="#d1d5db", alpha=0.45, lw=0.6); axes[1].set_xlabel("Matched fraction"); axes[1].set_title("(b) Matching coverage", loc="left", weight="bold")
        counts = [f"{matching['P2_valid_matched_images']}/{matching['NNS_valid_images_denominator']}", f"{matching['matched_coordinates']}/{matching['possible_coordinates']}", f"{matching['matched_controls']}/{matching['possible_controls']}"]
        for bar, fraction, count in zip(bars, fractions, counts): axes[1].text(fraction - 0.002, bar.get_y() + bar.get_height()/2, f"{fraction:.1%}  ({count})", ha="right", va="center", color="white", fontsize=7.2, weight="bold")
        fig.suptitle("Post-hoc exploratory matched occlusion evidence", fontsize=11.5, weight="bold")
        outputs += save(fig, root / "image_04_matched_occlusion_evidence")
    else:
        p2 = decisions["P2"]
        metrics = [("(a)", "Point estimate", p2["estimate"], 0.10, 0.07, True, (0, 0.11)), ("(b)", "Adjusted lower bound", p2["adjusted_interval"]["lower"], 0.10, 0.0, True, (-0.005, 0.11)), ("(c)", "BCa half-width", p2["ordinary_BCa_95"]["half_width"], 0.015, 0.02, False, (0, 0.022))]
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75), constrained_layout=True)
        for axis, (label, title, observed, original, posthoc, higher, limits) in zip(axes, metrics):
            axis.axvspan(posthoc if higher else limits[0], limits[1] if higher else posthoc, color="#16a34a", alpha=0.08); axis.axvline(original, color="#ea580c", lw=1.7, ls="--"); axis.axvline(posthoc, color="#16a34a", lw=1.7, ls=":"); axis.scatter([observed], [0.52], s=70, color="#1d4ed8", edgecolor="white", linewidth=0.8, zorder=5); axis.set_xlim(*limits); axis.set_ylim(0, 1); axis.set_yticks([]); axis.grid(axis="x", color="#d1d5db", alpha=0.45, lw=0.6); axis.set_title(f"{label} {title}", loc="left", weight="bold"); axis.text(0.04, 0.86, f"Observed {observed:.3f}", transform=axis.transAxes, weight="bold", color="#1d4ed8"); axis.text(0.04, 0.75, f"S5 gate {posthoc:.3f}  ·  original {original:.3f}", transform=axis.transAxes, fontsize=7.2, color="#4b5563"); axis.text(0.04, 0.16, "Original: FAIL", transform=axis.transAxes, color="#c2410c"); axis.text(0.96, 0.16, "S5: PASS", transform=axis.transAxes, color="#15803d", ha="right"); axis.set_xlabel("Metric value")
        fig.suptitle("Post-hoc exploratory P2 decision · original rule not passed", fontsize=11.5, weight="bold"); outputs += save(fig, root / "image_04_P2_decision_panels")
    plt.close(fig)
    return outputs


def png_geometry(path: Path) -> dict[str, int]:
    data = path.read_bytes(); require(data[:8] == b"\x89PNG\r\n\x1a\n", "invalid PNG")
    width, height = struct.unpack(">II", data[16:24]); return {"width": width, "height": height}
