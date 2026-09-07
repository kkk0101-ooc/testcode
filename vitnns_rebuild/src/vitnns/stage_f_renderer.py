"""Deterministic, display-only Stage F figure renderer.

F0 deliberately exposes only synthetic fixtures.  The module has no model,
Torch, timm, CUDA, checkpoint, JPEG, attribution, or scientific-statistics
execution surface.  D10 and E4 recipes are parsed as contracts only.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import font_manager, pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402

from .constants import REPO_ROOT
from .display import DisplayTransform, metric_ready, raw, validate_raw_input
from .stage_d_provenance import canonical_json_bytes, canonical_sha256


CONFIG_PATH = "config/stage_f_f0.json"
CONFIG_SCHEMA_PATH = "config/stage_f_f0.schema.json"
RUN_SPEC_PATH = "config/stage_f_run_spec.json"
RUN_SPEC_SCHEMA_PATH = "config/stage_f_run_spec.schema.json"
CATALOG_PATH = "config/stage_f_figure_catalog.json"
CATALOG_SCHEMA_PATH = "config/stage_f_figure_catalog.schema.json"
CAPTION_PATH = "config/stage_f_caption_contract.json"
CAPTION_SCHEMA_PATH = "config/stage_f_caption_contract.schema.json"
FIXTURE_SCHEMA_PATH = "config/stage_f_synthetic_fixture.schema.json"
RENDER_SCHEMA_PATH = "config/stage_f_render_artifact.schema.json"
VALIDATION_SCHEMA_PATH = "config/stage_f_validation_report.schema.json"
COMPLETION_SCHEMA_PATH = "config/stage_f_completion_receipt.schema.json"
FALSE_PASS_SCHEMA_PATH = "config/stage_f_false_pass_audit.schema.json"

D10_RECIPE_PATH = "evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/figure_recipes.json"
E4_RECIPE_PATH = "evidence/stage_e/e4_recovery_v2_20260907/formal/figure_recipes.json"
E4_COMPLETION_PATH = "evidence/stage_e/e4_recovery_v2_20260907/formal/completion/e4.json"
E4_INVENTORY_PATH = "evidence/stage_e/e4_recovery_v2_20260907/formal/whole_root_inventory.json"
DISPLAY_PROTOCOL_PATH = "docs/stage_f_display_protocol_v1.md"
REFERENCE_PDF_PATH = "../论文1/参考文章.pdf"

F0_ROOT_PATH = "evidence/stage_f/f0_r4_recovery_v2_20260907/formal"
F0_REPORT_PATH = "reports/stage_f_f0_r4_20260907"
ARCHIVE_REF = "refs/archive/stage-f-v1-f0-completion"

EXPECTED_E4_REF = "7c82eca4e7de488f4537eff2b3fdb59feb3413ab"
EXPECTED_D10_REF = "7520689bb031304e4789567f99d093062aa50f72"
EXPECTED_E4_COMPLETION_SHA256 = "42ad2b483e40c164b4e26e1d8d65ecfc00703a2c01c2f0589961bcbefa8d58dc"
EXPECTED_E4_INVENTORY_DIGEST = "75249c74e08cb5038acbc86f71051260f9d9b7b6cc80d23c517e8669f3765f67"
EXPECTED_D10_RECIPE_SHA256 = "aaa1858e5a2b0e820d4ea5cb3072a41e3cb6f4f507daefa319dc9ab66a1f0ad1"
EXPECTED_E4_RECIPE_SHA256 = "57306bc69e399d1ea2353d908ac0d8b22403d17ec29318a3eadf36b3b8d98a54"
EXPECTED_DISPLAY_PROTOCOL_SHA256 = "de3ee8804f59e5541eb3c170110ab3b6c47f663698638e3fd668ece7f19b35e3"
EXPECTED_REFERENCE_PDF_SHA256 = "33e323f8cb664864b12daad1f06b6ef0d3268bcafaea4e84ac695fd286f3855b"
EXPECTED_START_COMMIT = "cff6b72748c9fad70fd251337371d41bae983f6f"

ALLOWED_PANEL_KINDS = frozenset({
    "line_band", "ecdf", "point_interval", "heatmap",
    "signed_attribution_pair", "difference_map", "rgb_overlay",
    "fidelity_curve", "method_schematic",
})
FORBIDDEN_COLORMAPS = frozenset({"jet", "rainbow", "gist_rainbow", "nipy_spectral"})
FORBIDDEN_IMPORTS = frozenset({"torch", "torchvision", "timm", "cv2"})
JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})
FIXTURE_ROOT = "tests/fixtures/stage_f"


class StageFRendererError(RuntimeError):
    """Raised whenever the F0 display contract cannot be proven."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    raw_bytes = path.read_bytes()
    try:
        value = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageFRendererError(f"{context} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise StageFRendererError(f"{context} must be a JSON object")
    return value, raw_bytes


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != "canonical_digest"}
    if value.get("canonical_digest") != canonical_sha256(unsigned):
        raise StageFRendererError(f"{context} canonical digest mismatch")


def _schema_validate(document: Mapping[str, Any], schema_path: Path, context: str) -> None:
    schema, _ = _strict_json(schema_path, f"{context} schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda item: list(item.path))
    if errors:
        location = ".".join(str(item) for item in errors[0].path) or "<root>"
        raise StageFRendererError(f"{context} schema failure at {location}: {errors[0].message}")


def load_contracts(repo_root: Path | str = REPO_ROOT) -> dict[str, dict[str, Any]]:
    repo = Path(repo_root).resolve()
    specs = {
        "config": (CONFIG_PATH, CONFIG_SCHEMA_PATH),
        "run_spec": (RUN_SPEC_PATH, RUN_SPEC_SCHEMA_PATH),
        "catalog": (CATALOG_PATH, CATALOG_SCHEMA_PATH),
        "captions": (CAPTION_PATH, CAPTION_SCHEMA_PATH),
    }
    loaded: dict[str, dict[str, Any]] = {}
    for name, (document_path, schema_path) in specs.items():
        document, _ = _strict_json(repo / document_path, name)
        _schema_validate(document, repo / schema_path, name)
        _validate_signed(document, name)
        loaded[name] = document
    config = loaded["config"]
    if config["authorization"] != {
        "current": "F0_ONLY", "on_pass": "F1_ONLY",
        "f1_started": False, "f2_started": False,
        "f3_started": False, "f4_started": False,
    }:
        raise StageFRendererError("F0 authorization contract drifted")
    if config["safety"]["formal_scientific_figures_allowed"] is not False:
        raise StageFRendererError("F0 cannot authorize formal scientific figures")
    return loaded


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments), cwd=repo, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise StageFRendererError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _is_ancestor(repo: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    return subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def _reference_pdf(repo: Path) -> Path:
    return (repo / REFERENCE_PDF_PATH).resolve()


def verify_preconditions(
    repo_root: Path | str = REPO_ROOT, *, require_clean: bool = True, require_remote: bool = True,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    contracts = load_contracts(repo)
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    head = _git(repo, "rev-parse", "HEAD")
    tracking = _git(repo, "rev-parse", "origin/stage-d-d3-pass")
    if branch != "stage-d-d3-pass" or tracking != head:
        raise StageFRendererError("local branch and tracking ref must be stage-d-d3-pass at one commit")
    if not _is_ancestor(repo, EXPECTED_START_COMMIT):
        raise StageFRendererError("HEAD is not an undrifted successor of the frozen F0 start commit")
    if require_clean and _git(repo, "status", "--porcelain=v1"):
        raise StageFRendererError("F0 start requires a clean worktree and index")
    lfs_status = _git(repo, "lfs", "status")
    dirty_lfs = any(line.strip() for line in lfs_status.splitlines() if line.startswith("    "))
    if require_clean and dirty_lfs:
        raise StageFRendererError("F0 start requires clean Git LFS state")
    remote = None
    if require_remote:
        remote_lines = _git(repo, "ls-remote", "--heads", "origin", "stage-d-d3-pass").splitlines()
        if len(remote_lines) != 1:
            raise StageFRendererError("remote stage-d-d3-pass is absent or ambiguous")
        remote = remote_lines[0].split()[0]
        if remote != head:
            raise StageFRendererError("local, tracking, and remote branch are not identical")
    if _git(repo, "rev-parse", "refs/archive/stage-e-e4-completion^{commit}") != EXPECTED_E4_REF:
        raise StageFRendererError("E4 archive ref drifted")
    if _git(repo, "rev-parse", "refs/archive/stage-d-fidelity-v1-d10-completion^{commit}") != EXPECTED_D10_REF:
        raise StageFRendererError("D10 archive ref drifted")
    if not _is_ancestor(repo, EXPECTED_E4_REF):
        raise StageFRendererError("E4 archive commit must be an ancestor of HEAD")

    completion, _ = _strict_json(repo / E4_COMPLETION_PATH, "E4 completion")
    inventory, _ = _strict_json(repo / E4_INVENTORY_PATH, "E4 inventory")
    if sha256_file(repo / E4_COMPLETION_PATH) != EXPECTED_E4_COMPLETION_SHA256:
        raise StageFRendererError("E4 completion file hash drifted")
    if inventory.get("canonical_digest") != EXPECTED_E4_INVENTORY_DIGEST:
        raise StageFRendererError("E4 whole-root inventory digest drifted")
    required_e4 = {
        "status": "PASS", "package_completeness": "PASS",
        "workflow_integrity": "PASS", "scientific_validity": "MIXED",
        "next_authorization": "F0_ONLY",
    }
    if any(completion.get(key) != expected for key, expected in required_e4.items()):
        raise StageFRendererError("E4 terminal state drifted")
    bound_hashes = {
        D10_RECIPE_PATH: EXPECTED_D10_RECIPE_SHA256,
        E4_RECIPE_PATH: EXPECTED_E4_RECIPE_SHA256,
        DISPLAY_PROTOCOL_PATH: EXPECTED_DISPLAY_PROTOCOL_SHA256,
    }
    for relative, expected in bound_hashes.items():
        if sha256_file(repo / relative) != expected:
            raise StageFRendererError(f"frozen Stage F input drifted: {relative}")
    reference = _reference_pdf(repo)
    if sha256_file(reference) != EXPECTED_REFERENCE_PDF_SHA256:
        raise StageFRendererError("visual reference PDF hash drifted")
    parse_upstream_recipes(repo)
    return {
        "status": "PASS", "branch": branch, "head": head,
        "tracking": tracking, "remote": remote, "git_lfs_clean": not dirty_lfs,
        "e4_ref": EXPECTED_E4_REF, "d10_ref": EXPECTED_D10_REF,
        "e4_completion_sha256": EXPECTED_E4_COMPLETION_SHA256,
        "e4_inventory_canonical_digest": EXPECTED_E4_INVENTORY_DIGEST,
        "reference_pdf_sha256": EXPECTED_REFERENCE_PDF_SHA256,
        "contracts": {name: value["canonical_digest"] for name, value in contracts.items()},
        "next_authorization": "F0_ONLY",
    }


def parse_upstream_recipes(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    d10, _ = _strict_json(repo / D10_RECIPE_PATH, "D10 figure recipes")
    e4, _ = _strict_json(repo / E4_RECIPE_PATH, "E4 figure recipes")
    for name, value in (("D10", d10), ("E4", e4)):
        if value.get("numeric_source") != "paper_data_only" or value.get("renders_png") is not False:
            raise StageFRendererError(f"{name} recipes are not parse-only paper-data recipes")
        if not value.get("figures") or any(item.get("figure_id") is None for item in value["figures"]):
            raise StageFRendererError(f"{name} recipes lack figure identities")
    if d10.get("c10_stage_d_pooling_allowed") is not False:
        raise StageFRendererError("D10 recipe no-pooling boundary drifted")
    return {
        "status": "PASS_PARSE_ONLY", "renders_created": 0,
        "new_statistics_computed": False,
        "d10": {"figure_ids": [item["figure_id"] for item in d10["figures"]], "sha256": sha256_file(repo / D10_RECIPE_PATH)},
        "e4": {"figure_ids": [item["figure_id"] for item in e4["figures"]], "sha256": sha256_file(repo / E4_RECIPE_PATH)},
    }


def _font_path(config: Mapping[str, Any], *, bold: bool = False) -> Path:
    key = "bold_file" if bold else "regular_file"
    relative = config["style"]["font"][key]
    path = Path(matplotlib.get_data_path()) / relative
    return path.resolve()


def validate_font(config: Mapping[str, Any]) -> dict[str, Any]:
    font = config["style"]["font"]
    records = {}
    for label, bold in (("regular", False), ("bold", True)):
        path = _font_path(config, bold=bold)
        if not path.is_file():
            raise StageFRendererError(f"required {label} font file is missing: {path}")
        observed_hash = sha256_file(path)
        expected_hash = font[f"{label}_sha256"]
        if observed_hash != expected_hash:
            raise StageFRendererError(f"required {label} font file hash drifted")
        font_manager.fontManager.addfont(str(path))
        properties = font_manager.FontProperties(fname=str(path))
        actual_name = properties.get_name()
        resolved = Path(font_manager.findfont(properties, fallback_to_default=False)).resolve()
        if actual_name != font["family"] or resolved != path:
            raise StageFRendererError("font resolution changed or silently fell back")
        records[label] = {"path": str(path), "sha256": observed_hash, "family": actual_name}
    return records


@contextmanager
def renderer_style(config: Mapping[str, Any]):
    fonts = validate_font(config)
    style = config["style"]
    regular = fonts["regular"]["path"]
    regular_name = font_manager.FontProperties(fname=regular).get_name()
    rc = {
        "font.family": "serif", "font.serif": [regular_name],
        "font.size": style["minimum_font_pt"],
        "axes.labelsize": style["minimum_font_pt"],
        "axes.titlesize": style["title_font_pt"],
        "xtick.labelsize": style["minimum_font_pt"],
        "ytick.labelsize": style["minimum_font_pt"],
        "legend.fontsize": style["minimum_font_pt"],
        "lines.linewidth": style["minimum_linewidth_pt"],
        "axes.linewidth": style["minimum_linewidth_pt"],
        "patch.linewidth": style["minimum_linewidth_pt"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "pdf.compression": 0,
        "svg.fonttype": "path", "svg.hashsalt": config["determinism"]["svg_hashsalt"],
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    with matplotlib.rc_context(rc):
        yield fonts


def load_fixture(path: Path, repo_root: Path | str = REPO_ROOT) -> tuple[dict[str, Any], str]:
    repo = Path(repo_root).resolve()
    target = path.resolve()
    fixture_root = (repo / FIXTURE_ROOT).resolve()
    if fixture_root not in target.parents or target.suffix.lower() != ".json":
        raise StageFRendererError("F0 renders only versioned JSON files in tests/fixtures/stage_f")
    if any(suffix in JPEG_SUFFIXES for suffix in (part.lower() for part in target.suffixes)):
        raise StageFRendererError("F0 forbids JPEG fixture access")
    value, _ = _strict_json(target, "synthetic fixture")
    _schema_validate(value, repo / FIXTURE_SCHEMA_PATH, "synthetic fixture")
    if value.get("fixture_class") != "SYNTHETIC_GOLDEN_ONLY" or value.get("formal_scientific_figure") is not False:
        raise StageFRendererError("F0 fixture is not synthetic-only")
    _validate_fixture_semantics(value)
    return value, sha256_file(target)


def _finite_vector(value: Any, name: str, *, minimum: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum or not np.all(np.isfinite(array)):
        raise StageFRendererError(f"{name} must be a finite 1D vector with at least {minimum} values")
    return array


def _finite_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2 or not np.all(np.isfinite(array)):
        raise StageFRendererError(f"{name} must be a finite 2D map")
    return array


def _validate_fixture_semantics(fixture: Mapping[str, Any]) -> None:
    panels = fixture["panels"]
    if len(panels) != fixture["grid"]["rows"] * fixture["grid"]["columns"]:
        raise StageFRendererError("panel count must equal rows times columns")
    expected_labels = [f"({chr(ord('a') + index)})" for index in range(len(panels))]
    if [panel["label"] for panel in panels] != expected_labels:
        raise StageFRendererError("panel labels must be consecutive (a), (b), ... in render order")
    if [panel["caption_key"] for panel in panels] != fixture["caption_keys"]:
        raise StageFRendererError("panel order and caption-key order differ")
    for panel in panels:
        kind = panel["kind"]
        if kind not in ALLOWED_PANEL_KINDS:
            raise StageFRendererError(f"unsupported panel kind: {kind}")
        if any(str(value).lower() in FORBIDDEN_COLORMAPS for value in panel.values()):
            raise StageFRendererError("rainbow/jet colormaps are forbidden")
        data = panel["data"]
        if kind in {"line_band", "fidelity_curve"}:
            x = _finite_vector(data["x"], f"{kind}.x", minimum=2)
            for series in data["series"]:
                for key in ("y", "lower", "upper"):
                    if key in series and _finite_vector(series[key], f"{kind}.{key}").shape != x.shape:
                        raise StageFRendererError(f"{kind} series shape mismatch")
                if "lower" in series and (np.any(np.asarray(series["lower"]) > np.asarray(series["y"])) or np.any(np.asarray(series["upper"]) < np.asarray(series["y"]))):
                    raise StageFRendererError("confidence band must contain the line")
        elif kind == "ecdf":
            for series in data["series"]:
                _finite_vector(series["values"], "ecdf.values", minimum=2)
        elif kind == "point_interval":
            for item in data["points"]:
                if not np.isfinite([item["estimate"], item["lower"], item["upper"]]).all() or not item["lower"] <= item["estimate"] <= item["upper"]:
                    raise StageFRendererError("point interval is invalid")
        elif kind in {"heatmap", "difference_map"}:
            _finite_map(data["values"], f"{kind}.values")
        elif kind == "signed_attribution_pair":
            nns = _finite_map(data["nns"], "signed pair nns")
            matched = _finite_map(data["matched"], "signed pair matched")
            if nns.shape != matched.shape:
                raise StageFRendererError("NNS and matched maps must share one shape")
        elif kind == "rgb_overlay":
            rgb = np.asarray(data["rgb"], dtype=np.float64)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.all(np.isfinite(rgb)):
                raise StageFRendererError("overlay RGB must be a finite HxWx3 array")
            _finite_map(data["attribution"], "overlay attribution")
        elif kind == "method_schematic":
            if len(data["nodes"]) < 2 or not data["arrows"]:
                raise StageFRendererError("method schematic needs nodes and arrows")


def _method_style(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    methods = config["style"]["methods"]
    if method not in methods:
        raise StageFRendererError(f"unknown method identity: {method}")
    return methods[method]


def _style_axis(axis: Any, panel: Mapping[str, Any]) -> None:
    axis.set_title(panel["title"], pad=3)
    axis.set_xlabel(panel.get("x_label", ""))
    axis.set_ylabel(panel.get("y_label", ""))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.75, length=2.5)


def _line_panel(axis: Any, panel: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    data = panel["data"]; x = np.asarray(data["x"], dtype=float)
    series_records = []
    for series in data["series"]:
        style = _method_style(config, series["method"]); y = np.asarray(series["y"], dtype=float)
        line, = axis.plot(x, y, color=style["color"], linestyle=style["linestyle"], marker=style["marker"], markersize=3, label=series["label"])
        if "lower" in series:
            axis.fill_between(x, series["lower"], series["upper"], color=style["color"], alpha=config["style"]["confidence_band_alpha"], linewidth=0)
        series_records.append({"method": series["method"], "color": style["color"], "linestyle": style["linestyle"], "marker": style["marker"], "line_width_pt": line.get_linewidth()})
    _style_axis(axis, panel); axis.legend(frameon=False)
    return {"series": series_records}


def _ecdf_panel(axis: Any, panel: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for series in panel["data"]["series"]:
        values = np.sort(np.asarray(series["values"], dtype=float)); y = np.arange(1, len(values) + 1) / len(values)
        style = _method_style(config, series["method"]); axis.step(values, y, where="post", color=style["color"], linestyle=style["linestyle"], label=series["label"])
        records.append({"method": series["method"], "n": len(values), "color": style["color"]})
    _style_axis(axis, panel); axis.set_ylim(0, 1.02); axis.legend(frameon=False)
    return {"series": records}


def _point_panel(axis: Any, panel: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    points = panel["data"]["points"]; records = []
    for index, item in enumerate(points):
        style = _method_style(config, item["method"]); estimate = item["estimate"]
        axis.errorbar(index, estimate, yerr=[[estimate - item["lower"]], [item["upper"] - estimate]], fmt=style["marker"], color=style["color"], capsize=2, label=item["method"] if not any(record["method"] == item["method"] for record in records) else None)
        records.append({"label": item["label"], "method": item["method"], "color": style["color"]})
    axis.set_xticks(range(len(points)), [item["label"] for item in points], rotation=25, ha="right")
    _style_axis(axis, panel); axis.legend(frameon=False)
    return {"points": records}


def _display_map(values: Any, *, resize: tuple[int, int] | None = None, order: int = 0, clip: tuple[float, float] | None = None):
    return DisplayTransform(percentile_clip=clip, resize=resize, resize_order=order)(metric_ready(np.asarray(values, dtype=np.float32)))


def _heatmap_panel(axis: Any, panel: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    values = _display_map(panel["data"]["values"]); cmap = panel["data"].get("colormap", config["style"]["sequential_colormap"])
    if cmap in FORBIDDEN_COLORMAPS: raise StageFRendererError("forbidden colormap")
    image = axis.imshow(values.data, cmap=cmap, interpolation="nearest", rasterized=True, aspect="auto")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03); _style_axis(axis, panel)
    return {"colormap": cmap, "interpolation": "nearest", "rasterized_map_only": True, "display_sha256": values.sha256}


def _signed_pair_panel(axis: Any, panel: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    axis.set_axis_off(); data = panel["data"]
    nns = _display_map(data["nns"]); matched = _display_map(data["matched"]); limit = float(max(np.max(np.abs(nns.data)), np.max(np.abs(matched.data))))
    if not np.isfinite(limit) or limit <= 0: raise StageFRendererError("signed maps require a positive shared symmetric limit")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit); cmap = config["style"]["signed_colormap"]
    axes = [axis.inset_axes([0.00, 0.12, 0.41, 0.76]), axis.inset_axes([0.45, 0.12, 0.41, 0.76])]
    images = []
    for child, values, title in zip(axes, (nns, matched), ("NNS", "matched")):
        images.append(child.imshow(values.data, cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)); child.set_title(title, pad=1); child.set_xticks([]); child.set_yticks([])
    cax = axis.inset_axes([0.90, 0.18, 0.025, 0.64]); axis.figure.colorbar(images[0], cax=cax)
    axis.set_title(panel["title"], pad=3)
    return {"colormap": cmap, "shared_colorbar": True, "nns_clim": [-limit, limit], "matched_clim": [-limit, limit], "center": 0.0, "interpolation": "nearest", "rasterized_map_only": True}


def _difference_panel(axis: Any, panel: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    values = _display_map(panel["data"]["values"]); limit = float(np.max(np.abs(values.data)))
    if limit <= 0: raise StageFRendererError("difference map requires a positive symmetric limit")
    cmap = config["style"]["difference_colormap"]; norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = axis.imshow(values.data, cmap=cmap, norm=norm, interpolation="nearest", rasterized=True); axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03); _style_axis(axis, panel)
    return {"colormap": cmap, "color_limit": [-limit, limit], "center": 0.0, "interpolation": "nearest", "rasterized_map_only": True}


def _overlay_panel(axis: Any, panel: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    settings = config["style"]["overlay"]; rgb_tagged = raw(np.asarray(panel["data"]["rgb"], dtype=np.float32)); rgb = validate_raw_input(rgb_tagged)
    target_shape = (rgb.shape[0], rgb.shape[1]); displayed = _display_map(panel["data"]["attribution"], resize=target_shape, order=settings["resize_order"], clip=tuple(settings["percentile_clip"]))
    limit = float(np.max(np.abs(displayed.data))); norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit) if limit > 0 else Normalize(vmin=0, vmax=1)
    axis.imshow(np.clip(rgb, settings["rgb_clip"][0], settings["rgb_clip"][1]), interpolation=settings["rgb_interpolation"]); axis.imshow(displayed.data, cmap=settings["colormap"], norm=norm, interpolation=settings["map_interpolation"], alpha=settings["alpha"], rasterized=True)
    axis.set_xticks([]); axis.set_yticks([]); axis.set_title(panel["title"], pad=3)
    return {"resize": list(target_shape), "resize_order": settings["resize_order"], "clip_percentiles": settings["percentile_clip"], "rgb_clip": settings["rgb_clip"], "alpha": settings["alpha"], "colormap": settings["colormap"], "map_interpolation": settings["map_interpolation"], "rgb_interpolation": settings["rgb_interpolation"], "normalization": "symmetric_about_zero", "display_sha256": displayed.sha256}


def _schematic_panel(axis: Any, panel: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.set_axis_off(); data = panel["data"]
    node_lookup = {node["id"]: node for node in data["nodes"]}
    for node in data["nodes"]:
        x, y = node["position"]; width, height = node.get("size", [0.19, 0.20]); patch = FancyBboxPatch((x-width/2, y-height/2), width, height, boxstyle="round,pad=0.015", facecolor=node.get("facecolor", "#F2F2F2"), edgecolor="#333333", linewidth=config["style"]["minimum_linewidth_pt"]); axis.add_patch(patch); axis.text(x, y, node["text"], ha="center", va="center")
    for arrow in data["arrows"]:
        start=node_lookup[arrow["from"]]["position"]; end=node_lookup[arrow["to"]]["position"]; axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=config["style"]["minimum_linewidth_pt"], color="#333333", shrinkA=18, shrinkB=18))
    axis.set_title(panel["title"], pad=3)
    return {"backend": "matplotlib_vector_patches", "nodes": len(data["nodes"]), "arrows": len(data["arrows"]), "rasterized": False}


def _render_panel(axis: Any, panel: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    dispatch = {
        "line_band": _line_panel, "fidelity_curve": _line_panel,
        "ecdf": _ecdf_panel, "point_interval": _point_panel,
        "heatmap": _heatmap_panel, "signed_attribution_pair": _signed_pair_panel,
        "difference_map": _difference_panel, "rgb_overlay": _overlay_panel,
        "method_schematic": _schematic_panel,
    }
    record = dispatch[panel["kind"]](axis, panel, config)
    axis.text(-0.08, 1.04, panel["label"], transform=axis.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=config["style"]["panel_label_font_pt"], clip_on=False)
    return {"panel_id": panel["panel_id"], "label": panel["label"], "caption_key": panel["caption_key"], "kind": panel["kind"], **record}


def _pdf_object_digest(raw: bytes) -> str:
    normalized = re.sub(br"/(CreationDate|ModDate)\s*\([^)]*\)", br"/\1(FIXED)", raw)
    normalized = re.sub(br"/ID\s*\[<[^>]+><[^>]+>\]", b"/ID[FIXED]", normalized)
    return hashlib.sha256(normalized).hexdigest()


def _save_figure(fig: Any, output_base: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    export = config["export"]; fixed_date = "2026-09-07T00:00:00Z"
    paths = {extension: output_base.with_suffix(f".{extension}") for extension in ("png", "pdf", "svg")}
    if any(path.exists() for path in paths.values()):
        raise StageFRendererError("refusing to overwrite an existing F0 render artifact")
    fig.savefig(paths["png"], format="png", dpi=export["png_dpi"], metadata={"Software": export["producer"]})
    fig.savefig(paths["pdf"], format="pdf", dpi=export["png_dpi"], metadata={"Title": output_base.name, "Author": export["author"], "Creator": export["producer"], "Producer": export["producer"], "CreationDate": datetime(2026, 9, 7, tzinfo=timezone.utc), "ModDate": datetime(2026, 9, 7, tzinfo=timezone.utc)})
    fig.savefig(paths["svg"], format="svg", dpi=export["png_dpi"], metadata={"Date": fixed_date, "Creator": export["producer"], "Title": output_base.name})
    png = Image.open(paths["png"]); png.load()
    expected_pixels = tuple(int(round(value * export["png_dpi"])) for value in fig.get_size_inches())
    if png.size != expected_pixels:
        raise StageFRendererError(f"PNG dimensions {png.size} differ from exact profile pixels {expected_pixels}")
    pdf_raw = paths["pdf"].read_bytes(); svg_raw = paths["svg"].read_bytes()
    if b"/FontFile2" not in pdf_raw and b"/FontFile3" not in pdf_raw:
        raise StageFRendererError("PDF does not contain an embedded font program")
    if b"<path" not in svg_raw:
        raise StageFRendererError("SVG lacks embedded vector glyph paths")
    return {
        "png": {"path": paths["png"].name, "sha256": sha256_file(paths["png"]), "bytes": paths["png"].stat().st_size, "pixel_size": list(png.size), "mode": png.mode},
        "pdf": {"path": paths["pdf"].name, "sha256": sha256_file(paths["pdf"]), "bytes": paths["pdf"].stat().st_size, "internal_object_digest": _pdf_object_digest(pdf_raw), "metadata_normalized": True, "font_embedded": True},
        "svg": {"path": paths["svg"].name, "sha256": sha256_file(paths["svg"]), "bytes": paths["svg"].stat().st_size, "glyphs_embedded_as_paths": True},
    }


def _version(name: str) -> str:
    return importlib.metadata.version(name)


def render_fixture(
    fixture_path: Path, output_dir: Path, *, repo_root: Path | str = REPO_ROOT, renderer_commit: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); contracts = load_contracts(repo); config = contracts["config"]
    fixture, fixture_hash = load_fixture(fixture_path, repo)
    if str(matplotlib.get_backend()).lower() != config["determinism"]["backend"].lower():
        raise StageFRendererError("Matplotlib backend drifted from frozen Agg contract")
    profile = config["layouts"][fixture["layout_profile"]]
    content_digest = canonical_sha256({"fixture": fixture, "config_digest": config["canonical_digest"], "renderer_contract": config["renderer_contract_version"]})
    output_base = output_dir.resolve() / fixture["figure_id"]
    with renderer_style(config) as fonts:
        fig, axes = plt.subplots(fixture["grid"]["rows"], fixture["grid"]["columns"], figsize=(profile["width_in"], profile["height_in"]), squeeze=False, constrained_layout=True)
        panel_records = [_render_panel(axis, panel, config) for axis, panel in zip(axes.flat, fixture["panels"])]
        fig.suptitle(fixture["title"], fontsize=config["style"]["title_font_pt"])
        outputs = _save_figure(fig, output_base, config); plt.close(fig)
    commit = renderer_commit or _git(repo, "rev-parse", "HEAD")
    report = {
        "schema_version": 1, "document_type": "stage_f_render_artifact_report",
        "figure_id": fixture["figure_id"], "title": fixture["title"],
        "fixture_class": "SYNTHETIC_GOLDEN_ONLY", "formal_scientific_figure": False,
        "panel_ids": [panel["panel_id"] for panel in fixture["panels"]],
        "caption_keys": fixture["caption_keys"], "input_fixture_sha256": fixture_hash,
        "renderer_commit": commit, "config_digest": config["canonical_digest"],
        "content_digest": content_digest, "format_content_digests": {extension: content_digest for extension in outputs},
        "environment": {"python": platform.python_version(), "matplotlib": _version("matplotlib"), "pillow": _version("pillow"), "numpy": _version("numpy"), "backend": config["determinism"]["backend"]},
        "fonts": fonts, "layout_profile": fixture["layout_profile"],
        "size_inches": [profile["width_in"], profile["height_in"]], "dpi": config["export"]["png_dpi"],
        "style": {"methods": config["style"]["methods"], "minimum_font_pt": config["style"]["minimum_font_pt"], "minimum_linewidth_pt": config["style"]["minimum_linewidth_pt"], "signed_colormap": config["style"]["signed_colormap"], "difference_colormap": config["style"]["difference_colormap"], "overlay": config["style"]["overlay"], "cropping": config["export"]["cropping"]},
        "panels": panel_records, "outputs": outputs,
        "display_only": True, "primary_decision_modified": False, "stage_d_statistics_modified": False,
        "model_loaded": False, "torch_imported": False, "timm_imported": False, "cuda_used": False, "jpeg_read": False,
    }
    report["canonical_digest"] = canonical_sha256(report)
    _schema_validate(report, repo / RENDER_SCHEMA_PATH, "render artifact report")
    report_path = output_base.with_suffix(".report.json")
    if report_path.exists():
        raise StageFRendererError("refusing to overwrite an existing F0 render report")
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    return report


def validate_render_report(report_path: Path, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); report, raw = _strict_json(report_path, "render report")
    _schema_validate(report, repo / RENDER_SCHEMA_PATH, "render artifact report"); _validate_signed(report, "render report")
    if raw != canonical_json_bytes(report) + b"\n": raise StageFRendererError("render report is not byte-canonical")
    if report.get("formal_scientific_figure") is not False or report.get("display_only") is not True:
        raise StageFRendererError("F0 report crossed the formal/display boundary")
    if len(set(report["format_content_digests"].values())) != 1:
        raise StageFRendererError("PNG/PDF/SVG were not produced from one content identity")
    for extension, record in report["outputs"].items():
        candidate = Path(record["path"])
        if candidate.name != record["path"] or candidate.suffix.lower() != f".{extension}":
            raise StageFRendererError(f"render output path is not a local {extension} file")
        path = report_path.parent / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise StageFRendererError(f"render output tampered: {extension}")
    return {"status": "PASS", "figure_id": report["figure_id"], "content_digest": report["content_digest"], "formats": sorted(report["outputs"])}


def validate_cross_format_consistency(report_path: Path, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Validate one content identity and exact physical canvas in all formats."""

    validate_render_report(report_path, repo_root)
    report, _ = _strict_json(report_path, "render report")
    width_in, height_in = report["size_inches"]; dpi = report["dpi"]
    png_path = report_path.parent / report["outputs"]["png"]["path"]
    pdf_path = report_path.parent / report["outputs"]["pdf"]["path"]
    svg_path = report_path.parent / report["outputs"]["svg"]["path"]
    if png_path.suffix.lower() != ".png" or pdf_path.suffix.lower() != ".pdf" or svg_path.suffix.lower() != ".svg":
        raise StageFRendererError("cross-format report contains a wrong file extension")
    with Image.open(png_path) as image:
        if image.size != (round(width_in * dpi), round(height_in * dpi)):
            raise StageFRendererError("PNG canvas differs from the frozen physical size")
    pdf_raw = pdf_path.read_bytes()
    boxes = re.findall(br"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", pdf_raw)
    if not boxes:
        raise StageFRendererError("PDF media box is absent")
    pdf_size = tuple(float(value) / 72.0 for value in boxes[-1])
    if any(abs(observed - expected) > 1e-5 for observed, expected in zip(pdf_size, (width_in, height_in))):
        raise StageFRendererError("PDF canvas differs from the frozen physical size")
    svg_head = svg_path.read_text(encoding="utf-8")[:1000]
    svg_match = re.search(r'<svg[^>]*width="([0-9.]+)pt"[^>]*height="([0-9.]+)pt"', svg_head)
    if not svg_match:
        raise StageFRendererError("SVG physical dimensions are absent")
    svg_size = (float(svg_match.group(1)) / 72.0, float(svg_match.group(2)) / 72.0)
    if any(abs(observed - expected) > 1e-5 for observed, expected in zip(svg_size, (width_in, height_in))):
        raise StageFRendererError("SVG canvas differs from the frozen physical size")
    if len(set(report["format_content_digests"].values())) != 1:
        raise StageFRendererError("formats do not bind to one input/config content digest")
    return {"status": "PASS", "size_inches": [width_in, height_in], "content_digest": report["content_digest"]}


def assert_f0_source_boundary(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); paths = [repo / "src/vitnns/stage_f_renderer.py", repo / "src/vitnns/stage_f_cli.py"]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for name in FORBIDDEN_IMPORTS:
        if re.search(rf"^\s*(?:from|import)\s+{re.escape(name)}(?:\.|\s|$)", combined, re.MULTILINE):
            raise StageFRendererError(f"F0 source imports forbidden runtime: {name}")
    forbidden_tokens = (
        "." + "cuda(", "cuda" + ":", "load_" + "canonical_model",
        "load_" + "state_dict", "back" + "ward(", "integrated_" + "grad",
    )
    if any(token in combined for token in forbidden_tokens):
        raise StageFRendererError("F0 source exposes model/GPU/JPEG/scientific execution")
    return {"status": "PASS", "model_access": False, "torch_access": False, "timm_access": False, "cuda_access": False, "jpeg_access": False}


__all__ = [
    "ARCHIVE_REF", "F0_REPORT_PATH", "F0_ROOT_PATH", "StageFRendererError",
    "assert_f0_source_boundary", "load_contracts", "load_fixture",
    "parse_upstream_recipes", "render_fixture", "sha256_file",
    "validate_cross_format_consistency", "validate_font",
    "validate_render_report", "verify_preconditions",
]
