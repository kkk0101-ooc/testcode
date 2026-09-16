"""S6 reporting-only utilities for the sealed S5 post-hoc exploratory result."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-nns-s6")

import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Any, Mapping
import xml.etree.ElementTree as ET


STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
STAGE_ID = "S6-POST-HOC-EXPLORATORY-REPORTING"
RUN_ID = "s6_r1_posthoc_exploratory_reporting_20260916"
LABEL = "post-hoc exploratory"
ROOT_REL = Path("evidence/nns_vit_patch_depth/s6_r1_posthoc_exploratory_reporting_20260916")
REPORT_REL = Path("reports/nns_vit_patch_depth_s6_r1_posthoc_exploratory_20260916")
S5_ROOT_REL = Path("evidence/nns_vit_patch_depth/s5_r1_posthoc_exploratory_20260916")
S5_TERMINAL_REL = S5_ROOT_REL / "terminal_receipt.json"
S5_TERMINAL_SHA256 = "3048787522e34b411665bfee774f3d718f7920e9f854d70c5dd5fd5b938f9201"
S5_TERMINAL_CANONICAL = "081d2370b200d8105c71ca91377051b84bc423744617f72856b3db102e13f130"
S5_PUBLICATION_COMMIT = "9d7a8c3e6993e2f76e088948679dec90d0e3dc7b"
SOURCE_NAMES = (
    "endpoint_decisions.json", "image_level_endpoint_values.json", "original_vs_posthoc_P2_comparison.json",
    "depth_profiles_and_D1_D2_relationship.json", "intervention_controls_and_matching_diagnostics.json",
    "validity_accounting.json", "claim_ledger.json", "independent_validation_report.json", "false_pass_mutation_audit.json",
)


class S6Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S6Error(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"file changed while hashing: {path}")
    return digest.hexdigest()


def verify_signed(document: Mapping[str, Any]) -> bool:
    copy = dict(document)
    claimed = copy.pop("canonical_digest", None)
    return isinstance(claimed, str) and claimed == canonical_digest(copy)


def signed(document_type: str, status: str, payload: Mapping[str, Any], producer_commit: str) -> dict[str, Any]:
    document = {"schema_version": 1, "study_id": STUDY_ID, "stage_id": STAGE_ID, "run_id": RUN_ID, "document_type": document_type, "status": status, "payload": dict(payload), "producer_commit": producer_commit}
    document["canonical_digest"] = canonical_digest(document)
    return document


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)


def write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)


def artifact(repo: Path, path: Path) -> dict[str, Any]:
    value = {"analysis_label": LABEL, "path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if isinstance(document, dict) and isinstance(document.get("canonical_digest"), str):
            value["canonical_digest"] = document["canonical_digest"]
    return value


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def authorization_gate(repo: Path, check_remote: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    path = repo / S5_TERMINAL_REL
    require(path.is_file(), "S5 terminal missing")
    file_sha = sha_file(path)
    terminal = json.loads(path.read_text())
    recomputed = canonical_digest({key: value for key, value in terminal.items() if key != "canonical_digest"})
    payload = terminal.get("payload", {})
    branch = git(repo, "symbolic-ref", "--short", "HEAD")
    head = git(repo, "rev-parse", "HEAD")
    tracking = git(repo, "rev-parse", "@{upstream}")
    remote = tracking
    if check_remote:
        lines = git(repo, "ls-remote", "origin", f"refs/heads/{branch}").splitlines()
        require(len(lines) == 1, "live remote branch absent or ambiguous")
        remote = lines[0].split()[0]
    checks = {
        "S5_terminal_file_sha256": file_sha == S5_TERMINAL_SHA256,
        "S5_terminal_claimed_canonical_digest": terminal.get("canonical_digest") == S5_TERMINAL_CANONICAL,
        "S5_terminal_recomputed_canonical_digest": recomputed == S5_TERMINAL_CANONICAL,
        "S5_terminal_status": terminal.get("status") == "PASS_POST_HOC_EXPLORATORY",
        "S5_scientific_status": payload.get("SCIENTIFIC_STATUS") == "PASS_POST_HOC_EXPLORATORY",
        "S5_reporting_authorization": payload.get("next_authorization") == "S6_POST_HOC_EXPLORATORY_REPORTING_ONLY",
        "formal_confirmatory_gate_closed": payload.get("FORMAL_CONFIRMATORY_GATE_OPEN") is False,
        "S5_original_P2_rule_not_passed": payload.get("original_P2_rule_status") == "ORIGINAL_RULE_NOT_PASSED",
        "S5_strong_FWER_claim_false": payload.get("strong_FWER_preserved_claim") is False,
    }
    for label, commit in (("local_head_contains_S5", head), ("tracking_ref_contains_S5", tracking), ("live_remote_contains_S5", remote)):
        check = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", S5_PUBLICATION_COMMIT, commit], check=False)
        checks[label] = check.returncode == 0
    require(all(checks.values()), "S6_NOT_AUTHORIZED")
    return terminal, {"analysis_label": LABEL, "title": "S6 post-hoc exploratory reporting authorization gate", "checks": checks, "all_checks_pass": True, "S5_terminal": {"analysis_label": LABEL, "path": S5_TERMINAL_REL.as_posix(), "bytes": path.stat().st_size, "sha256": file_sha, "canonical_digest": recomputed}, "git": {"analysis_label": LABEL, "branch": branch, "local_head": head, "tracking_ref": tracking, "live_remote": remote, "required_S5_publication_commit": S5_PUBLICATION_COMMIT}, "new_scientific_analysis_authorized": False, "reporting_only": True}


def load_sources(repo: Path, terminal: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    required = terminal["payload"]["required_evidence"]
    sources = {}
    inventory = []
    for name in SOURCE_NAMES:
        require(name in required, f"S5 terminal does not bind required reporting source: {name}")
        binding = required[name]
        path = repo / binding["path"]
        require(path == repo / S5_ROOT_REL / name, f"S5 source path escaped bound root: {name}")
        require(path.is_file() and path.stat().st_size == binding["bytes"] and sha_file(path) == binding["sha256"], f"S5 reporting source drift: {name}")
        document = json.loads(path.read_text())
        require(verify_signed(document) and document["canonical_digest"] == binding["canonical_digest"], f"S5 reporting source canonical drift: {name}")
        sources[name] = document
        inventory.append(artifact(repo, path))
    inventory.insert(0, artifact(repo, repo / S5_TERMINAL_REL))
    return sources, inventory


def figure_data(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    endpoints = sources["endpoint_decisions.json"]["payload"]["endpoints"]
    depth = sources["depth_profiles_and_D1_D2_relationship.json"]["payload"]
    images = sources["image_level_endpoint_values.json"]["payload"]["endpoint_records"]
    comparison = sources["original_vs_posthoc_P2_comparison.json"]["payload"]
    d1 = {row["source_image_id"]: row["value"] for row in images["D1"]}
    d2 = {row["source_image_id"]: row["value"] for row in images["D2"]}
    overlap = sorted(set(d1) & set(d2))
    scatter = [{"analysis_label": LABEL, "source_image_id": source, "D1": d1[source], "D2": d2[source]} for source in overlap]
    require(len(scatter) == depth["D1_D2_relationship"]["paired_valid_source_count"], "S5 paired-source count drift")
    return {
        "analysis_label": LABEL,
        "title": "S6 post-hoc exploratory figure data",
        "endpoint_rows": [{"analysis_label": LABEL, **endpoints[key]} for key in ("P1", "P2", "D1", "D2")],
        "NNS_depth_rows": depth["NNS_equal_class_profile"],
        "D2_depth_rows": depth["D2_equal_class_profile"],
        "D1_D2_scatter_rows": scatter,
        "D1_D2_relationship": depth["D1_D2_relationship"],
        "P2_comparison": comparison,
        "source_transform": "selection_and_formatting_only_from_sealed_S5_outputs; no new inferential endpoint or decision",
    }


def _configure_plotting() -> Any:
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "figure.dpi": 120, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _save_figure(figure: Any, base: Path) -> list[Path]:
    output = []
    for suffix in ("png", "pdf", "svg"):
        path = base.with_suffix("." + suffix)
        figure.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        output.append(path)
    return output


def render(figures_root: Path, data: Mapping[str, Any]) -> list[Path]:
    plt = _configure_plotting()
    figures_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    colors = {"P1": "#0072B2", "P2": "#D55E00", "D1": "#009E73", "D2": "#CC79A7"}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), constrained_layout=True)
    groups = (("P1", "P2"), ("D1", "D2"))
    subtitles = ("Patch evidence", "Depth contrasts")
    for axis, endpoint_names, subtitle in zip(axes, groups, subtitles):
        for y, name in enumerate(endpoint_names):
            row = next(item for item in data["endpoint_rows"] if item["endpoint"] == name)
            lo, hi, estimate = row["ordinary_BCa_95"]["lower"], row["ordinary_BCa_95"]["upper"], row["estimate"]
            axis.errorbar(estimate, y, xerr=[[estimate - lo], [hi - estimate]], fmt="o", color=colors[name], capsize=3, lw=1.5, label=name)
            axis.axvline(row["posthoc_threshold"], color=colors[name], linestyle="--", linewidth=0.8, alpha=0.65)
        axis.set_yticks(range(len(endpoint_names)), endpoint_names)
        axis.set_title(f"Post-hoc exploratory — {subtitle}")
        axis.set_xlabel("Estimate with ordinary BCa 95% CI")
        axis.axvline(0, color="#777777", linewidth=0.6)
        axis.grid(axis="x", alpha=0.2)
    outputs += _save_figure(fig, figures_root / "image_01_endpoint_decisions")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), constrained_layout=True)
    nns_rows = data["NNS_depth_rows"]
    d2_rows = data["D2_depth_rows"]
    axes[0].plot([row["block"] for row in nns_rows], [row["equal_class_position_control_corrected_alignment"] for row in nns_rows], color="#0072B2", marker="o", linewidth=1.5, markersize=3)
    axes[0].axhline(0, color="#777777", linewidth=0.6)
    axes[0].axvspan(-0.25, 3.25, color="#999999", alpha=0.08); axes[0].axvspan(7.75, 11.25, color="#0072B2", alpha=0.08)
    axes[0].set(title="Post-hoc exploratory — NNS alignment", xlabel="ViT block", ylabel="Equal-class corrected alignment", xticks=range(12))
    axes[1].plot([row["block"] for row in d2_rows], [row["equal_class_probe_gain"] for row in d2_rows], color="#CC79A7", marker="s", linewidth=1.5, markersize=3)
    axes[1].axhline(0, color="#777777", linewidth=0.6)
    axes[1].axvspan(-0.25, 3.25, color="#999999", alpha=0.08); axes[1].axvspan(7.75, 11.25, color="#CC79A7", alpha=0.08)
    axes[1].set(title="Post-hoc exploratory — probe gain", xlabel="ViT block", ylabel="Equal-class control MSE − probe MSE", xticks=range(12))
    for axis in axes: axis.grid(alpha=0.2)
    outputs += _save_figure(fig, figures_root / "image_02_depth_profiles")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.5, 3.35), constrained_layout=True)
    scatter = data["D1_D2_scatter_rows"]
    axis.scatter([row["D1"] for row in scatter], [row["D2"] for row in scatter], s=10, alpha=0.45, color="#56B4E9", edgecolors="none")
    axis.axhline(0, color="#777777", linewidth=0.6); axis.axvline(0, color="#777777", linewidth=0.6)
    relationship = data["D1_D2_relationship"]
    axis.text(0.03, 0.97, f"n={relationship['paired_valid_source_count']}, r={relationship['paired_image_Pearson_correlation']:.3f}", transform=axis.transAxes, va="top", ha="left")
    axis.set(title="Post-hoc exploratory — D1/D2 relation", xlabel="D1: NNS deep − shallow", ylabel="D2: probe-gain deep − shallow")
    axis.grid(alpha=0.2)
    outputs += _save_figure(fig, figures_root / "image_03_D1_D2_relationship")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.16, 2.7), constrained_layout=True)
    p2 = next(row for row in data["endpoint_rows"] if row["endpoint"] == "P2")
    observed = [p2["estimate"], p2["adjusted_interval"]["lower"], p2["ordinary_BCa_95"]["half_width"]]
    original = [0.10, 0.10, 0.015]
    posthoc = [0.07, 0.0, 0.02]
    labels = ["Point estimate (higher)", "Adjusted lower bound (higher)", "BCa half-width (lower)"]
    y = list(range(3))
    axis.scatter(observed, y, s=36, color="#0072B2", label="Observed")
    axis.scatter(original, y, marker="x", s=42, color="#D55E00", label="Original rule")
    axis.scatter(posthoc, y, marker="|", s=120, color="#009E73", label="Post-hoc rule")
    axis.set_yticks(y, labels); axis.invert_yaxis(); axis.set_xlabel("Metric value")
    axis.set_title("Post-hoc exploratory — P2 rule comparison (original rule not passed)")
    axis.grid(axis="x", alpha=0.2); axis.legend(ncol=3, frameon=False, loc="lower right")
    outputs += _save_figure(fig, figures_root / "image_04_P2_rule_comparison")
    plt.close(fig)
    return outputs


def png_geometry(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR", f"invalid PNG: {path}")
    width, height = struct.unpack(">II", data[16:24])
    return {"width": width, "height": height}


def validate_render(path: Path) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 1000, f"empty render: {path}")
    record = {"analysis_label": LABEL, "path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path), "format": path.suffix[1:].upper()}
    if path.suffix == ".png":
        record["geometry"] = png_geometry(path)
    elif path.suffix == ".pdf":
        require(path.read_bytes().startswith(b"%PDF-"), f"invalid PDF: {path}")
        record["geometry"] = None
    elif path.suffix == ".svg":
        root = ET.parse(path).getroot()
        require(root.tag.endswith("svg"), f"invalid SVG: {path}")
        record["geometry"] = None
    else:
        raise S6Error(f"unapproved render format: {path}")
    return record


PAYLOAD_KEYS = {
    "s6_authorization_gate_report": {"analysis_label", "title", "checks", "all_checks_pass", "S5_terminal", "git", "new_scientific_analysis_authorized", "reporting_only"},
    "s6_source_binding": {"analysis_label", "title", "S5_terminal", "records", "record_count", "records_digest", "reporting_transform_only"},
    "s6_figure_data": {"analysis_label", "title", "endpoint_rows", "NNS_depth_rows", "D2_depth_rows", "D1_D2_scatter_rows", "D1_D2_relationship", "P2_comparison", "source_transform"},
    "s6_figure_inventory": {"analysis_label", "title", "figures", "figure_count", "formats", "style", "figures_digest"},
    "s6_claim_ledger": {"analysis_label", "title", "claims", "original_P2_rule_status", "forbidden_interpretations", "new_scientific_claim_count"},
    "s6_writing_audit": {"analysis_label", "title", "audit_rows", "paper_level_defensive_writing_score", "highest_priority_removals", "necessary_caveats_to_preserve", "repeated_phrase_audit"},
    "s6_cpu_access_audit": {"analysis_label", "title", "CUDA_VISIBLE_DEVICES", "S5_JSON_read_count", "S3_or_S4_array_read_count", "JPEG_or_mask_read_count", "model_or_checkpoint_load_count", "torch_timm_torchvision_import_count", "CUDA_or_GPU_initialization_count", "new_inference_or_intervention_count", "S0_to_S5_root_write_count"},
    "s6_independent_validation_report": {"analysis_label", "title", "checks", "source_records_digest", "figure_data_digest", "report_sha256", "figure_analysis_sha256", "render_count", "all_checks_pass"},
    "s6_false_pass_mutation_audit": {"analysis_label", "title", "attack_count", "all_rejected", "attacks"},
    "s6_whole_root_inventory": {"analysis_label", "title", "files", "file_count", "files_digest", "excluded_late_bound"},
    "s6_terminal_receipt": {"analysis_label", "title", "S6_WORKFLOW_STATUS", "SCIENTIFIC_STATUS", "reporting_only", "new_scientific_analysis_count", "new_intervention_count", "original_P2_rule_status", "endpoint_states", "paper_conclusion", "required_evidence", "report_copy", "next_authorization"},
}


def verify_closed_document(document: Mapping[str, Any]) -> None:
    require(set(document) == {"schema_version", "study_id", "stage_id", "run_id", "document_type", "status", "payload", "producer_commit", "canonical_digest"}, "open S6 envelope")
    require(verify_signed(document), "S6 canonical digest failed")
    kind = str(document["document_type"])
    require(kind in PAYLOAD_KEYS and set(document["payload"]) == PAYLOAD_KEYS[kind], f"open or incomplete S6 payload: {kind}")
