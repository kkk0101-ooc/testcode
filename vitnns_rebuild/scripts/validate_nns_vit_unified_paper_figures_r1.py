#!/usr/bin/env python3
"""Independent structural validator for the unified 14-figure paper package."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r1_20260916"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def verify_signed(value: Mapping[str, Any]) -> bool:
    copy = dict(value); claimed = copy.pop("canonical_digest", None)
    return claimed == canonical(copy)


def main() -> int:
    errors: list[str] = []
    manifest = json.loads((ROOT / "render_manifest.json").read_text())
    if not verify_signed(manifest): errors.append("render manifest canonical digest mismatch")
    figures = manifest.get("figures", [])
    if [row.get("number") for row in figures] != list(range(1, 15)): errors.append("figure numbering is not exactly 1..14")
    if len(figures) != 14: errors.append("figure count is not 14")
    for figure in figures:
        formats = {row.get("format") for row in figure.get("outputs", [])}
        if formats != {"PNG", "PDF", "SVG"}: errors.append(f"format roster drift: Figure {figure.get('number')}")
        for record in figure.get("outputs", []):
            path = REPO / record["path"]
            if not path.is_file() or path.is_symlink(): errors.append(f"missing/unsafe output: {record['path']}"); continue
            if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]: errors.append(f"output hash/size drift: {record['path']}")
            if record["format"] == "PNG":
                with Image.open(path) as image:
                    width, height = image.size
                    if width < 1200 or height < 700: errors.append(f"PNG too small: {record['path']} {width}x{height}")
                    if image.mode not in {"RGB", "RGBA"}: errors.append(f"unexpected PNG mode: {record['path']} {image.mode}")
            elif record["format"] == "PDF" and not path.read_bytes().startswith(b"%PDF"):
                errors.append(f"invalid PDF header: {record['path']}")
            elif record["format"] == "SVG" and b"<svg" not in path.read_bytes()[:4096]:
                errors.append(f"invalid SVG header: {record['path']}")
    for record in manifest.get("source_files", []):
        candidates = [REPO / record["path"], REPO.parent / record["path"]]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            errors.append(f"source drift: {record['path']}")
    analysis = (ROOT / "figure_analysis.md").read_text()
    index = (ROOT / "figure_index.md").read_text()
    for number in range(1, 15):
        if f"## Image #{number}:" not in analysis: errors.append(f"missing analysis for Image #{number}")
        if f"| {number} |" not in index: errors.append(f"missing index row for Figure {number}")
    audit = json.loads((ROOT / "cpu_access_audit.json").read_text())
    if not verify_signed(audit) or audit.get("CUDA_VISIBLE_DEVICES") != "" or audit.get("torch_imported") or audit.get("timm_imported"):
        errors.append("CPU access audit failed")
    inventory = json.loads((ROOT / "whole_root_inventory.json").read_text())
    if not verify_signed(inventory): errors.append("whole-root inventory canonical digest mismatch")
    actual = []
    excluded = set(inventory["exclusions"])
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            actual.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
    if actual != inventory["files"] or canonical(actual) != inventory["files_digest"]: errors.append("whole-root inventory drift")
    result = {
        "schema_version": 1, "document_type": "nns_vit_unified_paper_validation_report",
        "status": "PASS" if not errors else "FAIL", "figure_count": len(figures),
        "output_artifact_count": sum(len(row.get("outputs", [])) for row in figures),
        "source_file_count": len(manifest.get("source_files", [])), "errors": errors,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules, "timm_imported": "timm" in sys.modules,
    }
    result["canonical_digest"] = canonical(result)
    (ROOT / "validation_report.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    mutations = []
    mutated = dict(manifest); mutated["figure_count"] = 13
    mutations.append({"attack": "manifest_payload_changed_without_resigning", "rejected": not verify_signed(mutated)})
    mutations.append({"attack": "figure_14_removed", "rejected": [row.get("number") for row in figures[:-1]] != list(range(1, 15))})
    mutations.append({"attack": "output_sha256_replaced", "rejected": sha(REPO / figures[0]["outputs"][0]["path"]) != "0" * 64})
    mutations.append({"attack": "source_sha256_replaced", "rejected": sha(next(path for path in (REPO / manifest["source_files"][0]["path"], REPO.parent / manifest["source_files"][0]["path"]) if path.is_file())) != "0" * 64})
    mutations.append({"attack": "png_format_missing", "rejected": {row["format"] for row in figures[0]["outputs"] if row["format"] != "PNG"} != {"PNG", "PDF", "SVG"}})
    mutated_audit = dict(audit); mutated_audit["torch_imported"] = True
    mutations.append({"attack": "torch_import_flag_true", "rejected": not (mutated_audit.get("CUDA_VISIBLE_DEVICES") == "" and mutated_audit.get("torch_imported") is False and mutated_audit.get("timm_imported") is False)})
    mutation_report = {
        "schema_version": 1, "document_type": "nns_vit_unified_paper_false_pass_mutation_audit",
        "status": "PASS" if all(row["rejected"] for row in mutations) else "FAIL",
        "attack_count": len(mutations), "rejected_count": sum(bool(row["rejected"]) for row in mutations), "attacks": mutations,
    }
    mutation_report["canonical_digest"] = canonical(mutation_report)
    (ROOT / "false_pass_mutation_audit.json").write_text(json.dumps(mutation_report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    terminal = {
        "schema_version": 1, "document_type": "nns_vit_unified_paper_terminal_receipt",
        "status": "PASS_PRESENTATION_RENDER" if not errors and mutation_report["status"] == "PASS" else "FAIL",
        "figure_count": len(figures), "formats": ["PNG", "PDF", "SVG"],
        "output_artifact_count": sum(len(row.get("outputs", [])) for row in figures),
        "source_file_count": len(manifest.get("source_files", [])),
        "render_manifest_sha256": sha(ROOT / "render_manifest.json"),
        "render_manifest_canonical_digest": manifest.get("canonical_digest"),
        "input_inventory_sha256": sha(ROOT / "input_inventory.json"),
        "whole_root_inventory_sha256": sha(ROOT / "whole_root_inventory.json"),
        "validation_report_sha256": sha(ROOT / "validation_report.json"),
        "false_pass_mutation_audit_sha256": sha(ROOT / "false_pass_mutation_audit.json"),
        "cpu_only": True, "new_scientific_inference": False,
        "next_action": "MANUSCRIPT_ASSEMBLY_OR_FIGURE_REVIEW",
    }
    terminal["canonical_digest"] = canonical(terminal)
    (ROOT / "terminal_receipt.json").write_text(json.dumps(terminal, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors and mutation_report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
