#!/usr/bin/env python3
"""Validate R10-R4 Figure-2 attention-channel-255 positive display."""
from __future__ import annotations

import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re
from pathlib import Path

import numpy as np
from PIL import Image

import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r10_r4_20260917"
OLD = REPO / "reports/nns_vit_unified_paper_figures_r10_r2_20260917"
EXCLUSIONS = {"terminal_receipt.json", "validation_report.json", "whole_root_inventory.json"}


def validate(draft: bool = False) -> list[str]:
    errors = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def signed(path: Path) -> dict:
        value = common.read(path)
        check(common.verify(value), "digest " + str(path))
        return value

    manifest = signed(ROOT / "render_manifest.json")
    check(manifest["revision"] == "R10-R4", "revision")
    check(manifest["figure_count"] == 35 and manifest["artifact_count"] == 105, "counts")
    check([item["number"] for item in manifest["figures"]] == list(range(1, 36)), "roster")
    check(manifest["redesigned_figures"] == [2] and
          manifest["inherited_figures"] == [number for number in range(1, 36) if number != 2],
          "revision scope")
    expected = []
    old_manifest = signed(OLD / "render_manifest.json")
    for figure in manifest["figures"]:
        check({item["format"] for item in figure["outputs"]} == {"PNG", "PDF", "SVG"},
              f"formats {figure['number']}")
        for item in figure["outputs"]:
            path = REPO / item["path"]
            expected.append(path)
            check(path.is_relative_to(ROOT) and not path.is_symlink(), "local export " + str(path))
            check(path.is_file() and common.sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
                  "output identity " + str(path))
            if figure["number"] == 2:
                predecessor_item = next(row for row in old_manifest["figures"][1]["outputs"]
                                        if row["format"] == item["format"])
                check(item["sha256"] != predecessor_item["sha256"], "Figure 2 must be redesigned " + item["format"])
                check("inherited_from" not in item, "Figure 2 inherited marker")
            else:
                check("inherited_from" in item and common.sha(REPO / item["inherited_from"]) == item["sha256"],
                      "inheritance parity " + str(path))
            if item["format"] == "PNG" and path.is_file():
                with Image.open(path) as image:
                    image.load()
                    check(image.width >= 1200 and image.height >= 700, "PNG size " + str(path))
            elif item["format"] == "PDF" and path.is_file():
                check(path.read_bytes().startswith(b"%PDF"), "PDF header " + str(path))
            elif item["format"] == "SVG" and path.is_file():
                check(b"<svg" in path.read_bytes()[:4096], "SVG header " + str(path))
    actual = [path for folder in ["figures_png", "figures_pdf", "figures_svg"]
              for path in (ROOT / folder).iterdir()]
    check(set(actual) == set(expected) and len(actual) == 105, "exact export roster")
    check(common.canonical(manifest["source_files"]) == manifest["source_files_digest"], "source digest")
    for row in manifest["source_files"]:
        path = REPO / row["path"]
        check(path.is_file() and common.sha(path) == row["sha256"] and path.stat().st_size == row["bytes"],
              "source drift " + row["path"])

    predecessor = signed(ROOT / "predecessor_snapshot.json")
    check(predecessor["files"] == common.inventory(OLD), "predecessor changed")
    check(predecessor["files_digest"] == common.canonical(predecessor["files"]), "predecessor digest")
    old_terminal = signed(OLD / "terminal_receipt.json")
    check(old_terminal["status"] == "PASS_PRESENTATION_RENDER" and old_terminal["revision"] == "R10-R2",
          "predecessor terminal")
    arrays = signed(ROOT / "source_arrays_manifest.json")
    check(arrays["files"] == common.inventory(ROOT / "source_arrays"), "source array inventory")
    old_arrays = signed(OLD / "source_arrays_manifest.json")
    check(arrays["files"] == old_arrays["files"], "source arrays inherited byte-for-byte")

    display = signed(ROOT / "figure_02_display_receipt.json")
    check(display["status"] == "PASS_DISPLAY_ONLY_POSITIVE_RESTYLE", "display status")
    check(display["model_forward_calls"] == 0 and display["attribution_calls"] == 0 and
          not display["new_scientific_inference"], "display-only boundary")
    raw = np.load(ROOT / "source_arrays/module_response_maps.npy", allow_pickle=False)[1, 1]
    clean = np.load(ROOT / "source_arrays/attention_clean_tokens.npy", allow_pickle=False)
    baseline = np.load(ROOT / "source_arrays/attention_baseline_tokens.npy", allow_pickle=False)
    delta = float((clean - baseline)[:, 255].sum())
    positive = np.maximum(raw, 0)
    limit = float(np.percentile(positive, 99)) or float(positive.max())
    check(raw.shape == (224, 224) and np.isfinite(raw).all(), "Figure 2 raw map")
    check(abs(float(raw.sum()) - delta) < 1e-9, "signed completeness")
    check(abs(display["signed_sum"] - float(raw.sum())) < 1e-12, "reported signed sum")
    check(abs(display["token_recomputed_delta"] - delta) < 1e-12, "reported token delta")
    check(abs(display["positive_mass"] - float(positive.sum())) < 1e-12, "reported positive mass")
    check(abs(display["p99_limit"] - limit) < 1e-15, "reported P99")
    check(display["display_clipped_count"] == int((positive > limit).sum()), "display clipping")
    check("max(raw,0)" in display["display_rule"] and "R4 heat palette" in display["display_rule"],
          "display rule")
    check(display["source_array"] == "source_arrays/module_response_maps.npy[1,1]" and
          display["source_figure_panel"] == "Figure 30, channel 255, top row", "Figure 30 binding")
    check(len(display["signed_companions"]) == 3, "signed companion roster")
    for record in display["signed_companions"]:
        path = REPO / record["path"]
        source = REPO / record["source_path"]
        check(record["selected_channel"] == 255 and path.name.startswith("figure_30_signed") and
              path.is_file() and common.sha(path) == record["sha256"] == common.sha(source),
              "signed companion parity " + record["format"])
    check(len(display["predecessor_display_companions"]) == 3 and
          all(row["historical_target"] == "attention channel 0" for row in display["predecessor_display_companions"]),
          "historical Figure 2 companion boundary")

    numeric = signed(ROOT / "numeric_display_data.json")
    rows = [row for row in numeric["display"] if row["panel"] == "figure2_positive"]
    check(len(rows) == 1 and rows[0]["scale"].startswith("positive P99") and
          rows[0]["clipped_count"] == display["display_clipped_count"], "numeric display binding")
    for filename, marker in [("captions_zh.md", "## Figure "),
                              ("captions_en.md", "## Figure "),
                              ("figure_analysis.md", "## Image #")]:
        sections = common.sections(ROOT / filename, marker)
        check(set(sections) == set(range(1, 36)), "text coverage " + filename)
        check("P99" in sections[2] and ("正" in sections[2] if filename != "captions_en.md" else "max(Ls,0)" in sections[2]),
              "Figure 2 text " + filename)
    for path in [*ROOT.glob("*.md"), ROOT / "gallery.html",
                 REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md",
                 REPO / "docs/unified_paper_figures_r10_r4_figure2_attention_ch255_positive_design_20260917.md"]:
        text = path.read_text(encoding="utf-8")
        check(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text) is None, "control character " + str(path))
        links = re.findall(r'(?:href|src)="([^"]+)"', text) if path.suffix == ".html" else re.findall(r"\]\(([^)]+)\)", text)
        for target in links:
            target = target.strip("<>")
            if target.startswith("#"):
                check('id="' + target[1:] + '"' in text, "anchor " + target)
            elif "://" not in target:
                check((path.parent / target.split("#")[0]).exists(), "link " + str(path) + " " + target)
    evidence = signed(ROOT / "paper_evidence_audit.json")
    check(evidence["status"] == "PASS" and evidence["outline_sha256"] == common.sha(
        REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md"), "paper evidence")
    layout = signed(ROOT / "layout_audit.json")
    check(layout["status"] == "PASS" and layout["records"] == [{"number": 2, "text_outside_canvas": []}],
          "layout")
    if not draft:
        review = signed(ROOT / "manual_visual_qc.json")
        check(review["status"] == "PASS" and len(review["records"]) == 35, "visual review")
        for row, figure in zip(review["records"], manifest["figures"]):
            png = next(item for item in figure["outputs"] if item["format"] == "PNG")
            check(row["number"] == figure["number"] and row["png_sha256"] == png["sha256"] and row["status"] == "PASS",
                  f"QC binding {figure['number']}")
            expected_mode = "direct_visual_review" if figure["number"] == 2 else "inherited_sealed_review_and_byte_parity"
            check(row["review_mode"] == expected_mode, f"QC mode {figure['number']}")
    if (ROOT / "terminal_receipt.json").exists():
        terminal = signed(ROOT / "terminal_receipt.json")
        inventory = signed(ROOT / "whole_root_inventory.json")
        check(terminal["status"] == "PASS_PRESENTATION_RENDER" and terminal["revision"] == "R10-R4",
              "terminal status")
        check(terminal["validator_sha256"] == common.sha(Path(__file__)), "validator binding")
        check(inventory["files"] == common.inventory(ROOT, EXCLUSIONS), "sealed inventory")
        for name, digest in terminal["bindings"].items():
            check(common.sha(ROOT / name) == digest, "receipt " + name)
    return errors


def main() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    sealed = (ROOT / "terminal_receipt.json").exists()
    if args.seal and (sealed or args.draft):
        parser.error("Need an unsealed reviewed root")
    errors = validate(args.draft)
    report = common.signed({
        "status": "FAIL" if errors else "PASS_DRAFT_CHECKS" if args.draft else "PASS",
        "errors": errors, "revision": "R10-R4", "figure_count": 35, "artifact_count": 105,
        "redesigned_figures": [2], "inherited_figures": 34,
        "scope": "Figure 2 uses existing Figure 30 attention channel 255 positive display"})
    if not sealed:
        common.write(ROOT / "validation_report.json", report)
    if args.seal and not errors:
        files = common.inventory(ROOT, EXCLUSIONS)
        common.write(ROOT / "whole_root_inventory.json", common.signed({
            "files": files, "files_digest": common.canonical(files), "exclusions": sorted(EXCLUSIONS)}))
        common.write(ROOT / "terminal_receipt.json", common.signed({
            "status": "PASS_PRESENTATION_RENDER", "revision": "R10-R4",
            "figure_count": 35, "artifact_count": 105,
            "validator_sha256": common.sha(Path(__file__)),
            "bindings": {name: common.sha(ROOT / name) for name in [
                "render_manifest.json", "validation_report.json", "whole_root_inventory.json",
                "manual_visual_qc.json", "figure_02_display_receipt.json", "paper_evidence_audit.json"]},
            "new_inference": False}))
    print(json.dumps(report, ensure_ascii=False))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
