#!/usr/bin/env python3
"""Check R7 source identities, display parity, unchanged figures and visual review."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from PIL import Image

import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r7_20260916"
OLD = REPO / "reports/nns_vit_unified_paper_figures_r6_20260916"
CHANGED = [1, 2, 8, 9, 10]
EXCLUSIONS = {"whole_root_inventory.json", "validation_report.json", "terminal_receipt.json"}


def resolve_source(relative):
    path = REPO / relative
    return path if path.is_file() else REPO.parent / relative


def validate(require_review=True):
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    def checked(path):
        value = common.read(path)
        check(common.verify(value), "canonical digest: " + str(path))
        return value

    manifest = checked(ROOT / "render_manifest.json")
    previous = checked(OLD / "render_manifest.json")
    check(manifest["redesigned_figures"] == CHANGED, "redesign scope")
    check(manifest["figure_count"] == 14 and manifest["artifact_count"] == 42, "figure counts")
    check([record["number"] for record in manifest["figures"]] == list(range(1, 15)), "figure roster")
    expected_outputs = []
    for record in manifest["figures"]:
        number = record["number"]
        check(record["redesigned"] == (number in CHANGED), "redesigned flag")
        check(len(record["outputs"]) == 3 and
              {output["format"] for output in record["outputs"]} == {"PNG", "PDF", "SVG"}, "formats")
        for output in record["outputs"]:
            path = REPO / output["path"]
            expected_outputs.append(path)
            check(path.is_relative_to(ROOT) and not path.is_symlink(), "local regular output")
            check(path.is_file() and common.sha(path) == output["sha256"] and
                  path.stat().st_size == output["bytes"], "export identity: " + output["path"])
            prior = next(item for item in previous["figures"][number - 1]["outputs"]
                         if item["format"] == output["format"])
            check((prior["sha256"] == output["sha256"]) == (number not in CHANGED),
                  f"changed/inherited parity: {number}")
            if output["format"] == "PNG":
                with Image.open(path) as image:
                    image.load()
                    check(image.width >= 1200 and image.height >= 700, "PNG dimensions")
                    check(all(abs(value - 400) < 1 for value in image.info.get("dpi", (0, 0))), "PNG dpi")
            elif output["format"] == "PDF":
                check(path.read_bytes().startswith(b"%PDF"), "PDF signature")
            else:
                check(b"<svg" in path.read_bytes()[:4096], "SVG signature")
    actual = [path for folder in ("figures_png", "figures_pdf", "figures_svg")
              for path in (ROOT / folder).iterdir()]
    check(len(actual) == 42 and set(actual) == set(expected_outputs), "exact output roster")
    sources = manifest["source_files"]
    check(common.canonical(sources) == manifest["source_files_digest"], "source inventory digest")
    for source in sources:
        path = resolve_source(source["path"])
        check(path.is_file() and common.sha(path) == source["sha256"] and
              path.stat().st_size == source["bytes"], "source drift: " + source["path"])
    check(checked(ROOT / "input_inventory.json")["source_files"] == sources, "input inventory")
    predecessor = checked(ROOT / "predecessor_snapshot.json")
    check(predecessor["files"] == common.inventory(OLD) and
          common.canonical(predecessor["files"]) == predecessor["files_digest"], "R6 mutation")
    numeric = checked(ROOT / "numeric_display_data.json")
    check(numeric["inherited_R6_numeric"] == checked(OLD / "numeric_display_data.json"),
          "inherited numerical results")
    import render_nns_vit_unified_paper_figures_r7 as renderer

    _, _, recomputed = renderer.load_head_evidence()
    check(numeric["head_display"] == recomputed, "288 maps / 68 original display digests")
    check(len(recomputed["consensus_groups_checked"]) == 68, "all consensus groups")
    check(recomputed["target_query"] == "CLS" and not recomputed["patch_query_maps_available"],
          "CLS versus patch-query distinction")
    check(numeric["f3c_display"]["formal_positions"] == [1, 2, 3, 4, 5], "complete F3C first page")
    formal = common.read(renderer.base.F3C3 / "formal_manifest.json")["samples"]
    selected = [next(row for row in formal if row["formal_position"] == position)["selection_order"]
                for position in range(1, 6)]
    check(numeric["f3c_display"]["selection_orders"] == selected, "F3C sample identities")
    figure, recomputed_f3c = renderer.figure_10("viridis")
    renderer.plt.close(figure)
    check(numeric["f3c_display"] == recomputed_f3c and
          len(recomputed_f3c["original_scale_records_verified"]) == 35, "F3C original scale parity")
    references = checked(ROOT / "reference_gallery_manifest.json")
    groups = Counter(record["group"] for record in references["records"])
    check(groups == {"f3h": 20, "f3c_primary_pages": 10, "f3c_signed_pages": 10,
                     "f3c_unclipped_pages": 10, "method_detail": 3}, "full reference gallery")
    for record in references["records"]:
        source, destination = REPO / record["source_path"], ROOT / record["path"]
        check(common.sha(source) == record["sha256"] == common.sha(destination),
              "reference byte parity: " + record["path"])
        check(destination.stat().st_size == record["bytes"], "reference bytes")
    actual_references = [path for folder in ("reference_gallery", "supplement")
                         for path in (ROOT / folder).rglob("*") if path.is_file()]
    check(set(actual_references) == {ROOT / record["path"] for record in references["records"]},
          "reference file roster")
    for name, marker in (("captions_zh.md", "## Figure "), ("captions_en.md", "## Figure "),
                         ("figure_analysis.md", "## Image #")):
        sections = common.sections(ROOT / name, marker)
        old_sections = common.sections(OLD / name, marker)
        check(set(sections) == set(range(1, 15)), "text coverage: " + name)
        for number in range(1, 15):
            if number not in CHANGED:
                check(sections[number] == old_sections[number], "unchanged caption/analysis")
        if name == "captions_zh.md":
            check("中位效应" in sections[8] and "不是类别归因" in sections[8], "consensus description")
            check("patch-query" in sections[1] and "CLS" in sections[1], "patch-query wording")
        if name == "figure_analysis.md":
            check("500" not in sections[1], "Figure 1 is scanner overview, not cohort infographic")
    for path in [*ROOT.glob("*.md"), ROOT / "gallery.html"]:
        content = path.read_text(encoding="utf-8")
        links = (re.findall(r'(?:href|src)="([^"]+)"', content) if path.suffix == ".html"
                 else re.findall(r"\]\(([^)]+)\)", content))
        for target in links:
            if target.startswith("#"):
                check('id="' + target[1:] + '"' in content, "anchor: " + target)
            elif "://" not in target:
                check((path.parent / target.split("#")[0]).exists(), "broken link: " + target)
    layout = checked(ROOT / "layout_audit.json")
    check([row["number"] for row in layout["records"]] == CHANGED and
          all(not row["text_outside_canvas"] for row in layout["records"]), "layout bounds")
    audit = checked(ROOT / "cpu_access_audit.json")
    check(audit["CUDA_VISIBLE_DEVICES"] == "" and not audit["torch_imported"] and
          not audit["timm_imported"] and not audit["new_inference"], "CPU presentation boundary")
    review_path = ROOT / "manual_visual_qc.json"
    if require_review or review_path.exists():
        if not review_path.exists():
            errors.append("Direct visual review pending")
        else:
            review = checked(review_path)
            check(review["status"] == "PASS" and
                  [row["number"] for row in review["records"]] == list(range(1, 15)), "QC roster")
            for row, figure in zip(review["records"], manifest["figures"]):
                expected_mode = ("direct_visual_review" if figure["number"] in CHANGED
                                 else "inherited_sealed_review_and_byte_parity")
                png = next(output for output in figure["outputs"] if output["format"] == "PNG")
                check(row["status"] == "PASS" and row["png_sha256"] == png["sha256"]
                      and row["review_mode"] == expected_mode, "QC image binding")
    if (ROOT / "terminal_receipt.json").exists():
        terminal = checked(ROOT / "terminal_receipt.json")
        inventory = checked(ROOT / "whole_root_inventory.json")
        check(terminal["status"] == "PASS_PRESENTATION_RENDER" and
              terminal["validator_sha256"] == common.sha(Path(__file__)), "terminal identity")
        check(inventory["files"] == common.inventory(ROOT, EXCLUSIONS) and
              common.canonical(inventory["files"]) == inventory["files_digest"], "sealed inventory")
        for name, digest in terminal["bindings"].items():
            check(common.sha(ROOT / name) == digest, "terminal binding: " + name)
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    if args.seal and args.draft:
        parser.error("--seal requires completed visual review")
    sealed = (ROOT / "terminal_receipt.json").exists()
    if args.seal and sealed:
        parser.error("R7 already sealed")
    errors = validate(require_review=not args.draft)
    report = common.signed({
        "status": "FAIL" if errors else ("PASS_DRAFT_CHECKS" if args.draft else "PASS"),
        "revision": "R7", "errors": errors, "figure_count": 14, "artifact_count": 42,
        "new_figures": CHANGED, "unchanged_figures": 9, "source_map_count": 288,
        "source_consensus_digests_checked": 68, "unchanged_reference_pngs": 50,
        "cpu_only": True, "new_inference": False,
    })
    if not sealed:
        common.write(ROOT / "validation_report.json", report)
    if args.seal and not errors:
        files = common.inventory(ROOT, EXCLUSIONS)
        common.write(ROOT / "whole_root_inventory.json", common.signed({
            "files": files, "files_digest": common.canonical(files)}))
        common.write(ROOT / "terminal_receipt.json", common.signed({
            "status": "PASS_PRESENTATION_RENDER", "revision": "R7",
            "figure_count": 14, "artifact_count": 42,
            "validator_sha256": common.sha(Path(__file__)),
            "bindings": {name: common.sha(ROOT / name) for name in
                         ("render_manifest.json", "validation_report.json", "manual_visual_qc.json",
                          "whole_root_inventory.json", "reference_gallery_manifest.json")},
            "new_inference": False}))
    print(json.dumps(report, ensure_ascii=False))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
