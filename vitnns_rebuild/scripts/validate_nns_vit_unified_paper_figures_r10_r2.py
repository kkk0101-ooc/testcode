#!/usr/bin/env python3
"""Validate the R10-R2 classifier comparison figure package."""
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
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r10_r2_20260917"
OLD = REPO / "reports/nns_vit_unified_paper_figures_r10_r1_20260917"
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
    check(manifest["revision"] == "R10-R2", "revision")
    check(manifest["figure_count"] == 35 and manifest["artifact_count"] == 105, "counts")
    check([item["number"] for item in manifest["figures"]] == list(range(1, 36)), "roster")
    check(manifest["inherited_figures"] == list(range(1, 35)) and manifest["new_figures"] == [35],
          "revision scope")
    expected = []
    for figure in manifest["figures"]:
        check({item["format"] for item in figure["outputs"]} == {"PNG", "PDF", "SVG"},
              f"formats {figure['number']}")
        for item in figure["outputs"]:
            path = REPO / item["path"]
            expected.append(path)
            check(path.is_relative_to(ROOT) and not path.is_symlink(), "local export " + str(path))
            check(path.is_file() and common.sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
                  "output identity " + str(path))
            if "inherited_from" in item:
                source = REPO / item["inherited_from"]
                check(source.is_relative_to(OLD) and common.sha(source) == item["sha256"],
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
    check(old_terminal["status"] == "PASS_PRESENTATION_RENDER" and old_terminal["revision"] == "R10-R1",
          "predecessor terminal")

    array_manifest = signed(ROOT / "source_arrays_manifest.json")
    check(array_manifest["files"] == common.inventory(ROOT / "source_arrays"), "array inventory")
    receipt = signed(ROOT / "classifier_comparison_receipt.json")
    check(receipt["status"] == "PASS_SINGLE_IMAGE_CLASSIFIER_COMPARISON", "compute status")
    check(receipt["target_order"] == ["correct", "competitor", "margin", "low"], "target order")
    targets = receipt["targets"]
    check([target.get("class_id") for target in targets] == [235, 174, None, 171], "target ids")
    check(targets[0]["selection_rule"] == "official ground-truth class and clean argmax", "correct rule")
    check(targets[1]["selection_rule"] == "highest clean logit after excluding correct class", "competitor rule")
    check(targets[3]["selection_rule"] == "global argmin of 1000 clean logits", "low rule")
    check(receipt["official_ground_truth"] == {"validation_image_number": 7922, "wnid": "n02106662",
                                                   "model_class_index": 235, "label": "German shepherd"},
          "official label")
    check(receipt["runtime"]["device"] == "cpu" and receipt["runtime"]["CUDA_VISIBLE_DEVICES"] == "",
          "CPU compute")
    for row in receipt["source_files"]:
        path = REPO / row["path"]
        check(common.sha(path) == row["sha256"] and path.stat().st_size == row["bytes"],
              "compute source drift " + row["path"])

    data = ROOT / "source_arrays"
    rgb = np.load(data / "classifier_four_targets_rgb_ig.npy", allow_pickle=False)
    maps = np.load(data / "classifier_four_targets_maps.npy", allow_pickle=False)
    check(rgb.shape == (4, 3, 224, 224) and np.isfinite(rgb).all(), "RGB maps")
    check(np.array_equal(maps, rgb.sum(1)), "RGB sum")
    check(np.array_equal(rgb[2], rgb[0] - rgb[1]), "margin pixel identity")
    logits_clean = np.load(data / "r10_logits_clean.npy", allow_pickle=False)
    logits_base = np.load(data / "r10_logits_baseline.npy", allow_pickle=False)
    order = np.argsort(logits_clean)[::-1]
    check(int(order[0]) == 235 and int(order[1]) == 174 and int(logits_clean.argmin()) == 171, "selection recompute")
    deltas = np.array([logits_clean[235] - logits_base[235], logits_clean[174] - logits_base[174],
                       (logits_clean[235] - logits_clean[174]) - (logits_base[235] - logits_base[174]),
                       logits_clean[171] - logits_base[171]], dtype=float)
    check(np.allclose([target["delta"] for target in targets], deltas, atol=1e-12, rtol=0), "target deltas")

    previous = None
    for attempt in receipt["attempts"]:
        values = np.load(data / f"classifier_four_targets_rgb_ig_order_{attempt['order']}.npy", allow_pickle=False)
        closure = np.abs(values.sum((1, 2, 3)) - deltas)
        keys = ["correct", "competitor", "margin", "low"]
        check(np.allclose(closure, [attempt["completeness_abs"][key] for key in keys], atol=1e-12, rtol=0),
              "completeness report")
        if previous is not None:
            change = np.sum(np.abs(values - previous), axis=(1, 2, 3)) / np.maximum(
                np.sum(np.abs(values), axis=(1, 2, 3)), 1e-12)
            check(np.allclose(change, attempt["map_L1_relative_change"], atol=1e-12, rtol=0),
                  "convergence report")
        previous = values
    final_attempt = receipt["attempts"][-1]
    check(receipt["selected_order"] == 4096 and np.array_equal(previous, rgb), "selected map")
    check(max(final_attempt["map_L1_relative_change"]) < 0.005, "convergence threshold")
    check(all(final_attempt["completeness_abs"][key] < final_attempt["completeness_tolerance"][key]
              for key in ["correct", "competitor", "margin", "low"]), "completeness thresholds")

    display = signed(ROOT / "numeric_display_data.json")
    check(len(display["display"]) == 1 and display["display"][0]["scale"].startswith("shared signed"),
          "shared display scale")
    for filename, marker in [("captions_zh.md", "## Figure "),
                              ("captions_en.md", "## Figure "),
                              ("figure_analysis.md", "## Image #")]:
        check(set(common.sections(ROOT / filename, marker)) == set(range(1, 36)), "text coverage " + filename)
    for path in [*ROOT.glob("*.md"), ROOT / "gallery.html", REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md"]:
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
    check(layout["status"] == "PASS" and layout["records"] == [{"number": 35, "text_outside_canvas": []}],
          "layout")
    if not draft:
        review = signed(ROOT / "manual_visual_qc.json")
        check(review["status"] == "PASS" and len(review["records"]) == 35, "visual review")
        for row, figure in zip(review["records"], manifest["figures"]):
            png = next(item for item in figure["outputs"] if item["format"] == "PNG")
            check(row["number"] == figure["number"] and row["png_sha256"] == png["sha256"] and row["status"] == "PASS",
                  f"QC binding {figure['number']}")
            expected_mode = "direct_visual_review" if figure["number"] == 35 else "inherited_sealed_review_and_byte_parity"
            check(row["review_mode"] == expected_mode, f"QC mode {figure['number']}")
    if (ROOT / "terminal_receipt.json").exists():
        terminal = signed(ROOT / "terminal_receipt.json")
        inventory = signed(ROOT / "whole_root_inventory.json")
        check(terminal["status"] == "PASS_PRESENTATION_RENDER" and terminal["revision"] == "R10-R2",
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
        "errors": errors, "revision": "R10-R2", "figure_count": 35, "artifact_count": 105,
        "inherited_figures": 34, "new_figures": [35],
        "scope": "single-input classifier-output comparison; no population inference"})
    if not sealed:
        common.write(ROOT / "validation_report.json", report)
    if args.seal and not errors:
        files = common.inventory(ROOT, EXCLUSIONS)
        common.write(ROOT / "whole_root_inventory.json", common.signed({
            "files": files, "files_digest": common.canonical(files), "exclusions": sorted(EXCLUSIONS)}))
        common.write(ROOT / "terminal_receipt.json", common.signed({
            "status": "PASS_PRESENTATION_RENDER", "revision": "R10-R2",
            "figure_count": 35, "artifact_count": 105,
            "validator_sha256": common.sha(Path(__file__)),
            "bindings": {name: common.sha(ROOT / name) for name in [
                "render_manifest.json", "validation_report.json", "whole_root_inventory.json",
                "manual_visual_qc.json", "classifier_comparison_receipt.json", "paper_evidence_audit.json"]},
            "new_inference": "single-input comparison only"}))
    print(json.dumps(report, ensure_ascii=False))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
