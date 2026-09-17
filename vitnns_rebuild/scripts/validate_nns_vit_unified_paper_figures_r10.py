#!/usr/bin/env python3
"""Validate R10 figure inheritance, real maps, documentation, and review bindings."""
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
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r10_r1_20260917"
R9 = REPO / "reports/nns_vit_unified_paper_figures_r9_20260917"
EXCLUSIONS = {"terminal_receipt.json", "validation_report.json", "whole_root_inventory.json"}
CHANNELS = [0, 255, 511, 767]


def validate(draft: bool = False) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def signed(path: Path) -> dict:
        value = common.read(path)
        check(common.verify(value), "digest " + str(path))
        return value

    manifest = signed(ROOT / "render_manifest.json")
    check(manifest["revision"] == "R10-R1", "revision")
    check(manifest["figure_count"] == 34 and manifest["artifact_count"] == 102, "counts")
    check([item["number"] for item in manifest["figures"]] == list(range(1, 35)), "roster")
    check(manifest["inherited_figures"] == list(range(1, 32)), "inherited roster")
    check(manifest["new_figures"] == [32, 33, 34], "new roster")

    expected_outputs = []
    for figure in manifest["figures"]:
        check({item["format"] for item in figure["outputs"]} == {"PNG", "PDF", "SVG"},
              f"formats {figure['number']}")
        for item in figure["outputs"]:
            path = REPO / item["path"]
            expected_outputs.append(path)
            check(path.is_relative_to(ROOT) and not path.is_symlink(), "local export " + str(path))
            check(path.is_file() and common.sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
                  "output identity " + str(path))
            if "inherited_from" in item:
                source = REPO / item["inherited_from"]
                check(source.is_relative_to(R9) and common.sha(source) == item["sha256"],
                      "inherited parity " + str(path))
            if item["format"] == "PNG" and path.is_file():
                with Image.open(path) as image:
                    image.load()
                    check(image.width >= 1200 and image.height >= 700, "PNG size " + str(path))
            elif item["format"] == "PDF" and path.is_file():
                check(path.read_bytes().startswith(b"%PDF"), "PDF header " + str(path))
            elif item["format"] == "SVG" and path.is_file():
                check(b"<svg" in path.read_bytes()[:4096], "SVG header " + str(path))
    actual_outputs = [
        path for folder in ["figures_png", "figures_pdf", "figures_svg"]
        for path in (ROOT / folder).iterdir()
    ]
    check(set(actual_outputs) == set(expected_outputs) and len(actual_outputs) == 102, "exact export roster")

    check(common.canonical(manifest["source_files"]) == manifest["source_files_digest"], "source digest")
    for row in manifest["source_files"]:
        path = REPO / row["path"]
        check(path.is_file() and common.sha(path) == row["sha256"] and path.stat().st_size == row["bytes"],
              "source drift " + row["path"])

    predecessor = signed(ROOT / "predecessor_snapshot.json")
    check(predecessor["files"] == common.inventory(R9), "R9 predecessor changed")
    check(predecessor["files_digest"] == common.canonical(predecessor["files"]), "predecessor digest")
    check(signed(R9 / "terminal_receipt.json")["status"] == "PASS_PRESENTATION_RENDER", "R9 terminal")

    array_manifest = signed(ROOT / "source_arrays_manifest.json")
    check(array_manifest["files"] == common.inventory(ROOT / "source_arrays"), "source array identities")
    receipt = signed(ROOT / "module_compute_receipt.json")
    check(receipt["status"] == "PASS_SINGLE_IMAGE_VIT_EXTENSION_MAPS", "compute status")
    check(receipt["channels"] == CHANNELS and receipt["block"] == 0, "fixed module selection")
    check(receipt["classifier_target"]["class_id"] == 235, "fixed classifier target")
    check(receipt["architecture"] == {
        "blocks": 12, "embedding_width": 768, "patch_tokens": 196,
        "b0_mlp": [768, 3072, 768], "mlp_activation": "GELU",
        "classifier": [768, 1000], "global_pool": "token", "num_prefix_tokens": 1,
    }, "architecture")
    check(receipt["runtime"]["device"] == "cpu" and receipt["runtime"]["CUDA_VISIBLE_DEVICES"] == "",
          "CPU compute")
    check(max(receipt["float32_float64_forward_max_abs"].values()) < 1e-4, "forward parity")
    for row in receipt["source_files"]:
        path = REPO / row["path"]
        check(common.sha(path) == row["sha256"] and path.stat().st_size == row["bytes"],
              "compute source drift " + row["path"])

    data = ROOT / "source_arrays"
    rgb = np.load(data / "b0_extension_rgb_ig.npy", allow_pickle=False)
    maps = np.load(data / "b0_extension_maps.npy", allow_pickle=False)
    classifier_rgb = np.load(data / "classifier_logit_rgb_ig.npy", allow_pickle=False)
    classifier_map = np.load(data / "classifier_logit_map.npy", allow_pickle=False)
    check(rgb.shape == (3, 4, 3, 224, 224) and np.isfinite(rgb).all(), "module RGB shape")
    check(np.array_equal(maps, rgb.sum(2)), "module RGB sum")
    check(classifier_rgb.shape == (3, 224, 224) and np.isfinite(classifier_rgb).all(), "classifier RGB shape")
    check(np.array_equal(classifier_map, classifier_rgb.sum(0)), "classifier RGB sum")

    clean = np.asarray(receipt["module_clean_values"])
    baseline = np.asarray(receipt["module_baseline_values"])
    delta = np.asarray(receipt["module_target_deltas"])
    check(np.allclose(delta, clean - baseline, atol=1e-10, rtol=0), "module deltas")
    check(np.all(np.abs(rgb.sum((2, 3, 4)) - delta) < 1e-4 + 0.001 * np.abs(delta)), "module completeness")
    check(float(np.max(np.abs(rgb[2] - rgb[0] - rgb[1]))) < 1e-9, "second residual additivity")
    r9_rgb = np.load(data / "module_response_rgb_ig.npy", allow_pickle=False)
    check(np.array_equal(rgb[0], r9_rgb[2]), "R9 first residual equality")

    for index, name in enumerate(["first_residual", "mlp_branch", "block_output"]):
        source_name = "residual" if name == "first_residual" else name
        module_clean_tokens = np.load(data / f"{source_name}_clean_tokens.npy", allow_pickle=False)
        module_base_tokens = np.load(data / f"{source_name}_baseline_tokens.npy", allow_pickle=False)
        check(module_clean_tokens.shape == module_base_tokens.shape == (196, 768), "token shape " + name)
        check(np.allclose((module_clean_tokens - module_base_tokens)[:, CHANNELS].sum(0), delta[index], atol=1e-10),
              "token target recompute " + name)

    previous = None
    for attempt in receipt["module_attempts"]:
        values = np.load(data / f"b0_extension_rgb_ig_order_{attempt['order']}.npy", allow_pickle=False)
        closure = np.abs(values.sum((2, 3, 4)) - delta)
        check(np.allclose(closure, attempt["completeness_abs"], atol=1e-12, rtol=0), "module closure report")
        if previous is not None:
            change = np.sum(np.abs(values - previous), axis=(2, 3, 4)) / np.maximum(
                np.sum(np.abs(values), axis=(2, 3, 4)), 1e-12)
            check(np.allclose(change, attempt["map_L1_relative_change"], atol=1e-15, rtol=1e-10),
                  "module convergence report")
        previous = values
    check(receipt["selected_orders"]["module"] == 128 and np.array_equal(previous, rgb), "module selected map")
    check(np.max(receipt["module_attempts"][-1]["map_L1_relative_change"]) < 0.005, "module convergence")

    classifier_target = receipt["classifier_target"]
    logits_clean = np.load(data / "r10_logits_clean.npy", allow_pickle=False)
    logits_base = np.load(data / "r10_logits_baseline.npy", allow_pickle=False)
    classifier_delta = float(logits_clean[235] - logits_base[235])
    check(logits_clean.shape == logits_base.shape == (1000,) and int(logits_clean.argmax()) == 235, "classifier logits")
    check(abs(classifier_delta - classifier_target["delta"]) < 1e-12, "classifier delta")
    previous = None
    for attempt in receipt["classifier_attempts"]:
        values = np.load(data / f"classifier_logit_rgb_ig_order_{attempt['order']}.npy", allow_pickle=False)
        closure = abs(float(values.sum()) - classifier_delta)
        check(abs(closure - attempt["completeness_abs"]) < 1e-12, "classifier closure report")
        if previous is not None:
            change = float(np.sum(np.abs(values - previous))) / max(float(np.sum(np.abs(values))), 1e-12)
            check(abs(change - attempt["map_L1_relative_change"]) < 1e-12, "classifier convergence report")
        previous = values
    check(receipt["selected_orders"]["classifier"] == 4096 and np.array_equal(previous, classifier_rgb),
          "classifier selected map")
    check(receipt["classifier_attempts"][-1]["map_L1_relative_change"] < 0.005, "classifier convergence")
    check(abs(float(classifier_rgb.sum()) - classifier_delta) < 1e-4 + 0.001 * abs(classifier_delta),
          "classifier completeness")
    check(np.load(data / "final_norm_clean_tokens.npy", allow_pickle=False).shape == (197, 768), "final norm tokens")
    check(np.load(data / "classifier_input_clean.npy", allow_pickle=False).shape == (768,), "classifier input")

    for filename, marker in [
        ("captions_zh.md", "## Figure "), ("captions_en.md", "## Figure "),
        ("figure_analysis.md", "## Image #"),
    ]:
        check(set(common.sections(ROOT / filename, marker)) == set(range(1, 35)), "text coverage " + filename)
    documents = [*ROOT.glob("*.md"), ROOT / "gallery.html", REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md"]
    for path in documents:
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
    check(evidence["status"] == "PASS", "paper evidence audit")
    check(evidence["outline_sha256"] == common.sha(REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md"),
          "outline binding")
    layout = signed(ROOT / "layout_audit.json")
    check(layout["status"] == "PASS" and {item["number"] for item in layout["records"]} == {32, 33, 34},
          "layout coverage")

    if not draft:
        review = signed(ROOT / "manual_visual_qc.json")
        check(review["status"] == "PASS" and len(review["records"]) == 34, "visual review")
        for row, figure in zip(review["records"], manifest["figures"]):
            png = next(item for item in figure["outputs"] if item["format"] == "PNG")
            check(row["number"] == figure["number"] and row["png_sha256"] == png["sha256"] and row["status"] == "PASS",
                  f"QC binding {figure['number']}")
            expected_mode = "direct_visual_review" if figure["number"] >= 32 else "inherited_sealed_review_and_byte_parity"
            check(row["review_mode"] == expected_mode, f"QC mode {figure['number']}")

    if (ROOT / "terminal_receipt.json").exists():
        terminal = signed(ROOT / "terminal_receipt.json")
        inventory = signed(ROOT / "whole_root_inventory.json")
        check(terminal["status"] == "PASS_PRESENTATION_RENDER" and terminal["revision"] == "R10-R1", "terminal status")
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
        "errors": errors, "revision": "R10-R1", "figure_count": 34, "artifact_count": 102,
        "inherited_figures": 31, "new_figures": [32, 33, 34],
        "scope": "single-input method illustration; no new population inference",
    })
    if not sealed:
        common.write(ROOT / "validation_report.json", report)
    if args.seal and not errors:
        files = common.inventory(ROOT, EXCLUSIONS)
        common.write(ROOT / "whole_root_inventory.json", common.signed({
            "files": files, "files_digest": common.canonical(files), "exclusions": sorted(EXCLUSIONS),
        }))
        common.write(ROOT / "terminal_receipt.json", common.signed({
            "status": "PASS_PRESENTATION_RENDER", "revision": "R10-R1",
            "figure_count": 34, "artifact_count": 102,
            "validator_sha256": common.sha(Path(__file__)),
            "bindings": {name: common.sha(ROOT / name) for name in [
                "render_manifest.json", "validation_report.json", "whole_root_inventory.json",
                "manual_visual_qc.json", "module_compute_receipt.json", "paper_evidence_audit.json",
            ]},
            "new_inference": "single-input illustration only",
        }))
    print(json.dumps(report, ensure_ascii=False))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
