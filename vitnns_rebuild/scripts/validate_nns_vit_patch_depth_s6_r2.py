#!/usr/bin/env python3
"""Validate S6-R2 presentation-only figure revision."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s6_visual as visual  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--revision", choices=("r2", "r3"), default="r2"); args = parser.parse_args()
    root_rel, report_rel = visual.revision_paths(args.revision); root = REPO / root_rel; report = REPO / report_rel
    source = json.loads((root / "source_binding.json").read_text()); inventory = json.loads((root / "figure_inventory.json").read_text()); access = json.loads((root / "access_audit.json").read_text())
    require(visual.verify_signed(source) and visual.verify_signed(inventory) and visual.verify_signed(access), "canonical digest failure")
    inputs, source_records = visual.load_inputs(REPO, paper_safe=args.revision == "r3")
    require(source["payload"]["records"] == source_records and source["payload"]["records_digest"] == visual.canonical_digest(source_records), "source binding drift")
    require(source["payload"]["S6_R1_modified"] is False and source["payload"]["new_scientific_analysis_count"] == 0, "presentation scope drift")
    records = inventory["payload"]["figures"]
    require(len(records) == inventory["payload"]["figure_count"] == 12 and inventory["payload"]["figures_digest"] == visual.canonical_digest(records), "figure inventory drift")
    require(inventory["payload"]["S6_R1_scientific_values_changed"] is False, "scientific value change claimed")
    fourth = "image_04_matched_occlusion_evidence.png" if args.revision == "r3" else "image_04_P2_decision_panels.png"
    expected_png = {"image_01_evidence_atlas.png", "image_02_depth_progression.png", "image_03_D1_D2_relation.png", fourth}
    actual_png = set()
    for record in records:
        path = REPO / record["path"]; require(path.is_file() and path.stat().st_size == record["bytes"] and visual.sha_file(path) == record["sha256"], f"render drift: {record['path']}")
        if path.suffix == ".png":
            data = path.read_bytes(); require(data[:8] == b"\x89PNG\r\n\x1a\n", "PNG header"); width, height = struct.unpack(">II", data[16:24]); require(record["geometry"] == {"width": width, "height": height} and width >= 2000 and height >= 1500, "PNG geometry"); actual_png.add(path.name)
            require((report / "figures_png" / path.name).read_bytes() == path.read_bytes(), "report PNG parity")
        elif path.suffix == ".pdf": require(path.read_bytes().startswith(b"%PDF-"), "PDF header")
        elif path.suffix == ".svg": require(ET.parse(path).getroot().tag.endswith("svg"), "SVG root")
        else: raise RuntimeError("unapproved format")
    require(actual_png == expected_png, "PNG figure set drift")
    require((report / "figure_analysis.md").read_bytes() == (root / "figure_analysis.md").read_bytes(), "analysis parity")
    audit = access["payload"]
    require(audit["reference_JPEG_style_read_count"] == 0 and audit["scientific_array_read_count"] == 0 and audit["model_or_checkpoint_load_count"] == 0 and audit["CUDA_or_GPU_initialization_count"] == 0 and audit["new_scientific_analysis_count"] == 0 and audit["S6_R1_write_count"] == 0, "presentation isolation")
    require(visual.sha_file(REPO / visual.S6_ROOT / "terminal_receipt.json") == "56491d0b1e4cac51d8e664f2410e686af180c15a8cfe86e99296e0c09b1a963a", "S6-R1 terminal changed")
    if args.revision == "r3":
        analysis = (root / "figure_analysis.md").read_text(); require("Matched occlusion evidence" in analysis and "matching coverage" in analysis, "paper figure analysis drift")
        forbidden_paper_terms = ("Original: FAIL", "S5: PASS", "内部 PASS/FAIL", "改为", "规则调整", "outcome access 后确定")
        require(not any(term in analysis for term in forbidden_paper_terms), "internal governance wording leaked into paper-facing analysis")
        require(not any("P2_decision_panels" in record["path"] for record in records), "governance figure leaked into R3")
    result = {"status": "PASS_PRESENTATION_ONLY", "revision": args.revision, "figure_count": 12, "PNG_count": 4, "source_records_digest": source["payload"]["records_digest"], "figures_digest": inventory["payload"]["figures_digest"], "S6_R1_unchanged": True, "new_scientific_analysis_count": 0}
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
