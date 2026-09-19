#!/usr/bin/env python3
"""Validate synchronized Endpoint M6 documents and claim language."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m6_r1_20260919"
PAPER = REPO / "papers/nns_vit_endpoint"


def require(condition, code):
    if not condition: raise ValueError(code)


def main():
    selection = json.loads((ROOT / "selection_manifest.json").read_text())
    paths = [REPO / "docs/endpoint_m6_results_20260919.md", PAPER / "zh/endpoint_m6_atlas.md", PAPER / "en/endpoint_m6_atlas.md",
             PAPER / "zh/full_draft_m6.md", PAPER / "en/full_draft_m6.md"]
    texts = [path.read_text() for path in paths]
    for case in selection["cases"]:
        for text in texts: require(case["case_id"] in text, "CASE_TEXT:" + case["case_id"])
    for text in texts:
        for number in range(1, 6): require("Image #%d" % number in text, "IMAGE_TEXT")
        require("local input dependence" in text or "Endpoint dependence" in text, "DEPENDENCE_LANGUAGE")
        require("因果证据" not in text and "causal evidence" not in text.lower(), "CAUSAL_OVERCLAIM")
    require(texts[3].count("<!-- M6_ATLAS_START -->") == texts[3].count("<!-- M6_ATLAS_END -->") == 1, "ZH_MARKER")
    require(texts[4].count("<!-- M6_ATLAS_START -->") == texts[4].count("<!-- M6_ATLAS_END -->") == 1, "EN_MARKER")
    require((PAPER / "figure_manifest_m6.json").read_text() == (ROOT / "figure_manifest.json").read_text(), "FIGURE_MANIFEST_SYNC")
    evidence = (PAPER / "notes/endpoint_m6_evidence_matrix.md").read_text(); require(evidence.count("| PASS |") == 8, "EVIDENCE_MATRIX")
    audit = (PAPER / "qc/endpoint_m6_writing_audit.md").read_text(); require("1/10" in audit and "Reviewer-facing prebuttals: 0" in audit and "Promotional adjectives: 0" in audit, "WRITING_AUDIT")
    report = {"status": "PASS", "documents": len(paths), "bilingual_sync": True, "case_identifiers": 12,
              "numbered_figures": 5, "evidence_rows": 8, "writing_audit": "PASS"}
    (ROOT / "document_validation.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report))


if __name__ == "__main__":
    main()
