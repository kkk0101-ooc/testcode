#!/usr/bin/env python3
"""Validate synchronized M4 result documents and manuscript integration."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
PAPER = REPO / "papers/nns_vit_endpoint"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def main():
    zh = (PAPER / "zh/endpoint_m4_supplement.md").read_text()
    en = (PAPER / "en/endpoint_m4_supplement.md").read_text()
    doc = (REPO / "docs/endpoint_m4_results_20260919.md").read_text()
    full_zh = (PAPER / "zh/full_draft.md").read_text(); full_en = (PAPER / "en/full_draft.md").read_text()
    tests = json.loads((ROOT / "m4_tests.json").read_text())["tests"]
    for text in (zh, en, doc):
        for token in ("86,400", "19,200", "691,200", "230,400", "Image #1", "Image #2", "Image #3"):
            require(token in text, "CONTENT:" + token)
        require("local input dependence" in text or "局部输入依赖" in text, "SCOPE_TERM")
        require("causal importance" in text or "因果重要性" in text, "CAUSAL_BOUNDARY")
        for test in tests:
            require(test["test_id"] in text and f"{test['holm_p']:.3e}" in text, "TEST_SYNC:" + test["test_id"])
    require(full_zh.count("<!-- M4_MECHANISM_START -->") == full_zh.count("<!-- M4_MECHANISM_END -->") == 1, "ZH_MARKER")
    require(full_en.count("<!-- M4_MECHANISM_START -->") == full_en.count("<!-- M4_MECHANISM_END -->") == 1, "EN_MARKER")
    banned = ("causally important head", "causally important branch", "complete vit neuron scan")
    require(not any(term in (zh + en + full_zh + full_en).lower() for term in banned), "OVERCLAIM")
    evidence = (PAPER / "notes/endpoint_m4_evidence_matrix.md").read_text()
    require(evidence.count("| PASS |") >= 6 and "| BOUNDARY |" in evidence, "EVIDENCE_MATRIX")
    report = {"status": "PASS", "documents": 5, "tests_crosschecked": len(tests),
              "figures_described": 3, "scope_boundaries": 2, "bilingual_sync": True}
    (ROOT / "document_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
