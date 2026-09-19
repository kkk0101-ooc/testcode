#!/usr/bin/env python3
"""Validate synchronized Endpoint M5 result and manuscript documents."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m5_r2_20260919"
PAPER = REPO / "papers/nns_vit_endpoint"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def main():
    result = (REPO / "docs/endpoint_m5_results_20260919.md").read_text()
    en = (PAPER / "en/endpoint_m5_positioning.md").read_text()
    zh = (PAPER / "zh/endpoint_m5_positioning.md").read_text()
    full_en = (PAPER / "en/full_draft_m5.md").read_text()
    full_zh = (PAPER / "zh/full_draft_m5.md").read_text()
    for text in (result, en, zh, full_en, full_zh):
        for token in ("163,200", "46", "64", "9,600", "0.3782", "0.3000", "0.6742", "0.6238", "41/64", "40/64", "61/64", "512"):
            require(token in text, "CONTENT:" + token)
        require("Input × Gradient" in text, "IDENTITY_TERM")
    require(result.count("\[") == result.count("\]"), "RESULT_DISPLAY_MATH")
    require(full_en.count("<!-- M5_POSITIONING_START -->") == full_en.count("<!-- M5_POSITIONING_END -->") == 1, "EN_MARKER")
    require(full_zh.count("<!-- M5_POSITIONING_START -->") == full_zh.count("<!-- M5_POSITIONING_END -->") == 1, "ZH_MARKER")
    require("### 6.7 Method positioning and reference conditions" in full_en, "EN_SECTION")
    require("### 6.7 方法定位与 reference 信息条件" in full_zh, "ZH_SECTION")
    require("Endpoint 完全无 baseline" not in result + zh + full_zh, "FALSE_BASELINE_CLAIM")
    require("Endpoint is completely baseline-free" not in en + full_en, "FALSE_BASELINE_CLAIM_EN")
    matrix = json.loads((ROOT / "method_applicability_matrix.json").read_text())["methods"]
    for row in matrix:
        require(row["method"] in result and row["method"] in en and row["method"] in zh, "METHOD_MATRIX:" + row["method"])
    audit = (PAPER / "qc/endpoint_m5_writing_audit.md").read_text()
    require("2/10" in audit and "Reviewer-facing prebuttals: 0" in audit and "Promotional adjectives: 0" in audit, "WRITING_AUDIT")
    evidence = (PAPER / "notes/endpoint_m5_evidence_matrix.md").read_text()
    require(evidence.count("| PASS |") >= 5 and evidence.count("| BOUNDARY |") >= 2, "EVIDENCE_MATRIX")
    plan = (REPO / "docs/阶段M_endpointnnsvit.md").read_text()
    require("M5 已完成" in plan and "ENDPOINT_M5_PASS_EXECUTION_AND_AUDIT" in plan, "STAGE_PLAN")
    report = {
        "status": "PASS", "documents": 5, "bilingual_sync": True,
        "method_rows": len(matrix), "required_numeric_tokens": 12,
        "writing_audit": "PASS", "stage_plan_updated": True,
    }
    path = ROOT / "document_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
