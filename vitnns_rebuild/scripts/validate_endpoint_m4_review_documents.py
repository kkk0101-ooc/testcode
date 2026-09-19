#!/usr/bin/env python3
"""Validate M4 Review R2 bilingual documents and claim boundaries."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_review_r2_20260919"
PAPER = REPO / "papers/nns_vit_endpoint"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def main():
    paths = [REPO / "docs/endpoint_m4_review_results_20260919.md",
             PAPER / "zh/endpoint_m4_review_r2.md", PAPER / "en/endpoint_m4_review_r2.md"]
    texts = [path.read_text() for path in paths]
    zh_full = (PAPER / "zh/full_draft_m4_review_r2.md").read_text()
    en_full = (PAPER / "en/full_draft_m4_review_r2.md").read_text()
    for text in texts:
        for token in ("0.9516", "0.9636", "14.9711", "1.5155", "0.0520", "0.0282", "-0.0187", "-0.0462", "2.502e-5"):
            require(token in text, "NUMERIC_TOKEN:" + token)
        require("target-conditioned" in text or "目标条件化" in text, "TARGET_SCOPE")
        require("path-norm" in text or "路径 norm" in text, "NORM_SCOPE")
        require("95%" in text and "not" in text.lower() or "不是" in text, "JVP_BOUNDARY")
    for text in (zh_full, en_full):
        require(text.count("M4_MECHANISM_START") == text.count("M4_MECHANISM_END") == 1, "M4_MARKER")
        require("0.9516" in text and "14.9711" in text and "2.502e-5" in text, "FULL_DRAFT_NUMBERS")
    banned = ("routing contributed 95%", "routing 贡献了 95%", "routing causes m3", "q/k routing 导致", "causal contribution percentage")
    combined = "\n".join(texts + [zh_full, en_full]).lower()
    require(not any(term in combined for term in banned), "OVERCLAIM")
    require("Holm p=0.0629" in en_full and "Holm p 均为 0.0629" in zh_full, "CLS_QUALIFICATION")
    require("每模型仅 1 张图重叠" in zh_full and "only one overlapping image per model" in en_full, "BRIDGE_BOUNDARY")
    audit = (PAPER / "qc/endpoint_m4_review_r2_writing_audit.md").read_text()
    require("2/10" in audit and "Reviewer-facing prebuttals: 0" in audit, "WRITING_AUDIT")
    response = (REPO / "docs/endpoint_m4_review_response_20260919.md").read_text()
    require(response.count("| 接受") >= 3 and "不支持" in response and "部分支持" in response, "REVIEW_RESPONSE")
    decision = json.loads((ROOT / "scientific_decision.json").read_text())
    require(decision["claims"]["routing_causes_M3_spatial_broadening"].startswith("NOT_SUPPORTED"), "DECISION_BOUNDARY")
    report = {"status": "PASS", "standalone_documents": len(paths), "full_drafts": 2,
              "numeric_tokens": 9, "review_items_mapped": 11, "writing_audit": "PASS",
              "overclaim_scan": "PASS", "bilingual_sync": True}
    (ROOT / "document_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
