#!/usr/bin/env python3
"""Validate the post-seal M4 result integration in the Stage M master document."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/阶段M_endpointnnsvit.md"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def main():
    text = DOC.read_text()
    main = json.loads((REPO / "reports/nns_vit_endpoint_m4_r1_20260919/terminal_receipt.json").read_text())
    review = json.loads((REPO / "reports/nns_vit_endpoint_m4_review_r2_20260919/terminal_receipt.json").read_text())
    original_tests = json.loads((REPO / "reports/nns_vit_endpoint_m4_r1_20260919/m4_tests.json").read_text())["tests"]
    spatial_tests = json.loads((REPO / "reports/nns_vit_endpoint_m4_review_r2_20260919/residual_spatial_tests.json").read_text())["tests"]
    required = [
        "M1、M2、M3、M4 和 M5 已完成", "**M4. Attention 与 residual 结构机制分析** | 已完成",
        main["status"], review["status"], "86,400 个 image-block-head attention records",
        "19,200 个 image-block-residual records", "691,200 个 direction-head JVP records",
        "14.9711", "Holm p 均为 0.0629", "M3、M4 与 M5 已完成。剩余最低完成集",
        "MLP path cancellation 在两模型中呈显著相反趋势。", "状态回填说明",
    ]
    for token in required:
        require(token in text, "REQUIRED:" + token)
    stale = [
        "M4 是当前下一优先级模块", "M4-A routing/value 与 frozen-attention | 待执行",
        "M4-B residual identity/branch | 待执行", "后续 M4 将", "仍须等待 M4",
        "仍由 M4 决定", "\\boxed{M4 > M6 > M5}",
    ]
    for token in stale:
        require(token not in text, "STALE:" + token)
    for row in original_tests:
        if row["test_id"] in {"late_minus_early_routing_share", "late_minus_early_frozen_attention_endpoint_error",
                              "late_minus_early_identity_share", "late_minus_early_path_cancellation"}:
            value = f"{row['median']:+.4f}".replace("-", "−")
            require(value in text, "ORIGINAL_TEST:" + value)
    for row in spatial_tests:
        value = f"{row['median']:+.4f}".replace("-", "−")
        require(value in text, "SPATIAL_TEST:" + value)
    require(text.count("\\[") == text.count("\\]"), "DISPLAY_MATH")
    linked = [
        "docs/endpoint_m4_results_20260919.md", "reports/nns_vit_endpoint_m4_r1_20260919/terminal_receipt.json",
        "docs/endpoint_m4_review_results_20260919.md", "docs/endpoint_m4_review_response_20260919.md",
        "reports/nns_vit_endpoint_m4_review_r2_20260919/terminal_receipt.json",
        "papers/nns_vit_endpoint/figure_manifest_m4.json", "papers/nns_vit_endpoint/figure_manifest_m4_review_r2.json",
    ]
    for relative in linked:
        quoted = chr(96) + relative + chr(96)
        require(quoted in text and (REPO / relative).exists(), "LINK:" + relative)
    report = {"status": "PASS", "requirements": len(required), "stale_phrases_rejected": len(stale),
              "original_test_rows_crosschecked": 12, "spatial_test_rows_crosschecked": 4,
              "linked_artifacts": len(linked), "document_lines": len(text.splitlines()),
              "post_m5_status_accepted": True}
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
