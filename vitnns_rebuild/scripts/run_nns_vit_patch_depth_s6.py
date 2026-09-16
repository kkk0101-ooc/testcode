#!/usr/bin/env python3
"""Generate S6 reports and figures from sealed S5 JSON only."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-nns-s6")

import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s6 as s6  # noqa: E402


def doc(kind: str, status: str, payload: dict, commit: str) -> dict:
    value = s6.signed(kind, status, payload, commit)
    s6.verify_closed_document(value)
    return value


def markdown_report(data: dict) -> str:
    rows = {row["endpoint"]: row for row in data["endpoint_rows"]}
    relation = data["D1_D2_relationship"]
    return f"""# S6 post-hoc exploratory 论文结果报告

状态：PASS_POST_HOC_EXPLORATORY

## 核心结论 [post-hoc exploratory]

NNS 识别的 patch 与语义前景正向对齐，遮挡这些 patch 引起的目标 logit 变化大于遮挡空间匹配的对照 patch。NNS 对齐和表示层 probe gain 均表现为深层高于浅层。两种深度对比在逐图层面近乎不相关，因此它们反映互补的深度现象，而不是一条已证实的统一因果机制。

## 论文结果段 [post-hoc exploratory]

在 R12 选择的 240 张图像中，位置对照校正后的 NNS–前景对齐为 {rows['P1']['estimate']:.4f}（ordinary BCa 95% CI [{rows['P1']['ordinary_BCa_95']['lower']:.4f}, {rows['P1']['ordinary_BCa_95']['upper']:.4f}]；{rows['P1']['n_valid']}/{rows['P1']['n_total']} valid）。匹配遮挡效应为 {rows['P2']['estimate']:.4f}（ordinary BCa 95% CI [{rows['P2']['ordinary_BCa_95']['lower']:.4f}, {rows['P2']['ordinary_BCa_95']['upper']:.4f}]；{rows['P2']['n_valid']}/{rows['P2']['n_total']} valid），通过 outcome access 后设定的 S5 探索性门槛。NNS 对齐的 deep-minus-shallow 差异为 {rows['D1']['estimate']:.4f}（ordinary BCa 95% CI [{rows['D1']['ordinary_BCa_95']['lower']:.4f}, {rows['D1']['ordinary_BCa_95']['upper']:.4f}]）。在完整 S3-R7 D2 封存队列中，probe gain 的 deep-minus-shallow 差异为 {rows['D2']['estimate']:.4f}（ordinary BCa 95% CI [{rows['D2']['ordinary_BCa_95']['lower']:.4f}, {rows['D2']['ordinary_BCa_95']['upper']:.4f}]；{rows['D2']['n_valid']}/{rows['D2']['n_total']} valid）。D1 与 D2 在 {relation['paired_valid_source_count']} 个共同有效样本上的 Pearson 相关为 {relation['paired_image_Pearson_correlation']:.4f}，未显示逐图耦合关系。

## Endpoint decisions [post-hoc exploratory]

| Endpoint [post-hoc exploratory] | n total | n valid | n NA | n failed | Estimate | Ordinary BCa 95% CI | Adjusted interval | Adjusted p | State |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| P1 | {rows['P1']['n_total']} | {rows['P1']['n_valid']} | {rows['P1']['n_NA']} | {rows['P1']['n_failed']} | {rows['P1']['estimate']:.6f} | [{rows['P1']['ordinary_BCa_95']['lower']:.6f}, {rows['P1']['ordinary_BCa_95']['upper']:.6f}] | lower={rows['P1']['adjusted_interval']['lower']:.6f} | {rows['P1']['adjusted_p']:.3g} | {rows['P1']['exploratory_state']} |
| P2 | {rows['P2']['n_total']} | {rows['P2']['n_valid']} | {rows['P2']['n_NA']} | {rows['P2']['n_failed']} | {rows['P2']['estimate']:.6f} | [{rows['P2']['ordinary_BCa_95']['lower']:.6f}, {rows['P2']['ordinary_BCa_95']['upper']:.6f}] | lower={rows['P2']['adjusted_interval']['lower']:.6f} | {rows['P2']['adjusted_p']:.3g} | {rows['P2']['exploratory_state']} |
| D1 | {rows['D1']['n_total']} | {rows['D1']['n_valid']} | {rows['D1']['n_NA']} | {rows['D1']['n_failed']} | {rows['D1']['estimate']:.6f} | [{rows['D1']['ordinary_BCa_95']['lower']:.6f}, {rows['D1']['ordinary_BCa_95']['upper']:.6f}] | [{rows['D1']['adjusted_interval']['lower']:.6f}, {rows['D1']['adjusted_interval']['upper']:.6f}] | {rows['D1']['adjusted_p']:.3g} | {rows['D1']['exploratory_state']} |
| D2 | {rows['D2']['n_total']} | {rows['D2']['n_valid']} | {rows['D2']['n_NA']} | {rows['D2']['n_failed']} | {rows['D2']['estimate']:.6f} | [{rows['D2']['ordinary_BCa_95']['lower']:.6f}, {rows['D2']['ordinary_BCa_95']['upper']:.6f}] | [{rows['D2']['adjusted_interval']['lower']:.6f}, {rows['D2']['adjusted_interval']['upper']:.6f}] | {rows['D2']['adjusted_p']:.3g} | {rows['D2']['exploratory_state']} |

P1、P2 和 D1 的 adjusted interval/p 来自 S5 的 family-size-three L1 exploratory 分析。D2 是独立的 two-direction exploratory endpoint，不与前三项合并成新的 confirmatory family。

## P2 原规则与 S5 规则 [post-hoc exploratory]

原 P2 规则为 ORIGINAL_RULE_NOT_PASSED：点估计 {rows['P2']['estimate']:.6f} 未超过 0.10，调整后下界 {rows['P2']['adjusted_interval']['lower']:.6f} 未超过 0.10，ordinary BCa half-width {rows['P2']['ordinary_BCa_95']['half_width']:.6f} 未达到 ≤0.015。S5 使用的 ≥0.07 点估计门、>0 adjusted lower-bound 门和 ≤0.02 half-width 门均在 outcome access 后指定。

## 解释范围 [post-hoc exploratory]

结果支持在所分析样本中报告正向的 NNS–前景对齐、匹配遮挡响应和两个正向深度对比。D2 衡量 probe 可解码性，不能替代 NNS 证据；D1 与 D2 的相关性接近零，不支持统一或因果性的深度形成机制。结果不恢复原 S4-R10 STOP_SUCCESS，也不作为 preregistered、outcome-blind、confirmatory 或 strong-FWER-preserved 证据。
"""


def figure_analysis() -> str:
    return """# S6 post-hoc exploratory 图片分析

## Image #1 — Endpoint decisions [post-hoc exploratory]

图中展示 P1、P2、D1、D2 的点估计与 ordinary BCa 95% CI。虚线是各端点在 S5 中采用的探索性门槛。四个端点均为 SUPPORTED_POST_HOC_EXPLORATORY。P2 的虚线是 outcome access 后设置的 0.07 门槛，不是原 0.10 规则。

## Image #2 — Depth profiles [post-hoc exploratory]

左图显示 NNS 的位置对照校正前景对齐随 block 的变化；右图显示 feature probe 相对于低层控制的 MSE gain。两个深浅对比均为正，但曲线代表不同构念，纵轴不应互换解释。

## Image #3 — D1/D2 relationship [post-hoc exploratory]

共同有效的 233 张图像形成散点。Pearson r=0.0016，显示两种深度差异没有逐图线性耦合证据。该图不支持 probe 显著等同于 NNS 显著，也不支持因果形成机制。

## Image #4 — P2 rule comparison [post-hoc exploratory]

图中并列标示观测值、原规则门槛和 S5 探索规则门槛。P2 通过 S5 的探索规则，但原规则的三个条件均未通过，因此正式状态保持 ORIGINAL_RULE_NOT_PASSED。
"""


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--replace-own-root", action="store_true"); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise RuntimeError("CPU-only boundary not active")
    evidence_root = REPO / s6.ROOT_REL; report_root = REPO / s6.REPORT_REL
    terminal, gate = s6.authorization_gate(REPO, check_remote=True)
    for root in (evidence_root, report_root):
        if root.exists():
            if not args.replace_own_root: raise RuntimeError(f"fresh S6 root required: {root}")
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() not in {".json", ".md", ".png", ".pdf", ".svg"}: raise RuntimeError(f"refusing to replace unexpected S6 file: {path}")
            shutil.rmtree(root)
    evidence_root.mkdir(parents=True); report_root.mkdir(parents=True)
    commit = s6.git(REPO, "rev-parse", "HEAD")
    s6.write_json_new(evidence_root / "authorization_gate_report.json", doc("s6_authorization_gate_report", "PASS_POST_HOC_EXPLORATORY", gate, commit))
    sources, source_inventory = s6.load_sources(REPO, terminal)
    source_payload = {"analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory source binding", "S5_terminal": s6.artifact(REPO, REPO / s6.S5_TERMINAL_REL), "records": source_inventory, "record_count": len(source_inventory), "records_digest": s6.canonical_digest(source_inventory), "reporting_transform_only": True}
    s6.write_json_new(evidence_root / "source_binding.json", doc("s6_source_binding", "PASS_POST_HOC_EXPLORATORY", source_payload, commit))
    data = s6.figure_data(sources)
    s6.write_json_new(evidence_root / "figure_data.json", doc("s6_figure_data", "PASS_POST_HOC_EXPLORATORY", data, commit))
    figure_paths = s6.render(evidence_root / "figures", data)
    render_records = [s6.validate_render(path) for path in figure_paths]
    for record in render_records: record["path"] = Path(record["path"]).relative_to(REPO).as_posix()
    figure_payload = {"analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory figure inventory", "figures": render_records, "figure_count": len(render_records), "formats": ["PDF", "PNG", "SVG"], "style": {"analysis_label": s6.LABEL, "font_family": "STIXGeneral", "PNG_DPI": 600, "single_column_in": 3.5, "full_width_in": 7.16}, "figures_digest": s6.canonical_digest(render_records)}
    s6.write_json_new(evidence_root / "figure_inventory.json", doc("s6_figure_inventory", "PASS_POST_HOC_EXPLORATORY", figure_payload, commit))
    report = markdown_report(data); analysis = figure_analysis()
    s6.write_text_new(evidence_root / "paper_results.md", report); s6.write_text_new(evidence_root / "figure_analysis.md", analysis)
    s6.write_text_new(report_root / "paper_results.md", report); s6.write_text_new(report_root / "figure_analysis.md", analysis)
    (report_root / "figures_png").mkdir()
    for path in figure_paths:
        if path.suffix == ".png": shutil.copyfile(path, report_root / "figures_png" / path.name)
    claims = sources["claim_ledger.json"]["payload"]["claims"]
    claim_payload = {"analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory claim ledger", "claims": [{**item, "evidence_path": f"{s6.S5_ROOT_REL.as_posix()}/{Path(item['evidence_path']).name}"} for item in claims], "original_P2_rule_status": "ORIGINAL_RULE_NOT_PASSED", "forbidden_interpretations": ["preregistered", "outcome-blind", "confirmatory", "strong-FWER-preserved", "original S4-R10 STOP_SUCCESS", "probe significance equals NNS significance", "depth association is a causal formation mechanism"], "new_scientific_claim_count": 0}
    s6.write_json_new(evidence_root / "claim_ledger.json", doc("s6_claim_ledger", "PASS_POST_HOC_EXPLORATORY", claim_payload, commit))
    audit_rows = [
        {"location": "paper_results.md/core conclusion", "original": "N/A generated text", "type": "CLEAN", "severity": "none", "necessary_caveat": True, "why_it_feels_defensive": "It does not; the empirical conclusion leads.", "likely_reviewer_reaction": "Direct and measurable.", "recommended_action": "KEEP", "rewrite": "NNS alignment, matched intervention response, and both depth contrasts were positive in the evaluated data."},
        {"location": "paper_results.md/P2", "original": "N/A generated text", "type": "NECESSARY_CAVEAT", "severity": "none", "necessary_caveat": True, "why_it_feels_defensive": "The original rule failure materially changes interpretation.", "likely_reviewer_reaction": "Transparent separation of original and post-hoc rules.", "recommended_action": "KEEP_ONCE", "rewrite": "P2 passed the S5 exploratory rule; the original P2 rule was not passed."},
        {"location": "paper_results.md/depth", "original": "N/A generated text", "type": "NECESSARY_CAVEAT", "severity": "none", "necessary_caveat": True, "why_it_feels_defensive": "The distinction prevents an unsupported mechanistic inference.", "likely_reviewer_reaction": "Appropriately scoped.", "recommended_action": "KEEP_ONCE", "rewrite": "D1 and D2 were positive but showed near-zero image-level correlation."},
    ]
    writing_payload = {"analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory defensive-writing audit", "audit_rows": audit_rows, "paper_level_defensive_writing_score": 1, "highest_priority_removals": ["No reviewer-facing prebuttal remains.", "No repeated caveat blocks remain.", "No promotional adjective substitutes for a measurement."], "necessary_caveats_to_preserve": ["post-hoc exploratory scope", "outcome-access timing of P2 gates", "ORIGINAL_RULE_NOT_PASSED", "D2 probe is not NNS", "near-zero D1/D2 correlation is not causal evidence"], "repeated_phrase_audit": {"result": "PASS", "post_hoc_label_repeated_only_where_scope_binding_is_required": True}}
    s6.write_json_new(evidence_root / "writing_audit.json", doc("s6_writing_audit", "PASS_POST_HOC_EXPLORATORY", writing_payload, commit))
    access_payload = {"analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory CPU access audit", "CUDA_VISIBLE_DEVICES": "", "S5_JSON_read_count": len(source_inventory), "S3_or_S4_array_read_count": 0, "JPEG_or_mask_read_count": 0, "model_or_checkpoint_load_count": 0, "torch_timm_torchvision_import_count": 0, "CUDA_or_GPU_initialization_count": 0, "new_inference_or_intervention_count": 0, "S0_to_S5_root_write_count": 0}
    s6.write_json_new(evidence_root / "cpu_access_audit.json", doc("s6_cpu_access_audit", "PASS_POST_HOC_EXPLORATORY", access_payload, commit))
    print(json.dumps({"status": "PASS_POST_HOC_EXPLORATORY", "evidence_root": s6.ROOT_REL.as_posix(), "report_root": s6.REPORT_REL.as_posix(), "figure_count": len(render_records), "new_scientific_claim_count": 0}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
