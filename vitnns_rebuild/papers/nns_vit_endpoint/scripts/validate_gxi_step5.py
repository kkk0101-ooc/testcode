#!/usr/bin/env python3
"""CPU-only completion validator for GxI roadmap Step 5."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "papers/nns_vit_endpoint"
ROOT = PAPER / "publication_r2_20260920"
REPORT = ROOT / "qc/gxi_step5_validation.json"
CHECKS: list[dict] = []


def check(name: str, passed: bool, detail="") -> None:
    CHECKS.append({"name": name, "passed": bool(passed), "detail": str(detail)})


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirror(text: str, lang: str) -> str:
    return (text.replace("](../figures/", "](../publication_r2_20260920/figures/")
                .replace("](../references.md)", "](../publication_r2_20260920/references.md)")
                .replace("](supplement.md)", f"](../publication_r2_20260920/{lang}/supplement.md)")
                .replace("](supplement_experiments_v1.md)", f"](../publication_r2_20260920/{lang}/supplement_experiments_v1.md)"))


def table_numbers(lines: list[str]) -> list[list[str]]:
    number = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z])")
    return [number.findall(line) for line in lines if line.startswith("|")]


def validate_markdown(path: Path) -> None:
    text = path.read_text()
    label = str(path.relative_to(ROOT))
    check("markdown_fences:" + label, text.count("```") % 2 == 0)
    check("latex_blocks:" + label, text.count("$$") % 2 == 0)
    equations = re.findall(r"\$\$(.*?)\$\$", text, re.S)
    check("latex_braces:" + label, all(eq.count("{") == eq.count("}") for eq in equations))
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("http", "#")):
            continue
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        check("link:" + label + ":" + target, resolved.is_file() and (resolved == REPO.resolve() or REPO.resolve() in resolved.parents))


def main() -> int:
    texts = {lang: (ROOT / lang / "manuscript.md").read_text() for lang in ("zh", "en")}
    supplements = {lang: (ROOT / lang / "supplement.md").read_text() for lang in ("zh", "en")}
    experiments = {lang: (ROOT / lang / "supplement_experiments_v1.md").read_text() for lang in ("zh", "en")}
    title = {
        "zh": "# 面向视觉 Transformer 内部目标的双索引局部 Jacobian 扫描与结构审计",
        "en": "# Dual-Indexed Local Jacobian Scanning and Structural Auditing of Internal Targets in Vision Transformers",
    }
    for lang, text in texts.items():
        intro = text.split("## 2.", 1)[0]
        abstract = text.split("## 1.", 1)[0]
        check("step5_title:" + lang, text.startswith(title[lang]))
        abstract_terms = ((r"\mathcal L_{e,i}(s;x)", "实际使用边", "合法 differential cut") if lang == "zh" else (r"\mathcal L_{e,i}(s;x)", "concrete use edge", "valid differential cut"))
        check("step5_abstract_definition:" + lang, all(term in abstract for term in abstract_terms))
        contribution = ("目标条件化双索引扫描与结构审计" if lang == "zh" else "target-conditioned dual-indexed scanning and structural auditing")
        check("step5_contribution_two:" + lang, contribution in intro and all(term in intro for term in ("76,800", "28,800", "6,144")))
        check("step5_nov_joint_anchor:" + lang, text.count("NOV-JOINT-1") >= 4)
        check("step5_strong_neighbors:" + lang, all(term in text for term in ("Conductance", "DecompX", "LRP", "EAP", "Vi-CD", "Unpack")))
        check("step5_no_priority_claims:" + lang, not re.search(r"for the first time|\bfirst (?:method|framework|approach|study|to)\b|\bnovel (?:method|framework|approach|formula)\b|首次|首创|不同于既有工作", text, re.I))
        check("step5_no_defensive_phrases:" + lang, not re.search(r"We emphasize that|It should be noted that|We do not claim|To avoid misunderstanding|Taken together|需要强调的是|应当指出|为避免误解|我们不声称", text, re.I))
        adversarial_terms = (("总图给出空间扩展背景", "双索引图组揭示", "两类证据分别界定方法正确性和行为用途") if lang == "zh" else ("total maps establish background spatial broadening", "dual-indexed map sets expose", "these evidence types delimit correctness and behavioral use separately"))
        check("step5_adversarial_answers:" + lang, all(term in text for term in adversarial_terms))
        negative_terms = (("广泛但非普遍", "0.1136–0.1862", "相反方向") if lang == "zh" else ("broad but non-universal", "0.1136–0.1862", "opposite direction"))
        check("step5_negative_results_preserved:" + lang, all(term in text for term in negative_terms))
        scope_terms = (("不是独立训练种子", "不识别训练过程的因果效应") if lang == "zh" else ("not independent training seeds", "does not identify a causal effect of training"))
        check("step5_e5_scope:" + lang, all(term in text for term in scope_terms))
        check("step5_entry_mirror:" + lang, (PAPER / lang / "full_draft.md").read_text() == mirror(text, lang))

    check("step5_bilingual_sections", re.findall(r"^#{2,3} (\d+(?:\.\d+)?)", texts["zh"], re.M) == re.findall(r"^#{2,3} (\d+(?:\.\d+)?)", texts["en"], re.M))
    check("step5_bilingual_equations", re.findall(r"\$\$(.*?)\$\$", texts["zh"], re.S) == re.findall(r"\$\$(.*?)\$\$", texts["en"], re.S))
    check("step5_bilingual_tables", [line for line in texts["zh"].splitlines() if line.startswith("|")] == [line for line in texts["en"].splitlines() if line.startswith("|")])
    check("step5_bilingual_figures", re.findall(r"!\[[^]]*\]\(([^)]+)\)", texts["zh"]) == re.findall(r"!\[[^]]*\]\(([^)]+)\)", texts["en"]))
    for name, pair in (("supplement", supplements), ("supplement_experiments", experiments)):
        check("step5_" + name + "_headings", len(re.findall(r"^#{1,4} ", pair["zh"], re.M)) == len(re.findall(r"^#{1,4} ", pair["en"], re.M)))
        check("step5_" + name + "_figures", re.findall(r"!\[[^]]*\]\(([^)]+)\)", pair["zh"]) == re.findall(r"!\[[^]]*\]\(([^)]+)\)", pair["en"]))
        check("step5_" + name + "_table_numbers", table_numbers(pair["zh"].splitlines()) == table_numbers(pair["en"].splitlines()))
        check("step5_" + name + "_equations", re.findall(r"\$\$(.*?)\$\$", pair["zh"], re.S) == re.findall(r"\$\$(.*?)\$\$", pair["en"], re.S))

    refs = (ROOT / "references.md").read_text()
    citation_names = ("Dosovitskiy", "Dou", "Shrikumar", "Sundararajan", "Dhamdhere", "Srinivas", "Wróbel", "Modarressi", "Ferrando", "Achtibat", "Mehri", "Nishankar", "Buratti", "Chen", "Zhang", "Syed", "Kramár", "Żukowska", "Izadi", "Selvaraju", "Abnar", "Chefer")
    check("step5_citations_resolve", all(name in refs and all(name in texts[lang] for lang in ("zh", "en")) for name in citation_names) and "TO VERIFY" not in refs)

    required = {
        "claim_to_evidence": ROOT / "evidence/gxi_step5_claim_to_evidence.md",
        "novelty_conditions": ROOT / "evidence/gxi_step5_novelty_conditions.md",
        "adversarial_review": ROOT / "qc/gxi_step5_adversarial_review.md",
        "defensive_audit": ROOT / "qc/gxi_step5_defensive_writing_audit.md",
        "thesis_review": ROOT / "qc/gxi_step5_thesis_final_review.md",
        "mismatch_risk": ROOT / "qc/gxi_step5_evidence_mismatch_and_risk.md",
    }
    check("step5_deliverables_present", all(path.is_file() for path in required.values()))
    claim = required["claim_to_evidence"].read_text()
    check("step5_claim_evidence_fields", all(term in claim for term in ("Claim ID", "双语正文位置", "直接证据", "统计单位与范围", "边界或反证", "NOV-JOINT-1")))
    conditions = required["novelty_conditions"].read_text()
    check("step5_novelty_conditions", all(term in conditions for term in ("## 成立条件", "## 失效条件", "Conductance", "DecompX/LRP", "EAP/Vi-CD", "Unpack")))
    adversarial = required["adversarial_review"].read_text()
    check("step5_four_adversarial_checks", all(term in adversarial for term in ("是否只是 G×I 加 hooks", "是否只是已有组件拼装", "结构闭合是否被写成因果正确", "经验结果是否真的需要双索引")) and adversarial.count("已处理") >= 4)
    thesis = required["thesis_review"].read_text()
    dimensions = ("研究问题与科学价值", "研究设计与因果逻辑", "方法学与统计推断", "结果解释与外推边界", "学术写作与叙事结构", "改进路径")
    check("step5_six_thesis_dimensions", all(term in thesis for term in dimensions) and thesis.count("**问题。**") == 6 and thesis.count("**证据位置。**") == 6 and thesis.count("**影响。**") == 6 and thesis.count("**修改建议与判断。**") == 6)
    check("step5_thesis_verdict", "当前结论为 `可送审`" in thesis and "维持该结论的最小条件" in thesis)
    risk = required["mismatch_risk"].read_text()
    check("step5_mismatch_columns", all(term in risk for term in ("论文位置", "当前结论", "现有证据", "缺口", "允许表述")))
    check("step5_risk_levels", all(term in risk for term in ("### 阻断毕业", "### 重大修改", "### 一般修改", "### 修辞优化")) and "无未处理项" in risk)
    defensive = required["defensive_audit"].read_text()
    required_columns = ("Location", "Original", "Type", "Severity", "Necessary caveat?", "Likely reviewer reaction", "Paper-level defensive-writing score", "Necessary caveats to preserve", "Repeated phrase audit")
    check("step5_defensive_audit_format", all(term in defensive for term in required_columns) and "1/10" in defensive)
    plan = (REPO / "Endpoint_GxI创新定位与后续五步计划.md").read_text()
    check("step5_plan_status", "第五步已完成中英文同步重写与对抗式终审" in plan.splitlines()[2] and "## 十二、第五步执行记录：中英文同步重写与对抗式终审" in plan)
    step5_record = plan.split("## 十二、第五步执行记录：中英文同步重写与对抗式终审", 1)[1]
    check("step5_plan_prompt_artifact_audit", step5_record.split("### 12.6 第五步 prompt-to-artifact 验收表", 1)[1].split("### 12.7", 1)[0].count("| PASS |") == 14)

    step4 = load(ROOT / "qc/gxi_step4_validation.json")
    numeric = load(ROOT / "evidence/gxi_step4_numeric_recalculation.json")
    check("step5_numeric_recalculation", step4["status"] == numeric["status"] == "PASS" and not step4["errors"] and step4["e5_audit"]["maximum_raw_value_difference"] == 0 and numeric["new_experiments"] == 0)
    check("step5_four_checkpoint_gate", step4["e5_audit"]["decision"] == "SYNC_FOUR_CHECKPOINT_E5" and step4["e5_audit"]["image_checkpoint_tasks"] == 1600 and step4["e5_audit"]["recomputed_hypothesis_values"] == 6400)

    completion = ROOT / "qc/gxi_step5_completion_audit.md"
    check("step5_completion_audit_present", completion.is_file())
    if completion.is_file():
        completion_text = completion.read_text()
        check("step5_prompt_artifact_checklist", "## Prompt-to-artifact checklist" in completion_text and completion_text.count("| PASS |") >= 18)
        check("step5_completion_gates", all(term in completion_text for term in ("一句话回答", "G×I 加 hooks", "Conductance", "四 checkpoint", "六个固定维度", "阻断毕业", "重大修改")))

    markdown_files = [ROOT / lang / name for lang in ("zh", "en") for name in ("manuscript.md", "supplement.md", "supplement_experiments_v1.md")] + list(required.values()) + [ROOT / "README.md", ROOT / "references.md"]
    if completion.is_file():
        markdown_files.append(completion)
    for path in markdown_files:
        validate_markdown(path)

    errors = [row for row in CHECKS if not row["passed"]]
    report = {
        "status": "PASS" if not errors else "FAIL",
        "objective": "GxI roadmap step 5: bilingual rewrite and adversarial thesis review",
        "cpu_only": True,
        "new_experiments": 0,
        "check_count": len(CHECKS),
        "errors": errors,
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in [ROOT / "zh/manuscript.md", ROOT / "en/manuscript.md", ROOT / "references.md", ROOT / "evidence/gxi_step4_numeric_recalculation.json"]},
        "coverage": {"main_figures": 11, "main_tables": 10, "supplement_pages": 92, "e5_tasks": 1600, "e5_values_recomputed": 6400, "novelty_axis": "NOV-JOINT-1", "thesis_verdict": "可送审"},
        "checks": CHECKS,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "check_count", "errors", "coverage")}, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
