#!/usr/bin/env python3
"""Validate the revision-v3 Step-3 bilingual blind-review thesis package."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parents[1]
ZH = PACKAGE / "zh" / "manuscript.md"
EN = PACKAGE / "en" / "manuscript.md"
REFS = PACKAGE / "references.md"
OPINION = ROOT / "修改意见" / "修改意见v3.md"
SOURCE = ROOT / "papers" / "nns_vit_endpoint" / "publication_r2_20260920"
OUT = PACKAGE / "qc" / "validation_report.json"

EXPECTED_OPINION_SHA256 = "5e39a561b98454c9311e551bdafa28d17a11d1cc6ce6a56566d05b94a8eb097c"
FORMULAS = [f"3-{i}" for i in range(1, 6)] + [f"4-{i}" for i in range(1, 16)] + [f"5-{i}" for i in range(1, 4)]
TABLES = ["2-1", "3-1", "5-1", "5-2", "6-1", "6-2", "6-3", "7-1", "7-2", "7-3"]


class Audit:
    def __init__(self) -> None:
        self.checks = []
        self.errors = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            self.errors.append(f"{name}: {detail}")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chapters(text: str, prefix: str) -> list[int]:
    if prefix == "zh":
        return [int(x) for x in re.findall(r"^# 第 ([1-8]) 章 ", text, re.M)]
    return [int(x) for x in re.findall(r"^# Chapter ([1-8]) ", text, re.M)]


def chapter_text(text: str, number: int, prefix: str) -> str:
    marker = f"# 第 {number} 章 " if prefix == "zh" else f"# Chapter {number} "
    start = text.index(marker)
    next_marker = f"# 第 {number + 1} 章 " if prefix == "zh" else f"# Chapter {number + 1} "
    end = text.find(next_marker, start) if number < 8 else text.find("# 参考文献", start) if prefix == "zh" else text.find("# References", start)
    return text[start:] if end < 0 else text[start:end]


def table_numeric_signature(text: str, table_id: str) -> list[str]:
    header = re.search(rf"^\*\*(?:表|Table) {re.escape(table_id)}[^\n]*\*\*$", text, re.M)
    if not header:
        return []
    block = text[header.end():]
    next_heading = re.search(r"^#{1,2} |^\*\*(?:表|Table) ", block, re.M)
    if next_heading:
        block = block[:next_heading.start()]
    rows = [line for line in block.splitlines() if line.startswith("|")]
    data = rows[2:] if len(rows) >= 2 else []
    return ["|".join(re.findall(r"(?<![A-Za-z])[-−]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", row, re.I)).replace("−", "-") for row in data]


def cited_numbers(text: str) -> set[int]:
    found = set()
    for body in re.findall(r"\[([0-9, -]+)\]", text):
        for part in body.split(","):
            part = part.strip()
            if "-" in part:
                a, b = map(int, part.split("-", 1))
                found.update(range(a, b + 1))
            elif part:
                found.add(int(part))
    return found


def csv_value(path: Path, where: dict[str, str], field: str) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if all(row[k] == v for k, v in where.items()):
                return float(row[field])
    raise KeyError((path, where, field))


def main() -> int:
    audit = Audit()
    required = [
        ZH,
        EN,
        REFS,
        PACKAGE / "README.md",
        PACKAGE / "notes" / "claim_evidence_matrix.md",
        PACKAGE / "qc" / "defensive_writing_audit.md",
        PACKAGE / "qc" / "page_review.md",
        PACKAGE / "qc" / "page_audit.json",
        PACKAGE / "qc" / "page_contact_sheet.png",
        PACKAGE / "NNS_ViT_学位论文_匿名内容审阅版.pdf",
        PACKAGE / "qc" / "completion_audit.md",
    ]
    audit.check("required_files", all(p.is_file() for p in required), ", ".join(str(p) for p in required if not p.is_file()))
    zh, en, refs = read(ZH), read(EN), read(REFS)

    observed_sha = sha256(OPINION)
    audit.check("opinion_source_sha256", observed_sha == EXPECTED_OPINION_SHA256, observed_sha)
    audit.check("exact_eight_chapters_zh", chapters(zh, "zh") == list(range(1, 9)), str(chapters(zh, "zh")))
    audit.check("exact_eight_chapters_en", chapters(en, "en") == list(range(1, 9)), str(chapters(en, "en")))

    for n in (3, 4, 6, 7):
        zc, ec = chapter_text(zh, n, "zh"), chapter_text(en, n, "en")
        for key, zword, eword in (("question", "本章问题", "Chapter Question"), ("method", "方法", "Method"), ("results", "结果", "Results"), ("summary", "本章小结", "Chapter Summary")):
            audit.check(f"chapter_{n}_{key}_bilingual", zword in zc and eword in ec)

    for number in FORMULAS:
        audit.check(f"formula_{number}_numbered_bilingual", f"式（{number}）" in zh and f"Equation ({number})" in en)
        audit.check(f"formula_{number}_referenced_bilingual", zh.count(number) >= 2 and en.count(number) >= 2, f"zh={zh.count(number)}, en={en.count(number)}")

    for number in TABLES:
        zs, es = table_numeric_signature(zh, number), table_numeric_signature(en, number)
        audit.check(f"table_{number}_present", bool(zs) and bool(es), f"zh rows={len(zs)}, en rows={len(es)}")
        audit.check(f"table_{number}_numeric_parity", zs == es, f"zh={zs}, en={es}" if zs != es else "")

    zh_refs = re.findall(r"^\[(\d+)\] ", zh, re.M)
    en_refs = re.findall(r"^\[(\d+)\] ", en, re.M)
    standalone_refs = re.findall(r"^\[(\d+)\] ", refs, re.M)
    expected_refs = [str(i) for i in range(1, 38)]
    audit.check("references_embedded_zh", zh_refs == expected_refs, str(zh_refs))
    audit.check("references_embedded_en", en_refs == expected_refs, str(en_refs))
    audit.check("reference_source_complete", standalone_refs == expected_refs, str(standalone_refs))
    audit.check("all_references_cited_zh", cited_numbers(zh.split("# 参考文献", 1)[0]) == set(range(1, 38)), str(sorted(cited_numbers(zh.split("# 参考文献", 1)[0]))))
    audit.check("all_references_cited_en", cited_numbers(en.split("# References", 1)[0]) == set(range(1, 38)), str(sorted(cited_numbers(en.split("# References", 1)[0]))))

    links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", zh) + re.findall(r"!\[[^]]*\]\(([^)]+)\)", en)
    missing_links = []
    for language_dir, text in ((ZH.parent, zh), (EN.parent, en)):
        for rel in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
            if not (language_dir / rel).resolve().is_file():
                missing_links.append(rel)
    audit.check("all_figure_links_resolve", not missing_links and len(links) == 14, f"count={len(links)}, missing={missing_links}")

    forbidden_identity = re.compile(r"\[(?:AUTHOR|AFFILIATION|CORRESPONDING AUTHOR|REPOSITORY AND ARCHIVE) REQUIRED\]|/data01|yeziqi|ByteDance|字节", re.I)
    audit.check("blind_review_anonymity", not forbidden_identity.search(zh + en), "identity or repository marker found")
    forbidden_zh_terms = re.compile(r"\b(?:cohort|reference|Met|Unmet|Figure|Table|head|routing|value|absolute mass|differential|checkpoint)\b", re.I)
    zh_body = zh.split("# 参考文献", 1)[0]
    zh_prose = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|!\[[^]]*\]\([^)]+\)", "", zh_body, flags=re.S)
    audit.check("formal_chinese_terminology", not forbidden_zh_terms.search(zh_prose), str(forbidden_zh_terms.findall(zh_prose)))
    audit.check("no_slash_joined_model_numbers", not re.search(r"\d\s*/\s*\d", zh_body), "slash-joined number found")

    defensive_en = re.compile(r"we emphasize|it should be noted|importantly,? we do not claim|to avoid misunderstanding|one might argue|a potential concern|should not be interpreted|we make no claim|encouraging|promising|compelling|taken together|together, these results|these results further validate", re.I)
    defensive_zh = re.compile(r"需要强调的是|值得注意的是|为避免误解|可能有人认为|令人鼓舞|毋庸置疑|必须指出")
    audit.check("defensive_stock_phrases_absent", not defensive_en.search(en) and not defensive_zh.search(zh))

    audit.check("analysis_types_separated", all(x in zh for x in ("确认性分析", "补充分析", "事后分析")) and all(x in en for x in ("Confirmatory analysis", "Supplementary analysis", "Post-hoc analysis")))
    audit.check("unexecuted_experiments_only_in_limitations_zh", zh.count("未执行") == 1 and "未执行" in chapter_text(zh, 8, "zh"), str(zh.count("未执行")))
    audit.check("unexecuted_experiments_only_in_limitations_en", en.count("were not executed") == 1 and "were not executed" in chapter_text(en, 8, "en"), str(en.count("were not executed")))

    claim_sentinels = ["广泛但并非普遍", "1 600", "0.6095–0.6856", "五组独立", "依赖图、切面和参数化"]
    audit.check("zh_abstract_body_conclusion_claims", all(x in zh for x in claim_sentinels), str([x for x in claim_sentinels if x not in zh]))
    en_sentinels = ["broad but non-universal", "1,600", "0.6095–0.6856", "five independent", "depend on the graph, cut, and parameterization"]
    audit.check("en_abstract_body_conclusion_claims", all(x.lower() in en.lower() for x in en_sentinels), str([x for x in en_sentinels if x.lower() not in en.lower()]))

    amp = SOURCE / "evidence" / "revision_v3_step2" / "multi_amplitude_summary.csv"
    qkv = SOURCE / "evidence" / "revision_v3_step2" / "qkv_depth_contrasts.csv"
    sampling = SOURCE / "evidence" / "revision_v3_step2" / "independent_sampling_summary.csv"
    decisions = json.loads((SOURCE / "evidence" / "revision_v3_step2" / "multi_amplitude_decision.json").read_text())
    audit.check("source_multi_amplitude_monotonic", decisions["strong_boundary_supported"] is True and sum(x["image_monotonic_count"] for x in decisions["families"]) == 1600)
    source_checks = [
        (csv_value(amp, {"model_id": "vit_b16", "reference_id": "normalized_zero", "alpha": "1.0", "metric": "mean_within_block_spearman"}, "median"), "0.6742"),
        (csv_value(amp, {"model_id": "deit_s16", "reference_id": "blur_rgb", "alpha": "0.05000000074505806", "metric": "median_relative_error"}, "median"), "0.0241"),
        (csv_value(qkv, {"model_id": "vit_b16", "path": "Q", "metric": "absolute_mass", "contrast": "block11_minus_block0"}, "median_difference"), "208.8022"),
        (csv_value(sampling, {"model_id": "deit_s16", "group_id": "sample_5", "source_metric": "input_dependency_distance", "contrast": "late_minus_early"}, "median"), "0.1320"),
    ]
    for idx, (value, rendered) in enumerate(source_checks, 1):
        audit.check(f"source_numeric_binding_{idx}", rendered in zh and rendered in en, f"source={value}, rendered={rendered}")

    old_validation = json.loads((SOURCE / "qc" / "revision_v3_step2_validation.json").read_text())
    audit.check("step2_predecessor_pass", old_validation.get("status") == "PASS" and old_validation.get("check_count") == 47)
    page_audit = json.loads((PACKAGE / "qc" / "page_audit.json").read_text())
    audit.check("page_by_page_pdf_audit", page_audit.get("status") == "PASS" and page_audit.get("page_count") == 29 and page_audit.get("embedded_image_count") == 25, str({k: page_audit.get(k) for k in ("status", "page_count", "embedded_image_count", "errors")}))
    audit.check("claim_evidence_matrix_complete", read(PACKAGE / "notes" / "claim_evidence_matrix.md").count("| ") >= 50)
    audit.check("prompt_to_artifact_completion_audit", "第三步的所有显式工作和最终验收标准" in read(PACKAGE / "qc" / "completion_audit.md"))
    audit.check("manuscript_substantive_length", len(zh) >= 30000 and len(en) >= 58000, f"zh={len(zh)}, en={len(en)}")

    result = {
        "status": "PASS" if not audit.errors else "FAIL",
        "objective": "revision v3 Step 3 bilingual blind-review thesis rewrite",
        "source_opinion_sha256": observed_sha,
        "check_count": len(audit.checks),
        "errors": audit.errors,
        "checks": audit.checks,
        "coverage": [
            "eight-chapter bilingual structure",
            "problem-method-results-summary structure in Chapters 3, 4, 6, and 7",
            "numbered and cross-referenced formulas",
            "bilingual table numeric parity",
            "embedded and cited bibliography",
            "formal Chinese terminology and slash-number removal",
            "confirmatory/supplementary/post-hoc separation",
            "blind-review anonymity",
            "figure link resolution",
            "selected raw-source numeric recomputation",
            "defensive-writing stock-phrase scan",
            "claim-evidence matrix and predecessor validation",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "check_count": result["check_count"], "errors": result["errors"]}, ensure_ascii=False, indent=2))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
