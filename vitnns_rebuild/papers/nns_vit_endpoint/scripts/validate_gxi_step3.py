#!/usr/bin/env python3
"""Validate GxI positioning plan Step 3 against the active bilingual paper package."""

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "papers/nns_vit_endpoint"
ROOT = PAPER / "publication_r2_20260920"
PLAN = REPO / "Endpoint_GxI创新定位与后续五步计划.md"
REPORT = ROOT / "qc/gxi_step3_validation.json"
checks = []


def check(name, value, detail=""):
    checks.append({"name": name, "passed": bool(value), "detail": detail})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirror(text, lang):
    return (
        text.replace("](../figures/", "](../publication_r2_20260920/figures/")
        .replace("](../references.md)", "](../publication_r2_20260920/references.md)")
        .replace("](supplement.md)", f"](../publication_r2_20260920/{lang}/supplement.md)")
        .replace("](supplement_experiments_v1.md)", f"](../publication_r2_20260920/{lang}/supplement_experiments_v1.md)")
    )


def main():
    texts = {lang: (ROOT / lang / "manuscript.md").read_text() for lang in ("zh", "en")}
    plan = PLAN.read_text()
    manifest = json.loads((ROOT / "figure_manifest.json").read_text())
    binding = json.loads((PAPER / "figure_manifest.json").read_text())

    check("plan_step3_status", "第三步已完成方法章节和核心概念图重构" in plan and "第三步未启动" not in plan.splitlines()[2])
    check("formal_definition_bilingual", all(all(term in text for term in (
        r"e=(p_e,c_e,q_e,\nu_e)", r"\mathcal L_{e,i}(s;x)", r"J_{z_e,x}(x)^{\mathsf T}\bar z_e^{(s)}"
    )) for text in texts.values()))
    check("free_indices_and_current_point", "$i$ 与 $e$ 是自由索引" in texts["zh"] and "$i$ and $e$ are free indices" in texts["en"] and "same observed input $x$ and forward state" in texts["en"])
    check("consumer_use_identity", all(all(term in text for term in ("producer", "consumer", "input slot", "invocation", "fan-out")) for text in texts.values()))
    check("valid_cut_proposition", all(all(term in text for term in (r"\sum_{e\in C}\mathcal L_{e,i}(s;x)", r"x_i\frac{\partial s(x)}{\partial x_i}", "exactly one" if lang == "en" else "恰好穿过一个")) for lang, text in texts.items()))
    check("valid_cut_proof", "互不相交且穷尽" in texts["zh"] and "disjoint, exhaustive groups" in texts["en"] and "链式法则" in texts["zh"] and "chain rule" in texts["en"])
    cut_terms = (r"C_{\mathrm{QKV}}", r"C_{\mathrm{RV}}", r"C_{\mathrm{head}}", r"C_{\mathrm{res}}")
    check("cut_instances_bilingual", all(all(term in text for term in cut_terms) for text in texts.values()))
    check("degeneration_table_bilingual", all(all(term in text for term in ("no $i$", "no $e$", "no registered $s$", "no valid-cut contract")) for text in texts.values()))
    check("audit_table_bilingual", all(all(term in text for term in ("76,800 closure checks", "9,600 omit-head", "9,600 omit-routing", "9,600 duplicate-identity", "6,144 analytic JVP", "JVP/VJP", "nine-step central differences", "384 attention-detach", "230,400 E3")) for text in texts.values()))
    check("unsealed_mutation_boundary", all(all(term in text for term in ("wrong-slot", "merged fan-out", "detach mutation")) for text in texts.values()))
    claim_ids = ("NOV-ID-1", "NOV-JOINT-1", "NOV-ATTN-1", "NOV-RES-1", "NOV-AUD-1")
    check("claim_links_bilingual", all(all(term in text for term in claim_ids) for text in texts.values()))
    check("no_priority_overclaim", not re.search(r"for the first time|\bfirst (?:method|framework|approach|study|to)\b|首次|首创", texts["zh"] + texts["en"], re.I))
    check("bilingual_equations", re.findall(r"\$\$(.*?)\$\$", texts["zh"], re.S) == re.findall(r"\$\$(.*?)\$\$", texts["en"], re.S))
    check("bilingual_table_rows", [line for line in texts["zh"].splitlines() if line.startswith("|")] == [line for line in texts["en"].splitlines() if line.startswith("|")])
    check("bilingual_sections", re.findall(r"^#{2,3} (\d+(?:\.\d+)?)", texts["zh"], re.M) == re.findall(r"^#{2,3} (\d+(?:\.\d+)?)", texts["en"], re.M))
    check("active_mirrors", all((PAPER / lang / "full_draft.md").read_text() == mirror(texts[lang], lang) for lang in ("zh", "en")))

    check("figure_manifest_step3", manifest["revision"] == "PUBLICATION-R2-STEP3" and binding["source_revision"] == "PUBLICATION-R2-STEP3")
    expected_titles = {1: "From total Gradient × Input to auditable dual-indexed maps", 2: "Valid differential cuts partition live paths exactly once"}
    for number, title in expected_titles.items():
        row = next(item for item in manifest["figures"] if item["number"] == number)
        check(f"figure_title:{number}", row["title"] == title)
        for extension, record in row["files"].items():
            path = ROOT / record["path"]
            check(f"figure_hash:{number}:{extension}", path.is_file() and sha(path) == record["sha256"])
            if extension == "png":
                width, height = struct.unpack(">II", path.read_bytes()[16:24])
                check(f"figure_resolution:{number}", width >= 2500 and height >= 1000, f"{width}x{height}")
            elif extension == "svg":
                ET.parse(path)
    review = json.loads((ROOT / "qc/visual_review.json").read_text())
    check("visual_review_step3", review.get("status") == "PASS_VISUAL_REVIEW_STEP3" and review.get("step3_concept_figures") == [1, 2])
    check("writing_audit", (PAPER / "qc/gxi_step3_writing_audit.md").is_file())
    check("completion_audit", (ROOT / "qc/gxi_step3_completion_audit.md").is_file())

    errors = [row for row in checks if not row["passed"]]
    report = {
        "status": "PASS" if not errors else "FAIL",
        "check_count": len(checks),
        "errors": errors,
        "checks": checks,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
