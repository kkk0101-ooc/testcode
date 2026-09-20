#!/usr/bin/env python3
"""Validate every deliverable in revision-v3 step 1 against current artifacts."""

import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "papers/nns_vit_endpoint"
ROOT = PAPER / "publication_r2_20260920"
REPORT = ROOT / "qc/revision_v3_step1_validation.json"
EXPECTED_SOURCE_SHA256 = "5e39a561b98454c9311e551bdafa28d17a11d1cc6ce6a56566d05b94a8eb097c"
checks = []


def check(name, passed, detail=""):
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirror(text, lang):
    return (
        text.replace("](../figures/", "](../publication_r2_20260920/figures/")
        .replace("](../references.md)", "](../publication_r2_20260920/references.md)")
        .replace("](supplement.md)", f"](../publication_r2_20260920/{lang}/supplement.md)")
        .replace(
            "](supplement_experiments_v1.md)",
            f"](../publication_r2_20260920/{lang}/supplement_experiments_v1.md)",
        )
    )


def main():
    source = REPO / "修改意见/修改意见v3.md"
    zh = (ROOT / "zh/manuscript.md").read_text()
    en = (ROOT / "en/manuscript.md").read_text()
    refs = (ROOT / "references.md").read_text()
    claims = (PAPER / "notes/revision_v3_frozen_claims.md").read_text()

    check(
        "revised_bilingual_titles",
        zh.startswith("# 视觉 Transformer 计算图上的 NNS 结构路径分解与系统验证")
        and en.startswith("# NNS Structural Path Decomposition and Systematic Validation on Vision Transformer Computational Graphs"),
    )
    check(
        "abstract_positioning",
        all(term in zh.split("## 1.")[0] for term in ("结构化实现", "路径分解", "系统验证", "不具有一般实现不变性")),
    )
    check("research_questions_1_to_4", all(f"RQ{i}" in zh and f"RQ{i}" in en for i in range(1, 5)) and "RQ5" not in zh + en)
    check(
        "rq4_matches_executed_experiments",
        "局部路径图能否在规定的有限替换任务中排序成分和空间区域" in zh
        and "Can local path maps rank components and spatial regions in the prescribed finite-replacement tasks" in en
        and "0.05,0.1,0.25,0.5,1.0" not in zh + en,
    )
    check(
        "contributions_map_prior_delta_evidence",
        zh.count("已有") >= 3 and zh.count("本文新增") >= 2 and zh.count("直接证据") >= 3,
    )
    methods = (
        "Gradient × Input", "conductance", "FullGrad", "GlobEnc", "DecompX",
        "AttnLRP", "LibraGrad", "Activation patching", "Edge attribution patching",
    )
    check("all_requested_method_comparisons", all(term in zh and term in en for term in methods))
    dimensions = ("Primary object", "External reference", "Internal-path decomposition", "Completeness / closure", "Finite-intervention evaluation")
    check("method_table_five_dimensions", all(term in zh and term in en for term in dimensions))
    check(
        "graph_cut_parameterization_counterexample",
        all(term in zh.split("### 4.5")[0] and term in en.split("### 4.5")[0] for term in (r"R^{-\top}", "E'_1=", "E'_2=")),
    )
    check(
        "chain_rule_not_primary_innovation",
        "不列为主要创新" in zh and "not listed as primary innovations" in en,
    )
    check(
        "parameter_state_not_training_causality",
        "该对照不识别训练过程的因果效应" in zh
        and "does not identify a causal effect of training" in en,
    )
    check(
        "frozen_claim_inventory",
        all(term in claims for term in (
            "K1", "K2", "K3", "C1 数学身份", "C2 结构闭合", "C3 分解非唯一性",
            "C4 层深规律", "C5 参数状态", "C6 成分排序", "C7 区域排序", "RQ4 冻结合同",
        )),
    )
    check("expanded_references", all(term in refs for term in ("Srinivas, S.", "Modarressi, A.", "Syed, A.", "Zhang, F.")))

    fidelity = load(REPO / "reports/nns_vit_endpoint_m3_r1_20260919/m3a_fidelity.json")
    e3 = load(REPO / "reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json")
    e4 = load(REPO / "reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json")
    family = load(REPO / "reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json")
    training = load(ROOT / "evidence/supplement_v1/training_analysis.json")
    m2 = load(ROOT / "evidence/supplement_v1/m2_inherited.json")
    check("fidelity_evidence", fidelity["status"] == "PASS" and fidelity["comparison_count"] == 5000 and fidelity["failure_count"] == 0)
    check(
        "closure_evidence",
        e3["independent_validation"]["positive_closure_evaluations"] == 76800
        and e3["independent_validation"]["negative_control_evaluations"] == 28800,
    )
    check("local_direction_evidence", e4["local_component_evaluation_count"] == 6144 and "PASS" in e4["status"])
    c4a = [row for row in family["families"] if row["claim"] == "C4a"]
    check("broad_nonuniversal_evidence", len(c4a) == 12 and sum(row["status"] == "SUPPORTED" for row in c4a) == 11 and family["aggregate_claims"]["C4a"] == "NOT_SUPPORTED")
    check("regional_ranking_evidence", m2["taylor_greater_counts"] == {"full": 11, "isolated": 12} and family["aggregate_claims"]["C4b"] == "NOT_SUPPORTED")
    check(
        "parameter_state_evidence",
        len(training["comparisons"]) == 102
        and all(row["n_seeds"] == 3 and row["n_images"] == 32 for row in training["comparisons"] if row["metric"] not in ("saturation_routing", "saturation_error")),
    )

    check("bilingual_display_equations", re.findall(r"\$\$(.*?)\$\$", zh, re.S) == re.findall(r"\$\$(.*?)\$\$", en, re.S))
    check("bilingual_table_rows", [line for line in zh.splitlines() if line.startswith("|")] == [line for line in en.splitlines() if line.startswith("|")])
    check("bilingual_section_numbers", re.findall(r"^#{2,3} (\d+(?:\.\d+)?)", zh, re.M) == re.findall(r"^#{2,3} (\d+(?:\.\d+)?)", en, re.M))
    check("active_mirror_zh", (PAPER / "zh/full_draft.md").read_text() == mirror(zh, "zh"))
    check("active_mirror_en", (PAPER / "en/full_draft.md").read_text() == mirror(en, "en"))
    check("defensive_writing_audit", (PAPER / "qc/revision_v3_step1_writing_audit.md").is_file())
    check("prompt_to_artifact_completion_audit", (ROOT / "qc/revision_v3_step1_completion_audit.md").is_file())
    banned = re.compile(r"We emphasize that|It should be noted that|We do not claim|To avoid misunderstanding|Taken together|需要强调的是|应当指出|为避免误解|我们不声称", re.I)
    check("no_defensive_stock_phrases", not banned.search(zh + en))

    errors = [row for row in checks if not row["passed"]]
    report = {
        "status": "PASS" if not errors else "FAIL",
        "source_sha256_expected": EXPECTED_SOURCE_SHA256,
        "source_sha256_observed": sha256(source) if source.is_file() else None,
        "check_count": len(checks),
        "errors": errors,
        "checks": checks,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
