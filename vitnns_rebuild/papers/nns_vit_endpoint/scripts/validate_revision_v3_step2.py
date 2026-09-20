#!/usr/bin/env python3
"""Validate revision-v3 Step-2 experiments, portable evidence, and bilingual report."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "papers/nns_vit_endpoint"
ROOT = PAPER / "publication_r2_20260920"
EVIDENCE = ROOT / "evidence/revision_v3_step2"
REPORT = ROOT / "qc/revision_v3_step2_validation.json"
RESULT = REPO / "reports/nns_vit_endpoint_revision_v3_step2_r2_20260921"
EXPECTED_SOURCE_SHA256 = "5e39a561b98454c9311e551bdafa28d17a11d1cc6ce6a56566d05b94a8eb097c"
checks = []


def check(name, value, detail=""):
    checks.append({"name": name, "passed": bool(value), "detail": detail})


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    source = REPO / "修改意见/修改意见v3.md"
    check("opinion_source_sha", sha(source) == EXPECTED_SOURCE_SHA256)
    acquisition = load(RESULT / "acquisition_receipt.json")
    analysis = load(RESULT / "analysis_receipt.json")
    validation = load(RESULT / "validation_report.json")
    check("production_complete", acquisition["status"] == "PASS" and acquisition["states"] == {"completed": 1000} and acquisition["exits"] == [0, 0, 0])
    check("analysis_counts", analysis == {
        "status": "PASS", "statistical_unit": "image",
        "amplitude_image_rows": 8000, "amplitude_summary_rows": 100,
        "qkv_image_rows": 28800, "qkv_summary_rows": 360, "qkv_contrast_rows": 30,
        "sampling_image_rows": 3000, "sampling_summary_rows": 60, "sampling_variation_rows": 12,
    })
    check("independent_validation", validation["status"] == "PASS" and validation["task_count"] == 1000 and validation["raw_qkv_recompute"]["records"] == 1152)

    expected_files = {
        "acquisition_receipt.json", "analysis_receipt.json", "validation_report.json",
        "multi_amplitude_summary.csv", "multi_amplitude_decision.json",
        "qkv_block_summary.csv", "qkv_depth_contrasts.csv",
        "independent_sampling_summary.csv", "independent_sampling_variation.json",
        "multi_amplitude.png", "qkv_population.png", "independent_sampling.png",
    }
    check("portable_evidence_file_set", {path.name for path in EVIDENCE.iterdir() if path.is_file()} == expected_files)
    source_map = {
        "acquisition_receipt.json": RESULT / "acquisition_receipt.json",
        "analysis_receipt.json": RESULT / "analysis_receipt.json",
        "validation_report.json": RESULT / "validation_report.json",
        "multi_amplitude_summary.csv": RESULT / "multi_amplitude_summary.csv",
        "multi_amplitude_decision.json": RESULT / "multi_amplitude_decision.json",
        "qkv_block_summary.csv": RESULT / "qkv_block_summary.csv",
        "qkv_depth_contrasts.csv": RESULT / "qkv_depth_contrasts.csv",
        "independent_sampling_summary.csv": RESULT / "independent_sampling_summary.csv",
        "independent_sampling_variation.json": RESULT / "independent_sampling_variation.json",
        "multi_amplitude.png": RESULT / "figures/multi_amplitude.png",
        "qkv_population.png": RESULT / "figures/qkv_population.png",
        "independent_sampling.png": RESULT / "figures/independent_sampling.png",
    }
    for name, source_path in source_map.items():
        check("portable_hash:" + name, sha(EVIDENCE / name) == sha(source_path))

    amplitude = rows(EVIDENCE / "multi_amplitude_summary.csv")
    check("amplitude_summary_scope", len(amplitude) == 100 and {float(row["alpha"]) for row in amplitude} == {0.05000000074505806, 0.10000000149011612, 0.25, 0.5, 1.0} and all(int(row["n_images"]) == 400 for row in amplitude))
    decision = load(EVIDENCE / "multi_amplitude_decision.json")
    check("amplitude_monotonic", decision["strong_boundary_supported"] and len(decision["families"]) == 4 and all(row["image_monotonic_count"] == row["image_count"] == 400 for row in decision["families"]))
    qkv = rows(EVIDENCE / "qkv_block_summary.csv")
    contrasts = rows(EVIDENCE / "qkv_depth_contrasts.csv")
    check("qkv_scope", len(qkv) == 360 and len(contrasts) == 30 and all(int(row["n_images"]) == 400 for row in qkv + contrasts))
    q_spatial = [row for row in contrasts if row["path"] == "Q" and row["metric"] in {"normalized_spatial_entropy", "normalized_pairwise_distance"}]
    check("q_spatial_not_cross_model_consistent", len(q_spatial) == 4 and all(row["cross_model_direction_consistent"] == "False" for row in q_spatial))
    sampling = rows(EVIDENCE / "independent_sampling_summary.csv")
    variation = load(EVIDENCE / "independent_sampling_variation.json")["rows"]
    check("sampling_scope", len(sampling) == 60 and all(int(row["n_images"]) == 100 for row in sampling) and len(variation) == 12)
    check("five_real_groups", all(row["group_count"] == 5 and row["direction_consistent_groups"] == 5 for row in variation))

    zh = (ROOT / "zh/revision_v3_step2_experiments.md").read_text()
    en = (ROOT / "en/revision_v3_step2_experiments.md").read_text()
    check("bilingual_table_rows", [line for line in zh.splitlines() if line.startswith("|")] == [line for line in en.splitlines() if line.startswith("|")])
    check("bilingual_equations", re.findall(r"\$\$(.*?)\$\$", zh, re.S) == re.findall(r"\$\$(.*?)\$\$", en, re.S))
    check("report_requirements", all(term in zh for term in ("1600/1600", "Q/K/V", "30/30", "五组", "95% bootstrap")) and all(term in en for term in ("1,600", "Q/K/V", "30", "five independent", "95% bootstrap")))
    check("claim_successor", all(term in (PAPER / "notes/revision_v3_step2_claims.md").read_text() for term in ("S2-C1", "S2-C2", "S2-C3")))
    check("writing_audit", (PAPER / "qc/revision_v3_step2_writing_audit.md").is_file())
    banned = re.compile(r"We emphasize that|It should be noted that|We do not claim|Taken together|需要强调的是|应当指出|为避免误解|我们不声称", re.I)
    check("no_defensive_stock_phrases", not banned.search(zh + en))
    for path in (ROOT / "zh/revision_v3_step2_experiments.md", ROOT / "en/revision_v3_step2_experiments.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not target.startswith(("http", "#")):
                check("portable_link:" + path.parent.name + ":" + target, (path.parent / target).resolve().is_file())

    errors = [row for row in checks if not row["passed"]]
    report = {
        "status": "PASS" if not errors else "FAIL", "check_count": len(checks),
        "source_sha256": sha(source), "errors": errors, "checks": checks,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
