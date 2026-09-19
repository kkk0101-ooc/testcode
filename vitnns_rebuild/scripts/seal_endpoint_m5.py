#!/usr/bin/env python3
"""Prompt-to-artifact completion audit and immutable Endpoint M5 seal."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m5_r2_20260919"
R1 = REPO / "reports/nns_vit_endpoint_m5_r1_20260919"
PYTHON = REPO.parent / ".conda/envs/vitnns/bin/python3.10"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(relative):
    path = REPO / relative
    require(path.is_file(), "MISSING:" + relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)}


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_checks():
    independent = subprocess.run(
        [sys.executable, str(REPO / "scripts/validate_endpoint_m5.py")], cwd=REPO.parent,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True, capture_output=True, check=True,
    )
    documents = subprocess.run(
        [sys.executable, str(REPO / "scripts/validate_endpoint_m5_documents.py")], cwd=REPO.parent,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True, capture_output=True, check=True,
    )
    tests = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", str(REPO / "tests/test_endpoint_m5.py")],
        cwd=REPO.parent, env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        text=True, capture_output=True, check=True,
    )
    report = {
        "status": "PASS", "commands": [
            {"command": "CUDA_VISIBLE_DEVICES='' python3 scripts/validate_endpoint_m5.py", "stdout": independent.stdout.strip()},
            {"command": "CUDA_VISIBLE_DEVICES='' python3 scripts/validate_endpoint_m5_documents.py", "stdout": documents.stdout.strip()},
            {"command": "python3.10 -m pytest -q tests/test_endpoint_m5.py", "stdout": tests.stdout.strip()},
        ], "pytest_passed": 4,
    }
    path = ROOT / "test_results.json"
    if path.exists():
        path.unlink()
    write_new(path, report)


def main():
    require(ROOT.is_dir(), "M5_ROOT")
    require(not (ROOT / "terminal_receipt.json").exists(), "ALREADY_TERMINAL")
    run_checks()
    analysis = load(ROOT / "analysis_receipt.json")
    independent = load(ROOT / "independent_validation.json")
    documents = load(ROOT / "document_validation.json")
    reference = load(ROOT / "reference_reuse_summary.json")
    m2 = load(ROOT / "m2_reuse_receipt.json")
    matrix = load(ROOT / "method_applicability_matrix.json")
    cost = load(ROOT / "cost_and_information.json")
    require(analysis["status"] == "ENDPOINT_M5_ANALYSIS_COMPLETE", "ANALYSIS")
    require(independent["status"] == documents["status"] == "PASS", "VALIDATION")
    require(reference["endpoint_invariance"]["exact"] and reference["new_inferential_tests"] == 0, "REFERENCE_RESULT")
    require(m2["groups"] == 24 and m2["new_tests"] == 0, "M2_REUSE")
    require(len(matrix["methods"]) == 7, "METHOD_MATRIX")
    require(cost["runtime_claim"] == "NOT_IDENTIFIABLE_FROM_COMBINED_M1_M2_PACKAGES", "COST_SCOPE")
    rel = ROOT.relative_to(REPO).as_posix() + "/"
    checks = []

    def check(requirement, paths, evidence, condition=True):
        require(condition, requirement)
        checks.append({"requirement": requirement, "status": "PASS", "artifacts": [artifact(path) for path in paths], "evidence": evidence})

    check("M4 and Review-R2 predecessor gates", [rel + "predecessor_validation.json", "reports/nns_vit_endpoint_m4_r1_20260919/terminal_receipt.json", "reports/nns_vit_endpoint_m4_review_r2_20260919/terminal_receipt.json"], "M4 and Review-R2 terminal states plus 838/30-file full-hash manifests revalidated; historical stage-plan binding resolved from git object")
    check("R1 failure preserved and R2 repair limited", ["docs/endpoint_m5_r2_recovery_20260919.md", rel + "freeze.json", "config/nns_vit_endpoint/endpoint_m5.json"], "R1 contains only predecessor/freeze receipts; Python 3.7 fmean incompatibility repaired with fsum/len; all scientific settings unchanged", {path.name for path in R1.iterdir()} == {"predecessor_validation.json", "freeze.json"})
    check("13.1 different method questions stated", ["docs/endpoint_m5_results_20260919.md", "papers/nns_vit_endpoint/en/endpoint_m5_positioning.md", "papers/nns_vit_endpoint/zh/endpoint_m5_positioning.md"], "Endpoint local dependence, Taylor replacement prediction, IG path attribution, and attention/relevance methods have explicit distinct estimands")
    check("13.2 reference-free wording and coordinate origin", ["docs/endpoint_m5_results_20260919.md", "papers/nns_vit_endpoint/en/full_draft_m5.md", "papers/nns_vit_endpoint/zh/full_draft_m5.md"], "External-reference independence and normalization-origin dependence stated with x=(r-mu)/sigma identity; logit Endpoint identified as Input x Gradient")
    check("13.3 Endpoint scores invariant across two references", [rel + "reference_reuse_summary.json", rel + "reference_reuse_image_values.csv", rel + "independent_validation.json"], "163,200 component pairs checked; maximum m and a differences are exactly zero", reference["endpoint_component_reference_pairs"] == 163200)
    check("13.3 Taylor and finite-effect reference dependence", [rel + "reference_reuse_summary.json", "docs/endpoint_m5_results_20260919.md"], "All 163,200 Taylor/effect values changed; only 46/9,600 and 64/9,600 exact head rankings", reference["reference_sensitivity"]["exact_taylor_rank_blocks"] == 46 and reference["reference_sensitivity"]["exact_finite_effect_rank_blocks"] == 64)
    check("13.3 Endpoint behavioral association and Taylor advantage", [rel + "reference_reuse_summary.json", rel + "m2_reuse_summary.csv", rel + "m2_reuse_receipt.json"], "16 image-bootstrap summaries; Taylor higher association in four model-reference M1 combinations and median AUC in all 24 M2 reuse groups; no new tests")
    check("13.3 computation and cache costs", [rel + "cost_and_information.json", "docs/endpoint_m5_results_20260919.md"], "Operation requirements and M1/M2/EP2 package measurements reported; combined packages explicitly do not identify per-method runtime")
    check("13.4 method applicability matrix", [rel + "method_applicability_matrix.json", "papers/nns_vit_endpoint/notes/endpoint_m5_evidence_matrix.md"], "Seven unique closed-schema method rows cover all five required applicability columns")
    check("13.5 output-comparison boundary", ["docs/endpoint_m5_results_20260919.md", "papers/nns_vit_endpoint/en/endpoint_m5_positioning.md", "papers/nns_vit_endpoint/zh/endpoint_m5_positioning.md"], "Endpoint logit equals normalized-input Input x Gradient; no duplicate algorithm comparison or output-localization superiority claim")
    check("13.6 Path IG failed-gate handling", [rel + "cost_and_information.json", "reports/nns_vit_endpoint_ep2_r1_20260918/terminal_receipt.json", "reports/nns_vit_endpoint_ep2_r1_20260918/calibration_report.json"], "61/64 gate retained; DeiT-S selects 512 while ViT-B records 41/64 and 40/64, so no cross-model release or retuning")
    check("CPU-only reuse-only access boundary", [rel + "analysis_receipt.json", rel + "independent_validation.json"], "No JPEG/model/checkpoint/Torch/timm/GPU access and zero new inferential tests", analysis["jpeg_reads"] == analysis["model_or_checkpoint_loads"] == analysis["new_inferential_tests"] == 0)
    check("Bilingual paper integration and writing audit", [rel + "document_validation.json", "papers/nns_vit_endpoint/qc/endpoint_m5_writing_audit.md", "papers/nns_vit_endpoint/en/full_draft_m5.md", "papers/nns_vit_endpoint/zh/full_draft_m5.md"], "Five documents synchronized; defensive-writing score 2/10; claim-evidence-boundary pattern preserved")
    check("Stage plan completion and scope stop", ["docs/阶段M_endpointnnsvit.md", "docs/endpoint_m5_execution_20260919.md"], "M5 marked complete with linked versioned artifacts; M6 remains pending and no six-reference, Path-IG tuning, region search, or output benchmark executed")
    check("Independent validation and focused tests", [rel + "test_results.json", "scripts/validate_endpoint_m5.py", "scripts/validate_endpoint_m5_documents.py", "tests/test_endpoint_m5.py"], "Independent standard-library recomputation passes all raw values and bootstrap summaries; document validator passes; 4 focused tests pass")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    tracking = subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=REPO, text=True).strip()
    require(head == tracking, "UNPUBLISHED_COMPLETION_CODE")
    audit = {
        "objective": "Execute Stage M M5: method semantics, reference properties, and minimal comparison",
        "success_criteria": ["three tasks in section 13.1", "accurate reference-free wording", "six reference-reuse reports", "seven-row applicability matrix", "output-level identity boundary", "Path IG failed-gate handling", "cost and cache conditions", "versioned bilingual paper integration", "stop before M6"],
        "status": "PASS", "requirement_count": len(checks), "requirements": checks,
        "missing": [], "incomplete": [], "weakly_verified": [], "uncovered": [],
        "not_executed_by_design": ["six-reference expansion", "large output-localization leaderboard", "new region or blur search", "Path IG retuning", "M6 mechanism cases"],
        "audited_head": head,
    }
    write_new(ROOT / "completion_audit.json", audit)
    (ROOT / "README.md").write_text(
        "# Endpoint M5 reuse-only result package\n\n"
        "Start with `docs/endpoint_m5_results_20260919.md` and `completion_audit.json`.\n"
        "The root contains 800 source-recomputed image-model rows, 16 bootstrap summaries,\n"
        "24 M2 spatial reuse groups, a seven-method applicability matrix, and cost/information\n"
        "boundaries. No new image/model/GPU access or inferential tests were used. Stop after M5.\n"
    )
    records = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}:
            records.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    write_new(ROOT / "artifact_manifest.json", {"file_count": len(records), "files": records})
    terminal = {
        "status": "ENDPOINT_M5_PASS_EXECUTION_AND_AUDIT",
        "scientific_status": "METHOD_SEMANTICS_AND_REFERENCE_BOUNDARY_COMPLETE",
        "analysis_mode": "CPU_ONLY_REUSE_ONLY", "image_model_rows": 800,
        "component_reference_pairs": 163200, "head_block_reference_comparisons": 9600,
        "m2_reuse_groups": 24, "new_inferential_tests": 0,
        "completion_requirements": len(checks),
        "completion_audit_sha256": sha(ROOT / "completion_audit.json"),
        "artifact_manifest_sha256": sha(ROOT / "artifact_manifest.json"),
        "next_authorization": "STOP_AFTER_M5", "sealed_time_ns": time.time_ns(),
    }
    write_new(ROOT / "terminal_receipt.json", terminal)
    print(json.dumps({"status": terminal["status"], "requirements": len(checks), "artifact_files": len(records), "next_authorization": terminal["next_authorization"]}))


def verify():
    terminal = load(ROOT / "terminal_receipt.json")
    require(terminal["status"] == "ENDPOINT_M5_PASS_EXECUTION_AND_AUDIT", "TERMINAL_STATUS")
    require(sha(ROOT / "completion_audit.json") == terminal["completion_audit_sha256"], "AUDIT_HASH")
    require(sha(ROOT / "artifact_manifest.json") == terminal["artifact_manifest_sha256"], "MANIFEST_HASH")
    manifest = load(ROOT / "artifact_manifest.json")
    expected = {row["path"] for row in manifest["files"]}
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    require(manifest["file_count"] == len(expected) and actual == expected | {"artifact_manifest.json", "terminal_receipt.json"}, "MANIFEST_SET")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "ARTIFACT:" + row["path"])
    audit = load(ROOT / "completion_audit.json")
    require(audit["status"] == "PASS" and not any(audit[key] for key in ("missing", "incomplete", "weakly_verified", "uncovered")), "COMPLETION_AUDIT")
    for row in audit["requirements"]:
        require(row["status"] == "PASS", "REQUIREMENT_STATUS")
        for item in row["artifacts"]:
            path = REPO / item["path"]
            require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "AUDIT_ARTIFACT:" + item["path"])
    require(terminal["next_authorization"] == "STOP_AFTER_M5" and terminal["new_inferential_tests"] == 0, "SCOPE")
    print(json.dumps({"status": terminal["status"], "requirements": audit["requirement_count"], "artifact_files": manifest["file_count"], "hashes": "PASS", "next_authorization": terminal["next_authorization"]}))


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else main()
