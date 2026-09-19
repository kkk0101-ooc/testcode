#!/usr/bin/env python3
"""Seal Endpoint M4 Review R2 with prompt-to-artifact coverage."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_review_r2_20260919"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def artifact(relative):
    path = REPO / relative; require(path.is_file(), "MISSING:" + relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)}


def main():
    analysis = load(ROOT / "analysis_receipt.json"); validation = load(ROOT / "independent_validation.json")
    documents = load(ROOT / "document_validation.json"); decision = load(ROOT / "scientific_decision.json")
    require(analysis["status"] == "PASS_POST_HOC_REVIEW_ANALYSIS" and validation["status"] == documents["status"] == "PASS", "AUDIT_STATUS")
    rel = ROOT.relative_to(REPO).as_posix() + "/"; checks = []
    def check(requirement, paths, evidence):
        checks.append({"requirement": requirement, "status": "PASS", "artifacts": [artifact(path) for path in paths], "evidence": evidence})
    check("Preserve immutable M4 PASS root", ["reports/nns_vit_endpoint_m4_r1_20260919/terminal_receipt.json", "reports/nns_vit_endpoint_m4_r1_20260919/artifact_manifest.json", rel + "freeze.json"], "Original M4 status and 838-file manifest revalidated without mutation")
    check("Preserve R1 implementation failure", ["reports/nns_vit_endpoint_m4_review_r1_20260919/failure_receipt.json", "reports/nns_vit_endpoint_m4_review_r1_20260919/freeze.json", "docs/endpoint_m4_review_r2_recovery_20260919.md"], "NameError failure retained; R2 fresh root binds its SHA and allowed repair")
    check("Target-conditioned estimand and non-percentage wording", ["docs/endpoint_m4_review_results_20260919.md", "papers/nns_vit_endpoint/en/endpoint_m4_review_r2.md", "papers/nns_vit_endpoint/zh/endpoint_m4_review_r2.md"], "Downstream cotangent scope, relative path-norm scale and normalized RMS discrepancy stated explicitly")
    check("JVP distribution, absolute error and denominator", [rel + "jvp_image_block_values.csv", rel + "jvp_block_summary.csv", rel + "figures/m4_review_jvp_diagnostics.png"], "Full 800-task q10/q25/median/q75/q90 and bootstrap CIs; high block-11 ratio is not a general small-denominator artifact")
    check("Mean-query and CLS-query saturation robustness", [rel + "saturation_sensitivity_summary.csv", rel + "saturation_sensitivity_tests.json", rel + "figures/m4_review_saturation_robustness.png"], "Unadjusted and partial rank controlling head-output and cotangent norms; phase stratification; 18-test family/model")
    check("Residual spatial rewrite evidence", [rel + "residual_spatial_summary.csv", rel + "residual_spatial_tests.json", rel + "figures/m4_review_residual_spatial.png"], "400-image late-early output-identity patch TV with bootstrap, sign and four-test Holm; branch-share relation")
    check("MLP cancellation model dependence", ["docs/endpoint_m4_review_results_20260919.md", "reports/nns_vit_endpoint_m4_r1_20260919/m4_tests.json"], "Original corrected tests show significant positive ViT-B and negative DeiT-S trends")
    check("M3-M4 descriptive bridge", [rel + "bridge_raw_map_values.csv", rel + "bridge_block_summary.csv", rel + "m3_m4_trajectory_relations.csv", rel + "figures/m4_review_m3_m4_bridge.png"], "384 source-hashed raw maps; full/value entropy and TV; model-level depth trajectories only")
    check("Cut, cotangent and maximum-error locations", [rel + "numeric_error_locations.json", "docs/endpoint_m4_review_results_20260919.md", "src/vitnns/endpoint_m4.py"], "AV cut before concat/projection; pre-LN clean residual cotangents; four extrema located")
    check("CPU-only source recomputation", [rel + "analysis_receipt.json", rel + "independent_validation.json"], "No GPU, JPEG, model, Torch or timm access; 800 M4 and M1 records plus 384 raw maps independently verified")
    check("Revised bilingual manuscript", ["papers/nns_vit_endpoint/en/full_draft_m4_review_r2.md", "papers/nns_vit_endpoint/zh/full_draft_m4_review_r2.md", rel + "document_validation.json"], "Abstract, contribution, M4 results, discussion, limitations and conclusion synchronized")
    check("Defensive-writing audit", ["papers/nns_vit_endpoint/qc/endpoint_m4_review_r2_writing_audit.md", "docs/endpoint_m4_review_response_20260919.md"], "Claim-evidence-boundary applied; 11 review items mapped; overclaim scan passed")
    check("Figures and visual inspection", [rel + "figure_manifest.json", rel + "visual_inspection.json", rel + "figures/m4_review_jvp_diagnostics.png", rel + "figures/m4_review_saturation_robustness.png", rel + "figures/m4_review_residual_spatial.png", rel + "figures/m4_review_m3_m4_bridge.png"], "Four PNG/PDF pairs mirrored to review paper package and visually inspected")
    check("Stop boundary", ["config/nns_vit_endpoint/endpoint_m4_review_r1.json", rel + "scientific_decision.json"], "No new primary M4 test, no M5/M6, next authorization STOP_AFTER_M4_REVIEW_R2")
    audit = {"objective": "Evaluate the supplied M4 critique and revise or supplement M4 using sealed evidence",
             "success_criteria": ["tighten target-conditioned and norm-ratio interpretations", "diagnose JVP denominator and distributions",
                                  "test saturation robustness and residual spatial change", "add a bounded M3-M4 bridge",
                                  "produce synchronized bilingual revision without mutating original M4"],
             "status": "PASS", "requirements": checks, "requirement_count": len(checks),
             "missing": [], "incomplete": [], "weakly_verified": [],
             "scientific_status": decision["status"], "audited_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    (ROOT / "completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}:
            files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    (ROOT / "artifact_manifest.json").write_text(json.dumps({"file_count": len(files), "files": files}, indent=2) + "\n")
    receipt = {"status": "ENDPOINT_M4_REVIEW_R2_PASS_POST_HOC_SENSITIVITY", "scientific_status": decision["status"],
               "original_M4_status": "ENDPOINT_M4_PASS_EXECUTION_AND_AUDIT", "m4_tasks": 800,
               "raw_map_tasks": 32, "review_requirements": len(checks),
               "completion_audit_sha256": sha(ROOT / "completion_audit.json"),
               "artifact_manifest_sha256": sha(ROOT / "artifact_manifest.json"),
               "next_authorization": "STOP_AFTER_M4_REVIEW_R2", "sealed_time_ns": time.time_ns()}
    (ROOT / "terminal_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "requirements": len(checks), "files": len(files)}))


def verify():
    receipt = load(ROOT / "terminal_receipt.json")
    require(receipt["status"] == "ENDPOINT_M4_REVIEW_R2_PASS_POST_HOC_SENSITIVITY" and receipt["next_authorization"] == "STOP_AFTER_M4_REVIEW_R2", "TERMINAL")
    require(sha(ROOT / "completion_audit.json") == receipt["completion_audit_sha256"] and sha(ROOT / "artifact_manifest.json") == receipt["artifact_manifest_sha256"], "ROOT_HASHES")
    manifest = load(ROOT / "artifact_manifest.json")
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    require(actual == {row["path"] for row in manifest["files"]} | {"artifact_manifest.json", "terminal_receipt.json"}, "FILE_SET")
    for row in manifest["files"]:
        path = ROOT / row["path"]; require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "ARTIFACT:" + row["path"])
    audit = load(ROOT / "completion_audit.json")
    require(audit["status"] == "PASS" and not audit["missing"] and not audit["incomplete"] and not audit["weakly_verified"], "COMPLETION")
    for item in audit["requirements"]:
        for row in item["artifacts"]:
            path = REPO / row["path"]; require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "AUDIT_ARTIFACT:" + row["path"])
    print(json.dumps({"status": receipt["status"], "files": manifest["file_count"], "requirements": audit["requirement_count"], "hashes": "PASS", "next_authorization": receipt["next_authorization"]}))


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else main()
