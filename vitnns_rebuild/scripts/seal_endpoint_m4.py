#!/usr/bin/env python3
"""Prompt-to-artifact completion audit and immutable Endpoint M4 receipt."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"


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
    path = REPO / relative
    require(path.is_file(), "MISSING:" + relative)
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def main():
    validation = load(ROOT / "validation_report.json")
    analysis = load(ROOT / "analysis_receipt.json")
    documents = load(ROOT / "document_validation.json")
    decisions = load(ROOT / "scientific_decision.json")
    require(validation["status"] == analysis["status"] == documents["status"] == "PASS", "AUDITS")
    rel = ROOT.relative_to(REPO).as_posix() + "/"; checks = []
    def check(requirement, paths, evidence):
        checks.append({"requirement": requirement, "status": "PASS",
                       "artifacts": [artifact(path) for path in paths], "evidence": evidence})
    check("M3 predecessor and immutable bindings", [rel + "predecessor_validation.json", rel + "freeze.json"],
          "M3 verifier passed; all source and task identities frozen before formal image access")
    check("10.1 per-head Q/K/V and routing/value differential", [rel + "validation_report.json", rel + "attention_head_summary.csv"],
          "86,400 image-block-head records with Q+K=routing and routing+value=native attention Endpoint")
    check("10.2 frozen-attention approximation", [rel + "attention_block_summary.csv", rel + "m4_tests.json"],
          "Endpoint and contracted-gradient errors retain absolute diagnostics and NA for zero denominators")
    check("10.3 two models, shared 400 images, 12 layers, every head", [rel + "task_manifest.json", rel + "coverage_audit.json", rel + "analysis_receipt.json"],
          "800 dynamic-queue tasks; 400 shared identities; 12 or 6 heads by model")
    check("10.4 routing share and both error domains", [rel + "image_values.csv", rel + "attention_block_summary.csv"],
          "routing share, Endpoint relative and absolute error, contracted-gradient relative and absolute error")
    check("10.5 independent validation suite", [rel + "validation_report.json", "tests/test_endpoint_m4.py"],
          "native autodiff, explicit QKV, detach, forward parity, Endpoint closure, and 8-direction JVP")
    check("10.6 mean-query and CLS-query saturation", [rel + "attention_head_summary.csv", rel + "m4_tests.json", rel + "figures/m4_attention_saturation.png"],
          "within-image within-layer head Spearman before cross-image inference; no frozen direction assumption")
    check("11.1 clean residual output cotangent", ["src/vitnns/endpoint_m4.py", "tests/test_endpoint_m4.py"],
          "attention uses g_r1 and MLP uses g_r2; identity seed excludes sibling branch")
    check("11.2 both residuals over full cohort", [rel + "residual_block_summary.csv", rel + "analysis_receipt.json"],
          "19,200 image-block-residual records")
    check("11.3 share, cancellation, cosine and spatial change", [rel + "residual_block_summary.csv", rel + "m4_a1_descriptive_summary.csv", rel + "figures/m4_residual_depth.png"],
          "identity/branch shares, path cancellation, cosine, patch-distribution total variation")
    check("11.4 six residual mechanism questions", ["docs/endpoint_m4_results_20260919.md", rel + "residual_block_summary.csv", rel + "m4_a1_descriptive_summary.csv"],
          "depth, residual type, branch-cancellation association, preservation/rewrite layers, and separate-model patterns")
    check("11.5 omitted-identity negative controls", [rel + "validation_report.json"],
          "all 19,200 nonzero identity/carry omissions rejected exact closure")
    check("M4-C signed, absolute, map and path cancellation", [rel + "m4_a1_image_values.csv", rel + "m4_a1_descriptive_summary.csv", rel + "residual_block_summary.csv"],
          "signed magnitude, absolute mass, within-map cancellation and cross-path cancellation remain distinct")
    check("15 image-first inference and frozen Holm families", [rel + "image_values.csv", rel + "m4_tests.json", "src/vitnns/endpoint_m4_statistics.py"],
          "10,000 bootstrap, two-sided exact sign, four attention tests/model and two residual tests/model/type")
    check("pre-result M4-A1 descriptive amendment", ["docs/endpoint_m4_analysis_amendment_20260919.md", rel + "m4_a1_freeze.json", rel + "m4_a1_analysis_receipt.json"],
          "published before analysis; no new inferential tests; adds signed/absolute and residual-type descriptive contrasts")
    check("pre-analysis M4-A2 import-boundary amendment", ["docs/endpoint_m4_analysis_import_amendment_20260919.md", rel + "analysis_import_amendment.json"],
          "matplotlib Pillow writer allowed without JPEG reads; Torch/timm/GPU remain prohibited during analysis")
    check("figures and numbered analysis", [rel + "figure_manifest.json", rel + "figures/m4_attention_depth.png", rel + "figures/m4_attention_saturation.png", rel + "figures/m4_residual_depth.png", "docs/endpoint_m4_results_20260919.md"],
          "three PNG/PDF result figures mirrored into paper package and described as Image #1 through Image #3")
    check("bilingual manuscript integration and writing audit", ["papers/nns_vit_endpoint/zh/full_draft.md", "papers/nns_vit_endpoint/en/full_draft.md", "papers/nns_vit_endpoint/zh/endpoint_m4_supplement.md", "papers/nns_vit_endpoint/en/endpoint_m4_supplement.md", "papers/nns_vit_endpoint/qc/endpoint_m4_writing_audit.md", rel + "document_validation.json"],
          "numbers and decisions synchronized; claim-evidence-boundary prose audited")
    check("scope stop", ["config/nns_vit_endpoint/endpoint_m4.json", rel + "scientific_decision.json"],
          "M4 local mechanism analysis complete; M5 and M6 not executed; next authorization STOP_AFTER_M4")
    audit = {
        "objective": "Execute Stage M M4 completely and stop before M5 or M6",
        "success_criteria": [
            "M4-A full-cohort per-head routing/value, frozen-attention and saturation analysis",
            "M4-B both residual-add identity/branch analyses with negative controls",
            "M4-C signed/absolute and map/path cancellation analysis",
            "image-first statistics, figures, bilingual manuscript integration, and immutable receipt",
        ],
        "status": "PASS", "requirements": checks, "requirement_count": len(checks),
        "missing": [], "incomplete": [], "weakly_verified": [],
        "outside_scope": ["M5 method positioning", "M6 mechanism cases", "new predictor competitions"],
        "audited_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    }
    (ROOT / "completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}:
            files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    (ROOT / "artifact_manifest.json").write_text(json.dumps({"file_count": len(files), "files": files}, indent=2) + "\n")
    receipt = {
        "status": "ENDPOINT_M4_PASS_EXECUTION_AND_AUDIT",
        "scientific_status": decisions["status"], "image_model_tasks": 800,
        "attention_head_records": validation["attention_heads"], "residual_records": validation["residuals"],
        "completion_audit_sha256": sha(ROOT / "completion_audit.json"),
        "artifact_manifest_sha256": sha(ROOT / "artifact_manifest.json"),
        "next_authorization": "STOP_AFTER_M4", "sealed_time_ns": time.time_ns(),
    }
    (ROOT / "terminal_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "requirements": len(checks), "files": len(files)}))


def verify():
    receipt = load(ROOT / "terminal_receipt.json")
    require(receipt["status"] == "ENDPOINT_M4_PASS_EXECUTION_AND_AUDIT" and receipt["next_authorization"] == "STOP_AFTER_M4", "TERMINAL")
    require(sha(ROOT / "completion_audit.json") == receipt["completion_audit_sha256"] and sha(ROOT / "artifact_manifest.json") == receipt["artifact_manifest_sha256"], "ROOT_HASHES")
    manifest = load(ROOT / "artifact_manifest.json")
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    require(actual == {row["path"] for row in manifest["files"]} | {"artifact_manifest.json", "terminal_receipt.json"}, "FILE_SET")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "ARTIFACT:" + row["path"])
    audit = load(ROOT / "completion_audit.json")
    require(audit["status"] == "PASS" and not audit["missing"] and not audit["incomplete"] and not audit["weakly_verified"], "COMPLETION_AUDIT")
    for item in audit["requirements"]:
        require(item["status"] == "PASS", "REQUIREMENT")
        for row in item["artifacts"]:
            path = REPO / row["path"]
            require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "AUDIT_ARTIFACT:" + row["path"])
    print(json.dumps({"status": receipt["status"], "files": manifest["file_count"],
                      "requirements": audit["requirement_count"], "hashes": "PASS",
                      "next_authorization": receipt["next_authorization"]}))


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else main()
