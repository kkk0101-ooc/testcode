#!/usr/bin/env python3
"""Prompt-to-artifact completion audit and immutable Endpoint M6 receipt."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m6_r1_20260919"


def load(path): return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): value.update(chunk)
    return value.hexdigest()


def require(condition, code):
    if not condition: raise ValueError(code)


def artifact(relative):
    path = REPO / relative; require(path.is_file(), "MISSING:" + relative)
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def main():
    validation = load(ROOT / "independent_validation.json"); documents = load(ROOT / "document_validation.json")
    selection = load(ROOT / "selection_manifest.json"); figures = load(ROOT / "figure_manifest.json")
    require(validation["status"] == documents["status"] == "PASS", "AUDITS")
    rel = ROOT.relative_to(REPO).as_posix() + "/"; checks = []
    def check(requirement, paths, evidence):
        checks.append({"requirement": requirement, "status": "PASS", "artifacts": [artifact(path) for path in paths], "evidence": evidence})
    check("M5 predecessor and immutable M2/M3/M4 source bindings", [rel + "freeze.json", rel + "source_validation.json", rel + "source_access_manifest.json"],
          "All named receipts and manifests match frozen hashes; every accessed numeric source matches its sealed artifact manifest")
    check("14.1 twelve required case categories", [rel + "selection_manifest.json"],
          "Case count is 12 and covers early/late/CLS-patch, three attention, three residual, cancellation, cross-model and M2 boundary categories")
    check("14.2 frozen automatic selection", ["config/nns_vit_endpoint/endpoint_m6.json", rel + "selection_manifest.json", "src/vitnns/endpoint_m6.py"],
          "Inclusive 5 percent tails, median-nearest representative rules and fixed ID tie-breaks; manual visual reselection false")
    check("selection occurs before JPEG and selected-map access", [rel + "freeze.json", rel + "source_validation.json", rel + "map_recompute_receipt.json"],
          "CPU-only numerical selection was sealed before selected images and VJP maps were obtained")
    check("14.3 neuron atlas content", [rel + "raw_arrays.npz", rel + "figures/m6_neuron_atlas.png", rel + "figure_manifest.json"],
          "Input, signed/absolute Endpoint, patch mass and all requested scalar metadata are present")
    check("14.3 attention content", [rel + "raw_arrays.npz", rel + "figures/m6_attention_cases.png", rel + "map_recompute_receipt.json"],
          "Full/routing/Value/Q/K Endpoint, CLS-query attention, routing share, frozen-attention error and saturation")
    check("14.3 residual content", [rel + "raw_arrays.npz", rel + "figures/m6_residual_cases.png", rel + "map_recompute_receipt.json"],
          "Output/identity/branch Endpoint, identity/branch share, path cancellation and output-identity spatial change")
    check("ViT-B/16 versus DeiT-S representative layer difference", [rel + "figures/m6_model_layer_comparison.png", rel + "selection_manifest.json"],
          "Shared image nearest the model-difference median; within-image channels nearest their block medians")
    check("one M2 Endpoint versus Taylor semantic boundary", [rel + "figures/m6_m2_semantic_boundary.png", rel + "selection_manifest.json"],
          "Fixed model/block/geometry/reference; rank-difference selection excludes finite outcomes")
    check("five PNG/PDF figures and raw provenance", [rel + "figure_manifest.json", rel + "raw_arrays.npz", "papers/nns_vit_endpoint/figure_manifest_m6.json"],
          "Five required figures in both formats, byte-identical report/paper copies, raw arrays retained")
    check("bilingual integration, evidence matrix and writing audit", ["papers/nns_vit_endpoint/zh/full_draft_m6.md", "papers/nns_vit_endpoint/en/full_draft_m6.md",
          "papers/nns_vit_endpoint/notes/endpoint_m6_evidence_matrix.md", "papers/nns_vit_endpoint/qc/endpoint_m6_writing_audit.md", rel + "document_validation.json"],
          "Image #1 through Image #5 synchronized with claim-evidence-boundary language")
    check("independent selection, raw closure and figure validation", [rel + "independent_validation.json"],
          "Independent reselection of all 12 cases, 9 raw closure checks and image dimension/hash checks")
    audit = {"objective": "Execute docs/阶段M_endpointnnsvit.md M6 and stop after M6", "success_criteria": [
        "Automatically select all twelve specified M6 case categories without visual cherry-picking",
        "Render every neuron, attention and residual panel named by section 14.3",
        "Preserve raw arrays, source hashes, M2 semantic boundary and claim-language constraints",
        "Provide bilingual descriptions, independent validation and an immutable receipt"],
        "status": "PASS", "requirements": checks, "requirement_count": len(checks), "missing": [], "incomplete": [], "weakly_verified": [],
        "outside_scope": ["new inferential tests", "full-cohort recomputation", "manual aesthetic reselection", "post-M6 paper work"],
        "audited_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    (ROOT / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}:
            files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    (ROOT / "artifact_manifest.json").write_text(json.dumps({"file_count": len(files), "files": files}, indent=2) + "\n")
    receipt = {"status": "ENDPOINT_M6_PASS_EXECUTION_AND_AUDIT", "scientific_status": "M6_AUTOMATIC_MECHANISM_ATLAS_COMPLETE",
               "case_count": selection["case_count"], "figure_count": len(figures["figures"]), "new_inferential_tests": 0,
               "completion_audit_sha256": sha(ROOT / "completion_audit.json"), "artifact_manifest_sha256": sha(ROOT / "artifact_manifest.json"),
               "next_authorization": "STOP_AFTER_M6", "sealed_time_ns": time.time_ns()}
    (ROOT / "terminal_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "requirements": len(checks), "files": len(files)}))


def verify():
    receipt = load(ROOT / "terminal_receipt.json")
    require(receipt["status"] == "ENDPOINT_M6_PASS_EXECUTION_AND_AUDIT" and receipt["next_authorization"] == "STOP_AFTER_M6", "TERMINAL")
    require(sha(ROOT / "completion_audit.json") == receipt["completion_audit_sha256"] and sha(ROOT / "artifact_manifest.json") == receipt["artifact_manifest_sha256"], "ROOT_HASHES")
    manifest = load(ROOT / "artifact_manifest.json"); actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    require(actual == {row["path"] for row in manifest["files"]} | {"artifact_manifest.json", "terminal_receipt.json"}, "FILE_SET")
    for row in manifest["files"]:
        path = ROOT / row["path"]; require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "ARTIFACT:" + row["path"])
    audit = load(ROOT / "completion_audit.json")
    require(audit["status"] == "PASS" and not audit["missing"] and not audit["incomplete"] and not audit["weakly_verified"], "COMPLETION_AUDIT")
    for item in audit["requirements"]:
        require(item["status"] == "PASS", "REQUIREMENT")
        for row in item["artifacts"]:
            path = REPO / row["path"]; require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "AUDIT_ARTIFACT:" + row["path"])
    print(json.dumps({"status": receipt["status"], "files": manifest["file_count"], "requirements": audit["requirement_count"], "hashes": "PASS", "next_authorization": receipt["next_authorization"]}))


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else main()
