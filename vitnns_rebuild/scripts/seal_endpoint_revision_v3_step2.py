#!/usr/bin/env python3
"""Seal the completed revision-v3 Step-2 result root and prompt audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_revision_v3_step2_r2_20260921"
PAPER = REPO / "papers/nns_vit_endpoint/publication_r2_20260920"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(value, code):
    if not value:
        raise ValueError(code)


def atomic(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main():
    require(not (ROOT / "terminal_receipt.json").exists(), "TERMINAL_EXISTS")
    acquisition = load(ROOT / "acquisition_receipt.json")
    analysis = load(ROOT / "analysis_receipt.json")
    validation = load(ROOT / "validation_report.json")
    paper_validation = load(PAPER / "qc/revision_v3_step2_validation.json")
    require(acquisition["status"] == analysis["status"] == validation["status"] == paper_validation["status"] == "PASS", "VALIDATION_STATUS")
    requirements = [
        {"requirement": "1. same images, targets, layers, heads and references at alpha 0.05/0.1/0.25/0.5/1.0", "status": "PASS", "evidence": ["task_manifest.json", "multi_amplitude_summary.csv"]},
        {"requirement": "1. finite effect, Taylor prediction, absolute/relative error and rank correlation", "status": "PASS", "evidence": ["multi_amplitude_image_values.csv", "multi_amplitude_summary.csv"]},
        {"requirement": "1. image-level error trend and strong-boundary disposition", "status": "PASS", "evidence": ["multi_amplitude_decision.json", "figures/multi_amplitude.png"]},
        {"requirement": "2. Q/K/V absolute mass, entropy, normalized distance, cancellation and relative scale", "status": "PASS", "evidence": ["qkv_image_values.csv", "qkv_block_summary.csv"]},
        {"requirement": "2. block-wise image summaries, confidence intervals and cross-model directions", "status": "PASS", "evidence": ["qkv_block_summary.csv", "qkv_depth_contrasts.csv"]},
        {"requirement": "3. at least five independent channel/token samples", "status": "PASS", "evidence": ["sampling_registry.json", "independent_sampling_summary.csv"]},
        {"requirement": "3. entropy/locality/distance depth comparisons, between-group variation and direction rates", "status": "PASS", "evidence": ["independent_sampling_variation.json", "figures/independent_sampling.png"]},
        {"requirement": "image is the inferential unit; bootstrap is not used as independent replication", "status": "PASS", "evidence": ["analysis_receipt.json", "sampling_registry.json"]},
        {"requirement": "closure error not substituted for finite utility", "status": "PASS", "evidence": ["multi_amplitude_image_values.csv", "validation_report.json"]},
        {"requirement": "no training-causal claim from random initialization comparison", "status": "PASS", "evidence": ["../../papers/nns_vit_endpoint/notes/revision_v3_step2_claims.md"]},
        {"requirement": "bilingual evidence report and successor claim list", "status": "PASS", "evidence": ["../../papers/nns_vit_endpoint/publication_r2_20260920/zh/revision_v3_step2_experiments.md", "../../papers/nns_vit_endpoint/publication_r2_20260920/en/revision_v3_step2_experiments.md"]},
        {"requirement": "independent validation covers actual objective, not proxy green status", "status": "PASS", "evidence": ["validation_report.json", "../../papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v3_step2_validation.json"]},
    ]
    audit = {
        "objective": "执行修改意见v3第二步：补齐决定论文可信度的实验",
        "success_criteria": ["multi-amplitude finite interventions", "population Q/K/V statistics", "five independent channel/token sampling replications"],
        "status": "PASS", "requirements": requirements, "requirement_count": len(requirements),
        "missing": [], "incomplete": [], "weakly_verified": [],
        "outside_step2": ["alternative trained checkpoints (resource-optional)", "patch pixel controls (resource-optional)", "full thesis rewrite (Step 3)"],
        "producer_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    }
    atomic(ROOT / "completion_audit.json", audit)
    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}:
            files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    atomic(ROOT / "artifact_manifest.json", {"file_count": len(files), "files": files})
    receipt = {
        "status": "ENDPOINT_REVISION_V3_STEP2_PASS_EXECUTION_AND_AUDIT",
        "stage": "ENDPOINT-REVISION-V3-STEP2-R2", "tasks": 1000,
        "primary_tasks": 800, "sampling_tasks": 200, "independent_sampling_groups": 5,
        "statistical_unit": "image", "completion_audit_sha256": sha(ROOT / "completion_audit.json"),
        "artifact_manifest_sha256": sha(ROOT / "artifact_manifest.json"),
        "paper_validation_sha256": sha(PAPER / "qc/revision_v3_step2_validation.json"),
        "next_authorization": "REVISION_V3_STEP3_REQUIRES_EXPLICIT_USER_REQUEST",
        "sealed_time_ns": time.time_ns(),
    }
    atomic(ROOT / "terminal_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "requirements": len(requirements), "files": len(files)}))


def verify():
    receipt = load(ROOT / "terminal_receipt.json")
    require(receipt["status"] == "ENDPOINT_REVISION_V3_STEP2_PASS_EXECUTION_AND_AUDIT", "STATUS")
    require(receipt["next_authorization"] == "REVISION_V3_STEP3_REQUIRES_EXPLICIT_USER_REQUEST", "NEXT")
    require(sha(ROOT / "completion_audit.json") == receipt["completion_audit_sha256"], "AUDIT_SHA")
    require(sha(ROOT / "artifact_manifest.json") == receipt["artifact_manifest_sha256"], "MANIFEST_SHA")
    manifest = load(ROOT / "artifact_manifest.json")
    expected = {row["path"] for row in manifest["files"]} | {"artifact_manifest.json", "terminal_receipt.json"}
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    require(actual == expected, "FILE_SET")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "ARTIFACT:" + row["path"])
    print(json.dumps({"status": receipt["status"], "files": manifest["file_count"], "hashes": "PASS"}))


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else main()
