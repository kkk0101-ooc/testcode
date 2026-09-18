#!/usr/bin/env python3
"""Seal the verifier-only ENDPOINT-E3 R2 handoff without GPU reruns."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
REPO = Path(__file__).resolve().parents[1]
GIT_ROOT = REPO.parent
PYTHON = GIT_ROOT / ".conda/envs/vitnns/bin/python"
SOURCE = REPO / "reports/nns_vit_endpoint_e3_closure_r1_20260918"
ROOT = REPO / "reports/nns_vit_endpoint_e3_closure_r2_20260918"
AMENDMENT = REPO / "config/nns_vit_endpoint/endpoint_e3_closure_handoff_recovery_r2.json"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import digest, file_sha256, write_json_new  # noqa: E402
from vitnns.endpoint_e3_closure import PASS_STATUS  # noqa: E402

SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_e3_closure_handoff_recovery_r2.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE3封存恢复R2_20260918.md",
    "scripts/validate_endpoint_e3_closure.py",
    "scripts/finalize_endpoint_e3_closure_r2.py",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> int:
    if ROOT.exists():
        raise RuntimeError("E3_R2_ROOT_EXISTS")
    amendment = json.loads(AMENDMENT.read_text())
    for key, name in {
        "r1_terminal_receipt_sha256": "terminal_receipt.json",
        "r1_validation_report_sha256": "validation_report.json",
        "r1_artifact_manifest_sha256": "artifact_manifest.json",
    }.items():
        if file_sha256(SOURCE / name) != amendment["source_bindings"][key]:
            raise RuntimeError("R1_SOURCE_BINDING:" + key)
    head = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    fields = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=REPO, text=True).split()
    remote = fields[0] if len(fields) == 2 else ""
    if branch != "stage-d-d3-pass" or not head == upstream == remote:
        raise RuntimeError("RECOVERY_NOT_PUBLISHED")
    for relative in SOURCE_FILES:
        subprocess.check_call(["git", "cat-file", "-e", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO)
    completed = subprocess.run(
        [str(PYTHON), "scripts/validate_endpoint_e3_closure.py", str(SOURCE)],
        cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError("R2_INDEPENDENT_VALIDATION_FAILED:" + completed.stderr)
    independent = json.loads(completed.stdout.strip())
    if independent["status"] != PASS_STATUS:
        raise RuntimeError("R2_STATUS")
    ROOT.mkdir(parents=True)
    report = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E3-R2-HANDOFF",
        "claim_scope": "CORRECTNESS_ONLY_NOT_UTILITY",
        "source_r1_immutable": True, "gpu_rerun": False,
        "producer_commit": json.loads((SOURCE / "terminal_receipt.json").read_text())["producer_commit"],
        "recovery_commit": head, "amendment_sha256": file_sha256(AMENDMENT),
        "source_bindings": amendment["source_bindings"],
        "independent_validation": independent,
        "verifier_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "scientific_contract_changed": False,
        "next_authorization": "ENDPOINT_E4_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    write_json_new(ROOT / "validation_report.json", report)
    files = [{
        "path": "validation_report.json",
        "bytes": (ROOT / "validation_report.json").stat().st_size,
        "sha256": file_sha256(ROOT / "validation_report.json"),
    }]
    manifest = {"files": files, "canonical_digest": digest(files)}
    write_json_new(ROOT / "artifact_manifest.json", manifest)
    terminal = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E3-R2-HANDOFF",
        "claim_scope": "CORRECTNESS_ONLY_NOT_UTILITY",
        "source_r1_terminal_sha256": amendment["source_bindings"]["r1_terminal_receipt_sha256"],
        "validation_digest": digest(report), "manifest_digest": manifest["canonical_digest"],
        "producer_commit": report["producer_commit"], "recovery_commit": head,
        "gpu_rerun": False, "next_authorization": "ENDPOINT_E4_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    write_json_new(ROOT / "terminal_receipt.json", terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
