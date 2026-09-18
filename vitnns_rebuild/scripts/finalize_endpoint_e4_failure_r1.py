#!/usr/bin/env python3
"""Seal the immutable ENDPOINT-E4 R1 local-contract failure."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[1]
GIT_ROOT = REPO.parent
PYTHON = GIT_ROOT / ".conda/envs/vitnns/bin/python"
ROOT = REPO / "reports/nns_vit_endpoint_e4_intervention_r1_20260918"
FAIL_STATUS = "ENDPOINT_E4_FAIL_LOCAL_DIRECTION_CONTRACT"
SOURCE_FILES = [
    "scripts/validate_endpoint_e4_failure.py",
    "scripts/finalize_endpoint_e4_failure_r1.py",
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_new(path: Path, value) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> int:
    if any((ROOT / name).exists() for name in ("validation_report.json", "artifact_manifest.json", "terminal_receipt.json")):
        raise RuntimeError("E4_FAILURE_TERMINAL_EXISTS")
    head = git("rev-parse", "HEAD"); branch = git("symbolic-ref", "--short", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    fields = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=REPO, text=True).split()
    remote = fields[0] if len(fields) == 2 else ""
    if branch != "stage-d-d3-pass" or not head == upstream == remote:
        raise RuntimeError("FAILURE_FINALIZER_NOT_PUBLISHED")
    for relative in SOURCE_FILES:
        subprocess.check_call(["git", "cat-file", "-e", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO)
    completed = subprocess.run(
        [str(PYTHON), "scripts/validate_endpoint_e4_failure.py", str(ROOT)], cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError("FAILURE_VALIDATION_FAILED:" + completed.stderr)
    independent = json.loads(completed.stdout)
    coverage = {
        "status": "FAIL", "stage": "ENDPOINT-E4",
        "failure_code": "LOCAL_DIRECTION_CONTRACT_FAILED",
        "failed_task_ids": independent["failed_task_ids"],
        "failed_task_count": independent["failed_task_count"],
        "failed_attempt_count": independent["failed_attempt_count"],
        "queue_state_counts": independent["queue_state_counts"],
        "completed_tasks_excluded_from_scientific_interpretation": independent["completed_tasks_excluded_from_scientific_interpretation"],
        "unfinished_tasks": independent["unfinished_tasks"],
    }
    write_new(ROOT / "formal_coverage_audit.json", coverage)
    report = {
        **independent, "producer_commit": load_startup()["producer_commit"],
        "failure_finalizer_commit": head, "r1_root_immutable": True,
        "formal_outcome_subsets_interpreted": False,
        "diagnosis": "all authorized local-contract tasks exhausted two attempts at the frozen finite-eta acceptance gate",
        "implementation_bindings": [{"path": path, "commit": head, "sha256": file_sha256(REPO / path)} for path in SOURCE_FILES],
    }
    write_new(ROOT / "validation_report.json", report)
    files = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in sorted(ROOT.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}]
    manifest = {"files": files, "canonical_digest": digest(files)}; write_new(ROOT / "artifact_manifest.json", manifest)
    terminal = {
        "status": FAIL_STATUS, "stage": "ENDPOINT-E4", "failed_gate": "EP5-A_LOCAL_DIRECTION_CONTRACT",
        "producer_commit": report["producer_commit"], "failure_finalizer_commit": head,
        "failed_task_count": 16, "failed_attempt_count": 32,
        "completed_tasks_excluded_from_scientific_interpretation": 73,
        "validation_digest": digest(report), "manifest_digest": manifest["canonical_digest"],
        "next_authorization": "NONE_PENDING_PROSPECTIVE_RECOVERY_AMENDMENT",
    }
    write_new(ROOT / "terminal_receipt.json", terminal); print(json.dumps(terminal, sort_keys=True)); return 0


def load_startup():
    return json.loads((ROOT / "startup_snapshot.json").read_text())


if __name__ == "__main__":
    raise SystemExit(main())
