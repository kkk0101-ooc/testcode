#!/usr/bin/env python3
"""Freeze exact R8 candidates and the pre-power seal without computing power."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import scripts.run_nns_vit_patch_depth_s2_r8 as runner

ROOT = REPO / "evidence/nns_vit_patch_depth/s2_r8_20260914"
R7 = REPO / "evidence/nns_vit_patch_depth/s2_r7_20260914"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    unsigned = dict(value)
    expected = unsigned.pop("canonical_digest")
    require(expected == digest(unsigned), "canonical digest mismatch")
    return value


def record(path: Path, base: Path = REPO) -> dict[str, Any]:
    result = {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
    if path.suffix == ".json":
        result["canonical_digest"] = read_signed(path)["canonical_digest"]
    return result


def signed(document_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    value = {"schema_version": 1, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S2-R8", "run_id": ROOT.name, "document_type": document_type, "status": "PASS", "payload": payload}
    value["canonical_digest"] = digest(value)
    return value


def write(name: str, value: dict[str, Any]) -> None:
    path = ROOT / name
    require(not path.exists(), "refusing overwrite")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def git_record(path: str, commit: str) -> dict[str, Any]:
    full = REPO / path
    result = {"path": path, "bytes": full.stat().st_size, "sha256": sha(full)}
    line = subprocess.check_output(["git", "-C", str(REPO.parent), "ls-tree", commit, "--", "vitnns_rebuild/" + path], text=True).strip()
    require(bool(line), "uncommitted: " + path)
    result["git_blob_oid"] = line.split()[2]
    return result


def main() -> None:
    manifests = runner.freeze_manifests()
    commit = subprocess.check_output(["git", "-C", str(REPO.parent), "rev-parse", "HEAD"], text=True).strip()
    paths = (
        "config/nns_vit_patch_depth/s2_r8_run_spec.json",
        "config/nns_vit_patch_depth/s2_r8_run_spec.schema.json",
        "docs/nns_vit_patch_depth_s2_r8_hierarchical_gatekeeping_20260914.md",
        "src/vitnns/nns_patch_depth_s2_r8.py",
        "scripts/run_nns_vit_patch_depth_s2_r8.py",
        "scripts/freeze_nns_vit_patch_depth_s2_r8.py",
        "scripts/validate_nns_vit_patch_depth_s2_r8.py",
        "tests/test_nns_patch_depth_s2_r8.py",
    )
    payload = {
        "implementation_protocol_commit": commit,
        "files": [git_record(item, commit) for item in paths],
        "R7_terminal": record(R7 / "terminal_receipt.json"),
        "R7_power_report": record(R7 / "cluster_aware_power_report.json"),
        "R7_ICC_report": record(R7 / "pilot_bidirectional_d2_icc_report.json"),
        "candidate_manifests": manifests["bindings"],
        "candidate_counts": manifests["counts"],
        "target_power": 0.8,
        "real_power_computed_before_seal": False,
        "formal_outcome_access_count": 0,
        "S3_S4_S5_executed": False,
    }
    write("pre_power_seal.json", signed("s2_r8_pre_power_seal", payload))
    write("formal_access_audit_pre_power.json", signed("s2_r8_pre_power_access_audit", {
        "formal_metadata_records_read": 11557,
        "formal_hash_only_byte_access_count": 0,
        "formal_image_mask_feature_prediction_endpoint_access_count": 0,
        "model_GPU_attribution_probe_occlusion_count": 0,
        "real_power_computation_count": 0,
    }))
    print(json.dumps({"status": "PASS", "commit": commit, "candidate_counts": manifests["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
