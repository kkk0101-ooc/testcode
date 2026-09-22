#!/usr/bin/env python3
"""Create the N2-V2 I0 input-only preflight root without image or model access."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_i0_20260922"
LEDGER = ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json"
ATTEMPTS = ("n2_r1_20260922", "n2_r1b_20260922", "n2_r1c_20260922")
ATTEMPT_IDS = ("N2_R1_20260922", "N2_R1B_20260922", "N2_R1C_20260922")

LEGACY_BOUND_PATHS = (
    "执行规划.md", "研究目标.md", "evidence/n1_r3_20260922/terminal_receipt.json",
    "N2隔离Pilot协议.md", "N2-R1C最终生产器恢复修订.md", "config/n2_r1/pilot_contract.json",
    "evidence/n2_r1c_20260922/pair_manifest.json", "evidence/n2_r1c_20260922/pair_inputs.npz",
    "evidence/n2_r1c_20260922/blind_qc.json", "evidence/n2_r1c_20260922/validation.json",
    "N2完成审计.md", "evidence/n2_r1c_20260922/completion_audit.json",
    "schemas/n2_completion_audit.schema.json", "schemas/n2_terminal_receipt.schema.json",
    "evidence/n2_r1c_20260922/terminal_receipt.json",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(ROOT)).encode("utf-8")); digest.update(b"\0")
        digest.update(sha(item).encode("ascii")); digest.update(b"\n")
    return {"path": str(path.relative_to(ROOT)), "file_count": len(files), "tree_sha256": digest.hexdigest()}


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def build_ledger() -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    for attempt, expected_id in zip(ATTEMPTS, ATTEMPT_IDS):
        manifest = json.loads((ROOT / "evidence" / attempt / "pair_manifest.json").read_text())
        if manifest["attempt_id"] != expected_id:
            raise RuntimeError("LEGACY_ATTEMPT_ID_DRIFT")
        for row in manifest["records"]:
            for role in ("recipient", "base_donor", "semantic_donor", "within_stratum_donor"):
                source = row[role]
                key = source["source_group_id"]
                record = by_source.setdefault(key, {
                    "source_group_id": key, "image_sha256": source["image_sha256"],
                    "attempt_ids": set(), "roles": set(),
                    "v2_use_policy": "REGRESSION_ONLY_EXCLUDED_FROM_ALL_SCIENTIFIC_SPLITS",
                })
                if record["image_sha256"] != source["image_sha256"]:
                    raise RuntimeError("LEGACY_SOURCE_HASH_CONFLICT")
                record["attempt_ids"].add(expected_id); record["roles"].add(role)
    records = []
    for key in sorted(by_source):
        record = by_source[key]
        record["attempt_ids"] = sorted(record["attempt_ids"]); record["roles"] = sorted(record["roles"])
        records.append(record)
    qc = json.loads((ROOT / "evidence/n2_r1c_20260922/blind_qc.json").read_text())
    pair = json.loads((ROOT / "evidence/n2_r1c_20260922/pair_manifest.json").read_text())
    pair_by_id = {row["pair_id"]: row for row in pair["records"]}
    regression = []
    for review in qc["pair_reviews"]:
        if not review["artifact_free"]:
            regression.append({
                "attempt_id": "N2_R1C_20260922", "pair_id": review["pair_id"],
                "failure_codes": review["codes"], "arm_array_digests": pair_by_id[review["pair_id"]]["arm_array_digests"],
                "scientific_use_forbidden": True,
            })
    return {
        "schema_version": 1, "study_version": "N2_V2",
        "ledger_id": "N2_V2_LEGACY_LINEAGE_EXCLUSIONS_20260922", "source_attempts": list(ATTEMPT_IDS),
        "excluded_source_count": len(records), "records": records, "artifact_regression_cases": regression,
    }


def main() -> int:
    if OUT.exists() or LEDGER.exists():
        raise RuntimeError("N2_V2_I0_ROOT_OR_LEDGER_ALREADY_EXISTS")
    audit = json.loads(subprocess.check_output([sys.executable, str(ROOT / "audit_n2_completion.py")], text=True, cwd=ROOT))
    if audit != {"checklist_items": 30, "status": "PASS_N2_COMPLETION_AUDIT", "terminal_decision": "NO_GO_NOT_IDENTIFIABLE"}:
        raise RuntimeError("LEGACY_N2_AUDIT_NOT_CURRENT")
    terminal = json.loads((ROOT / "evidence/n2_r1c_20260922/terminal_receipt.json").read_text())
    if terminal["status"] != "NO_GO_NOT_IDENTIFIABLE":
        raise RuntimeError("LEGACY_TERMINAL_DRIFT")
    ledger = build_ledger()
    if ledger["excluded_source_count"] != 30 or len(ledger["artifact_regression_cases"]) != 7:
        raise RuntimeError("LEGACY_LEDGER_COUNT_DRIFT")
    write_new(LEDGER, ledger)
    zero_model_counts = {
        "checkpoint_deserializations": 0, "model_loads": 0, "inference_runs": 0, "behavior_outcome_reads": 0,
        "activation_extractions": 0, "endpoint_runs": 0, "path_nns_runs": 0, "jvp_runs": 0,
        "intervention_runs": 0, "gpu_uses": 0,
    }
    closure = {
        "schema_version": 1, "document_type": "n2_v2_legacy_closure_preflight", "attempt_id": "N2_V2_I0_20260922",
        "status": "PASS_LEGACY_N2_CLOSED", "terminal_status": terminal["status"], "legacy_audit_status": audit["status"],
        "bound_file_hashes": [{"path": path, "sha256": sha(ROOT / path)} for path in LEGACY_BOUND_PATHS],
        "immutable_root_snapshots": [tree_snapshot(ROOT / "evidence" / attempt) for attempt in ATTEMPTS],
        "model_access_counts": zero_model_counts,
    }
    asset = {
        "schema_version": 1, "document_type": "n2_v2_asset_preflight", "attempt_id": "N2_V2_I0_20260922",
        "status": "BLOCKED_INPUT_ASSETS", "required_asset_type": "NATURAL_SCENE_ONLY_NO_FOREGROUND_REMOVAL",
        "accepted_asset_registry_path": "manifests/n2_v2/accepted_scene_only_assets.json", "accepted_asset_registry_present": False,
        "accepted_aquatic_scene_count": 0, "accepted_non_aquatic_scene_count": 0, "minimum_per_context": 24,
        "category_names_are_not_scene_only_evidence": True, "old_lineage_overlap_count": 0,
        "new_image_payload_reads": 0, "new_mask_payload_reads": 0, "decision": "DO_NOT_GENERATE_PAIRS_OR_REVIEW_CARDS",
    }
    reviewer = {
        "schema_version": 1, "document_type": "n2_v2_reviewer_preflight", "attempt_id": "N2_V2_I0_20260922",
        "status": "BLOCKED_INDEPENDENT_REVIEWERS", "required_independent_reviewers": 2, "assigned_reviewer_count": 0,
        "sealed_review_bundle_count": 0, "semantic_agreement_estimated": False, "cohen_kappa_estimated": False,
        "discussion_before_seal_forbidden": True, "model_outcomes_visible": False,
        "decision": "DO_NOT_RUN_BLIND_REVIEW_OR_DOWNSTREAM_GATES",
    }
    write_new(OUT / "legacy_closure.json", closure); write_new(OUT / "asset_preflight.json", asset); write_new(OUT / "reviewer_preflight.json", reviewer)
    print(json.dumps({"status": "PASS_I0_PREFLIGHT_SEALED", "excluded_sources": len(ledger["records"]), "regression_cases": len(ledger["artifact_regression_cases"]), "model_accesses": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
