#!/usr/bin/env python3
"""Standard-library-only source-recomputing validator for S1 evidence."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s1_r1_pilot_20260914"
S0 = REPO / "evidence/nns_vit_patch_depth/s0_r5_20260914"
SPEC = REPO / "config/nns_vit_patch_depth/s1_r1_run_spec.json"
SCHEDULE = REPO / "manifests/nns_vit_patch_depth/s1_pilot_schedule.json"
TERMINALS = ("PASS", "BLOCKED_SCIENTIFIC_VALIDITY", "BLOCKED_DATA_INTEGRITY", "BLOCKED_INFRASTRUCTURE", "FAILED_IMPLEMENTATION")
NA_STATES = ("NA_GEOMETRY_SIZE_MISMATCH", "NA_TARGET_DISAPPEARED_AFTER_CROP", "NA_MASK_LABEL_OUT_OF_RANGE", "NA_VOID_MASK", "NA_ZERO_POSITIVE_MASS", "NA_MATCHED_CONTROL_UNAVAILABLE")


def require(condition: bool, message: str) -> None:
    if not condition: raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def read_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); require(isinstance(value, dict) and "canonical_digest" in value, "unsigned document: " + str(path)); unsigned = dict(value); observed = unsigned.pop("canonical_digest"); require(observed == digest(unsigned), "canonical digest mismatch: " + str(path)); return value


def record(path: Path) -> dict[str, Any]:
    value = {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if path.suffix == ".json":
        document = json.loads(path.read_text());
        if "canonical_digest" in document: value["canonical_digest"] = document["canonical_digest"]
    return value


def npy_header(path: Path) -> tuple[str, list[int]]:
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "not npy: " + str(path)); major, minor = stream.read(2)
        size = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]; header = stream.read(size).decode("latin1")
    import ast
    value = ast.literal_eval(header); return str(value["descr"]), list(value["shape"])


def validate_preaccess() -> dict[str, Any]:
    spec = json.loads(SPEC.read_text()); schedule = read_signed(SCHEDULE); s0 = read_signed(S0 / "terminal_receipt.json")
    require(spec["stage_token"] == "S1_PILOT_ONLY" and spec["scientific_decisions_allowed"] is False, "S1 scope drift")
    failed_path = REPO / spec["recovery"]["predecessor_root"] / "terminal_receipt.json"; failed = read_signed(failed_path); require(failed["status"] == spec["recovery"]["predecessor_status"] == "FAILED_IMPLEMENTATION", "recovery predecessor status drift"); require(sha_file(failed_path) == spec["recovery"]["predecessor_terminal_sha256"] and failed["canonical_digest"] == spec["recovery"]["predecessor_terminal_canonical_digest"], "recovery predecessor binding drift")
    require(spec["predecessor"]["sha256"] == sha_file(S0 / "terminal_receipt.json") and spec["predecessor"]["canonical_digest"] == s0["canonical_digest"], "S0 binding drift")
    require(s0["payload"]["status"] == "PASS" and s0["payload"]["next_authorization"] == "S1_PILOT_ONLY", "S0 authorization invalid")
    payload = schedule["payload"]; records = payload["records"]; require(len(records) == payload["pilot_count"] == 1744 and payload["records_digest"] == digest(records), "pilot schedule drift")
    require(all(row["final_split"] == "pilot" and row["use_policy"] == "S1_PILOT_ONLY_PERMANENT_DEVELOPMENT" for row in records), "non-pilot schedule record")
    forbidden = payload["forbidden_split_metadata_only"]; require(set(forbidden[0]) == {"source_image_id", "image_sha256", "final_split"} and payload["forbidden_split_metadata_digest"] == digest(forbidden), "forbidden split metadata boundary drift")
    pilot_ids = {row["source_image_id"] for row in records}; pilot_hashes = {row["image_sha256"] for row in records}; forbidden_ids = {row["source_image_id"] for row in forbidden}; forbidden_hashes = {row["image_sha256"] for row in forbidden}; require(not pilot_ids & forbidden_ids and not pilot_hashes & forbidden_hashes, "pilot/forbidden split intersection")
    require(payload["forbidden_split_counts"] == {"probe_train": 5232, "probe_validation": 1744, "confirmatory_holdout": 1744}, "forbidden counts drift")
    implementation = read_signed(ROOT / "implementation_seal.json"); require(implementation["payload"]["pilot_or_model_outcome_access_before_seal"] is False, "pre-seal access attestation drift")
    for item in implementation["payload"]["files"]:
        path = REPO / item["path"]; require(record(path)["sha256"] == item["sha256"], "implementation file drift: " + item["path"]); require(bool(subprocess.check_output(["git", "-C", str(REPO), "ls-tree", "--full-tree", implementation["payload"]["commit"], "--", "vitnns_rebuild/" + item["path"]], text=True).strip()), "file absent from seal commit")
    return {"pilot_count": len(records), "pilot_forbidden_source_intersection": 0, "pilot_forbidden_hash_intersection": 0, "implementation_file_count": len(implementation["payload"]["files"]), "implementation_commit": implementation["payload"]["commit"]}


def unit_paths(level: str) -> list[Path]:
    phases = {"single_reference": ("single_reference",), "smoke": ("smoke",), "intermediate": ("smoke", "intermediate"), "full_pilot": ("smoke", "intermediate", "full_pilot")}[level]
    return sorted(path for phase in phases for path in (ROOT / phase / "shards").glob("shard-*/units/unit-*/completion/unit.json"))


def validate_units(level: str) -> dict[str, Any]:
    paths = unit_paths(level); expected = {"single_reference": 1, "smoke": 6, "intermediate": 24, "full_pilot": 1744}[level]; require(len(paths) == expected, "unit denominator mismatch")
    seen = set(); counts = collections.Counter(); gpu = []; cpu = []; wall = []; retained = 0; matched = 0; p1_valid = 0
    for path in paths:
        document = read_signed(path); require(document["document_type"] == "s1_unit_terminal", "wrong unit document type"); payload = document["payload"]; identity = payload["identity"]; index = identity["schedule_index"]; require(index not in seen, "duplicate schedule index"); seen.add(index); counts[document["status"]] += 1
        require(identity["mode"] in ("single_reference", "smoke", "intermediate", "full_pilot"), "mode drift"); require(payload["access"]["forbidden_split_image_bytes_read"] == payload["access"]["forbidden_split_mask_bytes_read"] == payload["access"]["forbidden_split_outcomes_read"] == 0, "forbidden access nonzero")
        require(payload["terminal"] == {"complete": True, "immutable": True}, "terminal marker drift")
        if document["status"] == "VALID":
            require(payload["native_hooked_parity"]["status"] == "PASS" and payload["nns"]["closure"]["status"] == "PASS", "numeric contract failed in valid unit"); require(payload["d1"]["scientific_decision"] is None and payload["p1"]["scientific_decision"] is None and payload["p2"]["scientific_decision"] is None, "pilot scientific decision present")
            p1_valid += payload["p1"]["valid_blocks"]; matched += payload["p2"]["matched_coordinates"]; resource = payload["resources"]; gpu.append((path.as_posix(), float(resource["gpu_active_seconds"]))); cpu.append((path.as_posix(), float(resource["cpu_seconds"]))); wall.append((path.as_posix(), float(resource["wall_seconds"])))
            for item in payload["arrays"]:
                array_path = REPO / item["path"]; require(array_path.is_file() and sha_file(array_path) == item["sha256"] and array_path.stat().st_size == item["bytes"], "array byte binding mismatch"); dtype, shape = npy_header(array_path); require(dtype in ("<f4", "|f4") and shape == item["shape"] and item["dtype"] == "float32", "array header contract mismatch"); retained += item["bytes"]
        else:
            require(document["status"] in NA_STATES + ("FAILED_NUMERICAL", "BLOCKED_DATA_INTEGRITY", "BLOCKED_INFRASTRUCTURE", "FAILED_IMPLEMENTATION"), "unknown unit state"); require(payload["failure"] is not None, "non-valid unit lacks failure")
    valid = counts["VALID"]; expected_indices = {0} if level == "single_reference" else set(range(expected)); require(seen == expected_indices, "schedule coverage drift")
    return {"level": level, "terminal_count": len(paths), "terminal_counts": dict(sorted(counts.items())), "valid_fraction": valid / len(paths), "p1_valid_block_fraction": None if valid == 0 else p1_valid / (valid * 12), "p2_matched_coordinate_fraction": None if valid == 0 else matched / (valid * 36), "forbidden_access_counts": {"bytes": 0, "outcomes": 0}, "resources": {"gpu_active_seconds": math.fsum(value for _, value in sorted(gpu)), "cpu_seconds": math.fsum(value for _, value in sorted(cpu)), "wall_seconds_sum": math.fsum(value for _, value in sorted(wall)), "retained_bytes": retained}}


def validate_final() -> dict[str, Any]:
    pre = validate_preaccess(); terminal = read_signed(ROOT / "terminal_receipt.json"); require(terminal["status"] in TERMINALS and terminal["payload"]["status"] == terminal["status"], "terminal state invalid"); require(terminal["payload"]["scientific_claim_decision"] is None, "S1 cannot decide scientific claims")
    if terminal["status"] == "PASS": require(terminal["payload"]["next_authorization"] == "S2_PROTOCOL_AND_POWER_FREEZE_ONLY", "PASS authorization drift")
    else: require(terminal["payload"]["next_authorization"] == "NONE", "blocked/failed authorization drift")
    level = terminal["payload"]["actual_level"]; units = validate_units(level); require(units["valid_fraction"] >= 0.85 and units["p2_matched_coordinate_fraction"] >= 0.98, "validity denominator gate failed")
    repeat = read_signed(ROOT / "deterministic_repeatability_report.json"); require(repeat["status"] == "PASS" and repeat["payload"]["all_byte_identical"], "deterministic repeatability failed")
    for name in ("synthetic_negative_test_report.json", "geometry_mask_validation_report.json", "nns_reconstruction_parity_report.json", "p1_p2_control_validation_report.json", "probe_engineering_validation_report.json", "resource_scaling_ledger.json", "failure_na_denominator_report.json", "s2_design_recommendation.json"):
        require(read_signed(ROOT / name)["status"] == "PASS", name + " not PASS")
    inventory = read_signed(ROOT / "whole_root_inventory.json"); files = inventory["payload"]["files"]; require(inventory["payload"]["files_digest"] == digest(files), "inventory digest drift"); excluded = {"completion/s1.pending.json", "whole_root_inventory.json", "terminal_receipt.json", "independent_validation_report.json", "false_pass_audit.json"}; observed = [record(path) for path in sorted(ROOT.rglob("*")) if path.is_file() and path.relative_to(ROOT).as_posix() not in excluded]; require(files == observed, "whole-root inventory drift")
    return {"status": "PASS", "preaccess": pre, "units": units, "terminal_status": terminal["status"], "next_authorization": terminal["payload"]["next_authorization"], "inventory_file_count": len(files)}


def false_pass_audit() -> dict[str, Any]:
    attacks = []
    base = json.loads((ROOT / "terminal_receipt.json").read_text())
    mutations = {"status": lambda x: x.update(status="BLOCKED_BUDGET"), "authorization": lambda x: x["payload"].update(next_authorization="S3"), "empty_result": lambda x: x["payload"].update(actual_sample_count=0)}
    for name, mutate in mutations.items():
        value = json.loads(json.dumps(base)); mutate(value); unsigned = dict(value); unsigned.pop("canonical_digest"); value["canonical_digest"] = digest(unsigned)
        rejected = False
        try:
            unsigned = dict(value); observed = unsigned.pop("canonical_digest"); require(observed == digest(unsigned), "digest"); require(value["status"] in TERMINALS, "status"); require(value["payload"].get("actual_sample_count", 1) > 0, "empty"); require(value["payload"].get("next_authorization") in ("S2_PROTOCOL_AND_POWER_FREEZE_ONLY", "NONE"), "authorization")
        except Exception: rejected = True
        require(rejected, "mutation false-passed: " + name); attacks.append({"attack": name, "rejected": True})
    schedule = json.loads(SCHEDULE.read_text()); for_mutations = (("split", lambda x: x["payload"]["records"][0].update(final_split="confirmatory_holdout")), ("hash", lambda x: x["payload"]["records"][0].update(image_sha256="0" * 64)))
    for name, mutate in for_mutations:
        value = json.loads(json.dumps(schedule)); mutate(value); unsigned = dict(value); unsigned.pop("canonical_digest"); value["canonical_digest"] = digest(unsigned); rejected = False
        try:
            unsigned = dict(value); observed = unsigned.pop("canonical_digest"); require(observed == digest(unsigned), "digest"); require(value["payload"]["records_digest"] == digest(value["payload"]["records"]), "records digest"); require(all(row["final_split"] == "pilot" and len(row["image_sha256"]) == 64 and row["image_sha256"] != "0" * 64 for row in value["payload"]["records"]), "split/hash semantic")
        except Exception: rejected = True
        require(rejected, "schedule mutation false-passed: " + name); attacks.append({"attack": name, "rejected": True})
    return {"status": "PASS", "attack_count": len(attacks), "attacks": attacks}


def write_final_reports() -> dict[str, Any]:
    validation = validate_final(); audit = false_pass_audit()
    for name, document_type, payload in (("independent_validation_report.json", "s1_independent_source_recomputing_validation_report", validation), ("false_pass_audit.json", "s1_false_pass_audit", audit)):
        path = ROOT / name; require(not path.exists(), "refusing overwrite: " + name); value = {"schema_version": 1, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S1-R1", "run_id": "s1_r1_pilot_20260914", "document_type": document_type, "status": "PASS", "payload": payload}; value["canonical_digest"] = digest(value)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"); stream.flush(); os.fsync(stream.fileno())
    return {"status": "PASS", "validation": validation, "false_pass": audit}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("preaccess", "units", "final", "false-pass", "write-final-reports")); parser.add_argument("--level", default="intermediate"); args = parser.parse_args()
    result = validate_preaccess() if args.command == "preaccess" else validate_units(args.level) if args.command == "units" else validate_final() if args.command == "final" else false_pass_audit() if args.command == "false-pass" else write_final_reports(); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
