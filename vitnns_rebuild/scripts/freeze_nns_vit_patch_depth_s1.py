#!/usr/bin/env python3
"""Metadata-only S1 schedule and pre-access evidence freezer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
S0 = REPO / "evidence/nns_vit_patch_depth/s0_r5_20260914"
SPEC = REPO / "config/nns_vit_patch_depth/s1_r1_run_spec.json"
SCHEDULE = REPO / "manifests/nns_vit_patch_depth/s1_pilot_schedule.json"
SCHEMA = REPO / "config/nns_vit_patch_depth/s1_r1_run_spec.schema.json"
ROOT = REPO / "evidence/nns_vit_patch_depth/s1_r1_pilot_20260914"
STUDY = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
RUN = "s1_r1_pilot_20260914"
NAMESPACE = "NNS_VIT_PATCH_DEPTH_S1_CLASS_THEN_SOURCE_V1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(document_type: str, status: str, payload: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    value = {"schema_version": 1, "study_id": STUDY, "stage_id": "S1-R1", "run_id": RUN, "document_type": document_type, "status": status, "payload": dict(payload), **extra}
    value["canonical_digest"] = digest(value)
    return value


def read_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    observed = value.pop("canonical_digest")
    require(observed == digest(value), "canonical digest mismatch: " + str(path))
    value["canonical_digest"] = observed
    return value


def sha_file(path: Path) -> str:
    before = path.stat(); h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file changed during hash")
    return h.hexdigest()


def record(path: Path) -> dict[str, Any]:
    result = {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if "canonical_digest" in value:
            result["canonical_digest"] = value["canonical_digest"]
    return result


def write_replace(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), "temporary collision: " + str(temporary))
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, value: Any) -> None:
    require(not path.exists(), "refusing overwrite: " + str(path))
    write_replace(path, value)


def closed_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "object", "additionalProperties": False, "required": sorted(value), "properties": {key: closed_schema(item) for key, item in value.items()}}
    if isinstance(value, list):
        variants = []
        for item in value:
            schema = closed_schema(item)
            if schema not in variants:
                variants.append(schema)
        items: Any = False if not variants else variants[0] if len(variants) == 1 else {"anyOf": variants}
        return {"type": "array", "items": items}
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "boolean"}
    if type(value) is int:
        return {"type": "integer"}
    if type(value) is float:
        return {"type": "number"}
    return {"type": "string"}


def hash_key(*parts: str) -> str:
    return hashlib.sha256((NAMESPACE + "\0" + "\0".join(parts)).encode()).hexdigest()


def build_schedule() -> dict[str, Any]:
    terminal = read_signed(S0 / "terminal_receipt.json")
    split = read_signed(S0 / "split_manifest.json")
    require(terminal["payload"]["status"] == "PASS" and terminal["payload"]["next_authorization"] == "S1_PILOT_ONLY", "S0 authorization invalid")
    require(split["payload"]["split_counts"] == {"pilot": 1744, "probe_train": 5232, "probe_validation": 1744, "confirmatory_holdout": 1744}, "split counts drift")
    by_split = {name: [row for row in split["payload"]["records"] if row["final_split"] == name] for name in split["payload"]["split_counts"]}
    pilot = sorted(by_split["pilot"], key=lambda row: (hash_key(row["annotation_wnid"]), hash_key(row["annotation_wnid"], row["source_image_id"]), row["source_image_id"]))
    records = []
    for index, row in enumerate(pilot):
        records.append({
            "schedule_index": index, "source_image_id": row["source_image_id"], "source_group_id": row["source_group_id"],
            "candidate_id": row["candidate_id"], "annotation_wnid": row["annotation_wnid"],
            "target_model_class_index": row["target_model_class_index"], "target_annotation_label_id": row["annotation_label_id"],
            "image_path": row["image_local_path"], "image_sha256": row["image_sha256"], "image_bytes": row["image_bytes"],
            "image_size": [row["image_height"], row["image_width"]], "annotation_archive_path": row["annotation_archive_path"],
            "annotation_sha256": row["annotation_sha256"], "mask_size": [row["mask_height"], row["mask_width"]],
            "final_split": "pilot", "use_policy": "S1_PILOT_ONLY_PERMANENT_DEVELOPMENT",
            "class_order_key": hash_key(row["annotation_wnid"]), "source_order_key": hash_key(row["annotation_wnid"], row["source_image_id"]),
        })
    metadata_only = []
    for name in ("probe_train", "probe_validation", "confirmatory_holdout"):
        metadata_only.extend({"source_image_id": row["source_image_id"], "image_sha256": row["image_sha256"], "final_split": name} for row in by_split[name])
    payload = {
        "status": "FROZEN_BEFORE_PILOT_IMAGE_MASK_MODEL_OR_OUTCOME_ACCESS", "selection_namespace": NAMESPACE,
        "ordering": "class_namespace_sha256_then_source_namespace_sha256_then_source_image_id",
        "pilot_count": len(records), "pilot_class_count": len({row["annotation_wnid"] for row in records}),
        "records": records, "records_digest": digest(records),
        "levels": [
            {"name": "smoke", "cumulative_schedule_indices": list(range(6)), "new_schedule_indices": list(range(6))},
            {"name": "intermediate", "cumulative_schedule_indices": list(range(24)), "new_schedule_indices": list(range(6, 24))},
            {"name": "full_pilot", "cumulative_schedule_indices": list(range(1744)), "new_schedule_indices": list(range(24, 1744))}
        ],
        "forbidden_split_metadata_only": metadata_only, "forbidden_split_metadata_digest": digest(metadata_only),
        "forbidden_split_counts": {name: len(by_split[name]) for name in ("probe_train", "probe_validation", "confirmatory_holdout")},
        "pilot_pair_per_class_required": True, "dynamic_reordering": False, "scientific_outcome_used_in_schedule": False,
    }
    return signed("s1_deterministic_pilot_schedule", "FROZEN", payload, predecessor={"path": str((S0 / 'terminal_receipt.json').relative_to(REPO)), "sha256": sha_file(S0 / 'terminal_receipt.json'), "canonical_digest": terminal['canonical_digest']})


def generate() -> None:
    spec = json.loads(SPEC.read_text())
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Closed S1 run specification", **closed_schema(spec)}
    write_replace(SCHEMA, schema)
    write_replace(SCHEDULE, build_schedule())
    print(json.dumps({"status": "PASS", "schema": record(SCHEMA), "schedule": record(SCHEDULE)}, sort_keys=True))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def git_blob_record(commit: str, path: str) -> dict[str, Any]:
    tree_path = "vitnns_rebuild/" + path
    line = git("ls-tree", "--full-tree", "-l", commit, "--", tree_path)
    require(bool(line), "missing Git blob: " + tree_path)
    metadata, observed_path = line.split("\t", 1); mode, kind, oid, blob_bytes = metadata.split()
    require(kind == "blob" and observed_path == tree_path, "Git tree identity drift")
    blob = subprocess.check_output(["git", "-C", str(REPO), "cat-file", "blob", oid])
    return {"path": path, "commit": commit, "git_mode": mode, "git_blob_oid": oid, "tree_blob_bytes": int(blob_bytes), "content_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def smoke_bindings() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    units = []
    for path in ROOT.glob("smoke/shards/shard-*/units/unit-*/completion/unit.json"):
        value = read_signed(path); units.append({"schedule_index": value["payload"]["identity"]["schedule_index"], "status": value["status"], **record(path)})
    units.sort(key=lambda item: item["schedule_index"]); require([item["schedule_index"] for item in units] == list(range(6)) and all(item["status"] == "VALID" for item in units), "smoke unit baseline invalid")
    shards = []
    for path in ROOT.glob("smoke/shards/shard-*/completion/shard.json"):
        value = read_signed(path); shards.append({"shard_id": value["payload"]["shard_id"], "status": value["status"], "terminal_counts": value["payload"]["terminal_counts"], **record(path)})
    shards.sort(key=lambda item: item["shard_id"]); require([item["shard_id"] for item in shards] == [0, 1, 2] and all(item["status"] == "PASS" and item["terminal_counts"] == {"VALID": 2} for item in shards), "smoke shard baseline invalid")
    require(not list((ROOT / "smoke").rglob("*.pending.json")) and not list((ROOT / "smoke").rglob("*.tmp")), "smoke has partial files")
    files = [record(path) for path in sorted((ROOT / "smoke").rglob("*")) if path.is_file()]
    return units, shards, files


def freeze_a1() -> None:
    require(ROOT.is_dir() and (ROOT / "completion/s1.pending.json").is_file() and not (ROOT / "terminal_receipt.json").exists(), "S1-R1 is not active")
    for name in ("smoke_baseline_binding.json", "a1_implementation_seal.json", "s1_r1_a1_amendment.json"):
        require(not (ROOT / name).exists(), "A1 already frozen: " + name)
    require(not (ROOT / "intermediate").exists() and not (ROOT / "full_pilot").exists(), "A1 must precede intermediate/full access")
    require(not git("status", "--porcelain"), "worktree must be clean before A1 seal")
    head = git("rev-parse", "HEAD"); units, shards, smoke_files = smoke_bindings()
    source_paths = [
        "config/nns_vit_patch_depth/s1_r1_run_spec.json", "config/nns_vit_patch_depth/s1_r1_run_spec.schema.json",
        "docs/nns_vit_patch_depth_s1_r1_a1_amendment_20260914.md", "scripts/freeze_nns_vit_patch_depth_s1.py",
        "scripts/run_nns_vit_patch_depth_s1.py", "scripts/validate_nns_vit_patch_depth_s1.py", "tests/test_nns_patch_depth_s1.py",
        "src/vitnns/nns_patch_depth_s1_worker.py",
    ]
    sources = []
    for relative in source_paths:
        live = record(REPO / relative); blob = git_blob_record(head, relative); require(live["sha256"] == blob["sha256"], "A1 live/Git drift: " + relative); sources.append({**live, "commit": head, "git_blob_oid": blob["git_blob_oid"], "git_mode": blob["git_mode"]})
    old_commit = "5695d1226d39bc4b03282231a2a50171f05eb813"; old_contract = git_blob_record(old_commit, "config/nns_vit_patch_depth/s1_r1_run_spec.json"); new_contract = record(SPEC)
    old_value = json.loads(subprocess.check_output(["git", "-C", str(REPO), "show", old_commit + ":vitnns_rebuild/config/nns_vit_patch_depth/s1_r1_run_spec.json"], text=True)); new_value = json.loads(SPEC.read_text())
    require(old_value["escalation"]["engineering_stop_conditions"]["minimum_control_matching_fraction"] == 0.98, "old P2 gate drift")
    require(new_value["escalation"]["engineering_stop_conditions"]["minimum_control_matching_fraction"] == 0.90, "new P2 gate drift")
    require(new_value["parallel"]["minimum_workers"] == 1 and new_value["parallel"]["maximum_workers"] == 8 and new_value["parallel"]["existing_compute_pid_is_disqualifying"] is False, "A1 elastic policy drift")
    baseline = signed("s1_r1_a1_smoke_baseline_binding", "PASS", {"frozen_before_intermediate_access": True, "unit_count": 6, "unit_terminal_counts": {"VALID": 6}, "shard_count": 3, "shard_terminal_counts": {"PASS": 3}, "units": units, "shards": shards, "all_smoke_files": smoke_files, "all_smoke_files_digest": digest(smoke_files), "pending_count": 0, "writer_count": 0, "immutability_rule": "indices_0_to_5_must_not_be_deleted_overwritten_recomputed_or_replaced"})
    a1_seal = signed("s1_r1_a1_implementation_seal", "FROZEN", {"commit": head, "files": sources, "files_digest": digest(sources), "applies_before_intermediate_access": True, "smoke_scientific_arrays_or_outcomes_used_to_choose_changes": False})
    amendment = signed("s1_r1_a1_prospective_amendment", "FROZEN", {"amendment_id": "S1-R1-A1", "owner_authorized": True, "frozen_after_smoke_before_intermediate": True, "amendment_document": record(REPO / new_value["amendment"]["path"]), "old_contract": old_contract, "new_contract": new_contract, "old_aggregate_p2_gate": 0.98, "new_aggregate_p2_gate": 0.90, "elastic_workers": {"minimum": 1, "maximum": 8, "fewer_than_eight_is_failure_or_blocked": False}, "gpu_eligibility": {"name": "NVIDIA H20", "minimum_free_memory_mib": 21504, "maximum_utilization_percent": 10, "existing_compute_pid_is_disqualifying": False, "record_compute_pid_and_memory": True}, "scope": ["intermediate/full launch", "aggregate P2 matched-control engineering gate"], "unchanged": ["pilot records and schedule", "target", "component", "geometry", "quadrature", "P1", "P2 per-coordinate matching", "D1", "D2", "scientific claim boundary"], "scientific_decisions_allowed": False, "smoke_baseline_digest": baseline["canonical_digest"]})
    for name, value in (("smoke_baseline_binding.json", baseline), ("a1_implementation_seal.json", a1_seal), ("s1_r1_a1_amendment.json", amendment)):
        write_new(ROOT / name, value)
    print(json.dumps({"status": "PASS", "commit": head, "smoke_units": 6, "smoke_shards": 3, "smoke_files": len(smoke_files), "new_P2_gate": 0.90}, sort_keys=True))


def freeze_root() -> None:
    require(not ROOT.exists(), "S1 evidence root already exists")
    require(not git("status", "--porcelain"), "worktree must be clean before implementation seal")
    head = git("rev-parse", "HEAD")
    sources = [
        SPEC, SCHEMA, SCHEDULE, REPO / "manifests/nns_vit_patch_depth/s1_artifact_contracts.json",
        REPO / "docs/nns_vit_patch_depth_s1_protocol_20260914.md", REPO / "src/vitnns/nns_patch_depth_s1.py",
        REPO / "docs/nns_vit_patch_depth_s1_r1_recovery_20260914.md",
        REPO / "src/vitnns/nns_patch_depth_s1_worker.py", REPO / "scripts/run_nns_vit_patch_depth_s1.py",
        REPO / "scripts/validate_nns_vit_patch_depth_s1.py", REPO / "tests/test_nns_patch_depth_s1.py",
        REPO / "scripts/nns_patch_depth_s1_env.sh", REPO / "src/vitnns/stage_y_y1.py",
    ]
    records = []
    for path in sources:
        item = record(path); tree_path = "vitnns_rebuild/" + item["path"]
        line = git("ls-tree", "--full-tree", "-l", head, "--", tree_path)
        require(bool(line), "implementation source absent from seal commit: " + item["path"]); item["seal_commit"] = head
        records.append(item)
    ROOT.mkdir(parents=True)
    policy = signed("s1_pilot_access_policy", "FROZEN", {
        "allowed": {"split": "pilot", "count": 1744, "bytes": ["JPEG", "mask"], "outcomes": ["features", "logits", "NNS", "P1", "P2", "D1", "D2_engineering"]},
        "forbidden": [{"split": name, "allowed_fields": ["source_image_id", "image_sha256", "final_split"], "byte_access_limit": 0, "outcome_access_limit": 0} for name in ("probe_train", "probe_validation", "confirmatory_holdout")],
        "pilot_permanently_development_only": True, "confirmatory_label": "public-validation confirmatory holdout", "forbidden_claim": "ImageNet-S official test result",
    })
    failed_terminal = REPO / "evidence/nns_vit_patch_depth/s1_pilot_20260914/terminal_receipt.json"
    failed = read_signed(failed_terminal); require(failed["status"] == "FAILED_IMPLEMENTATION" and failed["payload"]["valid_unit_count"] == 0, "failed predecessor identity drift")
    implementation = signed("s1_r1_implementation_seal", "FROZEN", {"commit": head, "files": records, "files_digest": digest(records), "pilot_or_model_outcome_access_before_seal": False, "recovery_predecessor": {**record(failed_terminal), "canonical_digest": failed["canonical_digest"]}, "repair_scope": "JSON_NATIVE_NUMPY_SCALAR_SERIALIZATION_ONLY", "scientific_contract_changed": False})
    protocol = signed("s1_r1_prospective_protocol_binding", "FROZEN", {"protocol": record(REPO / "docs/nns_vit_patch_depth_s1_protocol_20260914.md"), "recovery_amendment": record(REPO / "docs/nns_vit_patch_depth_s1_r1_recovery_20260914.md"), "run_spec": record(SPEC), "run_spec_schema": record(SCHEMA), "artifact_contracts": record(REPO / "manifests/nns_vit_patch_depth/s1_artifact_contracts.json"), "scientific_decisions_allowed": False})
    escalation = signed("s1_deterministic_escalation_manifest", "FROZEN", {"schedule": record(SCHEDULE), "levels": json.loads(SCHEDULE.read_text())["payload"]["levels"], "minimum_stop_level": "intermediate", "decision_inputs": ["numeric_contracts", "typed_failure_rate", "control_matching_success", "deterministic_repeatability", "runtime_memory_storage", "variance_estimate_precision"], "forbidden_decision_inputs": ["effect_direction", "p_value", "heatmap_appearance", "support_convenience"]})
    source_manifest = signed("s1_pilot_source_manifest", "FROZEN", {"schedule": record(SCHEDULE), "pilot_count": 1744, "pilot_records_digest": json.loads(SCHEDULE.read_text())["payload"]["records_digest"], "permanent_use": "DEVELOPMENT_ONLY", "image_bytes_accessed": 0, "mask_bytes_accessed": 0, "model_loaded": False})
    for name, value in (("pilot_access_policy.json", policy), ("implementation_seal.json", implementation), ("prospective_protocol.json", protocol), ("deterministic_escalation_manifest.json", escalation), ("pilot_source_manifest.json", source_manifest)):
        write_new(ROOT / name, value)
    write_new(ROOT / "completion/s1.pending.json", signed("s1_pending", "PENDING", {"started_unix_ns": time.time_ns(), "implementation_commit": head, "next_authorization": "NONE_WHILE_PENDING", "terminal_published": False}))
    print(json.dumps({"status": "PASS", "implementation_commit": head, "root": str(ROOT)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("generate", "freeze-root", "freeze-a1")); args = parser.parse_args()
    if args.command == "generate": generate()
    elif args.command == "freeze-root": freeze_root()
    else: freeze_a1()


if __name__ == "__main__":
    main()
