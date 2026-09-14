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
SPEC = REPO / "config/nns_vit_patch_depth/s1_run_spec.json"
SCHEDULE = REPO / "manifests/nns_vit_patch_depth/s1_pilot_schedule.json"
SCHEMA = REPO / "config/nns_vit_patch_depth/s1_run_spec.schema.json"
ROOT = REPO / "evidence/nns_vit_patch_depth/s1_pilot_20260914"
STUDY = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
RUN = "s1_pilot_20260914"
NAMESPACE = "NNS_VIT_PATCH_DEPTH_S1_CLASS_THEN_SOURCE_V1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(document_type: str, status: str, payload: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    value = {"schema_version": 1, "study_id": STUDY, "stage_id": "S1", "run_id": RUN, "document_type": document_type, "status": status, "payload": dict(payload), **extra}
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


def freeze_root() -> None:
    require(not ROOT.exists(), "S1 evidence root already exists")
    require(not git("status", "--porcelain"), "worktree must be clean before implementation seal")
    head = git("rev-parse", "HEAD")
    sources = [
        SPEC, SCHEMA, SCHEDULE, REPO / "manifests/nns_vit_patch_depth/s1_artifact_contracts.json",
        REPO / "docs/nns_vit_patch_depth_s1_protocol_20260914.md", REPO / "src/vitnns/nns_patch_depth_s1.py",
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
    implementation = signed("s1_implementation_seal", "FROZEN", {"commit": head, "files": records, "files_digest": digest(records), "pilot_or_model_outcome_access_before_seal": False})
    protocol = signed("s1_prospective_protocol_binding", "FROZEN", {"protocol": record(REPO / "docs/nns_vit_patch_depth_s1_protocol_20260914.md"), "run_spec": record(SPEC), "run_spec_schema": record(SCHEMA), "artifact_contracts": record(REPO / "manifests/nns_vit_patch_depth/s1_artifact_contracts.json"), "scientific_decisions_allowed": False})
    escalation = signed("s1_deterministic_escalation_manifest", "FROZEN", {"schedule": record(SCHEDULE), "levels": json.loads(SCHEDULE.read_text())["payload"]["levels"], "minimum_stop_level": "intermediate", "decision_inputs": ["numeric_contracts", "typed_failure_rate", "control_matching_success", "deterministic_repeatability", "runtime_memory_storage", "variance_estimate_precision"], "forbidden_decision_inputs": ["effect_direction", "p_value", "heatmap_appearance", "support_convenience"]})
    source_manifest = signed("s1_pilot_source_manifest", "FROZEN", {"schedule": record(SCHEDULE), "pilot_count": 1744, "pilot_records_digest": json.loads(SCHEDULE.read_text())["payload"]["records_digest"], "permanent_use": "DEVELOPMENT_ONLY", "image_bytes_accessed": 0, "mask_bytes_accessed": 0, "model_loaded": False})
    for name, value in (("pilot_access_policy.json", policy), ("implementation_seal.json", implementation), ("prospective_protocol.json", protocol), ("deterministic_escalation_manifest.json", escalation), ("pilot_source_manifest.json", source_manifest)):
        write_new(ROOT / name, value)
    write_new(ROOT / "completion/s1.pending.json", signed("s1_pending", "PENDING", {"started_unix_ns": time.time_ns(), "implementation_commit": head, "next_authorization": "NONE_WHILE_PENDING", "terminal_published": False}))
    print(json.dumps({"status": "PASS", "implementation_commit": head, "root": str(ROOT)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("generate", "freeze-root")); args = parser.parse_args()
    if args.command == "generate": generate()
    else: freeze_root()


if __name__ == "__main__":
    main()
