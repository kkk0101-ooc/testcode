#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
AUTHORIZED_COMMIT = "66a783d15971308b3307ab4551f00e0abfae0b03"
STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
STAGE_ID = "S0-R5"
RUN_ID = "s0_r5_20260914"
TRAIN_NAMESPACE = "S0_R1_METADATA_RESERVATION_V1"
HOLDOUT_NAMESPACE = "S0_R5_PUBLIC_VALIDATION_CONFIRMATORY_HOLDOUT_V1"
SPLITS = ("pilot", "probe_train", "probe_validation", "confirmatory_holdout")
PRE_TERMINAL_NAMES = (
    "source_binding",
    "route_contract",
    "class_manifest",
    "split_manifest",
    "selection_ledger",
    "independence_audit",
)
SOURCE_PATHS = (
    "docs/nns_vit_patch_depth_research_goals_20260914.md",
    "docs/nns_vit_patch_depth_execution_plan_20260914.md",
    "evidence/nns_vit_patch_depth/s0_r4_20260914/artifact_contracts.schema.json",
    "evidence/nns_vit_patch_depth/s0_r4_20260914/train_original_binding.json",
    "evidence/nns_vit_patch_depth/s0_r4_20260914/byte_alias_audit.json",
    "evidence/nns_vit_patch_depth/s0_r4_20260914/split_feasibility.json",
    "evidence/nns_vit_patch_depth/s0_r4_20260914/feasibility_decision.json",
    "evidence/nns_vit_patch_depth/s0_r3_20260914/artifact_contracts.schema.json",
    "evidence/nns_vit_patch_depth/s0_r3_20260914/image_mask_qc.json",
    "evidence/nns_vit_patch_depth/s0_r3_20260914/split_feasibility.json",
    "evidence/nns_vit_patch_depth/s0_r3_20260914/feasibility_decision.json",
    "evidence/nns_vit_patch_depth/s0_r2_20260914/authorization_receipt.json",
    "evidence/nns_vit_patch_depth/s0_r2_20260914/annotation_package_inventory.json",
    "evidence/nns_vit_patch_depth/s0_r2_20260914/split_feasibility.json",
    "evidence/nns_vit_patch_depth/s0_r2_20260914/feasibility_decision.json",
    "evidence/nns_vit_patch_depth/s0_r2_operator_note_20260914/session_access_note.json",
    "assets/imagenet/imagenets919_train_semi/extraction_receipt.json",
    "evidence/nns_vit_patch_depth/s0_r1_20260914/image_annotation_class_manifest.json",
    "evidence/nns_vit_patch_depth/s0_r1_20260914/reuse_and_exclusion_manifest.json",
    "evidence/nns_vit_patch_depth/s0_r1_20260914/annotation_contract.json",
    "evidence/nns_vit_patch_depth/s0_r1_20260914/split_feasibility.json",
    "evidence/nns_vit_patch_depth/s0_r1_20260914/feasibility_decision.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(value):
    return dict(value, canonical_digest=digest(value))


def serialize(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"


def git(*args, binary=False):
    return subprocess.check_output(["git", "-C", str(WORKSPACE), *args], text=not binary)


def file_record(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file changed while hashing")
    result = {"path": str(path.relative_to(REPO)), "bytes": before.st_size, "sha256": h.hexdigest()}
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict) and "canonical_digest" in value:
            observed = digest({key: item for key, item in value.items() if key != "canonical_digest"})
            require(observed == value["canonical_digest"], "canonical digest mismatch: " + str(path))
            result["canonical_digest"] = observed
    return result


def read_signed(relative):
    path = REPO / relative
    value = json.loads(path.read_text())
    require(value["canonical_digest"] == digest({key: item for key, item in value.items() if key != "canonical_digest"}), "canonical digest mismatch: " + relative)
    return value


def commit_binding(relative):
    path = REPO / relative
    observed = file_record(path)
    line = git("ls-tree", "-l", AUTHORIZED_COMMIT, "--", "vitnns_rebuild/" + relative).strip()
    require(line, "source absent from authorized commit: " + relative)
    metadata, tree_path = line.split("\t", 1)
    mode, kind, blob_oid, blob_bytes = metadata.split()
    require(kind == "blob" and tree_path == "vitnns_rebuild/" + relative, "unexpected tree entry: " + relative)
    blob = git("cat-file", "blob", blob_oid, binary=True)
    if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        pointer = dict(line.split(" ", 1) for line in blob.decode().splitlines()[1:])
        require(pointer["oid"] == "sha256:" + observed["sha256"], "LFS OID mismatch: " + relative)
        require(int(pointer["size"]) == observed["bytes"], "LFS size mismatch: " + relative)
        storage = "git_lfs"
    else:
        expected_git_blob = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
        require(expected_git_blob == blob_oid, "Git blob identity mismatch: " + relative)
        require(blob == path.read_bytes(), "working file differs from authorized commit: " + relative)
        storage = "git_blob"
    return {**observed, "authorized_commit": AUTHORIZED_COMMIT, "git_blob_oid": blob_oid, "git_mode": mode, "storage": storage, "tree_blob_bytes": int(blob_bytes)}


def selection_key(namespace, source_image_id):
    return hashlib.sha256((namespace + "\0" + source_image_id).encode()).hexdigest()


def closed_contract(value):
    if isinstance(value, dict):
        return {"type": "object", "additionalProperties": False, "required": sorted(value), "properties": {key: closed_contract(item) for key, item in value.items()}}
    if isinstance(value, list):
        schemas = []
        for item in value:
            schema = closed_contract(item)
            if schema not in schemas:
                schemas.append(schema)
        return {"type": "array", "items": schemas[0] if len(schemas) == 1 else {"anyOf": schemas} if schemas else False}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    return {"type": "string"}


def validate_closed(value, schema):
    require(schema is not False, "unexpected array item")
    if "anyOf" in schema:
        for choice in schema["anyOf"]:
            try:
                validate_closed(value, choice)
                return
            except ValueError:
                pass
        raise ValueError("no matching closed variant")
    kind = schema["type"]
    types = {"object": dict, "array": list, "null": type(None), "boolean": bool, "integer": int, "number": float, "string": str}
    require(type(value) is types[kind], "wrong schema type")
    if kind == "object":
        require(set(value) == set(schema["required"]) == set(schema["properties"]), "unclosed object")
        require(schema["additionalProperties"] is False, "open object schema")
        for key, item in value.items():
            validate_closed(item, schema["properties"][key])
    elif kind == "array":
        for item in value:
            validate_closed(item, schema["items"])


def write_new(path, value):
    with path.open("xb") as stream:
        stream.write(serialize(value))
        stream.flush()
        os.fsync(stream.fileno())


def build_documents(implementation=None):
    current_head = git("rev-parse", "HEAD").strip()
    require(current_head == AUTHORIZED_COMMIT, "S0-R5 must start at authorized commit")
    require(not git("status", "--porcelain", "--untracked-files=no").strip(), "tracked worktree changed before S0-R5")
    bindings = [commit_binding(relative) for relative in SOURCE_PATHS]
    bindings_by_path = {item["path"]: item for item in bindings}

    r4_decision = read_signed("evidence/nns_vit_patch_depth/s0_r4_20260914/feasibility_decision.json")
    r4_alias = read_signed("evidence/nns_vit_patch_depth/s0_r4_20260914/byte_alias_audit.json")
    r4_split = read_signed("evidence/nns_vit_patch_depth/s0_r4_20260914/split_feasibility.json")
    r4_train = read_signed("evidence/nns_vit_patch_depth/s0_r4_20260914/train_original_binding.json")
    r3_qc = read_signed("evidence/nns_vit_patch_depth/s0_r3_20260914/image_mask_qc.json")
    manifest = read_signed("evidence/nns_vit_patch_depth/s0_r1_20260914/image_annotation_class_manifest.json")
    reuse = read_signed("evidence/nns_vit_patch_depth/s0_r1_20260914/reuse_and_exclusion_manifest.json")
    operator_note = read_signed("evidence/nns_vit_patch_depth/s0_r2_operator_note_20260914/session_access_note.json")
    require(r4_decision["payload"]["data_readiness_status"] == "BLOCKED_SPLIT", "S0-R4 terminal status changed")
    require(r4_decision["payload"]["next_authorization"] == "APPROVE_PROSPECTIVE_PUBLIC_VALIDATION_CONFIRMATORY_HOLDOUT_OR_SUPPLY_TEST_TRUTH", "S0-R4 next authorization changed")
    require(r4_alias["payload"]["status"] == "PASS_WITH_QUARANTINE", "S0-R4 byte audit changed")
    require(r4_split["payload"]["complete_train_2_6_2_class_count"] == 878, "S0-R4 train class count changed")
    require(r4_split["payload"]["full_train_classes_with_validation_ge_2"] == 872, "S0-R4 route count changed")
    require(r4_train["payload"]["status"] == "PASS" and r4_train["payload"]["selected_image_count"] == 9190, "train binding changed")

    manifest_by_id = {record["candidate_id"]: record for record in manifest["payload"]["records"]}
    qc_by_id = {record["candidate_id"]: record for record in r3_qc["payload"]["records"]}
    adjusted_by_id = {record["candidate_id"]: record for record in r4_split["payload"]["adjusted_records"]}
    require(len(qc_by_id) == len(adjusted_by_id) == 21609, "candidate universe mismatch")
    require(set(qc_by_id) == set(adjusted_by_id), "QC/R4 candidate identity mismatch")
    eligible = [record for record in r4_split["payload"]["adjusted_records"] if record["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE"]
    require(len(eligible) == 21136 and all(not record["exclusion_reasons"] for record in eligible), "R4 eligible universe changed")

    train_by_class = collections.defaultdict(lambda: collections.defaultdict(list))
    validation_by_class = collections.defaultdict(list)
    for record in eligible:
        if record["official_split"] == "train-semi":
            train_by_class[record["annotation_wnid"]][record["provisional_pool"]].append(record)
        elif record["official_split"] == "validation":
            validation_by_class[record["annotation_wnid"]].append(record)
        else:
            raise ValueError("unexpected eligible official split")

    accepted_classes = []
    all_wnids = sorted({record["annotation_wnid"] for record in r4_split["payload"]["adjusted_records"]})
    for wnid in all_wnids:
        pools = train_by_class[wnid]
        if len(pools["pilot"]) == 2 and len(pools["probe_train"]) == 6 and len(pools["probe_validation"]) == 2 and len(validation_by_class[wnid]) >= 2:
            accepted_classes.append(wnid)
    require(len(all_wnids) == 919 and len(accepted_classes) == 872, "accepted class count does not match S0-R4")
    accepted_set = set(accepted_classes)

    selected_ids = set()
    holdout_ranks = {}
    holdout_keys = {}
    for wnid in accepted_classes:
        ranked = sorted(
            validation_by_class[wnid],
            key=lambda record: (selection_key(HOLDOUT_NAMESPACE, manifest_by_id[record["candidate_id"]]["source_image_id"]), record["candidate_id"]),
        )
        for rank, record in enumerate(ranked[:2], 1):
            candidate_id = record["candidate_id"]
            selected_ids.add(candidate_id)
            holdout_ranks[candidate_id] = rank
            holdout_keys[candidate_id] = selection_key(HOLDOUT_NAMESPACE, manifest_by_id[candidate_id]["source_image_id"])
        for split in ("pilot", "probe_train", "probe_validation"):
            rows = pools = train_by_class[wnid][split]
            for record in rows:
                selected_ids.add(record["candidate_id"])
    require(len(selected_ids) == len(accepted_classes) * 12 == 10464, "selected record count mismatch")

    final_records = []
    for candidate_id in sorted(selected_ids):
        adjusted = adjusted_by_id[candidate_id]
        source = manifest_by_id[candidate_id]
        qc = qc_by_id[candidate_id]
        final_split = "confirmatory_holdout" if adjusted["official_split"] == "validation" else adjusted["provisional_pool"]
        namespace = HOLDOUT_NAMESPACE if final_split == "confirmatory_holdout" else TRAIN_NAMESPACE
        key = holdout_keys[candidate_id] if final_split == "confirmatory_holdout" else selection_key(TRAIN_NAMESPACE, source["source_image_id"])
        if final_split == "confirmatory_holdout":
            rank = holdout_ranks[candidate_id]
        else:
            same_pool = sorted(train_by_class[adjusted["annotation_wnid"]][final_split], key=lambda row: (selection_key(TRAIN_NAMESPACE, manifest_by_id[row["candidate_id"]]["source_image_id"]), row["candidate_id"]))
            rank = [row["candidate_id"] for row in same_pool].index(candidate_id) + 1
        final_records.append({
            "annotation_archive_path": qc["mask_archive_path"],
            "annotation_label_id": qc["annotation_label_id"],
            "annotation_sha256": qc["annotation_sha256"],
            "annotation_wnid": qc["annotation_wnid"],
            "candidate_id": candidate_id,
            "final_split": final_split,
            "image_archive": source["image_archive"],
            "image_bytes": qc["image_bytes"],
            "image_height": qc["image_height"],
            "image_local_path": qc["image_local_path"],
            "image_member": source["image_member"],
            "image_sha256": qc["image_sha256"],
            "image_width": qc["image_width"],
            "mask_height": qc["mask_height"],
            "mask_width": qc["mask_width"],
            "official_split": qc["official_split"],
            "r4_adjusted_record_digest": digest(adjusted),
            "r3_qc_record_digest": digest(qc),
            "selection_key_sha256": key,
            "selection_namespace": namespace,
            "selection_rank_within_class_split": rank,
            "source_group_id": source["group_id"],
            "source_image_id": source["source_image_id"],
            "target_model_class_index": source["exact_target_class_index"],
            "use_policy": "CONFIRMATORY_HOLDOUT_NO_ACCESS_BEFORE_FROZEN_EVALUATION" if final_split == "confirmatory_holdout" else "S1_PILOT_ONLY" if final_split == "pilot" else "PROBE_TRAIN_ONLY" if final_split == "probe_train" else "PROBE_VALIDATION_ONLY",
        })
    split_order = {name: index for index, name in enumerate(SPLITS)}
    final_records.sort(key=lambda row: (split_order[row["final_split"]], row["annotation_wnid"], row["selection_rank_within_class_split"], row["candidate_id"]))
    split_counts = collections.Counter(record["final_split"] for record in final_records)
    require(dict(split_counts) == {"pilot": 1744, "probe_train": 5232, "probe_validation": 1744, "confirmatory_holdout": 1744}, "final split counts mismatch")

    class_records = []
    for wnid in all_wnids:
        train_counts = {split: len(train_by_class[wnid][split]) for split in ("pilot", "probe_train", "probe_validation")}
        reasons = []
        if train_counts != {"pilot": 2, "probe_train": 6, "probe_validation": 2}:
            reasons.append("INCOMPLETE_TRAIN_2_6_2")
        if len(validation_by_class[wnid]) < 2:
            reasons.append("INSUFFICIENT_PUBLIC_VALIDATION_CONFIRMATORY_LT2")
        selected = wnid in accepted_set
        source_rows = [manifest_by_id[row["candidate_id"]] for rows in train_by_class[wnid].values() for row in rows] + [manifest_by_id[row["candidate_id"]] for row in validation_by_class[wnid]]
        exact_indices = sorted({row["exact_target_class_index"] for row in source_rows if row["exact_target_class_index"] is not None})
        require(not selected or len(exact_indices) == 1, "selected class mapping is not exact")
        class_records.append({
            "annotation_wnid": wnid,
            "class_status": "SELECTED" if selected else "EXCLUDED_CLASS",
            "exclusion_reasons": reasons,
            "qualified_public_validation_count": len(validation_by_class[wnid]),
            "qualified_train_counts": train_counts,
            "selected_counts": {"pilot": 2 if selected else 0, "probe_train": 6 if selected else 0, "probe_validation": 2 if selected else 0, "confirmatory_holdout": 2 if selected else 0},
            "target_model_class_index": exact_indices[0] if selected else None,
        })
    require(sum(record["class_status"] == "SELECTED" for record in class_records) == 872, "class manifest selected count mismatch")

    ledger = []
    for record in sorted(r4_split["payload"]["adjusted_records"], key=lambda row: row["candidate_id"]):
        candidate_id = record["candidate_id"]
        source = manifest_by_id[candidate_id]
        eligible_state = record["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE"
        selected = candidate_id in selected_ids
        if selected:
            disposition = "SELECTED_CONFIRMATORY_HOLDOUT" if record["official_split"] == "validation" else "SELECTED_" + record["provisional_pool"].upper()
        elif not eligible_state:
            disposition = "EXCLUDED_BY_S0_R4_QC_OR_ISOLATION"
        elif record["annotation_wnid"] not in accepted_set:
            disposition = "EXCLUDED_CLASS_NOT_COMPLETE_2_6_2_PLUS_2"
        else:
            disposition = "ELIGIBLE_PUBLIC_VALIDATION_NOT_SELECTED_FIXED_TOP2"
        ledger.append({
            "annotation_wnid": record["annotation_wnid"],
            "candidate_id": candidate_id,
            "disposition": disposition,
            "final_split": "confirmatory_holdout" if selected and record["official_split"] == "validation" else record["provisional_pool"] if selected else None,
            "image_sha256": record["image_sha256"],
            "official_split": record["official_split"],
            "r4_eligibility_status": record["eligibility_status"],
            "r4_exclusion_reasons": record["exclusion_reasons"],
            "selection_key_sha256": selection_key(HOLDOUT_NAMESPACE, source["source_image_id"]) if eligible_state and record["official_split"] == "validation" else None,
            "source_image_id": source["source_image_id"],
        })

    selected_by_split = {split: [record for record in final_records if record["final_split"] == split] for split in SPLITS}
    pairwise = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            left_ids = {record["source_image_id"] for record in selected_by_split[left]}
            right_ids = {record["source_image_id"] for record in selected_by_split[right]}
            left_hashes = {record["image_sha256"] for record in selected_by_split[left]}
            right_hashes = {record["image_sha256"] for record in selected_by_split[right]}
            pairwise.append({
                "left": left,
                "right": right,
                "source_image_id_intersection_count": len(left_ids & right_ids),
                "source_image_id_intersection_digest": digest(sorted(left_ids & right_ids)),
                "image_sha256_intersection_count": len(left_hashes & right_hashes),
                "image_sha256_intersection_digest": digest(sorted(left_hashes & right_hashes)),
            })
    require(all(row["source_image_id_intersection_count"] == row["image_sha256_intersection_count"] == 0 for row in pairwise), "split intersection detected")
    historical_ids = {row["source_image_id"] for row in reuse["payload"]["historical_exclusions"]}
    historical_hashes = {record["image_sha256"] for record in r4_alias["payload"]["train_to_historical_byte_overlaps"]}
    selected_source_ids = {record["source_image_id"] for record in final_records}
    selected_hashes = {record["image_sha256"] for record in final_records}
    require(not selected_source_ids & historical_ids and not selected_hashes & historical_hashes, "historical contamination detected")
    operator_masks = {record["annotation_archive_path"] for record in operator_note["payload"]["accessed_mask_records"]}
    require(not operator_masks & {record["annotation_archive_path"] for record in final_records}, "operator-accessed mask selected")

    source_binding_payload = {
        "status": "PASS",
        "authorized_commit": AUTHORIZED_COMMIT,
        "authorized_commit_required_exact_at_freeze": True,
        "current_head_at_freeze": current_head,
        "tracked_worktree_status_at_freeze": "CLEAN",
        "files": bindings,
        "files_digest": digest(bindings),
        "required_s0_r4_terminal": {
            "path": "evidence/nns_vit_patch_depth/s0_r4_20260914/feasibility_decision.json",
            "canonical_digest": r4_decision["canonical_digest"],
            "sha256": bindings_by_path["evidence/nns_vit_patch_depth/s0_r4_20260914/feasibility_decision.json"]["sha256"],
            "data_readiness_status": r4_decision["payload"]["data_readiness_status"],
        },
        "required_s0_r4_byte_alias_audit": {
            "path": "evidence/nns_vit_patch_depth/s0_r4_20260914/byte_alias_audit.json",
            "canonical_digest": r4_alias["canonical_digest"],
            "sha256": bindings_by_path["evidence/nns_vit_patch_depth/s0_r4_20260914/byte_alias_audit.json"]["sha256"],
            "status": r4_alias["payload"]["status"],
        },
    }
    route_payload = {
        "status": "FROZEN",
        "authorization": {
            "approved_stage": "S0-R5_ONLY",
            "approved_route": "PUBLIC_VALIDATION_CONFIRMATORY_HOLDOUT",
            "official_test_route": "ABANDONED_SERVER_ONLY_GROUND_TRUTH",
            "next_stage_if_pass": "S1_PILOT_ONLY",
            "recorded_user_authorization": "approve public-validation confirmatory holdout and freeze S0-R5 only",
        },
        "dataset_label": "ImageNet-S919",
        "confirmatory_label_required": "public-validation confirmatory holdout",
        "forbidden_claim": "ImageNet-S official test result",
        "class_rule": "retain only classes with all ten qualified train-semi records in frozen 2/6/2 pools and at least two qualified public-validation records",
        "class_count": len(accepted_classes),
        "split_counts_per_class": {"pilot": 2, "probe_train": 6, "probe_validation": 2, "confirmatory_holdout": 2},
        "train_split_rule": "inherit S0-R1 metadata reservation exactly after S0-R4 QC and byte-alias exclusions",
        "confirmatory_selection_rule": "within each retained annotation WNID sort qualified public-validation candidates by SHA256(namespace+NUL+source_image_id), tie-break candidate_id, select first two",
        "train_selection_namespace": TRAIN_NAMESPACE,
        "confirmatory_selection_namespace": HOLDOUT_NAMESPACE,
        "holdout_access_policy": [
            "never use for pilot",
            "never use for hyperparameter or endpoint selection",
            "never inspect images or masks for sample selection",
            "open only under a later frozen formal evaluation authorization",
        ],
        "prohibited_operations_observed": {
            "new_data_download": False,
            "rgb_decode": False,
            "model_load": False,
            "gpu_or_cuda": False,
            "probe_training": False,
            "nns_compute": False,
            "occlusion_experiment": False,
            "scientific_outcome_access": False,
            "entered_s1_or_later": False,
        },
    }
    class_payload = {
        "status": "FROZEN",
        "class_count_total": len(class_records),
        "class_count_selected": len(accepted_classes),
        "class_count_excluded": len(class_records) - len(accepted_classes),
        "records": class_records,
        "records_digest": digest(class_records),
    }
    split_payload = {
        "status": "FROZEN",
        "record_count": len(final_records),
        "class_count": len(accepted_classes),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "records": final_records,
        "records_digest": digest(final_records),
        "split_digests": {split: digest(selected_by_split[split]) for split in SPLITS},
    }
    ledger_payload = {
        "status": "COMPLETE",
        "candidate_count": len(ledger),
        "selected_count": len(final_records),
        "disposition_counts": dict(sorted(collections.Counter(record["disposition"] for record in ledger).items())),
        "records": ledger,
        "records_digest": digest(ledger),
    }
    independence_payload = {
        "status": "PASS",
        "pairwise_split_checks": pairwise,
        "selected_source_image_count": len(selected_source_ids),
        "selected_image_sha256_count": len(selected_hashes),
        "historical_source_id_intersection_count": len(selected_source_ids & historical_ids),
        "historical_byte_hash_intersection_count": len(selected_hashes & historical_hashes),
        "operator_accessed_mask_intersection_count": len(operator_masks & {record["annotation_archive_path"] for record in final_records}),
        "all_selected_r4_eligible": all(adjusted_by_id[record["candidate_id"]]["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE" for record in final_records),
        "all_selected_mask_qc_clean": all(not qc_by_id[record["candidate_id"]]["exclusion_reasons"] for record in final_records),
        "all_selected_exact_class_mapping": all(record["target_model_class_index"] is not None for record in final_records),
        "official_split_to_final_split": {"train-semi": ["pilot", "probe_train", "probe_validation"], "validation": ["confirmatory_holdout"], "test": []},
    }
    payloads = {
        "source_binding": source_binding_payload,
        "route_contract": route_payload,
        "class_manifest": class_payload,
        "split_manifest": split_payload,
        "selection_ledger": ledger_payload,
        "independence_audit": independence_payload,
    }
    if implementation is None:
        implementation = file_record(REPO / "scripts/freeze_nns_vit_patch_depth_s0_r5.py")
    predecessor = bindings_by_path["evidence/nns_vit_patch_depth/s0_r4_20260914/feasibility_decision.json"]
    documents = {}
    for name in PRE_TERMINAL_NAMES:
        documents[name] = signed({
            "schema_version": 1,
            "study_id": STUDY_ID,
            "stage_id": STAGE_ID,
            "run_id": RUN_ID,
            "document_type": name,
            "authorized_commit": AUTHORIZED_COMMIT,
            "implementation": implementation,
            "predecessor": predecessor,
            "payload": payloads[name],
        })
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="evidence/nns_vit_patch_depth/s0_r5_20260914")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    require(not (args.dry_run and args.seal), "choose only one mode")
    root = (REPO / args.root).resolve()
    require(REPO / "evidence/nns_vit_patch_depth" in root.parents, "root outside evidence namespace")
    if args.dry_run:
        documents = build_documents()
        split = documents["split_manifest"]["payload"]
        print(json.dumps({"status": split["status"], "class_count": split["class_count"], "record_count": split["record_count"], "split_counts": split["split_counts"], "records_digest": split["records_digest"]}, sort_keys=True))
        return
    if args.seal:
        require(root.is_dir() and not (root / "terminal_receipt.json").exists(), "root missing or already terminal")
        existing_documents = {name: read_signed(str((root / (name + ".json")).relative_to(REPO))) for name in PRE_TERMINAL_NAMES}
        producer_implementation = existing_documents["source_binding"]["implementation"]
        require(all(document["implementation"] == producer_implementation for document in existing_documents.values()), "pre-terminal implementation binding mismatch")
        documents = build_documents(implementation=producer_implementation)
        for name in PRE_TERMINAL_NAMES:
            observed = existing_documents[name]
            require(observed == documents[name], "pre-terminal artifact drift: " + name)
        validation = read_signed(str((root / "validation_report.json").relative_to(REPO)))
        require(validation["payload"]["status"] == "PASS", "validator did not pass")
        expected = {name: file_record(root / (name + ".json")) for name in PRE_TERMINAL_NAMES}
        for item in validation["payload"]["validated_artifacts"]:
            expected_item = expected[item["name"][:-5]]
            require(all(item[key] == expected_item[key] for key in ("path", "bytes", "sha256", "canonical_digest")), "validator artifact binding mismatch")
        artifacts = []
        for name in PRE_TERMINAL_NAMES:
            item = file_record(root / (name + ".json"))
            item["name"] = name + ".json"
            item["canonical_digest"] = documents[name]["canonical_digest"]
            artifacts.append(item)
        validation_record = file_record(root / "validation_report.json")
        validation_record["name"] = "validation_report.json"
        validation_record["canonical_digest"] = validation["canonical_digest"]
        artifacts.append(validation_record)
        terminal = signed({
            "schema_version": 1,
            "study_id": STUDY_ID,
            "stage_id": STAGE_ID,
            "run_id": RUN_ID,
            "document_type": "terminal_receipt",
            "authorized_commit": AUTHORIZED_COMMIT,
            "predecessor": documents["source_binding"]["payload"]["required_s0_r4_terminal"],
            "payload": {
                "status": "PASS",
                "workflow_status": "COMPLETE",
                "data_route": "PUBLIC_VALIDATION_CONFIRMATORY_HOLDOUT",
                "confirmatory_result_label": "public-validation confirmatory holdout",
                "official_test_route": "ABANDONED_BY_EXPLICIT_USER_AUTHORIZATION",
                "class_count": documents["split_manifest"]["payload"]["class_count"],
                "record_count": documents["split_manifest"]["payload"]["record_count"],
                "split_counts": documents["split_manifest"]["payload"]["split_counts"],
                "independence_status": documents["independence_audit"]["payload"]["status"],
                "validator_status": validation["payload"]["status"],
                "producer_implementation": producer_implementation,
                "sealing_implementation": file_record(REPO / "scripts/freeze_nns_vit_patch_depth_s0_r5.py"),
                "artifacts": artifacts,
                "next_authorization": "S1_PILOT_ONLY",
                "s1_started": False,
                "not_performed": ["new data download", "RGB decode", "model load", "GPU/CUDA", "probe training", "NNS", "occlusion", "scientific outcome access", "S1", "S2", "S3", "S4", "S5", "S6"],
            },
        })
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$defs": {}, "oneOf": []}
        for name in PRE_TERMINAL_NAMES:
            schema["$defs"][name] = closed_contract(documents[name])
            schema["oneOf"].append({"$ref": "#/$defs/" + name})
        schema["$defs"]["validation_report"] = closed_contract(validation)
        schema["oneOf"].append({"$ref": "#/$defs/validation_report"})
        schema["$defs"]["terminal_receipt"] = closed_contract(terminal)
        schema["oneOf"].append({"$ref": "#/$defs/terminal_receipt"})
        write_new(root / "artifact_contracts.schema.json", schema)
        validate_closed(terminal, schema["$defs"]["terminal_receipt"])
        write_new(root / "terminal_receipt.json", terminal)
        print("SEALED: S0-R5 PASS; next_authorization S1_PILOT_ONLY")
        return
    documents = build_documents()
    require(not root.exists(), "S0-R5 root already exists")
    root.mkdir(parents=True)
    for name in PRE_TERMINAL_NAMES:
        write_new(root / (name + ".json"), documents[name])
    print("PREPARED: S0-R5 deterministic split artifacts; run independent validator before sealing")


if __name__ == "__main__":
    main()
