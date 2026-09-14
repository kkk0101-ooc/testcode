#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import tarfile


REPO = Path(__file__).resolve().parents[1]
S0_R1 = REPO / "evidence/nns_vit_patch_depth/s0_r1_20260914"
S0_R3 = REPO / "evidence/nns_vit_patch_depth/s0_r3_20260914"
EXTRACTION_RECEIPT = REPO / "assets/imagenet/imagenets919_train_semi/extraction_receipt.json"
VAL_TAR = REPO / "assets/imagenet/ILSVRC2012_img_val.tar"
NAMES = ("train_original_binding", "byte_alias_audit", "split_feasibility", "feasibility_decision")


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


def read_signed(path):
    value = json.loads(path.read_text())
    unsigned = {key: item for key, item in value.items() if key != "canonical_digest"}
    require(value["canonical_digest"] == digest(unsigned), "canonical digest mismatch: " + str(path))
    return value


def record(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file changed while hashing")
    return {"path": str(path.relative_to(REPO)), "bytes": before.st_size, "sha256": h.hexdigest()}


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
    types = {"object": dict, "array": list, "null": type(None), "boolean": bool, "integer": int, "number": float, "string": str}
    kind = schema["type"]
    require(type(value) is types[kind], "wrong schema type")
    if kind == "object":
        require(set(value) == set(schema["required"]) == set(schema["properties"]), "unclosed object")
        require(schema["additionalProperties"] is False, "open object schema")
        for key, item in value.items():
            validate_closed(item, schema["properties"][key])
    elif kind == "array":
        for item in value:
            validate_closed(item, schema["items"])


def historical_hashes(reuse):
    wanted = {row["source_image_id"].rsplit("/", 1)[-1] for row in reuse["payload"]["historical_exclusions"]}
    by_hash = collections.defaultdict(list)
    with tarfile.open(VAL_TAR, "r:") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            stream = archive.extractfile(member)
            require(stream is not None, "cannot read historical source")
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
            by_hash[h.hexdigest()].append(member.name)
    require(sum(len(names) for names in by_hash.values()) == len(wanted) == 1101, "historical source coverage mismatch")
    return by_hash


def build(root):
    predecessor = read_signed(S0_R3 / "feasibility_decision.json")
    require(predecessor["payload"]["data_readiness_status"] == "BLOCKED_SPLIT", "unexpected S0-R3 state")
    qc = read_signed(S0_R3 / "image_mask_qc.json")
    extraction = read_signed(EXTRACTION_RECEIPT)
    manifest = read_signed(S0_R1 / "image_annotation_class_manifest.json")
    reuse = read_signed(S0_R1 / "reuse_and_exclusion_manifest.json")
    train_hash = {row["image_relative_path"]: row["sha256"] for row in extraction["payload"]["records"]}
    by_hash = collections.defaultdict(list)
    for row in manifest["payload"]["records"]:
        image_hash = train_hash.get(row["official_source_relative_path"]) if row["official_split"] == "train-semi" else row["image_sha256"]
        if image_hash:
            by_hash[image_hash].append({
                "annotation_wnid": row["annotation_wnid"],
                "candidate_id": row["candidate_id"],
                "official_split": row["official_split"],
                "provisional_pool": row["provisional_pool"],
                "source_image_id": row["source_image_id"],
            })
    cross_groups = []
    for image_hash, rows in sorted(by_hash.items()):
        splits = sorted({row["official_split"] for row in rows})
        if len(splits) > 1:
            cross_groups.append({"image_sha256": image_hash, "official_splits": splits, "records": sorted(rows, key=lambda row: row["candidate_id"])})
    require(len(cross_groups) == 9, "unexpected cross-split byte group count")
    train_cross_groups = [group for group in cross_groups if any(row["official_split"] == "train-semi" for row in group["records"])]
    require(len(train_cross_groups) == 4, "unexpected train cross-split byte group count")
    affected_ids = {row["candidate_id"] for group in train_cross_groups for row in group["records"] if row["official_split"] in ("train-semi", "validation")}
    require(len(affected_ids) == 5, "unexpected eligible-scope affected record count")
    old_hashes = historical_hashes(reuse)
    historical_overlaps = []
    for image_hash, train_rows in sorted((key, value) for key, value in by_hash.items() if any(row["official_split"] == "train-semi" for row in value)):
        if image_hash in old_hashes:
            historical_overlaps.append({
                "image_sha256": image_hash,
                "historical_validation_members": sorted(old_hashes[image_hash]),
                "train_records": sorted([row for row in train_rows if row["official_split"] == "train-semi"], key=lambda row: row["candidate_id"]),
            })
    require(len(historical_overlaps) == 1, "unexpected train-to-historical byte overlap count")
    adjusted_records = []
    pool_counts = collections.Counter()
    per_class = collections.defaultdict(lambda: collections.Counter())
    newly_excluded = []
    for row in qc["payload"]["records"]:
        reasons = list(row["exclusion_reasons"])
        if row["candidate_id"] in affected_ids:
            reasons.append("TRAIN_BOUND_CROSS_PURPOSE_BYTE_ALIAS")
        reasons = sorted(set(reasons))
        eligible = not reasons
        adjusted = {
            "annotation_wnid": row["annotation_wnid"],
            "candidate_id": row["candidate_id"],
            "eligibility_status": "ELIGIBLE_PRE_SPLIT_FREEZE" if eligible else "EXCLUDED",
            "exclusion_reasons": reasons,
            "image_sha256": row["image_sha256"],
            "official_split": row["official_split"],
            "provisional_pool": row["provisional_pool"],
        }
        adjusted_records.append(adjusted)
        if eligible:
            pool_counts[row["provisional_pool"]] += 1
            per_class[row["annotation_wnid"]][row["provisional_pool"]] += 1
        if row["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE" and not eligible:
            newly_excluded.append(adjusted)
    adjusted_records.sort(key=lambda row: row["candidate_id"])
    newly_excluded.sort(key=lambda row: row["candidate_id"])
    require(len(newly_excluded) == 5, "new exclusion count mismatch")
    class_rows = []
    for wnid in sorted({row["annotation_wnid"] for row in adjusted_records}):
        counts = per_class[wnid]
        class_rows.append({
            "annotation_wnid": wnid,
            "pilot": counts["pilot"],
            "probe_train": counts["probe_train"],
            "probe_validation": counts["probe_validation"],
            "official_validation_unassigned": counts["official_validation_unassigned"],
        })
    complete_train = [row for row in class_rows if (row["pilot"], row["probe_train"], row["probe_validation"]) == (2, 6, 2)]
    predecessor_record = record(S0_R3 / "feasibility_decision.json")
    predecessor_record["canonical_digest"] = predecessor["canonical_digest"]
    inherited = [
        predecessor_record,
        {**record(S0_R3 / "image_mask_qc.json"), "canonical_digest": qc["canonical_digest"]},
        {**record(EXTRACTION_RECEIPT), "canonical_digest": extraction["canonical_digest"]},
        {**record(S0_R1 / "image_annotation_class_manifest.json"), "canonical_digest": manifest["canonical_digest"]},
        {**record(S0_R1 / "reuse_and_exclusion_manifest.json"), "canonical_digest": reuse["canonical_digest"]},
    ]
    payloads = {
        "train_original_binding": {
            "status": "PASS",
            "source_receipt": inherited[2],
            "whole_archive": extraction["payload"]["whole_archive"],
            "parts": extraction["payload"]["parts"],
            "selected_image_count": extraction["payload"]["selected_image_count"],
            "records": extraction["payload"]["records"],
            "records_digest": extraction["payload"]["records_digest"],
            "access_boundary": extraction["payload"]["access_boundary"],
        },
        "byte_alias_audit": {
            "status": "PASS_WITH_QUARANTINE",
            "all_cross_split_byte_group_count": len(cross_groups),
            "all_cross_split_byte_groups": cross_groups,
            "train_cross_split_byte_group_count": len(train_cross_groups),
            "train_cross_split_byte_groups": train_cross_groups,
            "historical_source_count": 1101,
            "historical_source_hash_group_count": len(old_hashes),
            "train_to_historical_byte_overlap_count": len(historical_overlaps),
            "train_to_historical_byte_overlaps": historical_overlaps,
            "newly_excluded_records": newly_excluded,
            "newly_excluded_records_digest": digest(newly_excluded),
        },
        "split_feasibility": {
            "status": "BLOCKED_SPLIT_ROUTE_NOT_AUTHORIZED",
            "eligible_count_pre_split_freeze": sum(row["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE" for row in adjusted_records),
            "eligible_provisional_pool_counts": dict(sorted(pool_counts.items())),
            "complete_train_2_6_2_class_count": len(complete_train),
            "full_train_classes_with_validation_ge_1": sum(row["official_validation_unassigned"] >= 1 for row in complete_train),
            "full_train_classes_with_validation_ge_2": sum(row["official_validation_unassigned"] >= 2 for row in complete_train),
            "validation_classes_with_at_least_one_eligible_image": sum(row["official_validation_unassigned"] >= 1 for row in class_rows),
            "per_class": class_rows,
            "per_class_digest": digest(class_rows),
            "adjusted_records": adjusted_records,
            "adjusted_records_digest": digest(adjusted_records),
            "prospective_route": "train-semi provisional 2/6/2 pools plus an outcome-blind disjoint public-validation confirmatory holdout",
            "prospective_route_status": "REQUIRES_EXPLICIT_APPROVAL_BEFORE_SPLIT_FREEZE_OR_S1",
            "official_test_route": "BLOCKED_SERVER_ONLY_GROUND_TRUTH",
            "benchmark_boundary": "independent NNS analysis only; no claim of compliance with ImageNet-S leaderboard protocols that restrict val/test annotations",
        },
        "feasibility_decision": {
            "audit_workflow_status": "COMPLETE",
            "data_readiness_status": "BLOCKED_SPLIT",
            "adopted_dataset": "ImageNet-S919",
            "s1_started": False,
            "can_request_S1_now": False,
            "gates": [
                {"gate": "predecessor_identity", "status": "PASS", "detail": "S0-R3 source-recomputed QC verified"},
                {"gate": "train_originals", "status": "PASS", "detail": "9190 train-semi originals byte-bound"},
                {"gate": "source_byte_isolation", "status": "PASS_WITH_QUARANTINE", "detail": "four train cross-split byte groups found; five train/validation eligible-scope records quarantined"},
                {"gate": "train_pool", "status": "PARTIAL", "detail": "%d classes retain complete 2/6/2 train pools after all typed exclusions" % len(complete_train)},
                {"gate": "confirmatory_pool", "status": "AWAITING_APPROVAL", "detail": "%d complete-train classes have at least two eligible public-validation candidates" % sum(row["official_validation_unassigned"] >= 2 for row in complete_train)},
                {"gate": "official_test_truth", "status": "BLOCKED_DATA", "detail": "official test ground truth remains server-only"},
            ],
            "eligible_count_pre_split_freeze": sum(row["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE" for row in adjusted_records),
            "complete_train_2_6_2_class_count": len(complete_train),
            "full_train_classes_with_validation_ge_2": sum(row["official_validation_unassigned"] >= 2 for row in complete_train),
            "next_authorization": "APPROVE_PROSPECTIVE_PUBLIC_VALIDATION_CONFIRMATORY_HOLDOUT_OR_SUPPLY_TEST_TRUTH",
            "not_performed": ["RGB decode", "model load", "GPU", "probe training", "attribution", "Y4 changes", "S1", "formal split freeze"],
            "artifacts": [],
        },
    }
    docs = {}
    for name in NAMES:
        docs[name] = signed({
            "schema_version": 1,
            "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS",
            "stage_id": "S0-R4",
            "run_id": root.name,
            "document_type": name,
            "predecessor": predecessor_record,
            "inherited_files": inherited,
            "implementation": record(REPO / "scripts/audit_nns_vit_patch_depth_s0_r4.py"),
            "payload": payloads[name],
        })
    docs["feasibility_decision"]["payload"]["artifacts"] = [
        {"name": name + ".json", "sha256": hashlib.sha256(serialize(docs[name])).hexdigest(), "canonical_digest": docs[name]["canonical_digest"]}
        for name in NAMES[:-1]
    ]
    docs["feasibility_decision"]["canonical_digest"] = digest({key: value for key, value in docs["feasibility_decision"].items() if key != "canonical_digest"})
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="evidence/nns_vit_patch_depth/s0_r4_20260914")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = (REPO / args.root).resolve()
    require(REPO / "evidence/nns_vit_patch_depth" in root.parents, "root outside evidence namespace")
    docs = build(root)
    if args.verify:
        schema = json.loads((root / "artifact_contracts.schema.json").read_text())
        for name in NAMES:
            old = read_signed(root / (name + ".json"))
            validate_closed(old, schema["$defs"][name])
            require(old == docs[name], "source-recomputed artifact mismatch: " + name)
        print("PASS: S0-R4 artifacts equal source-recomputed byte-alias audit and adjusted split feasibility")
        return
    require(not root.exists(), "terminal root already exists")
    root.mkdir(parents=True)
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$defs": {name: closed_contract(docs[name]) for name in NAMES}, "oneOf": [{"$ref": "#/$defs/" + name} for name in NAMES]}
    with (root / "artifact_contracts.schema.json").open("xb") as stream:
        stream.write(serialize(schema))
    for name in NAMES:
        validate_closed(docs[name], schema["$defs"][name])
        with (root / (name + ".json")).open("xb") as stream:
            stream.write(serialize(docs[name]))
    print("SEALED: S0-R4 source-byte isolation PASS_WITH_QUARANTINE; readiness BLOCKED_SPLIT")
    print(json.dumps({key: docs["feasibility_decision"]["payload"][key] for key in ("eligible_count_pre_split_freeze", "complete_train_2_6_2_class_count", "full_train_classes_with_validation_ge_2", "data_readiness_status", "next_authorization")}, sort_keys=True))


if __name__ == "__main__":
    main()
