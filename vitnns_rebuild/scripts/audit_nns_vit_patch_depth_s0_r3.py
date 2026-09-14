#!/usr/bin/env python3
import argparse
import collections
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
S0_R1 = REPO / "evidence/nns_vit_patch_depth/s0_r1_20260914"
S0_R2 = REPO / "evidence/nns_vit_patch_depth/s0_r2_20260914"
OPERATOR_NOTE = REPO / "evidence/nns_vit_patch_depth/s0_r2_operator_note_20260914/session_access_note.json"
EXTRACTION_RECEIPT = REPO / "assets/imagenet/imagenets919_train_semi/extraction_receipt.json"
MASK_ZIP = REPO / "assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip"
VAL_TAR = REPO / "assets/imagenet/ILSVRC2012_img_val.tar"
STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
NAMES = ("image_mask_qc", "split_feasibility", "feasibility_decision")
MERGED_DESTINATIONS = {"n04355933", "n02808440", "n03832673", "n03773504", "n15075141"}


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


def decode_png(raw, name):
    require(raw.startswith(b"\x89PNG\r\n\x1a\n"), "not PNG: " + name)
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == "PNG" and image.mode == "RGB", "unsupported PNG layout: " + name)
        pixels = np.asarray(image, dtype=np.uint16)
    require(pixels.ndim == 3 and pixels.shape[2] == 3, "unexpected mask array shape: " + name)
    labels = pixels[:, :, 0] + 256 * pixels[:, :, 1]
    values, counts = np.unique(labels, return_counts=True)
    return {
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "label_counts": collections.Counter({int(value): int(count) for value, count in zip(values, counts)}),
        "blue_nonzero": int(np.count_nonzero(pixels[:, :, 2])),
    }


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
        for option in schema["anyOf"]:
            try:
                validate_closed(value, option)
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


def build(root):
    predecessor = read_signed(S0_R2 / "feasibility_decision.json")
    require(predecessor["payload"]["data_readiness_status"] == "BLOCKED_SPLIT", "unexpected predecessor state")
    operator_note = read_signed(OPERATOR_NOTE)
    extraction = read_signed(EXTRACTION_RECEIPT)
    require(extraction["payload"]["status"] == "PASS" and extraction["payload"]["selected_image_count"] == 9190, "train extraction not PASS")
    manifest = read_signed(S0_R1 / "image_annotation_class_manifest.json")
    train_by_path = {record["image_relative_path"]: record for record in extraction["payload"]["records"]}
    mask_inventory = read_signed(S0_R2 / "annotation_package_inventory.json")
    mask_identity = {record["annotation_archive_path"]: record for record in mask_inventory["payload"]["records"]}
    require(len(mask_identity) == 21609, "mask inventory count mismatch")
    accessed_masks = {record["annotation_archive_path"] for record in operator_note["payload"]["accessed_mask_records"]}
    qc_records = []
    reason_counts = collections.Counter()
    pool_counts = collections.Counter()
    per_class = collections.defaultdict(lambda: collections.Counter())
    with zipfile.ZipFile(MASK_ZIP) as archive:
        for candidate in manifest["payload"]["records"]:
            if candidate["official_split"] not in ("train-semi", "validation"):
                continue
            prefix = "train-semi-segmentation" if candidate["official_split"] == "train-semi" else "validation-segmentation"
            mask_path = "ImageNetS919/%s/%s" % (prefix, candidate["annotation_relative_path"])
            identity = mask_identity[mask_path]
            raw = archive.read(mask_path)
            require(len(raw) == identity["bytes"] and hashlib.sha256(raw).hexdigest() == identity["sha256"], "mask identity mismatch")
            decoded = decode_png(raw, mask_path)
            if candidate["official_split"] == "train-semi":
                image = train_by_path[candidate["official_source_relative_path"]]
                image_header = image["image_header"]
                image_sha256 = image["sha256"]
                image_bytes = image["bytes"]
                image_local_path = image["local_path"]
            else:
                image_header = candidate["image_header"]
                image_sha256 = candidate["image_sha256"]
                image_bytes = candidate["image_bytes"]
                image_local_path = None
            reasons = []
            labels = decoded["label_counts"]
            target = candidate["annotation_label_id"]
            if (decoded["width"], decoded["height"]) != (image_header["width"], image_header["height"]):
                reasons.append("IMAGE_MASK_DIMENSION_MISMATCH")
            if image_header["exif_state"] == "MALFORMED_OR_UNSUPPORTED":
                reasons.append("IMAGE_EXIF_UNRESOLVED")
            if image_header["exif_orientation"] not in (None, 1):
                reasons.append("IMAGE_EXIF_TRANSFORM_REQUIRED")
            if decoded["blue_nonzero"]:
                reasons.append("MASK_BLUE_CHANNEL_NONZERO")
            if any(label not in {0, 1000} and not 1 <= label <= 919 for label in labels):
                reasons.append("MASK_LABEL_OUT_OF_RANGE")
            if labels.get(target, 0) == 0:
                reasons.append("TARGET_LABEL_ABSENT")
            if candidate["annotation_wnid"] in MERGED_DESTINATIONS or candidate["exact_target_class_index"] is None:
                reasons.append("MERGED_OR_NONEXACT_TARGET_CLASS")
            if "HISTORICAL_SOURCE_OR_BYTE_ALIAS" in candidate["exclusion_reasons"]:
                reasons.append("HISTORICAL_SOURCE_OR_BYTE_ALIAS")
            if "OFFICIAL_SPLIT_BYTE_ALIAS_QUARANTINED" in candidate["exclusion_reasons"]:
                reasons.append("OFFICIAL_SPLIT_BYTE_ALIAS_QUARANTINED")
            if mask_path in accessed_masks:
                reasons.append("OPERATOR_MASK_PIXEL_ACCESSED")
            reasons = sorted(set(reasons))
            eligible = not reasons
            for reason in reasons:
                reason_counts[reason] += 1
            if eligible:
                pool_counts[candidate["provisional_pool"]] += 1
                per_class[candidate["annotation_wnid"]][candidate["provisional_pool"]] += 1
            qc_records.append({
                "annotation_label_id": target,
                "annotation_sha256": identity["sha256"],
                "annotation_wnid": candidate["annotation_wnid"],
                "blue_nonzero_pixels": decoded["blue_nonzero"],
                "candidate_id": candidate["candidate_id"],
                "eligibility_status": "ELIGIBLE_PRE_SPLIT_FREEZE" if eligible else "EXCLUDED",
                "exclusion_reasons": reasons,
                "image_bytes": image_bytes,
                "image_height": image_header["height"],
                "image_local_path": image_local_path,
                "image_sha256": image_sha256,
                "image_width": image_header["width"],
                "mask_archive_path": mask_path,
                "mask_height": decoded["height"],
                "mask_label_counts": [{"label": label, "pixels": count} for label, count in sorted(labels.items())],
                "mask_width": decoded["width"],
                "official_split": candidate["official_split"],
                "provisional_pool": candidate["provisional_pool"],
                "target_pixels": labels.get(target, 0),
                "total_pixels": decoded["width"] * decoded["height"],
                "void_pixels": labels.get(1000, 0),
            })
    qc_records.sort(key=lambda record: record["candidate_id"])
    require(len(qc_records) == 21609, "QC record count mismatch")
    class_rows = []
    for wnid in sorted({record["annotation_wnid"] for record in qc_records}):
        counts = per_class[wnid]
        class_rows.append({
            "annotation_wnid": wnid,
            "pilot": counts["pilot"],
            "probe_train": counts["probe_train"],
            "probe_validation": counts["probe_validation"],
            "official_validation_unassigned": counts["official_validation_unassigned"],
        })
    complete_train_classes = sum(row["pilot"] == 2 and row["probe_train"] == 6 and row["probe_validation"] == 2 for row in class_rows)
    test_ready_classes = sum(row["official_validation_unassigned"] >= 1 for row in class_rows)
    predecessor_record = record(S0_R2 / "feasibility_decision.json")
    predecessor_record["canonical_digest"] = predecessor["canonical_digest"]
    inherited = [
        predecessor_record,
        {**record(OPERATOR_NOTE), "canonical_digest": operator_note["canonical_digest"]},
        {**record(EXTRACTION_RECEIPT), "canonical_digest": extraction["canonical_digest"]},
        {**record(S0_R1 / "image_annotation_class_manifest.json"), "canonical_digest": manifest["canonical_digest"]},
        {**record(S0_R2 / "annotation_package_inventory.json"), "canonical_digest": mask_inventory["canonical_digest"]},
    ]
    qc_payload = {
        "status": "COMPLETE",
        "record_count": len(qc_records),
        "records": qc_records,
        "records_digest": digest(qc_records),
        "eligible_count": sum(record["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE" for record in qc_records),
        "reason_counts": dict(sorted(reason_counts.items())),
        "eligible_provisional_pool_counts": dict(sorted(pool_counts.items())),
        "access_boundary": {
            "mask_pixel_decode_count": len(qc_records),
            "rgb_pixel_decode_count": 0,
            "model_load_count": 0,
            "gpu_access_count": 0,
            "scientific_outcomes_used_for_selection": False,
        },
    }
    split_payload = {
        "status": "BLOCKED_SPLIT_ROUTE_NOT_AUTHORIZED",
        "eligible_provisional_pool_counts": dict(sorted(pool_counts.items())),
        "per_class": class_rows,
        "per_class_digest": digest(class_rows),
        "complete_train_2_6_2_class_count": complete_train_classes,
        "validation_classes_with_at_least_one_eligible_image": test_ready_classes,
        "official_train_semi_route": "TRAIN_ORIGINALS_AND_MASK_QC_BOUND",
        "official_test_route": "BLOCKED_SERVER_ONLY_GROUND_TRUTH",
        "prospective_route_available": "use qualified train-semi 2/6/2 pools for pilot/probe train/probe validation and freeze a disjoint qualified public-validation subset as confirmatory test",
        "prospective_route_status": "REQUIRES_EXPLICIT_APPROVAL_BEFORE_SPLIT_FREEZE_OR_S1",
        "benchmark_boundary": "this independent NNS analysis would not claim compliance with ImageNet-S leaderboard protocols that prohibit val/test annotation use in some tracks",
        "statistical_power_status": "NOT_EVALUATED_S1_PILOT_REQUIRED",
    }
    decision_payload = {
        "audit_workflow_status": "COMPLETE",
        "data_readiness_status": "BLOCKED_SPLIT",
        "adopted_dataset": "ImageNet-S919",
        "s1_started": False,
        "can_request_S1_now": False,
        "gates": [
            {"gate": "predecessor_identity", "status": "PASS", "detail": "S0-R2 and operator access note verified"},
            {"gate": "license", "status": "PASS_BY_USER_ATTESTATION", "detail": "authorization inherited from S0-R2"},
            {"gate": "train_archive", "status": "PASS", "detail": "four parts match fixed SHA-256 values and concatenated tar matches standard ILSVRC2012 MD5"},
            {"gate": "train_originals", "status": "PASS", "detail": "9190 official train-semi originals extracted and byte-bound without RGB decode"},
            {"gate": "mask_qc", "status": "PASS_WITH_TYPED_EXCLUSIONS", "detail": "%d masks decoded for label and geometry checks; exclusions retained" % len(qc_records)},
            {"gate": "split", "status": "BLOCKED_SPLIT", "detail": "official test truth remains server-only and public-validation holdout use has not been explicitly approved"},
        ],
        "eligible_count_pre_split_freeze": qc_payload["eligible_count"],
        "complete_train_2_6_2_class_count": complete_train_classes,
        "validation_classes_with_at_least_one_eligible_image": test_ready_classes,
        "next_authorization": "APPROVE_PROSPECTIVE_PUBLIC_VALIDATION_CONFIRMATORY_HOLDOUT_OR_SUPPLY_TEST_TRUTH",
        "not_performed": ["RGB decode", "model load", "GPU", "probe training", "attribution", "Y4 changes", "S1", "formal split freeze"],
        "artifacts": [],
    }
    payloads = {"image_mask_qc": qc_payload, "split_feasibility": split_payload, "feasibility_decision": decision_payload}
    docs = {}
    for name in NAMES:
        docs[name] = signed({
            "schema_version": 1,
            "study_id": STUDY_ID,
            "stage_id": "S0-R3",
            "run_id": root.name,
            "document_type": name,
            "predecessor": predecessor_record,
            "inherited_files": inherited,
            "implementation": record(REPO / "scripts/audit_nns_vit_patch_depth_s0_r3.py"),
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
    parser.add_argument("--root", default="evidence/nns_vit_patch_depth/s0_r3_20260914")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = (REPO / args.root).resolve()
    require(REPO / "evidence/nns_vit_patch_depth" in root.parents, "root outside evidence namespace")
    docs = build(root)
    require(not {"torch", "torchvision", "timm", "cv2", "scipy"}.intersection(sys.modules), "forbidden model or GPU module loaded")
    if args.verify:
        schema = json.loads((root / "artifact_contracts.schema.json").read_text())
        for name in NAMES:
            old = read_signed(root / (name + ".json"))
            validate_closed(old, schema["$defs"][name])
            require(old == docs[name], "source-recomputed artifact mismatch: " + name)
        print("PASS: S0-R3 artifacts equal source-recomputed image/mask QC and split decision")
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
    print("SEALED: S0-R3 train originals PASS; readiness BLOCKED_SPLIT")
    print(json.dumps({key: docs["feasibility_decision"]["payload"][key] for key in ("eligible_count_pre_split_freeze", "complete_train_2_6_2_class_count", "validation_classes_with_at_least_one_eligible_image", "data_readiness_status", "next_authorization")}, sort_keys=True))


if __name__ == "__main__":
    main()
