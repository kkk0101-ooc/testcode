#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
S0_R1 = REPO / "evidence/nns_vit_patch_depth/s0_r1_20260914"
ZIP_REL = "assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip"
ZIP_EXPECTED = {
    "name": "ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip",
    "bytes": 61281494,
    "sha256": "31de0324c3a36e0817eeab0d704a7241b77ae5e725aeab103fd60c5b78dadf5e",
}
PIN = "b2312ba7396151f883362bdc86d71469aa81066e"
STAGE_ID = "S0-R2"
STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
NAMES = (
    "authorization_receipt",
    "annotation_package_inventory",
    "split_feasibility",
    "feasibility_decision",
)
S0_R1_NAMES = (
    "asset_inventory",
    "license_and_source_review",
    "annotation_contract",
    "image_annotation_class_manifest",
    "reuse_and_exclusion_manifest",
    "split_feasibility",
    "annotation_qc_receipt",
    "feasibility_decision",
)
MERGED = {
    "n04356056": "n04355933",
    "n04493381": "n02808440",
    "n03642806": "n03832673",
    "n04008634": "n03773504",
    "n03887697": "n15075141",
}
PUBLIC_SPLIT_PROFILES = (
    {"profile": "PVH_2_6_2_2", "pilot": 2, "probe_train": 6, "probe_validation": 2, "confirmatory_test": 2},
    {"profile": "PVH_2_4_2_2", "pilot": 2, "probe_train": 4, "probe_validation": 2, "confirmatory_test": 2},
    {"profile": "PVH_2_3_2_1", "pilot": 2, "probe_train": 3, "probe_validation": 2, "confirmatory_test": 1},
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


def read_signed(path):
    value = json.loads(path.read_text())
    unsigned = {k: v for k, v in value.items() if k != "canonical_digest"}
    require(value["canonical_digest"] == digest(unsigned), "canonical digest mismatch: " + str(path))
    return value


def record(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file changed while hashing")
    return {"path": os.path.relpath(str(path), str(REPO)), "bytes": before.st_size, "sha256": h.hexdigest()}


def git(*args):
    return subprocess.check_output(["git", "-C", str(WORKSPACE), *args], text=True).strip()


def safe_relative(value):
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value, "unsafe relative path: " + value)
    return path


def png_header(raw, name):
    require(raw.startswith(b"\x89PNG\r\n\x1a\n"), "not PNG: " + name)
    require(len(raw) >= 33, "short PNG header: " + name)
    require(struct.unpack(">I", raw[8:12])[0] == 13 and raw[12:16] == b"IHDR", "missing IHDR: " + name)
    width, height = struct.unpack(">II", raw[16:24])
    return {
        "width": width,
        "height": height,
        "bit_depth": raw[24],
        "color_type": raw[25],
        "compression": raw[26],
        "filter": raw[27],
        "interlace": raw[28],
    }


def closed_contract(value):
    if isinstance(value, dict):
        return {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(value),
            "properties": {k: closed_contract(v) for k, v in value.items()},
        }
    if isinstance(value, list):
        schemas = []
        for item in value:
            schema = closed_contract(item)
            if schema not in schemas:
                schemas.append(schema)
        if not schemas:
            return {"type": "array", "items": False}
        return {"type": "array", "items": schemas[0] if len(schemas) == 1 else {"anyOf": schemas}}
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
        errors = []
        for choice in schema["anyOf"]:
            try:
                validate_closed(value, choice)
                return
            except ValueError as error:
                errors.append(str(error))
        raise ValueError("no matching closed variant: " + "; ".join(errors[:3]))
    kind = schema["type"]
    py_types = {
        "object": dict,
        "array": list,
        "null": type(None),
        "boolean": bool,
        "integer": int,
        "number": float,
        "string": str,
    }
    require(type(value) is py_types[kind], "wrong schema type: " + kind)
    if kind == "object":
        require(set(value) == set(schema["required"]) == set(schema["properties"]), "unclosed object")
        require(schema["additionalProperties"] is False, "object schema is not closed")
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


def load_predecessor():
    docs = {name: read_signed(S0_R1 / (name + ".json")) for name in S0_R1_NAMES}
    decision = docs["feasibility_decision"]["payload"]
    require(decision["data_readiness_status"] == "BLOCKED_LICENSE", "S0-R1 predecessor is not the expected license block")
    require(decision["s1_started"] is False, "S0-R1 unexpectedly started S1")
    inherited = []
    for name in S0_R1_NAMES:
        path = S0_R1 / (name + ".json")
        item = record(path)
        item["canonical_digest"] = docs[name]["canonical_digest"]
        inherited.append(item)
    return docs, inherited


def scan_zip(expected_paths):
    zip_record = record(REPO / ZIP_REL)
    for key, expected in ZIP_EXPECTED.items():
        observed = Path(zip_record["path"]).name if key == "name" else zip_record[key]
        require(observed == expected, "annotation zip %s mismatch" % key)
    records = []
    split_counts = collections.Counter()
    dimension_counts = collections.Counter()
    format_counts = collections.Counter()
    matched = set()
    unexpected_files = []
    crc_failures = []
    with zipfile.ZipFile(REPO / ZIP_REL) as archive:
        bad_member = archive.testzip()
        require(bad_member is None, "zip CRC test failed at " + str(bad_member))
        infos = archive.infolist()
        for info in infos:
            safe_relative(info.filename)
            if info.is_dir():
                continue
            if not info.filename.endswith(".png"):
                unexpected_files.append(info.filename)
                continue
            raw = archive.read(info)
            crc = "%08x" % (info.CRC & 0xFFFFFFFF)
            observed_crc = "%08x" % (zipfile.crc32(raw) & 0xFFFFFFFF)
            if observed_crc != crc:
                crc_failures.append(info.filename)
            header = png_header(raw, info.filename)
            format_key = "bit%d_color%d_interlace%d" % (header["bit_depth"], header["color_type"], header["interlace"])
            format_counts[format_key] += 1
            dimension_counts["%dx%d" % (header["width"], header["height"])] += 1
            parts = PurePosixPath(info.filename).parts
            require(parts[0] == "ImageNetS919" and len(parts) == 4, "unexpected mask path: " + info.filename)
            if parts[1] == "validation-segmentation":
                split = "validation"
            elif parts[1] == "train-semi-segmentation":
                split = "train-semi"
            else:
                split = "other"
            split_counts[split] += 1
            expected = expected_paths.get(info.filename)
            if expected is not None:
                matched.add(info.filename)
            records.append({
                "annotation_archive_path": info.filename,
                "annotation_wnid": parts[2],
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": crc,
                "expected_by_official_list": expected is not None,
                "image_filename": parts[3][:-4],
                "png": header,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "split": split,
            })
    records.sort(key=lambda item: item["annotation_archive_path"])
    missing = sorted(set(expected_paths) - matched)
    extra = sorted(r["annotation_archive_path"] for r in records if not r["expected_by_official_list"])
    return {
        "zip_record": zip_record,
        "zip_crc_test": "PASS",
        "zip_entry_count": len(infos),
        "mask_file_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "dimension_count": len(dimension_counts),
        "top_dimensions": [{"dimension": k, "count": v} for k, v in dimension_counts.most_common(20)],
        "format_counts": dict(sorted(format_counts.items())),
        "unexpected_non_png_files": unexpected_files,
        "crc_failure_count": len(crc_failures),
        "crc_failures": crc_failures[:20],
        "expected_mask_count": len(expected_paths),
        "matched_expected_mask_count": len(matched),
        "missing_expected_mask_count": len(missing),
        "missing_expected_masks": missing[:100],
        "extra_mask_count": len(extra),
        "extra_masks": extra[:100],
        "records": records,
        "records_digest": digest(records),
    }


def active_train_original_candidates(candidate_records):
    wanted = {PurePosixPath(r["official_source_relative_path"]).name for r in candidate_records if r["official_split"] == "train-semi"}
    found = []
    roots = [REPO / "assets"]
    for root in roots:
        if not root.exists():
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs.sort()
            for name in sorted(files):
                if name in wanted and re.fullmatch(r"n\d{8}_\d+\.JPEG", name):
                    path = Path(directory) / name
                    found.append(record(path))
    found.sort(key=lambda item: item["path"])
    return found


def split_candidate_records(candidate_records, mask_records):
    mask_paths = {r["annotation_archive_path"] for r in mask_records}
    official_counts = collections.Counter(r["official_split"] for r in candidate_records)
    revised_reason_counts = collections.Counter()
    qualified_public_validation = []
    public_per_class = collections.defaultdict(lambda: {
        "annotation_wnid": None,
        "total_validation": 0,
        "exact_validation": 0,
        "eligible_public_validation": 0,
        "historical_or_byte_quarantined": 0,
    })
    for row in candidate_records:
        reasons = set(row["exclusion_reasons"])
        if "ANNOTATION_LICENSE_UNRESOLVED" in reasons:
            reasons.remove("ANNOTATION_LICENSE_UNRESOLVED")
        if row["official_split"] in ("validation", "train-semi"):
            prefix = "validation-segmentation" if row["official_split"] == "validation" else "train-semi-segmentation"
            mask_path = "ImageNetS919/%s/%s" % (prefix, row["annotation_relative_path"])
            if mask_path in mask_paths and "MASK_NOT_ACQUIRED_OR_QC_VERIFIED" in reasons:
                reasons.remove("MASK_NOT_ACQUIRED_OR_QC_VERIFIED")
            elif mask_path not in mask_paths:
                reasons.add("MASK_PACKAGE_MISSING")
        if row["official_split"] == "validation":
            cls = public_per_class[row["annotation_wnid"]]
            cls["annotation_wnid"] = row["annotation_wnid"]
            cls["total_validation"] += 1
            if row["exact_target_class_index"] is not None:
                cls["exact_validation"] += 1
            quarantined = bool({"HISTORICAL_SOURCE_OR_BYTE_ALIAS", "OFFICIAL_SPLIT_BYTE_ALIAS_QUARANTINED"} & reasons)
            if quarantined:
                cls["historical_or_byte_quarantined"] += 1
            if row["exact_target_class_index"] is not None and row["image_sha256"] is not None and not quarantined and "MASK_PACKAGE_MISSING" not in reasons:
                qualified_public_validation.append(row)
                cls["eligible_public_validation"] += 1
        for reason in sorted(reasons):
            revised_reason_counts[reason] += 1
    class_rows = sorted(public_per_class.values(), key=lambda item: item["annotation_wnid"])
    hist = collections.Counter(item["eligible_public_validation"] for item in class_rows)
    profile_summaries = []
    for profile in PUBLIC_SPLIT_PROFILES:
        needed = profile["pilot"] + profile["probe_train"] + profile["probe_validation"] + profile["confirmatory_test"]
        classes = [row for row in class_rows if row["eligible_public_validation"] >= needed]
        profile_summaries.append({
            "profile": profile["profile"],
            "required_public_validation_images_per_class": needed,
            "class_count_meeting_requirement": len(classes),
            "candidate_image_count_under_profile": len(classes) * needed,
            "allocation_status": "NOT_CREATED_REQUIRES_PROSPECTIVE_APPROVAL",
            "counts_per_class": {k: profile[k] for k in ("pilot", "probe_train", "probe_validation", "confirmatory_test")},
        })
    return {
        "class_rows": class_rows,
        "class_rows_digest": digest(class_rows),
        "eligible_public_validation_count": len(qualified_public_validation),
        "eligible_public_validation_digest": digest([r["candidate_id"] for r in qualified_public_validation]),
        "eligible_count_distribution": [{"eligible_count": k, "class_count": v} for k, v in sorted(hist.items())],
        "official_counts": dict(sorted(official_counts.items())),
        "profile_summaries": profile_summaries,
        "revised_reason_counts": dict(sorted(revised_reason_counts.items())),
    }


def build_documents(root):
    predecessor, inherited = load_predecessor()
    candidate_doc = predecessor["image_annotation_class_manifest"]
    candidate_records = candidate_doc["payload"]["records"]
    expected_paths = {}
    for row in candidate_records:
        if row["official_split"] == "validation":
            path = "ImageNetS919/validation-segmentation/" + row["annotation_relative_path"]
        elif row["official_split"] == "train-semi":
            path = "ImageNetS919/train-semi-segmentation/" + row["annotation_relative_path"]
        else:
            continue
        safe_relative(path)
        expected_paths[path] = row
    require(len(expected_paths) == 21609, "unexpected public mask expectation count")
    zip_scan = scan_zip(expected_paths)
    train_found = active_train_original_candidates(candidate_records)
    split_summary = split_candidate_records(candidate_records, zip_scan["records"])
    source_refs = [
        "evidence/nns_vit_patch_depth/s0_r1_20260914/source_evidence/imagenets_README_md.json",
        "evidence/nns_vit_patch_depth/s0_r1_20260914/source_evidence/imagenets_releases.json",
        "evidence/nns_vit_patch_depth/s0_r1_20260914/source_evidence/imagenet_terms.json",
    ]
    implementation = [
        record(REPO / "scripts/audit_nns_vit_patch_depth_s0_r2.py"),
        record(REPO / "scripts/audit_nns_vit_patch_depth_s0_r1.py"),
        record(REPO / "scripts/acquire_nns_vit_patch_depth_s0_r1_metadata.py"),
    ]
    identity = {
        "branch": git("symbolic-ref", "--short", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "implementation": implementation,
        "python_executable_realpath": os.path.realpath(sys.executable),
        "python_version": sys.version,
        "tracked_status": git("status", "--porcelain", "--untracked-files=no"),
        "worktree_status": git("status", "--short"),
    }
    authorization_payload = {
        "status": "USER_ATTESTED_AUTHORIZATION_RECORDED",
        "authorization_scope": {
            "annotation_package": "ImageNet-S919 release annotation archive for this local research workflow",
            "imagenet_originals": "ImageNet use/access authorization confirmed by user for this local research workflow",
            "redistribution": "NOT_GRANTED_OR_INFERRED",
            "commercial_use": "NOT_GRANTED_OR_INFERRED",
        },
        "boundaries": [
            "S0-R1 remains immutable and terminal BLOCKED_LICENSE",
            "this receipt records a user authorization assertion rather than an independently discovered upstream license file",
            "no new public redistribution right is inferred",
            "authorization does not supply absent ImageNet train originals or server-only official test masks",
            "authorization does not by itself approve a new public-validation holdout split",
        ],
        "evidence_refs": source_refs,
        "recorded_user_statement": "\u6211\u7ed9\u4f60\u6807\u6ce8\u8bb8\u53ef\uff0c\u4f60\u53ef\u4ee5\u7528imagenet",
        "recorded_user_statement_language": "zh-CN",
        "recorded_user_statement_session_date_prc": "2026-09-14",
    }
    annotation_payload = {
        "status": "ANNOTATION_PACKAGE_BOUND_HEADER_ONLY",
        "access_boundary": {
            "annotation_archive_downloaded": True,
            "archive_crc_tested": True,
            "mask_file_bytes_hashed": zip_scan["mask_file_count"],
            "png_headers_parsed": zip_scan["mask_file_count"],
            "mask_pixel_decode_count": 0,
            "rgb_decode_count": 0,
            "model_load_count": 0,
            "gpu_access_count": 0,
            "inference_count": 0,
            "training_count": 0,
            "attribution_count": 0,
            "scientific_outcomes_used_for_selection": False,
        },
        "archive": {
            "expected_release": ZIP_EXPECTED,
            "local_file": zip_scan["zip_record"],
            "zip_crc_test": zip_scan["zip_crc_test"],
            "zip_entry_count": zip_scan["zip_entry_count"],
            "mask_file_count": zip_scan["mask_file_count"],
            "split_counts": zip_scan["split_counts"],
        },
        "coverage": {
            "expected_public_mask_count": zip_scan["expected_mask_count"],
            "matched_expected_mask_count": zip_scan["matched_expected_mask_count"],
            "missing_expected_mask_count": zip_scan["missing_expected_mask_count"],
            "extra_mask_count": zip_scan["extra_mask_count"],
            "missing_expected_masks": zip_scan["missing_expected_masks"],
            "extra_masks": zip_scan["extra_masks"],
        },
        "png_header_summary": {
            "dimension_count": zip_scan["dimension_count"],
            "top_dimensions": zip_scan["top_dimensions"],
            "format_counts": zip_scan["format_counts"],
            "crc_failure_count": zip_scan["crc_failure_count"],
            "crc_failures": zip_scan["crc_failures"],
            "unexpected_non_png_files": zip_scan["unexpected_non_png_files"],
        },
        "records": zip_scan["records"],
        "records_digest": zip_scan["records_digest"],
    }
    split_payload = {
        "status": "BLOCKED_SPLIT_DATA_PLAN_REQUIRED",
        "candidate_manifest_ref": {
            "path": "evidence/nns_vit_patch_depth/s0_r1_20260914/image_annotation_class_manifest.json",
            "record_count": candidate_doc["payload"]["record_count"],
            "records_digest": candidate_doc["payload"]["records_digest"],
            "canonical_digest": candidate_doc["canonical_digest"],
        },
        "license_and_payload_delta": {
            "license_gate": "PASS_BY_USER_ATTESTATION",
            "annotation_payload": "PASS_PUBLIC_TRAIN_SEMI_AND_VALIDATION_ARCHIVE_BOUND",
            "official_test_masks": "NOT_AVAILABLE_SERVER_ONLY",
        },
        "official_train_semi_route": {
            "status": "BLOCKED_DATA",
            "reason": "ImageNet train originals for the 9190 train-semi rows are still absent from active project assets",
            "required_train_original_count": 9190,
            "found_train_original_count": len(train_found),
            "found_train_originals": train_found[:20],
            "metadata_reservation": "S0-R1 reserved per-class pilot/train/validation as 2/6/2 from train-semi IDs, but images cannot be admitted until bytes are supplied and hashed",
        },
        "official_test_route": {
            "status": "BLOCKED_DATA",
            "reason": "official test ground-truth masks are held on the evaluation server and are not in the public release archive",
            "official_test_candidate_count": split_summary["official_counts"].get("test", 0),
        },
        "public_validation_holdout_candidate": {
            "status": "AVAILABLE_ONLY_AFTER_EXPLICIT_PROSPECTIVE_APPROVAL",
            "reason": "public validation masks and validation original bytes are locally bound, but S0-R1 did not approve repurposing official validation into pilot/probe/confirmatory splits",
            "eligible_public_validation_count": split_summary["eligible_public_validation_count"],
            "eligible_public_validation_digest": split_summary["eligible_public_validation_digest"],
            "profile_summaries": split_summary["profile_summaries"],
            "class_count_distribution": split_summary["eligible_count_distribution"],
            "per_class": split_summary["class_rows"],
            "per_class_digest": split_summary["class_rows_digest"],
            "required_before_use": [
                "freeze a reduced class universe or provide a rule for low-count classes",
                "freeze pilot/probe_train/probe_validation/confirmatory_test counts and hash namespace before looking at outcomes",
                "keep historical and byte-alias quarantines",
                "run mask pixel QC and crop-target presence checks before S1 scientific pilot",
                "evaluate statistical power after S1 costs and variances are measured",
            ],
        },
        "remaining_blockers": [
            "BLOCKED_SPLIT",
            "BLOCKED_DATA for train-semi originals if the official train route is retained",
            "BLOCKED_DATA for official test masks if the official test route is retained",
            "BLOCKED_DEFINITION for merged/low-count classes until a class-universe rule is frozen",
            "MASK_PIXEL_QC_NOT_RUN",
            "STATISTICAL_POWER_NOT_EVALUATED",
        ],
        "revised_candidate_reason_counts": split_summary["revised_reason_counts"],
    }
    decision_payload = {
        "audit_workflow_status": "COMPLETE",
        "data_readiness_status": "BLOCKED_SPLIT",
        "additional_blockers": [
            "BLOCKED_DATA",
            "BLOCKED_DEFINITION",
            "MASK_PIXEL_QC_NOT_RUN",
            "STATISTICAL_POWER_NOT_EVALUATED",
        ],
        "adopted_dataset": "ImageNet-S919",
        "adoption_status": "LICENSE_AND_PUBLIC_ANNOTATION_PAYLOAD_ACCEPTED_NOT_SPLIT_FROZEN",
        "candidate_count": candidate_doc["payload"]["record_count"],
        "qualified_count_for_current_s0_r1_split": 0,
        "public_validation_holdout_candidate_count": split_summary["eligible_public_validation_count"],
        "s1_started": False,
        "can_request_S1_now": False,
        "next_authorization": "DATA_SPLIT_ROUTE_REQUIRED",
        "gates": [
            {"gate": "predecessor_identity", "status": "PASS", "detail": "S0-R1 terminal and inherited canonical digests verified"},
            {"gate": "annotation_license", "status": "PASS_BY_USER_ATTESTATION", "detail": "user provided annotation permission and confirmed ImageNet use"},
            {"gate": "annotation_payload", "status": "PASS", "detail": "ImageNet-S919 release zip downloaded, SHA-256 and zip CRC verified, all public train-semi/validation mask paths present"},
            {"gate": "mask_headers", "status": "PASS_HEADER_ONLY", "detail": "21609 PNG headers parsed and file bytes hashed; mask pixels were not decoded"},
            {"gate": "original_images", "status": "BLOCKED_DATA", "detail": "validation/test originals are available from local val archive; 9190 train-semi originals are still absent"},
            {"gate": "official_test_truth", "status": "BLOCKED_DATA", "detail": "official test ground-truth masks are server-only and absent from the public archive"},
            {"gate": "public_validation_holdout", "status": "AWAITING_APPROVAL", "detail": "12074 exact, non-historical, public validation candidates are available before mask-pixel QC; split not frozen"},
            {"gate": "geometry_qc", "status": "PARTIAL_HEADER_ONLY", "detail": "PNG dimensions known; mask label values, void coverage and crop-target presence not yet tested"},
        ],
        "next_user_decisions": [
            "supply the required ImageNet train originals and legitimate official test truth access if retaining the official train/test route",
            "or explicitly approve a fresh public-validation holdout route with a reduced class universe and frozen split counts",
            "approve mask-pixel QC in the chosen route before any S1 scientific pilot",
            "do not overwrite S0-R1 or S0-R2 terminals; continue through a fresh successor root",
        ],
        "not_performed": [
            "RGB decode",
            "mask pixel decode",
            "model load",
            "GPU",
            "probe training",
            "attribution",
            "Y4 changes",
            "S1",
            "commit",
            "push",
        ],
        "artifacts": [],
    }
    payloads = {
        "authorization_receipt": authorization_payload,
        "annotation_package_inventory": annotation_payload,
        "split_feasibility": split_payload,
        "feasibility_decision": decision_payload,
    }
    docs = {}
    predecessor_record = inherited[-1]
    for name in NAMES:
        docs[name] = signed({
            "schema_version": 1,
            "study_id": STUDY_ID,
            "stage_id": STAGE_ID,
            "run_id": root.name,
            "document_type": name,
            "code_identity": identity,
            "predecessor": predecessor_record,
            "inherited_files": inherited,
            "payload": payloads[name],
        })
    docs["feasibility_decision"]["payload"]["artifacts"] = [
        {
            "name": name + ".json",
            "sha256": hashlib.sha256(serialize(docs[name])).hexdigest(),
            "canonical_digest": docs[name]["canonical_digest"],
        }
        for name in NAMES[:-1]
    ]
    docs["feasibility_decision"]["canonical_digest"] = digest({k: v for k, v in docs["feasibility_decision"].items() if k != "canonical_digest"})
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="evidence/nns_vit_patch_depth/s0_r2_20260914")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = (REPO / args.root).resolve()
    require(REPO / "evidence/nns_vit_patch_depth" in root.parents, "root outside evidence namespace")
    require(S0_R1.is_dir(), "S0-R1 predecessor root missing")
    docs = build_documents(root)
    require(not {"torch", "torchvision", "timm", "numpy", "PIL", "cv2", "scipy"}.intersection(sys.modules), "forbidden module loaded")
    if args.verify:
        schema = json.loads((root / "artifact_contracts.schema.json").read_text())
        for name in NAMES:
            old = read_signed(root / (name + ".json"))
            validate_closed(old, schema["$defs"][name])
            require(old == docs[name], "source-recomputed artifact mismatch: " + name)
        print("PASS: S0-R2 artifacts equal source-recomputed license-unblocked package binding and split decision; no writes")
    else:
        require(not root.exists(), "terminal root already exists")
        root.mkdir(parents=True)
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {name: closed_contract(docs[name]) for name in NAMES},
            "oneOf": [{"$ref": "#/$defs/" + name} for name in NAMES],
        }
        write_new(root / "artifact_contracts.schema.json", schema)
        for name in NAMES:
            validate_closed(docs[name], schema["$defs"][name])
            write_new(root / (name + ".json"), docs[name])
        print("SEALED: S0-R2 license gate PASS_BY_USER_ATTESTATION; readiness BLOCKED_SPLIT")
    print(json.dumps({
        "annotation_masks": docs["annotation_package_inventory"]["payload"]["archive"]["mask_file_count"],
        "public_validation_holdout_candidate_count": docs["feasibility_decision"]["payload"]["public_validation_holdout_candidate_count"],
        "can_request_S1_now": docs["feasibility_decision"]["payload"]["can_request_S1_now"],
        "data_readiness_status": docs["feasibility_decision"]["payload"]["data_readiness_status"],
        "next_authorization": docs["feasibility_decision"]["payload"]["next_authorization"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
