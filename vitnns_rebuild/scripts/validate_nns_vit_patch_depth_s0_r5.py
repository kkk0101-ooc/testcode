#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
AUTHORIZED_COMMIT = "66a783d15971308b3307ab4551f00e0abfae0b03"
PRE_TERMINAL_NAMES = (
    "source_binding",
    "route_contract",
    "class_manifest",
    "split_manifest",
    "selection_ledger",
    "independence_audit",
)
SPLITS = ("pilot", "probe_train", "probe_validation", "confirmatory_holdout")
TRAIN_NAMESPACE = "S0_R1_METADATA_RESERVATION_V1"
HOLDOUT_NAMESPACE = "S0_R5_PUBLIC_VALIDATION_CONFIRMATORY_HOLDOUT_V1"


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


def read_signed(path):
    value = json.loads(path.read_text())
    require(value["canonical_digest"] == digest({key: item for key, item in value.items() if key != "canonical_digest"}), "canonical digest mismatch: " + str(path))
    return value


def selection_key(namespace, source_image_id):
    return hashlib.sha256((namespace + "\0" + source_image_id).encode()).hexdigest()


def validate_source_binding(binding):
    payload = binding["payload"]
    require(payload["status"] == "PASS", "source binding not PASS")
    require(payload["authorized_commit"] == AUTHORIZED_COMMIT, "authorized commit mismatch")
    require(payload["current_head_at_freeze"] == AUTHORIZED_COMMIT, "freeze HEAD mismatch")
    observed_files = []
    for expected in payload["files"]:
        path = REPO / expected["path"]
        observed = file_record(path)
        for key in ("path", "bytes", "sha256", "canonical_digest"):
            if key in expected:
                require(observed.get(key) == expected[key], "working source binding mismatch: " + expected["path"] + " " + key)
        line = git("ls-tree", "-l", AUTHORIZED_COMMIT, "--", "vitnns_rebuild/" + expected["path"]).strip()
        require(line, "source absent from authorized commit")
        metadata, tree_path = line.split("\t", 1)
        mode, kind, blob_oid, blob_bytes = metadata.split()
        require(kind == "blob" and tree_path == "vitnns_rebuild/" + expected["path"], "tree binding mismatch")
        blob = git("cat-file", "blob", blob_oid, binary=True)
        if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            pointer = dict(item.split(" ", 1) for item in blob.decode().splitlines()[1:])
            require(pointer["oid"] == "sha256:" + expected["sha256"] and int(pointer["size"]) == expected["bytes"], "LFS pointer mismatch")
            storage = "git_lfs"
        else:
            require(blob == path.read_bytes(), "Git blob content mismatch")
            storage = "git_blob"
        require(expected["authorized_commit"] == AUTHORIZED_COMMIT, "per-file commit mismatch")
        require(expected["git_blob_oid"] == blob_oid and expected["git_mode"] == mode, "Git tree metadata mismatch")
        require(expected["storage"] == storage and expected["tree_blob_bytes"] == int(blob_bytes), "Git storage metadata mismatch")
        observed_files.append(expected)
    require(payload["files_digest"] == digest(observed_files), "source file list digest mismatch")


def validate_selection(root, documents):
    route = documents["route_contract"]["payload"]
    classes = documents["class_manifest"]["payload"]
    split = documents["split_manifest"]["payload"]
    ledger = documents["selection_ledger"]["payload"]
    independence = documents["independence_audit"]["payload"]
    require(route["status"] == classes["status"] == split["status"] == "FROZEN", "freeze status mismatch")
    require(route["authorization"]["approved_stage"] == "S0-R5_ONLY", "authorization stage mismatch")
    require(route["authorization"]["approved_route"] == "PUBLIC_VALIDATION_CONFIRMATORY_HOLDOUT", "route mismatch")
    require(route["authorization"]["official_test_route"] == "ABANDONED_SERVER_ONLY_GROUND_TRUTH", "official test route not abandoned")
    require(route["confirmatory_label_required"] == "public-validation confirmatory holdout", "confirmatory label mismatch")
    require(route["forbidden_claim"] == "ImageNet-S official test result", "forbidden claim missing")
    require(not any(route["prohibited_operations_observed"].values()), "prohibited operation observed")
    require(route["train_selection_namespace"] == TRAIN_NAMESPACE and route["confirmatory_selection_namespace"] == HOLDOUT_NAMESPACE, "selection namespace mismatch")

    r4_split = read_signed(REPO / "evidence/nns_vit_patch_depth/s0_r4_20260914/split_feasibility.json")
    r3_qc = read_signed(REPO / "evidence/nns_vit_patch_depth/s0_r3_20260914/image_mask_qc.json")
    manifest = read_signed(REPO / "evidence/nns_vit_patch_depth/s0_r1_20260914/image_annotation_class_manifest.json")
    reuse = read_signed(REPO / "evidence/nns_vit_patch_depth/s0_r1_20260914/reuse_and_exclusion_manifest.json")
    operator_note = read_signed(REPO / "evidence/nns_vit_patch_depth/s0_r2_operator_note_20260914/session_access_note.json")
    adjusted = {record["candidate_id"]: record for record in r4_split["payload"]["adjusted_records"]}
    qc = {record["candidate_id"]: record for record in r3_qc["payload"]["records"]}
    manifest_records = {record["candidate_id"]: record for record in manifest["payload"]["records"]}
    eligible = [record for record in adjusted.values() if record["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE"]
    train = collections.defaultdict(lambda: collections.defaultdict(list))
    validation = collections.defaultdict(list)
    for record in eligible:
        if record["official_split"] == "train-semi":
            train[record["annotation_wnid"]][record["provisional_pool"]].append(record)
        else:
            require(record["official_split"] == "validation", "unexpected eligible source split")
            validation[record["annotation_wnid"]].append(record)
    accepted = sorted(wnid for wnid in set(train) | set(validation)
                      if len(train[wnid]["pilot"]) == 2
                      and len(train[wnid]["probe_train"]) == 6
                      and len(train[wnid]["probe_validation"]) == 2
                      and len(validation[wnid]) >= 2)
    require(len(accepted) == route["class_count"] == classes["class_count_selected"] == split["class_count"] == 872, "selected class count mismatch")
    class_by_wnid = {record["annotation_wnid"]: record for record in classes["records"]}
    require(len(class_by_wnid) == classes["class_count_total"] == 919, "class manifest universe mismatch")
    require(classes["records_digest"] == digest(classes["records"]), "class manifest digest mismatch")
    for wnid in accepted:
        record = class_by_wnid[wnid]
        require(record["class_status"] == "SELECTED" and not record["exclusion_reasons"], "accepted class state mismatch")
        require(record["qualified_train_counts"] == {"pilot": 2, "probe_train": 6, "probe_validation": 2}, "accepted train counts mismatch")
        require(record["qualified_public_validation_count"] == len(validation[wnid]), "validation count mismatch")
        require(record["selected_counts"] == {"pilot": 2, "probe_train": 6, "probe_validation": 2, "confirmatory_holdout": 2}, "selected class allocation mismatch")
        require(record["target_model_class_index"] is not None, "selected class lacks exact mapping")

    final_records = split["records"]
    require(split["record_count"] == len(final_records) == 10464, "split record count mismatch")
    require(split["records_digest"] == digest(final_records), "split manifest digest mismatch")
    by_split = {name: [record for record in final_records if record["final_split"] == name] for name in SPLITS}
    require(split["split_counts"] == {name: len(by_split[name]) for name in SPLITS} == {"pilot": 1744, "probe_train": 5232, "probe_validation": 1744, "confirmatory_holdout": 1744}, "split counts mismatch")
    require(split["split_digests"] == {name: digest(by_split[name]) for name in SPLITS}, "per-split digest mismatch")
    seen_candidates = set()
    for record in final_records:
        candidate_id = record["candidate_id"]
        require(candidate_id not in seen_candidates, "duplicate selected candidate")
        seen_candidates.add(candidate_id)
        source = manifest_records[candidate_id]
        source_qc = qc[candidate_id]
        source_r4 = adjusted[candidate_id]
        require(source_r4["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE" and not source_r4["exclusion_reasons"], "selected R4-ineligible record")
        require(not source_qc["exclusion_reasons"], "selected QC-excluded record")
        require(record["r4_adjusted_record_digest"] == digest(source_r4) and record["r3_qc_record_digest"] == digest(source_qc), "record lineage digest mismatch")
        require(record["source_image_id"] == source["source_image_id"] and record["source_group_id"] == source["group_id"], "source identity mismatch")
        require(record["image_sha256"] == source_r4["image_sha256"] == source_qc["image_sha256"], "image hash mismatch")
        require(record["annotation_sha256"] == source_qc["annotation_sha256"], "mask hash mismatch")
        require(record["target_model_class_index"] == source["exact_target_class_index"] is not None, "class mapping mismatch")
        if record["final_split"] == "confirmatory_holdout":
            require(record["official_split"] == "validation", "holdout not from public validation")
            expected_key = selection_key(HOLDOUT_NAMESPACE, source["source_image_id"])
            ranked = sorted(validation[record["annotation_wnid"]], key=lambda item: (selection_key(HOLDOUT_NAMESPACE, manifest_records[item["candidate_id"]]["source_image_id"]), item["candidate_id"]))
            require(candidate_id == ranked[record["selection_rank_within_class_split"] - 1]["candidate_id"] and record["selection_rank_within_class_split"] in (1, 2), "holdout rank mismatch")
            require(record["selection_namespace"] == HOLDOUT_NAMESPACE and record["selection_key_sha256"] == expected_key, "holdout selection key mismatch")
            require(record["use_policy"] == "CONFIRMATORY_HOLDOUT_NO_ACCESS_BEFORE_FROZEN_EVALUATION", "holdout use policy mismatch")
        else:
            require(record["official_split"] == "train-semi", "non-holdout not from train-semi")
            require(record["selection_namespace"] == TRAIN_NAMESPACE and record["selection_key_sha256"] == selection_key(TRAIN_NAMESPACE, source["source_image_id"]), "train selection key mismatch")

    pairwise = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            left_ids = {record["source_image_id"] for record in by_split[left]}
            right_ids = {record["source_image_id"] for record in by_split[right]}
            left_hashes = {record["image_sha256"] for record in by_split[left]}
            right_hashes = {record["image_sha256"] for record in by_split[right]}
            pairwise.append({
                "left": left,
                "right": right,
                "source_image_id_intersection_count": len(left_ids & right_ids),
                "source_image_id_intersection_digest": digest(sorted(left_ids & right_ids)),
                "image_sha256_intersection_count": len(left_hashes & right_hashes),
                "image_sha256_intersection_digest": digest(sorted(left_hashes & right_hashes)),
            })
    require(pairwise == independence["pairwise_split_checks"], "pairwise audit mismatch")
    require(all(row["source_image_id_intersection_count"] == row["image_sha256_intersection_count"] == 0 for row in pairwise), "split intersection found")
    selected_ids = {record["source_image_id"] for record in final_records}
    selected_hashes = {record["image_sha256"] for record in final_records}
    historical_ids = {record["source_image_id"] for record in reuse["payload"]["historical_exclusions"]}
    operator_masks = {record["annotation_archive_path"] for record in operator_note["payload"]["accessed_mask_records"]}
    require(not selected_ids & historical_ids, "historical source selected")
    require(not operator_masks & {record["annotation_archive_path"] for record in final_records}, "operator-accessed mask selected")
    require(independence["status"] == "PASS" and independence["selected_source_image_count"] == len(selected_ids) == 10464, "independence summary mismatch")
    require(independence["selected_image_sha256_count"] == len(selected_hashes) == 10464, "selected byte uniqueness mismatch")
    require(independence["historical_source_id_intersection_count"] == 0 and independence["historical_byte_hash_intersection_count"] == 0, "historical intersection summary mismatch")
    require(independence["operator_accessed_mask_intersection_count"] == 0, "operator mask intersection summary mismatch")
    require(independence["all_selected_r4_eligible"] and independence["all_selected_mask_qc_clean"] and independence["all_selected_exact_class_mapping"], "selected validity summary mismatch")
    require(ledger["candidate_count"] == len(ledger["records"]) == 21609 and ledger["selected_count"] == 10464, "selection ledger count mismatch")
    require(ledger["records_digest"] == digest(ledger["records"]), "selection ledger digest mismatch")
    require(ledger["disposition_counts"] == dict(sorted(collections.Counter(record["disposition"] for record in ledger["records"]).items())), "selection ledger counts mismatch")
    return {
        "accepted_class_count": len(accepted),
        "selected_record_count": len(final_records),
        "split_counts": split["split_counts"],
        "split_manifest_records_digest": split["records_digest"],
        "pairwise_checks_digest": digest(pairwise),
        "selection_ledger_records_digest": ledger["records_digest"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="evidence/nns_vit_patch_depth/s0_r5_20260914")
    args = parser.parse_args()
    root = (REPO / args.root).resolve()
    require(root.is_dir() and not (root / "terminal_receipt.json").exists(), "root missing or already terminal")
    documents = {name: read_signed(root / (name + ".json")) for name in PRE_TERMINAL_NAMES}
    for name, document in documents.items():
        require(document["study_id"] == "NNS_VIT_PATCH_DEPTH_SEMANTICS" and document["stage_id"] == "S0-R5", "document identity mismatch: " + name)
        require(document["run_id"] == root.name and document["document_type"] == name, "document type mismatch: " + name)
        require(document["authorized_commit"] == AUTHORIZED_COMMIT, "document commit mismatch: " + name)
    validate_source_binding(documents["source_binding"])
    summary = validate_selection(root, documents)
    artifacts = []
    for name in PRE_TERMINAL_NAMES:
        item = file_record(root / (name + ".json"))
        item["name"] = name + ".json"
        item["canonical_digest"] = documents[name]["canonical_digest"]
        artifacts.append(item)
    report = signed({
        "schema_version": 1,
        "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS",
        "stage_id": "S0-R5",
        "run_id": root.name,
        "document_type": "validation_report",
        "authorized_commit": AUTHORIZED_COMMIT,
        "validator": file_record(REPO / "scripts/validate_nns_vit_patch_depth_s0_r5.py"),
        "payload": {
            "status": "PASS",
            "validated_artifacts": artifacts,
            "summary": summary,
            "checks": [
                "exact authorized commit and Git/LFS source bindings",
                "all canonical digests and referenced artifact byte hashes",
                "closed deterministic class and sample selection",
                "fixed-hash public-validation holdout selection",
                "source-image and byte-hash split disjointness",
                "historical exclusion and operator-access exclusion",
                "mask QC inheritance and exact class mapping",
                "prohibited-operation attestation and route labeling",
            ],
        },
    })
    path = root / "validation_report.json"
    require(not path.exists(), "validation report already exists")
    with path.open("xb") as stream:
        stream.write(serialize(report))
        stream.flush()
        os.fsync(stream.fileno())
    print("PASS: S0-R5 independent split and provenance validation")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
