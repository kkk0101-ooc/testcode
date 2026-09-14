#!/usr/bin/env python3
import argparse
import ast
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tarfile


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
Y4 = "evidence/stage_y/y4_r5_mechanism_analysis_20260914/formal"
Y2 = "evidence/stage_y/y2_clean_formal_20260910/formal"
DEFAULT_ROOT = "evidence/nns_vit_patch_depth/s0_20260914"
NATIVE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG")
PREFIX = "ILSVRC2012_VAL_OFFICIAL:ILSVRC2012_val/"
NAMES = ("asset_inventory", "annotation_contract", "reuse_and_exclusion_manifest", "feasibility_decision")
FORBIDDEN_IMPORTS = {"torch", "torchvision", "timm", "numpy", "PIL", "cv2", "scipy"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(["git", "-C", str(WORKSPACE), *args], text=True).strip()


def sid(value):
    match = NATIVE.search(value)
    require(match is not None and 1 <= int(match.group(1)) <= 50000, "invalid native ID: " + value)
    return PREFIX + match.group(0)


class Audit:
    def __init__(self):
        self.records = {}
        self.array_checks = []
        self.bytes_hashed = 0

    def path(self, relative):
        path = (REPO / relative).resolve()
        require(path == WORKSPACE or WORKSPACE in path.parents, "source outside workspace: " + relative)
        return path

    def record(self, relative, expected=None):
        if relative not in self.records:
            path = self.path(relative)
            require(path.suffix.lower() not in {".jpeg", ".jpg", ".png", ".pdf", ".svg"}, "direct image read forbidden")
            before = path.stat()
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "source changed while hashing")
            self.bytes_hashed += before.st_size
            record = {"path": relative, "bytes": before.st_size, "sha256": h.hexdigest()}
            if path.suffix == ".json":
                value = json.loads(path.read_text())
                if isinstance(value, dict) and "canonical_digest" in value:
                    unsigned = {k: v for k, v in value.items() if k != "canonical_digest"}
                    observed = digest(unsigned)
                    require(observed == value["canonical_digest"], "canonical digest mismatch: " + relative)
                    record["canonical_digest"] = observed
            self.records[relative] = record
        result = self.records[relative]
        if expected:
            for key in ("bytes", "sha256", "canonical_digest"):
                if key in expected:
                    require(result.get(key) == expected[key], "binding mismatch: " + relative + " " + key)
        return result

    def read(self, relative, expected=None):
        self.record(relative, expected)
        return json.loads(self.path(relative).read_text())

    def array(self, expected):
        relative = expected["path"]
        path = self.path(relative)
        require(path.suffix == ".npy", "only NPY headers and bytes allowed")
        before = path.stat()
        with path.open("rb") as stream:
            prefix = stream.read(8)
            require(prefix[:6] == b"\x93NUMPY", "not NPY")
            size_bytes = 2 if prefix[6] == 1 else 4
            require(prefix[6] in (1, 2, 3), "unsupported NPY version")
            raw_size = stream.read(size_bytes)
            header_size = int.from_bytes(raw_size, "little")
            require(header_size <= 65536, "excessive NPY header")
            header_bytes = stream.read(header_size)
            header = ast.literal_eval(header_bytes.decode("utf-8" if prefix[6] == 3 else "latin1"))
            require(header["descr"] == "<f4" and not header["fortran_order"], "unexpected scientific array layout")
            shape = list(header["shape"])
            require(shape == expected["shape"] and expected["dtype"] == "float32", "array shape/dtype mismatch")
            h = hashlib.sha256(prefix + raw_size + header_bytes)
            c = hashlib.sha256(b"float32\0" + str(tuple(shape)).encode() + b"\0")
            payload_bytes = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
                c.update(chunk)
                payload_bytes += len(chunk)
        elements = 1
        for size in shape:
            elements *= size
        require(payload_bytes == elements * 4, "NPY payload size mismatch")
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "array changed while hashing")
        require(before.st_size == expected["bytes"] and h.hexdigest() == expected["sha256"], "array byte identity mismatch")
        require(c.hexdigest() == expected["canonical_digest"], "array canonical identity mismatch")
        self.bytes_hashed += before.st_size
        result = dict(expected)
        self.array_checks.append(result)
        return result


def identity_strings(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("sample_id", "source_object_id", "sample_ids", "selected_sample_ids", "x1_allowed_audit_samples"):
                for text in item if isinstance(item, list) else [item]:
                    if isinstance(text, str):
                        found.add(sid(text))
            found.update(identity_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.update(identity_strings(item))
    return found


def metadata_inventory():
    records = []
    errors = []
    for directory, dirs, files in os.walk(REPO / "assets", followlinks=False, onerror=lambda e: errors.append(str(e))):
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            records.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "is_symlink": path.is_symlink()})
    require(not errors, "asset inventory incomplete")
    candidates = [r for r in records if re.search(r"annotation|segmentation|trimap|partimagenet|imagenet.?s|bbox|bounding", r["path"], re.I)]
    require(not candidates, "new annotation candidate requires manual review before decision")
    pngs = [r for r in records if Path(r["path"]).suffix.lower() in (".png", ".xml", ".json", ".tif", ".tiff")]
    require(not pngs, "unclassified possible annotation requires review")
    return {"scope": "recursive active-project assets directory including Git-ignored entries; no host-wide absence claim",
            "root": str(REPO / "assets"), "files": records, "records_digest": digest(records),
            "annotation_name_candidates": candidates, "other_possible_annotation_files": pngs,
            "access_errors": errors, "symlink_directories_followed": False,
            "excluded_locations": ["workspace ZIP backups", "generated evidence/reports figures and geometric validity masks", "unreferenced external host directories"],
            "segmentation_mask_count_verified": 0, "part_mask_count_verified": 0, "bounding_box_annotation_count_verified": 0}


def build(audit):
    print("S0: verify Y4 package and implementation identities", flush=True)
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    require(not tracked_status, "tracked worktree changed before S0")
    identity = {"head": git("rev-parse", "HEAD"), "branch": git("symbolic-ref", "--short", "HEAD"),
                "tracked_status": tracked_status, "status_at_audit_start": git("status", "--short"),
                "auditor": audit.record(str(Path(__file__).relative_to(REPO))),
                "python_executable_realpath": os.path.realpath(sys.executable), "python_version": sys.version,
                "model_python_executable_realpath": os.path.realpath(str(WORKSPACE / ".conda/envs/vitnns/bin/python")),
                "cublas_workspace_config_observed_not_modified": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "network_or_remote_refs_verified": False}
    completion = audit.read(Y4 + "/completion/y4_r5.json")
    require(completion["status"] == "PASS", "predecessor Y4 not PASS")
    for key in ("whole_root_inventory", "payload_manifest", "reconciliation", "report", "validation_report"):
        record = completion["payload"][key]
        audit.record(record["path"], record)
    inventory = audit.read(Y4 + "/whole_root_inventory.json")["payload"]
    require(digest({"files": inventory["files"]}) == inventory["files_digest"], "Y4 inventory digest mismatch")
    require(inventory["file_count"] == len(inventory["files"]), "Y4 inventory count mismatch")
    for record in inventory["files"]:
        audit.record(record["path"], record)
    require(sum(r["bytes"] for r in inventory["files"]) == inventory["total_bytes"], "Y4 inventory byte total mismatch")
    seal = audit.read(Y4 + "/producer_implementation_seal.json")["payload"]
    require(digest({"files": seal["source_files"]}) == seal["files_digest"], "producer seal mismatch")
    for record in seal["source_files"]:
        audit.record(record["path"], record)
    require(subprocess.run(["git", "-C", str(WORKSPACE), "merge-base", "--is-ancestor", seal["producer_commit"], identity["head"]]).returncode == 0, "producer is not ancestor")
    bindings = []
    for name in ("y2_completion_publication_binding.json", "y3_completion_publication_binding.json"):
        binding = audit.read(Y4 + "/" + name)["payload"]
        require(digest(binding["files"]) == binding["files_digest"], "predecessor binding digest mismatch")
        for record in binding["files"]:
            audit.record(record["path"], record)
        require(subprocess.run(["git", "-C", str(WORKSPACE), "merge-base", "--is-ancestor", binding["commit"], identity["head"]]).returncode == 0, "predecessor not ancestor")
        bindings.append(binding)
    for relative in ("docs/nns_vit_patch_depth_research_goals_20260914.md", "docs/nns_vit_patch_depth_execution_plan_20260914.md",
                     "docs/stage_y_y4_r3_attention_cut_recovery_20260914.md", "docs/stage_y_y4_r4_zero_mass_na_recovery_20260914.md",
                     "docs/stage_e_protocol.md", "assets/README.md", "assets/download_imagenet_val.sh",
                     "src/vitnns/stage_y_y1.py", "src/vitnns/stage_y_y2.py", "src/vitnns/stage_y_y2_worker.py",
                     "src/vitnns/labels.py", "src/vitnns/constants.py", "requirements.lock.txt"):
        audit.record(relative)
    binding = audit.read("config/recovery_v2_binding.json")
    model = audit.record(binding["model"]["state_dict_path"], {"sha256": binding["model"]["state_dict_sha256"], "bytes": binding["model"]["state_dict_bytes"]})
    require(audit.record("requirements.lock.txt")["sha256"] == binding["environment"]["requirements_lock_sha256"], "lock drift")
    model = {**model, "model_name": binding["model"]["name"], "file_identity_status": "PASS",
             "canonical_tensor_digest_from_sealed_binding": binding["model"]["canonical_tensor_digest"],
             "tensor_digest_recomputed": False, "deserialized": False,
             "tensor_digest_limitation": "S0 hashes checkpoint bytes only; no model/tensor loading"}

    print("S0: verify clean source array bytes and headers without numerical loading", flush=True)
    cohort = audit.read(Y2 + "/cohort_registry.json")["payload"]["records"]
    clean = audit.read(Y4 + "/clean_100_image_source_manifest.json")["payload"]
    require(digest(clean["records"]) == clean["records_digest"] and len(clean["records"]) == 100, "clean roster mismatch")
    units = []
    shapes = collections.Counter()
    target_matches = 0
    for source in clean["records"]:
        terminal = audit.read(source["source_terminal"]["path"], source["source_terminal"])
        payload = terminal["payload"]
        row = cohort[source["index"]]
        require(payload["source_object_id"] == row["source_object_id"] == source["source_object_id"], "unit identity mismatch")
        arrays = payload["array_records"]
        require(len(arrays) == 262, "unexpected clean array count")
        for record in arrays:
            audit.array(record)
            shapes[(record["dtype"], tuple(record["shape"]))] += 1
        names = {Path(r["path"]).stem for r in arrays}
        required = {"b%02d_first_attention_%s" % (b, suffix) for b in range(12) for suffix in ("cell", "matrix", "cls")}
        require(required <= names, "primary component missing")
        science = payload.get("scientific_record") or payload.get("scientific_result")
        if science:
            audit.record(science["path"], science)
        match = payload["target_identity"]["class_id"] == row["class_id"]
        target_matches += int(match)
        units.append({"selection_order": source["index"], "source_object_id": row["source_object_id"],
                      "true_class_id": row["class_id"], "stored_target": payload["target_identity"], "target_class_matches_true_class": match,
                      "original_terminal_status": terminal["status"], "inherited_Y4_array_status": source["status"],
                      "source_terminal": source["source_terminal"], "array_count": len(arrays), "array_bytes": sum(r["bytes"] for r in arrays),
                      "verified_array_records_digest": digest(arrays), "scientific_result": science,
                      "native_token_features_retained": False, "new_confirmatory_reuse_allowed": False,
                      "allowed_reuse": "EXPLORATORY_ENGINEERING_ONLY_AFTER_SEPARATE_S1_AUTHORIZATION"})
    raw_features = [s for dtype, s in shapes if len(s) >= 2 and s[-1] == 768]
    require(not raw_features, "unexpected token features require inspection")
    primitives = []
    with audit.path(Y4 + "/per_image_primitive_metrics.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["population"] == "clean" and row["condition_id"] == "clean" and row["block"] == "11":
                primitives.append({"source_object_id": row["source_object_id"], "off_diagonal_fraction": row["off_diagonal_absolute_mass_fraction"], "cls_routing_fraction": row["cls_routing_fraction"]})
    require(len(primitives) == 100 and all(r["off_diagonal_fraction"] == "" and r["cls_routing_fraction"] == "1.0" for r in primitives), "sealed zero-mass summary changed")

    print("S0: rebuild historical exclusion union and class mapping", flush=True)
    plan = audit.read("manifests/stage_y/cohort_selection_and_exclusion_plan.json")
    sealed_exclusion = audit.read(Y2 + "/exclusion_union.json")["payload"]
    sources = []
    reasons = collections.defaultdict(set)
    def include(stage, relative, rule):
        audit.record(relative)
        if relative.endswith(".csv"):
            with audit.path(relative).open() as stream:
                rows = list(csv.DictReader(stream))
            ids = {sid(r.get("source_object_id") or r["sample_id"]) for r in rows}
        else:
            ids = identity_strings(audit.read(relative))
        require(ids, "empty exclusion source: " + relative)
        for source_id in ids:
            reasons[source_id].add(stage)
        sources.append({"stage": stage, "source": audit.records[relative], "rule": rule, "count": len(ids), "source_ids_digest": digest(sorted(ids))})
        return ids
    old_union = set()
    for item in plan["prior_access_exclusion_sources"]:
        old_union.update(include(item["stage"], item["path"], item["rule"]))
    require(sorted(old_union) == sealed_exclusion["source_object_ids"] and digest(sorted(old_union)) == sealed_exclusion["union_digest"], "historical Y2 exclusion recomputation mismatch")
    old_count = len(old_union)
    c_all = include("Stage_C_full_registered_500_conservative", "manifests/stage_c_flat_val_500/manifest.csv", "new study conservatively excludes every historically registered Stage C image; not a retroactive Y4 rule")
    y_ids = include("Stage_Y_Y2_Y3_Y4_outcome_known", Y2 + "/cohort_registry.json", "all Y2 images excluded; Y3 is a subset and no new S0 images are selected")
    union = set(reasons)
    synset_file = "../.conda/envs/vitnns/lib/python3.10/site-packages/timm/data/_info/imagenet_synsets.txt"
    functional_file = "../.conda/envs/vitnns/lib/python3.10/site-packages/torchvision/transforms/functional.py"
    audit.record(synset_file, {"sha256": "70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15"})
    audit.record(functional_file)
    labels_file = "assets/imagenet/imagenet_2012_validation_synset_labels.txt"
    audit.record(labels_file, {"sha256": "313fc3f23c864dde9183c9a368809065765dd1b657a6904fded44e3e46d34604"})
    labels = audit.path(labels_file).read_text().splitlines()
    synsets = audit.path(synset_file).read_text().splitlines()
    require(len(labels) == 50000 and len(synsets) == 1000 and synsets == sorted(set(labels)), "class roster mismatch")
    require(set(collections.Counter(labels).values()) == {50}, "release per-class count mismatch")
    class_map = {wnid: index for index, wnid in enumerate(synsets)}
    for row in cohort:
        number = int(NATIVE.search(row["source_object_id"]).group(1))
        require(row["official_wnid"] == labels[number - 1] and row["class_id"] == class_map[labels[number - 1]], "cohort label mismatch")
    for relative in ("manifests/stage_c_sample_registry.csv", "manifests/stage_c_flat_val_500/manifest.csv"):
        with audit.path(relative).open() as stream:
            for row in csv.DictReader(stream):
                number = int(NATIVE.search(row["sample_id"]).group(1))
                require(int(row["imagenet1k_index"]) == class_map[labels[number - 1]], "historical class mapping mismatch")

    print("S0: inventory archive members and stream raw JPEG bytes for hashes only", flush=True)
    devkit_file = "assets/imagenet/ILSVRC2012_devkit_t12.tar.gz"
    audit.record(devkit_file, {"sha256": binding["stage_c_input"]["official_devkit_sha256"]})
    with tarfile.open(str(audit.path(devkit_file)), "r:gz") as archive:
        devkit_members = [{"name": m.name, "bytes": m.size, "file": m.isfile()} for m in archive]
        copying = archive.extractfile("ILSVRC2012_devkit_t12/COPYING").read()
        gt = archive.extractfile("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt").read()
    require(not any(Path(m["name"]).suffix.lower() in (".xml", ".png") for m in devkit_members), "devkit annotation candidate requires review")
    numeric_gt = [int(x) for x in gt.decode().splitlines()]
    require(len(numeric_gt) == 50000 and set(collections.Counter(numeric_gt).values()) == {50}, "devkit GT distribution mismatch")
    numeric_to_wnid = collections.defaultdict(set)
    for number, wnid in zip(numeric_gt, labels):
        numeric_to_wnid[number].add(wnid)
    require(set(numeric_to_wnid) == set(range(1, 1001)) and all(len(s) == 1 for s in numeric_to_wnid.values()), "official numeric labels disagree with WNID partition")
    tar_file = "assets/imagenet/ILSVRC2012_img_val.tar"
    tar_record = audit.record(tar_file)
    candidate_digest = hashlib.sha256()
    hashes = collections.defaultdict(list)
    class_counts = collections.Counter()
    available_counts = collections.Counter()
    seen = set()
    cohort_hashes = {r["source_object_id"]: r["file_sha256"] for r in cohort}
    excluded_hashes = {}
    candidate_rows = []
    member_bytes = 0
    with tarfile.open(str(audit.path(tar_file)), "r:") as archive:
        for member in archive:
            require(member.isfile() and NATIVE.fullmatch(member.name), "unexpected validation archive member")
            source_id = sid(member.name)
            require(source_id not in seen, "duplicate native source")
            seen.add(source_id)
            h = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
                    member_bytes += len(chunk)
            sha = h.hexdigest()
            if source_id in cohort_hashes:
                require(sha == cohort_hashes[source_id], "raw Y2 source JPEG identity mismatch")
            if source_id in union:
                excluded_hashes[source_id] = sha
            wnid = labels[int(NATIVE.search(member.name).group(1)) - 1]
            index = class_map[wnid]
            class_counts[index] += 1
            hashes[sha].append(source_id)
            candidate_rows.append({"source_object_id": source_id, "class_id": index, "official_wnid": wnid, "file_sha256": sha, "file_bytes": member.size})
    require(len(seen) == 50000 and union <= seen, "release or exclusion coverage incomplete")
    audit.bytes_hashed += member_bytes
    excluded_sha_set = set(excluded_hashes.values())
    same_byte_extra = set()
    for row in sorted(candidate_rows, key=lambda r: r["source_object_id"]):
        candidate_digest.update(canonical(row) + b"\n")
        if row["source_object_id"] not in union:
            if row["file_sha256"] in excluded_sha_set:
                same_byte_extra.add(row["source_object_id"])
            else:
                available_counts[row["class_id"]] += 1
    for source_id in same_byte_extra:
        reasons[source_id].add("byte_identical_to_historical_source")
    effective_union = union | same_byte_extra
    duplicate_groups = [{"file_sha256": h, "source_object_ids": sorted(ids)} for h, ids in sorted(hashes.items()) if len(ids) > 1]
    local_names = {p.name for p in (REPO / "assets/data/ILSVRC2012_val").iterdir() if p.is_file()}
    require(all(NATIVE.fullmatch(name) for name in local_names), "unexpected local image-cache file")
    asset_scan = metadata_inventory()
    proposals = audit.read("manifests/stage_x_model_data_inventory_v1.json")
    datasets = [{"dataset_id": d["id"], "historical_materialization_status": d["materialization_status"],
                 "annotation_description_from_prior_plan": d["annotation"], "historical_license_note": d["license"],
                 "current_materialization_status": "LOCAL_CLASS_LABELS_ONLY" if d["id"] == "imagenet_1k" else "NOT_FOUND_IN_AUDITED_ASSET_SCOPE",
                 "pixel_masks_verified": 0} for d in proposals["datasets"]]
    topology = {"source_file": "src/vitnns/stage_y_y1.py", "source_lines": [251, 290],
                "classifier_readout": "model.fc_norm(final_norm[:, 0]) then head; avg or attention pooling explicitly rejected",
                "block11_reason": "No later cross-token operation follows the final residual; patch-destination class-logit derivatives can be zero while CLS-directed source contributions remain nonzero.",
                "source_patch_formula": "a[b,p] = sum_q T[b,p,q] + C[b,p]",
                "aggregation_order": "sum signed input coordinates/channels within source patch, then max(score,0)",
                "sealed_block11_clean_rows_checked": len(primitives), "sealed_rows_with_NA_patch_fraction_and_CLS_fraction_one": len(primitives),
                "new_numeric_topology_computation": False, "evidence_scope": "source-code audit plus hash-verified existing summary; not a new scientific endpoint"}
    return identity, completion, bindings, inventory, model, units, shapes, target_matches, sources, old_count, c_all, y_ids, effective_union, reasons, available_counts, duplicate_groups, same_byte_extra, tar_record, candidate_digest.hexdigest(), member_bytes, local_names, asset_scan, datasets, devkit_members, copying, numeric_gt, synsets, labels_file, topology


def assemble(audit, result):
    (identity, completion, bindings, inventory, model, units, shapes, target_matches, sources, old_count, c_all, y_ids,
     union, reasons, available_counts, duplicates, same_byte_extra, tar_record, candidates_digest, member_bytes,
     local_names, asset_scan, datasets, devkit_members, copying, numeric_gt, synsets, labels_file, topology) = result
    require(not FORBIDDEN_IMPORTS.intersection(sys.modules), "forbidden compute/image library imported")
    common = {"schema_version": 1, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S0", "run_id": "S0_20260914",
              "authorization": "USER_AUTHORIZED_METADATA_AND_SOURCE_AUDIT_ONLY", "s1_authorized": False}
    records = sorted(audit.records.values(), key=lambda r: r["path"])
    asset = {"code_identity": identity, "prior_stage_verification": {
                 "status": "PASS_PACKAGE_BYTE_AND_BINDING_VERIFICATION", "scientific_statistics_recomputed": False,
                 "y4_inventory_file_count": len(inventory["files"]), "y4_inventory_files_digest": inventory["files_digest"],
                 "y4_claim_decisions_unchanged": completion["payload"]["per_claim_decisions"],
                 "prior_next_authorization_unchanged": completion["payload"]["next_authorization"], "predecessor_bindings": bindings},
             "model": model, "source_files": records, "source_files_digest": digest(records),
             "local_asset_scan": asset_scan, "dataset_candidates": datasets,
             "release": {"archive": tar_record, "raw_member_count": 50000, "raw_member_bytes_hashed": member_bytes,
                         "candidate_metadata_digest": candidates_digest, "candidate_digest_algorithm": "SHA256 over canonical JSON plus LF in source_object_id order",
                         "raw_image_decode_count": 0, "local_materialized_jpeg_count": len(local_names),
                         "local_materialized_filenames_digest": digest(sorted(local_names)), "local_cache_bytes_hashed": False,
                         "exact_duplicate_byte_groups": duplicates},
             "devkit": {"members": devkit_members, "copying_sha256": hashlib.sha256(copying).hexdigest(),
                        "copying_applies_to": "devkit software, not a blanket license for images or third-party masks",
                        "numeric_class_label_count": len(numeric_gt), "contains_semantic_or_part_masks": False,
                        "contains_localization_evaluation_code_not_box_ground_truth": True},
             "arrays": {"clean_image_count": len(units), "array_count": len(audit.array_checks),
                        "array_bytes": sum(r["array_bytes"] for r in units), "verified_records_digest": digest(audit.array_checks),
                        "checks": ["all clean NPY file SHA256", "all clean canonical array byte digests", "headers, shapes, dtype, C order, file lengths"],
                        "not_recomputed": ["finite-values scan", "numerical reconstruction residuals", "scientific statistics", "perturbation-source raw array identities"],
                        "shape_counts": [{"dtype": dtype, "shape": list(shape), "count": n} for (dtype, shape), n in sorted(shapes.items())],
                        "native_patch_token_features_retained": False, "stored_target_rule": "clean predicted-class raw logit",
                        "target_class_matches_annotation_class_count": target_matches, "target_class_differs_count": len(units) - target_matches,
                        "scientific_intervention_payload_count": sum(u["scientific_result"] is not None for u in units),
                        "formal_reuse_status": "NOT_ALLOWED_OUTCOME_KNOWN_AND_NEW_TARGET_ENDPOINT_CONTRACT", "units": units},
             "topology": topology,
             "audit_boundary": {"implementation": "Python standard library only; archive image members streamed solely for hashes; NPY coefficients not numerically interpreted",
                                "bytes_hashed": audit.bytes_hashed, "image_decode_count": 0, "model_load_count": 0,
                                "inference_count": 0, "training_count": 0, "attribution_count": 0, "cuda_access_count": 0,
                                "network_requests": 0, "new_scientific_samples_selected": 0}}
    annotation = {
        "status": "DRAFT_GEOMETRY_AND_LABEL_RULES_BLOCKED_DATA",
        "observed_available_annotation": "ImageNet-1k image-level labels only within audited asset scope",
        "verified_object_masks": 0, "verified_part_masks": 0, "verified_boxes": 0,
        "class_mapping": {"label_source": labels_file, "rows": 50000, "class_count": 1000,
                          "canonical_timm_synset_order_sha256": "70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15",
                          "index_rule": "index in verified timm imagenet_synsets.txt; equal to sorted unique release WNIDs",
                          "wnid_to_index": [{"wnid": wnid, "class_id": index} for index, wnid in enumerate(synsets)],
                          "numeric_devkit_partition_agrees_with_wnid_labels": True,
                          "meta_mat_semantics_reparsed": False,
                          "future_segmentation_taxonomy_mapping": "UNBOUND; exact WNID mapping required, no related-category substitution"},
        "preprocessing": {"image_source": "src/vitnns/stage_y_y2_worker.py:70-88",
                          "rounding_source": "torchvision/transforms/functional.py:352-374,582-584 in pinned environment",
                          "image": {"color": "RGB", "resize_short_edge": 248, "resize_long_edge": "int(248 * original_long / original_short)",
                                    "interpolation": "PIL bicubic with antialias=True", "crop_hw": [224, 224],
                                    "crop_top_left": "int(round((resized_height-224)/2)), int(round((resized_width-224)/2)); Python ties-to-even",
                                    "normalization": "(float(pixel)/255 - 0.5) / 0.5", "exif_auto_transpose": False},
                          "proposed_mask": {"source_coordinate_requirement": "same raw stored image orientation and dimensions; otherwise require an explicit proven transform",
                                            "resize_and_crop": "reuse the image's exact integer resized dimensions and crop box",
                                            "interpolation": "nearest-neighbor for discrete semantic/instance IDs; never bicubic label interpolation",
                                            "void_or_unlabeled": "retain as unknown; any patch with unknown pixels is not silently relabeled background; endpoint coverage policy must be frozen before S1/formal use",
                                            "patch_shape": [16, 16], "grid_shape": [14, 14],
                                            "occupancy": "foreground pixels / 256 only for completely annotated native patch",
                                            "execution_status": "NOT_EXECUTED_NO_MASKS_AVAILABLE"}},
        "target_rules": {"primary": "annotation target-class raw logit fixed for all interventions",
                         "same_class_instances": "union all instance masks with the selected exact WNID before occupancy aggregation",
                         "multiple_classes": "select ascending verified canonical class index from fully annotated mapped classes using metadata only; retain selection record",
                         "misclassified": "retain; do not switch target to prediction",
                         "unmapped_class": "BLOCKED_DEFINITION; no semantic-neighbor substitution",
                         "no_target_after_crop": "record typed TARGET_ABSENT_AFTER_CROP; no model-based exclusion or replacement",
                         "part_expansion": "NOT_SCHEDULED; requires independent taxonomy, masks, parent-object mapping, and sufficient per-class splits"},
        "license": {"image_terms": "assets/README.md states ImageNet license and authorized use; local possession does not prove redistribution permission",
                    "devkit_software_license_not_image_license": True,
                    "new_mask_release_terms": "NOT_AVAILABLE_NOT_REVIEWED", "download_authorized": False, "manual_annotation_authorized": False},
        "annotation_qc": {"annotator_input": "original image and written labeling rules only", "forbidden_inputs": ["NNS map", "attention map", "probe prediction", "intervention outcome"],
                          "qc_status": "NOT_STARTED", "required_before_use": ["independent QC sample", "recorded exclusions", "source-image grouping", "void-label policy", "orientation and mask transform tests"]}}
    exclusions = [{"source_object_id": s, "reasons": sorted(reasons[s])} for s in sorted(union)]
    reuse = {"status": "EXCLUSION_AUDIT_COMPLETE_ANNOTATED_COHORT_UNAVAILABLE", "normalization": PREFIX + "ILSVRC2012_val_XXXXXXXX.JPEG",
             "historical_sources": sources, "historical_Y2_union_recomputed_count": old_count,
             "conservative_C_full_registry_count": len(c_all), "stage_Y_count": len(y_ids),
             "stage_Y_overlap_with_new_conservative_C_registry": sorted(c_all & y_ids),
             "overlap_interpretation": "new-study broader exclusion only; old Y4 decisions use their frozen roster and are unchanged",
             "effective_exclusion_count": len(union), "exclusions": exclusions, "exclusions_digest": digest(exclusions),
             "additional_byte_duplicate_exclusions": sorted(same_byte_extra),
             "same_source_grouping": {"canonical_native_id": "release ID plus original source image ID; strip derivative crop/annotation version only with explicit parent lineage",
                                      "exact_byte_aliases": "full official archive member SHA256 compared across names and against historical source hashes",
                                      "cropped_or_reencoded_aliases": "NOT_PROVEN_ABSENT; future annotation release must provide original-image IDs/derivation graph; unresolved records ineligible",
                                      "semantic_near_duplicate_detection": "not performed; image decoding forbidden in S0"},
             "unannotated_release_pool": {"candidate_count": 50000, "remaining_after_native_and_byte_exclusion": sum(available_counts.values()),
                                          "classes_remaining": len(available_counts), "per_class_counts": [{"class_id": c, "remaining": available_counts[c]} for c in range(1000)],
                                          "not_a_confirmatory_cohort": True},
             "new_annotated_candidates": [], "qualified_annotated_candidate_count": 0,
             "split_feasibility": {"status": "BLOCKED_DATA", "train": [], "validation": [], "confirmatory_test": [], "new_development_selection": [],
                                   "reason": "zero verified segmentation labels; independent per-class annotation pools cannot be counted or split",
                                   "group_unit": "original source image and all its aliases, crops, instances and annotations",
                                   "probe_requirement": "each included target class has disjoint train/validation/test images; one image per class is insufficient",
                                   "no_split_selected": True},
             "reuse_policy": [{"asset": "sealed checkpoint bytes and complete-cut source definition", "physical": "READABLE_HASH_VERIFIED", "protocol": "candidate inheritance only; S1 not authorized", "target": "model reusable, attribution target separately fixed"},
                              {"asset": "100 Stage Y clean contribution bundles", "physical": "262 arrays per image verified", "protocol": "exploratory/engineering only; never new confirmatory rows", "target": "predicted-class target; even matching true-class IDs do not restore independence"},
                              {"asset": "native token features for ridge probe", "physical": "NOT_RETAINED_IN_CLEAN_BUNDLES", "protocol": "future extraction requires S1/S3 authorization", "target": "representation tensor is not a contribution matrix"},
                              {"asset": "Y4 valid masks and rendered heatmaps", "physical": "generated analysis assets", "protocol": "not semantic annotations", "target": "cannot act as ground truth"}]}
    decision = {"audit_workflow_status": "COMPLETE", "feasibility_status": "BLOCKED_DATA", "can_enter_S1": False, "next_authorization": "NONE",
                "decision_basis": "known local ImageNet images and class labels exist, but no verified object/part masks or qualified annotated train/validation/test pool",
                "gates": [{"id": "PREDECESSOR_PACKAGE", "status": "PASS", "detail": "Y4 package bytes/canonical bindings and clean array identities verified; statistics not rerun"},
                          {"id": "MODEL_BYTES", "status": "PASS", "detail": "checkpoint SHA256 matches recovery-v2; no deserialization"},
                          {"id": "IMAGENET_CLASS_INDEX", "status": "PASS", "detail": "50k labels, 1000-class timm order and official numeric label partition checked"},
                          {"id": "OBJECT_SEGMENTATION", "status": "BLOCKED_DATA", "detail": "zero verified masks in audited asset scope"},
                          {"id": "ANNOTATION_CLASS_AND_LICENSE_BINDING", "status": "BLOCKED_DATA", "detail": "new annotation release, taxonomy and terms absent"},
                          {"id": "INDEPENDENT_ANNOTATED_SPLITS", "status": "BLOCKED_DATA", "detail": "historical exclusions rebuilt; qualified annotated candidates=0"},
                          {"id": "TOPOLOGY_DEFINITION", "status": "PASS_SOURCE_AUDIT_ONLY", "detail": "CLS-only readout identified; source patches must include T plus C; S1 empirical tests not performed"}],
                "what_can_be_inherited": ["hash-verified checkpoint bytes", "complete-cut/CLS accounting definitions", "historical exclusion provenance", "old arrays for authorized exploration only"],
                "required_to_unblock": ["provide a licensed object segmentation release or authorize independent annotation acquisition",
                                        "bind every mask to original source image, exact WNID, version, hash, and orientation",
                                        "re-run source-level and byte-alias exclusion against the expanded historical roster",
                                        "demonstrate multiple independent annotated images per included class for train/validation/test and isolated pilot",
                                        "approve a prospective S0 continuation/new root; no patching of this terminal evidence"],
                "candidate_route_not_executed": "ImageNet-S is already listed in the prior inventory but not materialized; evaluate its actual release, taxonomy, source overlap and terms before any download or adoption",
                "not_authorized": ["S1", "image decode", "GPU", "inference", "probe training", "attribution", "downloads", "Y4 amendment", "Y5", "F3-Final", "Stage G", "commit", "push"],
                "limitations": ["absence statement is scoped to the audited local assets, not the whole machine",
                                "no new annotation dataset or crop-lineage pool was supplied",
                                "no tensor digest recomputation, new numerical reconstruction, power calculation or outcome analysis"],
                "artifacts": []}
    payloads = dict(zip(NAMES, (asset, annotation, reuse, decision)))
    docs = {}
    for name in NAMES:
        if name == "feasibility_decision":
            decision["artifacts"] = [{"name": n + ".json", "canonical_digest": docs[n]["canonical_digest"],
                                      "sha256": hashlib.sha256(serialized(docs[n])).hexdigest()} for n in NAMES[:-1]]
        document = {**common, "document_type": name, "payload": payloads[name]}
        document["canonical_digest"] = digest(document)
        docs[name] = document
    require(len(audit.array_checks) == 26200, "clean array coverage incomplete")
    require(sum(available_counts.values()) + len(union) == 50000, "candidate partition mismatch")
    require(git("rev-parse", "HEAD") == identity["head"] and not git("status", "--porcelain", "--untracked-files=no"), "tracked identity changed during S0")
    return docs


def serialized(document):
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_ROOT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = (REPO / args.output).resolve()
    require(REPO / "evidence/nns_vit_patch_depth" in root.parents, "output outside new-study evidence namespace")
    require(root.exists() if args.verify else not root.exists(), "verify needs existing root; production needs fresh root")
    audit = Audit()
    documents = assemble(audit, build(audit))
    if args.verify:
        require({p.name for p in root.iterdir()} == {n + ".json" for n in NAMES}, "unexpected evidence files")
        previous = {}
        for name in NAMES:
            old = json.loads((root / (name + ".json")).read_text())
            require(old["canonical_digest"] == digest({k: v for k, v in old.items() if k != "canonical_digest"}), "output canonical digest mismatch")
            previous[name] = old
        documents["asset_inventory"]["payload"]["code_identity"]["status_at_audit_start"] = previous["asset_inventory"]["payload"]["code_identity"]["status_at_audit_start"]
        asset = documents["asset_inventory"]
        asset["canonical_digest"] = digest({k: v for k, v in asset.items() if k != "canonical_digest"})
        decision = documents["feasibility_decision"]
        decision["payload"]["artifacts"] = [{"name": n + ".json", "canonical_digest": documents[n]["canonical_digest"], "sha256": hashlib.sha256(serialized(documents[n])).hexdigest()} for n in NAMES[:-1]]
        decision["canonical_digest"] = digest({k: v for k, v in decision.items() if k != "canonical_digest"})
        for name in NAMES:
            require(documents[name] == previous[name], "source-recomputed artifact mismatch: " + name)
        print("S0 verification PASS: exact source-recomputed document comparison; existing files unchanged")
    else:
        root.mkdir(parents=True, exist_ok=False)
        for name in NAMES:
            with (root / (name + ".json")).open("xb") as stream:
                stream.write(serialized(documents[name]))
        print("S0 audit COMPLETE; feasibility BLOCKED_DATA; S1 not authorized")
    print(json.dumps({"root": str(root), "array_count": len(audit.array_checks), "excluded_images": documents["reuse_and_exclusion_manifest"]["payload"]["effective_exclusion_count"],
                      "qualified_annotated_candidates": 0, "bytes_hashed": audit.bytes_hashed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
