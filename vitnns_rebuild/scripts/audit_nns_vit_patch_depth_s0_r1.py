#!/usr/bin/env python3
import argparse
import ast
import collections
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import tarfile


REPO = Path(__file__).resolve().parents[1]
S0 = REPO / "evidence/nns_vit_patch_depth/s0_20260914"
PIN = "b2312ba7396151f883362bdc86d71469aa81066e"
NAMES = ("asset_inventory", "license_and_source_review", "annotation_contract", "image_annotation_class_manifest",
         "reuse_and_exclusion_manifest", "split_feasibility", "annotation_qc_receipt", "feasibility_decision")
MERGE = {"n04356056": "n04355933", "n04493381": "n02808440", "n03642806": "n03832673", "n04008634": "n03773504", "n03887697": "n15075141"}
POOLS = ("pilot", "probe_train", "probe_validation", "confirmatory_test", "official_validation_unassigned")
SPEC = importlib.util.spec_from_file_location("s0_identity_helpers", REPO / "scripts/audit_nns_vit_patch_depth_s0.py")
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(value):
    return dict(value, canonical_digest=digest(value))


def read_signed(path):
    value = json.loads(path.read_text())
    require(value["canonical_digest"] == digest({k: v for k, v in value.items() if k != "canonical_digest"}), "digest mismatch: " + str(path))
    return value


def record(path):
    h = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file changed during hash")
    return {"path": os.path.relpath(str(path), str(REPO)), "bytes": before.st_size, "sha256": h.hexdigest()}


def jpeg_header(raw):
    require(raw[:2] == b"\xff\xd8", "not JPEG")
    offset = 2
    width = height = None
    orientation = None
    exif_state = "ABSENT"
    sof = {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}
    while offset < len(raw):
        require(raw[offset] == 0xff, "JPEG marker alignment")
        while offset < len(raw) and raw[offset] == 0xff:
            offset += 1
        require(offset < len(raw), "truncated marker")
        marker = raw[offset]
        offset += 1
        if marker in (0xda, 0xd9):
            break
        if marker in {0x01, 0xd8, *range(0xd0, 0xd8)}:
            continue
        require(offset + 2 <= len(raw), "truncated segment")
        size = int.from_bytes(raw[offset:offset + 2], "big")
        require(size >= 2 and offset + size <= len(raw), "invalid segment length")
        segment = raw[offset + 2:offset + size]
        if marker in sof:
            require(len(segment) >= 6, "short SOF")
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
        if marker == 0xe1 and segment.startswith(b"Exif\0\0"):
            t = segment[6:]
            try:
                require(len(t) >= 8 and t[:2] in (b"II", b"MM"), "bad TIFF header")
                order = "<" if t[:2] == b"II" else ">"
                require(struct.unpack_from(order + "H", t, 2)[0] == 42, "bad TIFF version")
                pos = struct.unpack_from(order + "I", t, 4)[0]
                count = struct.unpack_from(order + "H", t, pos)[0]
                require(pos + 2 + count * 12 <= len(t), "short EXIF IFD")
                exif_state = "ORIENTATION_TAG_ABSENT"
                for index in range(count):
                    p = pos + 2 + index * 12
                    tag, kind, count_value = struct.unpack_from(order + "HHI", t, p)
                    if tag == 274:
                        require(kind == 3 and count_value == 1, "invalid EXIF orientation")
                        orientation = struct.unpack_from(order + "H", t, p + 8)[0]
                        require(1 <= orientation <= 8, "out-of-range orientation")
                        exif_state = "PARSED"
            except (ValueError, struct.error):
                exif_state = "MALFORMED_OR_UNSUPPORTED"
                orientation = None
        offset += size
    require(width and height, "missing JPEG dimensions")
    return {"width": width, "height": height, "exif_orientation": orientation, "exif_state": exif_state,
            "rgb_pixels_decoded": False, "effective_transform": "no automatic EXIF transpose; annotation orientation not yet checked"}


def safe_relative(value):
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value, "unsafe official relative path")
    return path


def collect(root):
    acquisition = read_signed(root / "acquisition_plan.json")
    require(OLD.git("rev-parse", "HEAD") == acquisition["head"], "HEAD changed since metadata acquisition")
    require(not OLD.git("status", "--porcelain", "--untracked-files=no"), "tracked source changed during audit")
    a = OLD.Audit()
    inherited = []
    predecessor = {}
    for name in OLD.NAMES:
        path = S0 / (name + ".json")
        predecessor[name] = read_signed(path)
        inherited.append(dict(record(path), canonical_digest=predecessor[name]["canonical_digest"]))
    require(predecessor["feasibility_decision"]["payload"]["feasibility_status"] == "BLOCKED_DATA", "S0 terminal changed")
    for name in ("docs/nns_vit_patch_depth_research_goals_20260914.md", "docs/nns_vit_patch_depth_execution_plan_20260914.md", "config/recovery_v2_binding.json",
                 "manifests/stage_x_model_data_inventory_v1.json", "src/vitnns/stage_y_y2_worker.py", "src/vitnns/stage_y_y1.py"):
        a.record(name)
    binding = a.read("config/recovery_v2_binding.json")
    a.record(binding["model"]["state_dict_path"], {"sha256": binding["model"]["state_dict_sha256"], "bytes": binding["model"]["state_dict_bytes"]})
    sources = {}
    source_records = []
    for path in sorted((root / "source_evidence").iterdir()):
        require(path.suffix == ".json", "unexpected source evidence")
        value = read_signed(path)
        if value["http_status"] == 200:
            require(value["body_sha256"] == hashlib.sha256(value["body_text"].encode()).hexdigest(), "response body digest")
            if value["decoded_blob"] is not None:
                b = value["decoded_blob"]
                require(b["sha256"] == hashlib.sha256(b["text"].encode()).hexdigest(), "blob SHA256")
                require(b["git_blob_sha1"] == hashlib.sha1(b"blob " + str(b["bytes"]).encode() + b"\0" + b["text"].encode()).hexdigest(), "blob git identity")
        sources[path.stem] = value
        source_records.append(record(path))
    def blob(key):
        v = sources[key]
        require(v["http_status"] == 200 and v["decoded_blob"] is not None, "required official source unavailable: " + key)
        return v["decoded_blob"]["text"]
    def api(key):
        v = sources[key]
        require(v["http_status"] == 200, "required API unavailable: " + key)
        return json.loads(v["body_text"])
    tree = api("imagenets_tree")
    require(not tree["truncated"] and api("imagenets_head")["sha"] == PIN, "official commit mismatch")
    require(api("imagenets_repository")["license"] is None, "license changed; review before proceeding")
    require(not any(re.search(r"(^|/)(license|copying|terms)(\.|$)", x["path"], re.I) for x in tree["tree"]), "new license file needs review")
    readme = blob("imagenets_README_md")
    require("merge = " in readme, "missing class merge documentation")
    parsed_merge = ast.literal_eval(re.search(r"merge = (\{[^}]+\})", readme).group(1))
    require(parsed_merge == MERGE, "merge definition changed")
    require("the segmentation mask is stored on the online evalution server" in readme, "test access policy changed")
    require("R+G*256" in readme and "1000" in readme, "mask encoding contract changed")
    categories = blob("imagenets_data_categories_ImageNetS_categories_im919_txt").splitlines()
    require(len(categories) == len(set(categories)) == 919, "category roster invalid")
    label_file = "assets/imagenet/imagenet_2012_validation_synset_labels.txt"
    label_record = a.record(label_file, {"sha256": "313fc3f23c864dde9183c9a368809065765dd1b657a6904fded44e3e46d34604"})
    labels = a.path(label_file).read_text().splitlines()
    synset_file = "../.conda/envs/vitnns/lib/python3.10/site-packages/timm/data/_info/imagenet_synsets.txt"
    a.record(synset_file, {"sha256": "70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15"})
    synsets = a.path(synset_file).read_text().splitlines()
    require(len(labels) == 50000 and categories == sorted(categories) and set(categories) <= set(synsets), "class roster mismatch")
    index = {wnid: i for i, wnid in enumerate(synsets)}
    annotation_index = {wnid: i + 1 for i, wnid in enumerate(categories)}
    old_reasons = collections.defaultdict(set)
    old_sources = []
    exclusion_plan = a.read("manifests/stage_y/cohort_selection_and_exclusion_plan.json")
    def include(stage, path):
        a.record(path)
        if path.endswith(".csv"):
            import csv
            with a.path(path).open() as f:
                ids = {OLD.sid(r.get("source_object_id") or r["sample_id"]) for r in csv.DictReader(f)}
        else:
            ids = OLD.identity_strings(a.read(path))
        for value in ids:
            old_reasons[value].add(stage)
        old_sources.append({"stage": stage, "source": a.records[path], "count": len(ids), "ids_digest": digest(sorted(ids))})
    for source in exclusion_plan["prior_access_exclusion_sources"]:
        include(source["stage"], source["path"])
    sealed_y2 = a.read(OLD.Y2 + "/exclusion_union.json")["payload"]
    require(sorted(old_reasons) == sealed_y2["source_object_ids"], "old exclusion source drift")
    include("Stage_C_full_registered_500_conservative", "manifests/stage_c_flat_val_500/manifest.csv")
    include("Stage_Y_Y2_Y3_Y4_outcome_known", OLD.Y2 + "/cohort_registry.json")
    native_excluded = set(old_reasons)
    tar_path = REPO / "assets/imagenet/ILSVRC2012_img_val.tar"
    tar_record = record(tar_path)
    require(tar_record["sha256"] == predecessor["asset_inventory"]["payload"]["release"]["archive"]["sha256"], "official image archive drift")
    by_name = {}
    by_hash = collections.defaultdict(list)
    print("S0-R1: stream official image bytes and parse JPEG headers only", flush=True)
    with tarfile.open(str(tar_path), "r:") as archive:
        for member in archive:
            require(member.isfile() and OLD.NATIVE.fullmatch(member.name), "unexpected official archive entry")
            safe_relative(member.name)
            with archive.extractfile(member) as f:
                raw = f.read()
            name = member.name
            sha = hashlib.sha256(raw).hexdigest()
            try:
                header = jpeg_header(raw)
                header_error = None
            except ValueError as e:
                header = None
                header_error = str(e)
            source_id = OLD.sid(name)
            item = {"source_id": source_id, "member": name, "bytes": len(raw), "sha256": sha,
                    "header": header, "header_error": header_error,
                    "wnid": labels[int(OLD.NATIVE.fullmatch(name).group(1)) - 1]}
            require(name not in by_name, "duplicate archive filename")
            by_name[name] = item
            by_hash[sha].append(source_id)
    require(len(by_name) == 50000, "official archive incomplete")
    old_hashes = {item["sha256"] for item in by_name.values() if item["source_id"] in native_excluded}
    for item in by_name.values():
        if item["sha256"] in old_hashes and item["source_id"] not in native_excluded:
            old_reasons[item["source_id"]].add("byte_identical_to_historical_source")
    require(set(old_reasons) == {x["source_object_id"] for x in predecessor["reuse_and_exclusion_manifest"]["payload"]["exclusions"]}, "S0 expanded exclusions drift")
    candidates = []
    validation_names = {}
    for line in blob("imagenets_data_names_ImageNetS_im919_validation_txt").splitlines():
        image, mask = line.split()
        safe_relative(image)
        safe_relative(mask)
        validation_names[image] = mask
    def candidate(split, image, dest, mask):
        safe_relative(image)
        safe_relative(dest)
        if mask is not None:
            safe_relative(mask)
        annotation_wnid = dest.split("/")[0]
        require(annotation_wnid in annotation_index, "annotation category missing")
        name = PurePosixPath(image).name
        if split == "train-semi":
            original_wnid = name.split("_")[0]
            source_id = "ILSVRC2012_TRAIN_OFFICIAL:" + original_wnid + "/" + name
            src = None
        else:
            src = by_name[name]
            original_wnid = src["wnid"]
            source_id = src["source_id"]
            require(image.split("/")[0] == original_wnid, "source WNID disagrees with official labels")
        require(original_wnid in index, "original class outside classifier mapping")
        relation = "IDENTICAL" if original_wnid == annotation_wnid else "DOCUMENTED_CATEGORY_MERGE" if MERGE.get(original_wnid) == annotation_wnid else "OFFICIAL_SOURCE_TO_ANNOTATION_RELABELING"
        exact = annotation_wnid not in set(MERGE.values())
        reasons = ["ANNOTATION_LICENSE_UNRESOLVED", "MASK_NOT_ACQUIRED_OR_QC_VERIFIED"]
        if src is None:
            reasons.append("TRAIN_ORIGINAL_IMAGE_NOT_LOCAL")
        if split == "test":
            reasons.append("OFFICIAL_TEST_MASK_SERVER_ONLY")
        if source_id in old_reasons:
            reasons.append("HISTORICAL_SOURCE_OR_BYTE_ALIAS")
        if not exact:
            reasons.append("MERGED_ANNOTATION_CATEGORY_NOT_EXACT_SINGLE_CLASS")
        if src is not None and src["header_error"] is not None:
            reasons.append("IMAGE_HEADER_UNVERIFIED")
        group = "sha256:" + src["sha256"] if src else source_id
        candidates.append({"dataset_id": "ImageNet-S919", "dataset_commit": PIN, "official_split": split,
                           "candidate_id": "ImageNetS919/" + split + "/" + dest,
                           "annotation_id": "ImageNetS919/" + ("validation-segmentation/" if split == "validation" else "train-semi-segmentation/") + mask if mask is not None else None,
                           "source_image_id": source_id, "original_release": "ILSVRC2012_TRAIN_OFFICIAL" if src is None else "ILSVRC2012_VAL_OFFICIAL",
                           "official_source_relative_path": image, "official_dataset_relative_path": dest,
                           "original_wnid": original_wnid, "original_model_class_index": index[original_wnid],
                           "annotation_wnid": annotation_wnid, "annotation_label_id": annotation_index[annotation_wnid],
                           "exact_target_class_index": index[annotation_wnid] if exact else None,
                           "original_to_annotation_class_relation": relation,
                           "target_mapping_basis": "official split destination WNID and exact canonical class table; mask target presence not yet verified",
                           "image_archive": str(tar_path.relative_to(REPO)) if src else None, "image_member": name if src else None,
                           "image_sha256": src["sha256"] if src else None, "image_bytes": src["bytes"] if src else None,
                           "image_header": src["header"] if src else None, "image_header_error": src["header_error"] if src else None,
                           "annotation_relative_path": mask, "annotation_local_path": None, "annotation_sha256": None,
                           "annotation_dimensions": None, "annotation_encoding": "official RGB channels encode class_id=R+256*G; zero=other category; 1000=ignored",
                           "annotation_source_type": "OFFICIAL_HUMAN_LABEL_CLAIM_NOT_LOCALLY_QUALITY_VERIFIED",
                           "license_evidence_ref": "license_and_source_review.json", "parent_lineage": "official preparation scripts copy original bytes without crop or reencoding",
                           "group_id": group, "byte_alias_verified": src is not None,
                           "provisional_pool": None, "eligibility_status": "INELIGIBLE", "exclusion_reasons": sorted(reasons)})
    for split in ("validation", "test"):
        key = "imagenets_data_mapping_ImageNetS_im919_mapping_%s_txt" % split
        for line in blob(key).splitlines():
            image, dest = line.split()
            mask = validation_names[dest] if split == "validation" else None
            candidate(split, image, dest, mask)
    for line in blob("imagenets_data_names_ImageNetSemi_im919_train_txt").splitlines():
        image, mask = line.split()
        candidate("train-semi", image, image, mask)
    require(len(candidates) == 49032, "official candidate count drift")
    require(len({c["source_image_id"] for c in candidates}) == len(candidates), "official source split overlaps")
    train_by_class = collections.defaultdict(list)
    for c in candidates:
        if c["official_split"] == "train-semi":
            train_by_class[c["annotation_wnid"]].append(c)
        else:
            c["provisional_pool"] = "confirmatory_test" if c["official_split"] == "test" else "official_validation_unassigned"
    for wnid, rows in train_by_class.items():
        require(len(rows) == 10, "official train-semi class count changed")
        rows.sort(key=lambda c: (hashlib.sha256(("S0_R1_METADATA_RESERVATION_V1\0" + c["source_image_id"]).encode()).hexdigest(), c["source_image_id"]))
        for rank, c in enumerate(rows):
            c["provisional_pool"] = "pilot" if rank < 2 else "probe_train" if rank < 8 else "probe_validation"
    groups = collections.defaultdict(set)
    for c in candidates:
        groups[c["group_id"]].add(c["provisional_pool"])
    cross_groups = {g: sorted(p) for g, p in groups.items() if len(p) > 1}
    for c in candidates:
        if c["group_id"] in cross_groups:
            c["exclusion_reasons"].append("OFFICIAL_SPLIT_BYTE_ALIAS_QUARANTINED")
            c["exclusion_reasons"].sort()
    candidates.sort(key=lambda c: (c["official_split"], c["source_image_id"]))
    images_digest = digest(sorted(by_name.values(), key=lambda x: x["source_id"]))
    asset_roots = []
    for p in sorted((REPO / "assets").iterdir()):
        asset_roots.append({"path": str(p.relative_to(REPO)), "is_directory": p.is_dir()})
    train_files = []
    for d, ds, fs in os.walk(REPO / "assets"):
        for name in fs:
            if re.fullmatch(r"n\d{8}_\d+\.(JPEG|jpg)", name):
                train_files.append(str((Path(d) / name).relative_to(REPO)))
    require(not train_files, "train images newly present; rerun binding logic with explicit path")
    return {"acquisition": acquisition, "audit": a, "inherited": inherited, "predecessor": predecessor,
            "source_records": source_records, "sources": sources, "readme": readme, "categories": categories,
            "annotation_index": annotation_index, "class_index": index, "candidate_records": candidates,
            "old_sources": old_sources, "old_reasons": old_reasons, "images_digest": images_digest,
            "tar_record": tar_record, "asset_roots": asset_roots, "cross_groups": cross_groups,
            "header_failure_count": sum(x["header_error"] is not None for x in by_name.values()),
            "response_unavailable": [{"source_key": k, "url": v["url"], "error": v["error"]} for k, v in sources.items() if v["error"]]}


def build_documents(root, c):
    candidates = c["candidate_records"]
    source_ref = lambda key: str((root / "source_evidence" / (key + ".json")).relative_to(REPO))
    readme_ref = source_ref("imagenets_README_md")
    release_response = json.loads(c["sources"]["imagenets_releases"]["body_text"])
    assets = [a for r in release_response for a in r["assets"] if a["name"].startswith("ImageNetS919-")]
    require(len(assets) == 1, "ambiguous annotation release")
    meta = assets[0]
    require(meta["size"] == 61281494 and meta.get("digest") is None, "release metadata changed")
    files = c["inherited"] + sorted(c["audit"].records.values(), key=lambda r: r["path"]) + [c["tar_record"]]
    implementation = [record(REPO / "scripts" / name) for name in ("acquire_nns_vit_patch_depth_s0_r1_metadata.py", "audit_nns_vit_patch_depth_s0_r1.py", "audit_nns_vit_patch_depth_s0.py")]
    counts = collections.Counter(r["official_split"] for r in candidates)
    per_class = []
    for wnid in c["categories"]:
        rows = [r for r in candidates if r["annotation_wnid"] == wnid]
        per_class.append({"annotation_wnid": wnid, "annotation_label_id": c["annotation_index"][wnid],
                          "exact_single_class": wnid not in set(MERGE.values()), "model_class_index_for_unmerged_wnid": c["class_index"][wnid],
                          "official_counts": {s: sum(r["official_split"] == s for r in rows) for s in ("train-semi", "validation", "test")},
                          "local_image_bound_count": sum(r["image_sha256"] is not None for r in rows),
                          "historical_excluded_count": sum("HISTORICAL_SOURCE_OR_BYTE_ALIAS" in r["exclusion_reasons"] for r in rows),
                          "exact_class_bound_count": sum(r["exact_target_class_index"] is not None for r in rows),
                          "verified_mask_count": 0, "eligible_count": 0,
                          "provisional_pool_counts": {p: sum(r["provisional_pool"] == p for r in rows) for p in POOLS},
                          "qualified_pool_counts": {p: 0 for p in POOLS}})
    pair_checks = []
    for left, right in itertools.combinations(POOLS, 2):
        l = [r for r in candidates if r["provisional_pool"] == left]
        r = [r for r in candidates if r["provisional_pool"] == right]
        common_ids = {x["source_image_id"] for x in l} & {x["source_image_id"] for x in r}
        common_hashes = {x["image_sha256"] for x in l if x["image_sha256"]} & {x["image_sha256"] for x in r if x["image_sha256"]}
        complete = all(x["image_sha256"] is not None for x in l + r)
        pair_checks.append({"left": left, "right": right, "source_id_intersection_count": len(common_ids),
                            "known_byte_hash_intersection_count": len(common_hashes), "known_byte_hash_intersection_digest": digest(sorted(common_hashes)),
                            "byte_check_complete": complete, "status": "COLLISION_QUARANTINED" if common_hashes else "PASS_KNOWN_BYTES" if complete else "UNVERIFIED_MISSING_ORIGINALS"})
    block_reasons = collections.Counter(reason for r in candidates for reason in r["exclusion_reasons"])
    asset = {"status": "METADATA_ONLY_ACQUISITION_COMPLETE", "code_identity": {"head": c["acquisition"]["head"], "worktree_status_at_start": c["acquisition"]["worktree_status"],
             "implementation": implementation, "python_executable_realpath": os.path.realpath(sys.executable), "python_version": sys.version,
             "commit_or_push_performed": False}, "inherited_files": files, "inherited_files_digest": digest(files), "official_source_snapshots": c["source_records"],
             "official_source_snapshots_digest": digest(c["source_records"]), "annotation_release": {"name": meta["name"], "size_bytes": meta["size"], "url": meta["browser_download_url"],
             "published_digest": meta.get("digest"), "local_file": None, "local_sha256": None, "status": "NOT_DOWNLOADED_LICENSE_GATE"},
             "original_images": {"validation_archive": c["tar_record"], "member_count_verified": 50000, "all_member_hash_and_header_digest": c["images_digest"],
             "candidate_bound_count": sum(r["image_sha256"] is not None for r in candidates), "candidate_train_missing_count": counts["train-semi"],
             "all_release_header_failure_count": c["header_failure_count"], "train_scan_scope": "active project assets only", "asset_root_entries": c["asset_roots"], "new_original_download_bytes": 0},
             "acquisition_plan_ref": str((root / "acquisition_plan.json").relative_to(REPO)), "unavailable_sources": c["response_unavailable"],
             "access_boundary": {"rgb_decode_count": 0, "mask_decode_count": 0, "model_load_count": 0, "gpu_access_count": 0, "inference_count": 0,
             "training_count": 0, "attribution_count": 0, "scientific_outcomes_used_for_selection": False, "remote_upload_count": 0}}
    license_review = {"status": "BLOCKED_LICENSE", "adopted_dataset": None,
        "primary_candidate": {"dataset": "ImageNet-S919", "official_repository": "https://github.com/LUSSeg/ImageNet-S", "commit": PIN,
        "documentation_status": "official README, project, release, category and source-mapping metadata snapshotted",
        "code_license": "GitHub license field null and no LICENSE/COPYING/TERMS file in complete pinned tree; no blanket grant inferred",
        "annotation_license": "No explicit annotation-use terms found in audited official README/project/release; unresolved, not a determination that permission cannot exist",
        "image_license": "ImageNet access terms require non-commercial research/education and named-researcher agreement; prior local custody does not document acceptance for this new release",
        "annotation_truth_evidence": "README describes precise pixel-level masks and human-labeled semi-supervised training data; actual masks not acquired; no pseudo-label or model output accepted",
        "human_quality_control_evidence": "official high-quality claim only; detailed paper/QC procedure not verified because full text unavailable",
        "official_train_semi_claim": "9190 images, 10 per category, original ImageNet train",
        "official_validation_claim": "12419 original ImageNet validation images; public segmentation archive candidate",
        "official_test_claim": "27423 original ImageNet validation images; ground truth held by online evaluation server",
        "evidence_refs": [readme_ref, source_ref("imagenets_tree"), source_ref("imagenets_repository"), source_ref("imagenets_releases"), source_ref("imagenets_project"), source_ref("imagenet_terms")]},
        "alternative_candidate": {"dataset": "PartImageNet", "official_repository": "https://github.com/TACJu/PartImageNet", "commit": "f4bf3df88b126d3a2d5e8671a8c2ea90de39638e",
        "readme_distinction": "Seg has shared classes across splits; OOD has disjoint classes and cannot silently substitute for per-class probes",
        "reported_total": 24095, "reported_classes": 158, "reported_count_scope": "README legacy/OOD statistics, not locally verified Seg split counts",
        "license": "official repo license null; linked author dataset card unreachable, so annotation terms and package contents unverified",
        "status": "NOT_ADOPTED_LICENSE_AND_ACCESS_UNRESOLVED", "evidence_refs": [source_ref("partimagenet_repository"), source_ref("partimagenet_tree"), source_ref("partimagenet_readme"), source_ref("partimagenet_dataset_card")]},
        "download_policy": "No annotation archive downloaded before permission was established; no signup, agreement acceptance or third-party upload",
        "metadata_read_scope": "public source documentation and file indexes retained for audit only; not reused as licensed annotation payload",
        "resolution_required": ["explicit applicable annotation license or author permission", "user confirmation of applicable ImageNet access/use authorization", "precise intended use and no unauthorized redistribution"]}
    old_annotation = c["predecessor"]["annotation_contract"]["payload"]
    annotation = {"status": "DRAFT_DATASET_SPECIFIC_CONTRACT_NOT_MASK_VALIDATED", "source_ref": readme_ref,
        "dataset": "ImageNet-S919", "commit": PIN, "geometry": old_annotation["preprocessing"],
        "mask_encoding": {"file_format": "PNG with RGB label-encoding channels", "class_id_formula": "uint16(R)+256*uint16(G)",
        "object_ids": "1..919 indexed by category-file row", "other_category_id": 0, "ignored_id": 1000,
        "zero_semantics": "other category, not guaranteed absence of all physical objects", "ignored_handling": "unknown, never background",
        "blue_channel_policy": "document encoding examples use zero; unexpected values require typed review, not silent channel coercion",
        "not_allowed": ["convert RGB-coded labels to grayscale", "palette conversion that loses label IDs", "bicubic mask interpolation"]},
        "mapping": [{"annotation_label_id": i + 1, "wnid": w, "model_class_index": c["class_index"][w], "exact_single_class_eligible": w not in set(MERGE.values())} for i, w in enumerate(c["categories"])],
        "merged_categories": MERGE, "merged_category_policy": "quarantine all five merged destination categories; even destination-named images may contain pixels from either original class",
        "same_class_instances": "semantic mask union for the exact target label; no part/instance identity inferred",
        "multi_class_selection": "use existing S0 metadata-only smallest exact mapped class rule if explicit class-presence metadata exists; otherwise no guessing from unseen pixels",
        "target_scope": "annotation target-class raw logit, never predicted-class selection",
        "patch_occupancy": "foreground/256 for fully annotated 16x16 patch after exact shared resize/crop; void-containing patch remains typed invalid",
        "orientation": "source JPEG header available; annotation dimensions/orientation not yet available; require pairing before claiming geometry PASS",
        "failure_states": ["MASK_NOT_AVAILABLE", "UNMAPPED_OR_MERGED_CLASS", "MASK_LABEL_OUT_OF_RANGE", "MASK_IMAGE_DIMENSION_MISMATCH", "ORIENTATION_UNRESOLVED", "VOID_COVERAGE", "TARGET_ABSENT_AFTER_CROP"],
        "status_of_tests": "NOT_RUN_LICENSE_GATE; no mask pixel content read"}
    manifest = {"status": "METADATA_CANDIDATES_NOT_ADMITTED", "record_count": len(candidates), "official_split_counts": dict(counts),
                "record_contract": "every row has fixed fields; unavailable actual file identity/dimensions represented by null, never fabricated",
                "records": candidates, "records_digest": digest(candidates), "annotation_file_count_acquired": 0,
                "qualified_record_count": 0, "public_test_truth_available": False}
    exclusions = [{"source_image_id": s, "reasons": sorted(rs)} for s, rs in sorted(c["old_reasons"].items())]
    reuse = {"status": "SOURCE_ID_AND_AVAILABLE_BYTE_AUDIT_COMPLETE", "historical_sources": c["old_sources"],
             "historical_exclusions": exclusions, "historical_exclusion_count": len(exclusions), "historical_exclusions_digest": digest(exclusions),
             "candidate_manifest_records_digest": manifest["records_digest"], "candidate_historical_collision_count": block_reasons["HISTORICAL_SOURCE_OR_BYTE_ALIAS"],
             "candidate_reason_counts": dict(block_reasons), "cross_official_split_byte_groups": [{"group_id": g, "pools": p} for g, p in sorted(c["cross_groups"].items())],
             "grouping": {"validation": "full original JPEG SHA256 plus canonical original source ID", "train": "original ImageNet train WNID/file source ID only; bytes unavailable",
             "declared_transform": "official scripts copy originals unchanged, category-directory renaming recorded", "near_duplicates": "not assessed; no RGB decoding",
             "unresolved_lineage": "no derivative/crop assumption made beyond official copy mapping; external aliases remain unproven"},
             "quarantine_rule": "same bytes crossing official purposes are withheld, not moved into a favorable split",
             "formal_reuse_of_old_outcomes": False, "source_pool_ready": False}
    pool_counts = {p: sum(r["provisional_pool"] == p for r in candidates) for p in POOLS}
    splits = {"status": "BLOCKED_SPLIT", "candidate_manifest_records_digest": manifest["records_digest"], "provisional_counts": pool_counts,
              "qualified_counts": {p: 0 for p in POOLS}, "per_class": per_class, "per_class_digest": digest(per_class), "pairwise_checks": pair_checks,
              "rule": {"namespace": "S0_R1_METADATA_RESERVATION_V1", "train_semi": "per category ascending SHA256(namespace+NUL+original source ID): first 2 pilot, next 6 probe_train, final 2 probe_validation",
              "test": "official test reserved only for potential confirmatory_test; currently server-only truth and ineligible",
              "validation": "official validation left unassigned; no repurposing into training without prospective approval",
              "not_formal_cohort": True, "mask_pixels_used": False},
              "adequacy": "6 train and 2 validation image IDs per class is only a metadata reservation, not evidence of adequate probe training or statistical power; originals are absent",
              "blocking_conditions": ["no licensed locally validated masks", "train originals absent", "official test truth not locally released", "cross-source byte alias checks incomplete for absent training images"],
              "statistical_power_status": "NOT_EVALUATED_S2_REQUIRED"}
    qc = {"status": "NOT_RUN_BLOCKED_LICENSE", "candidate_manifest_records_digest": manifest["records_digest"],
          "scope": "official metadata and JPEG header audit only; no masks acquired or decoded", "planned_mask_count": counts["validation"] + counts["train-semi"],
          "acquired_mask_count": 0, "decoded_mask_count": 0, "mask_pass_count": 0, "mask_failed_count": 0, "mask_not_run_count": counts["validation"] + counts["train-semi"],
          "header_audit": {"all_local_release_members": 50000, "header_parse_failures": c["header_failure_count"], "candidate_bound_images": counts["validation"] + counts["test"]},
          "annotation_human_accuracy": "NOT_INDEPENDENTLY_VERIFIED; high-quality claim is not a machine-QC result",
          "future_ordering": ["license gate", "identity and purpose seal", "mask-only CPU decoding preserving R+256G", "typed label/size/void QC", "no replacements after failures"],
          "predecode_seal_required": True, "predecode_seal_created": False, "reason": "license gate stopped before mask acquisition, so no permission to start mask QC"}
    decision = {"audit_workflow_status": "COMPLETE", "data_readiness_status": "BLOCKED_LICENSE", "additional_blockers": ["BLOCKED_DATA", "BLOCKED_DEFINITION", "BLOCKED_SPLIT"],
                "statistical_power_status": "NOT_EVALUATED", "s1_started": False, "can_request_S1_now": False, "next_authorization": "NONE",
                "adopted_dataset": None, "primary_candidate": "ImageNet-S919", "candidate_count": len(candidates), "qualified_count": 0,
                "gates": [{"gate": "predecessor_identity", "status": "PASS", "detail": "S0 four artifacts plus actually inherited source bindings recomputed"},
                          {"gate": "annotation_license", "status": "BLOCKED_LICENSE", "detail": "no explicit applicable annotation grant found in audited official sources"},
                          {"gate": "original_images", "status": "BLOCKED_DATA", "detail": "39842 candidate validation/test originals bound; 9190 training originals absent"},
                          {"gate": "annotation_payload", "status": "BLOCKED_DATA", "detail": "no payload downloaded pending license; test truth server-only"},
                          {"gate": "exact_class", "status": "PARTIAL", "detail": "919 canonical category WNIDs recognized; five merged destination categories quarantined"},
                          {"gate": "data_splits", "status": "BLOCKED_SPLIT", "detail": "no qualified masks in any pool; unknown training byte aliases; official test truth inaccessible"},
                          {"gate": "geometry_qc", "status": "NOT_RUN", "detail": "JPEG headers inspected; mask dimensions, void and crop-target presence untested"}],
                "next_user_decisions": ["provide applicable ImageNet-S annotation terms or written author permission and confirm ImageNet access/use authorization",
                                        "decide whether to supply required train originals or authorize an alternative officially documented labeled split route",
                                        "resolve inaccessible official test ground truth: obtain legitimate access or approve a new public-validation holdout design with isolated train/pilot and no old outcomes",
                                        "approve any new data plan in a fresh root; do not overwrite this terminal or enter S1 automatically"],
                "not_performed": ["annotation archive download", "RGB decode", "mask decode", "model load", "GPU", "probe training", "attribution", "Y4 changes", "S1", "commit", "push"],
                "artifacts": []}
    payloads = dict(zip(NAMES, (asset, license_review, annotation, manifest, reuse, splits, qc, decision)))
    docs = {}
    for name in NAMES:
        if name == "feasibility_decision":
            payloads[name]["artifacts"] = [{"name": k + ".json", "sha256": hashlib.sha256(serialize(docs[k])).hexdigest(), "canonical_digest": docs[k]["canonical_digest"]} for k in NAMES[:-1]]
        docs[name] = signed({"schema_version": 1, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S0-R1", "run_id": root.name,
                             "document_type": name, "predecessor": c["inherited"][-1], "payload": payloads[name]})
    return docs


def serialize(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"


def write_new(path, value):
    with path.open("xb") as f:
        f.write(serialize(value))
        f.flush()
        os.fsync(f.fileno())


def closed_contract(value):
    if isinstance(value, dict):
        return {"type": "object", "additionalProperties": False, "required": sorted(value), "properties": {k: closed_contract(v) for k, v in value.items()}}
    if isinstance(value, list):
        schemas = []
        for item in value:
            s = closed_contract(item)
            if s not in schemas:
                schemas.append(s)
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
        require(set(value) == set(schema["required"]) == set(schema["properties"]) and schema["additionalProperties"] is False, "unclosed object")
        for key, item in value.items():
            validate_closed(item, schema["properties"][key])
    elif kind == "array":
        for item in value:
            validate_closed(item, schema["items"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="evidence/nns_vit_patch_depth/s0_r1_20260914")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    root = (REPO / args.root).resolve()
    require(REPO / "evidence/nns_vit_patch_depth" in root.parents, "wrong root")
    require(root.is_dir(), "metadata acquisition root missing")
    if not args.verify:
        require(not any((root / (n + ".json")).exists() for n in NAMES), "terminal artifacts already exist")
    c = collect(root)
    docs = build_documents(root, c)
    require(not {"torch", "numpy", "timm", "torchvision", "PIL", "cv2"}.intersection(sys.modules), "forbidden module loaded")
    if args.verify:
        schema = json.loads((root / "artifact_contracts.schema.json").read_text())
        for name in NAMES:
            old = read_signed(root / (name + ".json"))
            validate_closed(old, schema["$defs"][name])
            require(old == docs[name], "source-recomputed artifact mismatch: " + name)
        print("PASS: eight closed artifacts equal source-recomputed metadata, JPEG identities, exclusions, mappings, split counts and blocked decision; no writes")
    else:
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$defs": {n: closed_contract(docs[n]) for n in NAMES},
                  "oneOf": [{"$ref": "#/$defs/" + n} for n in NAMES]}
        write_new(root / "artifact_contracts.schema.json", schema)
        for name in NAMES:
            validate_closed(docs[name], schema["$defs"][name])
            write_new(root / (name + ".json"), docs[name])
        print("SEALED: S0-R1 audit COMPLETE; readiness BLOCKED_LICENSE with DATA/DEFINITION/SPLIT blockers")
    print(json.dumps({"candidate_count": len(c["candidate_records"]), "qualified_count": 0,
                      "historical_exclusions": len(c["old_reasons"]), "candidate_reasons": docs["reuse_and_exclusion_manifest"]["payload"]["candidate_reason_counts"],
                      "provisional_counts": docs["split_feasibility"]["payload"]["provisional_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
