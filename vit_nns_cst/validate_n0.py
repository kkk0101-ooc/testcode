#!/usr/bin/env python3
"""Independent, standard-library-only validator for the NNS-CST N0 package.

The validator reads source manifests and file metadata directly. It does not
decode images, deserialize checkpoints, import ML frameworks, or trust a
producer-side PASS field. Use --full-hash to stream-hash the four large binary
assets in addition to validating their sizes and all metadata-level facts.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REBUILD = REPO / "vitnns_rebuild"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_keys(obj: dict[str, Any], keys: Iterable[str], where: str) -> None:
    expected = set(keys)
    actual = set(obj)
    require(actual == expected, f"{where}: key mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: root must be object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = REPO / value
    require(path.is_file(), f"missing file: {value}")
    return path


def validate_closed_decision(decision: dict[str, Any]) -> None:
    require_keys(
        decision,
        [
            "schema_version", "document_type", "study_id", "stage_id",
            "decision_date_prc", "status", "authority", "repository_state",
            "deliverables", "gates", "aggregate_decision", "asset_facts",
            "license_boundary", "missing_assets_and_implementations",
            "execution_audit", "non_inheritance",
            "required_next_stage_fail_closed_checks",
        ],
        "decision",
    )
    require(decision["schema_version"] == 1, "unsupported schema version")
    require(decision["document_type"] == "nns_cst_n0_decision", "wrong document type")
    require(decision["study_id"] == "NNS_CST" and decision["stage_id"] == "N0", "wrong study/stage")

    require_keys(
        decision["authority"],
        ["plan_path", "plan_sha256", "goal_path", "goal_sha256", "authorized_stage", "n1_authorized", "next_action"],
        "authority",
    )
    require_keys(
        decision["repository_state"],
        ["repository_root", "branch", "head_commit", "preexisting_adjacent_changes_preserved", "commit_created", "push_performed"],
        "repository_state",
    )
    require_keys(decision["gates"], ["NOVELTY_CANDIDATE", "DATA_FEASIBLE", "IDENTIFIABLE"], "gates")
    for name, gate in decision["gates"].items():
        require_keys(gate, ["passed", "decision", "evidence_path", "basis", "falsifier"], f"gate {name}")
        require(gate["decision"] == name, f"gate {name}: decision label mismatch")
        require(isinstance(gate["passed"], bool), f"gate {name}: passed must be boolean")

    require_keys(decision["aggregate_decision"], ["rule", "all_required_gates_passed", "decision", "meaning"], "aggregate_decision")
    require_keys(
        decision["asset_facts"],
        ["candidate_pool", "archives", "mask_archive_png_counts", "local_train_semi", "models"],
        "asset_facts",
    )
    require_keys(
        decision["asset_facts"]["candidate_pool"],
        [
            "record_count", "unique_source_image_count", "unique_image_sha256_count",
            "unique_annotation_sha256_count", "class_count",
            "image_reference_present_count", "mask_reference_present_count",
            "historical_exclusion_count", "historical_candidate_collision_count", "split_counts",
        ],
        "candidate_pool",
    )
    require_keys(decision["asset_facts"]["candidate_pool"]["split_counts"], ["pilot", "probe_train", "probe_validation", "confirmatory_holdout"], "split_counts")
    require_keys(decision["asset_facts"]["mask_archive_png_counts"], ["total", "validation", "train_semi"], "mask_archive_png_counts")
    require_keys(decision["asset_facts"]["local_train_semi"], ["jpeg_count", "total_bytes", "extraction_receipt_present"], "local_train_semi")
    for index, item in enumerate(decision["deliverables"]):
        require_keys(item, ["path", "bytes", "sha256", "role"], f"deliverable[{index}]")
    for index, item in enumerate(decision["asset_facts"]["archives"]):
        require_keys(item, ["path", "bytes", "sha256", "metadata_role"], f"archive[{index}]")
    for index, item in enumerate(decision["asset_facts"]["models"]):
        require_keys(
            item,
            ["model_id", "timm_name", "depth", "heads", "embedding_dim", "checkpoint_path", "checkpoint_bytes", "checkpoint_sha256", "checkpoint_deserialized_in_n0"],
            f"model[{index}]",
        )
    require_keys(
        decision["license_boundary"],
        [
            "imagenet_s_upstream_explicit_annotation_license_found",
            "user_attested_local_research_authorization_recorded",
            "user_authorization_receipt_path", "commercial_use_granted_or_inferred",
            "redistribution_granted_or_inferred", "checkpoint_redistribution_right_verified",
            "allowed_scope",
        ],
        "license_boundary",
    )
    require_keys(
        decision["execution_audit"],
        [
            "new_data_downloads", "formal_image_decodes", "model_loads",
            "checkpoint_deserializations", "gpu_or_cuda_uses", "training_runs",
            "inference_runs", "nns_runs", "activation_patching_runs",
            "formal_outcome_reads", "archive_or_checkpoint_stream_hashing_allowed",
            "zip_central_directory_metadata_read_allowed",
        ],
        "execution_audit",
    )
    require_keys(
        decision["non_inheritance"],
        ["historical_scientific_outcomes_inherited", "reusable_metadata_and_code_inherited", "n0_scientific_success_claimed", "n0_scope_statement"],
        "non_inheritance",
    )


def validate_document_bindings(decision: dict[str, Any]) -> None:
    authority = decision["authority"]
    for path_key, hash_key in (("plan_path", "plan_sha256"), ("goal_path", "goal_sha256")):
        path = repo_path(authority[path_key])
        require(sha256_file(path) == authority[hash_key], f"authority hash mismatch: {path}")

    expected = {
        "vit_nns_cst/相关工作与新颖性矩阵.md",
        "vit_nns_cst/资产与许可清单.md",
        "vit_nns_cst/配对反事实可识别性审计.md",
        "vit_nns_cst/n0_decision.json",
    }
    require({item["path"] for item in decision["deliverables"]} == expected, "deliverable set mismatch")
    for item in decision["deliverables"]:
        path = repo_path(item["path"])
        if path.name == "n0_decision.json":
            require(item["bytes"] is None and item["sha256"] is None, "self digest must be null")
        else:
            require(path.stat().st_size == item["bytes"], f"deliverable byte mismatch: {path.name}")
            require(sha256_file(path) == item["sha256"], f"deliverable hash mismatch: {path.name}")


def validate_prompt_coverage() -> dict[str, bool]:
    novelty = (HERE / "相关工作与新颖性矩阵.md").read_text(encoding="utf-8")
    assets = (HERE / "资产与许可清单.md").read_text(encoding="utf-8")
    ident = (HERE / "配对反事实可识别性审计.md").read_text(encoding="utf-8")

    families = [
        "TCAV", "CBM", "Network Dissection", "Feature Visualization", "PatchSAE",
        "SAE-V", "transcoder", "crosscoder", "activation patching", "causal tracing",
        "mediation", "DAS", "ACDC", "EAP", "Vi-CD", "Prisma", "ViSAE",
        "IG", "Conductance", "GlobEnc", "DecompX", "AttnLRP", "ALTI",
        "LibraGrad", "NNS", "Endpoint", "complete-cut",
    ]
    missing_families = [value for value in families if value.lower() not in novelty.lower()]
    require(not missing_families, f"literature families missing: {missing_families}")
    require("已有工作已经做了什么" in novelty and "NNS-CST 仍需证明什么" in novelty, "novelty comparison columns missing")
    require("NO_GO_NOVELTY" in novelty and "首个" in novelty, "novelty fail-closed boundary missing")

    asset_terms = [
        "原图", "mask", "编辑图", "donor", "生成谱系", "source_group_id",
        "target model class", "历史 1,101", "ViT-B/16", "DeiT-S/16",
        "checkpoint", "许可", "complete-cut", "JVP/VJP", "intervention",
    ]
    missing_assets = [value for value in asset_terms if value.lower() not in assets.lower()]
    require(not missing_assets, f"asset audit topics missing: {missing_assets}")
    require("NOT_GENERATED" in assets and "再分发" in assets and "商业" in assets, "asset limitations missing")

    contrast_terms = [
        "同前景、换背景", "同背景、换前景", "一致/冲突背景", "null edit",
        "S1_SEMANTIC_SWAP", "N1_WITHIN_STRATUM_SWAP", "低级", "生成伪影",
        "source_group_id", "NO_GO_NOT_IDENTIFIABLE",
    ]
    missing_contrasts = [value for value in contrast_terms if value.lower() not in ident.lower()]
    require(not missing_contrasts, f"identifiability topics missing: {missing_contrasts}")
    return {
        "literature_families_and_novelty_boundary": True,
        "assets_models_licenses_lineage_and_code": True,
        "semantic_and_matched_null_identifiability": True,
        "four_required_deliverables": True,
    }


def validate_source_assets(decision: dict[str, Any], full_hash: bool) -> dict[str, Any]:
    split_path = REBUILD / "evidence/nns_vit_patch_depth/s0_r5_20260914/split_manifest.json"
    split = read_json(split_path)
    records = split["payload"]["records"]
    actual = {
        "record_count": len(records),
        "unique_source_image_count": len({r["source_image_id"] for r in records}),
        "unique_image_sha256_count": len({r["image_sha256"] for r in records}),
        "unique_annotation_sha256_count": len({r["annotation_sha256"] for r in records}),
        "class_count": len({r["target_model_class_index"] for r in records}),
        "split_counts": dict(collections.Counter(r["final_split"] for r in records)),
    }
    require(None not in {r["target_model_class_index"] for r in records}, "null target model class index")

    mask_zip = REBUILD / "assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip"
    with zipfile.ZipFile(mask_zip) as archive:
        names = {info.filename for info in archive.infolist()}
        pngs = [name for name in names if name.lower().endswith(".png")]
        mask_counts = {
            "total": len(pngs),
            "validation": sum("/validation-segmentation/" in name for name in pngs),
            "train_semi": sum("/train-semi-segmentation/" in name for name in pngs),
        }
    image_present = 0
    archive_records: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        if record["image_local_path"]:
            image_present += int((REBUILD / record["image_local_path"]).is_file())
        elif record["image_archive"] and record["image_member"]:
            archive_records[record["image_archive"]].append(record)
    for archive_rel, grouped_records in archive_records.items():
        # Tar member tables are metadata; getnames does not extract or decode JPEG payloads.
        with tarfile.open(REBUILD / archive_rel, mode="r:") as archive:
            member_names = set(archive.getnames())
        image_present += sum(record["image_member"] in member_names for record in grouped_records)
    mask_present = sum(r["annotation_archive_path"] in names for r in records)
    actual["image_reference_present_count"] = image_present
    actual["mask_reference_present_count"] = mask_present

    reuse = read_json(REBUILD / "evidence/nns_vit_patch_depth/s0_r1_20260914/reuse_and_exclusion_manifest.json")["payload"]
    actual["historical_exclusion_count"] = reuse["historical_exclusion_count"]
    actual["historical_candidate_collision_count"] = reuse["candidate_historical_collision_count"]
    expected_pool = decision["asset_facts"]["candidate_pool"]
    require(actual == expected_pool, f"candidate pool mismatch: actual={actual}, expected={expected_pool}")
    require(mask_counts == decision["asset_facts"]["mask_archive_png_counts"], "mask archive counts mismatch")

    independence = read_json(REBUILD / "evidence/nns_vit_patch_depth/s0_r5_20260914/independence_audit.json")["payload"]
    for check in independence["pairwise_split_checks"]:
        require(check["source_image_id_intersection_count"] == 0, f"source split overlap: {check}")
        require(check["image_sha256_intersection_count"] == 0, f"image split overlap: {check}")
    require(independence["historical_source_id_intersection_count"] == 0, "historical source overlap")
    require(independence["historical_byte_hash_intersection_count"] == 0, "historical byte overlap")

    train_dir = REBUILD / "assets/imagenet/imagenets919_train_semi"
    jpegs = list(train_dir.rglob("*.JPEG"))
    train_actual = {
        "jpeg_count": len(jpegs),
        "total_bytes": sum(path.stat().st_size for path in jpegs),
        "extraction_receipt_present": (train_dir / "extraction_receipt.json").is_file(),
    }
    require(train_actual == decision["asset_facts"]["local_train_semi"], "train-semi metadata mismatch")

    model_registry = read_json(REBUILD / "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json")
    registered = {item["model_id"]: item for item in model_registry["models"]}
    require(model_registry["ep0_checkpoint_deserialized"] is False, "EP0 registry says checkpoint was deserialized")

    large_files = list(decision["asset_facts"]["archives"])
    for model in decision["asset_facts"]["models"]:
        source = registered[model["model_id"]]
        for key in ("timm_name", "depth", "heads", "embedding_dim"):
            require(model[key] == source[key], f"model registry mismatch: {model['model_id']} {key}")
        large_files.append({
            "path": model["checkpoint_path"],
            "bytes": model["checkpoint_bytes"],
            "sha256": model["checkpoint_sha256"],
        })

    hashes_checked = 0
    for item in large_files:
        path = repo_path(item["path"] if item["path"].startswith("vitnns_rebuild/") else f"vitnns_rebuild/{item['path']}")
        require(path.stat().st_size == item["bytes"], f"large asset byte mismatch: {path}")
        if full_hash:
            require(sha256_file(path) == item["sha256"], f"large asset hash mismatch: {path}")
            hashes_checked += 1

    license_review = read_json(REBUILD / "evidence/nns_vit_patch_depth/s0_r1_20260914/license_and_source_review.json")["payload"]
    require("No explicit annotation-use terms found" in license_review["primary_candidate"]["annotation_license"], "upstream annotation license boundary changed")
    receipt = read_json(REBUILD / "evidence/nns_vit_patch_depth/s0_r2_20260914/authorization_receipt.json")["payload"]
    require(receipt["status"] == "USER_ATTESTED_AUTHORIZATION_RECORDED", "user authorization receipt missing")
    require(receipt["authorization_scope"]["commercial_use"] == "NOT_GRANTED_OR_INFERRED", "commercial scope widened")
    require(receipt["authorization_scope"]["redistribution"] == "NOT_GRANTED_OR_INFERRED", "redistribution scope widened")

    return {
        "candidate_records_recomputed": len(records),
        "image_references_checked": image_present,
        "mask_references_checked": mask_present,
        "zip_png_entries_checked": mask_counts["total"],
        "large_asset_hashes_checked": hashes_checked,
    }


def validate_decision_logic(decision: dict[str, Any]) -> None:
    gates = decision["gates"]
    all_pass = all(gate["passed"] for gate in gates.values())
    aggregate = decision["aggregate_decision"]
    require(all_pass == aggregate["all_required_gates_passed"], "aggregate gate boolean mismatch")
    require(aggregate["decision"] == ("GO_REQUEST_N1" if all_pass else "NO_GO_N0"), "aggregate decision mismatch")
    authority = decision["authority"]
    require(authority["authorized_stage"] == "N0_ONLY", "authorized stage widened")
    require(authority["n1_authorized"] is False, "N1 must remain unauthorized")
    require(authority["next_action"] == "WAIT_FOR_EXPLICIT_N1_AUTHORIZATION", "wrong next action")
    require(decision["status"] == "N0_COMPLETE_AWAITING_EXPLICIT_N1_AUTHORIZATION", "wrong stage status")
    for key, value in decision["execution_audit"].items():
        if key.endswith("_allowed"):
            require(value is True, f"allowed metadata operation not recorded: {key}")
        else:
            require(value == 0, f"prohibited operation recorded: {key}={value}")
    require(decision["non_inheritance"]["historical_scientific_outcomes_inherited"] is False, "historical outcomes inherited")
    require(decision["non_inheritance"]["n0_scientific_success_claimed"] is False, "N0 overclaims scientific success")

    branch = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    require(branch == decision["repository_state"]["branch"], "branch binding mismatch")
    require(head == decision["repository_state"]["head_commit"], "HEAD binding mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-hash", action="store_true", help="stream-hash data archives and model checkpoints")
    args = parser.parse_args()
    decision = read_json(HERE / "n0_decision.json")
    validate_closed_decision(decision)
    validate_document_bindings(decision)
    coverage = validate_prompt_coverage()
    asset_result = validate_source_assets(decision, args.full_hash)
    validate_decision_logic(decision)
    result = {
        "validator": "validate_n0.py",
        "status": "PASS",
        "full_hash": args.full_hash,
        "coverage": coverage,
        "asset_recomputation": asset_result,
        "aggregate_decision": decision["aggregate_decision"]["decision"],
        "n1_authorized": decision["authority"]["n1_authorized"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, KeyError, TypeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        print(json.dumps({"validator": "validate_n0.py", "status": "FAIL", "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)
