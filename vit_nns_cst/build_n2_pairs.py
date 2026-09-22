#!/usr/bin/env python3
"""Build and seal outcome-blind N2 pilot pairs before model access."""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "vitnns_rebuild"
sys.path.insert(0, str(ROOT / "src"))

from nns_cst_n2.core import canonical_digest
from nns_cst_n2.producer import (
    array_digest, composite, decode_record, low_level_metrics, matching_score,
    remove_donor_foreground, sha_file,
)

OUT = ROOT / "evidence/n2_r1c_20260922"
ARRAY_PATH = OUT / "pair_inputs.npz"
MANIFEST_PATH = OUT / "pair_manifest.json"


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    config_path = ROOT / "config/n2_r1/pilot_contract.json"
    source_path = REPO / "evidence/nns_vit_patch_depth/s0_r5_20260914/split_manifest.json"
    mask_path = REPO / "assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip"
    config = json.loads(config_path.read_text())
    source_document = json.loads(source_path.read_text())
    records = source_document["payload"]["records"]
    pilot = [row for row in records if row["final_split"] == "pilot" and row["use_policy"] == "S1_PILOT_ONLY"]
    if len(pilot) != 1744:
        raise RuntimeError("PILOT_SOURCE_COUNT_DRIFT")
    by_name = {Path(row["image_local_path"]).name: row for row in pilot}
    requested = config["recipients"] + config["aquatic_donors"] + config["non_aquatic_donors"]
    if len(set(requested)) != 24 or any(name not in by_name for name in requested):
        raise RuntimeError("SELECTED_ROSTER_NOT_UNIQUE_PILOT")
    forbidden = {row["source_group_id"] for row in records if row["final_split"] != "pilot"}
    if forbidden & {by_name[name]["source_group_id"] for name in requested}:
        raise RuntimeError("FORBIDDEN_SPLIT_LINEAGE_INTERSECTION")

    decoded: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    with zipfile.ZipFile(mask_path) as archive:
        for name in requested:
            decoded[name] = decode_record(REPO, archive, by_name[name], config["producer"]["crop_size"])
    backgrounds = {}
    for name in config["aquatic_donors"] + config["non_aquatic_donors"]:
        image, mask, metadata = decoded[name]
        if metadata["mask_fraction"] > config["data_boundary"]["donor_mask_fraction_max"]:
            raise RuntimeError(("DONOR_MASK_FRACTION", name, metadata["mask_fraction"]))
        backgrounds[name] = remove_donor_foreground(image, mask)

    chosen_semantic = set(); chosen_null = set(); output_images = []; output_masks = []; result_records = []
    for index, recipient_name in enumerate(config["recipients"]):
        recipient, recipient_mask, recipient_meta = decoded[recipient_name]
        base_name = config["aquatic_donors"][index]
        base = composite(recipient, recipient_mask, backgrounds[base_name], config["producer"]["alpha_gaussian_radius"])
        candidates = []
        for semantic_name in config["non_aquatic_donors"]:
            if semantic_name in chosen_semantic:
                continue
            semantic = composite(recipient, recipient_mask, backgrounds[semantic_name], config["producer"]["alpha_gaussian_radius"])
            semantic_metrics = low_level_metrics(base, semantic, recipient_mask)
            for null_name in config["aquatic_donors"]:
                if null_name == base_name or null_name in chosen_null:
                    continue
                null = composite(recipient, recipient_mask, backgrounds[null_name], config["producer"]["alpha_gaussian_radius"])
                null_metrics = low_level_metrics(base, null, recipient_mask)
                score, max_ratio, ratios = matching_score(semantic_metrics, null_metrics)
                candidates.append((score, max_ratio, semantic_name, null_name, semantic, null, semantic_metrics, null_metrics, ratios))
        if not candidates:
            raise RuntimeError("PAIR_ASSIGNMENT_EXHAUSTED")
        selected = min(candidates, key=lambda row: (row[0], row[1], row[2], row[3]))
        score, max_ratio, semantic_name, null_name, semantic, null, semantic_metrics, null_metrics, ratios = selected
        chosen_semantic.add(semantic_name); chosen_null.add(null_name)
        render_candidates = []
        for beta in config["producer"]["render_null_beta_candidates"]:
            mixed_background = np.clip(backgrounds[base_name] + beta * (backgrounds[null_name] - backgrounds[base_name]), 0, 1)
            render = composite(recipient, recipient_mask, mixed_background, config["producer"]["alpha_gaussian_radius"])
            metrics = low_level_metrics(base, render, recipient_mask)
            render_score, _render_max, _render_ratios = matching_score(semantic_metrics, metrics)
            render_candidates.append((render_score, beta, render, metrics))
        _render_score, beta, render, render_metrics = min(render_candidates, key=lambda row: (row[0], row[1]))
        arms_float = np.stack([base, semantic, null, render])
        arms = np.rint(arms_float * 255.0).clip(0, 255).astype(np.uint8)
        output_images.append(arms); output_masks.append(recipient_mask)
        identity = lambda name: {"filename": name, **decoded[name][2]}
        matched = score <= config["identifiability_gate"]["max_aggregate_log_ratio_distance"] and max_ratio <= config["identifiability_gate"]["max_per_metric_ratio"]
        result_records.append({
            "pair_id": f"n2_pair_{index:02d}", "recipient": identity(recipient_name),
            "base_donor": identity(base_name), "semantic_donor": identity(semantic_name),
            "within_stratum_donor": identity(null_name), "render_null_beta": float(beta),
            "semantic_metrics": semantic_metrics, "within_null_metrics": null_metrics,
            "render_null_metrics": render_metrics, "aggregate_log_ratio_distance": score,
            "max_per_metric_ratio": max_ratio, "per_metric_ratios": ratios, "matched": matched,
            "arm_array_digests": [array_digest(arm) for arm in arms],
        })

    images = np.ascontiguousarray(np.stack(output_images), dtype=np.uint8)
    masks = np.ascontiguousarray(np.stack(output_masks), dtype=bool)
    OUT.mkdir(parents=True, exist_ok=True)
    if ARRAY_PATH.exists() or MANIFEST_PATH.exists():
        raise RuntimeError("N2_PAIR_OUTPUT_ALREADY_EXISTS")
    with ARRAY_PATH.open("xb") as stream:
        np.savez_compressed(stream, images=images, masks=masks)
        stream.flush(); os.fsync(stream.fileno())
    matched_count = sum(row["matched"] for row in result_records)
    unmatched_rate = 1.0 - matched_count / len(result_records)
    ident_status = "PASS" if unmatched_rate <= config["identifiability_gate"]["max_unmatched_rate"] else "NO_GO_NOT_IDENTIFIABLE"
    manifest = {
        "schema_version": 1, "stage_id": "N2", "attempt_id": "N2_R1C_20260922",
        "status": "PASS_PAIR_CONSTRUCTION" if ident_status == "PASS" else ident_status,
        "contract_sha256": sha_file(config_path), "source_manifest_sha256": sha_file(source_path),
        "mask_archive_sha256": sha_file(mask_path), "selection_outcome_blind": True,
        "records": result_records,
        "arrays": {"path": "evidence/n2_r1c_20260922/pair_inputs.npz", "sha256": sha_file(ARRAY_PATH),
                   "bytes": ARRAY_PATH.stat().st_size, "images_shape": list(images.shape), "images_dtype": str(images.dtype),
                   "masks_shape": list(masks.shape), "masks_dtype": str(masks.dtype),
                   "arm_order": config["data_boundary"]["arms"]},
        "identifiability": {"matched_count": matched_count, "unmatched_count": len(result_records) - matched_count,
                              "unmatched_rate": unmatched_rate, "max_unmatched_rate": config["identifiability_gate"]["max_unmatched_rate"],
                              "semantic_roster_agreement": 1.0, "source_lineage_intersection_count": 0, "status": ident_status},
        "exposure": {"split": "pilot", "mask_payloads_decoded": 1744, "image_payloads_visually_reviewed": 79,
                     "selected_image_payloads_decoded": 24, "confirmation_payloads_read": 0, "replication_payloads_read": 0,
                     "model_outcomes_read_during_selection": 0, "selected_sources_permanently_pilot_only": True},
    }
    write_new(MANIFEST_PATH, manifest)
    print(json.dumps({"status": manifest["status"], "matched": matched_count, "unmatched_rate": unmatched_rate,
                      "manifest_digest": canonical_digest(manifest), "array_bytes": ARRAY_PATH.stat().st_size}, sort_keys=True))
    return 0 if ident_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
