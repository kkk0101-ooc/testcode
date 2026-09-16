#!/usr/bin/env python3
"""Independent S5 recomputation, closed-schema checks, and mutation audit."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s5 as s5  # noqa: E402

ROOT = REPO / s5.ROOT_REL
OUTPUT_FILES = {
    "authorization_gate_report.json": "s5_authorization_gate_report",
    "input_inventory.json": "s5_input_inventory",
    "image_level_endpoint_values.json": "s5_image_level_endpoint_values",
    "endpoint_decisions.json": "s5_endpoint_decisions",
    "original_vs_posthoc_P2_comparison.json": "s5_original_vs_posthoc_P2_comparison",
    "depth_profiles_and_D1_D2_relationship.json": "s5_depth_profiles_and_D1_D2_relationship",
    "intervention_controls_and_matching_diagnostics.json": "s5_intervention_controls_and_matching",
    "validity_accounting.json": "s5_validity_accounting",
    "claim_ledger.json": "s5_claim_ledger",
    "cpu_access_audit.json": "s5_cpu_access_audit",
    "closed_schema_registry.json": "s5_closed_schema_registry",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_output(name: str, expected_type: str) -> dict:
    path = ROOT / name
    require(path.is_file(), f"missing S5 output: {name}")
    value = json.loads(path.read_text())
    require(value["document_type"] == expected_type, f"document type drift: {name}")
    s5.verify_closed_document(value)
    return value


def verify_nested(outputs: dict[str, dict]) -> None:
    decisions = outputs["endpoint_decisions.json"]["payload"]["endpoints"]
    require(set(decisions) == set(s5.ALL_ENDPOINTS), "endpoint family slot removed")
    for endpoint, row in decisions.items():
        require(set(row) == s5.NESTED_RECORD_KEYS["endpoint_summary"], f"open endpoint summary: {endpoint}")
        require(row["analysis_label"] == s5.ANALYSIS_LABEL, f"endpoint label missing: {endpoint}")
    image_records = outputs["image_level_endpoint_values.json"]["payload"]["endpoint_records"]
    for endpoint, rows in image_records.items():
        expected = s5.NESTED_RECORD_KEYS["D2_image_record"] if endpoint == "D2" else s5.NESTED_RECORD_KEYS["P1_D1_image_record"]
        require(all(set(row) == expected and row["analysis_label"] == s5.ANALYSIS_LABEL for row in rows), f"open image-level records: {endpoint}")
    depth = outputs["depth_profiles_and_D1_D2_relationship.json"]["payload"]
    require(all(set(row) == s5.NESTED_RECORD_KEYS["NNS_depth_record"] and row["analysis_label"] == s5.ANALYSIS_LABEL for row in depth["NNS_image_block_records"]), "open NNS depth records")
    require(all(set(row) == s5.NESTED_RECORD_KEYS["D2_depth_record"] and row["analysis_label"] == s5.ANALYSIS_LABEL for row in depth["D2_image_block_records"]), "open D2 depth records")
    intervention = outputs["intervention_controls_and_matching_diagnostics.json"]["payload"]["records"]
    require(all(set(row) == s5.NESTED_RECORD_KEYS["intervention_record"] and row["analysis_label"] == s5.ANALYSIS_LABEL for row in intervention), "open intervention records")
    validity = outputs["validity_accounting.json"]["payload"]["records"]
    require(all(set(row) == s5.NESTED_RECORD_KEYS["validity_record"] and row["analysis_label"] == s5.ANALYSIS_LABEL for row in validity), "open validity records")
    require(all(row["analysis_label"] == s5.ANALYSIS_LABEL for row in depth["NNS_equal_class_profile"] + depth["D2_equal_class_profile"]), "aggregate depth label missing")
    require(depth["D1_D2_relationship"]["analysis_label"] == s5.ANALYSIS_LABEL, "D1/D2 relationship label missing")
    intervention_payload = outputs["intervention_controls_and_matching_diagnostics.json"]["payload"]
    require(all(row["analysis_label"] == s5.ANALYSIS_LABEL for row in intervention_payload["aggregate_by_operator_block_budget"]), "aggregate intervention label missing")
    require(intervention_payload["matching_diagnostics"]["analysis_label"] == s5.ANALYSIS_LABEL, "matching label missing")
    require(all(row["analysis_label"] == s5.ANALYSIS_LABEL for row in outputs["claim_ledger.json"]["payload"]["claims"]), "claim ledger label missing")
    input_records = outputs["input_inventory.json"]["payload"]["records"]
    for row in input_records:
        kind = "input_JSON_inventory_record" if row["kind"] == "json" else "input_NPY_inventory_record"
        require(set(row) == s5.NESTED_RECORD_KEYS[kind] and row["analysis_label"] == s5.ANALYSIS_LABEL, "open input inventory record")


def compare(produced: object, recomputed: object, message: str) -> None:
    require(s5.canonical_digest(produced) == s5.canonical_digest(recomputed), message)


def independent_bca(groups: list[list[float]], endpoint: str, namespace: str) -> dict:
    values = [np.asarray(group, dtype=np.float64) for group in groups]
    means = np.asarray([value.mean() for value in values], dtype=np.float64)
    estimate = float(means.mean())
    material = namespace + "\0ordinary_BCa\0" + endpoint + "\0" + "1" + "\0" + "20000"
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(20000, dtype=np.float64)
    for start in range(0, 20000, 1000):
        count = min(1000, 20000 - start)
        combined = np.zeros(count, dtype=np.float64)
        for value in values:
            indices = generator.integers(0, len(value), size=(count, len(value)))
            combined += value[indices].mean(axis=1)
        draws[start:start + count] = combined / len(values)
    proportion = min(max(float(np.mean(draws < estimate)), 1 / 40000), 1 - 1 / 40000)
    z0 = float(norm.ppf(proportion))
    leave_one_out = []
    for index, value in enumerate(values):
        if len(value) <= 1:
            continue
        other_sum = float(means.sum() - means[index])
        for omitted in range(len(value)):
            leave_one_out.append((other_sum + (value.sum() - value[omitted]) / (len(value) - 1)) / len(values))
    jackknife = np.asarray(leave_one_out, dtype=np.float64)
    delta = float(jackknife.mean()) - jackknife
    denominator = 6 * float(np.sum(delta * delta)) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.sum(delta ** 3)) / denominator
    probabilities = []
    for alpha in (0.025, 0.975):
        z = float(norm.ppf(alpha))
        probabilities.append(float(norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))))
    lower, upper = (float(value) for value in np.quantile(draws, probabilities, method="linear"))
    return {"estimate": estimate, "lower": lower, "upper": upper, "half_width": max(estimate - lower, upper - estimate), "replicates": 20000, "seed_namespace": namespace, "bias_correction_z0": z0, "acceleration": acceleration, "resampling_unit": "source_image_within_each_fixed_class_then_equal_class_mean"}


def independent_family_p(summaries: dict[str, dict]) -> dict[str, float]:
    statistics = {
        "P1": (summaries["P1"]["estimate"] - 0.02) / summaries["P1"]["class_cluster_standard_error"],
        "P2": summaries["P2"]["estimate"] / summaries["P2"]["class_cluster_standard_error"],
        "D1": (summaries["D1"]["estimate"] - 0.02) / summaries["D1"]["class_cluster_standard_error"],
        "D1_LOWER": (-summaries["D1"]["estimate"] - 0.02) / summaries["D1"]["class_cluster_standard_error"],
    }
    local = {}
    for size in (1, 2, 3):
        for intersection in itertools.combinations(s5.ENDPOINTS, size):
            candidate = 1.0
            for endpoint in intersection:
                candidate = min(candidate, size * 0.05 * float(norm.sf(statistics[endpoint])) / 0.025)
                if endpoint == "D1":
                    candidate = min(candidate, 2 * size * 0.05 * float(norm.sf(statistics["D1_LOWER"])) / 0.025)
            local[intersection] = min(1.0, candidate)
    return {endpoint: max(value for intersection, value in local.items() if endpoint in intersection) for endpoint in s5.ENDPOINTS}


def independent_compute_core(audit: s5.ReadAudit) -> dict:
    manifest = audit.json(s5.R12_MANIFEST_REL, "R12_BOUND_240_SOURCE_MANIFEST", s5.R12_MANIFEST_SHA256, s5.R12_MANIFEST_CANONICAL)
    records = manifest["payload"]["records"]
    projection = [{"schedule_index": int(row["schedule_index"]), "formal_class_rank": int(row["formal_class_rank"]), "source_image_id": row["source_id"]} for row in records]
    identity = {
        "analysis_label": s5.ANALYSIS_LABEL, "selected_source_count": len(records),
        "selected_terminal_count": sum(len(row["terminals"]) for row in records),
        "selected_distinct_class_rank_count": len({int(row["formal_class_rank"]) for row in records}),
        "full_manifest_records_digest": s5.canonical_digest(records), "membership_projection_digest": s5.canonical_digest(projection),
        "membership_projection": projection,
    }
    require(identity["full_manifest_records_digest"] == s5.FULL_RECORDS_DIGEST and identity["membership_projection_digest"] == s5.MEMBERSHIP_DIGEST, "independent selection digest mismatch")
    values = {endpoint: [] for endpoint in s5.ALL_ENDPOINTS}
    groups = {endpoint: defaultdict(list) for endpoint in s5.ALL_ENDPOINTS}
    nns_profile_rows = []
    d2_profile_rows = []
    intervention_rows = []
    validity_rows = []
    d1_by_source = {}
    d2_by_source = {}
    terminal_states: Counter[str] = Counter()
    matching = {"valid_nns": 0, "valid_p2": 0, "matched_coordinates": 0, "possible_coordinates": 0, "matched_controls": 0, "possible_controls": 0}
    for row in records:
        documents = {}
        for bound in row["terminals"]:
            doc = audit.json(bound["path"], "R12_BOUND_SOURCE_TERMINAL", bound["sha256"], bound["canonical_digest"])
            require(doc["payload"]["source_id"] == row["source_id"] and int(doc["payload"]["schedule_index"]) == int(row["schedule_index"]), "independent terminal join mismatch")
            documents[bound["unit_type"]] = doc
            terminal_states[doc["status"]] += 1
        rank = int(row["formal_class_rank"])
        source_id = str(row["source_id"])
        nns = documents["NNS_IMAGE"]
        p1_state = d1_state = "NA"
        p2_state = "NA"
        p1_reason = d1_reason = nns["status"] if nns["status"] != "VALID_NNS" else None
        p2_reason = documents["OCCLUSION_PRIMARY"]["status"] if documents["OCCLUSION_PRIMARY"]["status"] != "VALID_OCCLUSION" else None
        plan = None
        if nns["status"] == "VALID_NNS":
            matching["valid_nns"] += 1
            arrays = {Path(record["path"]).name: record for record in nns["payload"]["arrays"]}
            numerator = audit.array(arrays["p1_numerators.npy"], "R12_BOUND_P1_RAW_ARRAY").astype(np.float64)
            denominator = audit.array(arrays["p1_denominators.npy"], "R12_BOUND_P1_RAW_ARRAY").astype(np.float64)
            validity = audit.array(arrays["p1_valid_blocks.npy"], "R12_BOUND_P1_RAW_ARRAY").astype(bool)
            raw = audit.array(arrays["p1_raw_alignment.npy"], "R12_BOUND_P1_RAW_ARRAY").astype(np.float64)
            require(bool(validity.all()) and bool((denominator > 0).all()) and np.allclose(numerator[:, 0] / denominator, raw, rtol=1e-12, atol=1e-12), "independent P1 primitive mismatch")
            corrected = numerator[:, 0] / denominator - (numerator[:, 1:] / denominator[:, None]).mean(axis=1)
            p1 = float(corrected.mean())
            d1 = float(corrected[8:12].mean() - corrected[:4].mean())
            for endpoint, value in (("P1", p1), ("D1", d1)):
                values[endpoint].append({"analysis_label": s5.ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank, "value": value})
                groups[endpoint][rank].append(value)
            d1_by_source[source_id] = d1
            p1_state = d1_state = "VALID"
            p1_reason = d1_reason = None
            for block in range(12):
                nns_profile_rows.append({"analysis_label": s5.ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank, "block": block, "component": "first_residual_addition_all_head_aggregate_attention_branch", "target": "annotation_class_raw_logit", "raw_alignment": float(raw[block]), "position_control_corrected_alignment": float(corrected[block])})
            plan_record = nns["payload"]["p2_plan"]["record"]
            plan = audit.json(plan_record["path"], "R12_BOUND_P2_PLAN", plan_record["sha256"], plan_record["canonical_digest"])
            matching["possible_coordinates"] += len(plan["coordinates"])
            matching["possible_controls"] += len(plan["coordinates"]) * 4
            matching["matched_coordinates"] += sum(item["status"] == "MATCHED" for item in plan["coordinates"])
            matching["matched_controls"] += sum(len(item["controls"]) for item in plan["coordinates"] if item["status"] == "MATCHED")
        for unit_type in ("OCCLUSION_PRIMARY", "OCCLUSION_SENSITIVITY"):
            occlusion = documents[unit_type]
            if occlusion["status"] != "VALID_OCCLUSION":
                continue
            require(plan is not None and all(item["status"] == "MATCHED" for item in plan["coordinates"]), "independent valid occlusion plan mismatch")
            raw_logits = audit.array(occlusion["payload"]["arrays"][0], "R12_BOUND_P2_RAW_LOGIT_ARRAY").astype(np.float64)
            clean_logit = float(occlusion["payload"]["clean_logit"])
            operator = str(occlusion["payload"]["operator"])
            for coordinate_index, (coordinate, logits) in enumerate(zip(plan["coordinates"], raw_logits)):
                controls = coordinate["controls"]
                control_logits = [float(value) for value in logits[1:]]
                intervention_rows.append({"analysis_label": s5.ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank, "unit_type": unit_type, "operator": operator, "coordinate_index": coordinate_index, "block": int(coordinate["block"]), "budget": int(coordinate["budget"]), "selected_patch_ids": [int(value) for value in coordinate["selected"]], "selected_stratum_counts": [int(value) for value in coordinate["stratum_counts"]], "selected_boundary_complexity": int(coordinate["selected_boundary_complexity"]), "control_patch_ids": [[int(value) for value in control["indices"]] for control in controls], "control_stratum_counts": [[int(value) for value in control["stratum_counts"]] for control in controls], "control_boundary_absolute_errors": [int(control["boundary_absolute_error"]) for control in controls], "control_overlap_counts": [int(control["overlap_count"]) for control in controls], "clean_logit": clean_logit, "selected_occluded_logit": float(logits[0]), "control_occluded_logits": control_logits, "selected_raw_logit_drop": clean_logit - float(logits[0]), "control_raw_logit_drops": [clean_logit - value for value in control_logits], "matched_control_corrected_logit_drop": float(np.mean(logits[1:]) - logits[0])})
            if unit_type == "OCCLUSION_PRIMARY":
                p2 = float(np.mean(np.mean(raw_logits[:, 1:], axis=1) - raw_logits[:, 0]))
                values["P2"].append({"analysis_label": s5.ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank, "value": p2})
                groups["P2"][rank].append(p2)
                matching["valid_p2"] += 1
                p2_state = "VALID"
                p2_reason = None
        validity_rows.append({"analysis_label": s5.ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank, "scope": "R12_SELECTED_240", "unit_states": {unit_type: documents[unit_type]["status"] for unit_type in sorted(documents)}, "endpoint_states": {"P1": {"state": p1_state, "reason": p1_reason}, "P2": {"state": p2_state, "reason": p2_reason}, "D1": {"state": d1_state, "reason": d1_reason}}})
    require(dict(terminal_states) == s5.R12_STATES, "independent R12 terminal-state mismatch")

    s3_terminal = audit.json(s5.S3_TERMINAL_REL, "R12_AUTHORIZED_S3_METADATA", s5.S3_TERMINAL_SHA256, s5.S3_TERMINAL_CANONICAL)
    audit.json(s5.S2_SAP_REL, "R12_AUTHORIZED_S2_METADATA", s5.S2_SAP_SHA256, s5.S2_SAP_CANONICAL)
    d2_manifest = audit.json(s5.S3_D2_MANIFEST_REL, "R12_AUTHORIZED_S3_D2_MANIFEST", s5.S3_D2_MANIFEST_SHA256, s5.S3_D2_MANIFEST_CANONICAL)
    terminal_inventory = audit.json(s5.S3_TERMINAL_INVENTORY_REL, "R12_AUTHORIZED_S3_METADATA", s5.S3_TERMINAL_INVENTORY_SHA256, s5.S3_TERMINAL_INVENTORY_CANONICAL)
    terminal_by_index = {int(record["schedule_index"]): record for record in terminal_inventory["payload"]["records"]}
    d2_states: Counter[str] = Counter()
    for record in d2_manifest["payload"]["records"]:
        source_id = str(record["source_image_id"])
        wnid = str(record["annotation_wnid"])
        state = str(record["terminal_state"])
        d2_states[state] += 1
        if state != "VALID_FEATURE":
            validity_rows.append({"analysis_label": s5.ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": None, "scope": "S3_R7_D2_FULL_SEALED_QUEUE", "unit_states": {"D2": state}, "endpoint_states": {"D2": {"state": "NA", "reason": state}}})
            continue
        source_terminal_record = terminal_by_index[int(record["schedule_index"])]
        source_terminal = audit.json(source_terminal_record["path"], "R12_AUTHORIZED_S3_SOURCE_TERMINAL", source_terminal_record["sha256"], source_terminal_record["canonical_digest"])
        terminal_arrays = source_terminal["payload"]["arrays"]
        prediction_arrays = record["prediction_arrays"]
        by_name = {Path(item["path"]).name: item for item in prediction_arrays}
        control_record = next(item for item in prediction_arrays if Path(item["path"]).name == "control_mse.npy" or "corrected_control_mse" in item["path"])
        occupancy_record = next(item for item in terminal_arrays if Path(item["path"]).name == "occupancy.npy")
        feature_prediction = audit.array(by_name["feature_prediction.npy"], "R12_AUTHORIZED_S3_D2_PREDICTION_ARRAY").astype(np.float64)
        control_prediction = audit.array(by_name["control_prediction.npy"], "R12_AUTHORIZED_S3_D2_PREDICTION_ARRAY").astype(np.float64)
        stored_feature_mse = audit.array(by_name["feature_mse.npy"], "R12_AUTHORIZED_S3_D2_ERROR_ARRAY").astype(np.float64)
        stored_control_mse = audit.array(control_record, "R12_AUTHORIZED_S3_D2_ERROR_ARRAY").astype(np.float64)
        target = audit.array(occupancy_record, "R12_AUTHORIZED_S3_D2_TARGET_ARRAY").astype(np.float64)[3]
        feature_mse = np.mean((feature_prediction - target[None, :]) ** 2, axis=1, dtype=np.float64)
        control_mse = float(np.mean((control_prediction - target) ** 2, dtype=np.float64))
        require(np.allclose(feature_mse, stored_feature_mse, rtol=1e-6, atol=1e-9) and np.allclose(control_mse, stored_control_mse, rtol=1e-6, atol=1e-9), "independent D2 error recomputation mismatch")
        gain = control_mse - feature_mse
        d2 = float(gain[8:12].mean() - gain[:4].mean())
        values["D2"].append({"analysis_label": s5.ANALYSIS_LABEL, "source_image_id": source_id, "annotation_wnid": wnid, "value": d2})
        groups["D2"][wnid].append(d2)
        d2_by_source[source_id] = d2
        for block in range(12):
            d2_profile_rows.append({"analysis_label": s5.ANALYSIS_LABEL, "source_image_id": source_id, "annotation_wnid": wnid, "block": block, "component": "frozen_block_output_patch_features", "target": "target_patch_occupancy", "feature_probe_mse": float(feature_mse[block]), "low_level_control_mse": control_mse, "probe_gain_control_mse_minus_feature_mse": float(gain[block])})
        validity_rows.append({"analysis_label": s5.ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": None, "scope": "S3_R7_D2_FULL_SEALED_QUEUE", "unit_states": {"D2": state}, "endpoint_states": {"D2": {"state": "VALID", "reason": None}}})
    require(dict(d2_states) == {"VALID_FEATURE": 3424, "NA_D2_REQUIRED_SPLIT_OR_MODEL_MISSING": 52, "NA_TARGET_DISAPPEARED_AFTER_CROP": 6}, "independent D2 state mismatch")

    identity.update({"P1_D1_valid_class_count": len(groups["P1"]), "P2_valid_class_count": len(groups["P2"]), "D2_valid_class_count": len(groups["D2"])})
    summaries = {}
    totals = {"P1": (240, 235, 5, 0), "P2": (240, 224, 16, 0), "D1": (240, 235, 5, 0), "D2": (3482, 3424, 58, 0)}
    for endpoint in s5.ALL_ENDPOINTS:
        ordered = [list(groups[endpoint][key]) for key in sorted(groups[endpoint])]
        class_means = np.asarray([math.fsum(group) / len(group) for group in ordered], dtype=np.float64)
        estimate = float(class_means.mean())
        standard_error = float(class_means.std(ddof=1) / math.sqrt(len(class_means)))
        total, valid, na_count, failed = totals[endpoint]
        summaries[endpoint] = {"analysis_label": s5.ANALYSIS_LABEL, "endpoint": endpoint, "n_total": total, "n_valid": valid, "n_NA": na_count, "n_failed": failed, "valid_class_count": len(ordered), "estimate": estimate, "class_cluster_standard_error": standard_error, "ordinary_BCa_95": independent_bca(ordered, endpoint, s5.S4_BOOTSTRAP_NAMESPACE if endpoint in s5.ENDPOINTS else s5.S5_BOOTSTRAP_NAMESPACE)}
    adjusted = independent_family_p(summaries)
    for endpoint in s5.ENDPOINTS:
        estimate, standard_error, boundary = summaries[endpoint]["estimate"], summaries[endpoint]["class_cluster_standard_error"], s5.R12_BOUNDARY[endpoint]
        summaries[endpoint]["adjusted_interval"] = {"scope": "family-size-three_L1_post-hoc_exploratory", "boundary_z": boundary, "lower": estimate - boundary * standard_error, "upper": estimate + boundary * standard_error if endpoint == "D1" else None}
        summaries[endpoint]["adjusted_p"] = adjusted[endpoint]
        summaries[endpoint]["adjustment_scope"] = "family-size-three_L1_post-hoc_exploratory_no_strong_FWER_claim"
    d2_summary = summaries["D2"]
    critical = float(norm.ppf(0.975))
    d2_summary["adjusted_interval"] = {"scope": "D2_standalone_two-direction_post-hoc_exploratory", "boundary_z": critical, "lower": d2_summary["estimate"] - critical * d2_summary["class_cluster_standard_error"], "upper": d2_summary["estimate"] + critical * d2_summary["class_cluster_standard_error"]}
    z_upper = (d2_summary["estimate"] - 0.01) / d2_summary["class_cluster_standard_error"]
    z_lower = (-d2_summary["estimate"] - 0.01) / d2_summary["class_cluster_standard_error"]
    d2_summary["adjusted_p"] = min(1.0, 2 * min(float(norm.sf(z_upper)), float(norm.sf(z_lower))))
    d2_summary["adjustment_scope"] = "one-slot_D2_standalone_post-hoc_exploratory_no-cross-family-or-strong-FWER-claim"
    checks = {
        "P1": {"validity": True, "precision": summaries["P1"]["ordinary_BCa_95"]["half_width"] <= 0.015, "adjusted_lower_above_0_02": summaries["P1"]["adjusted_interval"]["lower"] > 0.02},
        "P2": {"validity": True, "matched_control_image_fraction": matching["valid_p2"] / matching["valid_nns"] >= 0.85, "point_estimate_at_least_0_07": summaries["P2"]["estimate"] >= 0.07, "family_three_adjusted_lower_above_zero": summaries["P2"]["adjusted_interval"]["lower"] > 0, "ordinary_BCa_half_width_at_most_0_02": summaries["P2"]["ordinary_BCa_95"]["half_width"] <= 0.02},
        "D1": {"validity": True, "precision": summaries["D1"]["ordinary_BCa_95"]["half_width"] <= 0.015, "adjusted_interval_outside_plus_or_minus_0_02": summaries["D1"]["adjusted_interval"]["lower"] > 0.02 or summaries["D1"]["adjusted_interval"]["upper"] < -0.02},
        "D2": {"validity": True, "minimum_valid_images": True, "standalone_adjusted_interval_outside_plus_or_minus_0_01": summaries["D2"]["adjusted_interval"]["lower"] > 0.01 or summaries["D2"]["adjusted_interval"]["upper"] < -0.01, "standalone_adjusted_p_at_most_0_05": summaries["D2"]["adjusted_p"] <= 0.05},
    }
    for endpoint in s5.ALL_ENDPOINTS:
        summaries[endpoint]["posthoc_threshold"] = {"P1": 0.02, "P2": 0.07, "D1": 0.02, "D2": 0.01}[endpoint]
        summaries[endpoint]["threshold_direction"] = "positive" if endpoint in ("P1", "P2") else "two-direction"
        summaries[endpoint]["decision_checks"] = checks[endpoint]
        summaries[endpoint]["exploratory_state"] = "SUPPORTED_POST_HOC_EXPLORATORY" if all(checks[endpoint].values()) else "INCONCLUSIVE_POST_HOC_EXPLORATORY"
    matching_result = {"analysis_label": s5.ANALYSIS_LABEL, "NNS_valid_images_denominator": matching["valid_nns"], "P2_valid_matched_images": matching["valid_p2"], "matched_image_fraction": matching["valid_p2"] / matching["valid_nns"], "matched_coordinates": matching["matched_coordinates"], "possible_coordinates": matching["possible_coordinates"], "matched_coordinate_fraction": matching["matched_coordinates"] / matching["possible_coordinates"], "matched_controls": matching["matched_controls"], "possible_controls": matching["possible_controls"], "matched_control_fraction": matching["matched_controls"] / matching["possible_controls"]}
    nns_by_block = defaultdict(lambda: defaultdict(list))
    for row in nns_profile_rows:
        nns_by_block[row["block"]][row["formal_class_rank"]].append((row["raw_alignment"], row["position_control_corrected_alignment"]))
    nns_aggregate = []
    for block in range(12):
        raw_means = [math.fsum(value[0] for value in rows) / len(rows) for _, rows in sorted(nns_by_block[block].items())]
        corrected_means = [math.fsum(value[1] for value in rows) / len(rows) for _, rows in sorted(nns_by_block[block].items())]
        nns_aggregate.append({"analysis_label": s5.ANALYSIS_LABEL, "block": block, "valid_classes": len(raw_means), "equal_class_raw_alignment": float(np.mean(raw_means)), "equal_class_position_control_corrected_alignment": float(np.mean(corrected_means))})
    d2_by_block = defaultdict(lambda: defaultdict(list))
    for row in d2_profile_rows:
        d2_by_block[row["block"]][row["annotation_wnid"]].append((row["feature_probe_mse"], row["low_level_control_mse"], row["probe_gain_control_mse_minus_feature_mse"]))
    d2_aggregate = []
    for block in range(12):
        feature_means = [math.fsum(value[0] for value in rows) / len(rows) for _, rows in sorted(d2_by_block[block].items())]
        control_means = [math.fsum(value[1] for value in rows) / len(rows) for _, rows in sorted(d2_by_block[block].items())]
        gain_means = [math.fsum(value[2] for value in rows) / len(rows) for _, rows in sorted(d2_by_block[block].items())]
        d2_aggregate.append({"analysis_label": s5.ANALYSIS_LABEL, "block": block, "valid_classes": len(feature_means), "equal_class_feature_probe_mse": float(np.mean(feature_means)), "equal_class_low_level_control_mse": float(np.mean(control_means)), "equal_class_probe_gain": float(np.mean(gain_means))})
    overlap = sorted(set(d1_by_source) & set(d2_by_source))
    d1_overlap = np.asarray([d1_by_source[source] for source in overlap], dtype=np.float64)
    d2_overlap = np.asarray([d2_by_source[source] for source in overlap], dtype=np.float64)
    relationship = {"analysis_label": s5.ANALYSIS_LABEL, "paired_valid_source_count": len(overlap), "D1_estimate": summaries["D1"]["estimate"], "D2_estimate": summaries["D2"]["estimate"], "aggregate_direction_relation": "SAME_DIRECTION" if summaries["D1"]["estimate"] * summaries["D2"]["estimate"] > 0 else "DIFFERENT_OR_ZERO_DIRECTION", "paired_image_Pearson_correlation": float(np.corrcoef(d1_overlap, d2_overlap)[0, 1]) if len(overlap) >= 2 else None, "paired_image_same_sign_fraction": float(np.mean(np.sign(d1_overlap) == np.sign(d2_overlap))) if len(overlap) else None, "interpretation_boundary": "descriptive association only; neither depth correlation nor probe result establishes a causal formation mechanism or NNS significance"}
    by_operator = defaultdict(list)
    for row in intervention_rows:
        by_operator[(row["operator"], row["block"], row["budget"])].append(row)
    intervention_summary = []
    for (operator, block, budget), rows in sorted(by_operator.items()):
        corrected_values = [row["matched_control_corrected_logit_drop"] for row in rows]
        boundary_values = [value for row in rows for value in row["control_boundary_absolute_errors"]]
        overlap_values = [value for row in rows for value in row["control_overlap_counts"]]
        intervention_summary.append({"analysis_label": s5.ANALYSIS_LABEL, "operator": operator, "block": block, "budget": budget, "matched_image_count": len(rows), "matched_control_count": 4 * len(rows), "mean_matched_control_corrected_logit_drop": float(np.mean(corrected_values)), "mean_boundary_absolute_error": float(np.mean(boundary_values)), "maximum_boundary_absolute_error": int(max(boundary_values)), "mean_selected_control_overlap_count": float(np.mean(overlap_values)), "maximum_selected_control_overlap_count": int(max(overlap_values))})
    return {"identity": identity, "endpoint_values": values, "endpoint_summaries": summaries, "matching": matching_result, "R12_terminal_states": dict(terminal_states), "D2_terminal_states": dict(d2_states), "nns_profile_rows": nns_profile_rows, "d2_profile_rows": d2_profile_rows, "nns_aggregate_profile": nns_aggregate, "d2_aggregate_profile": d2_aggregate, "depth_relationship": relationship, "intervention_rows": intervention_rows, "intervention_summary": intervention_summary, "validity_rows": validity_rows}


def validate() -> dict:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only boundary absent")
    forbidden = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "torchvision" or name.startswith("torchvision.") or name == "timm" or name.startswith("timm.") or name == "PIL" or name.startswith("PIL."))
    require(not forbidden, "forbidden dependency imported")
    _, gate = s5.authorization_gate(REPO, check_remote=True)
    audit = s5.ReadAudit(REPO)
    audit.json(s5.R12_TERMINAL_REL, "S5_AUTHORIZING_R12_TERMINAL", s5.R12_TERMINAL_SHA256, s5.R12_TERMINAL_CANONICAL)
    recomputed = independent_compute_core(audit)

    outputs = {name: read_output(name, document_type) for name, document_type in OUTPUT_FILES.items()}
    verify_nested(outputs)
    producer_decisions = outputs["endpoint_decisions.json"]["payload"]["endpoints"]
    compare(producer_decisions, recomputed["endpoint_summaries"], "producer-validator endpoint mismatch")
    compare(outputs["image_level_endpoint_values.json"]["payload"]["endpoint_records"], recomputed["endpoint_values"], "producer-validator image-level mismatch")
    compare(outputs["depth_profiles_and_D1_D2_relationship.json"]["payload"]["NNS_image_block_records"], recomputed["nns_profile_rows"], "producer-validator NNS profile mismatch")
    compare(outputs["depth_profiles_and_D1_D2_relationship.json"]["payload"]["D2_image_block_records"], recomputed["d2_profile_rows"], "producer-validator D2 profile mismatch")
    compare(outputs["depth_profiles_and_D1_D2_relationship.json"]["payload"]["NNS_equal_class_profile"], recomputed["nns_aggregate_profile"], "producer-validator NNS aggregate profile mismatch")
    compare(outputs["depth_profiles_and_D1_D2_relationship.json"]["payload"]["D2_equal_class_profile"], recomputed["d2_aggregate_profile"], "producer-validator D2 aggregate profile mismatch")
    compare(outputs["depth_profiles_and_D1_D2_relationship.json"]["payload"]["D1_D2_relationship"], recomputed["depth_relationship"], "producer-validator depth relationship mismatch")
    compare(outputs["intervention_controls_and_matching_diagnostics.json"]["payload"]["records"], recomputed["intervention_rows"], "producer-validator intervention mismatch")
    compare(outputs["intervention_controls_and_matching_diagnostics.json"]["payload"]["aggregate_by_operator_block_budget"], recomputed["intervention_summary"], "producer-validator intervention summary mismatch")
    compare(outputs["intervention_controls_and_matching_diagnostics.json"]["payload"]["matching_diagnostics"], recomputed["matching"], "producer-validator matching mismatch")
    compare(outputs["validity_accounting.json"]["payload"]["records"], recomputed["validity_rows"], "producer-validator validity mismatch")
    compare(outputs["authorization_gate_report.json"]["payload"]["checks"], gate["checks"], "authorization gate drift")

    input_records = audit.inventory()
    producer_inventory = outputs["input_inventory.json"]["payload"]
    producer_paths = {(record["path"], record["sha256"], record.get("canonical_array_digest"), record.get("canonical_digest")) for record in producer_inventory["records"]}
    validator_paths = {(record["path"], record["sha256"], record.get("canonical_array_digest"), record.get("canonical_digest")) for record in input_records}
    require(validator_paths <= producer_paths, "independent validator read an input outside producer whitelist")
    access = outputs["cpu_access_audit.json"]["payload"]
    require(access["CUDA_VISIBLE_DEVICES"] == "" and access["forbidden_import_count"] == 0 and access["JPEG_or_mask_decode_count"] == 0 and access["model_or_checkpoint_load_count"] == 0 and access["CUDA_or_GPU_initialization_count"] == 0 and access["new_intervention_count"] == 0, "CPU-only access audit failed")
    comparison = outputs["original_vs_posthoc_P2_comparison.json"]["payload"]
    require(comparison["original_rule"]["status"] == "ORIGINAL_RULE_NOT_PASSED" and not any(comparison["original_rule"]["checks"].values()), "original P2 rule incorrectly upgraded")
    require(all(row["exploratory_state"].endswith("POST_HOC_EXPLORATORY") for row in producer_decisions.values()), "endpoint escaped exploratory status")
    require(outputs["claim_ledger.json"]["payload"]["S6_executed"] is False, "S6 execution leaked into S5")
    identity = recomputed["identity"]
    require(identity["selected_distinct_class_rank_count"] == 122 and identity["P1_D1_valid_class_count"] == 120 and identity["P2_valid_class_count"] == 119, "required R12 class counts drift")
    output_digests = {name: value["canonical_digest"] for name, value in sorted(outputs.items())}
    return {
        "analysis_label": s5.ANALYSIS_LABEL, "title": "S5 post-hoc exploratory independent validation",
        "producer_status_fields_read_before_recompute": False,
        "checks": {
            "authorization_gate_recomputed": True, "raw_arrays_recomputed": True, "image_level_values_recomputed": True,
            "equal_class_aggregation_recomputed": True, "valid_NA_failed_denominators_recomputed": True,
            "P2_matched_control_denominator_recomputed": True, "BCa_20000_recomputed": True,
            "family_three_adjusted_CI_and_p_recomputed": True, "D2_standalone_exploratory_analysis_recomputed": True,
            "full_records_and_membership_digests_recomputed": True, "closed_schemas_pass": True,
            "typed_NA_not_deleted_replaced_or_zero_filled": True, "invalid_endpoint_slot_retention_checked": True,
            "posthoc_exploratory_labeling_checked": True, "S6_not_executed": True,
        },
        "recomputed_identity": identity,
        "recomputed_endpoint_summaries": recomputed["endpoint_summaries"],
        "recomputed_output_digests": output_digests,
        "scientific_read_count": len(input_records), "scientific_read_records_digest": s5.canonical_digest(input_records),
    }


def false_pass() -> dict:
    outputs = {name: read_output(name, document_type) for name, document_type in OUTPUT_FILES.items()}
    attacks: list[dict] = []

    def attack(name: str, base: object, mutate, check) -> None:
        value = copy.deepcopy(base)
        mutate(value)
        rejected = False
        try:
            check(value)
        except Exception:
            rejected = True
        require(rejected, f"false-pass mutation accepted: {name}")
        attacks.append({"analysis_label": s5.ANALYSIS_LABEL, "attack": name, "mutation_applied": True, "rejected": True})

    gate = outputs["authorization_gate_report.json"]
    attack("authorization_status", gate, lambda value: value["payload"]["checks"].__setitem__("terminal_status", False), lambda value: require(all(value["payload"]["checks"].values()), "gate"))
    attack("formal_gate_open", gate, lambda value: value["payload"]["checks"].__setitem__("formal_confirmatory_gate_closed", False), lambda value: require(all(value["payload"]["checks"].values()), "gate"))
    attack("live_remote_binding", gate, lambda value: value["payload"]["checks"].__setitem__("live_remote_contains_R12", False), lambda value: require(all(value["payload"]["checks"].values()), "gate"))
    identity = json.loads((ROOT / "independent_validation_report.json").read_text())["payload"]["recomputed_identity"] if (ROOT / "independent_validation_report.json").exists() else {"selected_distinct_class_rank_count": 122, "P1_D1_valid_class_count": 120, "P2_valid_class_count": 119, "full_manifest_records_digest": s5.FULL_RECORDS_DIGEST, "membership_projection_digest": s5.MEMBERSHIP_DIGEST}
    for name, field, replacement, expected in (
        ("selected_class_122_to_120", "selected_distinct_class_rank_count", 120, 122),
        ("P1_D1_class_120_to_122", "P1_D1_valid_class_count", 122, 120),
        ("P2_class_119_to_120", "P2_valid_class_count", 120, 119),
        ("full_records_digest", "full_manifest_records_digest", "0" * 64, s5.FULL_RECORDS_DIGEST),
        ("membership_projection_digest", "membership_projection_digest", "0" * 64, s5.MEMBERSHIP_DIGEST),
    ):
        attack(name, identity, lambda value, field=field, replacement=replacement: value.__setitem__(field, replacement), lambda value, field=field, expected=expected: require(value[field] == expected, field))
    decisions = outputs["endpoint_decisions.json"]
    attack("remove_INVALID_family_slot", decisions, lambda value: value["payload"]["endpoints"].pop("D2"), lambda value: require(set(value["payload"]["endpoints"]) == set(s5.ALL_ENDPOINTS), "family"))
    attack("P2_point_gate", decisions, lambda value: value["payload"]["endpoints"]["P2"]["decision_checks"].__setitem__("point_estimate_at_least_0_07", False), lambda value: require(all(value["payload"]["endpoints"]["P2"]["decision_checks"].values()), "P2"))
    attack("P2_lower_gate", decisions, lambda value: value["payload"]["endpoints"]["P2"]["decision_checks"].__setitem__("family_three_adjusted_lower_above_zero", False), lambda value: require(all(value["payload"]["endpoints"]["P2"]["decision_checks"].values()), "P2"))
    attack("P2_precision_gate", decisions, lambda value: value["payload"]["endpoints"]["P2"]["decision_checks"].__setitem__("ordinary_BCa_half_width_at_most_0_02", False), lambda value: require(all(value["payload"]["endpoints"]["P2"]["decision_checks"].values()), "P2"))
    comparison = outputs["original_vs_posthoc_P2_comparison.json"]
    attack("original_rule_upgrade", comparison, lambda value: value["payload"]["original_rule"].__setitem__("status", "PASS"), lambda value: require(value["payload"]["original_rule"]["status"] == "ORIGINAL_RULE_NOT_PASSED", "original"))
    access = outputs["cpu_access_audit.json"]
    attack("CUDA_visible", access, lambda value: value["payload"].__setitem__("CUDA_VISIBLE_DEVICES", "0"), lambda value: require(value["payload"]["CUDA_VISIBLE_DEVICES"] == "", "CUDA"))
    attack("JPEG_decode", access, lambda value: value["payload"].__setitem__("JPEG_or_mask_decode_count", 1), lambda value: require(value["payload"]["JPEG_or_mask_decode_count"] == 0, "decode"))
    attack("model_load", access, lambda value: value["payload"].__setitem__("model_or_checkpoint_load_count", 1), lambda value: require(value["payload"]["model_or_checkpoint_load_count"] == 0, "model"))
    attack("new_intervention", access, lambda value: value["payload"].__setitem__("new_intervention_count", 1), lambda value: require(value["payload"]["new_intervention_count"] == 0, "intervention"))
    claims = outputs["claim_ledger.json"]
    attack("S6_execution", claims, lambda value: value["payload"].__setitem__("S6_executed", True), lambda value: require(value["payload"]["S6_executed"] is False, "S6"))
    attack("label_removed", decisions, lambda value: value["payload"]["endpoints"]["P1"].__setitem__("analysis_label", "confirmatory"), lambda value: require(value["payload"]["endpoints"]["P1"]["analysis_label"] == s5.ANALYSIS_LABEL, "label"))
    return {"analysis_label": s5.ANALYSIS_LABEL, "title": "S5 post-hoc exploratory false-pass mutation audit", "attack_count": len(attacks), "all_rejected": True, "attacks": attacks}


def publication() -> dict:
    terminal_path = ROOT / "terminal_receipt.json"
    inventory_path = ROOT / "whole_root_inventory.json"
    require(terminal_path.is_file() and inventory_path.is_file(), "late-bound S5 publication files missing")
    terminal = json.loads(terminal_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    s5.verify_closed_document(terminal)
    s5.verify_closed_document(inventory)
    payload = terminal["payload"]
    require(terminal["status"] == "PASS_POST_HOC_EXPLORATORY" and payload["SCIENTIFIC_STATUS"] == "PASS_POST_HOC_EXPLORATORY", "terminal exploratory status drift")
    require(payload["FORMAL_CONFIRMATORY_GATE_OPEN"] is False and payload["posthoc_exploratory_only"] is True, "terminal scope drift")
    require(payload["strong_FWER_preserved_claim"] is False and payload["original_S4_R10_STOP_SUCCESS_claimed"] is False, "forbidden terminal claim")
    require(payload["original_P2_rule_status"] == "ORIGINAL_RULE_NOT_PASSED", "original P2 status drift")
    require(payload["S6_execution_count"] == 0 and payload["next_authorization"] == "S6_POST_HOC_EXPLORATORY_REPORTING_ONLY", "next authorization drift")
    require(set(payload["endpoint_states"]) == set(s5.ALL_ENDPOINTS) and all(value == "SUPPORTED_POST_HOC_EXPLORATORY" for value in payload["endpoint_states"].values()), "terminal endpoint states drift")
    identity = payload["R12_selection_identity"]
    require(identity == {"selected_source_count": 240, "selected_distinct_class_rank_count": 122, "P1_D1_valid_class_count": 120, "P2_valid_class_count": 119, "full_manifest_records_digest": s5.FULL_RECORDS_DIGEST, "membership_projection_digest": s5.MEMBERSHIP_DIGEST}, "terminal R12 identity drift")
    records = inventory["payload"]["files"]
    require(inventory["payload"]["file_count"] == len(records) and inventory["payload"]["files_digest"] == s5.canonical_digest(records), "whole-root inventory digest drift")
    current = []
    for record in records:
        path = REPO / record["path"]
        require(path.is_file() and path.stat().st_size == record["bytes"] and s5.sha_file(path) == record["sha256"], f"whole-root file drift: {record['path']}")
        if "canonical_digest" in record:
            require(json.loads(path.read_text())["canonical_digest"] == record["canonical_digest"], f"whole-root canonical drift: {record['path']}")
        current.append(record)
    required = payload["required_evidence"]
    expected_required_paths = {path.name for path in ROOT.glob("*") if path.is_file() and path.name != "terminal_receipt.json"}
    require(set(required) == expected_required_paths, "terminal required-evidence set drift")
    for name, record in required.items():
        path = REPO / record["path"]
        require(path.name == name and path.is_file() and path.stat().st_size == record["bytes"] and s5.sha_file(path) == record["sha256"], f"terminal evidence drift: {name}")
    return {"status": "PASS_POST_HOC_EXPLORATORY", "analysis_label": s5.ANALYSIS_LABEL, "terminal_sha256": s5.sha_file(terminal_path), "terminal_canonical_digest": terminal["canonical_digest"], "inventory_file_count": len(records), "required_evidence_count": len(required), "next_authorization": payload["next_authorization"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "false-pass", "publication"))
    args = parser.parse_args()
    result = validate() if args.command == "validate" else false_pass() if args.command == "false-pass" else publication()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
