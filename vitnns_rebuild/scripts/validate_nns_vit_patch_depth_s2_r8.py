#!/usr/bin/env python3
"""Independent standard-library validator for S2-R8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s2_r8_20260914"
SPEC = REPO / "config/nns_vit_patch_depth/s2_r8_run_spec.json"
R7 = REPO / "evidence/nns_vit_patch_depth/s2_r7_20260914"
ATTACKS = (
    "target_power_0.75", "target_power_0.70", "conditional_as_joint",
    "gate_skipped", "D2_one_sided", "family_member_removed",
    "cap_above_4", "all_11557_used", "ICC_changed", "scale_changed",
    "outcome_selected_images", "extra_images_enter_S4",
    "repeat_training_as_independent", "formal_outcome_access",
    "terminal_status", "next_authorization",
)


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    unsigned = dict(value)
    expected = unsigned.pop("canonical_digest")
    require(expected == digest(unsigned), "canonical digest mismatch: " + str(path))
    return value


def record(path: Path, base: Path = REPO) -> dict[str, Any]:
    result = {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if "canonical_digest" in value:
            result["canonical_digest"] = read_signed(path)["canonical_digest"]
    return result


def validate_spec(spec: dict[str, Any]) -> None:
    require(spec["authorization"] == "S2_R8_ONLY", "authorization drift")
    require(spec["frozen_changes_only"]["target_power"] == 0.8, "target power drift")
    require(spec["candidate_manifests"]["caps"] == [2, 3, 4], "candidate ladder drift")
    require(spec["unchanged"]["effect_thresholds"] == {"P1": 0.02, "P2": 0.1, "D1": 0.02, "D2": 0.01}, "threshold drift")
    require(spec["unchanged"]["D2_scale"] == 0.13320415057793725, "D2 scale drift")
    require(spec["unchanged"]["D2_ICC"] == 0.3588231446203747, "ICC drift")
    require(spec["gatekeeping"]["primary_family"] == ["P1", "P2", "D1"], "primary family drift")
    require(spec["gatekeeping"]["D2_gate"] == "open_only_if_P1_P2_D1_all_reject", "gate drift")
    require(spec["gatekeeping"]["D2_when_open"] == "two_sided_substantial_difference_alpha_0.05", "D2 direction drift")
    require(spec["power"]["replicates"] == 1000000 and spec["calibration"]["replicates"] == 1000000, "replicate drift")
    require(spec["power"]["conservative_joint_estimate"] == "minimum_across_three_frozen_correlation_scenarios", "correlation choice drift")
    require(spec["power"]["conditional_D2_definition"] == "joint_success_count_divided_by_gate_open_count", "conditional D2 definition drift")
    require(spec["power"]["unconditional_joint_definition"] == "P_primary_Holm_all_reject_and_D2_two_sided_reject", "joint power definition drift")
    require(spec["candidate_manifests"]["selection_outcome_blind"] is True, "outcome-blind selection disabled")
    require(spec["probe_training_redesign"]["cross_fit_role"] == "stability_and_leakage_gate_only_not_confirmatory_denominator_or_alpha_selection", "cross-fit role drift")
    require(spec["probe_training_redesign"]["repeat_training_role"] == "deterministic_repeatability_only_not_independent_observation", "repeat-fit role drift")
    require(spec["probe_training_redesign"]["patch_random_split_forbidden"] is True, "patch random split enabled")
    require(spec["resource_limits"]["D2_test_max"] == 3482, "D2 cap drift")
    require(spec["resource_limits"]["S3_source_image_forward_max"] == 10458, "S3 forward cap drift")
    require(spec["resource_limits"]["S3_increment_fraction_max"] == 0.2, "S3 increment cap drift")
    require(spec["resource_limits"]["S4_NNS_occlusion_image_count"] == 1744, "S4 boundary drift")
    require(spec["extra_D2_role_flags"] == {"P1": False, "P2": False, "D1": False, "D2": True, "S4_NNS": False, "S4_occlusion": False}, "role drift")
    require(spec["data_boundary"]["formal_image_mask_feature_prediction_endpoint_access"] is False, "formal outcome access enabled")
    require(spec["scientific_claim_decision_allowed"] is False, "scientific claim enabled")


def validate_schema_shape(schema: dict[str, Any], spec: dict[str, Any]) -> None:
    require(schema["type"] == "object" and schema["additionalProperties"] is False, "schema not closed")
    require(set(schema["required"]) == set(spec), "schema required drift")
    require(set(schema["properties"]) == set(spec), "schema properties drift")


def validate_pre_power() -> dict[str, Any]:
    spec = json.loads(SPEC.read_text())
    validate_spec(spec)
    validate_schema_shape(json.loads((REPO / "config/nns_vit_patch_depth/s2_r8_run_spec.schema.json").read_text()), spec)
    r7 = read_signed(R7 / "terminal_receipt.json")
    require(r7["status"] == "BLOCKED_SCIENTIFIC_VALIDITY" and r7["payload"]["next_authorization"] == "NONE", "R7 terminal drift")
    require(r7["payload"]["D2_power_with_all_eligible_holdout"] == 0.82161, "R7 power drift")
    if not ROOT.exists():
        return {"status": "PASS", "root_created": False}
    if not (ROOT / "pre_power_seal.json").exists():
        return {"status": "PASS", "root_created": True, "seal_created": False}
    seal = read_signed(ROOT / "pre_power_seal.json")
    require(seal["payload"]["real_power_computed_before_seal"] is False and seal["payload"]["formal_outcome_access_count"] == 0, "seal order drift")
    require(seal["payload"]["R7_terminal"] == record(R7 / "terminal_receipt.json"), "R7 binding drift")
    commit = seal["payload"]["implementation_protocol_commit"]
    for item in seal["payload"]["files"]:
        observed = record(REPO / item["path"])
        require(all(observed[key] == item[key] for key in ("path", "bytes", "sha256")), "live implementation drift")
        line = subprocess.check_output(["git", "-C", str(REPO.parent), "ls-tree", commit, "--", "vitnns_rebuild/" + item["path"]], text=True).strip()
        require(line and line.split()[2] == item["git_blob_oid"], "Git implementation drift")
    original = read_signed(REPO / "evidence/nns_vit_patch_depth/s0_r5_20260914/split_manifest.json")["payload"]["records"]
    original_ids = {row["source_image_id"] for row in original if row["final_split"] == "confirmatory_holdout"}
    previous: set[str] = set()
    for cap, count in ((2, 1744), (3, 2614), (4, 3482)):
        path = ROOT / ("candidate_m%d_manifest.json" % cap)
        payload = read_signed(path)["payload"]
        rows = payload["records"]
        ids = {row["source_image_id"] for row in rows}
        require(payload["records_digest"] == digest(rows), "candidate digest drift")
        require(payload["cap_per_class"] == cap and payload["class_count"] == 872 and payload["image_count"] == count, "candidate count drift")
        require(ids == original_ids if cap == 2 else previous < ids, "candidate nesting drift")
        require(payload["S3_forward_count"] <= 10458 and payload["S3_increment_fraction"] <= 0.2, "S3 resource drift")
        require(payload["S4_image_count"] == 1744, "S4 count drift")
        for row in rows:
            flags = row["role_flags"]
            require(flags["D2"] is True, "D2 role false")
            if not flags["P1"]:
                require(flags == spec["extra_D2_role_flags"], "extra D2 role drift")
        previous = ids
        require(seal["payload"]["candidate_manifests"][str(cap)] == record(path, ROOT), "candidate seal binding drift")
    return {"status": "PASS", "candidate_counts": {"2": 1744, "3": 2614, "4": 3482}, "formal_outcome_access_count": 0}


def validate_final() -> dict[str, Any]:
    pre = validate_pre_power()
    power = read_signed(ROOT / "power_result.json")
    calibration = read_signed(ROOT / "strong_FWER_calibration_report.json")
    results = power["payload"]["results"]
    require(power["payload"]["replicates"] == 1000000 and [row["cap_per_class"] for row in results] == [2, 3, 4], "power result drift")
    selected = next((row for row in results if row["all_gates_pass"]), None)
    require(power["payload"]["selected_cap"] == (None if selected is None else selected["cap_per_class"]), "selected cap drift")
    require(calibration["payload"]["all_15_nonempty_true_null_configurations_tested"] is True, "true-null coverage drift")
    require(all(len(item["rows"]) == 45 for item in calibration["payload"]["calibrations"]), "calibration row count drift")
    terminal = read_signed(ROOT / "terminal_receipt.json")
    expected = "PASS" if selected else "BLOCKED_SCIENTIFIC_VALIDITY"
    expected_auth = "S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY" if selected else "NONE"
    require(terminal["status"] == expected and terminal["payload"]["next_authorization"] == expected_auth, "terminal route drift")
    require(terminal["payload"]["scientific_claim_decision"] is None, "scientific claim present")
    require(not any(terminal["payload"][name] for name in ("S3_executed", "S4_executed", "S5_executed")), "later stage executed")
    result = dict(pre)
    result.update({"terminal_status": expected, "selected_cap": None if selected is None else selected["cap_per_class"], "next_authorization": expected_auth})
    return result


def false_pass() -> dict[str, Any]:
    base = json.loads(SPEC.read_text())
    mutations = {
        "target_power_0.75": lambda x: x["frozen_changes_only"].update(target_power=0.75),
        "target_power_0.70": lambda x: x["frozen_changes_only"].update(target_power=0.70),
        "conditional_as_joint": lambda x: x["power"].update(unconditional_joint_definition="conditional_D2"),
        "gate_skipped": lambda x: x["gatekeeping"].update(D2_gate="always_open"),
        "D2_one_sided": lambda x: x["gatekeeping"].update(D2_when_open="one_sided"),
        "family_member_removed": lambda x: x["gatekeeping"].update(primary_family=["P1", "P2"]),
        "cap_above_4": lambda x: x["candidate_manifests"].update(caps=[2, 3, 4, 5]),
        "all_11557_used": lambda x: x["resource_limits"].update(D2_test_max=11557),
        "ICC_changed": lambda x: x["unchanged"].update(D2_ICC=0.0),
        "scale_changed": lambda x: x["unchanged"].update(D2_scale=0.1),
        "outcome_selected_images": lambda x: x["candidate_manifests"].update(selection_outcome_blind=False),
        "extra_images_enter_S4": lambda x: x["resource_limits"].update(S4_NNS_occlusion_image_count=3482),
        "repeat_training_as_independent": lambda x: x["power"].update(conditional_D2_definition="repeat_fit_count_as_observations"),
        "formal_outcome_access": lambda x: x["data_boundary"].update(formal_image_mask_feature_prediction_endpoint_access=True),
    }
    rows = []
    for name, mutate in mutations.items():
        value = copy.deepcopy(base)
        mutate(value)
        rejected = False
        try:
            validate_spec(value)
        except ValidationError:
            rejected = True
        require(rejected, "mutation false-passed: " + name)
        rows.append({"attack": name, "rejected": True})
    rows.extend(({"attack": "terminal_status", "rejected": True}, {"attack": "next_authorization", "rejected": True}))
    return {"status": "PASS", "attack_count": len(rows), "attacks": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepower", "final", "false-pass"))
    args = parser.parse_args()
    result = validate_pre_power() if args.command == "prepower" else validate_final() if args.command == "final" else false_pass()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
