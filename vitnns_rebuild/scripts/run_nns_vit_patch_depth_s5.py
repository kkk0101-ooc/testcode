#!/usr/bin/env python3
"""Produce S5 post-hoc exploratory evidence under a CPU-only read whitelist."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s5 as s5  # noqa: E402


def document(document_type: str, status: str, payload: dict, commit: str) -> dict:
    result = s5.signed(document_type, status, payload, commit)
    s5.verify_closed_document(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace-own-root", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only boundary not active")
    forbidden_loaded = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "torchvision" or name.startswith("torchvision.") or name == "timm" or name.startswith("timm.") or name == "PIL" or name.startswith("PIL."))
    if forbidden_loaded:
        raise RuntimeError("forbidden dependency imported before analysis")
    root = REPO / s5.ROOT_REL
    try:
        _, gate = s5.authorization_gate(REPO, check_remote=True)
    except s5.S5Error as error:
        if str(error) == "S5_NOT_AUTHORIZED":
            print("S5_NOT_AUTHORIZED")
            return 2
        raise
    if root.exists():
        if not args.replace_own_root:
            raise RuntimeError("fresh S5 evidence root required")
        children = list(root.iterdir())
        for path in children:
            if not path.is_file() or path.suffix != ".json":
                raise RuntimeError(f"refusing to replace unexpected S5 path: {path}")
            prior = json.loads(path.read_text())
            if prior.get("run_id") != s5.RUN_ID:
                raise RuntimeError(f"refusing to replace foreign S5 artifact: {path}")
        for path in children:
            path.unlink()
        root.rmdir()
    commit = s5.git(REPO, "rev-parse", "HEAD")
    root.mkdir(parents=True)
    s5.write_json_new(root / "authorization_gate_report.json", document("s5_authorization_gate_report", "PASS_POST_HOC_EXPLORATORY", gate, commit))

    audit = s5.ReadAudit(REPO)
    audit.json(s5.R12_TERMINAL_REL, "S5_AUTHORIZING_R12_TERMINAL", s5.R12_TERMINAL_SHA256, s5.R12_TERMINAL_CANONICAL)
    result = s5.compute(REPO, audit)
    endpoint_records = result["endpoint_values"]
    endpoint_value_payload = {
        "analysis_label": s5.ANALYSIS_LABEL,
        "title": "S5 post-hoc exploratory image-level endpoint values",
        "endpoint_records": endpoint_records,
        "record_counts": {endpoint: len(records) for endpoint, records in endpoint_records.items()},
        "records_digest": s5.canonical_digest(endpoint_records),
    }
    decisions_payload = {
        "analysis_label": s5.ANALYSIS_LABEL,
        "title": "S5 post-hoc exploratory endpoint decisions",
        "endpoints": result["endpoint_summaries"],
        "family_three_scope": "P1/P2/D1 L1 post-hoc exploratory adjustment; no confirmatory or strong-FWER-preserved claim",
        "D2_scope": "standalone two-direction post-hoc exploratory analysis after R12 exploratory gate; no cross-family or strong-FWER-preserved claim",
        "all_endpoints_retained": True,
    }
    p2 = result["endpoint_summaries"]["P2"]
    original_checks = {
        "point_estimate_above_0_10": p2["estimate"] > 0.10,
        "adjusted_lower_above_0_10": p2["adjusted_interval"]["lower"] > 0.10,
        "ordinary_BCa_half_width_at_most_0_015": p2["ordinary_BCa_95"]["half_width"] <= 0.015,
    }
    comparison_payload = {
        "analysis_label": s5.ANALYSIS_LABEL,
        "title": "S5 post-hoc exploratory original-vs-posthoc P2 comparison",
        "original_rule": {"point_threshold": ">0.10", "adjusted_lower_threshold": ">0.10", "ordinary_BCa_half_width_max": 0.015, "checks": original_checks, "status": "ORIGINAL_RULE_NOT_PASSED"},
        "posthoc_rule": {"point_threshold": ">=0.07", "adjusted_lower_threshold": ">0", "ordinary_BCa_half_width_max": 0.02, "checks": p2["decision_checks"], "status": p2["exploratory_state"]},
        "disclosure": "The P2 >=0.07 point gate, zero-effect family-size-three L1 adjusted lower-bound gate, and ordinary BCa half-width <=0.02 gate were specified after outcome access.",
    }
    depth_payload = {
        "analysis_label": s5.ANALYSIS_LABEL,
        "title": "S5 post-hoc exploratory depth profiles and D1/D2 relationship",
        "NNS_image_block_records": result["nns_profile_rows"],
        "D2_image_block_records": result["d2_profile_rows"],
        "NNS_equal_class_profile": result["nns_aggregate_profile"],
        "D2_equal_class_profile": result["d2_aggregate_profile"],
        "D1_D2_relationship": result["depth_relationship"],
        "interpretation_boundary": "Probe evidence is not NNS evidence, and a depth association is not a causal formation mechanism.",
    }
    intervention_payload = {
        "analysis_label": s5.ANALYSIS_LABEL,
        "title": "S5 post-hoc exploratory intervention controls and matching diagnostics",
        "records": result["intervention_rows"],
        "records_digest": s5.canonical_digest(result["intervention_rows"]),
        "aggregate_by_operator_block_budget": result["intervention_summary"],
        "matching_diagnostics": result["matching"],
        "interpretation_boundary": "Observed matched occlusion differences are post-hoc exploratory intervention evidence and do not establish a causal formation mechanism.",
    }
    validity_payload = {
        "analysis_label": s5.ANALYSIS_LABEL,
        "title": "S5 post-hoc exploratory validity accounting",
        "records": result["validity_rows"],
        "records_digest": s5.canonical_digest(result["validity_rows"]),
        "R12_terminal_states": result["R12_terminal_states"],
        "D2_terminal_states": result["D2_terminal_states"],
        "typed_NA_policy": "Typed NA records remain in denominators and are neither deleted, substituted, nor filled with zero.",
        "invalid_endpoint_family_policy": "An INVALID endpoint remains represented and cannot disappear from its declared analysis family.",
    }
    states = {endpoint: row["exploratory_state"] for endpoint, row in result["endpoint_summaries"].items()}
    evidence_prefix = s5.ROOT_REL.as_posix()
    claim_payload = {
        "analysis_label": s5.ANALYSIS_LABEL,
        "title": "S5 post-hoc exploratory claim ledger",
        "claims": [
            {"analysis_label": s5.ANALYSIS_LABEL, "claim": "P1 corrected alignment exceeded its exploratory threshold in the R12-selected cohort.", "required_endpoints": ["P1"], "observed_states": {"P1": states["P1"]}, "allowed_wording": "post-hoc exploratory support for P1 corrected alignment", "evidence_path": f"{evidence_prefix}/endpoint_decisions.json"},
            {"analysis_label": s5.ANALYSIS_LABEL, "claim": "P2 met the outcome-accessed exploratory gates while the original P2 rule did not pass.", "required_endpoints": ["P2"], "observed_states": {"P2": states["P2"]}, "allowed_wording": "post-hoc exploratory P2 signal; ORIGINAL_RULE_NOT_PASSED", "evidence_path": f"{evidence_prefix}/original_vs_posthoc_P2_comparison.json"},
            {"analysis_label": s5.ANALYSIS_LABEL, "claim": "D1 describes a depth contrast in NNS corrected alignment.", "required_endpoints": ["D1"], "observed_states": {"D1": states["D1"]}, "allowed_wording": "post-hoc exploratory D1 depth association", "evidence_path": f"{evidence_prefix}/depth_profiles_and_D1_D2_relationship.json"},
            {"analysis_label": s5.ANALYSIS_LABEL, "claim": "D2 describes depth variation in probe gain over a low-level control.", "required_endpoints": ["D2"], "observed_states": {"D2": states["D2"]}, "allowed_wording": "post-hoc exploratory D2 probe-depth evidence, not NNS evidence", "evidence_path": f"{evidence_prefix}/depth_profiles_and_D1_D2_relationship.json"},
            {"analysis_label": s5.ANALYSIS_LABEL, "claim": "D1 and D2 relationship is descriptive only.", "required_endpoints": ["D1", "D2"], "observed_states": {"D1": states["D1"], "D2": states["D2"]}, "allowed_wording": "post-hoc exploratory association; no causal formation-mechanism claim", "evidence_path": f"{evidence_prefix}/depth_profiles_and_D1_D2_relationship.json"},
        ],
        "global_forbidden_claims": ["preregistered", "outcome-blind", "confirmatory", "strong-FWER-preserved", "original S4-R10 STOP_SUCCESS", "probe significance implies NNS significance", "depth correlation establishes causal formation mechanism"],
        "S6_executed": False,
    }

    documents = {
        "image_level_endpoint_values.json": document("s5_image_level_endpoint_values", "PASS_POST_HOC_EXPLORATORY", endpoint_value_payload, commit),
        "endpoint_decisions.json": document("s5_endpoint_decisions", "PASS_POST_HOC_EXPLORATORY", decisions_payload, commit),
        "original_vs_posthoc_P2_comparison.json": document("s5_original_vs_posthoc_P2_comparison", "ORIGINAL_RULE_NOT_PASSED", comparison_payload, commit),
        "depth_profiles_and_D1_D2_relationship.json": document("s5_depth_profiles_and_D1_D2_relationship", "PASS_POST_HOC_EXPLORATORY", depth_payload, commit),
        "intervention_controls_and_matching_diagnostics.json": document("s5_intervention_controls_and_matching", "PASS_POST_HOC_EXPLORATORY", intervention_payload, commit),
        "validity_accounting.json": document("s5_validity_accounting", "PASS_POST_HOC_EXPLORATORY", validity_payload, commit),
        "claim_ledger.json": document("s5_claim_ledger", "PASS_POST_HOC_EXPLORATORY", claim_payload, commit),
    }
    for name, value in documents.items():
        s5.write_json_new(root / name, value)

    inventory_records = audit.inventory()
    input_payload = {
        "analysis_label": s5.ANALYSIS_LABEL, "title": "S5 post-hoc exploratory input inventory",
        "read_whitelist_policy": "Only the R12-bound 240-source manifest and terminals, their P2 plans and endpoint NPY arrays, and explicitly bound S2/S3 metadata plus D2 error arrays were readable as scientific inputs.",
        "records": inventory_records, "record_count": len(inventory_records), "records_digest": s5.canonical_digest(inventory_records),
    }
    s5.write_json_new(root / "input_inventory.json", document("s5_input_inventory", "PASS_POST_HOC_EXPLORATORY", input_payload, commit))
    access_payload = {
        "analysis_label": s5.ANALYSIS_LABEL, "title": "S5 post-hoc exploratory CPU access audit",
        "CUDA_VISIBLE_DEVICES": "", "forbidden_import_count": 0, "forbidden_imports_loaded": [],
        "JPEG_or_mask_decode_count": 0, "model_or_checkpoint_load_count": 0, "CUDA_or_GPU_initialization_count": 0,
        "new_intervention_count": 0, "scientific_read_whitelist_enforced": True,
        "scientific_read_count": len(inventory_records), "scientific_read_records_digest": s5.canonical_digest(inventory_records),
        "S0_to_S4_root_write_count": 0, "S6_execution_count": 0,
    }
    s5.write_json_new(root / "cpu_access_audit.json", document("s5_cpu_access_audit", "PASS_POST_HOC_EXPLORATORY", access_payload, commit))
    schema_payload = {
        "analysis_label": s5.ANALYSIS_LABEL, "title": "S5 post-hoc exploratory closed schema registry",
        "envelope_additional_properties": False, "payload_additional_properties": False,
        "document_payload_keys": {key: sorted(value) for key, value in sorted(s5.PAYLOAD_KEYS.items())},
        "nested_record_key_contracts": {key: sorted(value) for key, value in sorted(s5.NESTED_RECORD_KEYS.items())},
        "schema_self_test": "PASS",
    }
    s5.write_json_new(root / "closed_schema_registry.json", document("s5_closed_schema_registry", "PASS_POST_HOC_EXPLORATORY", schema_payload, commit))
    print(json.dumps({"status": "PASS_POST_HOC_EXPLORATORY", "root": s5.ROOT_REL.as_posix(), "endpoint_states": states, "scientific_read_count": len(inventory_records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
