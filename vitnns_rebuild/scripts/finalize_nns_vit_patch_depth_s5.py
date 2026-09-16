#!/usr/bin/env python3
"""Finalize the validated S5 post-hoc exploratory evidence package."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s5 as s5  # noqa: E402

ROOT = REPO / s5.ROOT_REL
PYTHON = REPO.parent / ".conda/envs/vitnns/bin/python"


def document(document_type: str, status: str, payload: dict, commit: str) -> dict:
    result = s5.signed(document_type, status, payload, commit)
    s5.verify_closed_document(result)
    return result


def run_validator(command: str) -> dict:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO / "src")}
    output = subprocess.check_output([str(PYTHON), "scripts/validate_nns_vit_patch_depth_s5.py", command], cwd=REPO, text=True, env=environment)
    return json.loads(output)


def replace_derived(name: str, value: dict) -> None:
    path = ROOT / name
    if path.exists():
        prior = json.loads(path.read_text())
        s5.require(prior.get("run_id") == s5.RUN_ID, f"refusing to replace foreign artifact: {name}")
        path.unlink()
    s5.write_json_new(path, value)


def main() -> int:
    commit = s5.git(REPO, "rev-parse", "HEAD")
    validation = run_validator("validate")
    false_pass = run_validator("false-pass")
    s5.require(all(validation["checks"].values()) and false_pass["all_rejected"], "S5 validation failed")
    replace_derived("independent_validation_report.json", document("s5_independent_validation_report", "PASS_POST_HOC_EXPLORATORY", validation, commit))
    replace_derived("false_pass_mutation_audit.json", document("s5_false_pass_mutation_audit", "PASS_POST_HOC_EXPLORATORY", false_pass, commit))

    exclude = {"whole_root_inventory.json", "terminal_receipt.json"}
    files = [s5.artifact(REPO, path) for path in sorted(ROOT.glob("*")) if path.is_file() and path.name not in exclude]
    inventory_payload = {
        "analysis_label": s5.ANALYSIS_LABEL, "title": "S5 post-hoc exploratory whole-root inventory",
        "files": files, "file_count": len(files), "files_digest": s5.canonical_digest(files),
        "excluded_late_bound": sorted(exclude),
    }
    replace_derived("whole_root_inventory.json", document("s5_whole_root_inventory", "PASS_POST_HOC_EXPLORATORY", inventory_payload, commit))

    decisions = json.loads((ROOT / "endpoint_decisions.json").read_text())["payload"]["endpoints"]
    identity = validation["recomputed_identity"]
    required_paths = [path for path in sorted(ROOT.glob("*")) if path.is_file() and path.name != "terminal_receipt.json"]
    terminal_payload = {
        "analysis_label": s5.ANALYSIS_LABEL, "title": "S5 post-hoc exploratory terminal receipt",
        "S5_WORKFLOW_STATUS": "PASS", "SCIENTIFIC_STATUS": "PASS_POST_HOC_EXPLORATORY",
        "FORMAL_CONFIRMATORY_GATE_OPEN": False, "posthoc_exploratory_only": True,
        "strong_FWER_preserved_claim": False, "original_S4_R10_STOP_SUCCESS_claimed": False,
        "original_P2_rule_status": "ORIGINAL_RULE_NOT_PASSED",
        "endpoint_states": {endpoint: value["exploratory_state"] for endpoint, value in decisions.items()},
        "R12_selection_identity": {
            "selected_source_count": identity["selected_source_count"],
            "selected_distinct_class_rank_count": identity["selected_distinct_class_rank_count"],
            "P1_D1_valid_class_count": identity["P1_D1_valid_class_count"],
            "P2_valid_class_count": identity["P2_valid_class_count"],
            "full_manifest_records_digest": identity["full_manifest_records_digest"],
            "membership_projection_digest": identity["membership_projection_digest"],
        },
        "required_evidence": {path.name: s5.artifact(REPO, path) for path in required_paths},
        "S6_execution_count": 0, "next_authorization": "S6_POST_HOC_EXPLORATORY_REPORTING_ONLY",
    }
    terminal = document("s5_terminal_receipt", "PASS_POST_HOC_EXPLORATORY", terminal_payload, commit)
    replace_derived("terminal_receipt.json", terminal)
    print(json.dumps({"status": terminal["status"], "canonical_digest": terminal["canonical_digest"], "next_authorization": terminal_payload["next_authorization"], "endpoint_states": terminal_payload["endpoint_states"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
