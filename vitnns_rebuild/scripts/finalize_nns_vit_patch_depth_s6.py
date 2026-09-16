#!/usr/bin/env python3
"""Finalize the validated S6 post-hoc exploratory reporting package."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s6 as s6  # noqa: E402

ROOT = REPO / s6.ROOT_REL
REPORT = REPO / s6.REPORT_REL
PYTHON = REPO.parent / ".conda/envs/vitnns/bin/python"


def doc(kind: str, status: str, payload: dict, commit: str) -> dict:
    value = s6.signed(kind, status, payload, commit); s6.verify_closed_document(value); return value


def run(command: str) -> dict:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO / "src")}
    output = subprocess.check_output([str(PYTHON), "scripts/validate_nns_vit_patch_depth_s6.py", command], cwd=REPO, text=True, env=environment)
    return json.loads(output)


def replace(name: str, value: dict) -> None:
    path = ROOT / name
    if path.exists():
        prior = json.loads(path.read_text()); s6.require(prior.get("run_id") == s6.RUN_ID, f"foreign S6 artifact: {name}"); path.unlink()
    s6.write_json_new(path, value)


def main() -> int:
    commit = s6.git(REPO, "rev-parse", "HEAD")
    validation = run("validate"); mutation = run("false-pass")
    s6.require(validation["all_checks_pass"] and mutation["all_rejected"], "S6 validation failed")
    replace("independent_validation_report.json", doc("s6_independent_validation_report", "PASS_POST_HOC_EXPLORATORY", validation, commit))
    replace("false_pass_mutation_audit.json", doc("s6_false_pass_mutation_audit", "PASS_POST_HOC_EXPLORATORY", mutation, commit))
    excluded = {"whole_root_inventory.json", "terminal_receipt.json"}
    files = [s6.artifact(REPO, path) for path in sorted(ROOT.rglob("*")) if path.is_file() and path.name not in excluded]
    report_files = [s6.artifact(REPO, path) for path in sorted(REPORT.rglob("*")) if path.is_file()]
    inventory_payload = {"analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory whole-root inventory", "files": files + report_files, "file_count": len(files) + len(report_files), "files_digest": s6.canonical_digest(files + report_files), "excluded_late_bound": sorted(excluded)}
    replace("whole_root_inventory.json", doc("s6_whole_root_inventory", "PASS_POST_HOC_EXPLORATORY", inventory_payload, commit))
    endpoint_rows = json.loads((ROOT / "figure_data.json").read_text())["payload"]["endpoint_rows"]
    required = [path for path in sorted(ROOT.rglob("*")) if path.is_file() and path.name != "terminal_receipt.json"]
    terminal_payload = {
        "analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory reporting terminal receipt",
        "S6_WORKFLOW_STATUS": "PASS", "SCIENTIFIC_STATUS": "PASS_POST_HOC_EXPLORATORY", "reporting_only": True,
        "new_scientific_analysis_count": 0, "new_intervention_count": 0, "original_P2_rule_status": "ORIGINAL_RULE_NOT_PASSED",
        "endpoint_states": {row["endpoint"]: row["exploratory_state"] for row in endpoint_rows},
        "paper_conclusion": "NNS patch alignment, matched occlusion response, and both depth contrasts were positive; D1 and D2 showed near-zero image-level coupling.",
        "required_evidence": {path.relative_to(ROOT).as_posix(): s6.artifact(REPO, path) for path in required},
        "report_copy": {path.relative_to(REPORT).as_posix(): s6.artifact(REPO, path) for path in sorted(REPORT.rglob("*")) if path.is_file()},
        "next_authorization": "NONE",
    }
    terminal = doc("s6_terminal_receipt", "PASS_POST_HOC_EXPLORATORY_REPORTING", terminal_payload, commit)
    replace("terminal_receipt.json", terminal)
    print(json.dumps({"status": terminal["status"], "canonical_digest": terminal["canonical_digest"], "next_authorization": "NONE", "render_count": validation["render_count"]}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
