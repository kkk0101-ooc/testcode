#!/usr/bin/env python3
"""Source-recomputing validator for formal Stage Y2 terminals."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from vitnns.constants import REPO_ROOT
from vitnns import stage_y_y2 as y2

PASS_REF = "refs/archive/stage-y-y2-clean-cohort-evidence"
BLOCKED_REF = "refs/archive/stage-y-y2-clean-cohort-evidence-blocked"
R1_REF = "refs/archive/stage-y-y2-p0-r1-elastic-schedule-freeze"
R1_COMPLETION = Path("evidence/stage_y/y2_p0_r1_elastic_schedule_20260911/formal/completion/y2_p0_r1.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise y2.StageYY2ProvenanceError(message)


def remote_ref(repo: Path, ref: str) -> str:
    parts = y2.git(repo, "ls-remote", "origin", ref).split()
    require(len(parts) == 2 and parts[1] == ref, f"remote ref missing or ambiguous: {ref}")
    return parts[0]


def validate_schema(repo: Path, relative: str, value: Mapping[str, Any]) -> None:
    schema = y2.read_json(repo / relative)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path))
    require(not errors, f"schema failure {relative}: {errors[0].message if errors else ''}")


def record(repo: Path, relative: Path) -> dict[str, Any]:
    result = y2.file_record(repo / relative); result["path"] = relative.as_posix(); return result


def validate_implementation_binding(repo: Path) -> dict[str, Any]:
    seal = y2.read_json(repo / y2.ROOT / "implementation_seal.json"); y2.validate_signed(seal)
    records = [record(repo, Path(row["path"])) for row in seal["payload"]["source_files"]]
    require(records == seal["payload"]["source_files"], "implementation seal source drift")
    require(y2.canonical_digest({"files": records}) == seal["payload"]["files_digest"], "implementation seal digest drift")
    return {"source_file_count": len(records), "files_digest": seal["payload"]["files_digest"]}


def validate_scientific_result(value: Mapping[str, Any], terminal: Mapping[str, Any]) -> None:
    y2.validate_signed(value)
    require(value["document_type"] == "stage_y_y2_unit_scientific_result" and value["status"] == "VALID", "scientific result identity drift")
    payload = value["payload"]
    keys = {"target_identity","clean_logit","baseline_logit","delta","selected_steps","closure","ablations","deletion_operator","calls","phase_telemetry"}
    require(set(payload) == keys, "scientific result payload is not closed")
    require(payload["target_identity"] == terminal["payload"]["target_identity"], "scientific target identity drift")
    require(abs(float(payload["clean_logit"]) - float(payload["baseline_logit"]) - float(payload["delta"])) < 1e-9, "scientific delta drift")
    attempts = terminal["payload"]["quadrature_attempts"]
    require(payload["selected_steps"] in (128,256,512,1024,2048,4096,8192) and attempts and attempts[-1]["steps"] == payload["selected_steps"] and attempts[-1]["all_contracts_pass"] is True and not any(row["all_contracts_pass"] for row in attempts[:-1]), "first-passing quadrature selection drift")
    closure = payload["closure"]; require(closure["status"] == "PASS" and closure["failures"] == [] and closure["input_abs_error"] <= closure["input_threshold"], "closure validation failed")
    ablations = payload["ablations"]; require(ablations["status"] == "PASS" and ablations["single_head_coordinates"] == 144 and ablations["curve_points"] == 192 and len(ablations["blocks"]) == 12, "ablation coverage drift")
    for block in ablations["blocks"]:
        require(len(block["single_head"]) == 12 and [row["k"] for row in block["curves"]["top"]] == [1,2,3,4,6,8,10,12] and [row["k"] for row in block["curves"]["bottom"]] == [1,2,3,4,6,8,10,12], "ablation block coverage drift")
    deletion = payload["deletion_operator"]; require(deletion["status"] == "PASS" and deletion["scientific_decision"] is None and len(deletion["blocks"]) == 12, "deletion control drift")
    for block in deletion["blocks"]: require(block["percentiles"] == list(range(0,101,5)) and len(block["controls"]) == 32, "deletion block coverage drift")
    required_calls = {"forward_calls","backward_calls","jvp_calls","jvp_direction_lanes","vjp_calls","quadrature_alpha_batches"}
    require(set(payload["calls"]) == required_calls and all(int(payload["calls"][key]) > 0 for key in required_calls), "scientific call telemetry incomplete")
    phase = payload["phase_telemetry"]; require(set(phase) == {"forward_backward_wall_seconds","jvp_groups","vjp_groups","jvp_wall_seconds","vjp_wall_seconds"} and phase["forward_backward_wall_seconds"] > 0 and phase["jvp_wall_seconds"] > 0 and phase["vjp_wall_seconds"] > 0, "grouped integration phase telemetry incomplete")


def validate(repo: Path, publication: bool = False, allow_blocked: bool = False) -> dict[str, Any]:
    root = repo / y2.ROOT
    completion_path = root / "completion/y2.json"
    require(completion_path.is_file() and not (root / "completion/y2.pending.json").exists(), "Y2 terminal transition incomplete")
    completion = y2.read_json(completion_path); y2.validate_signed(completion)
    validate_schema(repo, "config/stage_y/y2_completion.schema.json", completion)
    status = completion["status"]
    if allow_blocked:
        require(status != "PASS" and completion["payload"]["next_authorization"] == "NONE", "blocked terminal semantics drift")
    else:
        require(status == "PASS" and completion["payload"]["next_authorization"] == "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY", "PASS terminal semantics drift")
    implementation_path = root / "implementation_seal.json"
    if implementation_path.is_file():
        binding = validate_implementation_binding(repo)
    else:
        require(status != "PASS", "PASS terminal omitted implementation seal")
        binding = {"source_file_count": 0, "files_digest": None, "seal_absent_before_control_failure": True}
    inventory = y2.read_json(root / "whole_root_inventory.json"); y2.validate_signed(inventory)
    require(y2.whole_root_inventory(root, status)["payload"] == inventory["payload"], "whole-root inventory drift")
    require(completion["payload"]["whole_root_inventory"] == record(repo, y2.ROOT / "whole_root_inventory.json"), "completion inventory link drift")
    summary = y2.read_json(root / "terminal_access_summary.json"); y2.validate_signed(summary)
    require(completion["payload"]["terminal_access_summary"] == record(repo, y2.ROOT / "terminal_access_summary.json"), "completion access-summary link drift")
    require(completion["payload"]["report"] == record(repo, y2.REPORT_ROOT / "y2_report.md"), "completion report link drift")
    require(summary["payload"]["non_clean_condition_count"] == 0 and not summary["payload"]["later_stages_started"], "Y2 scope boundary violated")
    require(not summary["payload"]["bootstrap_executed"] and not summary["payload"]["holm_executed"] and not summary["payload"]["claim_decision_executed"], "Y2 statistical boundary violated")
    if status == "PASS":
        reconciliation = y2.reconcile_shards(root)
        require(reconciliation["status"] == "PASS" and reconciliation == y2.read_json(root / "reconciliation.json"), "reconciliation source recomputation drift")
        require(completion["payload"]["reconciliation"] == record(repo, y2.ROOT / "reconciliation.json") and completion["payload"]["failure"] is None, "PASS completion result links drift")
        for terminal_path in sorted(root.glob("shards/shard-*/units/unit-*/completion/unit.json")):
            terminal = y2.read_json(terminal_path); validate_schema(repo, "config/stage_y/y2_unit_terminal.schema.json", terminal)
            if terminal["status"] == "VALID":
                require(len(terminal["payload"]["array_records"]) == 262, "VALID unit array roster must contain 262 arrays")
                scientific_relative = Path(terminal["payload"]["scientific_record"]["path"])
                scientific_path = scientific_relative if scientific_relative.is_absolute() else repo / scientific_relative
                require(y2.file_record(scientific_path) == terminal["payload"]["scientific_record"], "scientific result file binding drift")
                validate_scientific_result(y2.read_json(scientific_path), terminal)
        gate = y2.read_json(root / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
        validate_schema(repo, "config/stage_y/y2_startup_gate.schema.json", gate)
        worker_count = y2.validate_worker_count(gate["payload"]["locked_worker_count"])
        require(len(gate["payload"]["devices"]) == worker_count, "startup device count differs from locked worker count")
        ledger = y2.read_json(repo / "evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json")
        expected_projection = y2.schedule_from_source_ledger(ledger, worker_count)["y2"]
        require(gate["payload"]["resource_projection"] == expected_projection, "startup resource projection differs from source ledger")
        manifest_paths = sorted(root.glob("shards/shard-*/shard_manifest.json"))
        require(len(manifest_paths) == worker_count, "manifest roster differs from locked worker count")
        for shard_id, manifest_path in enumerate(manifest_paths):
            manifest = y2.read_json(manifest_path)
            validate_schema(repo, "config/stage_y/y2_shard_manifest.schema.json", manifest)
            require(manifest["payload"]["resource_caps"] == expected_projection["per_shard"][shard_id], "manifest cap differs from source projection")
    else:
        require(completion["payload"]["failure"] is not None, "blocked terminal missing failure receipt")
        require((root / "failure.json").is_file(), "blocked failure receipt absent")
        require(completion["payload"]["failure"] == record(repo, y2.ROOT / "failure.json"), "blocked failure link drift")
    if publication:
        head = y2.git(repo, "rev-parse", "HEAD")
        require(y2.git(repo, "rev-parse", "@{upstream}") == remote_ref(repo, f"refs/heads/{y2.EXPECTED_BRANCH}") == head, "published branch drift")
        ref = PASS_REF if status == "PASS" else BLOCKED_REF
        require(y2.git(repo, "rev-parse", ref) == remote_ref(repo, ref) == head, "published Y2 archive drift")
        require(not y2.git(repo, "status", "--porcelain"), "publication worktree/index dirty")
        lfs = y2.git(repo, "lfs", "status")
        require("Git LFS objects to be committed:" in lfs and "Git LFS objects not staged for commit:" in lfs, "publication LFS status unavailable")
    return {"status": status, "source_recomputed": True, "implementation_binding": binding, "completion_sha256": y2.sha_file(completion_path), "completion_canonical_digest": completion["canonical_digest"], "inventory_sha256": y2.sha_file(root / "whole_root_inventory.json"), "inventory_canonical_digest": inventory["canonical_digest"], "inventory_files_digest": inventory["payload"]["files_digest"], "next_authorization": completion["payload"]["next_authorization"], "publication": publication}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("validate","validate-blocked","validate-publication","validate-blocked-publication")); parser.add_argument("--repo-root", type=Path, default=REPO_ROOT); args = parser.parse_args()
    try:
        result = validate(args.repo_root.resolve(), "publication" in args.command, "blocked" in args.command)
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error), "next_authorization": "NONE"}, sort_keys=True)); return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
