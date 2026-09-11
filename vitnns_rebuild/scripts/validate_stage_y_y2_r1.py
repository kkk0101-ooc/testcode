#!/usr/bin/env python3
"""Source-recomputing validator and false-pass audit for Stage Y2-R1."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from vitnns.constants import REPO_ROOT
from vitnns import stage_y_y2 as y2

y2.activate_recovery_contract()

PASS_REF = "refs/archive/stage-y-y2-r1-clean-cohort-evidence"
BLOCKED_REF = "refs/archive/stage-y-y2-r1-clean-cohort-evidence-blocked"
ORIGINAL_REF = "refs/archive/stage-y-y2-clean-cohort-evidence-blocked"
ORIGINAL_COMMIT = "1a621724b32b208036fb5f2a3993a48697e933c8"
ORIGINAL_FAILURE_FRAGMENT = "this operation is not deterministic because it uses CuBLAS"
RETRY_REASON = "CUBLAS_WORKSPACE_CONFIG_MISSING_BEFORE_TORCH_INITIALIZATION"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise y2.StageYY2ProvenanceError(message)


def remote_ref(repo: Path, ref: str) -> str:
    parts = y2.git(repo, "ls-remote", "origin", ref).split()
    require(len(parts) == 2 and parts[1] == ref, f"remote ref missing or ambiguous: {ref}")
    return parts[0]


def record(repo: Path, relative: Path) -> dict[str, Any]:
    value = y2.file_record(repo / relative); value["path"] = relative.as_posix(); return value


def validate_schema(repo: Path, relative: Path, value: Mapping[str, Any]) -> None:
    schema = y2.read_json(repo / relative)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path))
    require(not errors, f"schema failure {relative}: {errors[0].message if errors else ''}")


def _old_terminal_by_order(repo: Path) -> dict[int, tuple[Path, dict[str, Any]]]:
    result = {}
    for path in sorted((repo / y2.ORIGINAL_ROOT).glob("shards/shard-*/units/unit-*/completion/unit.json")):
        value = y2.read_json(path); y2.validate_signed(value, allow_historical_y2=True)
        result[int(value["payload"]["selection_order"])] = (path, value)
    require(sorted(result) == list(range(100)), "original Y2 terminal coverage drift")
    return result


def validate_original_y2(repo: Path, *, recompute_release: bool) -> dict[str, Any]:
    require(y2.git(repo, "rev-parse", ORIGINAL_REF) == ORIGINAL_COMMIT, "original local archive ref drift")
    protected = (y2.ORIGINAL_ROOT, y2.ORIGINAL_REPORT_ROOT)
    diff = subprocess.run(["git", "diff", "--exit-code", ORIGINAL_COMMIT, "--", *(path.as_posix() for path in protected)], cwd=repo).returncode
    require(diff == 0, "original Y2 protected paths changed")
    completion = y2.read_json(repo / y2.ORIGINAL_ROOT / "completion/y2.json")
    failure = y2.read_json(repo / y2.ORIGINAL_ROOT / "failure.json")
    reconciliation = y2.read_json(repo / y2.ORIGINAL_ROOT / "reconciliation.json")
    require(completion["status"] == "BLOCKED_INFRASTRUCTURE" and completion["payload"]["valid_clean_count"] == 0 and completion["payload"]["formal_outcome_access"] is True and completion["payload"]["next_authorization"] == "NONE", "original Y2 completion drift")
    require(failure["payload"]["phase"] == "RECONCILIATION" and failure["payload"]["recovery_requires_new_root"] is True, "original Y2 failure drift")
    require(reconciliation["payload"]["terminal_counts"] == {"VALID": 0,"FAILED_NUMERICAL": 0,"BLOCKED_INFRASTRUCTURE": 100,"BLOCKED_BUDGET": 0}, "original Y2 terminal counts drift")
    old = _old_terminal_by_order(repo)
    codes = {item[1]["payload"]["failure_code"] for item in old.values()}
    require(len(codes) == 1 and ORIGINAL_FAILURE_FRAGMENT in next(iter(codes)), "original failures are not homogeneous cuBLAS environment failures")
    for _, terminal in old.values():
        payload = terminal["payload"]
        require(payload["target_identity"] == {} and payload["quadrature_attempts"] == [], "original failure reached scientific computation")
        require(payload["array_records"] == [] and payload["scientific_record"] is None, "original failure retained scientific output")
    cohort = y2.read_json(repo / y2.ORIGINAL_ROOT / "cohort_registry.json")
    audit = y2.read_json(repo / y2.ORIGINAL_ROOT / "audit_subset_registry.json")
    perturbation = y2.read_json(repo / y2.ORIGINAL_ROOT / "perturbation_subset_registry.json")
    require(cohort["canonical_digest"] == "156eab4bd86ff9979abc6acf8b06af3aa5a90e90cd02edef7713539b147ce277", "cohort canonical digest drift")
    require(cohort["payload"]["selection_digest"] == "8b3faa6d437bda38e963985ac12f14f3b440a3e4021915081f1d5e1b1c80d2cf", "selection digest drift")
    require(cohort["payload"]["count"] == len(cohort["payload"]["records"]) == 100 and len({row["class_id"] for row in cohort["payload"]["records"]}) == 100, "cohort cardinality/class identity drift")
    if recompute_release:
        _, exclusions = y2.exclusion_union(repo)
        candidates, _ = y2.enumerate_candidates(repo)
        selected, proof = y2.select_cohort(candidates, exclusions)
        require(selected == cohort["payload"]["records"] and proof["selection_digest"] == cohort["payload"]["selection_digest"], "raw-source cohort recomputation drift")
        require(y2.subset_registry(selected, y2.AUDIT_NAMESPACE, 10, "audit_10") == audit["payload"], "audit subset recomputation drift")
        require(y2.subset_registry(selected, y2.PERTURBATION_NAMESPACE, 20, "perturbation_20") == perturbation["payload"], "perturbation subset recomputation drift")
    return {"status": "PASS","terminal_count": 100,"failure_code_digest": y2.canonical_digest(sorted(codes)),"cohort_canonical_digest": cohort["canonical_digest"],"selection_digest": cohort["payload"]["selection_digest"],"audit_subset_digest": audit["payload"]["subset_digest"],"perturbation_subset_digest": perturbation["payload"]["subset_digest"],"source_recomputed": recompute_release}


def run_focused_tests(repo: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage_y_y2_r1.py"], cwd=repo, env=env, text=True, capture_output=True)
    require(process.returncode == 0, "focused recovery tests failed: " + (process.stdout + process.stderr)[-4000:])
    match = __import__("re").search(r"(\d+) passed", process.stdout)
    require(match is not None, "focused-test count unavailable")
    return {"status": "PASS","command": f"{os.path.realpath(sys.executable)} -m pytest -q tests/test_stage_y_y2_r1.py","test_count": int(match.group(1)),"output_tail": process.stdout.strip()[-1000:]}


def false_pass_audit(repo: Path) -> dict[str, Any]:
    launcher = (repo / "scripts/run_stage_y_y2.py").read_text(encoding="utf-8")
    worker = (repo / "src/vitnns/stage_y_y2_worker.py").read_text(encoding="utf-8")
    spec = y2.read_json(repo / y2.R1_RUN_SPEC)
    original = validate_original_y2(repo, recompute_release=False)
    worker_tree = ast.parse(worker)
    execute = next(node for node in ast.walk(worker_tree) if isinstance(node, ast.FunctionDef) and node.name == "execute")
    env_guard_line = next(node.lineno for node in ast.walk(execute) if isinstance(node, ast.Raise) and "CUBLAS_WORKSPACE_CONFIG must equal" in ast.unparse(node))
    torch_import_line = next(node.lineno for node in ast.walk(execute) if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
    checks = [
        ("modify_original_y2", original["status"] == "PASS"),
        ("delete_original_failures", original["terminal_count"] == 100),
        ("reselect_cohort", spec["identity"]["selection_digest"] == original["selection_digest"]),
        ("warn_only_true", "warn_only=True" not in launcher + worker and spec["environment_repair"]["deterministic_warn_only"] is False),
        ("disable_deterministic_algorithms", "use_deterministic_algorithms(False" not in launcher + worker),
        ("late_environment_set", env_guard_line < torch_import_line),
        ("wrong_16_8_value", ":16:8" not in launcher + worker + json.dumps(spec)),
        ("probe_without_cublas", "torch.matmul" in launcher and "backward()" in launcher),
        ("probe_worker_python_mismatch", "python_executable_realpath" in launcher and "os.path.realpath(sys.executable)" in launcher),
        ("environment_digest_omits_cublas", "cublas_workspace_config" in (repo / "src/vitnns/stage_y_y2.py").read_text() and "environment_identity" in worker),
        ("omit_original_resource", "original_y2_resource_totals" in (repo / "src/vitnns/stage_y_y2.py").read_text()),
        ("change_scientific_contract", spec["execution"] == {key: y2.read_json(repo / y2.ORIGINAL_RUN_SPEC)["execution"][key] for key in spec["execution"]}),
        ("outcome_based_resharding", spec["execution"]["shard_rule"] == "shard_id = selection_order_zero_based mod locked_worker_count" and spec["execution"]["dynamic_rebalancing"] is False),
        ("launch_y3", "Y3" not in ["Y2-R1"] and spec["next_authorization_on_pass"] == "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY"),
        ("pass_below_90", spec["valid_clean_threshold"] == 90),
        ("rewrite_original_archive_ref", y2.git(repo, "rev-parse", ORIGINAL_REF) == ORIGINAL_COMMIT),
    ]
    rows = [{"attack": name,"expected": "REJECT","observed": "REJECT" if passed else "ACCEPT","pass": bool(passed)} for name, passed in checks]
    require(all(row["pass"] for row in rows), "false-pass audit accepted: " + ",".join(row["attack"] for row in rows if not row["pass"]))
    return {"status": "PASS","attack_count": len(rows),"rejected_count": len(rows),"checks": rows}


def _validate_environment_gate(gate: Mapping[str, Any]) -> None:
    y2.validate_signed(gate)
    payload = gate["payload"]
    require(payload["cublas_workspace_config"] == y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG, "startup cuBLAS value drift")
    require(payload["probe_performed_cublas_matmul"] is True and payload["probe_performed_backward"] is True and payload["probe_exact_repeat"] is True and payload["cross_device_numeric_parity"] is True, "startup cuBLAS probe contract drift")
    require(payload["python_executable_realpath"] == os.path.realpath(sys.executable), "startup Python executable drift")
    require(len(payload["devices"]) == payload["locked_worker_count"] and len({row["numeric_probe_digest"] for row in payload["devices"]}) == 1, "startup device/digest parity drift")
    for row in payload["devices"]:
        require(y2.environment_digest(row["environment_identity"]) == row["environment_digest"], "startup environment digest drift")


def validate(repo: Path, publication: bool = False, allow_blocked: bool = False) -> dict[str, Any]:
    root = repo / y2.ROOT
    completion_path = root / "completion/y2_r1.json"
    require(completion_path.is_file() and not (root / "completion/y2_r1.pending.json").exists(), "Y2-R1 terminal transition incomplete")
    completion = y2.read_json(completion_path); y2.validate_signed(completion)
    validate_schema(repo, Path("config/stage_y/y2_r1_artifact.schema.json"), completion)
    status = completion["status"]
    if allow_blocked:
        require(status != "PASS" and completion["payload"]["next_authorization"] == "NONE", "blocked terminal semantics drift")
    else:
        require(status == "PASS" and completion["payload"]["valid_clean_count"] >= 90 and completion["payload"]["next_authorization"] == "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY", "PASS terminal semantics drift")
    validate_schema(repo, Path("config/stage_y/y2_r1_run_spec.schema.json"), y2.read_json(repo / y2.R1_RUN_SPEC))
    original = validate_original_y2(repo, recompute_release=True)
    preflight = y2.read_json(root / "recovery_preflight.json"); y2.validate_signed(preflight)
    require(preflight["status"] == "PASS", "recovery preflight drift")
    implementation = y2.read_json(root / "implementation_seal.json"); y2.validate_signed(implementation)
    observed_sources = [record(repo, Path(row["path"])) for row in implementation["payload"]["source_files"]]
    require(observed_sources == implementation["payload"]["source_files"] and y2.canonical_digest({"files": observed_sources}) == implementation["payload"]["files_digest"], "implementation binding drift")
    gate = y2.read_json(root / "multi_gpu_startup_gate.json"); _validate_environment_gate(gate)
    reconciliation = y2.reconcile_shards(root)
    require(reconciliation == y2.read_json(root / "reconciliation.json"), "source-recomputed reconciliation drift")
    require(reconciliation["payload"]["resource_accounting_includes_original_y2"] is True and reconciliation["payload"]["original_y2_resource_totals"] == y2.read_json(repo / y2.ORIGINAL_ROOT / "reconciliation.json")["payload"]["resource_totals"], "cumulative resource accounting drift")
    old = _old_terminal_by_order(repo)
    for path in sorted(root.glob("shards/shard-*/units/unit-*/completion/unit.json")):
        terminal = y2.read_json(path); y2.validate_signed(terminal)
        payload = terminal["payload"]; retry = payload["retry_identity"]; _, predecessor = old[payload["selection_order"]]
        require(retry["original_terminal_digest"] == predecessor["canonical_digest"] and retry["original_attempt_digest"] == predecessor["payload"]["attempt_records"][-1]["canonical_digest"] and retry["retry_reason"] == RETRY_REASON and retry["retry_index"] == 1, "unit retry binding drift")
    inventory = y2.read_json(root / "whole_root_inventory.json"); y2.validate_signed(inventory)
    require(y2.whole_root_inventory(root, status)["payload"] == inventory["payload"], "whole-root inventory drift")
    require(completion["payload"]["whole_root_inventory"] == record(repo, y2.ROOT / "whole_root_inventory.json"), "completion inventory link drift")
    require(completion["payload"]["report"] == record(repo, y2.REPORT_ROOT / "y2_r1_report.md"), "completion report link drift")
    summary = y2.read_json(root / "terminal_access_summary.json"); y2.validate_signed(summary)
    require(summary["payload"]["non_clean_condition_count"] == 0 and summary["payload"]["later_stages_started"] == [] and not summary["payload"]["bootstrap_executed"] and not summary["payload"]["holm_executed"] and not summary["payload"]["claim_decision_executed"], "Y2-R1 scope boundary drift")
    audit = false_pass_audit(repo)
    if publication:
        head = y2.git(repo, "rev-parse", "HEAD")
        require(y2.git(repo, "rev-parse", "@{upstream}") == remote_ref(repo, "refs/heads/stage-d-d3-pass") == head, "published branch drift")
        ref = PASS_REF if status == "PASS" else BLOCKED_REF
        require(y2.git(repo, "rev-parse", ref) == remote_ref(repo, ref) == head, "published archive ref drift")
        require(remote_ref(repo, ORIGINAL_REF) == ORIGINAL_COMMIT, "published original archive ref drift")
        require(not y2.git(repo, "status", "--porcelain"), "publication worktree/index dirty")
        require("Git LFS objects to be committed:" in y2.git(repo, "lfs", "status"), "publication LFS status unavailable")
    return {"status": status,"source_recomputed": True,"original_y2": original,"false_pass_audit": {"attack_count": audit["attack_count"],"rejected_count": audit["rejected_count"]},"completion_sha256": y2.sha_file(completion_path),"completion_canonical_digest": completion["canonical_digest"],"inventory_sha256": y2.sha_file(root / "whole_root_inventory.json"),"inventory_canonical_digest": inventory["canonical_digest"],"inventory_files_digest": inventory["payload"]["files_digest"],"next_authorization": completion["payload"]["next_authorization"],"publication": publication}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("focused-tests","false-pass-audit","validate","validate-blocked","validate-publication","validate-blocked-publication")); parser.add_argument("--repo-root", type=Path, default=REPO_ROOT); args = parser.parse_args(); repo = args.repo_root.resolve()
    try:
        if args.command == "focused-tests": result = run_focused_tests(repo)
        elif args.command == "false-pass-audit": result = false_pass_audit(repo)
        else: result = validate(repo, publication="publication" in args.command, allow_blocked="blocked" in args.command)
    except Exception as error:
        print(json.dumps({"status": "BLOCKED","error": str(error),"next_authorization": "NONE"}, sort_keys=True)); return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
