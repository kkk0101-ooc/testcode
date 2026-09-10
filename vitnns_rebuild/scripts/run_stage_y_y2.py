#!/usr/bin/env python3
"""Stage Y2 orchestration: metadata seal, worker launch, and reconcile.

`seal-cohort` is Python-standard-library-only and cannot decode images, load a
model, import Torch/timm, or access CUDA. The empirical path lives in the
separate stage_y_y2_worker module and requires a completed pre-execution seal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from vitnns.constants import REPO_ROOT
from vitnns import stage_y_y2 as y2

EXPECTED_REFS = {
    y2.Y0_REF: "a6efa1426902708d83de1d83ef257d524cacf95b",
    y2.Y1_BLOCKED_REF: "d200a5b3de2e46e5d6de2ef07a6c0f5230b1c082",
    y2.Y1_R1_REF: y2.EXPECTED_BASE,
}
SOURCE_FILES = (
    Path("docs/stage_y_y2_clean_cohort_production_20260911.md"),
    Path("config/stage_y/y2_run_spec.json"),
    Path("config/stage_y/y2_run_spec.schema.json"),
    Path("config/stage_y/y2_artifact.schema.json"),
    Path("config/stage_y/y2_unit_terminal.schema.json"),
    Path("config/stage_y/y2_completion.schema.json"),
    Path("config/stage_y/y2_parallel_schedule.json"),
    Path("docs/stage_y_y2_p0_parallel_schedule_amendment_20260911.md"),
    Path("src/vitnns/stage_y_y2.py"),
    Path("src/vitnns/stage_y_y2_worker.py"),
    Path("scripts/run_stage_y_y2.py"),
    Path("scripts/validate_stage_y_y2.py"),
    Path("tests/test_stage_y_y2.py"),
    Path("scripts/validate_stage_y_y2_p0.py"),
    Path("tests/test_stage_y_y2_p0.py"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise y2.StageYY2ProvenanceError(message)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def remote_ref(repo: Path, ref: str) -> str:
    output = y2.git(repo, "ls-remote", "origin", ref).split()
    if len(output) != 2 or output[1] != ref:
        raise y2.StageYY2ProvenanceError(f"remote ref missing or ambiguous: {ref}")
    return output[0]


def _history_binding(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for ref, expected in EXPECTED_REFS.items():
        require(y2.git(repo, "rev-parse", ref) == expected, f"local archive ref drift: {ref}")
        require(remote_ref(repo, ref) == expected, f"remote archive ref drift: {ref}")
    y0_completion = y2.file_record(Path("evidence/stage_y/y0_protocol_20260910/formal/completion/y0.json"))
    y0_inventory = y2.file_record(Path("evidence/stage_y/y0_protocol_20260910/formal/whole_root_inventory.json"))
    y1_completion_path = Path("evidence/stage_y/y1_validation_pilot_20260910/formal/completion/y1.json")
    y1_failure_path = Path("evidence/stage_y/y1_validation_pilot_20260910/formal/failure.json")
    y1_completion = y2.read_json(repo / y1_completion_path)
    y1_failure = y2.read_json(repo / y1_failure_path)
    require(y1_completion["status"] == "BLOCKED_BUDGET", "original Y1 completion status drift")
    require(y1_completion["payload"]["next_authorization"] == "NONE", "original Y1 authorization drift")
    require(y1_failure["payload"]["failure_layer"] == "RESOURCE_TELEMETRY", "original Y1 failure layer drift")
    diff = subprocess.run(["git", "diff", "--exit-code", EXPECTED_REFS[y2.Y1_BLOCKED_REF], "--", "evidence/stage_y/y1_validation_pilot_20260910/formal"], cwd=repo).returncode
    require(diff == 0, "original Y1 formal root differs from blocked archive commit")
    r1_path = Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/completion/y1_r1.json")
    r1 = y2.read_json(repo / r1_path)
    require(r1["status"] == "PASS", "Y1-R1 completion is not PASS")
    require(r1["payload"]["completion_basis"] == "PASS_WITH_EXPLICIT_OWNER_RESOURCE_TELEMETRY_WAIVER", "Y1-R1 basis drift")
    require(r1["payload"]["next_authorization"] == y2.TOKEN, "Y1-R1 authorization drift")
    y0 = {"status": "PASS", "archive_ref": y2.Y0_REF, "commit": EXPECTED_REFS[y2.Y0_REF], "completion": y0_completion, "whole_root_inventory": y0_inventory}
    original = {"status": "BLOCKED_BUDGET", "failure_layer": "RESOURCE_TELEMETRY", "next_authorization": "NONE", "archive_ref": y2.Y1_BLOCKED_REF, "commit": EXPECTED_REFS[y2.Y1_BLOCKED_REF], "completion": y2.file_record(y1_completion_path), "failure": y2.file_record(y1_failure_path), "formal_tree_exact_at_blocked_commit": True}
    r1_binding = {"status": "PASS", "completion_basis": "PASS_WITH_EXPLICIT_OWNER_RESOURCE_TELEMETRY_WAIVER", "archive_ref": y2.Y1_R1_REF, "commit": y2.EXPECTED_BASE, "completion": y2.file_record(r1_path), "conservative_budget_receipt": y2.file_record(Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json")), "waiver_scope": "HISTORICAL_PHASE_LEVEL_TEMETRY_ONLY", "waiver_applies_to_y2": False}
    return y0, original, r1_binding


def _startup(repo: Path) -> dict[str, Any]:
    branch = y2.git(repo, "symbolic-ref", "--short", "HEAD")
    head = y2.git(repo, "rev-parse", "HEAD")
    upstream = y2.git(repo, "rev-parse", "@{upstream}")
    remote = remote_ref(repo, f"refs/heads/{y2.EXPECTED_BRANCH}")
    require(branch == y2.EXPECTED_BRANCH and head == upstream == remote, "Y2 startup branch binding drift")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", y2.EXPECTED_BASE, head], cwd=repo).returncode == 0, "Y1-R1 is not an ancestor of Y2 startup HEAD")
    p0_ref = "refs/archive/stage-y-y2-p0-parallel-schedule-freeze"
    require(y2.git(repo, "rev-parse", p0_ref) == head and remote_ref(repo, p0_ref) == head, "Y2-P0 archive binding drift")
    status = y2.git(repo, "status", "--porcelain")
    allowed = tuple(path.as_posix() for path in SOURCE_FILES)
    for line in status.splitlines():
        path = line[3:]
        require(path in allowed or path.startswith(y2.ROOT.as_posix() + "/") or path.startswith(y2.REPORT_ROOT.as_posix() + "/"), f"unexplained worktree path: {path}")
    lfs = y2.git(repo, "lfs", "status")
    require("(LFS:" not in lfs and "(Git:" not in lfs, "Git LFS is not clean")
    for ref in ("refs/archive/stage-y-y2-clean-cohort-evidence", "refs/archive/stage-y-y2-clean-cohort-evidence-blocked"):
        require(subprocess.run(["git", "show-ref", "--verify", "--quiet", ref], cwd=repo).returncode != 0, f"Y2 archive ref already exists: {ref}")
    return {"status": "PASS", "branch": branch, "base_commit": head, "tracking_commit": upstream, "remote_branch_commit": remote, "p0_archive_commit": head, "initial_worktree_clean_before_y2_edits": True, "initial_index_clean_before_y2_edits": True, "initial_lfs_clean_before_y2_edits": True, "current_authorized_y2_paths": list(allowed), "unexplained_current_paths": [], "archive_ref_absent_before_y2": True}


def _pre_selection(repo: Path) -> dict[str, Any]:
    paths = (y2.RUN_SPEC, y2.PLAN, y2.TAR_PATH, y2.LABEL_PATH, y2.DEVKIT_PATH, *y2.EXCLUSION_PATHS)
    records = [y2.file_record(path) for path in paths]
    spec = y2.read_json(repo / y2.RUN_SPEC)
    require(spec["stage_token"] == y2.TOKEN and spec["selection"]["metadata_only_before_seal"] is True, "Y2 run spec drift")
    require(spec["hard_caps"] == {"h20_gpu_hours":64.0,"critical_path_worker_wall_hours":9.0,"overall_stage_wall_hours":12.0,"cpu_hours":218.0,"peak_reserved_gpu_gib_per_process":21.0,"fleet_concurrent_reserved_gpu_gib":168.0,"retained_storage_gib":6.0,"devices":8,"download_bytes":0}, "Y2 hard-cap drift")
    return {"status": "PASS", "sealed_at": now(), "source_files": records, "source_bundle_digest": y2.canonical_digest(records), "allowed_selection_fields": ["dataset_release_id","class_id","source_object_id","file_sha256"], "forbidden_outcome_fields": ["prediction","correctness","raw_logit","confidence","attribution","scientific_metric","runtime","visual_appearance"], "candidate_image_decode_count": 0, "model_load_count": 0}


def seal_cohort(repo: Path) -> dict[str, Any]:
    root = repo / y2.ROOT
    report = repo / y2.REPORT_ROOT
    require(not root.exists() and not report.exists(), "Y2 formal/report root already exists")
    startup = _startup(repo)
    y0, original_y1, y1_r1 = _history_binding(repo)
    root.joinpath("completion").mkdir(parents=True)
    report.mkdir(parents=True)
    started = now()
    y2.write_json_new(root / "completion/y2.pending.json", y2.signed("stage_y_y2_pending_completion", "PENDING", {"started_at": started, "terminal_published": False, "next_authorization": "NONE_WHILE_PENDING"}))
    y2.write_json_new(root / "startup_gate.json", y2.signed("stage_y_y2_startup_gate", "PASS", startup))
    y2.write_json_new(root / "y0_binding.json", y2.signed("stage_y_y2_y0_binding", "PASS", y0))
    y2.write_json_new(root / "y1_r1_binding.json", y2.signed("stage_y_y2_y1_r1_binding", "PASS", {"original_y1": original_y1, "y1_r1": y1_r1}))
    access_policy = {"status": "PASS", "before_cohort_seal": {"candidate_raw_byte_sha256": "ALLOWED", "image_decode": "FORBIDDEN", "model_or_checkpoint_load": "FORBIDDEN", "outcome_access": "FORBIDDEN"}, "after_cohort_seal": {"decoded_source_objects": "exactly cohort_registry records", "conditions": ["clean"], "devices": 8, "one_process_per_physical_gpu": True, "worker_internal_device": "cuda:0"}, "display_extensions_forbidden": [".png",".svg",".pdf",".jpg",".jpeg",".JPEG"], "later_stages_forbidden": ["Y3","Y4","Y5","F3-Final","Stage G"]}
    y2.write_json_new(root / "access_policy.json", y2.signed("stage_y_y2_access_policy", "PASS", access_policy))
    pre = _pre_selection(repo)
    y2.write_json_new(root / "pre_selection_plan_seal.json", y2.signed("stage_y_y2_pre_selection_plan_seal", "PASS", pre))
    exclusion, exclusions = y2.exclusion_union(repo)
    y2.write_json_new(root / "exclusion_union.json", y2.signed("stage_y_y2_exclusion_union", "PASS", exclusion))
    candidates, metadata = y2.enumerate_candidates(repo)
    selected, proof = y2.select_cohort(candidates, exclusions)
    proof["candidate_metadata"] = metadata
    proof["exclusion_union_digest"] = exclusion["union_digest"]
    cohort = {"status": "PASS", "sealed_at": now(), "dataset_release_id": y2.RELEASE_ID, "namespace": y2.NAMESPACE, "count": 100, "distinct_true_class_count": 100, "selection_digest": proof["selection_digest"], "records": selected, "outcome_access_before_seal": False}
    audit = y2.subset_registry(selected, y2.AUDIT_NAMESPACE, 10, "audit_10")
    perturbation = y2.subset_registry(selected, y2.PERTURBATION_NAMESPACE, 20, "perturbation_20")
    y2.write_json_new(root / "cohort_registry.json", y2.signed("stage_y_y2_cohort_registry", "PASS", cohort))
    y2.write_json_new(root / "cohort_selection_proof.json", y2.signed("stage_y_y2_cohort_selection_proof", "PASS", proof))
    y2.write_json_new(root / "audit_subset_registry.json", y2.signed("stage_y_y2_audit_subset_registry", "PASS", audit))
    y2.write_json_new(root / "perturbation_subset_registry.json", y2.signed("stage_y_y2_perturbation_subset_registry", "PASS", perturbation))
    seals = [y2.file_record(path) for path in (y2.RUN_SPEC, y2.PLAN, y2.ROOT/"pre_selection_plan_seal.json", y2.ROOT/"exclusion_union.json", y2.ROOT/"cohort_registry.json", y2.ROOT/"cohort_selection_proof.json", y2.ROOT/"audit_subset_registry.json", y2.ROOT/"perturbation_subset_registry.json")]
    pre_execution = {"status": "PASS", "sealed_at": now(), "sealed_before_any_image_decode_or_model_load": True, "candidate_image_decode_count": 0, "model_or_checkpoint_load_count": 0, "scientific_outcome_access_count": 0, "sealed_files": seals, "seal_bundle_digest": y2.canonical_digest(seals), "authorized_next_action_within_y2": "BUILD_EIGHT_DETERMINISTIC_SHARD_MANIFESTS"}
    y2.write_json_new(root / "pre_execution_plan_seal.json", y2.signed("stage_y_y2_pre_execution_plan_seal", "PASS", pre_execution))
    access = {"status": "PASS", "phase": "COHORT_SEALED_PRE_EXECUTION", "candidate_raw_byte_hash_count": 50000, "candidate_raw_bytes_hashed": metadata["raw_bytes_hashed"], "image_decode_count": 0, "decoded_source_object_ids": [], "model_load_count": 0, "checkpoint_load_count": 0, "gpu_access_count": 0, "download_bytes": 0, "empirical_conditions": [], "later_stages_started": []}
    y2.write_json_new(root / "access_ledger.json", y2.signed("stage_y_y2_access_ledger", "PASS", access))
    for relative in ("pre_selection_plan_seal.json", "exclusion_union.json", "cohort_registry.json", "cohort_selection_proof.json", "audit_subset_registry.json", "perturbation_subset_registry.json", "pre_execution_plan_seal.json"):
        (root / relative).chmod(0o444)
    descriptor = os.open(root, os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)
    return {"status": "PASS", "phase": "COHORT_SEALED_PRE_EXECUTION", "cohort_count": 100, "selection_digest": proof["selection_digest"], "audit_subset_digest": audit["subset_digest"], "perturbation_subset_digest": perturbation["subset_digest"], "next_action": "build-shards"}


def build_shards(repo: Path) -> dict[str, Any]:
    formal = repo / y2.ROOT
    cohort = y2.read_json(formal / "cohort_registry.json")
    require(not (formal / "multi_gpu_startup_gate.json").exists(), "startup gate already exists")
    manifests = y2.build_shard_manifests(cohort, formal)
    return {"status": "PASS", "shard_count": len(manifests), "shard_unit_counts": [row["payload"]["unit_count"] for row in manifests], "next_action": "validate-startup"}


def _device_probe_script() -> str:
    return """import hashlib,json,os,platform\n+import torch,torchvision,timm\n+torch.cuda.set_device(0)\n+torch.use_deterministic_algorithms(True)\n+x=torch.arange(4096,dtype=torch.float64,device='cuda:0')\n+y=(x.sin().square().sum()).cpu().item()\n+p=torch.cuda.get_device_properties(0)\n+print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__,'cuda':torch.version.cuda,'device_name':torch.cuda.get_device_name(0),'total_memory_bytes':p.total_memory,'numeric_probe':format(y,'.17g')}))"""


def validate_startup(repo: Path) -> dict[str, Any]:
    """Future formal gate. This is deliberately never called by P0 tests."""
    formal = repo / y2.ROOT
    require((formal / "cohort_registry.json").is_file(), "cohort is not sealed")
    require(all((formal / "shards" / f"shard-{i:02d}/shard_manifest.json").is_file() for i in y2.SHARD_IDS), "all shard manifests must exist")
    require(not list((formal / "shards").glob("shard-*/units/unit-*/completion/unit.json")), "Y2 unit outcome already exists")
    ps = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True)
    stale = []
    for line in ps.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and int(parts[0]) != os.getpid() and re.search(r"stage_y_y2_worker|run_stage_y_y2.py\s+run-shard", parts[1]): stale.append({"pid": int(parts[0]), "command": parts[1]})
    require(not stale, "stale Y2 worker/PID exists")
    process = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits"], cwd=repo, text=True, capture_output=True)
    if process.returncode != 0: raise y2.StageYY2ScheduleError("nvidia-smi unavailable: " + process.stderr.strip())
    rows = []
    for line in process.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        require(len(values) == 6, "malformed nvidia-smi row")
        rows.append({"physical_index": int(values[0]), "device_name": values[1], "gpu_uuid": values[2], "pci_bus_id": values[3], "total_memory_mib": int(values[4]), "driver": values[5]})
    rows.sort(key=lambda row: row["physical_index"])
    if len(rows) != 8 or any(row["device_name"] != "NVIDIA H20" for row in rows): raise y2.StageYY2ScheduleError("exactly eight NVIDIA H20 devices required")
    require([row["physical_index"] for row in rows] == list(range(8)), "physical GPU indices must be exactly 0..7")
    require(len({row["gpu_uuid"] for row in rows}) == len({row["pci_bus_id"] for row in rows}) == 8, "GPU UUID/PCI identities not unique")
    require(all(row["total_memory_mib"] >= 21 * 1024 for row in rows), "a device cannot satisfy the 21 GiB cap")
    checkpoint_sha = y2.sha_file(repo / Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"))
    require(checkpoint_sha == "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d", "checkpoint identity drift")
    probes = []
    for row in rows:
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(row["physical_index"])
        probe = subprocess.run([sys.executable, "-c", _device_probe_script()], cwd=repo, env=env, text=True, capture_output=True)
        if probe.returncode != 0: raise y2.StageYY2ScheduleError(f"device probe failed for {row['physical_index']}: {probe.stderr.strip()}")
        value = json.loads(probe.stdout); require(value.pop("device_name") == "NVIDIA H20", "probe device-name drift")
        require(int(value.pop("total_memory_bytes")) >= 21 * 1024 ** 3, "probe memory below 21 GiB")
        numeric = value.pop("numeric_probe"); identity = {**value, "driver": row["driver"], "checkpoint_sha256": checkpoint_sha, "numeric_contract_digest": "stage-y-y1-grouped-jvp-vjp-frozen-v1"}
        row["environment_digest"] = y2.environment_digest(identity); row["numeric_probe"] = numeric; probes.append(identity)
    require(len({y2.canonical_digest(item) for item in probes}) == 1 and len({row["numeric_probe"] for row in rows}) == 1, "cross-device environment or numeric parity failure")
    gate = y2.signed("stage_y_y2_multi_gpu_startup_gate", "PASS", {"required_device_count": 8, "devices": rows, "checkpoint_sha256": checkpoint_sha, "model_identity": "vit_base_patch16_224.augreg2_in21k_ft_in1k", "numeric_contract_digest": "stage-y-y1-grouped-jvp-vjp-frozen-v1", "cross_device_numeric_parity": True, "isolated_shard_roots": True, "shared_mutable_output": False, "stale_worker_pids": stale, "formal_unit_outcome_count_before_launch": 0, "started_unix_ns": __import__("time").time_ns()})
    y2.write_json_new(formal / "multi_gpu_startup_gate.json", gate)
    return {"status": "PASS", "device_count": 8, "next_action": "run-shard --shard-id N"}


def run_shard(repo: Path, shard_id: int) -> dict[str, Any]:
    gate = y2.read_json(repo / y2.ROOT / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
    require(shard_id in y2.SHARD_IDS, "invalid shard id")
    physical = gate["payload"]["devices"][shard_id]["physical_index"]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(physical)
    env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = subprocess.run([sys.executable, "-m", "vitnns.stage_y_y2_worker", "--repo-root", str(repo), "--shard-id", str(shard_id)], cwd=repo, env=env, text=True, capture_output=True)
    require(process.returncode == 0, process.stdout.strip() or process.stderr.strip() or "shard worker failed")
    return json.loads(process.stdout)


def reconcile(repo: Path) -> dict[str, Any]:
    result = y2.reconcile_shards(repo / y2.ROOT)
    y2.write_json_new(repo / y2.ROOT / "reconciliation.json", result)
    return {"status": result["status"], **result["payload"]}


def seal(repo: Path) -> dict[str, Any]:
    reconciliation = y2.read_json(repo / y2.ROOT / "reconciliation.json"); y2.validate_signed(reconciliation)
    pending = repo / y2.ROOT / "completion/y2.pending.json"
    status = reconciliation["status"]
    require(status in ("PASS", "BLOCKED_BUDGET", "BLOCKED_INFRASTRUCTURE", "BLOCKED_SCIENTIFIC_VALIDITY"), "unsupported reconciliation terminal")
    next_authorization = "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY" if status == "PASS" else "NONE"
    completion = y2.signed("stage_y_y2_terminal_completion", status, {"terminal": True, "reconciliation": y2.file_record(y2.ROOT / "reconciliation.json"), "valid_clean_count": reconciliation["payload"]["valid_count"], "formal_outcome_access": True, "conditions": ["clean"], "bootstrap_executed": False, "holm_executed": False, "claim_decisions": [], "next_authorization": next_authorization})
    y2.publish_terminal(pending, repo / y2.ROOT / "completion/y2.json", completion)
    return {"status": status, "next_authorization": next_authorization}


def validate(repo: Path, publication: bool = False) -> dict[str, Any]:
    completion = y2.read_json(repo / y2.ROOT / "completion/y2.json"); y2.validate_signed(completion)
    result = y2.reconcile_shards(repo / y2.ROOT)
    require(completion["status"] == result["status"] == "PASS", "Y2 terminal validation failed")
    output = {"status": "PASS", "source_recomputed": True, "valid_clean_count": result["payload"]["valid_count"], "next_authorization": completion["payload"]["next_authorization"]}
    if publication:
        head = y2.git(repo, "rev-parse", "HEAD"); tracking = y2.git(repo, "rev-parse", "@{upstream}")
        require(remote_ref(repo, f"refs/heads/{y2.EXPECTED_BRANCH}") == head == tracking, "published branch drift")
        require(remote_ref(repo, "refs/archive/stage-y-y2-clean-cohort-evidence") == head == y2.git(repo, "rev-parse", "refs/archive/stage-y-y2-clean-cohort-evidence"), "published Y2 archive drift")
        require(not y2.git(repo, "status", "--porcelain"), "publication worktree dirty")
        output["publication"] = True
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal-cohort", "build-shards", "validate-startup", "run-shard", "reconcile", "seal", "validate", "validate-publication"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--shard-id", type=int)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    try:
        if args.command == "seal-cohort": result = seal_cohort(repo)
        elif args.command == "build-shards": result = build_shards(repo)
        elif args.command == "validate-startup": result = validate_startup(repo)
        elif args.command == "run-shard":
            require(args.shard_id is not None, "run-shard requires --shard-id"); result = run_shard(repo, args.shard_id)
        elif args.command == "reconcile": result = reconcile(repo)
        elif args.command == "seal": result = seal(repo)
        else: result = validate(repo, args.command == "validate-publication")
    except Exception as error:
        print(json.dumps({"status": getattr(error, "code", "BLOCKED"), "error": str(error), "next_authorization": "NONE"}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
