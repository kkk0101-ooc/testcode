#!/usr/bin/env python3
"""Stage Y2 orchestration: metadata seal, worker launch, and reconcile.

`seal-cohort` is Python-standard-library-only and cannot decode images, load a
model, import Torch/timm, or access CUDA. The empirical path lives in the
separate stage_y_y2_worker module and requires a completed pre-execution seal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
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
    Path("config/stage_y/y2_startup_gate.schema.json"),
    Path("config/stage_y/y2_shard_manifest.schema.json"),
    Path("config/stage_y/y2_parallel_schedule.json"),
    Path("docs/stage_y_y2_p0_parallel_schedule_amendment_20260911.md"),
    Path("docs/stage_y_y2_p0_r1_elastic_idle_h20_schedule_20260911.md"),
    Path("src/vitnns/stage_y_y2.py"),
    Path("src/vitnns/stage_y_y2_worker.py"),
    Path("scripts/run_stage_y_y2.py"),
    Path("scripts/validate_stage_y_y2.py"),
    Path("tests/test_stage_y_y2.py"),
    Path("scripts/validate_stage_y_y2_p0.py"),
    Path("tests/test_stage_y_y2_p0.py"),
    Path("scripts/validate_stage_y_y2_p0_r1.py"),
    Path("tests/test_stage_y_y2_p0_r1.py"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise y2.StageYY2ProvenanceError(message)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def repo_file_record(repo: Path, relative: Path) -> dict[str, Any]:
    result = y2.file_record(repo / relative)
    result["path"] = relative.as_posix()
    return result


def remote_ref(repo: Path, ref: str) -> str:
    output = y2.git(repo, "ls-remote", "origin", ref).split()
    if len(output) != 2 or output[1] != ref:
        raise y2.StageYY2ProvenanceError(f"remote ref missing or ambiguous: {ref}")
    return output[0]


def _history_binding(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for ref, expected in EXPECTED_REFS.items():
        require(y2.git(repo, "rev-parse", ref) == expected, f"local archive ref drift: {ref}")
        require(remote_ref(repo, ref) == expected, f"remote archive ref drift: {ref}")
    y0_completion = repo_file_record(repo, Path("evidence/stage_y/y0_protocol_20260910/formal/completion/y0.json"))
    y0_inventory = repo_file_record(repo, Path("evidence/stage_y/y0_protocol_20260910/formal/whole_root_inventory.json"))
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
    original = {"status": "BLOCKED_BUDGET", "failure_layer": "RESOURCE_TELEMETRY", "next_authorization": "NONE", "archive_ref": y2.Y1_BLOCKED_REF, "commit": EXPECTED_REFS[y2.Y1_BLOCKED_REF], "completion": repo_file_record(repo, y1_completion_path), "failure": repo_file_record(repo, y1_failure_path), "formal_tree_exact_at_blocked_commit": True}
    r1_binding = {"status": "PASS", "completion_basis": "PASS_WITH_EXPLICIT_OWNER_RESOURCE_TELEMETRY_WAIVER", "archive_ref": y2.Y1_R1_REF, "commit": y2.EXPECTED_BASE, "completion": repo_file_record(repo, r1_path), "conservative_budget_receipt": repo_file_record(repo, Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json")), "waiver_scope": "HISTORICAL_PHASE_LEVEL_TEMETRY_ONLY", "waiver_applies_to_y2": False}
    return y0, original, r1_binding


def _startup(repo: Path) -> dict[str, Any]:
    branch = y2.git(repo, "symbolic-ref", "--short", "HEAD")
    head = y2.git(repo, "rev-parse", "HEAD")
    upstream = y2.git(repo, "rev-parse", "@{upstream}")
    remote = remote_ref(repo, f"refs/heads/{y2.EXPECTED_BRANCH}")
    require(branch == y2.EXPECTED_BRANCH and head == upstream == remote, "Y2 startup branch binding drift")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", y2.EXPECTED_BASE, head], cwd=repo).returncode == 0, "Y1-R1 is not an ancestor of Y2 startup HEAD")
    p0_ref = "refs/archive/stage-y-y2-p0-parallel-schedule-freeze"
    p0_commit = "66a0dfb889fdb0d45d9e566ed422ba9a40bd30a1"
    require(y2.git(repo, "rev-parse", p0_ref) == remote_ref(repo, p0_ref) == p0_commit, "Y2-P0 archive binding drift")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", p0_commit, head], cwd=repo).returncode == 0, "Y2-P0 is not an ancestor of Y2 startup HEAD")
    r1_ref = "refs/archive/stage-y-y2-p0-r1-elastic-schedule-freeze"
    require(y2.git(repo, "rev-parse", r1_ref) == head and remote_ref(repo, r1_ref) == head, "Y2-P0-R1 archive binding drift")
    status = y2.git(repo, "status", "--porcelain")
    allowed = tuple(path.as_posix() for path in SOURCE_FILES)
    require(not status, "Y2 startup worktree/index must be clean")
    lfs = y2.git(repo, "lfs", "status")
    require("(LFS:" not in lfs and "(Git:" not in lfs, "Git LFS is not clean")
    for ref in ("refs/archive/stage-y-y2-clean-cohort-evidence", "refs/archive/stage-y-y2-clean-cohort-evidence-blocked"):
        require(subprocess.run(["git", "show-ref", "--verify", "--quiet", ref], cwd=repo).returncode != 0, f"Y2 archive ref already exists: {ref}")
    return {"status": "PASS", "branch": branch, "base_commit": head, "tracking_commit": upstream, "remote_branch_commit": remote, "p0_archive_commit": p0_commit, "p0_r1_archive_commit": head, "initial_worktree_clean_before_y2_edits": True, "initial_index_clean_before_y2_edits": True, "initial_lfs_clean_before_y2_edits": True, "current_authorized_y2_paths": list(allowed), "unexplained_current_paths": [], "archive_ref_absent_before_y2": True}


def _pre_selection(repo: Path) -> dict[str, Any]:
    paths = (y2.RUN_SPEC, y2.PLAN, y2.TAR_PATH, y2.LABEL_PATH, y2.DEVKIT_PATH, *y2.EXCLUSION_PATHS)
    records = [repo_file_record(repo, path) for path in paths]
    spec = y2.read_json(repo / y2.RUN_SPEC)
    require(spec["stage_token"] == y2.TOKEN and spec["selection"]["metadata_only_before_seal"] is True, "Y2 run spec drift")
    require(spec["hard_caps"]["h20_gpu_hours"] == 64.0 and spec["hard_caps"]["cpu_hours"] == 218.0, "Y2 aggregate hard-cap drift")
    require(spec["execution"]["minimum_device_count"] == 1 and spec["execution"]["maximum_device_count"] == 8, "Y2 elastic device bounds drift")
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
    access_policy = {"status": "PASS", "before_cohort_seal": {"candidate_raw_byte_sha256": "ALLOWED", "image_decode": "FORBIDDEN", "model_or_checkpoint_load": "FORBIDDEN", "outcome_access": "FORBIDDEN"}, "after_cohort_seal": {"decoded_source_objects": "exactly cohort_registry records", "conditions": ["clean"], "device_policy": "lock 1..8 eligible idle NVIDIA H20 devices once at startup", "one_process_per_physical_gpu": True, "worker_internal_device": "cuda:0"}, "display_extensions_forbidden": [".png",".svg",".pdf",".jpg",".jpeg",".JPEG"], "later_stages_forbidden": ["Y3","Y4","Y5","F3-Final","Stage G"]}
    y2.write_json_new(root / "access_policy.json", y2.signed("stage_y_y2_access_policy", "PASS", access_policy))
    implementation_files = [repo_file_record(repo, path) for path in SOURCE_FILES]
    y2.write_json_new(root / "implementation_seal.json", y2.signed("stage_y_y2_implementation_seal", "PASS", {"source_files": implementation_files, "files_digest": y2.canonical_digest({"files": implementation_files}), "frozen_before_formal_cohort_access": True}))
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
    seals = [repo_file_record(repo, path) for path in (y2.RUN_SPEC, y2.PLAN, y2.ROOT/"implementation_seal.json", y2.ROOT/"pre_selection_plan_seal.json", y2.ROOT/"exclusion_union.json", y2.ROOT/"cohort_registry.json", y2.ROOT/"cohort_selection_proof.json", y2.ROOT/"audit_subset_registry.json", y2.ROOT/"perturbation_subset_registry.json")]
    pre_execution = {"status": "PASS", "sealed_at": now(), "sealed_before_any_image_decode_or_model_load": True, "candidate_image_decode_count": 0, "model_or_checkpoint_load_count": 0, "scientific_outcome_access_count": 0, "sealed_files": seals, "seal_bundle_digest": y2.canonical_digest(seals), "authorized_next_action_within_y2": "LOCK_ELIGIBLE_IDLE_H20_ROSTER"}
    y2.write_json_new(root / "pre_execution_plan_seal.json", y2.signed("stage_y_y2_pre_execution_plan_seal", "PASS", pre_execution))
    access = {"status": "PASS", "phase": "COHORT_SEALED_PRE_EXECUTION", "candidate_raw_byte_hash_count": 50000, "candidate_raw_bytes_hashed": metadata["raw_bytes_hashed"], "image_decode_count": 0, "decoded_source_object_ids": [], "model_load_count": 0, "checkpoint_load_count": 0, "gpu_access_count": 0, "download_bytes": 0, "empirical_conditions": [], "later_stages_started": []}
    y2.write_json_new(root / "access_ledger.json", y2.signed("stage_y_y2_access_ledger", "PASS", access))
    for relative in ("implementation_seal.json", "pre_selection_plan_seal.json", "exclusion_union.json", "cohort_registry.json", "cohort_selection_proof.json", "audit_subset_registry.json", "perturbation_subset_registry.json", "pre_execution_plan_seal.json"):
        (root / relative).chmod(0o444)
    descriptor = os.open(root, os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)
    return {"status": "PASS", "phase": "COHORT_SEALED_PRE_EXECUTION", "cohort_count": 100, "selection_digest": proof["selection_digest"], "audit_subset_digest": audit["subset_digest"], "perturbation_subset_digest": perturbation["subset_digest"], "next_action": "validate-startup"}


def build_shards(repo: Path) -> dict[str, Any]:
    formal = repo / y2.ROOT
    cohort = y2.read_json(formal / "cohort_registry.json")
    gate = y2.read_json(formal / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
    worker_count = y2.validate_worker_count(gate["payload"]["locked_worker_count"])
    manifests = y2.build_shard_manifests(cohort, formal, worker_count, gate["payload"]["resource_projection"])
    return {"status": "PASS", "shard_count": len(manifests), "shard_unit_counts": [row["payload"]["unit_count"] for row in manifests], "next_action": "run-shard --shard-id N"}


def _device_probe_script() -> str:
    return """import json,platform
import torch,torchvision,timm
torch.cuda.set_device(0)
torch.use_deterministic_algorithms(True)
x=torch.arange(4096,dtype=torch.float64,device='cuda:0')
y=(x.sin().square().sum()).cpu().item()
p=torch.cuda.get_device_properties(0)
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__,'cuda':torch.version.cuda,'device_name':torch.cuda.get_device_name(0),'total_memory_bytes':p.total_memory,'numeric_probe':format(y,'.17g')}))"""


def validate_startup(repo: Path) -> dict[str, Any]:
    """Future formal gate. This is deliberately never called by P0 tests."""
    formal = repo / y2.ROOT
    require((formal / "cohort_registry.json").is_file(), "cohort is not sealed")
    require(not (formal / "multi_gpu_startup_gate.json").exists(), "startup roster is already locked")
    require(not list((formal / "shards").glob("shard-*/units/unit-*/completion/unit.json")), "Y2 unit outcome already exists")
    ps = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True)
    stale = []
    for line in ps.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and int(parts[0]) != os.getpid() and re.search(r"stage_y_y2_worker|run_stage_y_y2.py\s+(run-shard|run-all)", parts[1]): stale.append({"pid": int(parts[0]), "command": parts[1]})
    require(not stale, "stale Y2 worker/PID exists")
    process = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"], cwd=repo, text=True, capture_output=True)
    if process.returncode != 0: raise y2.StageYY2ScheduleError("nvidia-smi unavailable: " + process.stderr.strip())
    rows = []
    for line in process.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        require(len(values) == 7, "malformed nvidia-smi row")
        rows.append({"physical_index": int(values[0]), "device_name": values[1], "gpu_uuid": values[2], "pci_bus_id": values[3], "total_memory_mib": int(values[4]), "free_memory_mib": int(values[5]), "driver": values[6]})
    rows.sort(key=lambda row: row["physical_index"])
    compute = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], cwd=repo, text=True, capture_output=True)
    if compute.returncode != 0: raise y2.StageYY2ScheduleError("cannot query active GPU compute processes: " + compute.stderr.strip())
    busy_uuids = {line.split(",", 1)[0].strip() for line in compute.stdout.splitlines() if line.strip() and not line.lstrip().startswith("No running")}
    rows = y2.select_idle_h20_devices(rows, busy_uuids)
    worker_count = y2.validate_worker_count(len(rows))
    require(len({row["gpu_uuid"] for row in rows}) == len({row["pci_bus_id"] for row in rows}) == worker_count, "GPU UUID/PCI identities not unique")
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
    ledger = y2.read_json(repo / Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json"))
    projection = y2.schedule_from_source_ledger(ledger, worker_count)["y2"]
    gate = y2.signed("stage_y_y2_multi_gpu_startup_gate", "PASS", {"locked_worker_count": worker_count, "selection_policy": "all eligible idle H20 devices in ascending physical-index order, capped at eight", "devices": rows, "checkpoint_sha256": checkpoint_sha, "model_identity": "vit_base_patch16_224.augreg2_in21k_ft_in1k", "numeric_contract_digest": "stage-y-y1-grouped-jvp-vjp-frozen-v1", "cross_device_numeric_parity": True, "isolated_shard_roots": True, "shared_mutable_output": False, "dynamic_rebalancing": False, "stale_worker_pids": stale, "formal_unit_outcome_count_before_launch": 0, "resource_projection": projection, "started_unix_ns": __import__("time").time_ns()})
    y2.write_json_new(formal / "multi_gpu_startup_gate.json", gate)
    return {"status": "PASS", "locked_worker_count": worker_count, "physical_indices": [row["physical_index"] for row in rows], "next_action": "build-shards"}


def run_shard(repo: Path, shard_id: int) -> dict[str, Any]:
    gate = y2.read_json(repo / y2.ROOT / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
    worker_count = y2.validate_worker_count(gate["payload"]["locked_worker_count"])
    require(shard_id in y2.shard_ids(worker_count), "invalid shard id")
    physical = gate["payload"]["devices"][shard_id]["physical_index"]
    pending = repo / y2.ROOT / "shards" / f"shard-{shard_id:02d}" / "completion/shard.pending.json"
    y2.write_json_new(pending, y2.signed("stage_y_y2_shard_pending", "PENDING", {"shard_id": shard_id, "shard_count": worker_count, "launcher_pid": os.getpid(), "terminal_published": False}))
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(physical)
    env["STAGE_Y2_LAUNCH_AUTHORIZATION"] = gate["canonical_digest"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = subprocess.run([sys.executable, "-m", "vitnns.stage_y_y2_worker", "--repo-root", str(repo), "--shard-id", str(shard_id)], cwd=repo, env=env, text=True, capture_output=True)
    if process.returncode != 0:
        _publish_failed_shard(repo, shard_id, worker_count, process.stdout.strip(), process.stderr.strip())
        raise y2.StageYY2InfrastructureError(process.stdout.strip() or process.stderr.strip() or "shard worker failed")
    return json.loads(process.stdout)


def _publish_failed_shard(repo: Path, shard_id: int, worker_count: int, stdout: str, stderr: str) -> None:
    shard_root = repo / y2.ROOT / "shards" / f"shard-{shard_id:02d}"
    pending = shard_root / "completion/shard.pending.json"; terminal = shard_root / "completion/shard.json"
    if not pending.is_file() or terminal.exists():
        return
    counts = {state: 0 for state in y2.UNIT_TERMINAL_STATES}
    for path in shard_root.glob("units/unit-*/completion/unit.json"):
        document = y2.read_json(path); counts[document["status"]] += 1
    manifest = y2.read_json(shard_root / "shard_manifest.json")
    identity_path = shard_root / "worker_identity.json"
    identity = y2.read_json(identity_path)["payload"] if identity_path.is_file() else None
    failure = y2.signed("stage_y_y2_shard_terminal", "BLOCKED_INFRASTRUCTURE", {"shard_id": shard_id, "shard_count": worker_count, "unit_count": sum(counts.values()), "expected_unit_count": manifest["payload"]["unit_count"], "terminal_counts": counts, "worker_identity": identity, "resource_caps": manifest["payload"]["resource_caps"], "stdout": stdout, "stderr": stderr, "failure_code": "WORKER_PROCESS_NONZERO_EXIT", "completed_unix_ns": __import__("time").time_ns()})
    y2.publish_terminal(pending, terminal, failure)


def run_all(repo: Path) -> dict[str, Any]:
    formal = repo / y2.ROOT
    gate = y2.read_json(formal / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
    worker_count = y2.validate_worker_count(gate["payload"]["locked_worker_count"])
    require(all((formal / "shards" / f"shard-{shard:02d}/shard_manifest.json").is_file() for shard in y2.shard_ids(worker_count)), "locked shard manifest roster incomplete")
    require(not (formal / "worker_launch_roster.json").exists() and not (formal / "worker_launch_results.json").exists(), "formal worker launch already recorded")
    commands = []
    for shard in y2.shard_ids(worker_count):
        physical = gate["payload"]["devices"][shard]["physical_index"]
        commands.append({"shard_id": shard, "physical_index": physical, "cuda_visible_devices": str(physical)})
    y2.write_json_new(formal / "worker_launch_roster.json", y2.signed("stage_y_y2_worker_launch_roster", "PASS", {"locked_worker_count": worker_count, "commands": commands, "concurrent_launch": True, "dynamic_rebalancing": False}))
    for row in commands:
        pending = formal / "shards" / f"shard-{row['shard_id']:02d}" / "completion/shard.pending.json"
        y2.write_json_new(pending, y2.signed("stage_y_y2_shard_pending", "PENDING", {"shard_id": row["shard_id"], "shard_count": worker_count, "launcher_pid": os.getpid(), "terminal_published": False}))
    processes = []
    try:
        for row in commands:
            env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = row["cuda_visible_devices"]
            env["STAGE_Y2_LAUNCH_AUTHORIZATION"] = gate["canonical_digest"]
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            process = subprocess.Popen([sys.executable, "-m", "vitnns.stage_y_y2_worker", "--repo-root", str(repo), "--shard-id", str(row["shard_id"])], cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append((row, process))
    except Exception as error:
        for _row, process in processes:
            if process.poll() is None:
                process.terminate()
        for _row, process in processes:
            process.communicate()
        for row in commands:
            _publish_failed_shard(repo, row["shard_id"], worker_count, "", f"launcher spawn failure: {type(error).__name__}:{error}")
        raise y2.StageYY2InfrastructureError("worker fleet launch failed before the locked roster was fully started") from error
    results = []
    for row, process in processes:
        stdout, stderr = process.communicate()
        results.append({**row, "pid": process.pid, "returncode": process.returncode, "stdout": stdout.strip(), "stderr": stderr.strip()})
        if process.returncode != 0:
            _publish_failed_shard(repo, row["shard_id"], worker_count, stdout.strip(), stderr.strip())
    status = "PASS" if all(row["returncode"] == 0 for row in results) else "BLOCKED_INFRASTRUCTURE"
    y2.write_json_new(formal / "worker_launch_results.json", y2.signed("stage_y_y2_worker_launch_results", status, {"locked_worker_count": worker_count, "results": results, "all_workers_joined": True, "dynamic_rebalancing": False}))
    if status != "PASS":
        raise y2.StageYY2InfrastructureError("one or more locked shard workers failed; inspect worker_launch_results.json")
    return {"status": status, "locked_worker_count": worker_count, "next_action": "reconcile"}


def reconcile(repo: Path) -> dict[str, Any]:
    result = y2.reconcile_shards(repo / y2.ROOT)
    y2.write_json_new(repo / y2.ROOT / "reconciliation.json", result)
    return {"status": result["status"], **result["payload"]}


def _seal_terminal(repo: Path, status: str, reconciliation: Mapping[str, Any] | None, failure: Mapping[str, Any] | None) -> dict[str, Any]:
    root = repo / y2.ROOT; report_root = repo / y2.REPORT_ROOT
    pending = root / "completion/y2.pending.json"
    require(pending.is_file() and not (root / "completion/y2.json").exists(), "Y2 terminal transition is unavailable")
    valid_count = 0 if reconciliation is None else int(reconciliation["payload"]["valid_count"])
    outcome_count = len(list(root.glob("shards/shard-*/units/unit-*/completion/unit.json")))
    if failure is not None:
        y2.write_json_new(root / "failure.json", y2.signed("stage_y_y2_failure", status, failure))
    summary = y2.signed("stage_y_y2_terminal_access_summary", status, {
        "formal_unit_terminal_count": outcome_count,
        "formal_outcome_access": outcome_count > 0,
        "conditions_observed": ["clean"] if outcome_count else [],
        "non_clean_condition_count": 0,
        "bootstrap_executed": False, "holm_executed": False,
        "claim_decision_executed": False, "later_stages_started": [],
    })
    y2.write_json_new(root / "terminal_access_summary.json", summary)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "y2_report.md"
    report_text = (f"# Stage Y2 terminal report\n\nStatus: **{status}**. Valid clean units: {valid_count}. "
                   f"Formal unit terminals: {outcome_count}. No non-clean condition, bootstrap, Holm procedure, or claim decision ran.\n\n"
                   f"Next authorization: `{'STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY' if status == 'PASS' else 'NONE'}`.\n")
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(report_text); stream.flush(); os.fsync(stream.fileno())
    inventory = y2.whole_root_inventory(root, status)
    y2.write_json_new(root / "whole_root_inventory.json", inventory)
    completion = y2.signed("stage_y_y2_terminal_completion", status, {
        "terminal": True,
        "reconciliation": None if reconciliation is None else repo_file_record(repo, y2.ROOT / "reconciliation.json"),
        "failure": None if failure is None else repo_file_record(repo, y2.ROOT / "failure.json"),
        "whole_root_inventory": repo_file_record(repo, y2.ROOT / "whole_root_inventory.json"),
        "terminal_access_summary": repo_file_record(repo, y2.ROOT / "terminal_access_summary.json"),
        "report": repo_file_record(repo, y2.REPORT_ROOT / "y2_report.md"),
        "valid_clean_count": valid_count,
        "formal_outcome_access": outcome_count > 0,
        "conditions": ["clean"],
        "bootstrap_executed": False, "holm_executed": False,
        "claim_decisions": [],
        "next_authorization": "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY" if status == "PASS" else "NONE",
    })
    y2.publish_terminal(pending, root / "completion/y2.json", completion)
    return {"status": status, "valid_clean_count": valid_count, "next_authorization": completion["payload"]["next_authorization"]}


def seal_control_failure(repo: Path, error: Exception, phase: str) -> dict[str, Any]:
    status = getattr(error, "code", "BLOCKED_INFRASTRUCTURE")
    if status not in ("BLOCKED_COHORT", "BLOCKED_PROVENANCE", "BLOCKED_RESOURCE_SCHEDULE", "BLOCKED_INFRASTRUCTURE", "BLOCKED_BUDGET", "BLOCKED_SCIENTIFIC_VALIDITY"):
        status = "BLOCKED_INFRASTRUCTURE"
    failure = {"phase": phase, "failure_type": type(error).__name__, "message": str(error), "recovery_requires_new_root": True, "next_authorization": "NONE"}
    return _seal_terminal(repo, status, None, failure)


def seal(repo: Path) -> dict[str, Any]:
    reconciliation = y2.read_json(repo / y2.ROOT / "reconciliation.json"); y2.validate_signed(reconciliation)
    pending = repo / y2.ROOT / "completion/y2.pending.json"
    status = reconciliation["status"]
    require(status in ("PASS", "BLOCKED_BUDGET", "BLOCKED_INFRASTRUCTURE", "BLOCKED_SCIENTIFIC_VALIDITY"), "unsupported reconciliation terminal")
    failure = None if status == "PASS" else {"phase": "RECONCILIATION", "failure_type": status, "message": "source-recomputed reconciliation did not pass", "recovery_requires_new_root": True, "next_authorization": "NONE"}
    return _seal_terminal(repo, status, reconciliation, failure)


def validate(repo: Path, publication: bool = False) -> dict[str, Any]:
    path = repo / "scripts/validate_stage_y_y2.py"
    spec = importlib.util.spec_from_file_location("stage_y_y2_formal_validator", path)
    require(spec is not None and spec.loader is not None, "cannot load formal Y2 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(repo, publication=publication, allow_blocked=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal-cohort", "validate-startup", "build-shards", "run-shard", "run-all", "reconcile", "seal", "validate", "validate-publication"))
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
        elif args.command == "run-all": result = run_all(repo)
        elif args.command == "reconcile": result = reconcile(repo)
        elif args.command == "seal": result = seal(repo)
        else: result = validate(repo, args.command == "validate-publication")
    except Exception as error:
        pending = repo / y2.ROOT / "completion/y2.pending.json"
        terminal = repo / y2.ROOT / "completion/y2.json"
        if args.command not in ("validate", "validate-publication") and pending.is_file() and not terminal.exists():
            try:
                blocked = seal_control_failure(repo, error, args.command)
                print(json.dumps({**blocked, "error": str(error)}, sort_keys=True))
                return 2
            except Exception as seal_error:
                print(json.dumps({"status": "BLOCKED_INFRASTRUCTURE", "error": str(error), "terminal_seal_error": str(seal_error), "next_authorization": "NONE"}, sort_keys=True))
                return 2
        print(json.dumps({"status": getattr(error, "code", "BLOCKED"), "error": str(error), "next_authorization": "NONE"}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
