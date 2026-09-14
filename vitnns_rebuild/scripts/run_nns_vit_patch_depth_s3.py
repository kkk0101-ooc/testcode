#!/usr/bin/env python3
"""Launch S3 global-queue feature workers and preserve recovery evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa: E402

PYTHON = REPO.parent / ".conda/envs/vitnns/bin/python"
RUNTIME = REPO.parent / ".runtime/nns_vit_patch_depth_s3"


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def runtime_env() -> dict[str, str]:
    value = os.environ.copy()
    mapping = {"TMPDIR":"tmp","TMP":"tmp","TEMP":"tmp","XDG_CACHE_HOME":"cache","PIP_CACHE_DIR":"pip","CONDA_PKGS_DIRS":"conda_pkgs","HF_HOME":"hf","HF_DATASETS_CACHE":"hf_datasets","TRANSFORMERS_CACHE":"transformers","TORCH_HOME":"torch","TORCH_EXTENSIONS_DIR":"torch_extensions","TORCHINDUCTOR_CACHE_DIR":"torchinductor","TRITON_CACHE_DIR":"triton","CUDA_CACHE_PATH":"cuda","MPLCONFIGDIR":"matplotlib","NUMBA_CACHE_DIR":"numba"}
    for key, suffix in mapping.items():
        path = RUNTIME / suffix
        path.mkdir(parents=True, exist_ok=True)
        value[key] = str(path)
    value.update({"CUBLAS_WORKSPACE_CONFIG":":4096:8","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","NUMEXPR_NUM_THREADS":"2","PYTHONPATH":str(REPO / "src")})
    return value


def gpu_snapshot() -> list[dict[str, Any]]:
    query = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode","--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader,nounits"], text=True)
    active: dict[str, list[dict[str, int]]] = {}
    for line in processes.splitlines():
        if not line.strip():
            continue
        uuid, pid, used = [item.strip() for item in line.split(",")]
        active.setdefault(uuid, []).append({"pid":int(pid),"used_memory_mib":int(used)})
    rows = []
    for line in query.splitlines():
        index, uuid, name, total, used, free, utilization, mode = [item.strip() for item in line.split(",")]
        eligible = name == "NVIDIA H20" and int(free) >= 21504 and int(utilization) <= 10
        rows.append({"physical_index":int(index),"gpu_uuid":uuid,"name":name,"memory_total_mib":int(total),"memory_used_mib":int(used),"memory_free_mib":int(free),"utilization_percent":int(utilization),"compute_mode":mode,"existing_compute_processes":active.get(uuid,[]),"existing_compute_pid_disqualifying":False,"eligible":eligible,"eligibility_reason":"PASS_NAME_FREE_MEMORY_UTILIZATION" if eligible else "FAILED_NAME_OR_FREE_MEMORY_OR_UTILIZATION"})
    return sorted(rows, key=lambda row: (row["physical_index"], row["gpu_uuid"]))


def remote_tip() -> str:
    branch = git("symbolic-ref", "--short", "HEAD")
    output = subprocess.check_output(["git","ls-remote","origin",f"refs/heads/{branch}"], cwd=REPO, text=True).strip()
    return output.split()[0] if output else ""


def gate(phase: str) -> tuple[str, Mapping[str, Any]]:
    status = subprocess.check_output(["git","status","--porcelain=v2","--untracked-files=all"], cwd=REPO, text=True)
    unexplained = [line for line in status.splitlines() if "../.root_cleanup_quarantine/" not in line]
    s3.require(not unexplained, f"unexplained worktree changes: {unexplained[:5]}")
    head = git("rev-parse", "HEAD")
    seal_path = REPO / s3.ROOT_REL / "implementation_seal.json"
    s3.require(seal_path.is_file(), "pre-access implementation seal missing")
    seal = json.loads(seal_path.read_text())
    s3.require(s3.verify_signed(seal) and seal["status"] == "PASS", "implementation seal invalid")
    seal_commit = str(seal["payload"]["seal_commit"])
    s3.require(subprocess.run(["git","merge-base","--is-ancestor",seal_commit,head], cwd=REPO).returncode == 0, "HEAD does not contain implementation seal commit")
    s3.require(remote_tip() == head, "local HEAD is not the remotely published pre-access state")
    if phase == "test":
        gate_path = REPO / s3.ROOT_REL / "test_access_gate.json"
        weight_path = REPO / s3.ROOT_REL / "probe_weight_seal.json"
        s3.require(gate_path.is_file() and weight_path.is_file(), "test gate or weight seal missing")
        test_gate = json.loads(gate_path.read_text())
        weight = json.loads(weight_path.read_text())
        s3.require(s3.verify_signed(test_gate) and s3.verify_signed(weight) and test_gate["status"] == "PASS" and weight["status"] == "PASS", "test gate or weight seal invalid")
        gate_commit = str(test_gate["payload"]["publication_commit"])
        s3.require(gate_commit == head and remote_tip() == head, "weight-seal commit not present as local and remote published HEAD")
        s3.require(s3.sha_file(REPO / "assets/imagenet/ILSVRC2012_img_val.tar") == s3.JPEG_ARCHIVE_SHA256, "D2 JPEG archive SHA drift")
        s3.require(s3.sha_file(REPO / "assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip") == s3.MASK_ARCHIVE_SHA256, "mask archive SHA drift")
    return head, seal


def acquire_lock(row: Mapping[str, Any], launch_id: str) -> Path:
    lock = RUNTIME / "gpu_locks" / f"{row['gpu_uuid']}.lock"
    if lock.exists():
        previous = json.loads(lock.read_text())
        previous_pid = int(previous["pid"])
        try:
            os.kill(previous_pid, 0)
        except ProcessLookupError:
            stale = REPO / s3.ROOT_REL / "stale_gpu_locks" / f"{row['gpu_uuid']}-{previous['launch_id']}-{time.time_ns()}.json"
            stale.parent.mkdir(parents=True, exist_ok=True)
            os.replace(lock, stale)
            s3.fsync_directory(lock.parent)
            s3.fsync_directory(stale.parent)
        else:
            raise s3.S3Error(f"GPU UUID lock is held by live PID {previous_pid}: {row['gpu_uuid']}")
    s3.write_json_new(lock, {"launch_id":launch_id,"gpu_uuid":row["gpu_uuid"],"physical_index":row["physical_index"],"pid":os.getpid(),"created_time_ns":time.time_ns()})
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train_validation","test"))
    args = parser.parse_args()
    head, seal = gate(args.phase)
    rows = gpu_snapshot()
    eligible = [row for row in rows if row["eligible"]]
    selected = eligible[:8]
    s3.require(len(selected) == min(8, len(eligible)), "elastic worker selection drift")
    launch_index = 1
    launch_parent = REPO / s3.ROOT_REL / "launches"
    while (launch_parent / f"{args.phase}-launch-{launch_index:03d}").exists():
        launch_index += 1
    launch_id = f"{args.phase}-launch-{launch_index:03d}"
    launch_root = launch_parent / launch_id
    launch_root.mkdir(parents=True)
    manifest = s3.signed("s3_launch_manifest", "PENDING" if selected else "BLOCKED_INFRASTRUCTURE", {"launch_id":launch_id,"phase":args.phase,"gpu_snapshot":rows,"selected_gpus":selected,"worker_count":len(selected),"fixed_gpu_set":True,"dynamic_queue":True,"static_sharding":False,"created_time_ns":time.time_ns()}, producer_commit=head)
    s3.write_json_new(launch_root / "launch_manifest.json", manifest)
    if not selected:
        s3.write_json_new(launch_root / "launch_result.json", s3.signed("s3_launch_result", "BLOCKED_INFRASTRUCTURE", {"launch_id":launch_id,"phase":args.phase,"reason":"no eligible H20 at launch snapshot","missing_units_preserved":True}, producer_commit=head))
        print(json.dumps({"status":"BLOCKED_INFRASTRUCTURE","launch_id":launch_id}))
        return 2
    locks: list[Path] = []
    processes: list[tuple[subprocess.Popen[bytes], Any, Any, Mapping[str, Any], str]] = []
    started = time.monotonic()
    try:
        for row in selected:
            locks.append(acquire_lock(row, launch_id))
        for worker_index, row in enumerate(selected):
            worker_id = f"worker-{worker_index:02d}"
            log_path = launch_root / "logs" / f"{worker_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("xb")
            env = runtime_env()
            env["CUDA_VISIBLE_DEVICES"] = str(row["physical_index"])
            command = [str(PYTHON),"-m","vitnns.nns_patch_depth_s3_worker","--repo",str(REPO),"--launch-root",str(launch_root),"--launch-id",launch_id,"--phase",args.phase,"--worker-id",worker_id,"--physical-index",str(row["physical_index"]),"--gpu-uuid",str(row["gpu_uuid"]),"--producer-commit",head]
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, log, log_path, row, worker_id))
        exit_rows = []
        for process, log, log_path, row, worker_id in processes:
            return_code = process.wait()
            log.flush(); os.fsync(log.fileno()); log.close()
            exit_rows.append({"worker_id":worker_id,"physical_index":row["physical_index"],"gpu_uuid":row["gpu_uuid"],"pid":process.pid,"return_code":return_code,"log_path":log_path.relative_to(REPO).as_posix()})
    finally:
        for process, log, _path, _row, _worker_id in processes:
            if process.poll() is None:
                process.wait()
            if not log.closed:
                log.close()
        for lock in locks:
            if lock.exists():
                lock.unlink()
                s3.fsync_directory(lock.parent)
    schedule = json.loads((REPO / s3.SCHEDULE_REL).read_text())["payload"]["records"]
    allowed = {"probe_train","probe_validation"} if args.phase == "train_validation" else {"d2_test"}
    expected = [row for row in schedule if row["split"] in allowed]
    terminals = REPO / s3.ROOT_REL / "unit_terminals"
    missing = [int(row["schedule_index"]) for row in expected if not (terminals / f"unit-{int(row['schedule_index']):05d}.json").is_file()]
    status = "PASS" if not missing else "RECOVERY_REQUIRED"
    result = s3.signed("s3_launch_result", status, {"launch_id":launch_id,"phase":args.phase,"worker_exits":exit_rows,"expected_unit_count":len(expected),"missing_unit_count":len(missing),"missing_schedule_indices":missing,"launch_elapsed_seconds":time.monotonic()-started,"healthy_workers_continued_after_peer_exit":True,"fresh_launch_required_for_missing":bool(missing)}, producer_commit=head)
    s3.write_json_new(launch_root / "launch_result.json", result)
    print(json.dumps({"status":status,"launch_id":launch_id,"worker_count":len(selected),"missing_unit_count":len(missing),"launch_elapsed_seconds":result["payload"]["launch_elapsed_seconds"]}, sort_keys=True))
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
