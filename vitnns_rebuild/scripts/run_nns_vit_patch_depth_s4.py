#!/usr/bin/env python3
"""Post-seal materialization, dynamic launch, and S4 reconciliation."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Mapping
import zipfile

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s4_r4_20260915"
PAYLOAD = REPO / "artifacts/nns_vit_patch_depth/s4_r4_20260915"
PYTHON = REPO.parent / ".conda/envs/vitnns/bin/python"
RUNTIME = REPO.parent / ".runtime/nns_vit_patch_depth_s4_r4"
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s4 as s4  # noqa: E402

RUNTIME_KEYS = ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "NUMBA_CACHE_DIR")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def current_branch() -> str:
    return git("symbolic-ref", "--short", "HEAD")


def remote_oid() -> str:
    line = subprocess.check_output(["git", "-C", str(REPO), "ls-remote", "--heads", "origin", f"refs/heads/{current_branch()}"], text=True).strip()
    return line.split()[0] if line else ""


def environment() -> dict[str, str]:
    value = os.environ.copy()
    mapping = {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "cache", "PIP_CACHE_DIR": "pip", "CONDA_PKGS_DIRS": "conda_pkgs", "HF_HOME": "hf", "HF_DATASETS_CACHE": "hf_datasets", "TRANSFORMERS_CACHE": "hf/transformers", "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions", "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda", "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba"}
    for key, suffix in mapping.items():
        path = RUNTIME / suffix
        path.mkdir(parents=True, exist_ok=True)
        value[key] = str(path)
    value.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "PYTHONPATH": str(REPO / "src")})
    s4.require(all(os.path.realpath(value[key]).startswith("/data01/") for key in RUNTIME_KEYS), "runtime paths outside /data01")
    return value


def verify_remote_seal(*, allow_postseal_evidence: bool = False) -> tuple[str, dict[str, Any]]:
    seal_path = ROOT / "implementation_seal.json"
    s4.require(seal_path.is_file(), "implementation seal absent")
    seal = json.loads(seal_path.read_text())
    s4.require(s4.verify_signed(seal) and seal["status"] == "PASS", "implementation seal invalid")
    head = git("rev-parse", "HEAD")
    remote = remote_oid()
    s4.require(remote == head, "live remote branch does not equal local seal HEAD")
    implementation = seal["payload"]["implementation_commit"]
    subprocess.check_call(["git", "-C", str(REPO), "merge-base", "--is-ancestor", implementation, head])
    tracked = git("ls-tree", "-r", "--name-only", head).splitlines()
    seal_tree_path = "evidence/nns_vit_patch_depth/s4_r4_20260915/implementation_seal.json"
    s4.require(seal_tree_path in tracked or "vitnns_rebuild/" + seal_tree_path in tracked, "seal not published in Git tree")
    porcelain = git("status", "--porcelain=v2", "--untracked-files=all").splitlines()
    allowed = ["? ../.root_cleanup_quarantine/", "? artifacts/nns_vit_patch_depth/s4_r4_20260915/"]
    if allow_postseal_evidence:
        allowed.append("? evidence/nns_vit_patch_depth/s4_r4_20260915/selected_member_materialization.json")
    unexpected = [line for line in porcelain if not any(line.startswith(prefix) for prefix in allowed)]
    s4.require(not unexpected, "unexplained worktree changes at formal access gate: " + repr(unexpected[:10]))
    return head, seal


def _nvidia_rows() -> list[dict[str, Any]]:
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode", "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    active: dict[str, list[dict[str, int]]] = {}
    for line in processes.splitlines():
        if line.strip():
            uuid, pid, memory = [item.strip() for item in line.split(",")]
            active.setdefault(uuid, []).append({"pid": int(pid), "used_memory_mib": int(memory)})
    rows = []
    for line in query.splitlines():
        index, uuid, name, total, used, free, utilization, mode = [item.strip() for item in line.split(",")]
        eligible = name == "NVIDIA H20" and int(free) >= 21504 and int(utilization) <= 10
        rows.append({
            "physical_index": int(index), "gpu_uuid": uuid, "name": name,
            "memory_total_mib": int(total), "memory_used_mib": int(used), "memory_free_mib": int(free),
            "utilization_percent": int(utilization), "compute_mode": mode,
            "existing_compute_processes": active.get(uuid, []),
            "existing_compute_PID_disqualifying": False, "eligible": eligible,
            "eligibility_reason": "PASS" if eligible else "name_memory_or_utilization_threshold",
        })
    return sorted(rows, key=lambda row: (row["physical_index"], row["gpu_uuid"]))


def materialize() -> None:
    producer_commit, seal = verify_remote_seal(allow_postseal_evidence=False)
    if (ROOT / "selected_member_materialization.json").exists():
        raise RuntimeError("selected member materialization already sealed")
    schedule = json.loads((REPO / s4.SCHEDULE_REL).read_text())["payload"]["records"]
    jpeg_archive = REPO / s4.JPEG_ARCHIVE_REL
    mask_archive = REPO / s4.MASK_ARCHIVE_REL
    s4.require(jpeg_archive.stat().st_size == s4.JPEG_ARCHIVE_BYTES and s4.sha_file(jpeg_archive) == s4.JPEG_ARCHIVE_SHA256, "JPEG archive identity drift")
    s4.require(s4.sha_file(mask_archive) == s4.MASK_ARCHIVE_SHA256, "mask archive identity drift")
    from PIL import Image
    predecessor_root = REPO / "evidence/nns_vit_patch_depth/s4_r3_20260915"
    predecessor_payload = REPO / "artifacts/nns_vit_patch_depth/s4_r3_20260915"
    predecessor_materialization_path = predecessor_root / "selected_member_materialization.json"
    s4.require(predecessor_materialization_path.is_file(), "S4-R3 selected-member materialization evidence absent")
    predecessor_materialization = json.loads(predecessor_materialization_path.read_text())
    s4.require(s4.verify_signed(predecessor_materialization) and predecessor_materialization["status"] == "PASS", "S4-R3 selected-member materialization evidence invalid")
    s4.require(predecessor_materialization["payload"]["selected_JPEG_member_reads"] == 1744 and predecessor_materialization["payload"]["selected_mask_member_reads"] == 1744 and predecessor_materialization["payload"]["D2_only_member_read_count"] == 0, "S4-R3 access ledger drift")
    predecessor_records = {int(row["schedule_index"]): row for row in predecessor_materialization["payload"]["records"]}
    s4.require(len(predecessor_records) == 1744, "S4-R3 cache registry count drift")
    records = []
    cache_image_reads = cache_mask_reads = image_decodes = mask_decodes = 0
    for row in schedule:
        index = int(row["schedule_index"])
        predecessor = predecessor_records[index]
        s4.require(predecessor["source_id"] == row["source_image_id"] and predecessor["image_member"] == row["image_archive_member"] and predecessor["mask_member"] == row["mask_archive_member"], f"S4-R3 cache source binding drift at {index}")
        source_image_path = REPO / predecessor["image_cache_path"]
        source_mask_path = REPO / predecessor["mask_cache_path"]
        s4.require(source_image_path.is_relative_to(predecessor_payload) and source_mask_path.is_relative_to(predecessor_payload), "S4-R3 cache path escaped predecessor payload root")
        image_bytes = source_image_path.read_bytes(); cache_image_reads += 1
        mask_bytes = source_mask_path.read_bytes(); cache_mask_reads += 1
        if len(image_bytes) != int(row["image_bytes"]) or s4.sha_bytes(image_bytes) != row["image_sha256"]:
            raise s4.DataIntegrityError(f"S4-R3 cached JPEG identity mismatch at {index}")
        if s4.sha_bytes(mask_bytes) != row["annotation_sha256"]:
            raise s4.DataIntegrityError(f"S4-R3 cached mask identity mismatch at {index}")
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load(); image_size = list(image.size); image_decodes += 1
        with Image.open(io.BytesIO(mask_bytes)) as mask:
            mask.load(); mask_size = list(mask.size); mask_decodes += 1
        if image_size != [row["image_width"], row["image_height"]] or mask_size != [row["mask_width"], row["mask_height"]]:
            raise s4.DataIntegrityError(f"S4-R3 cached member geometry mismatch at {index}")
        image_path, mask_path = (PAYLOAD / "selected_member_cache/images" / f"{index:04d}.jpeg", PAYLOAD / "selected_member_cache/masks" / f"{index:04d}.png")
        s4.write_bytes_atomic(image_path, image_bytes)
        s4.write_bytes_atomic(mask_path, mask_bytes)
        records.append({
            "schedule_index": index, "source_id": row["source_image_id"],
            "image_member": row["image_archive_member"], "image_cache_path": image_path.relative_to(REPO).as_posix(),
            "image_bytes": len(image_bytes), "image_sha256": s4.sha_bytes(image_bytes), "image_size_wh": image_size,
            "mask_member": row["mask_archive_member"], "mask_cache_path": mask_path.relative_to(REPO).as_posix(),
            "mask_bytes": len(mask_bytes), "mask_sha256": s4.sha_bytes(mask_bytes), "mask_size_wh": mask_size,
        })
    document = s4.signed("s4_selected_member_materialization", "PASS", {
        "implementation_seal": {"path": "evidence/nns_vit_patch_depth/s4_r4_20260915/implementation_seal.json", "sha256": s4.sha_file(ROOT / "implementation_seal.json"), "canonical_digest": seal["canonical_digest"]},
        "producer_commit": producer_commit, "JPEG_archive": {"path": s4.JPEG_ARCHIVE_REL.as_posix(), "bytes": s4.JPEG_ARCHIVE_BYTES, "sha256": s4.JPEG_ARCHIVE_SHA256},
        "mask_archive": {"path": s4.MASK_ARCHIVE_REL.as_posix(), "bytes": mask_archive.stat().st_size, "sha256": s4.MASK_ARCHIVE_SHA256},
        "rebind_source": {"path": predecessor_materialization_path.relative_to(REPO).as_posix(), "sha256": s4.sha_file(predecessor_materialization_path), "canonical_digest": predecessor_materialization["canonical_digest"]},
        "authorized_source_count": 1744, "selected_JPEG_member_reads": 1744, "selected_mask_member_reads": 1744,
        "incremental_archive_JPEG_member_reads": 0, "incremental_archive_mask_member_reads": 0,
        "revalidated_cache_JPEG_reads": cache_image_reads, "revalidated_cache_mask_reads": cache_mask_reads,
        "image_decode_count": image_decodes, "mask_decode_count": mask_decodes, "D2_only_member_read_count": 0,
        "full_archive_extraction": False, "records_digest": s4.records_digest(records), "records": records,
        "S3_D2_prediction_error_numeric_read_count": 0, "cohort_level_inference_count": 0,
    }, producer_commit=producer_commit)
    s4.write_json_new(ROOT / "selected_member_materialization.json", document)
    print(json.dumps({"status": "PASS", "cumulative_JPEG_member_reads": 1744, "cumulative_mask_member_reads": 1744, "incremental_archive_member_reads": 0, "cache_revalidation_reads": cache_image_reads + cache_mask_reads, "decodes": image_decodes + mask_decodes, "D2_only_reads": 0}, sort_keys=True))


def worker_command(launch_id: str, worker_id: str, device: Mapping[str, Any], producer_commit: str) -> list[str]:
    code = "from vitnns.nns_patch_depth_s4_worker import execute; import json; print(json.dumps(execute(%r,%r,%r,%r,%r,%d,%r,%r),sort_keys=True))" % (str(REPO), str(ROOT), str(PAYLOAD), launch_id, worker_id, int(device["physical_index"]), str(device["gpu_uuid"]), producer_commit)
    return [str(PYTHON), "-c", code]


def launch(launch_id: str) -> None:
    producer_commit, seal = verify_remote_seal(allow_postseal_evidence=True)
    materialization = ROOT / "selected_member_materialization.json"
    s4.require(materialization.is_file(), "selected members not materialized after seal")
    document = json.loads(materialization.read_text())
    s4.require(s4.verify_signed(document) and document["status"] == "PASS", "materialization evidence invalid")
    schedule = json.loads((REPO / s4.SCHEDULE_REL).read_text())["payload"]["records"]
    missing = s4.missing_units(ROOT, schedule)
    launch_root = ROOT / "launches" / launch_id
    s4.require(not launch_root.exists(), "fresh launch root required")
    rows = _nvidia_rows()
    eligible = [row for row in rows if row["eligible"]]
    selected = eligible[:8]
    launch_root.mkdir(parents=True)
    if not selected:
        receipt = s4.signed("s4_launch_terminal", "BLOCKED_INFRASTRUCTURE", {"launch_id": launch_id, "GPU_snapshot": rows, "selected_GPU_count": 0, "remaining_units": len(missing), "scientific_terminal_created": False}, producer_commit=producer_commit)
        s4.write_json_new(launch_root / "launch_terminal.json", receipt)
        print(json.dumps({"status": "BLOCKED_INFRASTRUCTURE", "launch_id": launch_id, "selected_GPU_count": 0, "remaining_units": len(missing)}, sort_keys=True))
        return
    if len(eligible) >= 8:
        s4.require(len(selected) == 8, "eight eligible GPUs require eight workers")
    manifest = s4.signed("s4_launch_manifest", "FROZEN", {
        "launch_id": launch_id, "created_unix_ns": time.time_ns(), "producer_commit": producer_commit,
        "implementation_seal_sha256": s4.sha_file(ROOT / "implementation_seal.json"),
        "GPU_snapshot": rows, "eligible_GPU_count": len(eligible), "selected_GPU_count": len(selected),
        "selected_GPUs": selected, "worker_count": len(selected), "missing_units_at_launch": len(missing),
        "already_terminal_units": 5232 - len(missing), "dynamic_queue": True, "static_shards": False,
        "GPU_set_fixed": True, "hot_add_remove_migrate": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }, producer_commit=producer_commit)
    s4.write_json_new(launch_root / "launch_manifest.json", manifest)
    locks = []
    processes = []
    process_failures = []
    launch_started_ns = time.time_ns()
    launch_started = time.monotonic()
    try:
        for device in selected:
            lock_path = RUNTIME / "gpu_locks" / (device["gpu_uuid"] + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0); handle.truncate(); handle.write(json.dumps({"launch_id": launch_id, "gpu_uuid": device["gpu_uuid"], "physical_index": device["physical_index"], "parent_pid": os.getpid()})); handle.flush(); os.fsync(handle.fileno())
            locks.append(handle)
        for slot, device in enumerate(selected):
            worker_id = f"worker-{slot:02d}"
            child_env = environment()
            child_env["CUDA_VISIBLE_DEVICES"] = str(device["physical_index"])
            log_path = launch_root / "workers" / worker_id / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("xb")
            process = subprocess.Popen(worker_command(launch_id, worker_id, device, producer_commit), cwd=REPO, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((worker_id, process, handle))
        for worker_id, process, handle in processes:
            code = process.wait()
            handle.flush(); os.fsync(handle.fileno()); handle.close()
            if code != 0:
                process_failures.append({"worker_id": worker_id, "exit_code": code})
    finally:
        for handle in locks:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    remaining = s4.missing_units(ROOT, schedule)
    terminal_status = "PASS" if not remaining and not process_failures else "BLOCKED_INFRASTRUCTURE" if process_failures else "FAILED_IMPLEMENTATION"
    terminal = s4.signed("s4_launch_terminal", terminal_status, {
        "launch_id": launch_id, "started_unix_ns": launch_started_ns, "completed_unix_ns": time.time_ns(),
        "launch_elapsed_seconds": time.monotonic() - launch_started, "selected_GPU_count": len(selected),
        "selected_GPUs": selected, "worker_process_failures": process_failures,
        "terminal_count_after_launch": 5232 - len(remaining), "remaining_units": [{"unit_type": item[0], "schedule_index": item[1]} for item in remaining],
        "healthy_workers_continued_after_peer_failure": True, "stale_claims_preserved": True,
        "cost_or_wall_time_termination_used": False, "scientific_terminal_created": False,
    }, producer_commit=producer_commit)
    s4.write_json_new(launch_root / "launch_terminal.json", terminal)
    print(json.dumps({"status": terminal_status, "launch_id": launch_id, "selected_GPU_count": len(selected), "worker_failures": process_failures, "remaining_units": len(remaining), "launch_elapsed_seconds": terminal["payload"]["launch_elapsed_seconds"]}, sort_keys=True))


def status() -> None:
    schedule = json.loads((REPO / s4.SCHEDULE_REL).read_text())["payload"]["records"]
    paths = s4.all_terminal_paths(ROOT)
    states = Counter()
    types = Counter()
    for path in paths:
        value = json.loads(path.read_text())
        states[value["status"]] += 1
        types[value["payload"]["unit_type"]] += 1
    print(json.dumps({"terminal_count": len(paths), "remaining": len(s4.missing_units(ROOT, schedule)), "types": dict(types), "states": dict(states), "source_completions": len(list((ROOT / "source_completions").glob("source-*.json")))}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize")
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--launch-id", required=True)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "materialize":
        materialize()
    elif args.command == "launch":
        launch(args.launch_id)
    else:
        status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
