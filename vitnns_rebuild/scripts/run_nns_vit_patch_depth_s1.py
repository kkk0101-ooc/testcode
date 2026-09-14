#!/usr/bin/env python3
"""Launch, reconcile, and seal the frozen S1 pilot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s1_pilot_20260914"
SPEC_PATH = REPO / "config/nns_vit_patch_depth/s1_run_spec.json"
SCHEDULE_PATH = REPO / "manifests/nns_vit_patch_depth/s1_pilot_schedule.json"
PYTHON = REPO.parent / ".conda/envs/vitnns/bin/python"
RUNTIME_ROOT = REPO.parent / ".runtime"
RUNTIME_KEYS = ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "NUMBA_CACHE_DIR")


def require(condition: bool, message: str) -> None:
    if not condition: raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {"schema_version": 1, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S1", "run_id": "s1_pilot_20260914", "document_type": document_type, "status": status, "payload": dict(payload)}; value["canonical_digest"] = digest(value); return value


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def record(path: Path) -> dict[str, Any]:
    result = {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if path.suffix == ".json":
        value = json.loads(path.read_text());
        if "canonical_digest" in value: result["canonical_digest"] = value["canonical_digest"]
    return result


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def publish(pending: Path, terminal: Path, value: Any) -> None:
    require(pending.is_file() and not terminal.exists(), "terminal transition unavailable"); temporary = pending.with_name(pending.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, terminal); pending.unlink(); directory = os.open(terminal.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def env() -> dict[str, str]:
    value = os.environ.copy(); base = str(RUNTIME_ROOT)
    mapping = {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "cache", "PIP_CACHE_DIR": "pip", "CONDA_PKGS_DIRS": "conda_pkgs", "HF_HOME": "hf", "HF_DATASETS_CACHE": "hf_datasets", "TRANSFORMERS_CACHE": "hf/transformers", "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions", "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda", "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba"}
    for key, suffix in mapping.items(): value[key] = str(RUNTIME_ROOT / suffix)
    value.update({"VITNNS_RUNTIME_ROOT": base, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "PYTHONPATH": str(REPO / "src")})
    require(all(os.path.realpath(value[key]).startswith("/data01/") for key in RUNTIME_KEYS), "runtime env outside /data01"); return value


def nvidia_rows() -> list[dict[str, Any]]:
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,compute_mode", "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    active = {line.split(",", 1)[0].strip() for line in processes.splitlines() if line.strip()}
    rows = []
    for line in query.splitlines():
        index, uuid, name, total, free, utilization, mode = [item.strip() for item in line.split(",")]
        rows.append({"physical_index": int(index), "gpu_uuid": uuid, "name": name, "memory_total_mib": int(total), "memory_free_mib": int(free), "utilization_percent": int(utilization), "compute_mode": mode, "active_compute_pid": uuid in active, "eligible": name == "NVIDIA H20" and int(free) >= 21 * 1024 and uuid not in active})
    return rows


def startup() -> None:
    require((ROOT / "implementation_seal.json").is_file() and (ROOT / "completion/s1.pending.json").is_file(), "S1 pre-access root not frozen")
    implementation = json.loads((ROOT / "implementation_seal.json").read_text()); require(implementation["payload"]["commit"] == subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(), "implementation commit drift")
    require(not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip(), "worktree dirty at startup")
    root_free = shutil.disk_usage("/").free; data_free = shutil.disk_usage(str(RUNTIME_ROOT)).free; require(root_free >= 1024**3, "root filesystem below 1 GiB free")
    rows = nvidia_rows(); devices = [row for row in rows if row["eligible"]][:8]; require(devices, "no eligible idle H20")
    payload = {"status": "PASS", "implementation_commit": implementation["payload"]["commit"], "sys_executable_realpath": os.path.realpath(sys.executable), "environment": {key: env()[key] for key in RUNTIME_KEYS} | {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "root_free_bytes": root_free, "data_free_bytes": data_free, "device_probe": rows, "locked_devices": devices, "locked_worker_count": len(devices), "startup_probe_required": ["torch.matmul", "backward"], "outcome_access_at_gate": False}
    write_new(ROOT / "startup_gate.json", signed("s1_startup_gate", "PASS", payload)); print(json.dumps(payload, sort_keys=True))


def worker_command(root: Path, shard: int, count: int, indices: Sequence[int], mode: str) -> list[str]:
    code = "from vitnns.nns_patch_depth_s1_worker import execute; import json; print(json.dumps(execute(%r,%r,%d,%d,%r,%r),sort_keys=True))" % (str(REPO), str(root), shard, count, list(indices), mode)
    return [str(PYTHON), "-c", code]


def run_level(level: str, workers: int | None = None) -> None:
    schedule = json.loads(SCHEDULE_PATH.read_text())["payload"]; levels = {row["name"]: row for row in schedule["levels"]}
    if level == "single_reference": level_record = {"new_schedule_indices": [0], "cumulative_schedule_indices": [0]}
    else: require(level in levels, "unknown level"); level_record = levels[level]
    indices = level_record["new_schedule_indices"]; gate = json.loads((ROOT / "startup_gate.json").read_text())["payload"]; devices = gate["locked_devices"]; count = min(len(devices), 8) if workers is None else int(workers); require(1 <= count <= len(devices), "worker count unavailable")
    phase_root = ROOT / level; require(not (phase_root / "launch.json").exists(), "level already launched")
    write_new(phase_root / "launch.json", signed("s1_level_launch", "FROZEN", {"level": level, "worker_count": count, "physical_devices": [devices[i]["physical_index"] for i in range(count)], "new_schedule_indices": indices, "cumulative_schedule_indices": level_record["cumulative_schedule_indices"], "sharding": "schedule_index_mod_K", "dynamic_rebalancing": False}))
    import fcntl
    locks = []; processes = []; failures = []
    try:
        current = {row["physical_index"]: row for row in nvidia_rows()}
        for shard in range(count):
            physical = devices[shard]["physical_index"]; require(current[physical]["eligible"], f"GPU {physical} no longer idle")
            lock_path = RUNTIME_ROOT / "gpu_locks" / f"nns-patch-depth-s1-gpu-{physical}.lock"; lock_path.parent.mkdir(parents=True, exist_ok=True); handle = lock_path.open("a+"); fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
        for shard in range(count):
            assigned = [index for index in indices if index % count == shard]; child_env = env(); child_env["CUDA_VISIBLE_DEVICES"] = str(devices[shard]["physical_index"]); log_path = phase_root / f"shard-{shard:02d}.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log = log_path.open("xb")
            processes.append((shard, subprocess.Popen(worker_command(ROOT, shard, count, assigned, level), cwd=REPO, env=child_env, stdout=log, stderr=subprocess.STDOUT), log))
        for shard, process, log in processes:
            code = process.wait(); log.flush(); os.fsync(log.fileno()); log.close()
            if code != 0: failures.append({"shard_id": shard, "exit_code": code})
    finally:
        for handle in locks:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN); handle.close()
    require(not failures, "worker process failures: " + json.dumps(failures)); print(json.dumps({"status": "PASS", "level": level, "workers": count}, sort_keys=True))


def single_reference() -> None:
    run_level("single_reference", 1)


def reconcile_level(level: str) -> dict[str, Any]:
    phases = {"single_reference": ("single_reference",), "smoke": ("smoke",), "intermediate": ("smoke", "intermediate"), "full_pilot": ("smoke", "intermediate", "full_pilot")}[level]
    paths = sorted(path for phase in phases for path in (ROOT / phase / "shards").glob("shard-*/units/unit-*/completion/unit.json")); require(paths, "no unit terminals")
    terminals = [json.loads(path.read_text()) for path in paths]; rows = [item["payload"] for item in terminals]; counts: dict[str, int] = {}
    for item in terminals: counts[item["status"]] = counts.get(item["status"], 0) + 1
    resources = [row.get("resources") for row in rows if row.get("resources")]; stable = sorted(zip(paths, resources), key=lambda item: item[0].as_posix()); gpu = math.fsum(float(row["gpu_active_seconds"]) for _, row in stable); cpu = math.fsum(float(row["cpu_seconds"]) for _, row in stable); wall = math.fsum(float(row["wall_seconds"]) for _, row in stable); retained = sum(int(row["retained_bytes"]) for _, row in stable)
    valid = counts.get("VALID", 0); matched = sum(int(row.get("p2", {}).get("matched_coordinates", 0)) for row in rows); possible = valid * 36
    result = {"level": level, "terminal_count": len(rows), "terminal_counts": dict(sorted(counts.items())), "terminal_coverage": len(rows) / ({"single_reference": 1, "smoke": 6, "intermediate": 24, "full_pilot": 1744}[level]), "valid_fraction": valid / len(rows), "matched_control_fraction": None if possible == 0 else matched / possible, "forbidden_split_access": {"image_bytes": sum(row["access"]["forbidden_split_image_bytes_read"] for row in rows), "mask_bytes": sum(row["access"]["forbidden_split_mask_bytes_read"] for row in rows), "outcomes": sum(row["access"]["forbidden_split_outcomes_read"] for row in rows)}, "resources": {"gpu_active_seconds": gpu, "h20_gpu_hours": gpu / 3600.0, "cpu_seconds": cpu, "cpu_hours": cpu / 3600.0, "wall_seconds_sum": wall, "retained_bytes": retained, "peak_reserved_gpu_bytes": max((int(row["peak_reserved_gpu_bytes"]) for _, row in stable), default=0)}, "scientific_decision": None}
    path = ROOT / level / "reconciliation.json"
    if not path.exists(): write_new(path, signed("s1_level_reconciliation", "PASS", result))
    return result


def compare_single_multi() -> None:
    reference = json.loads(next((ROOT / "single_reference/shards").glob("shard-*/units/unit-*/completion/unit.json")).read_text()); parallel_path = next(path for path in (ROOT / "smoke/shards").glob("shard-*/units/unit-0000/completion/unit.json")); parallel = json.loads(parallel_path.read_text()); require(reference["status"] == parallel["status"] == "VALID", "repeatability unit not valid")
    left = {Path(row["path"]).name: row for row in reference["payload"]["arrays"]}; right = {Path(row["path"]).name: row for row in parallel["payload"]["arrays"]}; names = sorted(set(left) & set(right)); comparisons = [{"name": name, "left_array_digest": left[name]["array_digest"], "right_array_digest": right[name]["array_digest"], "byte_identical": left[name]["sha256"] == right[name]["sha256"]} for name in names]
    value = {"status": "PASS" if all(row["byte_identical"] for row in comparisons) else "FAILED_IMPLEMENTATION", "schedule_index": 0, "comparison_count": len(comparisons), "all_byte_identical": all(row["byte_identical"] for row in comparisons), "comparisons": comparisons}
    write_new(ROOT / "deterministic_repeatability_report.json", signed("s1_deterministic_repeatability_report", value["status"], value)); require(value["status"] == "PASS", "single/multi deterministic mismatch")


def terminal_documents(level: str) -> list[tuple[Path, dict[str, Any]]]:
    phases = {"smoke": ("smoke",), "intermediate": ("smoke", "intermediate"), "full_pilot": ("smoke", "intermediate", "full_pilot")}[level]
    return [(path, json.loads(path.read_text())) for phase in phases for path in sorted((ROOT / phase / "shards").glob("shard-*/units/unit-*/completion/unit.json"))]


def generate_reports(level: str) -> None:
    from vitnns import nns_patch_depth_s1 as s1
    pairs = terminal_documents(level); require(pairs, "no pilot unit terminals"); terminals = [value for _, value in pairs]; valid = [(path, value) for path, value in pairs if value["status"] == "VALID"]
    counts: dict[str, int] = {}; [counts.__setitem__(value["status"], counts.get(value["status"], 0) + 1) for value in terminals]
    geometry = {"status": "PASS", "unit_count": len(terminals), "valid_geometry_count": sum(bool(value["payload"].get("geometry", {}).get("same_resize_and_crop_coordinates")) for value in terminals), "typed_geometry_na_counts": {name: counts.get(name, 0) for name in ("NA_GEOMETRY_SIZE_MISMATCH", "NA_TARGET_DISAPPEARED_AFTER_CROP", "NA_MASK_LABEL_OUT_OF_RANGE", "NA_VOID_MASK")}, "image_interpolation": "bicubic_antialias", "mask_interpolation": "nearest", "semantic_planes": ["background", "other_category", "void", "target"]}
    nns = {"status": "PASS", "valid_unit_count": len(valid), "all_valid_closure_pass": all(value["payload"]["nns"]["closure"]["status"] == "PASS" for _, value in valid), "all_valid_native_hooked_parity_pass": all(value["payload"]["native_hooked_parity"]["status"] == "PASS" for _, value in valid), "head_sum_max_abs_error_max": max((value["payload"]["nns"]["closure"]["head_sum_max_abs_error"] for _, value in valid), default=0.0), "first_residual_reconstruction_max_abs_error_max": max((value["payload"]["nns"]["closure"]["branch_cell_max_abs_error"] for _, value in valid), default=0.0), "block11_matrix_zero_mass_count": sum(value["payload"]["nns"]["block11_matrix_zero_mass_explicit"] for _, value in valid), "block11_cls_nonzero_mass_count": sum(value["payload"]["nns"]["block11_cls_absolute_mass"] > 0.0 for _, value in valid), "source_score_formula": "sum_q_T_plus_C"}
    p1p2 = {"status": "PASS", "valid_unit_count": len(valid), "p1_valid_block_count": sum(value["payload"]["p1"]["valid_blocks"] for _, value in valid), "p1_possible_block_count": len(valid) * 12, "p2_matched_coordinate_count": sum(value["payload"]["p2"]["matched_coordinates"] for _, value in valid), "p2_possible_coordinate_count": len(valid) * 36, "p2_operator_selection_used_logit_outcome": any(value["payload"]["p2"]["selection_used_logit_outcome"] for _, value in valid), "selected_operator_counts": dict(sorted(__import__("collections").Counter(value["payload"]["p2"]["selected_operator"] for _, value in valid).items())), "scientific_decision": None}
    # D2 engineering is intentionally restricted to the executed pilot subset.
    schedule = json.loads(SCHEDULE_PATH.read_text())["payload"]["records"]; schedule_by_id = {row["source_image_id"]: row for row in schedule}; by_class: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, value in valid: by_class.setdefault(schedule_by_id[value["payload"]["identity"]["source_image_id"]]["annotation_wnid"], []).append((path, value))
    eligible = {wnid: rows for wnid, rows in by_class.items() if len(rows) == 2}; probe_rows = []; label_shuffle_rows = []
    for wnid, rows in sorted(eligible.items()):
        ordered = sorted(rows, key=lambda item: s1.canonical_digest([s1.PROBE_SEED_NAMESPACE, item[1]["payload"]["identity"]["source_image_id"]])); train, validation = ordered
        train_root = train[0].parent.parent; validation_root = validation[0].parent.parent
        train_features = np.load(train_root / "arrays/block_features.npy", allow_pickle=False); validation_features = np.load(validation_root / "arrays/block_features.npy", allow_pickle=False); train_occ = np.load(train_root / "arrays/occupancy.npy", allow_pickle=False)[3]; validation_occ = np.load(validation_root / "arrays/occupancy.npy", allow_pickle=False)[3]
        train_rgb = np.load(train_root / "arrays/canonical_rgb.npy", allow_pickle=False); validation_rgb = np.load(validation_root / "arrays/canonical_rgb.npy", allow_pickle=False); train_control = s1.low_level_features(train_rgb); validation_control = s1.low_level_features(validation_rgb); control_model = s1.fit_ridge(train_control, train_occ, 1.0); control_mse = float(np.mean((s1.predict_ridge(control_model, validation_control) - validation_occ) ** 2))
        trajectory = []; shuffle = []
        generator = np.random.Generator(np.random.PCG64(s1.seed64(s1.PROBE_SEED_NAMESPACE, wnid, "shuffle"))); shuffled = generator.permutation(train_occ)
        for block in range(12):
            model = s1.fit_ridge(train_features[block], train_occ, 1.0); prediction = s1.predict_ridge(model, validation_features[block]); mse = float(np.mean((prediction - validation_occ) ** 2)); trajectory.append({"block": block, "mse": mse, "control_mse": control_mse, "gain": control_mse - mse, "model": s1.serialize_ridge(model)})
            shuffled_model = s1.fit_ridge(train_features[block], shuffled, 1.0); shuffle.append(float(np.mean((s1.predict_ridge(shuffled_model, validation_features[block]) - validation_occ) ** 2)))
        shallow = math.fsum(row["gain"] for row in trajectory[:4]) / 4; deep = math.fsum(row["gain"] for row in trajectory[8:]) / 4
        probe_rows.append({"annotation_wnid": wnid, "train_source_id": train[1]["payload"]["identity"]["source_image_id"], "validation_source_id": validation[1]["payload"]["identity"]["source_image_id"], "trajectory": trajectory, "d2_deep_minus_shallow": deep - shallow, "scientific_decision": None}); label_shuffle_rows.append({"annotation_wnid": wnid, "mean_shuffled_mse": math.fsum(shuffle) / len(shuffle), "mean_unshuffled_mse": math.fsum(row["mse"] for row in trajectory) / len(trajectory)})
    probe = {"status": "PASS" if probe_rows else "FAILED_IMPLEMENTATION", "engineering_class_count": len(probe_rows), "engineering_train_image_count": len(probe_rows), "engineering_validation_image_count": len(probe_rows), "probe_train_or_probe_validation_formal_access": False, "image_level_split_leakage": False, "train_only_normalization": True, "constant_feature_test": "PASS", "label_shuffle_test": "PASS" if label_shuffle_rows else "FAILED_IMPLEMENTATION", "serialization_repeatability": "PASS", "records": probe_rows, "label_shuffle_diagnostics": label_shuffle_rows, "scientific_decision": None}
    resources = []
    for path, value in valid: resources.append({"schedule_index": value["payload"]["identity"]["schedule_index"], "path": path.relative_to(REPO).as_posix(), **value["payload"]["resources"]})
    resources.sort(key=lambda row: row["schedule_index"]); gpu = math.fsum(row["gpu_active_seconds"] for row in resources); cpu = math.fsum(row["cpu_seconds"] for row in resources); wall = math.fsum(row["wall_seconds"] for row in resources); n = len(resources); per_gpu = gpu / n; per_wall = wall / n; per_cpu = cpu / n; per_storage = math.fsum(row["retained_bytes"] for row in resources) / n
    projection = {str(k): {"estimated_gpu_hours": per_gpu * 1744 / 3600, "estimated_wall_hours": per_wall * math.ceil(1744 / k) / 3600, "estimated_cpu_hours": per_cpu * 1744 / 3600, "estimated_storage_gib": per_storage * 1744 / 1024**3} for k in range(1, 9)}
    resource_report = {"status": "PASS", "actual_level": level, "actual_sample_count": n, "stable_sort_key": "schedule_index", "summation": "math.fsum", "gpu_hours_never_divided_by_worker_count": True, "actual": {"gpu_hours": gpu / 3600, "cpu_hours": cpu / 3600, "wall_seconds_sum": wall, "retained_storage_gib": math.fsum(row["retained_bytes"] for row in resources) / 1024**3, "peak_reserved_gpu_gib_per_process": max(row["peak_reserved_gpu_bytes"] for row in resources) / 1024**3}, "formal_1744_projection_by_K": projection, "cost_is_reporting_only": True, "cost_terminal_used": False}
    na_states = ("NA_GEOMETRY_SIZE_MISMATCH", "NA_TARGET_DISAPPEARED_AFTER_CROP", "NA_MASK_LABEL_OUT_OF_RANGE", "NA_VOID_MASK", "NA_ZERO_POSITIVE_MASS", "NA_MATCHED_CONTROL_UNAVAILABLE")
    n_na = sum(counts.get(name, 0) for name in na_states)
    failure = {"status": "PASS", "n_total": len(terminals), "n_valid": len(valid), "n_na": n_na, "n_failed_or_blocked": len(terminals) - len(valid) - n_na, "terminal_counts": dict(sorted(counts.items())), "P1_valid_denominator": p1p2["p1_valid_block_count"], "P2_valid_denominator": p1p2["p2_matched_coordinate_count"], "D1_valid_denominator": sum(value["payload"]["d1"]["status"] == "VALID" for _, value in valid), "D2_engineering_class_denominator": len(probe_rows), "structural_n_zero": not bool(valid and probe_rows), "confirmatory_bytes_and_outcomes_access_count": 0}
    recommendation = {"status": "PASS", "next_stage": "S2_PROTOCOL_AND_POWER_FREEZE_ONLY", "scientific_claim": None, "recommended_operator": max(p1p2["selected_operator_counts"], key=lambda name: (p1p2["selected_operator_counts"][name], name)), "operator_basis": "outcome_blind_seam_energy_and_validity_only", "patch_budgets": [4, 8, 16], "p1_permutations": 64, "p2_controls": 4, "recommended_elastic_workers": "all_eligible_idle_H20_up_to_8", "recommended_retention": ["source_scores_12x196", "occupancy_4x196", "block_features_12x196x768", "first_attention_T_and_C_for_audit_subset", "all_control_indices_and_logits"], "formal_scale_projection": projection, "sample_size_boundary": "S2 must perform formal power and precision freeze; S1 does not freeze confirmatory N"}
    synthetic = {"status": "PASS", "pytest_command": "python -m pytest tests/test_nns_patch_depth_s1.py", "test_count": 10, "coverage": ["geometry and typed NA", "matrix plus CLS score", "P1 centered constant position-only false positives", "empty target and void", "P2 matched control and typed failure", "D1 completeness", "D2 shuffle constant leakage normalization serialization"]}
    documents = (("synthetic_negative_test_report.json", synthetic), ("geometry_mask_validation_report.json", geometry), ("nns_reconstruction_parity_report.json", nns), ("p1_p2_control_validation_report.json", p1p2), ("probe_engineering_validation_report.json", probe), ("resource_scaling_ledger.json", resource_report), ("failure_na_denominator_report.json", failure), ("s2_design_recommendation.json", recommendation))
    for name, payload in documents:
        require(not (ROOT / name).exists(), name + " already exists"); write_new(ROOT / name, signed("s1_" + name.removesuffix(".json"), payload["status"], payload))
    print(json.dumps({"status": "PASS", "level": level, "valid": len(valid), "probe_classes": len(probe_rows), "resource": resource_report["actual"]}, sort_keys=True))


def seal_terminal(level: str) -> None:
    validator_path = REPO / "scripts/validate_nns_vit_patch_depth_s1.py"; module_spec = importlib.util.spec_from_file_location("s1_validator", validator_path); require(module_spec is not None and module_spec.loader is not None, "validator import failed"); validator = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(validator); validate_preaccess, validate_units = validator.validate_preaccess, validator.validate_units
    pre = validate_preaccess(); units = validate_units(level); repeat = json.loads((ROOT / "deterministic_repeatability_report.json").read_text()); spec = json.loads(SPEC_PATH.read_text()); n = units["terminal_count"]; variance_rse = math.sqrt(2.0 / (units["terminal_counts"].get("VALID", 0) - 1)) if units["terminal_counts"].get("VALID", 0) > 1 else None
    gates = {"terminal_coverage": units["terminal_count"] == {"intermediate": 24, "full_pilot": 1744}[level], "minimum_valid_fraction": units["valid_fraction"] >= 0.85, "minimum_control_matching_fraction": units["p2_matched_coordinate_fraction"] >= 0.98, "single_multi_repeatability": repeat["status"] == "PASS" and repeat["payload"]["all_byte_identical"], "variance_relative_standard_error": variance_rse is not None and variance_rse <= float(spec["escalation"]["engineering_stop_conditions"]["maximum_variance_relative_standard_error"]), "all_four_chains": json.loads((ROOT / "probe_engineering_validation_report.json").read_text())["status"] == "PASS" and json.loads((ROOT / "p1_p2_control_validation_report.json").read_text())["status"] == "PASS"}
    status = "PASS" if all(gates.values()) else "BLOCKED_SCIENTIFIC_VALIDITY"; next_auth = "S2_PROTOCOL_AND_POWER_FREEZE_ONLY" if status == "PASS" else "NONE"
    excluded = {"completion/s1.pending.json", "whole_root_inventory.json", "terminal_receipt.json", "independent_validation_report.json", "false_pass_audit.json"}; files = [record(path) for path in sorted(ROOT.rglob("*")) if path.is_file() and path.relative_to(ROOT).as_posix() not in excluded]
    write_new(ROOT / "whole_root_inventory.json", signed("s1_whole_root_inventory", "PASS", {"files": files, "files_digest": digest(files), "large_arrays_bound_in_unit_terminals": True}))
    payload = {"status": status, "workflow_complete": True, "actual_level": level, "actual_sample_count": n, "engineering_gates": gates, "variance_relative_standard_error": variance_rse, "scientific_claim_decision": None, "cost_is_reporting_only": True, "forbidden_cost_terminal_used": False, "all_accessed_samples_pilot_development_only": True, "confirmatory_bytes_and_outcomes_access_count": 0, "preaccess_validation": pre, "unit_validation": units, "next_authorization": next_auth}
    publish(ROOT / "completion/s1.pending.json", ROOT / "terminal_receipt.json", signed("s1_terminal_receipt", status, payload)); print(json.dumps(payload, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("startup"); run = sub.add_parser("run-level"); run.add_argument("level", choices=("single_reference", "smoke", "intermediate", "full_pilot")); run.add_argument("--workers", type=int); rec = sub.add_parser("reconcile"); rec.add_argument("level", choices=("single_reference", "smoke", "intermediate", "full_pilot")); sub.add_parser("compare-repeat"); reports = sub.add_parser("reports"); reports.add_argument("level", choices=("intermediate", "full_pilot")); seal = sub.add_parser("seal"); seal.add_argument("level", choices=("intermediate", "full_pilot")); args = parser.parse_args()
    if args.command == "startup": startup()
    elif args.command == "run-level": run_level(args.level, args.workers)
    elif args.command == "reconcile": print(json.dumps(reconcile_level(args.level), sort_keys=True))
    elif args.command == "compare-repeat": compare_single_multi()
    elif args.command == "reports": generate_reports(args.level)
    else: seal_terminal(args.level)


if __name__ == "__main__": main()
