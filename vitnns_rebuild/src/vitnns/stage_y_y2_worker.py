"""Isolated one-visible-GPU worker for a frozen Stage Y2 shard.

This module is imported only by the explicitly authorized ``run-shard`` path.
Metadata sealing, P0 validation, reconciliation, and tests do not import Torch,
Pillow, timm, torchvision, or touch CUDA.
"""

from __future__ import annotations

import io
import os
import platform
import json
import socket
import tarfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import stage_y_y1 as y1
from . import stage_y_y2 as y2
from .stage_y_y1_worker import _axes

MODEL_PATH = Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth")
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
MODEL_LOADER_IDENTITY = "F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES"
NUMERIC_CONTRACT_DIGEST = "stage-y-y1-grouped-jvp-vjp-frozen-v1"


class UnitNumericalError(y1.StageYY1NumericalError):
    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message); self.attempts = attempts


class UnitInfrastructureError(y2.StageYY2Error):
    code = "BLOCKED_INFRASTRUCTURE"
    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message); self.attempts = attempts


def _write_array_new(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value, dtype="<f4")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return {
        "path": path.as_posix(), "bytes": path.stat().st_size,
        "sha256": y2.sha_file(path), "canonical_digest": y1.array_digest(array),
        "dtype": str(array.dtype), "shape": list(array.shape),
        "axes": _axes(path.name, array.shape),
    }


def _load_selected_image(repo: Path, row: Mapping[str, Any], device: str):
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf

    with tarfile.open(repo / y2.TAR_PATH, "r:") as archive:
        member = archive.getmember(str(row["tar_member_name"]))
        stream = archive.extractfile(member)
        if stream is None:
            raise y2.StageYY2ProvenanceError("selected tar member unavailable")
        raw = stream.read()
    if __import__("hashlib").sha256(raw).hexdigest() != row["file_sha256"]:
        raise y2.StageYY2ProvenanceError("selected JPEG raw-byte digest drift")
    with Image.open(io.BytesIO(raw)) as source:
        rgb = source.convert("RGB")
        resized = tf.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        cropped = tf.center_crop(resized, [224, 224])
        return ((tf.pil_to_tensor(cropped).float().div(255).unsqueeze(0) - 0.5) / 0.5).to(device)


def _compute_unit(model: Any, image: Any, row: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Execute the frozen clean Y1 identity once for one independent image."""
    import math
    import torch
    from .stage_x_nns_v2 import input_ig

    source_id = str(row["source_object_id"])
    started = time.monotonic(); cpu_started = time.process_time()
    torch.cuda.reset_peak_memory_stats(image.device)
    gpu_start = torch.cuda.Event(enable_timing=True); gpu_end = torch.cuda.Event(enable_timing=True); gpu_start.record()
    baseline_image = torch.zeros_like(image)
    with torch.no_grad():
        logits = model(image); target = int(logits.argmax(1)); clean_logit = float(logits[0, target])
        baseline_logit = float(model(baseline_image)[0, target])
    calls = {"forward_calls": 2, "backward_calls": 0, "jvp_calls": 0, "jvp_direction_lanes": 0, "vjp_calls": 0, "quadrature_alpha_batches": 0}
    attempts: list[dict[str, Any]] = []
    selected = None; selected_arrays = None; selected_closure = None; selected_integration = None
    for steps in spec["execution"]["quadrature_schedule"]:
        attempt_started = time.monotonic()
        try:
            candidate = input_ig(model, image, baseline_image, target, int(steps), int(spec["execution"]["alpha_chunk"]))
        except Exception as error:
            attempts.append({"steps": int(steps), "all_contracts_pass": False, "failure_code": f"{type(error).__name__}:{error}", "wall_seconds": time.monotonic() - attempt_started})
            raise UnitInfrastructureError(str(error), attempts) from error
        batches = math.ceil(int(steps) / int(spec["execution"]["alpha_chunk"]))
        calls["forward_calls"] += batches; calls["backward_calls"] += batches
        delta = clean_logit - baseline_logit
        input_error = abs(float(candidate.double().sum()) - delta)
        input_threshold = 1e-4 + 0.005 * abs(delta)
        closure = None; identity_error = None; arrays = None; integration = None
        if input_error <= input_threshold:
            try:
                integration = y1.integrate_routes(
                    model, image, baseline_image, target, int(steps),
                    alpha_chunk=int(spec["execution"]["alpha_chunk"]),
                    source_patch_chunk=int(spec["execution"]["source_patch_jvp_chunk"]),
                    component_vjp_chunk=int(spec["execution"]["component_vjp_chunk"]),
                )
                calls = {key: calls.get(key, 0) + integration.calls.get(key, 0) for key in set(calls) | set(integration.calls)}
                identity_error = float((integration.attribution - candidate).abs().max())
                arrays = y1.materialize_arrays(integration); closure = y1.validate_closure(arrays, delta)
            except Exception as error:
                attempts.append({"steps": int(steps), "input_sum": float(candidate.double().sum()), "input_delta": delta, "input_abs_error": input_error, "threshold": input_threshold, "all_contracts_pass": False, "failure_code": f"{type(error).__name__}:{error}", "wall_seconds": time.monotonic() - attempt_started, "array_digest": y1.array_digest(candidate)})
                raise UnitInfrastructureError(str(error), attempts) from error
        passed = input_error <= input_threshold and closure is not None and closure["status"] == "PASS" and identity_error is not None and identity_error <= 1e-5
        attempts.append({
            "steps": int(steps), "input_sum": float(candidate.double().sum()),
            "input_delta": delta, "input_abs_error": input_error,
            "threshold": input_threshold, "all_contracts_pass": passed,
            "closure_status": None if closure is None else closure["status"],
            "identity_max_abs_error": identity_error,
            "wall_seconds": time.monotonic() - attempt_started,
            "array_digest": y1.array_digest(candidate),
        })
        if passed:
            selected = int(steps); selected_arrays = arrays; selected_closure = closure; selected_integration = integration; break
    if selected is None or selected_arrays is None or selected_integration is None:
        raise UnitNumericalError("Y2 clean unit nonconverged at 8192", attempts)
    head_maps = {block: selected_arrays[f"b{block:02d}_first_heads_cell"] for block in range(12)}
    attention_maps = {block: selected_arrays[f"b{block:02d}_first_attention_cell"] for block in range(12)}
    try:
        ablations, ablation_calls = y1.run_head_ablations(model, image, target, clean_logit, baseline_logit, head_maps, source_id)
        deletion, deletion_calls = y1.run_deletion_operator(model, image, target, clean_logit, baseline_logit, attention_maps, source_id, int(spec["execution"]["deletion_forward_batch"]))
    except Exception as error:
        attempts.append({"steps": selected, "all_contracts_pass": False, "failure_phase": "intervention", "failure_code": f"{type(error).__name__}:{error}"})
        raise UnitInfrastructureError(str(error), attempts) from error
    for extra in (ablation_calls, deletion_calls):
        calls = {key: calls.get(key, 0) + extra.get(key, 0) for key in set(calls) | set(extra)}
    gpu_end.record(); torch.cuda.synchronize()
    resource = {
        "gpu_active_seconds": gpu_start.elapsed_time(gpu_end) / 1000.0,
        "wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu_started,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(image.device)),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(image.device)),
        "retained_bytes": 0, "calls": calls,
    }
    result = {
        "target_identity": {"class_id": target, "rule": "clean predicted-class raw logit"},
        "clean_logit": clean_logit, "baseline_logit": baseline_logit,
        "delta": clean_logit - baseline_logit, "selected_steps": selected,
        "closure": selected_closure, "ablations": ablations,
        "deletion_operator": deletion, "calls": calls,
        "phase_telemetry": selected_integration.phase_telemetry,
        "resource_telemetry": resource,
    }
    return result, attempts, selected_arrays


def execute(repo_root: Path | str, shard_id: int) -> dict[str, Any]:
    """Run exactly one preassigned shard; never writes outside its root."""
    import torch
    import torchvision
    import timm
    from .modeling import load_canonical_model

    repo = Path(repo_root).resolve(); shard = int(shard_id); worker_started = time.monotonic()
    if shard not in y2.SHARD_IDS: raise y2.StageYY2ScheduleError("invalid shard id")
    formal = repo / y2.ROOT; shard_root = formal / "shards" / f"shard-{shard:02d}"
    manifest = y2.read_json(shard_root / "shard_manifest.json"); y2.validate_signed(manifest)
    gate = y2.read_json(formal / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
    if gate["status"] != "PASS" or gate["payload"]["required_device_count"] != 8:
        raise y2.StageYY2ScheduleError("eight-device startup gate is not PASS")
    expected = gate["payload"]["devices"][shard]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(expected["physical_index"]) or torch.cuda.device_count() != 1:
        raise y2.StageYY2ScheduleError("worker must expose only its assigned physical GPU")
    if torch.cuda.get_device_name(0) != "NVIDIA H20": raise y2.StageYY2ScheduleError("visible device is not NVIDIA H20")
    torch.cuda.set_device(0); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    env = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "timm": timm.__version__,
        "cuda": torch.version.cuda, "driver": expected["driver"],
        "checkpoint_sha256": MODEL_SHA256, "numeric_contract_digest": NUMERIC_CONTRACT_DIGEST,
    }
    binding = y2.worker_binding({
        "physical_index": expected["physical_index"], "cuda_visible_devices": visible,
        "gpu_uuid": expected["gpu_uuid"], "pci_bus_id": expected["pci_bus_id"],
        "device_name": torch.cuda.get_device_name(0), "hostname": socket.gethostname(),
        "pid": os.getpid(), "environment_digest": y2.environment_digest(env),
    })
    if binding["environment_digest"] != expected["environment_digest"]:
        raise y2.StageYY2ScheduleError("worker environment differs from startup gate")
    shard_pending = shard_root / "completion/shard.pending.json"
    y2.write_json_new(shard_pending, y2.signed("stage_y_y2_shard_pending", "PENDING", {"shard_id": shard, "pid": os.getpid(), "terminal_published": False}))
    y2.write_json_new(shard_root / "worker_identity.json", y2.signed("stage_y_y2_worker_identity", "PASS", binding))
    if y2.sha_file(repo / MODEL_PATH) != MODEL_SHA256: raise y2.StageYY2ProvenanceError("checkpoint SHA-256 drift")
    fixed_started = time.monotonic(); fixed_cpu_started = time.process_time()
    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0")); fixed_gpu_start = torch.cuda.Event(enable_timing=True); fixed_gpu_end = torch.cuda.Event(enable_timing=True); fixed_gpu_start.record()
    model = load_canonical_model(repo / MODEL_PATH, device="cuda:0", unfused_attention=True, expected_state_dict_sha256=MODEL_SHA256, expected_model_name=MODEL_NAME, expected_model_identity=MODEL_LOADER_IDENTITY, expected_canonical_tensor_digest=MODEL_TENSOR_DIGEST).eval()
    fixed_gpu_end.record(); torch.cuda.synchronize()
    fixed_resource = {"gpu_active_seconds": fixed_gpu_start.elapsed_time(fixed_gpu_end) / 1000.0, "cpu_seconds": time.process_time() - fixed_cpu_started, "wall_seconds": time.monotonic() - fixed_started, "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(0)), "retained_bytes": (shard_root / "worker_identity.json").stat().st_size}
    spec = y2.read_json(repo / y2.RUN_SPEC); terminal_counts = {state: 0 for state in y2.UNIT_TERMINAL_STATES}
    shard_unit_count = int(manifest["payload"]["unit_count"]); shard_caps = {"gpu_hours": 8.4 if shard_unit_count == 13 else 7.7, "cpu_hours": 29.0 if shard_unit_count == 13 else 27.0, "wall_hours": 9.0 if shard_unit_count == 13 else 8.0, "storage_bytes": 1024 ** 3}
    cumulative_gpu = float(fixed_resource["gpu_active_seconds"]); cumulative_cpu = float(fixed_resource["cpu_seconds"]); cumulative_retained = int(fixed_resource["retained_bytes"])
    for row in manifest["payload"]["records"]:
        order = int(row["selection_order"]); unit_root = shard_root / "units" / f"unit-{order:03d}"
        y2.begin_unit(unit_root, shard_id=shard, selection_order=order, source_object_id_value=row["source_object_id"])
        status = "VALID"; failure_code = None; arrays_records: list[dict[str, Any]] = []; attempts: list[dict[str, Any]] = []; result: dict[str, Any] = {}
        try:
            if cumulative_gpu / 3600.0 > shard_caps["gpu_hours"] or cumulative_cpu / 3600.0 > shard_caps["cpu_hours"] or (time.monotonic() - worker_started) / 3600.0 > shard_caps["wall_hours"] or cumulative_retained > shard_caps["storage_bytes"]:
                raise y1.StageYY1BudgetError("per-shard resource cap already exhausted")
            image = _load_selected_image(repo, row, "cuda:0")
            result, attempts, arrays = _compute_unit(model, image, row, spec)
            for name, value in sorted(arrays.items()): arrays_records.append(_write_array_new(unit_root / "arrays" / f"{name}.npy", value))
            result["resource_telemetry"]["retained_bytes"] = sum(item["bytes"] for item in arrays_records)
        except UnitNumericalError as error:
            status = "FAILED_NUMERICAL"; failure_code = str(error); attempts = error.attempts
        except UnitInfrastructureError as error:
            status = "BLOCKED_INFRASTRUCTURE"; failure_code = str(error); attempts = error.attempts
        except y1.StageYY1BudgetError as error:
            status = "BLOCKED_BUDGET"; failure_code = str(error)
        except Exception as error:
            status = "BLOCKED_INFRASTRUCTURE"; failure_code = f"{type(error).__name__}:{error}"
        if attempts:
            for index, attempt in enumerate(attempts): y2.append_attempt(unit_root, {"attempt_id": f"q-{index:03d}", **attempt})
        else:
            y2.append_attempt(unit_root, {"attempt_id": "failure-000", "status": status, "failure_code": failure_code})
        attempt_records = [y2.file_record(path) for path in sorted((unit_root / "attempts").glob("attempt-*.json"))]
        resource = result.get("resource_telemetry", {"gpu_active_seconds": 0.0, "cpu_seconds": 0.0, "wall_seconds": 0.0, "peak_allocated_gpu_bytes": 0, "peak_reserved_gpu_bytes": 0, "retained_bytes": sum(item["bytes"] for item in arrays_records), "calls": {}})
        cumulative_gpu += float(resource["gpu_active_seconds"]); cumulative_cpu += float(resource["cpu_seconds"]); cumulative_retained += int(resource["retained_bytes"])
        if status == "VALID" and (cumulative_gpu / 3600.0 > shard_caps["gpu_hours"] or cumulative_cpu / 3600.0 > shard_caps["cpu_hours"] or (time.monotonic() - worker_started) / 3600.0 > shard_caps["wall_hours"] or cumulative_retained > shard_caps["storage_bytes"]):
            status = "BLOCKED_BUDGET"; failure_code = "per-shard resource cap exceeded by completed unit"
        payload = {
            "unit_digest": y2.canonical_digest({"selection_order": order, "source_object_id": row["source_object_id"]}),
            "attempt_id": attempt_records[-1]["path"], "shard_id": shard,
            "selection_order": order, "source_object_id": row["source_object_id"],
            "condition_id": "clean", "failure_code": failure_code,
            "target_identity": result.get("target_identity", {}),
            "source_binding": {"cohort_digest": manifest["payload"]["cohort_canonical_digest"], "file_sha256": row["file_sha256"]},
            "checkpoint_sha256": MODEL_SHA256, "environment_digest": binding["environment_digest"],
            "numeric_contract_digest": NUMERIC_CONTRACT_DIGEST,
            "quadrature_attempts": attempts, "resource_telemetry": resource,
            "array_records": arrays_records, "attempt_records": attempt_records,
        }
        y2.finish_unit(unit_root, status, payload); terminal_counts[status] += 1
    if terminal_counts["BLOCKED_BUDGET"]: shard_status = "BLOCKED_BUDGET"
    elif terminal_counts["BLOCKED_INFRASTRUCTURE"]: shard_status = "BLOCKED_INFRASTRUCTURE"
    elif terminal_counts["FAILED_NUMERICAL"]: shard_status = "FAILED_NUMERICAL"
    else: shard_status = "VALID"
    shard_summary = y2.signed("stage_y_y2_shard_terminal", shard_status, {"shard_id": shard, "unit_count": sum(terminal_counts.values()), "terminal_counts": terminal_counts, "worker_identity": binding, "fixed_initialization_resource": fixed_resource, "shard_elapsed_seconds": time.monotonic() - worker_started, "completed_unix_ns": time.time_ns()})
    y2.publish_terminal(shard_pending, shard_root / "completion/shard.json", shard_summary)
    return {"status": shard_status, "shard_id": shard, "terminal_counts": terminal_counts}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", type=Path, required=True); parser.add_argument("--shard-id", type=int, required=True); args = parser.parse_args()
    try: result = execute(args.repo_root, args.shard_id)
    except Exception as error:
        print(json.dumps({"status": getattr(error, "code", "BLOCKED_INFRASTRUCTURE"), "error": str(error), "next_authorization": "NONE"}, sort_keys=True)); return 2
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
