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

from . import stage_y_y2 as y2

# This check is intentionally before every Torch/torchvision/timm import and
# before any JPEG byte is read.  The launcher owns environment initialization.
if os.environ.get("STAGE_Y2_R1_RECOVERY") == "1":
    y2.activate_recovery_contract()

MODEL_PATH = Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth")
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
MODEL_LOADER_IDENTITY = "F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES"
NUMERIC_CONTRACT_DIGEST = "stage-y-y1-grouped-jvp-vjp-frozen-v1"


class UnitNumericalError(y2.StageYY2Error):
    code = "FAILED_NUMERICAL"
    def __init__(self, message: str, attempts: list[dict[str, Any]], calls: Mapping[str, int]):
        super().__init__(message); self.attempts = attempts; self.calls = dict(calls)


class UnitInfrastructureError(y2.StageYY2Error):
    code = "BLOCKED_INFRASTRUCTURE"
    def __init__(self, message: str, attempts: list[dict[str, Any]], calls: Mapping[str, int]):
        super().__init__(message); self.attempts = attempts; self.calls = dict(calls)


def _write_array_new(path: Path, value: np.ndarray) -> dict[str, Any]:
    from . import stage_y_y1 as y1
    from .stage_y_y1_worker import _axes

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value, dtype="<f4")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    record = y2.file_record(path)
    return {
        **record, "canonical_digest": y1.array_digest(array),
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
    from . import stage_y_y1 as y1
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
            raise UnitInfrastructureError(str(error), attempts, calls) from error
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
                raise UnitInfrastructureError(str(error), attempts, calls) from error
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
        raise UnitNumericalError("Y2 clean unit nonconverged at 8192", attempts, calls)
    head_maps = {block: selected_arrays[f"b{block:02d}_first_heads_cell"] for block in range(12)}
    attention_maps = {block: selected_arrays[f"b{block:02d}_first_attention_cell"] for block in range(12)}
    try:
        ablations, ablation_calls = y1.run_head_ablations(model, image, target, clean_logit, baseline_logit, head_maps, source_id)
        deletion, deletion_calls = y1.run_deletion_operator(model, image, target, clean_logit, baseline_logit, attention_maps, source_id, int(spec["execution"]["deletion_forward_batch"]))
    except Exception as error:
        attempts.append({"steps": selected, "all_contracts_pass": False, "failure_phase": "intervention", "failure_code": f"{type(error).__name__}:{error}"})
        raise UnitInfrastructureError(str(error), attempts, calls) from error
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
    repo = Path(repo_root).resolve(); shard = int(shard_id); worker_started = time.monotonic()
    if y2.RECOVERY_MODE and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise y2.StageYY2ScheduleError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8 before Torch initialization")
    formal = repo / y2.ROOT; shard_root = formal / "shards" / f"shard-{shard:02d}"
    manifest = y2.read_json(shard_root / "shard_manifest.json"); y2.validate_signed(manifest)
    gate = y2.read_json(formal / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
    if os.environ.get("STAGE_Y2_LAUNCH_AUTHORIZATION") != gate["canonical_digest"]:
        raise y2.StageYY2ScheduleError("worker was not launched by the sealed Y2 launcher")
    worker_count = y2.validate_worker_count(gate["payload"]["locked_worker_count"])
    if gate["status"] != "PASS" or shard not in y2.shard_ids(worker_count):
        raise y2.StageYY2ScheduleError("locked startup gate does not authorize this shard")
    if manifest["payload"]["shard_count"] != worker_count:
        raise y2.StageYY2ScheduleError("manifest worker count differs from startup gate")
    expected = gate["payload"]["devices"][shard]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(expected["physical_index"]):
        raise y2.StageYY2ScheduleError("worker visibility differs from its sealed physical GPU")
    shard_pending = shard_root / "completion/shard.pending.json"
    if not shard_pending.is_file() or (shard_root / "completion/shard.json").exists():
        raise y2.StageYY2ProvenanceError("launcher-owned shard PENDING receipt is absent or already terminal")
    if y2.RECOVERY_MODE and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise y2.StageYY2ScheduleError("cuBLAS workspace configuration missing before Torch import")

    # Heavy framework imports and all CUDA calls occur only after the sealed
    # launcher authorization and immutable device assignment have passed.
    import torch
    import torchvision
    import timm
    from . import stage_y_y1 as y1
    from .modeling import load_canonical_model

    if visible != str(expected["physical_index"]) or torch.cuda.device_count() != 1:
        raise y2.StageYY2ScheduleError("worker must expose only its assigned physical GPU")
    if torch.cuda.get_device_name(0) != "NVIDIA H20": raise y2.StageYY2ScheduleError("visible device is not NVIDIA H20")
    torch.cuda.set_device(0); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    env = y2.build_environment_identity(
        torch, torchvision, timm, driver=expected["driver"],
        checkpoint_sha256=MODEL_SHA256,
        numeric_contract_digest=NUMERIC_CONTRACT_DIGEST,
    )
    binding = y2.worker_binding({
        "physical_index": expected["physical_index"], "cuda_visible_devices": visible,
        "gpu_uuid": expected["gpu_uuid"], "pci_bus_id": expected["pci_bus_id"],
        "device_name": torch.cuda.get_device_name(0), "hostname": socket.gethostname(),
        "pid": os.getpid(), "environment_digest": y2.environment_digest(env),
        **({"environment_identity": env} if y2.RECOVERY_MODE else {}),
    })
    if binding["environment_digest"] != expected["environment_digest"]:
        raise y2.StageYY2ScheduleError("worker environment differs from startup gate")
    y2.write_json_new(shard_root / "worker_identity.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_worker_identity", "PASS", binding))
    if y2.sha_file(repo / MODEL_PATH) != MODEL_SHA256: raise y2.StageYY2ProvenanceError("checkpoint SHA-256 drift")
    fixed_started = time.monotonic(); fixed_cpu_started = time.process_time()
    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0")); fixed_gpu_start = torch.cuda.Event(enable_timing=True); fixed_gpu_end = torch.cuda.Event(enable_timing=True); fixed_gpu_start.record()
    model = load_canonical_model(repo / MODEL_PATH, device="cuda:0", unfused_attention=True, expected_state_dict_sha256=MODEL_SHA256, expected_model_name=MODEL_NAME, expected_model_identity=MODEL_LOADER_IDENTITY, expected_canonical_tensor_digest=MODEL_TENSOR_DIGEST).eval()
    fixed_gpu_end.record(); torch.cuda.synchronize()
    fixed_resource = {"gpu_active_seconds": fixed_gpu_start.elapsed_time(fixed_gpu_end) / 1000.0, "cpu_seconds": time.process_time() - fixed_cpu_started, "wall_seconds": time.monotonic() - fixed_started, "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(0)), "retained_bytes": (shard_root / "worker_identity.json").stat().st_size}
    spec = y2.read_json(repo / y2.RUN_SPEC); terminal_counts = {state: 0 for state in y2.UNIT_TERMINAL_STATES}
    shard_unit_count = int(manifest["payload"]["unit_count"]); sealed_caps = manifest["payload"]["resource_caps"]
    shard_caps = {"gpu_hours": float(sealed_caps["cap_h20_gpu_hours"]), "cpu_hours": float(sealed_caps["cap_cpu_hours"]), "wall_hours": float(sealed_caps["cap_worker_wall_hours"]), "storage_bytes": int(float(sealed_caps["cap_storage_gib"]) * 1024 ** 3)}
    cumulative_gpu = float(fixed_resource["gpu_active_seconds"]); cumulative_cpu = float(fixed_resource["cpu_seconds"]); cumulative_retained = int(fixed_resource["retained_bytes"])
    for row in manifest["payload"]["records"]:
        order = int(row["selection_order"]); unit_root = shard_root / "units" / f"unit-{order:03d}"
        y2.begin_unit(unit_root, shard_id=shard, selection_order=order, source_object_id_value=row["source_object_id"], worker_count=worker_count)
        status = "VALID"; failure_code = None; arrays_records: list[dict[str, Any]] = []; attempts: list[dict[str, Any]] = []; result: dict[str, Any] = {}
        failure_calls = {"forward_calls": 0, "backward_calls": 0, "jvp_calls": 0, "jvp_direction_lanes": 0, "vjp_calls": 0, "quadrature_alpha_batches": 0}
        unit_started = time.monotonic(); unit_cpu_started = time.process_time()
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0")); unit_gpu_start = torch.cuda.Event(enable_timing=True); unit_gpu_end = torch.cuda.Event(enable_timing=True); unit_gpu_start.record()
        try:
            if cumulative_gpu / 3600.0 > shard_caps["gpu_hours"] or cumulative_cpu / 3600.0 > shard_caps["cpu_hours"] or (time.monotonic() - worker_started) / 3600.0 > shard_caps["wall_hours"] or cumulative_retained > shard_caps["storage_bytes"]:
                raise y1.StageYY1BudgetError("per-shard resource cap already exhausted")
            image = _load_selected_image(repo, row, "cuda:0")
            result, attempts, arrays = _compute_unit(model, image, row, spec)
            for name, value in sorted(arrays.items()): arrays_records.append(_write_array_new(unit_root / "arrays" / f"{name}.npy", value))
            result["resource_telemetry"]["retained_bytes"] = sum(item["bytes"] for item in arrays_records)
        except UnitNumericalError as error:
            status = "FAILED_NUMERICAL"; failure_code = str(error); attempts = error.attempts; failure_calls = error.calls
        except UnitInfrastructureError as error:
            status = "BLOCKED_INFRASTRUCTURE"; failure_code = str(error); attempts = error.attempts; failure_calls = error.calls
        except y1.StageYY1BudgetError as error:
            status = "BLOCKED_BUDGET"; failure_code = str(error)
        except Exception as error:
            status = "BLOCKED_INFRASTRUCTURE"; failure_code = f"{type(error).__name__}:{error}"
        if attempts:
            for index, attempt in enumerate(attempts): y2.append_attempt(unit_root, {"attempt_id": f"q-{index:03d}", **attempt})
        else:
            y2.append_attempt(unit_root, {"attempt_id": "failure-000", "status": status, "failure_code": failure_code})
        attempt_records = [y2.file_record(path) for path in sorted((unit_root / "attempts").glob("attempt-*.json"))]
        final_attempt = y2.read_json(Path(attempt_records[-1]["path"]))["payload"]["attempt_id"]
        unit_gpu_end.record(); torch.cuda.synchronize()
        resource = {
            "gpu_active_seconds": unit_gpu_start.elapsed_time(unit_gpu_end) / 1000.0,
            "cpu_seconds": time.process_time() - unit_cpu_started,
            "wall_seconds": time.monotonic() - unit_started,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(0)),
            "retained_bytes": sum(path.stat().st_size for path in unit_root.rglob("*") if path.is_file()),
            "calls": result.get("calls", failure_calls),
        }
        cumulative_gpu += float(resource["gpu_active_seconds"]); cumulative_cpu += float(resource["cpu_seconds"]); cumulative_retained += int(resource["retained_bytes"])
        if status == "VALID" and (cumulative_gpu / 3600.0 > shard_caps["gpu_hours"] or cumulative_cpu / 3600.0 > shard_caps["cpu_hours"] or (time.monotonic() - worker_started) / 3600.0 > shard_caps["wall_hours"] or cumulative_retained > shard_caps["storage_bytes"]):
            status = "BLOCKED_BUDGET"; failure_code = "per-shard resource cap exceeded by completed unit"
        scientific_record = None
        if status == "VALID":
            scientific_payload = {
                key: result[key] for key in ("target_identity", "clean_logit", "baseline_logit", "delta", "selected_steps", "closure", "ablations", "deletion_operator", "calls", "phase_telemetry")
            }
            scientific_path = unit_root / "scientific_result.json"
            y2.write_json_new(scientific_path, y2.signed(f"{y2.DOCUMENT_PREFIX}_unit_scientific_result", "VALID", scientific_payload))
            scientific_record = y2.file_record(scientific_path)
        payload = {
            "unit_digest": y2.canonical_digest({"selection_order": order, "source_object_id": row["source_object_id"]}),
            "attempt_id": final_attempt, "shard_id": shard,
            "selection_order": order, "source_object_id": row["source_object_id"],
            "condition_id": "clean", "failure_code": failure_code,
            "target_identity": result.get("target_identity", {}),
            "source_binding": {"cohort_digest": manifest["payload"]["cohort_canonical_digest"], "file_sha256": row["file_sha256"]},
            "checkpoint_sha256": MODEL_SHA256, "environment_digest": binding["environment_digest"],
            "numeric_contract_digest": NUMERIC_CONTRACT_DIGEST,
            "quadrature_attempts": attempts, "resource_telemetry": resource,
            "scientific_record": scientific_record, "array_records": arrays_records, "attempt_records": attempt_records,
        }
        if y2.RECOVERY_MODE:
            original_path = y2.ORIGINAL_ROOT / "shards" / f"shard-{order % 4:02d}" / "units" / f"unit-{order:03d}" / "completion/unit.json"
            original_terminal = y2.read_json(repo / original_path)
            payload["retry_identity"] = {
                "supersedes_original_y2_unit": True,
                "original_y2_unit_path": original_path.as_posix(),
                "original_attempt_digest": original_terminal["payload"]["attempt_records"][-1]["canonical_digest"],
                "original_terminal_digest": original_terminal["canonical_digest"],
                "original_failure_code": original_terminal["payload"]["failure_code"],
                "retry_reason": "CUBLAS_WORKSPACE_CONFIG_MISSING_BEFORE_TORCH_INITIALIZATION",
                "retry_index": 1,
                "scientific_scope_changed": False,
                "target_metric_tolerance_changed": False,
            }
        y2.finish_unit(unit_root, status, payload, worker_count=worker_count); terminal_counts[status] += 1
    if terminal_counts["BLOCKED_BUDGET"]: shard_status = "BLOCKED_BUDGET"
    elif terminal_counts["BLOCKED_INFRASTRUCTURE"]: shard_status = "BLOCKED_INFRASTRUCTURE"
    elif terminal_counts["FAILED_NUMERICAL"]: shard_status = "FAILED_NUMERICAL"
    else: shard_status = "VALID"
    shard_summary = y2.signed(f"{y2.DOCUMENT_PREFIX}_shard_terminal", shard_status, {"shard_id": shard, "shard_count": worker_count, "unit_count": sum(terminal_counts.values()), "terminal_counts": terminal_counts, "worker_identity": binding, "fixed_initialization_resource": fixed_resource, "resource_caps": sealed_caps, "shard_elapsed_seconds": time.monotonic() - worker_started, "completed_unix_ns": time.time_ns()})
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
