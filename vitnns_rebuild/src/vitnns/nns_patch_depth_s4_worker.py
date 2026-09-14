"""Long-lived one-visible-H20 worker for formal S4 units."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import nns_patch_depth_s4 as s4

REQUIRED_CUBLAS = ":4096:8"
RUNTIME_KEYS = (
    "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR",
    "CONDA_PKGS_DIRS", "HF_HOME", "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
    "MPLCONFIGDIR", "NUMBA_CACHE_DIR",
)


def _runtime_gate() -> dict[str, Any]:
    s4.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == REQUIRED_CUBLAS, "CUBLAS_WORKSPACE_CONFIG gate failed")
    values = {key: os.path.realpath(os.environ.get(key, "")) for key in RUNTIME_KEYS}
    s4.require(all(value.startswith("/data01/") for value in values.values()), "runtime/cache path outside /data01")
    import tempfile
    s4.require(os.path.realpath(tempfile.gettempdir()).startswith("/data01/"), "Python tempdir outside /data01")
    return {"variables": values, "python_tempdir": os.path.realpath(tempfile.gettempdir()), "cublas_workspace_config": REQUIRED_CUBLAS}


def _install_read_guard(repo: Path) -> None:
    """Fail closed on forbidden S3 outcome and S5/S6 paths in this process."""
    forbidden = (
        str((repo / "evidence/nns_vit_patch_depth/s3_r7_20260915/test_prediction_error_manifest.json").resolve()),
        str((repo / "artifacts/nns_vit_patch_depth/s3_r3_20260914").resolve()),
        str((repo / "artifacts/nns_vit_patch_depth/s3_r4_20260915").resolve()),
        str((repo / "artifacts/nns_vit_patch_depth/s3_r5_20260915").resolve()),
        str((repo / "evidence/nns_vit_patch_depth/s5").resolve()),
        str((repo / "evidence/nns_vit_patch_depth/s6").resolve()),
    )
    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        try:
            path = os.path.realpath(os.fspath(args[0]))
        except TypeError:
            return
        if any(path == item or path.startswith(item + os.sep) for item in forbidden):
            raise PermissionError("S4 command-level read whitelist rejected forbidden outcome path")
    sys.addaudithook(audit)


def _initialize_worker(repo: Path, physical_index: int, gpu_uuid: str) -> tuple[Any, dict[str, Any]]:
    runtime = _runtime_gate()
    _install_read_guard(repo)
    import torch
    import torchvision
    import timm
    from .modeling import load_canonical_model
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    s4.require(visible == str(physical_index), "worker visibility does not match physical index")
    s4.require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "NVIDIA H20", "worker must see exactly one H20")
    s4.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == REQUIRED_CUBLAS, "worker cuBLAS environment drift")
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    left = torch.arange(1024, dtype=torch.float32, device="cuda:0", requires_grad=True).reshape(32, 32)
    right = torch.arange(1024, dtype=torch.float32, device="cuda:0").reshape(32, 32)
    (left @ right).square().sum().backward()
    torch.cuda.synchronize()
    model_path = repo / s4.MODEL_REL
    s4.require(model_path.stat().st_size == s4.MODEL_BYTES, "checkpoint byte count drift")
    s4.require(s4.sha_file(model_path) == s4.MODEL_SHA256, "checkpoint SHA drift")
    model = load_canonical_model(
        model_path, device="cuda:0", unfused_attention=True,
        expected_state_dict_sha256=s4.MODEL_SHA256, expected_model_name=s4.MODEL_NAME,
        expected_model_identity=s4.MODEL_LOADER_IDENTITY,
        expected_canonical_tensor_digest=s4.MODEL_TENSOR_DIGEST,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    s4.require(not any(parameter.requires_grad for parameter in model.parameters()), "model parameters not frozen")
    environment = {
        "sys_executable_realpath": os.path.realpath(sys.executable), "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        "device_name": torch.cuda.get_device_name(0), "physical_gpu_index": physical_index,
        "gpu_uuid": gpu_uuid, "cuda_visible_devices": visible, "internal_device": "cuda:0",
        "hostname": socket.gethostname(), "pid": os.getpid(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "startup_matmul_backward": "PASS", "runtime": runtime, "model_load_count": 1,
    }
    return model, environment


def _cache_paths(payload_root: Path, index: int) -> tuple[Path, Path]:
    return (payload_root / "selected_member_cache/images" / f"{index:04d}.jpeg", payload_root / "selected_member_cache/masks" / f"{index:04d}.png")


def _preprocess(payload_root: Path, row: Mapping[str, Any], *, include_mask: bool) -> tuple[Any, np.ndarray | None, dict[str, Any]]:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf
    image_path, mask_path = _cache_paths(payload_root, int(row["schedule_index"]))
    raw_image = image_path.read_bytes()
    if len(raw_image) != int(row["image_bytes"]) or s4.sha_bytes(raw_image) != row["image_sha256"]:
        raise s4.DataIntegrityError("selected JPEG cache identity mismatch")
    raw_mask = None
    if include_mask:
        raw_mask = mask_path.read_bytes()
        if s4.sha_bytes(raw_mask) != row["annotation_sha256"]:
            raise s4.DataIntegrityError("selected mask cache identity mismatch")
    with Image.open(io.BytesIO(raw_image)) as image_source:
        image_source.load()
        if image_source.size != (int(row["image_width"]), int(row["image_height"])):
            raise s4.DataIntegrityError("selected JPEG geometry mismatch")
        rgb = image_source.convert("RGB")
        resized_rgb = tf.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        cropped_rgb = tf.center_crop(resized_rgb, [224, 224])
        rgb_tensor = tf.pil_to_tensor(cropped_rgb).float().div(255.0)
        original_size = list(rgb.size)
        resized_size = list(resized_rgb.size)
    labels = None
    if include_mask:
        assert raw_mask is not None
        with Image.open(io.BytesIO(raw_mask)) as mask_source:
            mask_source.load()
            if mask_source.size != (int(row["mask_width"]), int(row["mask_height"])):
                raise s4.DataIntegrityError("selected mask geometry mismatch")
            labels_rgb = mask_source.convert("RGB")
            resized_mask = tf.resize(labels_rgb, 248, interpolation=InterpolationMode.NEAREST)
            cropped_mask = tf.center_crop(resized_mask, [224, 224])
            mask_bytes = np.asarray(cropped_mask, dtype=np.uint16)
        if int(np.count_nonzero(mask_bytes[:, :, 2])) != 0:
            raise s4.TypedUnitError("NA_MASK_LABEL_OUT_OF_RANGE", "mask blue channel is nonzero")
        labels = np.ascontiguousarray(mask_bytes[:, :, 0] + 256 * mask_bytes[:, :, 1], dtype=np.int32)
    normalized = ((rgb_tensor.unsqueeze(0) - 0.5) / 0.5).to("cuda:0")
    left = int(round((resized_size[0] - 224) / 2.0))
    top = int(round((resized_size[1] - 224) / 2.0))
    geometry = {
        "original_size_wh": original_size, "resized_size_wh": resized_size,
        "crop_box_ltrb": [left, top, left + 224, top + 224],
        "image_interpolation": "bicubic_antialias", "mask_interpolation": "nearest",
        "same_integer_geometry": True, "image_cache_bytes_read": len(raw_image),
        "mask_cache_bytes_read": 0 if raw_mask is None else len(raw_mask),
    }
    return normalized, labels, geometry


def _occupancy(labels: np.ndarray, target_label: int) -> np.ndarray:
    if labels.shape != (224, 224):
        raise s4.TypedUnitError("NA_GEOMETRY_SIZE_MISMATCH", "mask after preprocessing is not 224x224")
    allowed = (labels == 0) | (labels == 1000) | ((labels >= 1) & (labels <= 919))
    if not bool(allowed.all()):
        raise s4.TypedUnitError("NA_MASK_LABEL_OUT_OF_RANGE", "mask contains an unsupported label")
    target = labels == int(target_label)
    if not bool(target.any()):
        raise s4.TypedUnitError("NA_TARGET_DISAPPEARED_AFTER_CROP", "target label absent after crop")
    void = labels == 1000
    if bool(void.any()):
        raise s4.TypedUnitError("NA_VOID_MASK", "void pixels remain after crop")
    planes = (labels == 0, (labels != 0) & (labels != 1000) & ~target, void, target)
    output = np.stack([plane.reshape(14, 16, 14, 16).transpose(0, 2, 1, 3).mean((2, 3), dtype=np.float64).reshape(196) for plane in planes]).astype("<f4")
    s4.require(np.array_equal(output.astype(np.float64).sum(0), np.ones(196)), "occupancy does not close to one")
    return output


def _feature_capture(model: Any, image: Any) -> tuple[Any, Any, np.ndarray]:
    import torch
    from . import stage_y_y1 as y1
    with torch.no_grad():
        native = model(image)
        captured = y1.native_forward_capture(model, image)
    features = np.stack([captured.tensors[f"b{block}.output"][0, 1:].detach().float().cpu().numpy() for block in range(12)]).astype("<f4")
    return native, captured.logits, features


def _score(matrix: np.ndarray, cls: np.ndarray) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    t = np.asarray(matrix, dtype=np.float64)
    c = np.asarray(cls, dtype=np.float64)
    s4.require(t.shape == (196, 196) and c.shape == (196,), "NNS matrix/CLS geometry drift")
    signed = t.sum(1, dtype=np.float64) + c
    positive = np.maximum(signed, 0.0)
    negative = np.minimum(signed, 0.0)
    absolute = np.abs(signed)
    absolute_total = float(absolute.sum(dtype=np.float64))
    summary = {
        "signed_mass": float(signed.sum(dtype=np.float64)), "positive_mass": float(positive.sum(dtype=np.float64)),
        "negative_mass": float(negative.sum(dtype=np.float64)), "absolute_mass": absolute_total,
        "cancellation": None if absolute_total == 0.0 else 1.0 - abs(float(signed.sum(dtype=np.float64))) / absolute_total,
        "matrix_signed_mass": float(t.sum(dtype=np.float64)), "matrix_absolute_mass": float(np.abs(t).sum(dtype=np.float64)),
        "cls_signed_mass": float(c.sum(dtype=np.float64)), "cls_absolute_mass": float(np.abs(c).sum(dtype=np.float64)),
    }
    return summary, tuple(np.ascontiguousarray(value, dtype="<f4") for value in (signed, positive, negative, absolute))


def _record_path(repo: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value["path"] = Path(value["path"]).relative_to(repo).as_posix()
    return value


def _artifact_json(repo: Path, path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document["canonical_digest"] = s4.canonical_digest(document)
    s4.write_bytes_atomic(path, json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
    return {
        "path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size,
        "sha256": s4.sha_file(path), "canonical_digest": document["canonical_digest"],
    }


def _nns_unit(repo: Path, root: Path, payload_root: Path, model: Any, row: Mapping[str, Any], claim: Mapping[str, Any], producer_commit: str) -> tuple[str, dict[str, Any]]:
    import torch
    from . import stage_y_y1 as y1
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    started_ns = time.time_ns()
    torch.cuda.reset_peak_memory_stats(0)
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    arrays: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    quadrature_calls = {"forward_calls": 0, "backward_calls": 0, "jvp_calls": 0, "jvp_direction_lanes": 0, "vjp_calls": 0, "quadrature_alpha_batches": 0}
    geometry: dict[str, Any] = {}
    outcome: dict[str, Any] = {}
    failure = None
    status = "VALID_NNS"
    try:
        image, labels, geometry = _preprocess(payload_root, row, include_mask=True)
        assert labels is not None
        occupancy = _occupancy(labels, int(row["annotation_label_id"]))
        native, hooked, features = _feature_capture(model, image)
        parity = float((native - hooked).abs().max())
        threshold = float(1e-6 + 1e-6 * native.abs().max())
        s4.require(parity <= threshold, "native/hooked logits parity failed")
        target = int(row["target_model_class_index"])
        clean_logit = float(native[0, target])
        baseline = torch.zeros_like(image)
        with torch.no_grad():
            baseline_logit = float(model(baseline)[0, target])
        delta = clean_logit - baseline_logit
        integration = materialized = closure = None
        spec = json.loads((repo / s4.SPEC_REL).read_text())
        for steps in spec["nns"]["quadrature_schedule"]:
            attempt_start = time.monotonic()
            candidate = y1.integrate_routes(model, image, baseline, target, int(steps), alpha_chunk=4, source_patch_chunk=8, component_vjp_chunk=16)
            candidate_arrays = y1.materialize_arrays(candidate)
            candidate_closure = y1.validate_closure(candidate_arrays, delta)
            for key, value in candidate.calls.items():
                quadrature_calls[key] += int(value)
            passed = candidate_closure["status"] == "PASS"
            attempts.append({
                "steps": int(steps), "status": "PASS" if passed else "FAILED_NUMERICAL",
                "input_abs_error": candidate_closure["input_abs_error"],
                "input_threshold": candidate_closure["input_threshold"],
                "head_sum_max_abs_error": candidate_closure["head_sum_max_abs_error"],
                "branch_cell_max_abs_error": candidate_closure["branch_cell_max_abs_error"],
                "scalar_map_matrix_max_abs_error": candidate_closure["scalar_map_matrix_max_abs_error"],
                "wall_seconds": time.monotonic() - attempt_start,
            })
            if passed:
                integration, materialized, closure = candidate, candidate_arrays, candidate_closure
                break
            del candidate, candidate_arrays
            torch.cuda.empty_cache()
        if integration is None or materialized is None or closure is None:
            raise s4.TypedUnitError("FAILED_NUMERICAL", "quadrature schedule exhausted")
        score_rows = []
        signed_rows = []
        positive_rows = []
        negative_rows = []
        absolute_rows = []
        for block in range(12):
            summary, values = _score(materialized[f"b{block:02d}_first_attention_matrix"], materialized[f"b{block:02d}_first_attention_cls"])
            score_rows.append({"block": block, **summary})
            signed_rows.append(values[0]); positive_rows.append(values[1]); negative_rows.append(values[2]); absolute_rows.append(values[3])
        score_arrays = np.stack((signed_rows, positive_rows, negative_rows, absolute_rows)).astype("<f4")
        positive = score_arrays[1]
        p1 = s4.p1_raw(positive, occupancy[3], row["source_image_id"])
        p2_plan = s4.build_p2_plan(positive, row["source_image_id"])
        p2_status = "MATCHED" if all(item["status"] == "MATCHED" for item in p2_plan) else "NA_MATCHED_CONTROL_UNAVAILABLE"
        p2_reason = None if p2_status == "MATCHED" else "one or more of 36 coordinates exhausted 512 matching attempts"
        unit_dir = payload_root / "units" / f"unit-{int(row['schedule_index']):04d}"
        values_to_write = [
            ("source_scores", score_arrays, "<f4", ("mass_kind_signed_positive_negative_absolute", "block", "source_patch")),
            ("occupancy", occupancy, "<f4", ("semantic_plane_background_other_void_target", "source_patch")),
            ("block_features", features, "<f4", ("block", "source_patch", "channel")),
            ("p1_permutation_indices", p1["permutation_indices"], "<i2", ("block", "repeat", "source_patch")),
            ("p1_seed_digests", p1["seed_digests"], "S64", ("block", "repeat")),
            ("p1_numerators", p1["numerators"], "<f8", ("block", "observed_then_64_permutations")),
            ("p1_denominators", p1["denominators"], "<f8", ("block",)),
            ("p1_positive_mass", p1["positive_mass"], "<f8", ("block",)),
            ("p1_raw_alignment", np.divide(p1["numerators"][:, 0], p1["denominators"], out=np.zeros(12, dtype="<f8"), where=p1["denominators"] > 0), "<f8", ("block",)),
            ("p1_permutation_alignment_draws", np.divide(p1["numerators"][:, 1:], p1["denominators"][:, None], out=np.zeros((12, 64), dtype="<f8"), where=p1["denominators"][:, None] > 0), "<f8", ("block", "repeat")),
            ("p1_valid_blocks", p1["valid_blocks"], "u1", ("block",)),
        ]
        for name, value, dtype, axes in values_to_write:
            record = s4.write_array_atomic(
                unit_dir / f"{name}.npy", value, dtype=dtype, axes=axes,
                producer_commit=producer_commit, source_identity=row["source_image_id"],
                recomputation_rule="rerun sealed S4 NNS_IMAGE on bound selected JPEG mask checkpoint and implementation commit",
            )
            arrays.append(_record_path(repo, record))
        audit_indices = set(json.loads((repo / s4.SCHEDULE_REL).read_text())["payload"]["audit_subset_schedule_indices"])
        if int(row["schedule_index"]) in audit_indices:
            matrices = np.stack([materialized[f"b{block:02d}_first_attention_matrix"] for block in range(12)]).astype("<f4")
            classes = np.stack([materialized[f"b{block:02d}_first_attention_cls"] for block in range(12)]).astype("<f4")
            for name, value, axes in (("audit_first_attention_T", matrices, ("block", "source_patch", "destination_patch")), ("audit_first_attention_C", classes, ("block", "source_patch"))):
                record = s4.write_array_atomic(unit_dir / f"{name}.npy", value, dtype="<f4", axes=axes, producer_commit=producer_commit, source_identity=row["source_image_id"], recomputation_rule="rerun sealed S4 NNS_IMAGE audit-subset retention")
                arrays.append(_record_path(repo, record))
        plan_record = _artifact_json(repo, unit_dir / "p2_coordinate_plan.json", {
            "schema_version": 1, "document_type": "s4_p2_coordinate_plan",
            "source_id": row["source_image_id"], "schedule_index": int(row["schedule_index"]),
            "status": p2_status, "reason": p2_reason, "coordinates": p2_plan,
            "selected_and_control_plan_shared_by_operators": True, "selection_used_logit_outcome": False,
            "namespace": s4.P2_NAMESPACE, "maximum_attempts": 512,
        })
        outcome = {
            "target": {"class_index": target, "annotation_label_id": int(row["annotation_label_id"]), "rule": "annotation_class_raw_logit", "fixed_across_clean_occlusion_controls": True, "clean_logit": clean_logit, "baseline_logit": baseline_logit, "delta": delta},
            "native_hooked_parity": {"max_abs_error": parity, "threshold": threshold, "status": "PASS"},
            "nns": {
                "component": "first_residual_addition_all_head_aggregate_attention_branch", "blocks": list(range(12)),
                "source_patch_formula": "sum_q_T_plus_C", "aggregation_order": "signed_sum_then_positive_part",
                "quadrature_attempts": attempts, "selected_steps": attempts[-1]["steps"], "closure": closure,
                "mass_records": score_rows, "block11_matrix_zero_mass_explicit": score_rows[11]["matrix_absolute_mass"] == 0.0,
                "block11_cls_routing_preserved": True, "full_dense_pixel_hidden_jacobian": False,
            },
            "p1": {
                "status": "VALID_RAW" if bool(p1["valid_blocks"].all()) else "PARTIAL_NA_ZERO_POSITIVE_MASS",
                "namespace": s4.P1_NAMESPACE, "permutations_per_block": 64,
                "valid_block_count": int(p1["valid_blocks"].sum()), "raw_only": True,
                "zero_positive_mass_blocks": [int(block) for block in range(12) if not bool(p1["valid_blocks"][block])],
                "cohort_level_estimate_count": 0, "p_value_count": 0, "CI_count": 0,
            },
            "p2_plan": {"status": p2_status, "coordinate_count": len(p2_plan), "record": plan_record, "outcome_used": False},
            "arrays": arrays, "model_forward_counts": {"clean_native": 1, "hooked_capture": 1, "baseline": 1, **quadrature_calls},
        }
    except s4.TypedUnitError as error:
        status = error.code
        failure = {"code": error.code, "type": type(error).__name__, "message": str(error), "failure_class": "TYPED_NA" if error.code.startswith("NA_") else "NUMERICAL"}
    except s4.DataIntegrityError as error:
        status = "BLOCKED_DATA_INTEGRITY"
        failure = {"code": status, "type": type(error).__name__, "message": str(error), "failure_class": "DATA_INTEGRITY"}
    except torch.cuda.OutOfMemoryError as error:
        status = "BLOCKED_INFRASTRUCTURE"
        failure = {"code": status, "type": type(error).__name__, "message": str(error), "failure_class": "INFRASTRUCTURE"}
    except Exception as error:
        status = "FAILED_NNS" if type(error).__module__.startswith("vitnns.stage_y_y1") else "FAILED_IMPLEMENTATION"
        failure = {"code": status, "type": type(error).__name__, "message": str(error), "failure_class": "NNS" if status == "FAILED_NNS" else "IMPLEMENTATION"}
    gpu_end.record()
    torch.cuda.synchronize()
    resources = {
        "gpu_active_seconds": gpu_start.elapsed_time(gpu_end) / 1000.0,
        "cpu_seconds": time.process_time() - started_cpu, "unit_wall_seconds": time.monotonic() - started_wall,
        "retained_bytes": sum(int(item["bytes"]) for item in arrays),
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    payload = {
        "unit_type": "NNS_IMAGE", "schedule_index": int(row["schedule_index"]),
        "source_id": row["source_image_id"], "image_sha256": row["image_sha256"],
        "mask_sha256": row["annotation_sha256"], "class_target": int(row["target_model_class_index"]),
        "annotation_label_id": int(row["annotation_label_id"]), "role_flags": row["role_flags"],
        "component": "first_residual_addition_all_head_aggregate_attention_branch", "operator": None,
        "dependency_terminal": None, "launch_id": claim["launch_id"],
        "claim_path": claim["claim_path"], "claim_time_unix_ns": claim["claimed_unix_ns"],
        "worker_id": claim["worker_id"], "worker_pid": claim["worker_pid"],
        "physical_gpu_index": claim["physical_gpu_index"], "gpu_uuid": claim["gpu_uuid"],
        "started_unix_ns": started_ns, "completed_unix_ns": time.time_ns(),
        "python_executable_realpath": os.path.realpath(sys.executable), "producer_commit": producer_commit,
        "quadrature_attempts": attempts, "quadrature_selected_steps": attempts[-1]["steps"] if attempts and attempts[-1]["status"] == "PASS" else None,
        "model_forward_counts_summary": outcome.get("model_forward_counts", {"clean_native": 0, "hooked_capture": 0, "baseline": 0, **quadrature_calls}),
        "geometry": geometry, **outcome, "resources": resources,
        "terminal_state": status, "typed_failure_or_na": failure,
        "formal_archive_member_reads_by_worker": 0, "S3_D2_prediction_error_numeric_reads": 0,
        "cohort_level_inference_count": 0, "scientific_claim_decision": None,
    }
    return status, payload


def _apply_masks(image: Any, coordinate_sets: Sequence[Sequence[int]], operator: str) -> Any:
    import torch
    output = image.repeat(len(coordinate_sets), 1, 1, 1)
    if operator == "normalized_zero":
        replacement = torch.zeros((len(coordinate_sets), 3, 1, 1), device=image.device, dtype=image.dtype)
    elif operator == "image_channel_mean":
        replacement = image.mean((2, 3), keepdim=True).repeat(len(coordinate_sets), 1, 1, 1)
    else:
        raise s4.S4Error("unregistered occlusion operator")
    for batch_index, indices in enumerate(coordinate_sets):
        for patch in indices:
            row, column = divmod(int(patch), 14)
            output[batch_index, :, row * 16:(row + 1) * 16, column * 16:(column + 1) * 16] = replacement[batch_index]
    return output


def _occlusion_unit(repo: Path, root: Path, payload_root: Path, model: Any, row: Mapping[str, Any], claim: Mapping[str, Any], producer_commit: str) -> tuple[str, dict[str, Any]]:
    import torch
    unit_type = str(claim["unit_type"])
    operator = str(claim["operator"])
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    started_ns = time.time_ns()
    torch.cuda.reset_peak_memory_stats(0)
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    nns_path = s4.terminal_path(root, "NNS_IMAGE", int(row["schedule_index"]))
    nns = json.loads(nns_path.read_text())
    s4.require(s4.verify_signed(nns), "NNS dependency digest invalid")
    arrays: list[dict[str, Any]] = []
    geometry: dict[str, Any] = {}
    failure = None
    outcome: dict[str, Any] = {}
    if nns["status"] != "VALID_NNS":
        status = "NA_UPSTREAM_NNS"
        failure = {"code": status, "upstream_state": nns["status"], "failure_class": "UPSTREAM_NNS_TYPED_TERMINAL"}
    elif nns["payload"]["p2_plan"]["status"] != "MATCHED":
        status = "NA_MATCHED_CONTROL_UNAVAILABLE"
        failure = {"code": status, "upstream_state": nns["payload"]["p2_plan"]["status"], "failure_class": "P2_TYPED_NA"}
    else:
        status = "VALID_OCCLUSION"
        try:
            plan_record = nns["payload"]["p2_plan"]["record"]
            plan_path = repo / plan_record["path"]
            s4.require(s4.sha_file(plan_path) == plan_record["sha256"], "P2 plan SHA drift")
            plan = json.loads(plan_path.read_text())
            claimed_digest = plan.pop("canonical_digest")
            s4.require(claimed_digest == s4.canonical_digest(plan), "P2 plan canonical digest drift")
            coordinate_sets = []
            coordinate_mask_digests = []
            for coordinate in plan["coordinates"]:
                sets = [coordinate["selected"]] + [control["indices"] for control in coordinate["controls"]]
                s4.require(len(sets) == 5, "P2 coordinate must contain selected plus four controls")
                coordinate_sets.extend(sets)
                coordinate_mask_digests.append(hashlib.sha256(s4.canonical_bytes({"operator": operator, "sets": sets})).hexdigest())
            s4.require(len(coordinate_sets) == 180, "P2 expected 36 coordinates times five patch sets")
            image, _, geometry = _preprocess(payload_root, row, include_mask=False)
            target = int(row["target_model_class_index"])
            with torch.no_grad():
                clean_logit = float(model(image)[0, target])
            masked = _apply_masks(image, coordinate_sets, operator)
            values = []
            calls = 1
            with torch.no_grad():
                for begin in range(0, masked.shape[0], 64):
                    logits = model(masked[begin:begin + 64])[:, target]
                    values.extend(float(item) for item in logits)
                    calls += 1
            masked_logits = np.asarray(values, dtype="<f4").reshape(36, 5)
            output_path = payload_root / "units" / f"unit-{int(row['schedule_index']):04d}" / f"occlusion_{operator}_logits.npy"
            record = s4.write_array_atomic(output_path, masked_logits, dtype="<f4", axes=("block_budget_coordinate", "selected_then_four_controls"), producer_commit=producer_commit, source_identity=row["source_image_id"], recomputation_rule=f"rerun sealed S4 {unit_type} with frozen shared coordinate plan and {operator}")
            arrays.append(_record_path(repo, record))
            outcome = {
                "target": {"class_index": target, "rule": "annotation_class_raw_logit", "fixed": True},
                "clean_logit": clean_logit, "coordinate_count": 36, "selected_plus_control_logits_per_coordinate": 5,
                "patch_ids_and_matching_diagnostics": plan_record, "mask_digests": coordinate_mask_digests,
                "matching_maximum_attempts": 512, "operator_identity": operator,
                "coordinate_plan_shared_between_operators": True, "operator_selected_from_logit_drop": False,
                "model_forward_calls": calls, "arrays": arrays,
            }
        except s4.DataIntegrityError as error:
            status = "BLOCKED_DATA_INTEGRITY"
            failure = {"code": status, "type": type(error).__name__, "message": str(error), "failure_class": "DATA_INTEGRITY"}
        except torch.cuda.OutOfMemoryError as error:
            status = "BLOCKED_INFRASTRUCTURE"
            failure = {"code": status, "type": type(error).__name__, "message": str(error), "failure_class": "INFRASTRUCTURE"}
        except Exception as error:
            status = "FAILED_OCCLUSION" if type(error).__module__.startswith("torch") else "FAILED_IMPLEMENTATION"
            failure = {"code": status, "type": type(error).__name__, "message": str(error), "failure_class": "OCCLUSION" if status == "FAILED_OCCLUSION" else "IMPLEMENTATION"}
    gpu_end.record()
    torch.cuda.synchronize()
    resources = {
        "gpu_active_seconds": gpu_start.elapsed_time(gpu_end) / 1000.0,
        "cpu_seconds": time.process_time() - started_cpu, "unit_wall_seconds": time.monotonic() - started_wall,
        "retained_bytes": sum(int(item["bytes"]) for item in arrays),
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    payload = {
        "unit_type": unit_type, "schedule_index": int(row["schedule_index"]), "source_id": row["source_image_id"],
        "image_sha256": row["image_sha256"], "mask_sha256": row["annotation_sha256"],
        "class_target": int(row["target_model_class_index"]), "annotation_label_id": int(row["annotation_label_id"]),
        "role_flags": row["role_flags"], "component": None, "operator": operator,
        "dependency_terminal": {"path": nns_path.relative_to(repo).as_posix(), "sha256": s4.sha_file(nns_path), "canonical_digest": nns["canonical_digest"], "status": nns["status"]},
        "launch_id": claim["launch_id"], "claim_path": claim["claim_path"], "claim_time_unix_ns": claim["claimed_unix_ns"],
        "worker_id": claim["worker_id"], "worker_pid": claim["worker_pid"], "physical_gpu_index": claim["physical_gpu_index"], "gpu_uuid": claim["gpu_uuid"],
        "started_unix_ns": started_ns, "completed_unix_ns": time.time_ns(), "python_executable_realpath": os.path.realpath(sys.executable), "producer_commit": producer_commit,
        "quadrature_attempts": [], "quadrature_selected_steps": None,
        "model_forward_counts_summary": {"occlusion_forward_calls": int(outcome.get("model_forward_calls", 0))},
        "geometry": geometry, **outcome, "resources": resources, "terminal_state": status, "typed_failure_or_na": failure,
        "formal_archive_member_reads_by_worker": 0, "S3_D2_prediction_error_numeric_reads": 0, "cohort_level_inference_count": 0, "scientific_claim_decision": None,
    }
    return status, payload


def _publish_source_completion(repo: Path, root: Path, row: Mapping[str, Any], producer_commit: str) -> None:
    paths = {unit_type: s4.terminal_path(root, unit_type, int(row["schedule_index"])) for unit_type in s4.UNIT_TYPES}
    if not all(path.is_file() for path in paths.values()):
        return
    terminals = {unit_type: json.loads(path.read_text()) for unit_type, path in paths.items()}
    for value in terminals.values():
        s4.require(s4.verify_signed(value), "source completion dependency digest invalid")
    destination = root / "source_completions" / f"source-{int(row['schedule_index']):04d}.json"
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    document = s4.signed("s4_source_completion", "COMPLETE", {
        "schedule_index": int(row["schedule_index"]), "source_id": row["source_image_id"],
        "image_sha256": row["image_sha256"], "mask_sha256": row["annotation_sha256"],
        "unit_terminals": [{"unit_type": unit_type, "path": paths[unit_type].relative_to(repo).as_posix(), "sha256": s4.sha_file(paths[unit_type]), "canonical_digest": terminals[unit_type]["canonical_digest"], "status": terminals[unit_type]["status"]} for unit_type in s4.UNIT_TYPES],
        "all_three_units_terminal": True, "producer_commit": producer_commit, "scientific_claim_decision": None,
    }, producer_commit=producer_commit)
    s4.publish_json_no_overwrite(temporary, destination, document)


def execute(repo_root: str, evidence_root: str, payload_root: str, launch_id: str, worker_id: str, physical_gpu_index: int, gpu_uuid: str, producer_commit: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = Path(evidence_root).resolve()
    payload = Path(payload_root).resolve()
    launch_root = root / "launches" / launch_id
    schedule = json.loads((repo / s4.SCHEDULE_REL).read_text())["payload"]["records"]
    model, environment = _initialize_worker(repo, int(physical_gpu_index), gpu_uuid)
    environment.update({"worker_id": worker_id, "launch_id": launch_id, "producer_commit": producer_commit})
    worker_root = launch_root / "workers" / worker_id
    s4.write_json_new(worker_root / "worker_identity.json", s4.signed("s4_worker_identity", "PASS", environment, producer_commit=producer_commit))
    counts: dict[str, int] = {}
    claimed: list[dict[str, Any]] = []
    started = time.monotonic()
    while True:
        claim = s4.claim_next(launch_root, root, schedule, {"worker_id": worker_id, "worker_pid": os.getpid(), "physical_gpu_index": int(physical_gpu_index), "gpu_uuid": gpu_uuid})
        if claim is None:
            break
        row = schedule[int(claim["schedule_index"])]
        if claim["unit_type"] == "NNS_IMAGE":
            status, terminal_payload = _nns_unit(repo, root, payload, model, row, claim, producer_commit)
        else:
            status, terminal_payload = _occlusion_unit(repo, root, payload, model, row, claim, producer_commit)
        terminal = s4.signed("s4_unit_terminal", status, terminal_payload, producer_commit=producer_commit)
        destination = s4.terminal_path(root, str(claim["unit_type"]), int(claim["schedule_index"]))
        temporary = destination.with_name(destination.name + f".tmp-{launch_id}-{worker_id}-{os.getpid()}-{time.time_ns()}")
        s4.publish_json_no_overwrite(temporary, destination, terminal)
        _publish_source_completion(repo, root, row, producer_commit)
        counts[status] = counts.get(status, 0) + 1
        claimed.append({"unit_type": claim["unit_type"], "schedule_index": claim["schedule_index"], "status": status})
        print(json.dumps({"event": "s4_unit_terminal", "unit_type": claim["unit_type"], "schedule_index": claim["schedule_index"], "status": status, "worker_id": worker_id, "gpu": physical_gpu_index}, sort_keys=True), flush=True)
    result = s4.signed("s4_worker_terminal", "PASS", {
        "launch_id": launch_id, "worker_id": worker_id, "worker_pid": os.getpid(),
        "physical_gpu_index": int(physical_gpu_index), "gpu_uuid": gpu_uuid,
        "claimed_units": claimed, "terminal_counts": dict(sorted(counts.items())),
        "worker_wall_seconds": time.monotonic() - started, "model_load_count": 1,
    }, producer_commit=producer_commit)
    s4.write_json_new(worker_root / "worker_terminal.json", result)
    return result
