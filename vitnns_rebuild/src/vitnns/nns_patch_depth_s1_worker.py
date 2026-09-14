"""One-visible-H20 S1 pilot worker; imported only after launcher gates."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

REQUIRED_CUBLAS = ":4096:8"
REQUIRED_RUNTIME_PREFIX = "/data01/"
RUNTIME_KEYS = ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "NUMBA_CACHE_DIR")
MODEL_PATH = Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth")
MASK_ZIP = Path("assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip")
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
MODEL_LOADER_IDENTITY = "F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES"


class DataIntegrityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    before = path.stat(); h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    after = path.stat(); require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file changed while hashing")
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {"schema_version": 1, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S1", "run_id": "s1_pilot_20260914", "document_type": document_type, "status": status, "payload": dict(payload)}
    value["canonical_digest"] = digest(value); return value


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def publish_terminal(pending: Path, terminal: Path, value: Any) -> None:
    require(pending.is_file() and not terminal.exists(), "pending to terminal transition unavailable")
    temporary = pending.with_name(pending.name + ".terminal.tmp")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, terminal); pending.unlink()
    directory = os.open(terminal.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(); h.update(str(array.dtype).encode()); h.update(b"\0"); h.update(str(tuple(array.shape)).encode()); h.update(b"\0"); h.update(array.tobytes())
    return h.hexdigest()


def write_array_new(path: Path, value: np.ndarray, axes: Sequence[str], recompute_rule: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype="<f4"); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path), "array_digest": array_digest(array), "dtype": str(array.dtype), "shape": list(array.shape), "axes": list(axes), "provenance": "S1_PILOT_ONLY", "recompute_rule": recompute_rule}


def runtime_gate() -> dict[str, Any]:
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == REQUIRED_CUBLAS, "CUBLAS_WORKSPACE_CONFIG gate failed")
    values = {key: os.path.realpath(os.environ.get(key, "")) for key in RUNTIME_KEYS}
    require(all(value.startswith(REQUIRED_RUNTIME_PREFIX) for value in values.values()), "runtime/cache path outside /data01")
    import tempfile
    require(os.path.realpath(tempfile.gettempdir()).startswith(REQUIRED_RUNTIME_PREFIX), "Python tempdir outside /data01")
    return {"variables": values, "python_tempdir": os.path.realpath(tempfile.gettempdir()), "cublas_workspace_config": REQUIRED_CUBLAS}


def preprocess_pair(repo: Path, row: Mapping[str, Any], device: str) -> tuple[Any, np.ndarray, dict[str, Any], np.ndarray]:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf

    image_path = repo / row["image_path"]
    raw_image = image_path.read_bytes()
    if not (len(raw_image) == row["image_bytes"] and sha_bytes(raw_image) == row["image_sha256"]):
        raise DataIntegrityError("pilot JPEG identity mismatch")
    with zipfile.ZipFile(repo / MASK_ZIP) as archive:
        raw_mask = archive.read(row["annotation_archive_path"])
    if sha_bytes(raw_mask) != row["annotation_sha256"]:
        raise DataIntegrityError("pilot mask identity mismatch")
    with Image.open(io.BytesIO(raw_image)) as image_source, Image.open(io.BytesIO(raw_mask)) as mask_source:
        if image_source.size != mask_source.size or image_source.size != (row["image_size"][1], row["image_size"][0]):
            raise DataIntegrityError("image/mask original size mismatch")
        rgb = image_source.convert("RGB"); labels_rgb = mask_source.convert("RGB")
        resized_rgb = tf.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        resized_mask = tf.resize(labels_rgb, 248, interpolation=InterpolationMode.NEAREST)
        require(resized_rgb.size == resized_mask.size, "image/mask resized geometry mismatch")
        left = int(round((resized_rgb.width - 224) / 2.0)); top = int(round((resized_rgb.height - 224) / 2.0))
        crop_box = [left, top, left + 224, top + 224]
        cropped_rgb = tf.center_crop(resized_rgb, [224, 224]); cropped_mask = tf.center_crop(resized_mask, [224, 224])
        rgb_tensor = tf.pil_to_tensor(cropped_rgb).float().div(255)
        mask_bytes = np.asarray(cropped_mask, dtype=np.uint16)
    labels = (mask_bytes[:, :, 0] + 256 * mask_bytes[:, :, 1]).astype(np.int32)
    blue_nonzero = int(np.count_nonzero(mask_bytes[:, :, 2]))
    require(blue_nonzero == 0, "mask blue channel nonzero")
    canonical_rgb = np.ascontiguousarray(rgb_tensor.numpy(), dtype="<f4")
    normalized = ((rgb_tensor.unsqueeze(0) - 0.5) / 0.5).to(device)
    geometry = {"original_size_wh": list(rgb.size), "resized_size_wh": list(resized_rgb.size), "crop_box_ltrb": crop_box, "image_interpolation": "bicubic_antialias", "mask_interpolation": "nearest", "same_resize_and_crop_coordinates": True, "blue_nonzero_pixels": blue_nonzero, "image_raw_bytes": len(raw_image), "mask_raw_bytes": len(raw_mask)}
    return normalized, labels, geometry, canonical_rgb


def feature_capture(model: Any, image: Any) -> tuple[Any, np.ndarray]:
    from . import stage_y_y1 as y1
    with __import__("torch").no_grad():
        native = model(image); captured = y1.native_forward_capture(model, image)
    features = np.stack([captured.tensors[f"b{block}.output"][0, 1:].detach().float().cpu().numpy() for block in range(12)]).astype("<f4")
    return (native, captured.logits, features)


def compute_unit(repo: Path, model: Any, image: Any, labels: np.ndarray, row: Mapping[str, Any], spec: Mapping[str, Any], canonical_rgb: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch
    from . import nns_patch_depth_s1 as s1
    from . import stage_y_y1 as y1

    started = time.monotonic(); cpu_started = time.process_time(); torch.cuda.reset_peak_memory_stats(0)
    gpu_started = torch.cuda.Event(enable_timing=True); gpu_ended = torch.cuda.Event(enable_timing=True); gpu_started.record()
    occupancy = s1.patch_occupancy(labels, int(row["target_annotation_label_id"]))
    native, hooked, features = feature_capture(model, image)
    parity = float((native - hooked).abs().max()); threshold = float(spec["nns"]["hooked_native_logits_atol"] + spec["nns"]["hooked_native_logits_rtol"] * native.abs().max())
    require(parity <= threshold, "native/hooked logits parity failed")
    target = int(row["target_model_class_index"]); clean_logit = float(native[0, target]); baseline = torch.zeros_like(image)
    with torch.no_grad(): baseline_logit = float(model(baseline)[0, target])
    delta = clean_logit - baseline_logit; attempts = []; integration = arrays = closure = None
    for steps in spec["nns"]["quadrature_schedule"]:
        attempt_started = time.monotonic()
        candidate = y1.integrate_routes(model, image, baseline, target, int(steps), alpha_chunk=int(spec["nns"]["alpha_chunk"]), source_patch_chunk=int(spec["nns"]["source_patch_jvp_chunk"]), component_vjp_chunk=int(spec["nns"]["component_vjp_chunk"]))
        candidate_arrays = y1.materialize_arrays(candidate); candidate_closure = y1.validate_closure(candidate_arrays, delta)
        passed = candidate_closure["status"] == "PASS"
        attempts.append({"steps": int(steps), "status": "PASS" if passed else "FAILED_NUMERICAL", "input_abs_error": candidate_closure["input_abs_error"], "input_threshold": candidate_closure["input_threshold"], "head_sum_max_abs_error": candidate_closure["head_sum_max_abs_error"], "branch_cell_max_abs_error": candidate_closure["branch_cell_max_abs_error"], "scalar_map_matrix_max_abs_error": candidate_closure["scalar_map_matrix_max_abs_error"], "wall_seconds": time.monotonic() - attempt_started})
        if passed:
            integration, arrays, closure = candidate, candidate_arrays, candidate_closure; break
    if integration is None or arrays is None or closure is None:
        raise RuntimeError("FAILED_NUMERICAL: quadrature schedule exhausted")
    signed_scores = []; positive_scores = []; negative_scores = []; absolute_scores = []; p1_rows = []; mass_rows = []
    for block in range(12):
        score = s1.source_patch_scores(arrays[f"b{block:02d}_first_attention_matrix"], arrays[f"b{block:02d}_first_attention_cls"])
        signed_scores.append(score["signed"]); positive_scores.append(score["positive"]); negative_scores.append(score["negative"]); absolute_scores.append(score["absolute"])
        mass_rows.append({key: value for key, value in score.items() if not isinstance(value, np.ndarray)} | {"block": block})
        try:
            result = s1.p1_alignment(score["positive"], occupancy["target"], row["source_image_id"], block, int(spec["p1"]["permutations"])); result["block"] = block
        except s1.S1TypedNA as error:
            result = {"block": block, "status": error.code, "reason": str(error)}
        p1_rows.append(result)
    try:
        d1 = s1.d1_contrast(p1_rows)
    except s1.S1TypedNA as error:
        d1 = {"status": error.code, "reason": str(error), "scientific_decision": None}
    p2_plans = []; candidate_sets = []
    for block in range(12):
        for budget in spec["p2"]["patch_budgets"]:
            try:
                plan = s1.topk_and_controls(positive_scores[block], row["source_image_id"], block, int(budget), int(spec["p2"]["controls_per_coordinate"]), int(spec["p2"]["control_attempts"]), 0.25)
                sets = [plan["selected"]] + [control["indices"] for control in plan["controls"]]; candidate_sets.extend(sets)
                p2_plans.append({"block": block, "budget": budget, "status": "MATCHED", **plan})
            except s1.S1TypedNA as error:
                p2_plans.append({"block": block, "budget": budget, "status": error.code, "reason": str(error)})
    operator_scores = {name: s1.operator_artifact_score(image, candidate_sets, name) for name in spec["p2"]["operator_candidates"]} if candidate_sets else {}
    operator = min(operator_scores, key=lambda name: (operator_scores[name], name)) if operator_scores else None
    forward_calls = 0
    if operator is not None:
        masked = s1.apply_patch_mask(image, candidate_sets, operator); masked_logits = []
        with torch.no_grad():
            for begin in range(0, masked.shape[0], int(spec["p2"]["forward_batch_size"])):
                masked_logits.extend(float(item) for item in model(masked[begin:begin + int(spec["p2"]["forward_batch_size"]), :])[:, target]); forward_calls += 1
        cursor = 0
        for plan in p2_plans:
            if plan["status"] != "MATCHED": continue
            count = 1 + len(plan["controls"]); values = masked_logits[cursor:cursor + count]; cursor += count
            selected_drop = clean_logit - values[0]; control_drops = [clean_logit - item for item in values[1:]]
            plan["selected_logit_drop"] = selected_drop; plan["control_logit_drops"] = control_drops; plan["matched_effect"] = selected_drop - math.fsum(control_drops) / len(control_drops); plan["target_fixed"] = target
    gpu_ended.record(); torch.cuda.synchronize()
    resources = {"gpu_active_seconds": gpu_started.elapsed_time(gpu_ended) / 1000.0, "wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu_started, "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(0)), "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(0)), "calls": {**integration.calls, "p2_batched_forward_calls": forward_calls}}
    outcome = {
        "target": {"class_index": target, "rule": "annotation_class_raw_logit", "fixed_before_after_occlusion": True, "clean_logit": clean_logit, "baseline_logit": baseline_logit, "delta": delta},
        "native_hooked_parity": {"max_abs_error": parity, "threshold": threshold, "status": "PASS"},
        "nns": {"selected_steps": attempts[-1]["steps"], "attempts": attempts, "closure": closure, "component": "first_residual_addition_all_head_aggregate_attention_branch", "source_patch_formula": "sum_q_T_plus_C", "masses": mass_rows, "block11_matrix_zero_mass_explicit": mass_rows[11]["matrix_absolute_mass"] == 0.0, "block11_cls_absolute_mass": mass_rows[11]["cls_absolute_mass"]},
        "p1": {"blocks": p1_rows, "valid_blocks": sum(item["status"] == "VALID" for item in p1_rows), "scientific_decision": None},
        "p2": {"coordinates": p2_plans, "matched_coordinates": sum(item["status"] == "MATCHED" for item in p2_plans), "operator_candidates_outcome_blind_scores": operator_scores, "selected_operator": operator, "selection_used_logit_outcome": False, "scientific_decision": None},
        "d1": d1, "probe_features": {"status": "EXTRACTED", "blocks": 12, "patches": 196, "channels": 768, "scientific_decision": None}, "resources": resources,
    }
    retained = {
        "source_signed": np.stack(signed_scores), "source_positive": np.stack(positive_scores), "source_negative": np.stack(negative_scores), "source_absolute": np.stack(absolute_scores),
        "occupancy": np.stack([occupancy[name] for name in ("background", "other_category", "void", "target")]), "block_features": features, "canonical_rgb": canonical_rgb,
    }
    for block in range(12):
        retained[f"b{block:02d}_first_attention_matrix"] = arrays[f"b{block:02d}_first_attention_matrix"]; retained[f"b{block:02d}_first_attention_cls"] = arrays[f"b{block:02d}_first_attention_cls"]
    return outcome, retained


def execute(repo_root: str, evidence_root: str, shard_id: int, worker_count: int, indices: Sequence[int], mode: str = "parallel") -> dict[str, Any]:
    runtime = runtime_gate()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", ""); require(visible and "," not in visible, "worker must see exactly one physical GPU")
    import torch, torchvision, timm
    from .modeling import load_canonical_model
    from . import nns_patch_depth_s1 as s1
    repo = Path(repo_root).resolve(); root = Path(evidence_root).resolve(); spec = json.loads((repo / "config/nns_vit_patch_depth/s1_run_spec.json").read_text()); schedule = json.loads((repo / "manifests/nns_vit_patch_depth/s1_pilot_schedule.json").read_text())["payload"]["records"]; by_index = {row["schedule_index"]: row for row in schedule}
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "NVIDIA H20", "one visible H20 required")
    torch.cuda.set_device(0); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    a = torch.arange(1024, dtype=torch.float32, device="cuda:0", requires_grad=True).reshape(32, 32); b = torch.arange(1024, dtype=torch.float32, device="cuda:0").reshape(32, 32); (a @ b).square().sum().backward(); torch.cuda.synchronize()
    environment = {"sys_executable_realpath": os.path.realpath(__import__("sys").executable), "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__, "cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(0), "physical_index": int(visible), "cuda_visible_devices": visible, "hostname": socket.gethostname(), "pid": os.getpid(), "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "runtime": runtime, "startup_matmul_backward": "PASS"}
    phase = "single_reference" if mode == "single_reference" else mode
    shard_root = root / phase / "shards" / f"shard-{shard_id:02d}"
    shard_pending = shard_root / "completion/shard.pending.json"; shard_terminal_path = shard_root / "completion/shard.json"
    if shard_terminal_path.exists():
        return json.loads(shard_terminal_path.read_text())
    write_json_new(shard_pending, signed("s1_shard_pending", "PENDING", {"shard_id": shard_id, "worker_count": worker_count, "mode": mode, "schedule_indices": list(indices)}))
    write_json_new(shard_root / "worker_identity.json", signed("s1_worker_identity", "PASS", environment))
    require(sha_file(repo / MODEL_PATH) == MODEL_SHA256, "checkpoint hash drift")
    model = load_canonical_model(repo / MODEL_PATH, device="cuda:0", unfused_attention=True, expected_state_dict_sha256=MODEL_SHA256, expected_model_name=MODEL_NAME, expected_model_identity=MODEL_LOADER_IDENTITY, expected_canonical_tensor_digest=MODEL_TENSOR_DIGEST).eval()
    counts: dict[str, int] = {}; shard_started = time.monotonic()
    for index in indices:
        row = by_index[int(index)]; unit_root = shard_root / "units" / f"unit-{index:04d}"; pending = unit_root / "completion/unit.pending.json"; terminal = unit_root / "completion/unit.json"
        if terminal.exists():
            old = json.loads(terminal.read_text()); counts[old["status"]] = counts.get(old["status"], 0) + 1; continue
        write_json_new(pending, signed("s1_unit_pending", "PENDING", {"schedule_index": index, "source_image_id": row["source_image_id"], "shard_id": shard_id, "worker_count": worker_count, "mode": mode}))
        status = "VALID"; failure = None; geometry = {}; arrays_records = []; access = {"pilot_image_bytes_read": 0, "pilot_mask_bytes_read": 0, "forbidden_split_image_bytes_read": 0, "forbidden_split_mask_bytes_read": 0, "forbidden_split_outcomes_read": 0}
        try:
            image, labels, geometry, canonical_rgb = preprocess_pair(repo, row, "cuda:0"); access["pilot_image_bytes_read"] = int(geometry["image_raw_bytes"]); access["pilot_mask_bytes_read"] = int(geometry["mask_raw_bytes"])
            outcome, retained = compute_unit(repo, model, image, labels, row, spec, canonical_rgb)
            axes = {"source_signed": ["block", "source_patch"], "source_positive": ["block", "source_patch"], "source_negative": ["block", "source_patch"], "source_absolute": ["block", "source_patch"], "occupancy": ["semantic_plane", "patch"], "block_features": ["block", "patch", "channel"], "canonical_rgb": ["rgb", "pixel_y", "pixel_x"]}
            for name, value in sorted(retained.items()):
                item_axes = axes.get(name, ["source_patch", "destination_patch"] if name.endswith("_matrix") else ["source_patch"]); item = write_array_new(unit_root / "arrays" / f"{name}.npy", value, item_axes, "rerun sealed S1 worker on bound pilot JPEG mask checkpoint and implementation commit"); item["path"] = (unit_root / "arrays" / f"{name}.npy").relative_to(repo).as_posix(); arrays_records.append(item)
            outcome["resources"]["retained_bytes"] = sum(item["bytes"] for item in arrays_records)
        except s1.S1TypedNA as error:
            status = error.code; outcome = {}; failure = {"code": error.code, "message": str(error), "failure_class": "TYPED_SCIENTIFIC_NA"}
        except DataIntegrityError as error:
            status = "BLOCKED_DATA_INTEGRITY"; outcome = {}; failure = {"code": status, "type": type(error).__name__, "message": str(error), "failure_class": "DATA_INTEGRITY"}
        except torch.cuda.OutOfMemoryError as error:
            status = "BLOCKED_INFRASTRUCTURE"; outcome = {}; failure = {"code": status, "type": type(error).__name__, "message": str(error), "failure_class": "INFRASTRUCTURE"}
        except Exception as error:
            status = "FAILED_NUMERICAL" if str(error).startswith("FAILED_NUMERICAL") else "FAILED_IMPLEMENTATION"; outcome = {}; failure = {"code": status, "type": type(error).__name__, "message": str(error), "failure_class": "NUMERICAL" if status == "FAILED_NUMERICAL" else "IMPLEMENTATION"}
        identity = {key: row[key] for key in ("schedule_index", "source_image_id", "image_sha256", "annotation_sha256", "target_model_class_index", "target_annotation_label_id")} | {"shard_id": shard_id, "worker_count": worker_count, "mode": mode}
        payload = {"identity": identity, "access": access, "geometry": geometry, **outcome, "arrays": arrays_records, "failure": failure, "terminal": {"complete": True, "immutable": True}}
        publish_terminal(pending, terminal, signed("s1_unit_terminal", status, payload)); counts[status] = counts.get(status, 0) + 1
        print(json.dumps({"event": "unit_terminal", "schedule_index": index, "status": status, "shard_id": shard_id, "mode": mode}), flush=True)
    shard_terminal = signed("s1_shard_terminal", "PASS" if counts.get("FAILED_IMPLEMENTATION", 0) == 0 else "FAILED_IMPLEMENTATION", {"shard_id": shard_id, "worker_count": worker_count, "mode": mode, "schedule_indices": list(indices), "terminal_counts": dict(sorted(counts.items())), "worker_wall_seconds": time.monotonic() - shard_started})
    publish_terminal(shard_pending, shard_terminal_path, shard_terminal)
    return shard_terminal
