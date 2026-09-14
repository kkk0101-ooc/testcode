"""One-visible-H20 long-lived representation worker for S3."""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
from pathlib import Path
import socket
import sys
import tarfile
import time
from typing import Any, Mapping
import zipfile

import numpy as np

from . import nns_patch_depth_s3 as s3

REQUIRED_CUBLAS = ":4096:8"
MODEL_PATH = Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth")
MASK_ZIP = Path("assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip")
JPEG_TAR = Path("assets/imagenet/ILSVRC2012_img_val.tar")
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_IDENTITY = "F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES"
RUNTIME_KEYS = ("TMPDIR","TMP","TEMP","XDG_CACHE_HOME","PIP_CACHE_DIR","CONDA_PKGS_DIRS","HF_HOME","HF_DATASETS_CACHE","TRANSFORMERS_CACHE","TORCH_HOME","TORCH_EXTENSIONS_DIR","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","CUDA_CACHE_PATH","MPLCONFIGDIR","NUMBA_CACHE_DIR")


def runtime_gate() -> dict[str, Any]:
    s3.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == REQUIRED_CUBLAS, "CUBLAS_WORKSPACE_CONFIG must be set before worker import")
    values = {key: os.path.realpath(os.environ.get(key, "")) for key in RUNTIME_KEYS}
    s3.require(all(value.startswith("/data01/") for value in values.values()), "runtime path outside /data01")
    return {"cublas_workspace_config": REQUIRED_CUBLAS, "paths": values}


def initialize(repo: Path, physical_index: int, gpu_uuid: str) -> tuple[Any, list[Any], dict[str, Any], list[Any]]:
    runtime = runtime_gate()
    import timm
    import torch
    import torchvision
    from .modeling import load_canonical_model

    s3.require(torch.cuda.device_count() == 1, "worker must see exactly one GPU")
    s3.require(torch.cuda.get_device_name(0) == "NVIDIA H20", "worker visible device is not NVIDIA H20")
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
    s3.require(s3.sha_file(repo / MODEL_PATH) == s3.MODEL_SHA256, "checkpoint SHA-256 drift")
    model = load_canonical_model(
        repo / MODEL_PATH, device="cuda:0", unfused_attention=False,
        expected_state_dict_sha256=s3.MODEL_SHA256, expected_model_name=MODEL_NAME,
        expected_model_identity=MODEL_IDENTITY, expected_canonical_tensor_digest=s3.MODEL_TENSOR_DIGEST,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured: list[Any] = [None] * 12
    handles = []
    for block_index, block in enumerate(model.blocks):
        def hook(_module: Any, _arguments: Any, output: Any, index: int = block_index) -> None:
            captured[index] = output.detach()
        handles.append(block.register_forward_hook(hook))
    identity = {
        "physical_gpu_index": physical_index, "gpu_uuid": gpu_uuid, "visible_device_count": torch.cuda.device_count(),
        "visible_device_name": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(), "hostname": socket.gethostname(), "python_executable_realpath": os.path.realpath(sys.executable),
        "python_version": platform.python_version(), "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__, "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "startup_matmul_backward": "PASS",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "runtime": runtime,
    }
    return model, captured, identity, handles


def read_pair(repo: Path, row: Mapping[str, Any], mask_archive: zipfile.ZipFile, jpeg_archive: tarfile.TarFile | None) -> tuple[bytes, bytes, dict[str, int]]:
    split = str(row["split"])
    if split == "d2_test":
        s3.require(jpeg_archive is not None, "D2 JPEG archive unavailable")
        member = jpeg_archive.getmember(str(row["image_archive_member"]))
        stream = jpeg_archive.extractfile(member)
        s3.require(stream is not None, "D2 JPEG member is not a regular file")
        raw_image = stream.read()
        access = {"image_member_read_count": 1, "mask_member_read_count": 1, "decode_count": 0, "feature_extraction_count": 0, "prediction_count": 0}
    else:
        raw_image = (repo / str(row["image_local_path"])).read_bytes()
        access = {"image_member_read_count": 0, "mask_member_read_count": 0, "decode_count": 0, "feature_extraction_count": 0, "prediction_count": 0}
    raw_mask = mask_archive.read(str(row["annotation_archive_path"]))
    access["mask_member_read_count"] = 1
    if len(raw_image) != int(row["image_bytes"]) or s3.sha_bytes(raw_image) != row["image_sha256"]:
        raise s3.DataIntegrityError("JPEG identity mismatch")
    if s3.sha_bytes(raw_mask) != row["annotation_sha256"]:
        raise s3.DataIntegrityError("mask identity mismatch")
    return raw_image, raw_mask, access


def preprocess(raw_image: bytes, raw_mask: bytes, row: Mapping[str, Any]) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf

    with Image.open(io.BytesIO(raw_image)) as image_source, Image.open(io.BytesIO(raw_mask)) as mask_source:
        expected = (int(row["image_width"]), int(row["image_height"]))
        if image_source.size != mask_source.size or image_source.size != expected or mask_source.size != (int(row["mask_width"]), int(row["mask_height"])):
            raise s3.TypedUnitError("NA_GEOMETRY_SIZE_MISMATCH", "original image/mask geometry mismatch")
        rgb = image_source.convert("RGB")
        mask_rgb = mask_source.convert("RGB")
        resized_rgb = tf.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        resized_mask = tf.resize(mask_rgb, 248, interpolation=InterpolationMode.NEAREST)
        if resized_rgb.size != resized_mask.size:
            raise s3.TypedUnitError("NA_GEOMETRY_SIZE_MISMATCH", "resized image/mask geometry mismatch")
        left = int(round((resized_rgb.width - 224) / 2.0))
        top = int(round((resized_rgb.height - 224) / 2.0))
        rgb_tensor = tf.pil_to_tensor(tf.center_crop(resized_rgb, [224, 224])).float().div(255.0)
        mask_array = np.asarray(tf.center_crop(resized_mask, [224, 224]), dtype=np.uint16)
    if int(np.count_nonzero(mask_array[:, :, 2])) != 0:
        raise s3.TypedUnitError("NA_MASK_LABEL_OUT_OF_RANGE", "mask blue channel is nonzero")
    labels = (mask_array[:, :, 0] + 256 * mask_array[:, :, 1]).astype(np.int32)
    rgb_array = np.ascontiguousarray(rgb_tensor.numpy(), dtype="<f4")
    normalized = ((rgb_tensor.unsqueeze(0) - 0.5) / 0.5).to("cuda:0")
    geometry = {"original_size_wh": list(expected), "resized_size_wh": list(resized_rgb.size), "crop_box_ltrb": [left, top, left + 224, top + 224], "shared_integer_geometry": True}
    return normalized, labels, rgb_array, geometry


def capture_features(model: Any, captured: list[Any], image: Any) -> tuple[np.ndarray, dict[str, float]]:
    import torch
    started = time.monotonic()
    cpu_started = time.process_time()
    torch.cuda.reset_peak_memory_stats(0)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for index in range(12):
        captured[index] = None
    begin.record()
    with torch.inference_mode():
        model(image)
    end.record()
    torch.cuda.synchronize()
    s3.require(all(item is not None for item in captured), "not all block outputs were captured")
    features = np.stack([item[0, 1:, :].float().cpu().numpy() for item in captured]).astype("<f4")
    s3.require(features.shape == (12, 196, 768) and np.isfinite(features).all(), "feature array contract failed")
    return features, {
        "gpu_active_seconds": begin.elapsed_time(end) / 1000.0, "cpu_seconds": time.process_time() - cpu_started,
        "unit_wall_seconds": time.monotonic() - started, "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def load_class_weights(repo: Path, wnid: str) -> dict[str, np.ndarray]:
    path = repo / s3.PAYLOAD_REL / "weights" / f"{wnid}.npz"
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def predictions(features: np.ndarray, control: np.ndarray, occupancy: np.ndarray, bundle: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    target = np.asarray(occupancy[3], dtype=np.float64)
    feature_prediction = np.empty((12, 196), dtype="<f4")
    feature_mse = np.empty(12, dtype="<f8")
    for block in range(12):
        model = s3.RidgeModel(np.asarray(bundle["feature_mean"])[block], np.asarray(bundle["feature_scale"])[block], np.asarray(bundle["feature_weight"])[block], float(np.asarray(bundle["feature_intercept"])[block]), float(np.asarray(bundle["feature_alpha"])[block]))
        predicted = s3.predict_ridge(model, features[block])
        feature_prediction[block] = predicted.astype("<f4")
        feature_mse[block] = np.mean((predicted - target) ** 2, dtype=np.float64)
    control_model = s3.RidgeModel(np.asarray(bundle["control_mean"]), np.asarray(bundle["control_scale"]), np.asarray(bundle["control_weight"]), float(np.asarray(bundle["control_intercept"])), float(np.asarray(bundle["control_alpha"])))
    control_pred64 = s3.predict_ridge(control_model, control)
    return {"feature_prediction": feature_prediction, "control_prediction": control_pred64.astype("<f4"), "feature_mse": feature_mse, "control_mse": np.asarray(np.mean((control_pred64 - target) ** 2, dtype=np.float64), dtype="<f8")}


def claim_next(repo: Path, launch_root: Path, schedule: list[Mapping[str, Any]], phase: str, worker: Mapping[str, Any], cursor: int) -> tuple[Mapping[str, Any], Mapping[str, Any], int] | None:
    allowed = {"probe_train", "probe_validation"} if phase == "train_validation" else {"d2_test"}
    terminal_root = repo / s3.ROOT_REL / "unit_terminals"
    for position in range(cursor, len(schedule)):
        row = schedule[position]
        if row["split"] not in allowed:
            continue
        index = int(row["schedule_index"])
        terminal = terminal_root / f"unit-{index:05d}.json"
        if terminal.exists():
            document = json.loads(terminal.read_text())
            s3.require(document["payload"]["source_image_id"] == row["source_image_id"], "existing terminal source identity mismatch")
            continue
        claim = launch_root / "claims" / f"unit-{index:05d}.json"
        claim.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schedule_index": index, "source_image_id": row["source_image_id"], "worker_id": worker["worker_id"], "worker_pid": os.getpid(), "claim_time_ns": time.time_ns()}
        try:
            s3.write_json_new(claim, s3.signed("s3_unit_claim", "CLAIMED", payload, producer_commit=worker["producer_commit"]))
        except FileExistsError:
            continue
        return row, {**payload, "claim_path": claim.relative_to(repo).as_posix()}, position + 1
    return None


def execute_unit(repo: Path, model: Any, captured: list[Any], row: Mapping[str, Any], claim: Mapping[str, Any], worker: Mapping[str, Any], mask_archive: zipfile.ZipFile, jpeg_archive: tarfile.TarFile | None, weight_cache: dict[str, dict[str, np.ndarray]], typed_na_classes: set[str]) -> str:
    index = int(row["schedule_index"])
    terminal_path = repo / s3.ROOT_REL / "unit_terminals" / f"unit-{index:05d}.json"
    pending_path = repo / s3.ROOT_REL / "unit_pending" / worker["launch_id"] / f"unit-{index:05d}.json"
    s3.write_json_new(pending_path, s3.signed("s3_unit_pending", "PENDING", {"schedule_index": index, "source_image_id": row["source_image_id"], "claim": dict(claim)}, producer_commit=worker["producer_commit"]))
    started_ns = time.time_ns()
    arrays: list[dict[str, Any]] = []
    state = "VALID_FEATURE"
    reason = None
    access = {"image_member_read_count": 0, "mask_member_read_count": 0, "decode_count": 0, "feature_extraction_count": 0, "prediction_count": 0}
    resources = {"gpu_active_seconds": 0.0, "cpu_seconds": 0.0, "unit_wall_seconds": 0.0, "retained_bytes": 0, "peak_allocated_gpu_bytes": 0, "peak_reserved_gpu_bytes": 0}
    unit_started = time.monotonic()
    unit_cpu_started = time.process_time()
    try:
        if row["split"] == "d2_test" and str(row["annotation_wnid"]) in typed_na_classes:
            raise s3.TypedUnitError("NA_D2_REQUIRED_SPLIT_OR_MODEL_MISSING", "class lacks all six VALID train and two VALID validation units; D2 bytes not read")
        raw_image, raw_mask, access = read_pair(repo, row, mask_archive, jpeg_archive)
        image, labels, rgb, geometry = preprocess(raw_image, raw_mask, row)
        access["decode_count"] = 1
        occupancy = s3.patch_occupancy(labels, int(row["annotation_label_id"]))
        control = s3.low_level_features(rgb)
        features, measured = capture_features(model, captured, image)
        access["feature_extraction_count"] = 1
        resources.update(measured)
        array_root = repo / s3.PAYLOAD_REL / "units" / f"unit-{index:05d}"
        rule = "rerun sealed S3 worker on bound source JPEG mask checkpoint preprocessing and implementation commit"
        arrays.append(s3.write_array_atomic(array_root / "block_features.npy", features, dtype="<f4", axes=["block","patch","channel"], producer_commit=worker["producer_commit"], source_identity=row["source_image_id"], recomputation_rule=rule))
        arrays.append(s3.write_array_atomic(array_root / "occupancy.npy", occupancy, dtype="<f4", axes=["semantic_plane","patch"], producer_commit=worker["producer_commit"], source_identity=row["source_image_id"], recomputation_rule=rule))
        arrays.append(s3.write_array_atomic(array_root / "low_level_features.npy", control, dtype="<f4", axes=["patch","feature"], producer_commit=worker["producer_commit"], source_identity=row["source_image_id"], recomputation_rule=rule))
        if row["split"] == "d2_test":
            wnid = str(row["annotation_wnid"])
            if wnid not in weight_cache:
                weight_cache[wnid] = load_class_weights(repo, wnid)
            result = predictions(features, control, occupancy, weight_cache[wnid])
            access["prediction_count"] = 1
            for name, dtype, axes in (("feature_prediction","<f4",["block","patch"]),("control_prediction","<f4",["patch"]),("feature_mse","<f8",["block"]),("control_mse","<f8",[])):
                arrays.append(s3.write_array_atomic(array_root / f"{name}.npy", result[name], dtype=dtype, axes=axes, producer_commit=worker["producer_commit"], source_identity=row["source_image_id"], recomputation_rule="apply sealed class weights and normalization to sealed S3 feature arrays without cohort aggregation"))
        for array_record in arrays:
            array_record["path"] = Path(array_record["path"]).relative_to(repo).as_posix()
        resources["retained_bytes"] = sum(int(record["bytes"]) for record in arrays)
    except s3.TypedUnitError as error:
        state, reason = error.code, str(error)
    except s3.DataIntegrityError as error:
        state, reason = "BLOCKED_DATA_INTEGRITY", str(error)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        state, reason = "FAILED_NUMERICAL", repr(error)
    except Exception as error:
        state, reason = "FAILED_FEATURE_EXTRACTION", repr(error)
    resources["unit_wall_seconds"] = max(resources["unit_wall_seconds"], time.monotonic() - unit_started)
    resources["cpu_seconds"] = max(resources["cpu_seconds"], time.process_time() - unit_cpu_started)
    payload = {
        "schedule_index": index, "source_image_id": row["source_image_id"], "image_sha256": row["image_sha256"], "annotation_sha256": row["annotation_sha256"],
        "split": row["split"], "annotation_wnid": row["annotation_wnid"], "target_model_class_index": int(row["target_model_class_index"]), "role_flags": row["role_flags"],
        "launch_id": worker["launch_id"], "claim_path": claim["claim_path"], "claim_time_ns": int(claim["claim_time_ns"]), "worker_id": worker["worker_id"],
        "worker_pid": os.getpid(), "physical_gpu_index": int(worker["physical_gpu_index"]), "gpu_uuid": worker["gpu_uuid"], "started_time_ns": started_ns,
        "completed_time_ns": time.time_ns(), "python_executable_realpath": os.path.realpath(sys.executable), "producer_commit": worker["producer_commit"],
        "arrays": arrays, "resources": resources, "terminal_state": state, "failure_reason": reason, "access": access,
    }
    terminal_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = terminal_path.with_name(terminal_path.name + f".tmp-{worker['launch_id']}-{worker['worker_id']}-{os.getpid()}")
    s3.write_json_new(temporary, s3.signed("s3_unit_terminal", state, payload, producer_commit=worker["producer_commit"]))
    try:
        os.link(temporary, terminal_path)
        s3.fsync_directory(terminal_path.parent)
    except FileExistsError:
        raise s3.S3Error(f"duplicate immutable terminal attempt for {index}")
    finally:
        if temporary.exists():
            temporary.unlink()
            s3.fsync_directory(temporary.parent)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--launch-root", type=Path, required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--phase", choices=("train_validation","test"), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--producer-commit", required=True)
    arguments = parser.parse_args()
    repo = arguments.repo.resolve()
    schedule = json.loads((repo / s3.SCHEDULE_REL).read_text())["payload"]["records"]
    worker = {"launch_id": arguments.launch_id, "worker_id": arguments.worker_id, "physical_gpu_index": arguments.physical_index, "gpu_uuid": arguments.gpu_uuid, "producer_commit": arguments.producer_commit}
    model, captured, identity, handles = initialize(repo, arguments.physical_index, arguments.gpu_uuid)
    s3.write_json_new(arguments.launch_root / "workers" / f"{arguments.worker_id}.json", s3.signed("s3_worker_identity", "PASS", identity, producer_commit=arguments.producer_commit))
    mask_archive = zipfile.ZipFile(repo / MASK_ZIP)
    jpeg_archive = tarfile.open(repo / JPEG_TAR, mode="r:") if arguments.phase == "test" else None
    counts: dict[str, int] = {}
    weight_cache: dict[str, dict[str, np.ndarray]] = {}
    typed_na_classes: set[str] = set()
    if arguments.phase == "test":
        report_path = repo / s3.ROOT_REL / "probe_class_typed_na_report.json"
        s3.require(report_path.is_file(), "probe class typed-NA report missing")
        report = json.loads(report_path.read_text())
        s3.require(s3.verify_signed(report) and report["status"] == "PASS", "probe class typed-NA report invalid")
        typed_na_classes = {str(row["annotation_wnid"]) for row in report["payload"]["records"]}
    try:
        cursor = 0
        while True:
            claimed = claim_next(repo, arguments.launch_root, schedule, arguments.phase, worker, cursor)
            if claimed is None:
                break
            row, claim, cursor = claimed
            state = execute_unit(repo, model, captured=captured, row=row, claim=claim, worker=worker, mask_archive=mask_archive, jpeg_archive=jpeg_archive, weight_cache=weight_cache, typed_na_classes=typed_na_classes)
            counts[state] = counts.get(state, 0) + 1
    finally:
        if jpeg_archive is not None:
            jpeg_archive.close()
        mask_archive.close()
        for handle in handles:
            handle.remove()
    s3.write_json_new(arguments.launch_root / "workers" / f"{arguments.worker_id}.result.json", s3.signed("s3_worker_result", "PASS", {"counts": counts, "completed_time_ns": time.time_ns()}, producer_commit=arguments.producer_commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
