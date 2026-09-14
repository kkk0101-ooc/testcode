"""Frozen metadata, numeric, and serialization primitives for patch-depth S3."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
STAGE_ID = "S3"
RUN_ID = "s3_r3_20260914"
ROOT_REL = Path("evidence/nns_vit_patch_depth/s3_r3_20260914")
PAYLOAD_REL = Path("artifacts/nns_vit_patch_depth/s3_r3_20260914")
SCHEDULE_REL = Path("manifests/nns_vit_patch_depth/s3_r3_source_schedule.json")
SPEC_REL = Path("config/nns_vit_patch_depth/s3_run_spec.json")
SCHEMA_REL = Path("config/nns_vit_patch_depth/s3_artifact.schema.json")
PREDECESSOR_COMMIT = "7b8dd924ee0160e7ca61796b266d840da34d4ed0"
PREDECESSOR_TERMINAL_SHA256 = "8abfd24bc7a6536694bb1fb765400395ff302c064b47fd2e1b8f571fdc1f4949"
PREDECESSOR_TERMINAL_DIGEST = "409e76b56f74a6e2a951d7105af9c95e35a7f787149589bae232d6ccb2d90cad"
SELECTED_SHA256 = "9309a73c1cc2d91cc4c6ca2dc6a13e5b6ed3fed680ed20bc35e1152fc259e4f8"
SELECTED_DIGEST = "2260a21c949f364eafc21fc3d722243a65c2f05aa82873c9aeb7779a8b1c3f6b"
SELECTED_RECORDS_DIGEST = "d5e9d5a4873d51b6ddedfe3ed2c9cb08be50fac4a8cfc053d8782c83ffafbd16"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
JPEG_ARCHIVE_SHA256 = "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0"
MASK_ARCHIVE_SHA256 = "31de0324c3a36e0817eeab0d704a7241b77ae5e725aeab103fd60c5b78dadf5e"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CROSSFIT_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S3_D2_TRAIN_CROSSFIT_V1"
TERMINAL_STATES = (
    "VALID_FEATURE", "NA_GEOMETRY_SIZE_MISMATCH", "NA_TARGET_DISAPPEARED_AFTER_CROP",
    "NA_MASK_LABEL_OUT_OF_RANGE", "NA_VOID_MASK", "FAILED_NUMERICAL",
    "FAILED_FEATURE_EXTRACTION", "BLOCKED_DATA_INTEGRITY", "BLOCKED_INFRASTRUCTURE",
    "FAILED_IMPLEMENTATION", "NA_D2_REQUIRED_SPLIT_OR_MODEL_MISSING",
)


class S3Error(RuntimeError):
    pass


class DataIntegrityError(S3Error):
    pass


class TypedUnitError(S3Error):
    def __init__(self, code: str, message: str):
        if code not in TERMINAL_STATES:
            raise ValueError(f"unsupported S3 unit state: {code}")
        super().__init__(message)
        self.code = code


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S3Error(message)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def signed(document_type: str, status: str, payload: Mapping[str, Any], *, producer_commit: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "$schema": "s3_artifact.schema.json", "schema_version": 1, "study_id": STUDY_ID,
        "stage_id": STAGE_ID, "run_id": RUN_ID, "document_type": document_type,
        "status": status, "payload": jsonable(dict(payload)),
    }
    if producer_commit is not None:
        value["producer_commit"] = producer_commit
    value["canonical_digest"] = canonical_digest(value)
    return value


def verify_signed(value: Mapping[str, Any]) -> bool:
    copy = dict(value)
    claimed = copy.pop("canonical_digest", None)
    return isinstance(claimed, str) and claimed == canonical_digest(copy)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    before = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            hasher.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"file changed while hashing: {path}")
    return hasher.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(jsonable(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(jsonable(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest(list(records))


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode())
    hasher.update(b"\0")
    hasher.update(str(tuple(array.shape)).encode())
    hasher.update(b"\0")
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def write_array_atomic(path: Path, value: np.ndarray, *, dtype: str, axes: Sequence[str], producer_commit: str, source_identity: str, recomputation_rule: str) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.dtype(dtype))
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    require(array.dtype.byteorder in ("<", "=", "|"), "array must be little endian on this environment")
    require(np.isfinite(array).all(), "array contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
    return {
        "path": path.as_posix(), "sha256": sha_file(path), "canonical_array_digest": array_digest(array),
        "bytes": path.stat().st_size, "shape": list(array.shape), "dtype": str(array.dtype),
        "axes": list(axes), "producer_commit": producer_commit, "source_identity": source_identity,
        "recomputation_rule": recomputation_rule,
    }


def artifact_record(repo: Path, path: Path) -> dict[str, Any]:
    value = {"path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if isinstance(document, dict) and isinstance(document.get("canonical_digest"), str):
            value["canonical_digest"] = document["canonical_digest"]
    return value


def _role_flags(split: str) -> dict[str, bool]:
    return {
        "probe_train": split == "probe_train", "probe_validation": split == "probe_validation",
        "P1": False, "P2": False, "D1": False, "D2": split == "d2_test",
        "S4_NNS": False, "S4_occlusion": False,
    }


def build_schedule(repo: Path) -> list[dict[str, Any]]:
    split_document = json.loads((repo / "evidence/nns_vit_patch_depth/s0_r5_20260914/split_manifest.json").read_text())
    qc_document = json.loads((repo / "evidence/nns_vit_patch_depth/s0_r3_20260914/image_mask_qc.json").read_text())
    selected_document = json.loads((repo / "evidence/nns_vit_patch_depth/s2_r8_20260914/candidate_m4_manifest.json").read_text())
    require(sha_file(repo / "evidence/nns_vit_patch_depth/s2_r8_20260914/candidate_m4_manifest.json") == SELECTED_SHA256, "selected manifest SHA drift")
    require(selected_document["canonical_digest"] == SELECTED_DIGEST, "selected manifest canonical digest drift")
    require(selected_document["payload"]["records_digest"] == SELECTED_RECORDS_DIGEST, "selected manifest records digest drift")
    source_rows = split_document["payload"]["records"]
    formal = [row for row in source_rows if row["final_split"] in ("probe_train", "probe_validation")]
    formal.sort(key=lambda row: (0 if row["final_split"] == "probe_train" else 1, row["annotation_wnid"], int(row["selection_rank_within_class_split"]), row["source_image_id"]))
    rows: list[dict[str, Any]] = []
    for row in formal:
        split = str(row["final_split"])
        rows.append({
            "schedule_index": len(rows), "source_image_id": row["source_image_id"], "source_group_id": row["source_group_id"],
            "image_sha256": row["image_sha256"], "image_bytes": int(row["image_bytes"]),
            "image_width": int(row["image_width"]), "image_height": int(row["image_height"]),
            "image_access_kind": "local_path", "image_local_path": row["image_local_path"], "image_archive_member": None,
            "annotation_sha256": row["annotation_sha256"], "annotation_archive_path": row["annotation_archive_path"],
            "mask_width": int(row["mask_width"]), "mask_height": int(row["mask_height"]),
            "annotation_wnid": row["annotation_wnid"], "annotation_label_id": int(row["annotation_label_id"]),
            "target_model_class_index": int(row["target_model_class_index"]), "split": split,
            "split_rank_within_class": int(row["selection_rank_within_class_split"]), "source_role": split,
            "role_flags": _role_flags(split), "resolution_join": "S0_R5_FROZEN_RECORD_DIRECT",
        })
    qc_index: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in qc_document["payload"]["records"]:
        key = (row["image_sha256"], row["annotation_sha256"], row["annotation_wnid"], int(row["image_width"]), int(row["image_height"]), int(row["mask_width"]), int(row["mask_height"]))
        qc_index.setdefault(key, []).append(row)
    for selected_rank, row in enumerate(selected_document["payload"]["records"]):
        key = (row["image_sha256"], row["annotation_sha256"], row["annotation_wnid"], int(row["image_width"]), int(row["image_height"]), int(row["mask_width"]), int(row["mask_height"]))
        matches = qc_index.get(key, [])
        require(len(matches) == 1, f"D2 QC identity join is not one-to-one at selected rank {selected_rank}")
        qc = matches[0]
        require(qc["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE", "D2 QC join selected an ineligible record")
        source_tail = str(row["source_image_id"]).split(":", 1)[1]
        member = source_tail.split("/", 1)[1]
        flags = {key: bool(value) for key, value in row["role_flags"].items()}
        flags.update({"probe_train": False, "probe_validation": False})
        rows.append({
            "schedule_index": len(rows), "source_image_id": row["source_image_id"], "source_group_id": row["source_image_id"],
            "image_sha256": row["image_sha256"], "image_bytes": int(row["image_bytes"]),
            "image_width": int(row["image_width"]), "image_height": int(row["image_height"]),
            "image_access_kind": "selected_tar_member", "image_local_path": None, "image_archive_member": member,
            "annotation_sha256": row["annotation_sha256"], "annotation_archive_path": qc["mask_archive_path"],
            "mask_width": int(row["mask_width"]), "mask_height": int(row["mask_height"]),
            "annotation_wnid": row["annotation_wnid"], "annotation_label_id": int(qc["annotation_label_id"]),
            "target_model_class_index": int(row["target_model_class_index"]), "split": "d2_test",
            "split_rank_within_class": int(row["D2_cap"]), "selected_manifest_rank": selected_rank,
            "source_role": row["source_role"], "role_flags": flags,
            "selection_key_sha256": row["selection_key_sha256"], "resolution_join": "S0_R3_EXACT_IDENTITY_ONE_TO_ONE",
        })
    require(len(rows) == 10458, "S3 schedule count drift")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("probe_train", "probe_validation", "d2_test")}
    require(counts == {"probe_train": 5232, "probe_validation": 1744, "d2_test": 3482}, "S3 split count drift")
    require([row["schedule_index"] for row in rows] == list(range(10458)), "S3 schedule indices not contiguous")
    return rows


def independence_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split = {split: [row for row in rows if row["split"] == split] for split in ("probe_train", "probe_validation", "d2_test")}
    pairs = []
    for left, right in (("probe_train", "probe_validation"), ("probe_train", "d2_test"), ("probe_validation", "d2_test")):
        source_overlap = sorted({row["source_image_id"] for row in by_split[left]} & {row["source_image_id"] for row in by_split[right]})
        sha_overlap = sorted({row["image_sha256"] for row in by_split[left]} & {row["image_sha256"] for row in by_split[right]})
        pairs.append({"left": left, "right": right, "source_id_overlap_count": len(source_overlap), "image_sha256_overlap_count": len(sha_overlap)})
    per_class: dict[str, dict[str, int]] = {}
    for row in rows:
        per_class.setdefault(str(row["annotation_wnid"]), {"probe_train": 0, "probe_validation": 0, "d2_test": 0})[str(row["split"])] += 1
    require(len(per_class) == 872, "class count drift")
    require(all(value["probe_train"] == 6 and value["probe_validation"] == 2 for value in per_class.values()), "train/validation per-class count drift")
    require(all(item["source_id_overlap_count"] == 0 and item["image_sha256_overlap_count"] == 0 for item in pairs), "split independence failure")
    return {"pairwise": pairs, "class_count": len(per_class), "per_class_counts_digest": canonical_digest(per_class), "patch_random_split": False, "source_image_is_atomic_unit": True, "patches_per_image": 196, "status": "PASS"}


def crossfit_assignment(annotation_wnid: str, source_ids: Sequence[str]) -> list[dict[str, Any]]:
    require(len(source_ids) == 6 and len(set(source_ids)) == 6, "cross-fit requires six unique train source IDs")
    keyed = []
    for source_id in source_ids:
        key = hashlib.sha256((CROSSFIT_NAMESPACE + "\0" + annotation_wnid + "\0" + source_id).encode()).hexdigest()
        keyed.append((key, source_id))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [{"source_image_id": source_id, "ordering_sha256": key, "rank": rank, "fold": rank // 2} for rank, (key, source_id) in enumerate(keyed)]


@dataclass(frozen=True)
class RidgeDecomposition:
    mean: np.ndarray
    scale: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    rhs_rotated: np.ndarray
    intercept: float


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    intercept: float
    alpha: float


def ridge_decompose(train_x: np.ndarray, train_y: np.ndarray) -> RidgeDecomposition:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    require(x.ndim == 2 and x.shape[0] == y.size and x.shape[0] > 0, "invalid ridge training geometry")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "non-finite ridge training input")
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, ddof=0, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    z = (x - mean) / scale
    intercept = float(y.mean(dtype=np.float64))
    centered = y - intercept
    gram = z.T @ z
    rhs = z.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues[eigenvalues < 0.0] = 0.0
    return RidgeDecomposition(mean, scale, eigenvalues, eigenvectors, eigenvectors.T @ rhs, intercept)


def model_from_decomposition(decomposition: RidgeDecomposition, alpha: float) -> RidgeModel:
    require(float(alpha) in ALPHAS, "alpha outside frozen grid")
    coefficients = decomposition.rhs_rotated / (decomposition.eigenvalues + float(alpha))
    weight = decomposition.eigenvectors @ coefficients
    return RidgeModel(decomposition.mean.copy(), decomposition.scale.copy(), weight, decomposition.intercept, float(alpha))


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> RidgeModel:
    return model_from_decomposition(ridge_decompose(train_x, train_y), float(alpha))


def predict_ridge(model: RidgeModel, value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == model.mean.size and np.isfinite(x).all(), "invalid ridge prediction array")
    result = ((x - model.mean) / model.scale) @ model.weight + model.intercept
    require(np.isfinite(result).all(), "non-finite ridge prediction")
    return result


def equal_image_mse(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> float:
    require(len(predictions) == len(targets) and len(predictions) > 0, "equal-image MSE needs matched nonempty images")
    values = []
    for prediction, target in zip(predictions, targets):
        p = np.asarray(prediction, dtype=np.float64).reshape(-1)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        require(p.shape == y.shape and p.size == 196, "validation image patch geometry drift")
        values.append(float(np.mean((p - y) ** 2, dtype=np.float64)))
    return math.fsum(values) / len(values)


def select_alpha(validation_mse: Mapping[float, float]) -> float:
    require(set(float(key) for key in validation_mse) == set(ALPHAS), "validation ledger alpha grid drift")
    return min(ALPHAS, key=lambda alpha: (float(validation_mse[alpha]), float(alpha)))


def low_level_features(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.float64)
    require(image.shape == (3, 224, 224), "low-level RGB geometry drift")
    require(np.isfinite(image).all() and float(image.min()) >= 0.0 and float(image.max()) <= 1.0, "low-level RGB domain drift")
    pooled = image.reshape(3, 14, 16, 14, 16).transpose(0, 1, 3, 2, 4).mean((3, 4), dtype=np.float64).reshape(3, 196).T
    yy, xx = np.meshgrid((np.arange(14, dtype=np.float64) + 0.5) / 14.0, (np.arange(14, dtype=np.float64) + 0.5) / 14.0, indexing="ij")
    return np.ascontiguousarray(np.column_stack((xx.reshape(-1), yy.reshape(-1), pooled)), dtype="<f4")


def patch_occupancy(labels: np.ndarray, target_label: int) -> np.ndarray:
    value = np.asarray(labels)
    if value.shape != (224, 224) or not np.issubdtype(value.dtype, np.integer):
        raise TypedUnitError("NA_GEOMETRY_SIZE_MISMATCH", "mask must be integer [224,224]")
    allowed = (value == 0) | (value == 1000) | ((value >= 1) & (value <= 919))
    if not bool(allowed.all()):
        raise TypedUnitError("NA_MASK_LABEL_OUT_OF_RANGE", "mask contains unsupported labels")
    void = value == 1000
    if bool(void.all()):
        raise TypedUnitError("NA_VOID_MASK", "crop is entirely void")
    target = value == int(target_label)
    if not bool(target.any()):
        raise TypedUnitError("NA_TARGET_DISAPPEARED_AFTER_CROP", "target absent after crop")
    planes = [value == 0, (value != 0) & (value != 1000) & ~target, void, target]
    result = np.stack([plane.reshape(14, 16, 14, 16).transpose(0, 2, 1, 3).mean((2, 3), dtype=np.float64).reshape(-1) for plane in planes]).astype("<f4")
    require(np.allclose(result.sum(axis=0, dtype=np.float64), 1.0, atol=0.0, rtol=0.0), "occupancy planes do not close")
    return result


def count_states(terminals: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for terminal in terminals:
        state = str(terminal["payload"]["terminal_state"])
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))
