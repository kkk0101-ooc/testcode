"""Frozen S4 metadata, queue, P1, P2, and evidence primitives.

The module deliberately has no Torch/PIL imports.  Formal decoding and model
execution live in :mod:`nns_patch_depth_s4_worker` and are imported only after
the launcher has verified the implementation seal and deterministic runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
STAGE_ID = "S4"
RUN_ID = "s4_r4_20260915"
ROOT_REL = Path("evidence/nns_vit_patch_depth/s4_r4_20260915")
PAYLOAD_REL = Path("artifacts/nns_vit_patch_depth/s4_r4_20260915")
CACHE_REL = Path("artifacts/nns_vit_patch_depth/s4_r4_20260915/selected_member_cache")
SCHEDULE_REL = Path("manifests/nns_vit_patch_depth/s4_r4_source_schedule.json")
PATH_REGISTRY_REL = Path("manifests/nns_vit_patch_depth/s4_r4_path_resolution_registry.json")
SPEC_REL = Path("config/nns_vit_patch_depth/s4_run_spec.json")
SCHEMA_REL = Path("config/nns_vit_patch_depth/s4_artifact.schema.json")
S3_COMMIT = "a2fa81082ba9b49e0aa44407cbbb373de18accff"
S3_TERMINAL_SHA256 = "8c1a4f82d3bb553246cc4c019e4e2286a8dfb7dfd8b058265af93ad5f1e192d4"
S3_TERMINAL_DIGEST = "1b6f7308f135dc69ad0146635a31eb529cfac703710fe5a340c009c56ce1215d"
SELECTED_SHA256 = "9309a73c1cc2d91cc4c6ca2dc6a13e5b6ed3fed680ed20bc35e1152fc259e4f8"
SELECTED_DIGEST = "2260a21c949f364eafc21fc3d722243a65c2f05aa82873c9aeb7779a8b1c3f6b"
SELECTED_RECORDS_DIGEST = "d5e9d5a4873d51b6ddedfe3ed2c9cb08be50fac4a8cfc053d8782c83ffafbd16"
QC_REL = Path("evidence/nns_vit_patch_depth/s0_r3_20260914/image_mask_qc.json")
SELECTED_REL = Path("evidence/nns_vit_patch_depth/s2_r8_20260914/candidate_m4_manifest.json")
S3_TERMINAL_REL = Path("evidence/nns_vit_patch_depth/s3_r7_20260915/terminal_receipt.json")
JPEG_ARCHIVE_REL = Path("assets/imagenet/ILSVRC2012_img_val.tar")
JPEG_ARCHIVE_BYTES = 6_744_924_160
JPEG_ARCHIVE_SHA256 = "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0"
MASK_ARCHIVE_REL = Path("assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip")
MASK_ARCHIVE_SHA256 = "31de0324c3a36e0817eeab0d704a7241b77ae5e725aeab103fd60c5b78dadf5e"
MODEL_REL = Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth")
MODEL_BYTES = 346_329_934
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_LOADER_IDENTITY = "F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES"
P1_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S2_R4_P1_POSITION_PERMUTATION_V1"
P2_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S2_R4_P2_MATCHED_CONTROL_V1"
P2_TIE_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S2_R4_P2_PATCH_TIE_V1"
AUDIT_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S2_R4_AUDIT_SUBSET_V1"
PATCH_GRID = 14
PATCH_COUNT = 196
BLOCKS = tuple(range(12))
BUDGETS = (4, 8, 16)
OPERATORS = ("image_channel_mean", "normalized_zero")
UNIT_TYPES = ("NNS_IMAGE", "OCCLUSION_PRIMARY", "OCCLUSION_SENSITIVITY")
NNS_TYPED_NA = (
    "NA_GEOMETRY_SIZE_MISMATCH", "NA_TARGET_DISAPPEARED_AFTER_CROP",
    "NA_MASK_LABEL_OUT_OF_RANGE", "NA_VOID_MASK",
)
TERMINAL_STATES = (
    "VALID_NNS", "VALID_OCCLUSION", *NNS_TYPED_NA,
    "NA_ZERO_POSITIVE_MASS", "NA_MATCHED_CONTROL_UNAVAILABLE",
    "NA_INCOMPLETE_DEPTH_TRAJECTORY", "NA_UPSTREAM_NNS",
    "FAILED_NUMERICAL", "FAILED_NNS", "FAILED_OCCLUSION",
    "BLOCKED_DATA_INTEGRITY", "BLOCKED_INFRASTRUCTURE", "FAILED_IMPLEMENTATION",
)
POSITION_STRATA = np.asarray([
    max(abs(2 * row - 13), abs(2 * column - 13)) // 2
    for row in range(PATCH_GRID) for column in range(PATCH_GRID)
], dtype=np.int16)


class S4Error(RuntimeError):
    pass


class DataIntegrityError(S4Error):
    pass


class TypedUnitError(S4Error):
    def __init__(self, code: str, message: str):
        if code not in TERMINAL_STATES:
            raise ValueError(f"unsupported S4 terminal state: {code}")
        super().__init__(message)
        self.code = code


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S4Error(message)


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
        "$schema": "s4_artifact.schema.json", "schema_version": 1,
        "study_id": STUDY_ID, "stage_id": STAGE_ID, "run_id": RUN_ID,
        "document_type": document_type, "status": status,
        "payload": jsonable(dict(payload)),
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


def write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def publish_json_no_overwrite(temporary: Path, terminal: Path, value: Any) -> None:
    """Publish complete JSON via same-filesystem hard link without overwrite."""
    write_json_new(temporary, value)
    terminal.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, terminal)
    except FileExistsError:
        existing = json.loads(terminal.read_text())
        require(existing == jsonable(value), f"conflicting immutable terminal: {terminal}")
    fsync_directory(terminal.parent)
    temporary.unlink()
    fsync_directory(temporary.parent)


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
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype(dtype)))
    if np.issubdtype(array.dtype, np.floating):
        require(np.isfinite(array).all(), "array contains NaN/Inf")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
    return {
        "path": path.as_posix(), "sha256": sha_file(path),
        "canonical_array_digest": array_digest(array), "bytes": path.stat().st_size,
        "shape": list(array.shape), "dtype": str(array.dtype), "axes": list(axes),
        "producer_identity": producer_commit, "source_identity": source_identity,
        "recomputation_rule": recomputation_rule,
    }


def records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest(list(records))


def build_schedule(repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_path = repo / SELECTED_REL
    require(sha_file(selected_path) == SELECTED_SHA256, "candidate_m4 SHA drift")
    selected_doc = json.loads(selected_path.read_text())
    require(selected_doc["canonical_digest"] == SELECTED_DIGEST, "candidate_m4 canonical digest drift")
    require(selected_doc["payload"]["records_digest"] == SELECTED_RECORDS_DIGEST, "candidate_m4 records digest drift")
    all_rows = selected_doc["payload"]["records"]
    role_keys = ("S4_NNS", "S4_occlusion", "P1", "P2", "D1", "D2")
    selected = [row for row in all_rows if all(row["role_flags"].get(key) is True for key in role_keys)]
    require(len(selected) == 1744, "S4 selected count drift")
    require(sum(row["role_flags"].get("D2") is True and not all(row["role_flags"].get(key) is True for key in role_keys[:-1]) for row in all_rows) == 1738, "D2-only exclusion count drift")
    require(not any(any(row["role_flags"].get(key) for key in ("pilot", "probe_train", "probe_validation")) for row in selected), "forbidden role entered S4")
    class_counts: dict[int, int] = {}
    for row in selected:
        target = int(row["target_model_class_index"])
        class_counts[target] = class_counts.get(target, 0) + 1
    require(len(class_counts) == 872 and set(class_counts.values()) == {2}, "S4 class identity drift")
    require(len({row["source_image_id"] for row in selected}) == 1744, "S4 source IDs not unique")
    require(len({row["image_sha256"] for row in selected}) == 1744, "S4 image hashes not unique")
    split_doc = json.loads((repo / "evidence/nns_vit_patch_depth/s0_r5_20260914/split_manifest.json").read_text())
    confirmatory = [row for row in split_doc["payload"]["records"] if row["final_split"] == "confirmatory_holdout"]
    confirmatory_index = {row["source_image_id"]: row for row in confirmatory}
    require(len(confirmatory) == len(confirmatory_index) == 1744, "S0 confirmatory holdout identity drift")
    class_doc = json.loads((repo / "evidence/nns_vit_patch_depth/s2_r4_20260914/formal_class_cohort_manifest.json").read_text())
    class_rank = {row["annotation_wnid"]: int(row["class_rank"]) for row in class_doc["payload"]["records"]}
    require(len(class_rank) == 872 and set(class_rank.values()) == set(range(872)), "S2-R4 formal class order drift")
    qc_doc = json.loads((repo / QC_REL).read_text())
    qc_index: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in qc_doc["payload"]["records"]:
        key = (row["image_sha256"], row["annotation_sha256"], row["annotation_wnid"], int(row["image_width"]), int(row["image_height"]), int(row["mask_width"]), int(row["mask_height"]))
        qc_index.setdefault(key, []).append(row)
    schedule: list[dict[str, Any]] = []
    resolution: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        original = confirmatory_index.get(row["source_image_id"])
        require(original is not None, f"candidate_m4 source absent from S0 confirmatory holdout: {index}")
        require(all(original[key] == row[key] for key in ("image_sha256", "annotation_sha256", "annotation_wnid", "image_width", "image_height", "mask_width", "mask_height", "target_model_class_index")), f"candidate_m4 and S0 confirmatory identity mismatch: {index}")
        key = (row["image_sha256"], row["annotation_sha256"], row["annotation_wnid"], int(row["image_width"]), int(row["image_height"]), int(row["mask_width"]), int(row["mask_height"]))
        matches = qc_index.get(key, [])
        require(len(matches) == 1, f"S4 QC join not one-to-one: {index}")
        qc = matches[0]
        require(qc["eligibility_status"] == "ELIGIBLE_PRE_SPLIT_FREEZE", f"S4 QC join ineligible: {index}")
        member = str(row["source_image_id"]).split(":", 1)[1].split("/", 1)[1]
        role_flags = {key: bool(row["role_flags"].get(key, False)) for key in role_keys}
        role_flags.update({"pilot": False, "probe_train": False, "probe_validation": False})
        record = {
            "schedule_index": index, "source_image_id": row["source_image_id"],
            "source_role": row["source_role"], "image_sha256": row["image_sha256"],
            "image_bytes": int(row["image_bytes"]), "image_width": int(row["image_width"]),
            "image_height": int(row["image_height"]), "annotation_sha256": row["annotation_sha256"],
            "annotation_wnid": row["annotation_wnid"], "annotation_label_id": int(qc["annotation_label_id"]),
            "formal_class_rank": class_rank[row["annotation_wnid"]],
            "mask_width": int(row["mask_width"]), "mask_height": int(row["mask_height"]),
            "target_model_class_index": int(row["target_model_class_index"]),
            "role_flags": role_flags, "selection_key_sha256": row["selection_key_sha256"],
            "image_archive_member": member, "mask_archive_member": qc["mask_archive_path"],
            "resolution_join": "S0_R3_EXACT_SEVEN_FIELD_IDENTITY_ONE_TO_ONE",
        }
        schedule.append(record)
        resolution.append({
            "schedule_index": index, "source_image_id": row["source_image_id"],
            "join_key": list(key), "image_archive_member": member,
            "mask_archive_member": qc["mask_archive_path"],
            "annotation_label_id": int(qc["annotation_label_id"]), "match_count": 1,
            "selection_sort_exclusion_replacement_use": False,
        })
    require([row["schedule_index"] for row in schedule] == list(range(1744)), "S4 schedule not contiguous")
    return schedule, resolution


def audit_subset_indices(schedule: Sequence[Mapping[str, Any]]) -> list[int]:
    by_class: dict[int, list[Mapping[str, Any]]] = {}
    for row in schedule:
        by_class.setdefault(int(row["target_model_class_index"]), []).append(row)
    class_rank = {int(row["target_model_class_index"]): int(row["formal_class_rank"]) for row in schedule}
    classes = sorted(by_class, key=lambda target: class_rank[target])[:64]
    selected = []
    for target in classes:
        row = min(by_class[target], key=lambda item: hashlib.sha256(f"{AUDIT_NAMESPACE}\0{item['source_image_id']}".encode()).hexdigest())
        selected.append(int(row["schedule_index"]))
    require(len(selected) == 64 and len(set(selected)) == 64, "audit subset construction drift")
    return sorted(selected)


def _hash_order(namespace: str, parts: Sequence[object], indices: Sequence[int]) -> np.ndarray:
    return np.asarray(sorted((int(index) for index in indices), key=lambda index: hashlib.sha256((namespace + "\0" + "\0".join(str(part) for part in (*parts, index))).encode()).digest()), dtype=np.int16)


def p1_permutation_indices(source_id: str) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty((12, 64, 196), dtype="<i2")
    digests = np.empty((12, 64), dtype="S64")
    for block in BLOCKS:
        for repeat in range(64):
            permutation = np.arange(PATCH_COUNT, dtype=np.int16)
            for stratum in range(7):
                indices = np.flatnonzero(POSITION_STRATA == stratum)
                permutation[indices] = _hash_order(P1_NAMESPACE, (source_id, block, repeat), indices)
            output[block, repeat] = permutation
            digests[block, repeat] = hashlib.sha256(permutation.tobytes()).hexdigest().encode()
    return output, digests


def p1_raw(positive: np.ndarray, occupancy: np.ndarray, source_id: str) -> dict[str, np.ndarray]:
    scores = np.asarray(positive, dtype=np.float64)
    target = np.asarray(occupancy, dtype=np.float64)
    require(scores.shape == (12, 196) and target.shape == (196,), "P1 geometry drift")
    require(np.isfinite(scores).all() and np.isfinite(target).all(), "P1 nonfinite input")
    permutations, seed_digests = p1_permutation_indices(source_id)
    numerator = np.empty((12, 65), dtype="<f8")
    denominator = scores.sum(1, dtype=np.float64).astype("<f8")
    valid = denominator > 0
    numerator.fill(0.0)
    for block in BLOCKS:
        if not valid[block]:
            continue
        numerator[block, 0] = np.dot(scores[block], target)
        for repeat in range(64):
            numerator[block, repeat + 1] = np.dot(scores[block, permutations[block, repeat]], target)
    return {
        "permutation_indices": permutations, "seed_digests": seed_digests,
        "numerators": numerator, "denominators": denominator,
        "positive_mass": denominator.copy(), "valid_blocks": valid.astype(np.uint8),
    }


def boundary_complexity(indices: Sequence[int]) -> int:
    chosen = {int(item) for item in indices}
    require(all(item in range(PATCH_COUNT) for item in chosen), "patch index out of range")
    result = 0
    for patch in chosen:
        row, column = divmod(patch, PATCH_GRID)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y, x = row + dy, column + dx
            if y not in range(PATCH_GRID) or x not in range(PATCH_GRID) or y * PATCH_GRID + x not in chosen:
                result += 1
    return result


def selected_and_controls(positive: np.ndarray, source_id: str, block: int, budget: int) -> dict[str, Any]:
    values = np.asarray(positive, dtype=np.float64).reshape(-1)
    require(values.shape == (196,) and np.isfinite(values).all(), "P2 score drift")
    order = _hash_order(P2_TIE_NAMESPACE, (source_id, block, budget), range(196))
    order = np.asarray(sorted((int(index) for index in order), key=lambda index: (-float(values[index]), hashlib.sha256(f"{P2_TIE_NAMESPACE}\0{source_id}\0{block}\0{budget}\0{index}".encode()).digest())), dtype=np.int16)
    selected = tuple(int(item) for item in order[:budget])
    counts = np.bincount(POSITION_STRATA[list(selected)], minlength=7)
    selected_boundary = boundary_complexity(selected)
    tolerance = max(4, int(math.ceil(0.25 * selected_boundary)))
    controls: list[dict[str, Any]] = []
    seen = {tuple(sorted(selected))}
    for attempt in range(512):
        candidate: list[int] = []
        for stratum, needed in enumerate(counts):
            if not needed:
                continue
            pool = np.flatnonzero(POSITION_STRATA == stratum)
            ranked = _hash_order(P2_NAMESPACE, (source_id, block, budget, attempt, stratum), pool)
            candidate.extend(int(item) for item in ranked[:int(needed)])
        key = tuple(sorted(candidate))
        if key in seen:
            continue
        seen.add(key)
        boundary = boundary_complexity(key)
        error = abs(boundary - selected_boundary)
        if error <= tolerance:
            controls.append({
                "indices": list(key), "attempt": attempt,
                "boundary_complexity": boundary, "boundary_absolute_error": error,
                "overlap_count": len(set(key) & set(selected)),
                "stratum_counts": [int(item) for item in counts],
            })
            if len(controls) == 4:
                break
    if len(controls) != 4:
        return {
            "block": block, "budget": budget, "status": "NA_MATCHED_CONTROL_UNAVAILABLE",
            "selected": list(selected), "selected_boundary_complexity": selected_boundary,
            "selected_positive_count": sum(values[index] > 0 for index in selected),
            "stratum_counts": [int(item) for item in counts], "controls": controls,
            "attempts_exhausted": 512, "reason": "four matched controls unavailable within 512 attempts",
        }
    return {
        "block": block, "budget": budget, "status": "MATCHED", "selected": list(selected),
        "selected_boundary_complexity": selected_boundary,
        "selected_positive_count": sum(values[index] > 0 for index in selected),
        "stratum_counts": [int(item) for item in counts], "controls": controls,
    }


def build_p2_plan(positive: np.ndarray, source_id: str) -> list[dict[str, Any]]:
    values = np.asarray(positive)
    require(values.shape == (12, 196), "P2 positive score shape drift")
    return [selected_and_controls(values[block], source_id, block, budget) for block in BLOCKS for budget in BUDGETS]


def terminal_path(root: Path, unit_type: str, schedule_index: int) -> Path:
    require(unit_type in UNIT_TYPES, "unknown unit type")
    return root / "unit_terminals" / unit_type.lower() / f"unit-{schedule_index:04d}.json"


def all_terminal_paths(root: Path) -> list[Path]:
    return sorted(root.glob("unit_terminals/*/unit-*.json"))


def claim_next(launch_root: Path, root: Path, schedule: Sequence[Mapping[str, Any]], worker: Mapping[str, Any]) -> dict[str, Any] | None:
    """Claim one globally ready dependency unit with O_EXCL semantics."""
    # Source-major scanning is intentional: as soon as one NNS terminal is
    # published, its occlusion dependants become claimable without waiting for
    # a cohort-wide NNS barrier.  Claims remain globally shared across workers.
    for row in schedule:
        for unit_type in UNIT_TYPES:
            index = int(row["schedule_index"])
            if terminal_path(root, unit_type, index).is_file():
                continue
            if unit_type != "NNS_IMAGE" and not terminal_path(root, "NNS_IMAGE", index).is_file():
                continue
            claim_path = launch_root / "claims" / unit_type.lower() / f"unit-{index:04d}.claim.json"
            payload = {
                "launch_id": launch_root.name, "unit_type": unit_type,
                "schedule_index": index, "source_id": row["source_image_id"],
                "operator": None if unit_type == "NNS_IMAGE" else (OPERATORS[0] if unit_type == "OCCLUSION_PRIMARY" else OPERATORS[1]),
                "component": "first_residual_addition_all_head_aggregate_attention_branch" if unit_type == "NNS_IMAGE" else None,
                "worker_id": worker["worker_id"], "worker_pid": int(worker["worker_pid"]),
                "physical_gpu_index": int(worker["physical_gpu_index"]), "gpu_uuid": worker["gpu_uuid"],
                "claimed_unix_ns": time.time_ns(), "claim_scope": "THIS_LAUNCH_ONLY",
                "scientific_seed_depends_on_worker_gpu_claim_launch_or_time": False,
            }
            claim = signed("s4_unit_claim", "CLAIMED", payload)
            claim_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                continue
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(json.dumps(claim, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            fsync_directory(claim_path.parent)
            return {**payload, "claim_path": claim_path.as_posix(), "claim_canonical_digest": claim["canonical_digest"]}
    return None


def missing_units(root: Path, schedule: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    return [(unit_type, int(row["schedule_index"])) for unit_type in UNIT_TYPES for row in schedule if not terminal_path(root, unit_type, int(row["schedule_index"])).is_file()]
