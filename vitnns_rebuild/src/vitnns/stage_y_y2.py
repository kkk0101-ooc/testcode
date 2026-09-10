"""Stage Y2 metadata-only cohort sealing and evidence primitives.

The cohort path intentionally imports no image decoder, Torch, timm, or CUDA
runtime. Candidate members from the official validation tar are streamed only
as raw bytes for SHA-256 before the immutable cohort and subset seals exist.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROTOCOL_ID = "STAGE_Y_VIT_BLOCK_LEVEL_NNS_MECHANISM_V1_20260910"
STAGE_ID = "Y2"
TOKEN = "STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY"
RELEASE_ID = "ILSVRC2012_VAL_OFFICIAL"
NAMESPACE = "STAGE_Y_FORMAL_COHORT_V1_20260910"
AUDIT_NAMESPACE = "STAGE_Y_AUDIT_SUBSET_V1_20260910"
PERTURBATION_NAMESPACE = "STAGE_Y_PERTURBATION_SUBSET_V1_20260910"
ROOT = Path("evidence/stage_y/y2_clean_formal_20260910/formal")
REPORT_ROOT = Path("reports/stage_y_y2_20260911")
RUN_SPEC = Path("config/stage_y/y2_run_spec.json")
PLAN = Path("manifests/stage_y/cohort_selection_and_exclusion_plan.json")
TAR_PATH = Path("assets/imagenet/ILSVRC2012_img_val.tar")
LABEL_PATH = Path("assets/imagenet/imagenet_2012_validation_synset_labels.txt")
DEVKIT_PATH = Path("assets/imagenet/ILSVRC2012_devkit_t12.tar.gz")
Y1_ORDER44 = "ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"
EXPECTED_BRANCH = "stage-d-d3-pass"
EXPECTED_BASE = "70f70ca7290316531406791a6533d65590424a24"
Y0_REF = "refs/archive/stage-y-y0-protocol-freeze"
Y1_BLOCKED_REF = "refs/archive/stage-y-y1-implementation-cost-pilot-blocked"
Y1_R1_REF = "refs/archive/stage-y-y1-r1-resource-telemetry-waiver"
SHARD_COUNT = 8
SHARD_IDS = tuple(range(SHARD_COUNT))
UNIT_TERMINAL_STATES = ("VALID", "FAILED_NUMERICAL", "BLOCKED_INFRASTRUCTURE", "BLOCKED_BUDGET")
SCHEDULE = Path("config/stage_y/y2_parallel_schedule.json")

EXCLUSION_PATHS = (
    Path("manifests/stage_c_sample_registry.csv"),
    Path("manifests/stage_d_fidelity_v1/registry.json"),
    Path("evidence/stage_f/f3_r7_recovery_v2_20260907/formal/selection_manifest.json"),
    Path("manifests/stage_f_f3c_v1/sample_split.json"),
    Path("manifests/stage_x_prior_access_ledger_v1.json"),
    Path("evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal/manifests/development_samples.csv"),
)


class StageYY2Error(RuntimeError):
    code = "BLOCKED"


class StageYY2CohortError(StageYY2Error):
    code = "BLOCKED_COHORT"


class StageYY2ProvenanceError(StageYY2Error):
    code = "BLOCKED_PROVENANCE"


class StageYY2ScheduleError(StageYY2Error):
    code = "BLOCKED_RESOURCE_SCHEDULE"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "$schema": "stage_y_y2_artifact.schema.json",
        "schema_version": 1,
        "document_type": document_type,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": status,
        "payload": dict(payload),
    }
    value["canonical_digest"] = canonical_digest(value)
    return value


def validate_signed(value: Mapping[str, Any]) -> None:
    required = {"$schema", "schema_version", "document_type", "protocol_id", "stage_id", "status", "payload", "canonical_digest"}
    if set(value) != required:
        raise StageYY2ProvenanceError("artifact envelope is not closed")
    unsigned = dict(value)
    observed = unsigned.pop("canonical_digest")
    if observed != canonical_digest(unsigned):
        raise StageYY2ProvenanceError("artifact canonical digest mismatch")
    if value["protocol_id"] != PROTOCOL_ID or value["stage_id"] != STAGE_ID:
        raise StageYY2ProvenanceError("artifact identity drift")


def read_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StageYY2ProvenanceError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result
    def reject(token: str) -> None:
        raise StageYY2ProvenanceError(f"non-finite JSON constant: {path}:{token}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageYY2ProvenanceError(f"cannot read strict JSON {path}: {error}") from error


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_sha(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StageYY2ProvenanceError(f"required file missing: {path}")
    item: dict[str, Any] = {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if path.suffix == ".json":
        value = read_json(path)
        if isinstance(value, Mapping) and "canonical_digest" in value:
            unsigned = dict(value)
            claimed = unsigned.pop("canonical_digest")
            if canonical_digest(unsigned) != claimed:
                raise StageYY2ProvenanceError(f"canonical digest drift: {path}")
            item["canonical_digest"] = claimed
    return item


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def publish_terminal(pending: Path, terminal: Path, value: Mapping[str, Any]) -> None:
    if not pending.is_file() or terminal.exists():
        raise StageYY2ProvenanceError(f"atomic terminal precondition failed: {terminal}")
    temporary = terminal.with_name(f".{terminal.name}.tmp")
    write_json_new(temporary, value)
    # The fully fsynced terminal bytes first replace the PENDING inode and the
    # PENDING pathname then moves atomically to its one terminal pathname.
    os.replace(temporary, pending)
    os.replace(pending, terminal)
    directory = os.open(terminal.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout.strip()


def digest_text(namespace: str, *parts: object) -> str:
    return hashlib.sha256((namespace + "\0" + "\0".join(str(part) for part in parts)).encode("utf-8")).hexdigest()


def native_object_id(value: str) -> str:
    text = str(value).strip()
    if text.startswith(RELEASE_ID + ":"):
        text = text[len(RELEASE_ID) + 1:]
    if text.startswith("assets/data/"):
        text = text[len("assets/data/"):]
    if text.startswith("ILSVRC2012_val_"):
        text = "ILSVRC2012_val/" + text
    if not text.startswith("ILSVRC2012_val/ILSVRC2012_val_") or not text.endswith(".JPEG"):
        raise StageYY2CohortError(f"unparseable release-native object identifier: {value}")
    number = text.removeprefix("ILSVRC2012_val/ILSVRC2012_val_").removesuffix(".JPEG")
    if len(number) != 8 or not number.isdigit() or not 1 <= int(number) <= 50000:
        raise StageYY2CohortError(f"invalid validation object identifier: {value}")
    return text


def source_object_id(value: str) -> str:
    return RELEASE_ID + ":" + native_object_id(value)


def _collect_sample_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"sample_id", "source_object_id"} and isinstance(item, str):
                try:
                    result.add(native_object_id(item))
                except StageYY2CohortError:
                    pass
            elif key in {"sample_ids", "selected_sample_ids", "x1_allowed_audit_samples"} and isinstance(item, list):
                for entry in item:
                    if isinstance(entry, str):
                        result.add(native_object_id(entry))
            result.update(_collect_sample_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_collect_sample_ids(item))
    return result


def _csv_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise StageYY2CohortError(f"CSV has no header: {path}")
        field = "source_object_id" if "source_object_id" in reader.fieldnames else "sample_id" if "sample_id" in reader.fieldnames else None
        if field is None:
            raise StageYY2CohortError(f"CSV has no source identity column: {path}")
        return {native_object_id(row[field]) for row in reader}


def exclusion_union(repo: Path) -> tuple[dict[str, Any], set[str]]:
    plan = read_json(repo / PLAN)
    configured = tuple(Path(row["path"]) for row in plan["prior_access_exclusion_sources"])
    if configured != EXCLUSION_PATHS:
        raise StageYY2CohortError("frozen exclusion source roster drift")
    sources: list[dict[str, Any]] = []
    union: set[str] = set()
    for row, relative in zip(plan["prior_access_exclusion_sources"], EXCLUSION_PATHS):
        path = repo / relative
        if not path.is_file():
            raise StageYY2CohortError(f"exclusion source missing: {relative}")
        ids = _csv_ids(path) if path.suffix == ".csv" else _collect_sample_ids(read_json(path))
        if not ids:
            raise StageYY2CohortError(f"exclusion source yielded no identities: {relative}")
        normalized = {source_object_id(item) for item in ids}
        union.update(normalized)
        sources.append({"stage": row["stage"], **file_record(relative), "parsed_identity_count": len(normalized), "identity_digest": canonical_digest(sorted(normalized))})
    expected = source_object_id(Y1_ORDER44)
    if expected not in union:
        raise StageYY2CohortError("Y1 order-44 audit image is not excluded")
    payload = {
        "status": "PASS",
        "normalization": "dataset_release_id + ':' + release-native object identifier",
        "sources": sources,
        "source_count": len(sources),
        "union_count": len(union),
        "union_digest": canonical_digest(sorted(union)),
        "y1_order_44_source_object_id": expected,
        "y1_order_44_excluded": True,
        "source_object_ids": sorted(union),
    }
    return payload, union


def official_class_index(labels: Sequence[str]) -> dict[str, int]:
    unique = sorted(set(labels))
    if len(labels) != 50000 or len(unique) != 1000 or any(len(item) != 9 or not item.startswith("n") or not item[1:].isdigit() for item in unique):
        raise StageYY2CohortError("official label metadata shape or WNID roster drift")
    return {wnid: index for index, wnid in enumerate(unique)}


def enumerate_candidates(repo: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tar_path = repo / TAR_PATH
    label_path = repo / LABEL_PATH
    labels = label_path.read_text(encoding="utf-8").splitlines()
    mapping = official_class_index(labels)
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    total_bytes = 0
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive:
            if not member.isfile():
                raise StageYY2CohortError("validation tar contains a non-file member")
            native = native_object_id(member.name)
            if native in names:
                raise StageYY2CohortError("duplicate source-object ID in validation tar")
            names.add(native)
            image_number = int(Path(member.name).stem.rsplit("_", 1)[1])
            stream = archive.extractfile(member)
            if stream is None:
                raise StageYY2CohortError(f"cannot stream tar member: {member.name}")
            file_sha = stream_sha(stream)
            wnid = labels[image_number - 1]
            class_id = mapping[wnid]
            sid = source_object_id(native)
            records.append({
                "dataset_release_id": RELEASE_ID,
                "class_id": class_id,
                "official_wnid": wnid,
                "source_object_id": sid,
                "release_native_object_id": native,
                "tar_member_name": member.name,
                "file_bytes": member.size,
                "file_sha256": file_sha,
            })
            total_bytes += member.size
    if len(records) != 50000 or len(names) != 50000:
        raise StageYY2CohortError("official validation release count drift")
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda item: item["source_object_id"].encode("utf-8")):
        digest.update(canonical_bytes(row)); digest.update(b"\n")
    metadata = {
        "candidate_count": len(records),
        "distinct_class_count": len({row["class_id"] for row in records}),
        "raw_bytes_hashed": total_bytes,
        "candidate_metadata_digest": digest.hexdigest(),
        "image_decode_count": 0,
        "model_load_count": 0,
    }
    return records, metadata


def select_cohort(candidates: Sequence[Mapping[str, Any]], exclusions: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[str] = set()
    eligible_by_class: dict[int, list[dict[str, Any]]] = {}
    for original in candidates:
        row = dict(original)
        sid = row["source_object_id"]
        if sid in seen:
            raise StageYY2CohortError("duplicate candidate source-object ID")
        seen.add(sid)
        sha = str(row.get("file_sha256", ""))
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            raise StageYY2CohortError("candidate missing lowercase file SHA-256")
        class_id = int(row["class_id"])
        row["class_digest"] = digest_text(NAMESPACE, RELEASE_ID, class_id)
        row["object_digest"] = digest_text(NAMESPACE, RELEASE_ID, class_id, sid, sha)
        row["excluded"] = sid in exclusions
        if not row["excluded"]:
            eligible_by_class.setdefault(class_id, []).append(row)
    if len(eligible_by_class) < 100:
        raise StageYY2CohortError("fewer than 100 eligible true classes")
    class_order = sorted(eligible_by_class, key=lambda item: (digest_text(NAMESPACE, RELEASE_ID, item), item))[:100]
    selected = []
    for class_id in class_order:
        ordered = sorted(eligible_by_class[class_id], key=lambda item: (item["object_digest"], item["source_object_id"].encode("utf-8")))
        selected.append(ordered[0])
    selected.sort(key=lambda item: (item["object_digest"], item["source_object_id"].encode("utf-8")))
    for order, row in enumerate(selected):
        row["selection_order"] = order
        row["selection_digest"] = canonical_digest({key: row[key] for key in ("dataset_release_id", "class_id", "source_object_id", "file_sha256", "class_digest", "object_digest", "selection_order")})
        row.pop("excluded", None)
    if len(selected) != 100 or len({row["class_id"] for row in selected}) != 100 or len({row["source_object_id"] for row in selected}) != 100:
        raise StageYY2CohortError("selected cohort cardinality contract failed")
    proof = {
        "status": "PASS",
        "algorithm": "frozen class digest then object digest ordering",
        "namespace": NAMESPACE,
        "eligible_class_count": len(eligible_by_class),
        "selected_class_order": class_order,
        "selected_count": len(selected),
        "distinct_true_class_count": len({row["class_id"] for row in selected}),
        "selection_digest": canonical_digest(selected),
        "replacement_allowed": False,
        "selection_fields_used": ["dataset_release_id", "class_id", "source_object_id", "file_sha256"],
        "outcome_fields_used": [],
        "candidate_image_decode_count": 0,
        "model_load_count": 0,
        "y1_order_44_eligible": source_object_id(Y1_ORDER44) not in exclusions,
        "y1_order_44_selected": source_object_id(Y1_ORDER44) in {row["source_object_id"] for row in selected},
    }
    return selected, proof


def subset_registry(selected: Sequence[Mapping[str, Any]], namespace: str, count: int, subset_id: str) -> dict[str, Any]:
    rows = []
    for row in selected:
        item = {
            "selection_order": row["selection_order"],
            "source_object_id": row["source_object_id"],
            "selected_object_digest": row["object_digest"],
            "subset_selection_digest": digest_text(namespace, row["object_digest"]),
        }
        rows.append(item)
    rows.sort(key=lambda item: (item["subset_selection_digest"], item["source_object_id"].encode("utf-8")))
    rows = rows[:count]
    for rank, row in enumerate(rows):
        row["subset_rank"] = rank
    return {
        "subset_id": subset_id,
        "namespace": namespace,
        "count": count,
        "records": rows,
        "subset_digest": canonical_digest(rows),
        "selected_before_outcome_access": True,
    }


def selected_by_order(cohort_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_signed(cohort_artifact)
    records = [dict(row) for row in cohort_artifact["payload"]["records"]]
    if [row["selection_order"] for row in records] != list(range(100)):
        raise StageYY2CohortError("cohort selection order drift")
    return records


def manifest_digest(paths: Iterable[Path]) -> str:
    rows = [file_record(path) for path in sorted(paths)]
    return canonical_digest(rows)


def y2_shard_id(selection_order_zero_based: int) -> int:
    order = int(selection_order_zero_based)
    if order not in range(100):
        raise StageYY2ScheduleError(f"Y2 selection order outside 0..99: {order}")
    return order % SHARD_COUNT


def y3_shard_id(perturbation_subset_order: int, transform_registry_order: int) -> int:
    image_order = int(perturbation_subset_order)
    transform_order = int(transform_registry_order)
    if image_order not in range(20) or transform_order not in range(8):
        raise StageYY2ScheduleError("Y3 canonical coordinate outside 20 x 8 roster")
    return (image_order * 8 + transform_order) % SHARD_COUNT


def expected_y2_orders(shard_id: int) -> list[int]:
    shard = int(shard_id)
    if shard not in SHARD_IDS:
        raise StageYY2ScheduleError(f"invalid shard id: {shard}")
    return list(range(shard, 100, SHARD_COUNT))


def schedule_from_source_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the eight-worker projection from immutable Y1-R1 inputs."""
    validate_signed_ledger = dict(ledger)
    unsigned = dict(validate_signed_ledger)
    observed = unsigned.pop("canonical_digest", None)
    if observed != canonical_digest(unsigned):
        raise StageYY2ProvenanceError("Y1-R1 budget ledger canonical digest drift")
    payload = ledger["payload"]
    fixed = payload["fixed_cost_per_metric_maximum"]
    unit = payload["clean_unit_cost_per_metric_maximum"]
    fixed_bytes = int(payload["fixed_bytes_maximum"])
    unit_bytes = int(payload["retained_bytes_per_condition_maximum"])
    multiplier = float(payload["safety_multiplier"])

    def ceil_tenth(value: float) -> float:
        return math.ceil(value * 10.0 - 1e-12) / 10.0

    def one_shard(units: int) -> dict[str, float]:
        raw_gpu = (float(fixed["gpu_active_seconds"]) + units * float(unit["gpu_active_seconds"])) / 3600.0
        raw_cpu = (float(fixed["cpu_seconds"]) + units * float(unit["cpu_seconds"])) / 3600.0
        raw_wall = (float(fixed["wall_seconds"]) + units * float(unit["wall_seconds"])) / 3600.0
        raw_storage = (fixed_bytes + units * unit_bytes) / 1024.0 ** 3
        return {
            "units": units,
            "raw_h20_gpu_hours": raw_gpu,
            "raw_cpu_hours": raw_cpu,
            "raw_wall_hours": raw_wall,
            "raw_storage_gib": raw_storage,
            "cap_h20_gpu_hours": ceil_tenth(multiplier * raw_gpu),
            "cap_cpu_hours": float(math.ceil(multiplier * raw_cpu - 1e-12)),
            "cap_worker_wall_hours": float(math.ceil(multiplier * raw_wall - 1e-12)),
            "cap_storage_gib": float(math.ceil(multiplier * raw_storage - 1e-12)),
        }

    def stage(counts: Sequence[int], aggregate: Mapping[str, float], overall_wall: float) -> dict[str, Any]:
        units = sum(counts)
        raw_gpu = (SHARD_COUNT * float(fixed["gpu_active_seconds"]) + units * float(unit["gpu_active_seconds"])) / 3600.0
        raw_cpu = (SHARD_COUNT * float(fixed["cpu_seconds"]) + units * float(unit["cpu_seconds"])) / 3600.0
        raw_summed_wall = (SHARD_COUNT * float(fixed["wall_seconds"]) + units * float(unit["wall_seconds"])) / 3600.0
        raw_storage = (SHARD_COUNT * fixed_bytes + units * unit_bytes) / 1024.0 ** 3
        shards = [one_shard(count) for count in counts]
        projected_cpu = multiplier * raw_cpu
        return {
            "shard_counts": list(counts),
            "per_shard": shards,
            "raw_aggregate": {"h20_gpu_hours": raw_gpu, "cpu_hours": raw_cpu, "summed_worker_wall_hours": raw_summed_wall, "storage_gib": raw_storage},
            "safety_projected": {"h20_gpu_hours": multiplier * raw_gpu, "cpu_hours": projected_cpu, "storage_gib": multiplier * raw_storage},
            "aggregate_caps": dict(aggregate),
            "critical_path_worker_wall_hours": max(row["cap_worker_wall_hours"] for row in shards),
            "overall_stage_wall_hours": overall_wall,
            "cpu_only_reconciliation_allowance_hours": float(aggregate["cpu_hours"]) - projected_cpu,
            "per_worker_fixed_initialization": {**fixed, "retained_bytes": fixed_bytes},
            "fleet_concurrent_reserved_gpu_gib": SHARD_COUNT * float(aggregate["peak_reserved_gpu_gib_per_process"]),
            "gpu_accounting": "SUM_ACROSS_WORKERS_NEVER_DIVIDE_BY_DEVICE_COUNT",
        }

    y2_caps = {"h20_gpu_hours": 64.0, "cpu_hours": 218.0, "storage_gib": 6.0, "peak_reserved_gpu_gib_per_process": 21.0, "devices": 8}
    y3_caps = {"h20_gpu_hours": 102.4, "cpu_hours": 349.0, "storage_gib": 9.0, "peak_reserved_gpu_gib_per_process": 21.0, "devices": 8}
    return {
        "source_canonical_digest": observed,
        "safety_multiplier": multiplier,
        "y2": stage([13, 13, 13, 13, 12, 12, 12, 12], y2_caps, 12.0),
        "y3": stage([20] * 8, y3_caps, 18.0),
    }


def build_shard_manifests(cohort_artifact: Mapping[str, Any], formal_root: Path) -> list[dict[str, Any]]:
    """Write eight immutable manifests after the shared cohort has sealed."""
    records = selected_by_order(cohort_artifact)
    cohort_digest = str(cohort_artifact["canonical_digest"])
    root_label = ROOT if formal_root.as_posix().endswith(ROOT.as_posix()) else formal_root
    manifests: list[dict[str, Any]] = []
    for shard_id in SHARD_IDS:
        assigned = [dict(row) for row in records if y2_shard_id(int(row["selection_order"])) == shard_id]
        expected = expected_y2_orders(shard_id)
        if [int(row["selection_order"]) for row in assigned] != expected:
            raise StageYY2ScheduleError("deterministic shard mapping drift")
        payload = {
            "shard_id": shard_id,
            "shard_name": f"shard-{shard_id:02d}",
            "root": (root_label / "shards" / f"shard-{shard_id:02d}").as_posix(),
            "cohort_canonical_digest": cohort_digest,
            "mapping_rule": "selection_order_zero_based mod 8",
            "unit_count": len(assigned),
            "selection_orders": expected,
            "records": assigned,
            "dynamic_rebalancing": False,
            "condition_roster": ["clean"],
        }
        document = signed("stage_y_y2_shard_manifest", "PASS", payload)
        shard_root = formal_root / "shards" / f"shard-{shard_id:02d}"
        manifest_path = shard_root / "shard_manifest.json"
        write_json_new(manifest_path, document); manifest_path.chmod(0o444)
        manifests.append(document)
    return manifests


def environment_digest(identity: Mapping[str, Any]) -> str:
    required = {"python", "torch", "torchvision", "timm", "cuda", "driver", "checkpoint_sha256", "numeric_contract_digest"}
    if set(identity) != required:
        raise StageYY2ScheduleError("worker environment identity is not closed")
    return canonical_digest(dict(identity))


def worker_binding(identity: Mapping[str, Any]) -> dict[str, Any]:
    required = {"physical_index", "cuda_visible_devices", "gpu_uuid", "pci_bus_id", "device_name", "hostname", "pid", "environment_digest"}
    if set(identity) != required:
        raise StageYY2ScheduleError("worker binding fields are not closed")
    if identity["device_name"] != "NVIDIA H20" or not identity["gpu_uuid"] or not identity["pci_bus_id"]:
        raise StageYY2ScheduleError("worker GPU identity violates H20 contract")
    if not isinstance(identity["pid"], int) or identity["pid"] <= 0:
        raise StageYY2ScheduleError("worker PID invalid")
    return dict(identity)


def begin_unit(unit_root: Path, *, shard_id: int, selection_order: int, source_object_id_value: str) -> Path:
    if y2_shard_id(selection_order) != shard_id:
        raise StageYY2ScheduleError("unit assigned to wrong shard")
    pending = unit_root / "completion" / "unit.pending.json"
    value = signed("stage_y_y2_unit_pending", "PENDING", {
        "shard_id": shard_id, "selection_order": selection_order,
        "source_object_id": source_object_id_value, "condition_id": "clean",
        "terminal_published": False,
    })
    write_json_new(pending, value)
    return pending


def append_attempt(unit_root: Path, attempt: Mapping[str, Any]) -> dict[str, Any]:
    attempts = unit_root / "attempts"
    index = len(list(attempts.glob("attempt-*.json"))) if attempts.exists() else 0
    path = attempts / f"attempt-{index:03d}.json"
    document = signed("stage_y_y2_unit_attempt", "PASS", {"attempt_index": index, **dict(attempt)})
    write_json_new(path, document)
    return file_record(path)


def finish_unit(unit_root: Path, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if status not in UNIT_TERMINAL_STATES:
        raise StageYY2ScheduleError(f"invalid unit terminal: {status}")
    attempts = sorted((unit_root / "attempts").glob("attempt-*.json"))
    if not attempts:
        raise StageYY2ScheduleError("unit terminal cannot omit all attempts")
    expected_keys = {"unit_digest", "attempt_id", "shard_id", "selection_order", "source_object_id", "condition_id", "failure_code", "target_identity", "source_binding", "checkpoint_sha256", "environment_digest", "numeric_contract_digest", "quadrature_attempts", "resource_telemetry", "array_records", "attempt_records"}
    if set(payload) != expected_keys:
        raise StageYY2ProvenanceError("unit terminal payload is not closed")
    if payload["condition_id"] != "clean" or y2_shard_id(int(payload["selection_order"])) != int(payload["shard_id"]):
        raise StageYY2ScheduleError("unit terminal scope or shard drift")
    observed_attempts = [file_record(path) for path in attempts]
    if payload["attempt_records"] != observed_attempts:
        raise StageYY2ProvenanceError("unit terminal does not bind every retained attempt")
    document = signed("stage_y_y2_unit_terminal", status, payload)
    publish_terminal(unit_root / "completion/unit.pending.json", unit_root / "completion/unit.json", document)
    return document


def reconcile_shards(formal_root: Path, *, minimum_valid: int = 90) -> dict[str, Any]:
    """CPU-only, JSON/hash-only reconciliation of all eight shard roots."""
    shard_names = sorted(path.name for path in (formal_root / "shards").iterdir() if path.is_dir())
    expected_names = [f"shard-{item:02d}" for item in SHARD_IDS]
    if shard_names != expected_names:
        raise StageYY2ScheduleError("shard roster must be exactly shard-00 through shard-07")
    terminals: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    all_attempt_records: list[dict[str, Any]] = []
    shard_terminals: list[dict[str, Any]] = []
    for shard_id, name in enumerate(expected_names):
        root = formal_root / "shards" / name
        manifest = read_json(root / "shard_manifest.json"); validate_signed(manifest)
        if manifest["payload"]["shard_id"] != shard_id or manifest["payload"]["selection_orders"] != expected_y2_orders(shard_id):
            raise StageYY2ScheduleError("shard manifest mapping drift")
        identity_doc = read_json(root / "worker_identity.json"); validate_signed(identity_doc)
        identities.append(worker_binding(identity_doc["payload"]))
        require_shard_terminal = root / "completion/shard.json"
        if not require_shard_terminal.is_file() or (root / "completion/shard.pending.json").exists():
            raise StageYY2ScheduleError(f"shard terminal transition incomplete in {name}")
        shard_terminal = read_json(require_shard_terminal); validate_signed(shard_terminal)
        if shard_terminal["document_type"] != "stage_y_y2_shard_terminal" or shard_terminal["status"] not in UNIT_TERMINAL_STATES or shard_terminal["payload"]["shard_id"] != shard_id:
            raise StageYY2ScheduleError(f"shard terminal invalid in {name}")
        if shard_terminal["payload"]["worker_identity"] != identity_doc["payload"]:
            raise StageYY2ProvenanceError(f"shard worker identity link drift in {name}")
        shard_terminals.append(shard_terminal)
        unit_paths = sorted(root.glob("units/unit-*/completion/unit.json"))
        if len(unit_paths) != len(expected_y2_orders(shard_id)):
            raise StageYY2ScheduleError(f"terminal count drift in {name}")
        for path in unit_paths:
            terminal = read_json(path); validate_signed(terminal)
            if terminal["document_type"] != "stage_y_y2_unit_terminal" or terminal["status"] not in UNIT_TERMINAL_STATES:
                raise StageYY2ProvenanceError("invalid accepted unit terminal")
            payload = terminal["payload"]
            if payload["shard_id"] != shard_id or y2_shard_id(payload["selection_order"]) != shard_id or payload["condition_id"] != "clean":
                raise StageYY2ScheduleError("terminal mapping or condition drift")
            if payload["environment_digest"] != identity_doc["payload"]["environment_digest"]:
                raise StageYY2ScheduleError("unit environment does not match its worker identity")
            observed_attempts = [file_record(item) for item in sorted(path.parents[1].glob("attempts/attempt-*.json"))]
            if payload["attempt_records"] != observed_attempts or not observed_attempts:
                raise StageYY2ProvenanceError("attempt retention drift")
            unit_root = path.parents[1].resolve()
            for array_record in payload["array_records"]:
                array_path = Path(str(array_record["path"]))
                if not array_path.is_absolute(): array_path = Path.cwd() / array_path
                array_path = array_path.resolve()
                try: array_path.relative_to(unit_root / "arrays")
                except ValueError as error: raise StageYY2ScheduleError("array path escaped its unit root") from error
                if not array_path.is_file() or int(array_record["bytes"]) != array_path.stat().st_size or array_record["sha256"] != sha_file(array_path):
                    raise StageYY2ProvenanceError("array source record drift")
                import numpy as np
                array = np.load(array_path, allow_pickle=False)
                if str(array.dtype) != array_record["dtype"] or list(array.shape) != array_record["shape"] or not array.flags.c_contiguous or not np.isfinite(array).all():
                    raise StageYY2ProvenanceError("scientific array dtype, shape, layout, or finite contract drift")
                digest = hashlib.sha256(); digest.update(str(array.dtype).encode("utf-8")); digest.update(b"\0"); digest.update(str(tuple(array.shape)).encode("utf-8")); digest.update(b"\0"); digest.update(np.ascontiguousarray(array).tobytes(order="C"))
                if digest.hexdigest() != array_record["canonical_digest"]:
                    raise StageYY2ProvenanceError("scientific array canonical digest drift")
            all_attempt_records.extend(observed_attempts)
            terminals.append(terminal)
        observed_counts = {state: sum(row["status"] == state for row in terminals if row["payload"]["shard_id"] == shard_id) for state in UNIT_TERMINAL_STATES}
        if shard_terminal["payload"]["terminal_counts"] != observed_counts or shard_terminal["payload"]["unit_count"] != sum(observed_counts.values()):
            raise StageYY2ProvenanceError(f"shard terminal counts are not source-recomputed in {name}")
    orders = [int(row["payload"]["selection_order"]) for row in terminals]
    if sorted(orders) != list(range(100)) or len(set(orders)) != 100:
        raise StageYY2ScheduleError("Y2 selection-order coverage is not exactly 0..99")
    for field in ("checkpoint_sha256", "environment_digest", "numeric_contract_digest"):
        if len({row["payload"][field] for row in terminals}) != 1:
            raise StageYY2ScheduleError(f"cross-worker {field} mismatch")
    if len({row["gpu_uuid"] for row in identities}) != 8 or len({row["pci_bus_id"] for row in identities}) != 8:
        raise StageYY2ScheduleError("worker GPU UUID or PCI identity is not unique")
    if len({row["physical_index"] for row in identities}) != 8 or len({row["cuda_visible_devices"] for row in identities}) != 8:
        raise StageYY2ScheduleError("physical GPU or CUDA_VISIBLE_DEVICES assignment is not unique")
    if len({row["environment_digest"] for row in identities}) != 1:
        raise StageYY2ScheduleError("worker environments are not homogeneous")
    totals = {"gpu_active_seconds": 0.0, "cpu_seconds": 0.0, "wall_seconds_sum": 0.0, "retained_bytes": 0, "peak_reserved_gpu_bytes_per_process": 0}
    fixed_initializations = []
    per_shard_resources = []
    for shard_id, shard_terminal in enumerate(shard_terminals):
        fixed = shard_terminal["payload"]["fixed_initialization_resource"]
        fixed_initializations.append(fixed)
        totals["gpu_active_seconds"] += float(fixed["gpu_active_seconds"])
        totals["cpu_seconds"] += float(fixed["cpu_seconds"])
        totals["wall_seconds_sum"] += float(fixed["wall_seconds"])
        totals["retained_bytes"] += int(fixed["retained_bytes"])
        totals["peak_reserved_gpu_bytes_per_process"] = max(totals["peak_reserved_gpu_bytes_per_process"], int(fixed["peak_reserved_gpu_bytes"]))
        shard_rows = [row for row in terminals if row["payload"]["shard_id"] == shard_id]
        shard_resource = {
            "shard_id": shard_id, "unit_count": len(shard_rows),
            "gpu_active_seconds": float(fixed["gpu_active_seconds"]) + sum(float(row["payload"]["resource_telemetry"]["gpu_active_seconds"]) for row in shard_rows),
            "cpu_seconds": float(fixed["cpu_seconds"]) + sum(float(row["payload"]["resource_telemetry"]["cpu_seconds"]) for row in shard_rows),
            "wall_seconds_sum": float(fixed["wall_seconds"]) + sum(float(row["payload"]["resource_telemetry"]["wall_seconds"]) for row in shard_rows),
            "worker_elapsed_seconds": float(shard_terminal["payload"]["shard_elapsed_seconds"]),
            "peak_reserved_gpu_bytes": max([int(fixed["peak_reserved_gpu_bytes"])] + [int(row["payload"]["resource_telemetry"]["peak_reserved_gpu_bytes"]) for row in shard_rows]),
        }
        per_shard_resources.append(shard_resource)
    for row in terminals:
        resource = row["payload"]["resource_telemetry"]
        totals["gpu_active_seconds"] += float(resource["gpu_active_seconds"])
        totals["cpu_seconds"] += float(resource["cpu_seconds"])
        totals["wall_seconds_sum"] += float(resource["wall_seconds"])
        totals["retained_bytes"] += int(resource["retained_bytes"])
        totals["peak_reserved_gpu_bytes_per_process"] = max(totals["peak_reserved_gpu_bytes_per_process"], int(resource["peak_reserved_gpu_bytes"]))
    valid_count = sum(row["status"] == "VALID" for row in terminals)
    terminal_counts = {state: sum(row["status"] == state for row in terminals) for state in UNIT_TERMINAL_STATES}
    actual_retained_bytes = sum(path.stat().st_size for path in (formal_root / "shards").rglob("*") if path.is_file())
    totals["retained_bytes"] = actual_retained_bytes
    cap_failures = []
    if totals["gpu_active_seconds"] / 3600.0 > 64.0: cap_failures.append("aggregate_h20_gpu_hours")
    if totals["cpu_seconds"] / 3600.0 > 218.0: cap_failures.append("aggregate_cpu_hours")
    if totals["retained_bytes"] / 1024.0 ** 3 > 6.0: cap_failures.append("aggregate_storage_gib")
    if totals["peak_reserved_gpu_bytes_per_process"] / 1024.0 ** 3 > 21.0: cap_failures.append("peak_reserved_gpu_gib_per_process")
    overall_stage_wall_seconds = None
    startup_gate_path = formal_root / "multi_gpu_startup_gate.json"
    if formal_root.as_posix().endswith(ROOT.as_posix()):
        if not startup_gate_path.is_file(): raise StageYY2ScheduleError("formal reconciliation requires the eight-device startup gate")
        startup_gate = read_json(startup_gate_path); validate_signed(startup_gate)
        overall_stage_wall_seconds = max(int(row["payload"]["completed_unix_ns"]) for row in shard_terminals) / 1e9 - int(startup_gate["payload"]["started_unix_ns"]) / 1e9
        if overall_stage_wall_seconds / 3600.0 > 12.0: cap_failures.append("overall_stage_wall_hours")
    for row in per_shard_resources:
        count = row["unit_count"]; gpu_cap = 8.4 if count == 13 else 7.7; cpu_cap = 29.0 if count == 13 else 27.0; wall_cap = 9.0 if count == 13 else 8.0
        if row["gpu_active_seconds"] / 3600.0 > gpu_cap: cap_failures.append(f"shard_{row['shard_id']:02d}_gpu_hours")
        if row["cpu_seconds"] / 3600.0 > cpu_cap: cap_failures.append(f"shard_{row['shard_id']:02d}_cpu_hours")
        if row["worker_elapsed_seconds"] / 3600.0 > wall_cap: cap_failures.append(f"shard_{row['shard_id']:02d}_critical_path_wall_hours")
    if cap_failures or terminal_counts["BLOCKED_BUDGET"]:
        stage_status = "BLOCKED_BUDGET"
    elif terminal_counts["BLOCKED_INFRASTRUCTURE"]:
        stage_status = "BLOCKED_INFRASTRUCTURE"
    elif valid_count < minimum_valid:
        stage_status = "BLOCKED_SCIENTIFIC_VALIDITY"
    else:
        stage_status = "PASS"
    summary = {
        "status": stage_status,
        "shard_roster": expected_names, "selection_orders": sorted(orders),
        "unit_count": len(terminals), "valid_count": valid_count,
        "terminal_counts": terminal_counts,
        "attempt_record_count": len(all_attempt_records),
        "fixed_initialization_records": fixed_initializations,
        "per_shard_resources": per_shard_resources,
        "resource_totals": {**totals, "h20_gpu_hours": totals["gpu_active_seconds"] / 3600.0, "cpu_hours": totals["cpu_seconds"] / 3600.0, "retained_storage_gib": totals["retained_bytes"] / 1024.0 ** 3, "worker_critical_path_seconds": max(row["worker_elapsed_seconds"] for row in per_shard_resources)},
        "overall_stage_wall_seconds": overall_stage_wall_seconds,
        "gpu_hours_accounting": "SUM_ACROSS_ALL_SHARDS_NOT_DIVIDED_BY_EIGHT",
        "budget_cap_failures": cap_failures,
        "conditions_observed": ["clean"],
        "failures_and_na_preserved": True,
        "bootstrap_executed": False, "holm_executed": False, "claim_decision_executed": False,
        "next_authorization": "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY" if stage_status == "PASS" else "NONE",
    }
    return signed("stage_y_y2_reconciliation", stage_status, summary)
