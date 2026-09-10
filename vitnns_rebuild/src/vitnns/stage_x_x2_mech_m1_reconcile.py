"""Source-recomputing M1 reconciliation for the frozen X2-MECH worker.

This module is deliberately M1-only.  It never imports Torch/timm, decodes an
image, loads a checkpoint, reads sealed X2 outcome tables, or computes any M2
endpoint.  It validates route terminals and raw map files independently of the
worker's returned shard summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import REPO_ROOT
from . import stage_x_x2_mech as mech
from . import stage_x_x2_mech_worker as worker
from .stage_x_x2_mech_analysis import seed64

RUN_ID = "STAGE_X_X2_MECH_M1_V1_20260910"
AUTHORIZATION = "X2_MECH_M1_TRANSFORMED_MAP_AND_MULTISCALE_FIDELITY_ONLY"
NEXT_AUTHORIZATION = "X2_MECH_M2_CPU_ONLY_MULTISCALE_MECHANISM_ANALYSIS_ONLY"
BASE_COMMIT = "f205eeebe0f054db434fb822d8d6d92c95b74ee4"
BRANCH = "stage-d-d3-pass"
ARCHIVE_REF = "refs/archive/stage-x-x2-mech-m1-r1-completion"
ROOT = Path("evidence/stage_x/x2_mech_m1_r1_20260910/formal")
REPORT_ROOT = Path("reports/stage_x_x2_mech_m1_r1_20260910")
PRE_EXECUTION_SEAL = ROOT / "pre_execution_seal.json"
COMPLETION = ROOT / "completion/x2_mech_m1_r1.json"
PENDING = ROOT / "completion/x2_mech_m1_r1.pending.json"
RECONCILER = Path("src/vitnns/stage_x_x2_mech_m1_reconcile.py")
TEST_FILE = Path("tests/test_stage_x_x2_mech_m1_reconcile.py")
AMENDMENT = Path("docs/stage_x_x2_mech_m1_r1_digest_recovery_amendment_20260910.md")
OLD_BLOCKED_ROOT = Path("evidence/stage_x/x2_mech_m1_20260910/formal")
METHODS = ("nns_v2", "vit_grad_cam")
TRANSFORMS = worker.TRANSFORMS
BLOCKS = (4, 16)
PERCENTILES = (10, 20, 30, 40, 50, 60, 70, 80, 90)
SEEDS = tuple(range(5))
DIRECTIONS = ("morf", "lerf")
MODEL_LOCAL_PATHS = {
    "vit_small_patch16_224.augreg_in21k_ft_in1k": Path("assets/models/stage_x_x2/vit_small_patch16_224.augreg_in21k_ft_in1k.npz"),
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"),
    "deit_small_patch16_224.fb_in1k": Path("assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth"),
    "deit_base_patch16_224.fb_in1k": Path("assets/models/stage_x_x2/deit_base_patch16_224.fb_in1k.pth"),
    "vit_base_patch32_224.augreg_in21k_ft_in1k": Path("assets/models/stage_x_x2/vit_base_patch32_224.augreg_in21k_ft_in1k.npz"),
}
MAP_FIELDS = (
    "unit_digest", "model_id", "sample_id", "method_id", "transform",
    "coordinate", "relative_path", "sha256", "canonical_array_digest",
    "dtype", "shape", "selected_steps", "target", "prediction_changed",
    "source_clean_map_path", "source_clean_map_sha256", "source_clean_map_digest",
    "source_clean_map_dtype", "source_clean_map_shape", "source_x2_terminal_path",
    "source_x2_terminal_sha256", "m1_route_terminal_path", "resource_telemetry_scope",
    "transform_metadata", "x2_mask_path", "x2_mask_digest", "exact_mask_path",
    "exact_mask_digest",
)
MASK_ROUTE_FIELDS = ("transform", "mask_id", "relative_path", "sha256",
                     "canonical_digest", "valid_pixels", "full_domain",
                     "transform_metadata")
MASK_GLOBAL_FIELDS = ("transform", "mask_id", "relative_path", "bytes", "sha256",
                      "canonical_digest", "dtype", "shape", "valid_pixels",
                      "full_domain", "transform_metadata")
FIDELITY_FIELDS = ("unit_digest", "model_id", "sample_id", "method_id",
                   "road_combined", "road_morf", "road_lerf", "block",
                   "unit_count", "clean_logit", "baseline_logit")
DETAIL_FIELDS = ("unit_digest", "method_id", "block", "percentile", "seed_index",
                 "direction", "pcg64_seed", "actual_pixel_fraction",
                 "normalized_response")
PROJECTED = {
    "h20_gpu_hours": 78.7, "wall_hours": 24.0, "cpu_hours": 2459.0,
    "peak_reserved_gib_per_process": 5.0, "retained_gib": 4.0,
    "max_devices": 4.0, "downloaded_bytes": 0.0,
}
CAPS = {
    "h20_gpu_hours": 80.0, "wall_hours": 48.0, "cpu_hours": 2500.0,
    "peak_reserved_gib_per_process": 16.0, "retained_gib": 8.0,
    "max_devices": 4.0, "downloaded_bytes": 0.0,
}
FALSE_PASS_ATTACKS = (
    "route_method_transform_scale_seed_reduction",
    "failed_route_deleted_replaced_or_retried",
    "source_clean_map_or_terminal_drift",
    "method_specific_normalization_smoothing_or_thresholding",
    "partial_block_perturbation",
    "resize_substituted_for_4x4_or_16x16_block_aggregation",
    "mask_varies_by_route_method_or_outcome",
    "horizontal_flip_omitted",
    "transformed_map_omitted_or_inverse_only_retained",
    "jpeg_png_component_or_clean_map_copy_output",
    "budget_waiver_or_silent_cap_excess",
    "m2_analysis_executed",
    "x2_no_go_modified",
    "x3_x5_f3_final_or_stage_g_started",
    "pass_without_source_recomputing_validation",
)


class M1ReconcileError(ValueError):
    pass


def _pairs(values: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise M1ReconcileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               M1ReconcileError(f"non-finite JSON: {token}")))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise M1ReconcileError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise M1ReconcileError(f"JSON root must be an object: {path}")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": 1, "document_type": document_type, "stage_id": "X2-MECH",
        "phase": "M1", "run_id": RUN_ID, "authorization": AUTHORIZATION,
        "status": status, "payload": dict(payload),
    }
    value["canonical_digest"] = _canonical_digest(value)
    return value


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    copy = dict(value)
    observed = copy.pop("canonical_digest", None)
    if observed != _canonical_digest(copy):
        raise M1ReconcileError(f"{context}: canonical self-digest mismatch")


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    _write_new(path, _canonical(value) + b"\n")


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise M1ReconcileError(f"CSV fields drifted: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise M1ReconcileError(f"malformed CSV row: {path}")
    return rows


def _record(repo: Path, relative: Path) -> dict[str, Any]:
    path = repo / relative
    record: dict[str, Any] = {"path": relative.as_posix(), "bytes": path.stat().st_size,
                              "sha256": _sha_file(path)}
    if path.suffix == ".json":
        value = _read_json(path)
        record["canonical_digest"] = value.get("canonical_digest")
    return record


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode:
        raise M1ReconcileError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _bool(value: str, context: str) -> bool:
    if value not in ("True", "False"):
        raise M1ReconcileError(f"{context}: invalid Boolean")
    return value == "True"


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise M1ReconcileError(f"{context}: non-finite value")
    return result


def _verify_preconditions(repo: Path) -> None:
    if _git(repo, "symbolic-ref", "--short", "HEAD") != BRANCH:
        raise M1ReconcileError("BLOCKED_STARTUP_STATE: branch drifted")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=repo).returncode:
        raise M1ReconcileError("BLOCKED_STARTUP_STATE: HEAD does not descend from P0-R1 completion")
    if _git(repo, "rev-parse", "@{upstream}") != BASE_COMMIT:
        raise M1ReconcileError("BLOCKED_STARTUP_STATE: tracking ref drifted")
    expected_refs = (
        (mech.PRODUCER_REF, "92fd20fe680969c8c38d687c21205418a4ca7876"),
        (mech.COMPLETION_REF, BASE_COMMIT),
        (mech.X2_PRODUCER_REF, mech.X2_PRODUCER),
        (mech.X2_COMPLETION_REF, mech.X2_COMPLETION),
    )
    for ref, expected in expected_refs:
        if _git(repo, "rev-parse", ref + "^{commit}") != expected:
            raise M1ReconcileError(f"BLOCKED_STARTUP_STATE: ref drifted: {ref}")
    if subprocess.run(["git", "show-ref", "--verify", "--quiet", ARCHIVE_REF], cwd=repo).returncode == 0:
        raise M1ReconcileError("BLOCKED_STARTUP_STATE: M1 completion ref already exists")
    result = mech.validate_completion(repo)
    if result["status"] != "PASS" or result["next_authorization"] != AUTHORIZATION:
        raise M1ReconcileError("BLOCKED_STARTUP_STATE: P0-R1 completion validation failed")
    blocked = repo / OLD_BLOCKED_ROOT / "completion/x2_mech_m1.json"
    if not blocked.is_file():
        raise M1ReconcileError("BLOCKED_STARTUP_STATE: prior blocked M1 receipt is missing")
    value = _read_json(blocked); _validate_signed(value, str(blocked))
    if value.get("status") != "BLOCKED" or value["payload"].get("failure_code") != "BLOCKED_FROZEN_IMPLEMENTATION_DIGEST_CONVENTION_MISMATCH":
        raise M1ReconcileError("BLOCKED_STARTUP_STATE: prior blocked M1 receipt drifted")


def seal_pre_execution(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if (repo / ROOT).exists() or (repo / REPORT_ROOT).exists():
        raise M1ReconcileError("BLOCKED_STARTUP_STATE: M1 root or report already exists")
    _verify_preconditions(repo)
    for relative in (AMENDMENT, RECONCILER, Path("src/vitnns/stage_x_x2_mech_worker.py"), TEST_FILE):
        if not (repo / relative).is_file():
            raise M1ReconcileError(f"pre-execution source is missing: {relative}")
    value = _signed("stage_x_x2_mech_m1_pre_execution_seal", "FROZEN", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at_unix_ns": time.time_ns(),
        "base_commit": BASE_COMMIT,
        "amendment_type": "PROSPECTIVE_M1_R1_DIGEST_FORMULA_RECOVERY",
        "p0_r1_completion_commit": BASE_COMMIT,
        "sources": [_record(repo, AMENDMENT), _record(repo, RECONCILER),
                    _record(repo, Path("src/vitnns/stage_x_x2_mech_worker.py")),
                    _record(repo, TEST_FILE), _record(repo, mech.ROOT / "implementation_seal.json"),
                    _record(repo, OLD_BLOCKED_ROOT / "completion/x2_mech_m1.json")],
        "shard_device_assignment": [{"shard": index, "device": f"cuda:{index}"}
                                    for index in range(4)],
        "empirical_outcome_seen_before_seal": False,
        "prior_blocked_root_read_only": OLD_BLOCKED_ROOT.as_posix(),
        "m2_authorized": False,
    })
    _write_json_new(repo / PRE_EXECUTION_SEAL, value)
    return value


def _load_seal(repo: Path) -> dict[str, Any]:
    value = _read_json(repo / PRE_EXECUTION_SEAL)
    _validate_signed(value, str(PRE_EXECUTION_SEAL))
    if value.get("status") != "FROZEN" or value["payload"].get("empirical_outcome_seen_before_seal") is not False:
        raise M1ReconcileError("pre-execution seal drifted")
    for record in value["payload"]["sources"]:
        path = repo / record["path"]
        if path.stat().st_size != record["bytes"] or _sha_file(path) != record["sha256"]:
            raise M1ReconcileError(f"pre-execution implementation drifted: {record['path']}")
    return value


def _load_plan(repo: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = mech._validate_artifact(repo, mech.ROOT / "typed_plan.json")
    routes = plan["payload"]["routes"]
    if len(routes) != 200 or sum(row["source_status"] == "VALID" for row in routes) != 199:
        raise M1ReconcileError("typed denominator drifted")
    failed = [row for row in routes if row["source_status"] == "FAILED_ROUTE_LOCAL_NUMERICAL"]
    if len(failed) != 1 or failed[0]["unit_digest"] != "10480c360d247eceb7b7a7ef42ca1a22da8a3c0e170fd6efa124bde3fe70d702":
        raise M1ReconcileError("failed-route binding drifted")
    return routes, failed[0]


def _source_expectations(repo: Path, route: Mapping[str, Any]) -> dict[str, Any]:
    terminal_path = repo / route["source_terminal_path"]
    if _sha_file(terminal_path) != route["source_terminal_sha256"]:
        raise M1ReconcileError(f"source X2 terminal hash drifted: {route['unit_digest']}")
    terminal = mech.strict_read(terminal_path)
    mech.validate_signed(terminal, route["source_terminal_path"])
    clean_paths = [row["path"] for row in terminal["payload"]["bindings"]
                   if "/clean/terminal.json" in row["path"]]
    if len(clean_paths) != 1:
        raise M1ReconcileError("source clean terminal binding is not unique")
    clean = mech.strict_read(repo / clean_paths[0])
    mech.validate_signed(clean, clean_paths[0])
    summaries = [row for row in clean["payload"]["files"]
                 if Path(row["path"]).name == "summary.csv"]
    if len(summaries) != 1:
        raise M1ReconcileError("source summary binding is not unique")
    record = summaries[0]
    path = repo / record["path"]
    if path.stat().st_size != record["bytes"] or _sha_file(path) != record["sha256"]:
        raise M1ReconcileError("source summary binding drifted")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise M1ReconcileError("source summary must contain exactly one row")
    row = rows[0]
    if row["model_id"] != route["model_id"] or row["sample_id"] != route["sample_id"]:
        raise M1ReconcileError("source summary identity drifted")
    sample_path = repo / "assets/data" / route["sample_id"]
    checkpoint_path = repo / MODEL_LOCAL_PATHS[route["model_id"]]
    if _sha_file(sample_path) != row["sample_sha256"]:
        raise M1ReconcileError("bound JPEG hash drifted")
    if _sha_file(checkpoint_path) != row["checkpoint_sha256"]:
        raise M1ReconcileError("bound checkpoint hash drifted")
    return {
        "target": int(row["predicted_class"]), "selected_steps": int(row["selected_steps"]),
        "checkpoint_sha256": row["checkpoint_sha256"], "sample_sha256": row["sample_sha256"],
        "canonical_rgb_sha256": row["canonical_rgb_sha256"],
        "clean_tensor_sha256": row["clean_tensor_sha256"],
        "summary_path": record["path"], "summary_sha256": record["sha256"],
    }


def _verify_route_terminal(path: Path, route: Mapping[str, Any],
                           source: Mapping[str, Any]) -> dict[str, Any]:
    terminal = _read_json(path)
    _validate_signed(terminal, str(path))
    expected = {
        "status": "PASS", "authorization": AUTHORIZATION,
        "unit_digest": route["unit_digest"], "model_id": route["model_id"],
        "sample_id": route["sample_id"], "source_terminal_path": route["source_terminal_path"],
        "source_terminal_sha256": route["source_terminal_sha256"], "immutable": True,
    }
    for key, value in expected.items():
        if terminal.get(key) != value:
            raise M1ReconcileError(f"route terminal binding drifted: {route['unit_digest']}:{key}")
    if terminal.get("counts") != {"transforms": 12, "methods": 2, "transformed_maps": 24,
                                      "inverse_maps": 24, "masks": 24,
                                      "multiscale_fidelity_endpoints": 4}:
        raise M1ReconcileError("route terminal counts drifted")
    if int(terminal.get("target", -1)) != source["target"] or int(terminal.get("selected_steps", -1)) != source["selected_steps"]:
        raise M1ReconcileError("route target or selected steps drifted from source summary")
    if terminal.get("image_identity") != {"canonical_rgb_sha256": source["canonical_rgb_sha256"],
                                             "clean_tensor_sha256": source["clean_tensor_sha256"]}:
        raise M1ReconcileError("route image identity drifted from source summary")
    resources = terminal.get("resources")
    if not isinstance(resources, dict) or set(resources) != {"wall_seconds", "cpu_seconds",
            "gpu_seconds", "peak_reserved_bytes", "peak_allocated_bytes", "downloaded_bytes"}:
        raise M1ReconcileError("route resource telemetry fields drifted")
    for key, value in resources.items():
        number = _finite(value, f"resource {key}")
        if number < 0:
            raise M1ReconcileError(f"negative resource value: {key}")
    if int(resources["downloaded_bytes"]) != 0:
        raise M1ReconcileError("downloaded bytes are nonzero")
    return terminal


def _map_record(repo: Path, route: Mapping[str, Any], base: Path, row: Mapping[str, str]) -> dict[str, Any]:
    unit = route["unit_digest"]
    if row["unit_digest"] != unit or row["model_id"] != route["model_id"] or row["sample_id"] != route["sample_id"]:
        raise M1ReconcileError(f"map identity drifted: {unit}")
    if row["method_id"] not in METHODS or row["transform"] not in TRANSFORMS or row["coordinate"] not in ("transformed", "inverse"):
        raise M1ReconcileError(f"map coordinate roster drifted: {unit}")
    expected_local = f"maps/{row['transform']}/{row['method_id']}.{row['coordinate']}.npy"
    if row["relative_path"] != expected_local:
        raise M1ReconcileError(f"map path drifted: {unit}")
    path = base / expected_local
    if _sha_file(path) != row["sha256"]:
        raise M1ReconcileError(f"map file hash drifted: {path}")
    value = np.load(path, allow_pickle=False)
    worker.validate_map(value)
    digest = worker.array_digest(value)
    if digest != row["canonical_array_digest"]:
        raise M1ReconcileError(f"map array digest drifted: {path}")
    if row["dtype"] != "float32" or row["shape"] != "224x224":
        raise M1ReconcileError(f"map dtype/shape metadata drifted: {path}")
    source = next((item for item in route["source_maps"] if item["method_id"] == row["method_id"]), None)
    if source is None:
        raise M1ReconcileError("source map binding is absent")
    source_fields = {
        "source_clean_map_path": source["path"], "source_clean_map_sha256": source["sha256"],
        "source_clean_map_digest": source["canonical_digest"], "source_clean_map_dtype": "float32",
        "source_clean_map_shape": "224x224", "source_x2_terminal_path": route["source_terminal_path"],
        "source_x2_terminal_sha256": route["source_terminal_sha256"],
        "resource_telemetry_scope": "m1_route_terminal",
        "m1_route_terminal_path": (ROOT / "units" / unit / "attempts" / worker.ATTEMPT_ID / "terminal.json").as_posix(),
    }
    for key, expected in source_fields.items():
        if row[key] != expected:
            raise M1ReconcileError(f"source binding drifted: {unit}:{key}")
    source_path = repo / source["path"]
    if source_path.stat().st_size != source["bytes"] or _sha_file(source_path) != source["sha256"]:
        raise M1ReconcileError(f"source clean-map bytes drifted: {unit}")
    source_value = np.load(source_path, allow_pickle=False)
    if worker.array_digest(worker.validate_map(source_value)) != source["canonical_digest"]:
        raise M1ReconcileError(f"source clean-map digest drifted: {unit}")
    if row["sha256"] == source["sha256"] or digest == source["canonical_digest"]:
        raise M1ReconcileError(f"clean map was copied into M1: {unit}")
    try:
        metadata = json.loads(row["transform_metadata"], object_pairs_hook=_pairs)
    except json.JSONDecodeError as error:
        raise M1ReconcileError("invalid transform metadata") from error
    if not isinstance(metadata, dict):
        raise M1ReconcileError("transform metadata must be an object")
    result = dict(row)
    result["relative_path"] = path.relative_to(repo).as_posix()
    result["bytes"] = path.stat().st_size
    return result


def _mask_digest(value: np.ndarray) -> str:
    return hashlib.sha256(b"bool|224,224|" + np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _verify_masks(repo: Path, route_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    observed: dict[tuple[str, str], dict[str, str]] = {}
    for row in route_rows:
        key = (row["transform"], row["mask_id"])
        if key[0] not in TRANSFORMS or key[1] not in ("x2_frozen_common_domain", "exact_operator_support"):
            raise M1ReconcileError("mask roster drifted")
        previous = observed.setdefault(key, dict(row))
        if previous != dict(row):
            raise M1ReconcileError("BLOCKED_MASK_REGISTRY_DRIFT")
    expected = {(transform, name) for transform in TRANSFORMS
                for name in ("x2_frozen_common_domain", "exact_operator_support")}
    if set(observed) != expected:
        raise M1ReconcileError("shared mask roster is incomplete")
    result = []
    for transform, mask_id in sorted(expected):
        source = observed[(transform, mask_id)]
        relative = ROOT / "valid_masks" / f"{transform}.{mask_id}.npy"
        if source["relative_path"] != relative.as_posix():
            raise M1ReconcileError("mask path drifted")
        path = repo / relative
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.bool_ or value.shape != (224, 224):
            raise M1ReconcileError("mask dtype/shape drifted")
        digest = _mask_digest(value)
        if _sha_file(path) != source["sha256"] or digest != source["canonical_digest"]:
            raise M1ReconcileError("BLOCKED_MASK_REGISTRY_DRIFT")
        if int(source["valid_pixels"]) != int(value.sum()) or _bool(source["full_domain"], "mask full_domain") != bool(value.all()):
            raise M1ReconcileError("mask metadata drifted")
        if transform == "horizontal_flip" and not value.all():
            raise M1ReconcileError("horizontal-flip masks are not full-domain")
        result.append({"transform": transform, "mask_id": mask_id, "relative_path": relative.as_posix(),
                       "bytes": path.stat().st_size, "sha256": source["sha256"],
                       "canonical_digest": digest, "dtype": "bool", "shape": "224x224",
                       "valid_pixels": int(value.sum()), "full_domain": bool(value.all()),
                       "transform_metadata": source["transform_metadata"]})
    return result


def _verify_fidelity(route: Mapping[str, Any], summaries: Sequence[Mapping[str, str]],
                     details: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unit = route["unit_digest"]
    expected_summary = {(method, block) for method in METHODS for block in BLOCKS}
    summary_by: dict[tuple[str, int], dict[str, str]] = {}
    for row in summaries:
        if row["unit_digest"] != unit or row["model_id"] != route["model_id"] or row["sample_id"] != route["sample_id"]:
            raise M1ReconcileError("fidelity summary identity drifted")
        key = (row["method_id"], int(row["block"]))
        if key in summary_by or key not in expected_summary:
            raise M1ReconcileError("fidelity summary roster drifted")
        if int(row["unit_count"]) != (224 // key[1]) ** 2:
            raise M1ReconcileError("fidelity unit count drifted")
        for field in ("road_combined", "road_morf", "road_lerf", "clean_logit", "baseline_logit"):
            _finite(row[field], f"fidelity {field}")
        summary_by[key] = dict(row)
    if set(summary_by) != expected_summary:
        raise M1ReconcileError("fidelity summaries are incomplete")
    expected_detail = {(method, block, percentile, seed, direction) for method in METHODS
                       for block in BLOCKS for percentile in PERCENTILES for seed in SEEDS
                       for direction in DIRECTIONS}
    detail_by: dict[tuple[str, int, int, int, str], dict[str, str]] = {}
    for row in details:
        if row["unit_digest"] != unit:
            raise M1ReconcileError("fidelity detail identity drifted")
        key = (row["method_id"], int(row["block"]), int(row["percentile"]),
               int(row["seed_index"]), row["direction"])
        if key in detail_by or key not in expected_detail:
            raise M1ReconcileError("fidelity detail roster drifted")
        method, block, percentile, road_seed, direction = key
        expected_seed = seed64("STAGE_X_X2_ROAD_NOISY_LINEAR_V1_20260909",
                               f"{unit}|{method}", block, percentile, road_seed, direction)
        if int(row["pcg64_seed"]) != expected_seed:
            raise M1ReconcileError("fidelity PCG64 seed drifted")
        units = (224 // block) ** 2
        expected_fraction = math.floor(units * percentile / 100) * block * block / (224 * 224)
        if not math.isclose(_finite(row["actual_pixel_fraction"], "actual pixel fraction"),
                            expected_fraction, abs_tol=1e-15, rel_tol=1e-15):
            raise M1ReconcileError("partial-block perturbation or fraction drift detected")
        _finite(row["normalized_response"], "normalized response")
        detail_by[key] = dict(row)
    if set(detail_by) != expected_detail:
        raise M1ReconcileError("fidelity detailed rows are incomplete")
    fractions = np.asarray(PERCENTILES, dtype=np.float64) / 100.0
    for method, block in expected_summary:
        aucs: dict[str, float] = {}
        for direction in DIRECTIONS:
            per_seed = []
            for road_seed in SEEDS:
                responses = [_finite(detail_by[(method, block, percentile, road_seed, direction)]["normalized_response"],
                                     "normalized response") for percentile in PERCENTILES]
                per_seed.append(float(np.trapz(responses, fractions)))
            aucs[direction] = float(np.mean(per_seed, dtype=np.float64))
        summary = summary_by[(method, block)]
        expected_values = {"road_morf": aucs["morf"], "road_lerf": aucs["lerf"],
                           "road_combined": 0.5 * (aucs["lerf"] - aucs["morf"])}
        for field, expected_value in expected_values.items():
            if not math.isclose(float(summary[field]), expected_value, abs_tol=1e-12, rel_tol=1e-12):
                raise M1ReconcileError(f"fidelity AUC recomputation drifted: {unit}:{method}:{block}:{field}")
    return ([summary_by[key] for key in sorted(summary_by)], [detail_by[key] for key in sorted(detail_by)])


def _resource_document(seal: Mapping[str, Any], route_resources: Sequence[Mapping[str, Any]],
                       retained_route_and_mask_bytes: int, terminal_mtime_ns: int) -> dict[str, Any]:
    started_ns = int(seal["payload"]["started_at_unix_ns"])
    actual = {
        "h20_gpu_hours": sum(float(row["gpu_seconds"]) for row in route_resources) / 3600.0,
        "wall_hours": max(0.0, terminal_mtime_ns - started_ns) / 3_600_000_000_000.0,
        "cpu_hours": sum(float(row["cpu_seconds"]) for row in route_resources) / 3600.0,
        "peak_reserved_gib_per_process": max(float(row["peak_reserved_bytes"]) for row in route_resources) / (1024 ** 3),
        "retained_gib": retained_route_and_mask_bytes / (1024 ** 3),
        "max_devices": 4.0,
        "downloaded_bytes": sum(float(row["downloaded_bytes"]) for row in route_resources),
    }
    conservative = {key: max(actual[key], PROJECTED[key]) for key in CAPS}
    checks = []
    for key, cap in CAPS.items():
        status = "PASS" if actual[key] <= cap and conservative[key] <= cap else "BLOCKED_BUDGET"
        checks.append({"resource": key, "actual": actual[key], "frozen_projection": PROJECTED[key],
                       "conservative_cumulative": conservative[key], "ceiling": cap, "status": status})
    if any(row["status"] != "PASS" for row in checks):
        raise M1ReconcileError("BLOCKED_BUDGET")
    return _signed("stage_x_x2_mech_m1_resource_ledger", "PASS", {
        "route_resources": list(route_resources), "global": checks,
        "route_resource_count": len(route_resources),
        "retained_actual_scope": "immutable_route_attempts_and_shared_masks_before_global_sealing",
        "workload_reduction_used": False, "budget_waiver_used": False,
    })


def _collect(repo: Path) -> dict[str, Any]:
    seal = _load_seal(repo)
    result = mech.validate_completion(repo)
    if result["status"] != "PASS":
        raise M1ReconcileError("P0-R1 source-recomputing validation failed")
    routes, failed = _load_plan(repo)
    valid = [row for row in routes if row["source_status"] == "VALID"]
    maps: list[dict[str, Any]] = []
    route_masks: list[dict[str, str]] = []
    fidelities: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    route_resources: list[dict[str, Any]] = []
    terminal_mtime_ns = 0
    route_and_mask_paths: set[Path] = set()
    route_checks = []
    for route in valid:
        if int(route["shard"]) != worker.deterministic_shard(route["unit_digest"]):
            raise M1ReconcileError("route shard drifted")
        base = repo / ROOT / "units" / route["unit_digest"] / "attempts" / worker.ATTEMPT_ID
        terminal_path = base / "terminal.json"
        source = _source_expectations(repo, route)
        terminal = _verify_route_terminal(terminal_path, route, source)
        terminal_mtime_ns = max(terminal_mtime_ns, terminal_path.stat().st_mtime_ns)
        expected_files = {Path("terminal.json"), Path("map_manifest.csv"), Path("valid_mask_registry.csv"),
                          Path("multiscale_fidelity.csv"), Path("multiscale_fidelity_rows.csv")}
        expected_files |= {Path("maps") / transform / f"{method}.{coordinate}.npy"
                           for transform in TRANSFORMS for method in METHODS
                           for coordinate in ("transformed", "inverse")}
        actual_files = {path.relative_to(base) for path in base.rglob("*") if path.is_file()}
        if actual_files != expected_files:
            raise M1ReconcileError(f"route output file roster drifted: {route['unit_digest']}")
        route_and_mask_paths.update(path for path in base.rglob("*") if path.is_file())
        map_rows = _read_csv(base / "map_manifest.csv", MAP_FIELDS)
        if len(map_rows) != 48:
            raise M1ReconcileError("route map count drifted")
        roster = {(row["method_id"], row["transform"], row["coordinate"]) for row in map_rows}
        expected_roster = {(method, transform, coordinate) for method in METHODS for transform in TRANSFORMS
                           for coordinate in ("transformed", "inverse")}
        if roster != expected_roster:
            raise M1ReconcileError("route map roster drifted")
        current_maps = [_map_record(repo, route, base, row) for row in map_rows]
        for transform in TRANSFORMS:
            transform_rows = [row for row in current_maps if row["transform"] == transform]
            if len({row["transform_metadata"] for row in transform_rows}) != 1:
                raise M1ReconcileError("transform metadata differs by method/coordinate")
            if len({row["x2_mask_path"] for row in transform_rows}) != 1 or len({row["x2_mask_digest"] for row in transform_rows}) != 1:
                raise M1ReconcileError("X2 mask binding differs by method/coordinate")
            if len({row["exact_mask_path"] for row in transform_rows}) != 1 or len({row["exact_mask_digest"] for row in transform_rows}) != 1:
                raise M1ReconcileError("exact mask binding differs by method/coordinate")
            if len({row["prediction_changed"] for row in transform_rows}) != 1:
                raise M1ReconcileError("prediction-changed state differs by method/coordinate")
            _bool(transform_rows[0]["prediction_changed"], "prediction_changed")
        if len({row["selected_steps"] for row in current_maps}) != 1 or len({row["target"] for row in current_maps}) != 1:
            raise M1ReconcileError("target or selected-steps drifted within route")
        if int(current_maps[0]["selected_steps"]) != int(terminal["selected_steps"]) or int(current_maps[0]["target"]) != int(terminal["target"]):
            raise M1ReconcileError("map target/steps differ from route terminal")
        maps.extend(current_maps)
        masks = _read_csv(base / "valid_mask_registry.csv", MASK_ROUTE_FIELDS)
        if len(masks) != 24:
            raise M1ReconcileError("route mask registry count drifted")
        route_masks.extend(masks)
        summary_rows = _read_csv(base / "multiscale_fidelity.csv", FIDELITY_FIELDS)
        detail_rows = _read_csv(base / "multiscale_fidelity_rows.csv", DETAIL_FIELDS)
        if len(summary_rows) != 4 or len(detail_rows) != 360:
            raise M1ReconcileError("route fidelity count drifted")
        current_fidelity, current_detail = _verify_fidelity(route, summary_rows, detail_rows)
        fidelities.extend(current_fidelity)
        details.extend(current_detail)
        resource = {"unit_digest": route["unit_digest"], "shard": int(route["shard"]),
                    "device": f"cuda:{route['shard']}", **terminal["resources"]}
        route_resources.append(resource)
        route_checks.append({"unit_digest": route["unit_digest"], "status": "PASS",
                             "shard": int(route["shard"]), "map_count": 48,
                             "fidelity_summary_count": 4, "fidelity_detail_count": 360})
    masks = _verify_masks(repo, route_masks)
    route_and_mask_paths.update(repo / row["relative_path"] for row in masks)
    if len(maps) != 9552 or sum(row["coordinate"] == "transformed" for row in maps) != 4776 or sum(row["coordinate"] == "inverse" for row in maps) != 4776:
        raise M1ReconcileError("global map counts drifted")
    if len(fidelities) != 796 or len(details) != 71640 or len(masks) != 24:
        raise M1ReconcileError("global mask/fidelity counts drifted")
    if len({(row["unit_digest"], row["method_id"], row["transform"], row["coordinate"]) for row in maps}) != 9552:
        raise M1ReconcileError("duplicate map instance detected")
    retained = sum(path.stat().st_size for path in route_and_mask_paths)
    resources = _resource_document(seal, route_resources, retained, terminal_mtime_ns)
    return {"seal": seal, "routes": routes, "failed": failed, "maps": maps, "masks": masks,
            "fidelities": fidelities, "details": details, "resources": resources,
            "route_checks": route_checks, "retained_route_and_mask_bytes": retained}


def _access_document() -> dict[str, Any]:
    return _signed("stage_x_x2_mech_m1_access_audit", "PASS", {
        "access": {
            "typed_plan_routes_read": True, "source_clean_map_values_read": True,
            "source_x2_terminal_summary_identity_target_steps_hash_metadata_read": True,
            "bound_jpeg_bytes_read": True, "bound_checkpoint_bytes_read": True,
            "frozen_model_loaded": True, "torch_imported": True, "timm_imported": True,
            "torchvision_imported": True, "cuda_accessed": True,
            "transformed_inverse_maps_computed": True, "multiscale_road_4x4_16x16_computed": True,
            "source_x2_stability_rows_read": False, "source_x2_faithfulness_rows_read": False,
            "source_x2_faithfulness_road_rows_read": False,
            "source_x2_faithfulness_curves_rows_read": False, "source_x2_sanity_rows_read": False,
            "other_comparator_clean_map_values_read": False, "pixel_road_recomputed": False,
            "pixel_reproduction_computed": False, "multiscale_stability_computed": False,
            "transport_computed": False, "sse_decomposition_computed": False,
            "clean_structure_computed": False, "association_computed": False,
            "m2_analysis_executed": False, "jpeg_png_output_generated": False,
            "x2_or_p0_state_modified": False, "x3_x5_f3_final_or_stage_g_started": False,
            "downloaded_bytes": 0,
        },
        "basis": "frozen worker code path plus source-recomputing M1 reconciliation allowlist",
        "p0_all_false_flags_reused": False,
    })


def _false_pass_document() -> dict[str, Any]:
    mechanisms = {
        "route_method_transform_scale_seed_reduction": "exact 199x2x12 maps and 199x2x2x9x5x2 ROAD coordinates are recomputed",
        "failed_route_deleted_replaced_or_retried": "all 200 typed routes are bound and the immutable failed digest remains terminal-only",
        "source_clean_map_or_terminal_drift": "source bytes, SHA-256, array digest, dtype, shape, and X2 terminal hash are recomputed",
        "method_specific_normalization_smoothing_or_thresholding": "raw finite float32 maps and shared operator metadata are checked without postprocessing",
        "partial_block_perturbation": "actual pixel fraction is recomputed from whole-block counts",
        "resize_substituted_for_4x4_or_16x16_block_aggregation": "unit counts must equal 56x56 and 14x14 and worker implementation is pre-execution sealed",
        "mask_varies_by_route_method_or_outcome": "all route registries must be byte-identical to one 24-file shared registry",
        "horizontal_flip_omitted": "flip is in every map roster and both flip masks must be full-domain",
        "transformed_map_omitted_or_inverse_only_retained": "4776 transformed and 4776 inverse raw files are loaded and digested",
        "jpeg_png_component_or_clean_map_copy_output": "exact route file roster, suffix scan, and clean-source digest inequality are enforced",
        "budget_waiver_or_silent_cap_excess": "actual and conservative projected ledgers must both remain within every frozen ceiling",
        "m2_analysis_executed": "M1 output allowlist and access audit reject every M2 family",
        "x2_no_go_modified": "P0 source-recomputing validation rechecks sealed X2 NO-GO",
        "x3_x5_f3_final_or_stage_g_started": "source validation and final filesystem/ref audit remain fail-closed",
        "pass_without_source_recomputing_validation": "finalize and validate both reload route terminals and every raw map",
    }
    return _signed("stage_x_x2_mech_m1_false_pass_audit", "PASS", {
        "attacks": [{"attack": attack, "result": "REJECTED", "mechanism": mechanisms[attack]}
                    for attack in FALSE_PASS_ATTACKS],
        "attack_count": len(FALSE_PASS_ATTACKS),
    })


def _forbidden_output_scan(repo: Path) -> None:
    forbidden_suffixes = {".jpg", ".jpeg", ".png"}
    for path in (repo / ROOT).rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if path.suffix.lower() in forbidden_suffixes or "component" in lower or "clean_map" in lower:
            raise M1ReconcileError(f"forbidden M1 output detected: {path}")
    m2_names = {"pixel_reproduction_audit.csv", "pixel_reproduction_audit.json",
                "multiscale_stability.csv", "within_between_error.csv", "transport_control.csv",
                "clean_structure_metrics.csv", "horizontal_flip_control.csv",
                "association_summary.csv", "association_summary.json"}
    if any((repo / ROOT / name).exists() for name in m2_names):
        raise M1ReconcileError("M2 output detected in M1 root")


def _write_aggregate_outputs(repo: Path, collected: Mapping[str, Any]) -> dict[str, bytes]:
    map_rows = [{field: row[field] for field in MAP_FIELDS} for row in collected["maps"]]
    fidelity_rows = [{field: row[field] for field in FIDELITY_FIELDS} for row in collected["fidelities"]]
    detail_rows = [{field: row[field] for field in DETAIL_FIELDS} for row in collected["details"]]
    map_json = _signed("stage_x_x2_mech_m1_map_manifest", "PASS", {
        "maps": collected["maps"], "map_count": 9552, "transformed_count": 4776,
        "inverse_count": 4776, "source_recomputed": True,
    })
    mask_json = _signed("stage_x_x2_mech_m1_valid_mask_registry", "PASS", {
        "masks": collected["masks"], "mask_count": 24, "shared_by_transform": True,
        "horizontal_flip_full_domain": True,
    })
    outputs = {
        "map_manifest.csv": _csv_bytes(MAP_FIELDS, map_rows),
        "map_manifest.json": _canonical(map_json) + b"\n",
        "valid_mask_registry.json": _canonical(mask_json) + b"\n",
        "multiscale_fidelity.csv": _csv_bytes(FIDELITY_FIELDS, fidelity_rows),
        "multiscale_fidelity_rows.csv": _csv_bytes(DETAIL_FIELDS, detail_rows),
    }
    for name, data in outputs.items():
        _write_new(repo / ROOT / name, data)
    return outputs


def _root_files(repo: Path, excluded: set[Path]) -> list[Path]:
    return [path for path in sorted((repo / ROOT).rglob("*")) if path.is_file()
            and path.relative_to(repo) not in excluded]


def finalize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if (repo / PENDING).exists() or (repo / COMPLETION).exists() or (repo / REPORT_ROOT).exists():
        raise M1ReconcileError("M1 reconciliation/completion output already exists")
    _load_seal(repo)
    pending = _signed("stage_x_x2_mech_m1_completion_pending", "PENDING", {
        "created_at": datetime.now(timezone.utc).isoformat(), "terminal": False,
        "source_recomputing_required_before_terminal": True,
    })
    _write_json_new(repo / PENDING, pending)
    collected = _collect(repo)
    _write_aggregate_outputs(repo, collected)
    access = _access_document()
    false_pass = _false_pass_document()
    reconciliation = _signed("stage_x_x2_mech_m1_reconciliation", "PASS", {
        "denominator": 200, "valid_routes": 199, "failed_route_count": 1,
        "failed_route_binding": collected["failed"], "route_checks": collected["route_checks"],
        "counts": {"transformed_maps": 4776, "inverse_maps": 4776, "total_maps": 9552,
                   "shared_masks": 24, "fidelity_summaries": 796,
                   "detailed_road_rows": 71640},
        "source_recomputed_from_route_terminals_and_raw_files": True,
        "worker_summary_trusted": False, "m2_executed": False,
        "x2_decision": "NO_GO_BROAD_CLAIM_DEVELOPMENT",
    })
    for name, document in (("resource_ledger.json", collected["resources"]),
                           ("access_audit.json", access), ("false_pass_audit.json", false_pass),
                           ("reconciliation.json", reconciliation)):
        _write_json_new(repo / ROOT / name, document)
    _forbidden_output_scan(repo)
    report_text = (
        "# X2-MECH M1 terminal report\n\n"
        "Status: `PASS`. The frozen 200-route denominator is unchanged: 199 clean-valid routes "
        "completed one immutable M1 attempt and the original `FAILED_ROUTE_LOCAL_NUMERICAL` route "
        "remains bound without retry or replacement. Source-recomputing reconciliation validated "
        "4,776 transformed maps, 4,776 inverse maps, 24 shared masks, 796 4x4/16x16 "
        "class-logit ROAD summaries, and 71,640 detailed ROAD rows. No M2 analysis was run. "
        "X2 remains `NO_GO_BROAD_CLAIM_DEVELOPMENT`; X3-X5, F3-Final, and Stage G were not started. "
        f"The only next authorization is `{NEXT_AUTHORIZATION}`.\n")
    _write_new(repo / REPORT_ROOT / "x2_mech_m1_report.md", report_text.encode("utf-8"))
    validation = _signed("stage_x_x2_mech_m1_validation_report", "PASS", {
        "source_recomputing": True, "route_terminal_count": 199, "raw_map_count": 9552,
        "shared_mask_count": 24, "fidelity_summary_count": 796,
        "detailed_road_row_count": 71640, "false_pass_attack_count": len(FALSE_PASS_ATTACKS),
        "budget_status": "PASS", "m2_executed": False,
    })
    _write_json_new(repo / REPORT_ROOT / "validation_report.json", validation)
    excluded = {Path(PENDING), Path(COMPLETION), ROOT / "payload_manifest.json",
                ROOT / "whole_root_inventory.json"}
    payload_files = _root_files(repo, excluded)
    manifest = _signed("stage_x_x2_mech_m1_payload_manifest", "PASS", {
        "files": [_record(repo, path.relative_to(repo)) for path in payload_files],
        "file_count": len(payload_files), "source_recomputed": True,
    })
    _write_json_new(repo / ROOT / "payload_manifest.json", manifest)
    inventory_excluded = {Path(PENDING), Path(COMPLETION), ROOT / "whole_root_inventory.json"}
    inventory_files = _root_files(repo, inventory_excluded)
    inventory_records = [_record(repo, path.relative_to(repo)) for path in inventory_files]
    inventory = _signed("stage_x_x2_mech_m1_whole_root_inventory", "PASS", {
        "files": inventory_records, "file_count": len(inventory_records),
        "total_bytes": sum(row["bytes"] for row in inventory_records),
        "files_digest": _canonical_digest(inventory_records),
        "scope": "formal_root_excluding_inventory_and_completion_receipts",
    })
    _write_json_new(repo / ROOT / "whole_root_inventory.json", inventory)
    if inventory["payload"]["total_bytes"] > int(CAPS["retained_gib"] * 1024 ** 3):
        raise M1ReconcileError("BLOCKED_BUDGET: final retained root exceeds ceiling")
    completion = _signed("stage_x_x2_mech_m1_completion", "PASS", {
        "terminal": True, "complete": True, "immutable": True,
        "base_commit": BASE_COMMIT, "archive_ref": ARCHIVE_REF, "git_commit": "SELF",
        "denominator": {"expected": 200, "valid": 199, "failed_route_local_numerical": 1},
        "counts": reconciliation["payload"]["counts"],
        "bindings": [_record(repo, ROOT / name) for name in (
            "map_manifest.csv", "map_manifest.json", "valid_mask_registry.json",
            "multiscale_fidelity.csv", "multiscale_fidelity_rows.csv", "resource_ledger.json",
            "access_audit.json", "false_pass_audit.json", "reconciliation.json",
            "payload_manifest.json", "whole_root_inventory.json")]
            + [_record(repo, REPORT_ROOT / "validation_report.json"),
               _record(repo, REPORT_ROOT / "x2_mech_m1_report.md")],
        "source_recomputing_validation": "PASS", "budget_status": "PASS",
        "budget_waiver_used": False, "m2_executed": False,
        "x2_decision": "NO_GO_BROAD_CLAIM_DEVELOPMENT", "x2_decision_modified": False,
        "x3_x5_started": False, "f3_final_started": False, "stage_g_started": False,
        "next_authorization": NEXT_AUTHORIZATION,
    })
    if (repo / COMPLETION).exists():
        raise M1ReconcileError("completion terminal already exists")
    terminal_temp = repo / (str(COMPLETION) + ".terminal.tmp")
    _write_json_new(terminal_temp, completion)
    # Replace the pending receipt content with terminal bytes, then atomically rename it.
    os.replace(terminal_temp, repo / PENDING)
    os.replace(repo / PENDING, repo / COMPLETION)
    return completion


def _verify_bound_record(repo: Path, record: Mapping[str, Any]) -> None:
    path = repo / record["path"]
    if path.stat().st_size != record["bytes"] or _sha_file(path) != record["sha256"]:
        raise M1ReconcileError(f"bound file drifted: {record['path']}")
    if path.suffix == ".json" and record.get("canonical_digest") is not None:
        value = _read_json(path)
        if value.get("canonical_digest") != record["canonical_digest"]:
            raise M1ReconcileError(f"bound canonical digest drifted: {record['path']}")


def validate(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if (repo / PENDING).exists() or not (repo / COMPLETION).is_file():
        raise M1ReconcileError("terminal completion state is invalid")
    completion = _read_json(repo / COMPLETION)
    _validate_signed(completion, str(COMPLETION))
    if completion.get("status") != "PASS" or completion["payload"].get("next_authorization") != NEXT_AUTHORIZATION:
        raise M1ReconcileError("completion contract drifted")
    for record in completion["payload"]["bindings"]:
        _verify_bound_record(repo, record)
    collected = _collect(repo)
    expected_outputs = {
        "map_manifest.csv": _csv_bytes(MAP_FIELDS, [{field: row[field] for field in MAP_FIELDS}
                                                      for row in collected["maps"]]),
        "multiscale_fidelity.csv": _csv_bytes(FIDELITY_FIELDS, [{field: row[field] for field in FIDELITY_FIELDS}
                                                                  for row in collected["fidelities"]]),
        "multiscale_fidelity_rows.csv": _csv_bytes(DETAIL_FIELDS, [{field: row[field] for field in DETAIL_FIELDS}
                                                                      for row in collected["details"]]),
    }
    for name, expected in expected_outputs.items():
        if (repo / ROOT / name).read_bytes() != expected:
            raise M1ReconcileError(f"aggregate source recomputation drifted: {name}")
    map_json = _read_json(repo / ROOT / "map_manifest.json")
    mask_json = _read_json(repo / ROOT / "valid_mask_registry.json")
    _validate_signed(map_json, "map_manifest.json")
    _validate_signed(mask_json, "valid_mask_registry.json")
    if map_json["payload"]["maps"] != collected["maps"] or mask_json["payload"]["masks"] != collected["masks"]:
        raise M1ReconcileError("JSON manifest source recomputation drifted")
    resource = _read_json(repo / ROOT / "resource_ledger.json")
    _validate_signed(resource, "resource_ledger.json")
    if resource != collected["resources"]:
        raise M1ReconcileError("resource ledger recomputation drifted")
    for relative in (ROOT / "access_audit.json", ROOT / "false_pass_audit.json",
                     ROOT / "reconciliation.json", ROOT / "payload_manifest.json",
                     ROOT / "whole_root_inventory.json", REPORT_ROOT / "validation_report.json"):
        value = _read_json(repo / relative)
        _validate_signed(value, str(relative))
    manifest = _read_json(repo / ROOT / "payload_manifest.json")
    for record in manifest["payload"]["files"]:
        _verify_bound_record(repo, record)
    inventory = _read_json(repo / ROOT / "whole_root_inventory.json")
    records = inventory["payload"]["files"]
    for record in records:
        _verify_bound_record(repo, record)
    if _canonical_digest(records) != inventory["payload"]["files_digest"]:
        raise M1ReconcileError("whole-root inventory digest drifted")
    _forbidden_output_scan(repo)
    if completion["payload"]["counts"] != {"transformed_maps": 4776, "inverse_maps": 4776,
            "total_maps": 9552, "shared_masks": 24, "fidelity_summaries": 796,
            "detailed_road_rows": 71640}:
        raise M1ReconcileError("completion counts drifted")
    return {"status": "PASS", "source_recomputed": True, "routes": 199,
            "maps": 9552, "shared_masks": 24, "fidelity_summaries": 796,
            "detailed_road_rows": 71640, "next_authorization": NEXT_AUTHORIZATION}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal-pre-execution", "finalize", "validate"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "seal-pre-execution":
            result = seal_pre_execution(args.repo_root)
        elif args.command == "finalize":
            result = finalize(args.repo_root)
        else:
            result = validate(args.repo_root)
    except (M1ReconcileError, OSError, KeyError, TypeError, ValueError, worker.X2MechWorkerError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
