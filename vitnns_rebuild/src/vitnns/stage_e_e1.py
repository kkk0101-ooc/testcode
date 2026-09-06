"""Stage E E1 native-patch sensitivity execution and validation.

E1 consumes the sealed D9 maps and the sealed D10 fine-scale result.  It never
recomputes an attribution map or the D10 anchor.  The only model work exposed
by this module is evaluation of the two frozen rankings under native 16x16
patch interventions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from .stage_e_e0 import (
    EXPECTED_D10_INVENTORY, EXPECTED_D9_INVENTORY, StageEE0Error,
    _d10_inventory, validate_completion_document, verify_upstream,
)

CONFIG_PATH = "config/stage_e_e1.json"
CONFIG_SCHEMA_PATH = "config/stage_e_e1.schema.json"
OUTPUT_SCHEMA_PATH = "config/stage_e_e1_output.schema.json"
E0_COMPLETION_PATH = "evidence/stage_e/e0_recovery_v2_20260906/completion/e0_completion.json"
D8_ROOT_PATH = "evidence/stage_d/d8_recovery_v2_20260904/formal_r2"
D10_ROOT_PATH = "evidence/stage_d/d10_recovery_v2_20260906/formal_r2"
D10_ROWS_PATH = D10_ROOT_PATH + "/paper/paper_data/per_sample_delta.json"
REGISTRY_PATH = "manifests/stage_d_fidelity_v1/registry.json"
ROOT_PATH = "evidence/stage_e/e1_recovery_v2_20260906/formal"
REPORT_ROOT_PATH = "reports/stage_e_e1_20260906"
PLAN_PATH = ROOT_PATH + "/control/typed_plan.json"
RUN_SPEC_PATH = ROOT_PATH + "/control/run_spec.json"
AUTH_PATH = ROOT_PATH + "/control/execution_authorization.json"
PREFLIGHT_PATH = ROOT_PATH + "/control/budget_preflight.json"
UPSTREAM_BEFORE_PATH = ROOT_PATH + "/control/protected_upstream_before.json"
COMPLETION_PENDING_PATH = ROOT_PATH + "/completion/e1_completion.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/e1_completion.json"
ACCESS_PENDING_PATH = REPORT_ROOT_PATH + "/outcome_access.pending.json"
ACCESS_PATH = REPORT_ROOT_PATH + "/outcome_access.json"

E0_COMMIT = "6cd8a481966940dd786aff03f402262921c180fe"
E0_COMPLETION_DIGEST = "e754ab85577ff00b7f840153dae65b15570bad3366ba0fbcf91ee0961be327fd"
E0_CONFIG_DIGEST = "8b8a0066dad44b2681a8e12b92465c72f2b405fa7258313137260ce267a7184a"
E0_RUN_SPEC_DIGEST = "7799b5b75e3bba04a4bfb9043b21289d79811b9dfed9a1b7a1805f889d81f339"
E0_REUSE_DIGEST = "1eabf626336010f880420ac3f3b268ac15ab7ae4100fb6e1a68881e7fdd66522"
D10_COMMIT = "7520689bb031304e4789567f99d093062aa50f72"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
EXPECTED_E1_CONFIG_DIGEST = "f3d178382bed4eb27870ff088d24498099c5f295cef21668f97b4034a6e8cbb9"
VARIANTS = (
    "d10_fine_4x4_primary_anchor",
    "nns_native_14x14_patch",
    "matched_native_14x14_patch",
)
PATCH_VARIANTS = VARIANTS[1:]
METHODS = {
    PATCH_VARIANTS[0]: "conditional_fixed_clean_attention_head_direction_value_path_ig",
    PATCH_VARIANTS[1]: "matched_fixed_clean_attention_patch_token_contribution",
}
COMMON_SEED_IDENTITY = "e1_native_14x14_patch_comparison"
FRACTIONS = tuple(float(i) / 20.0 for i in range(21))
UNIT_COUNTS = (0,) + tuple(int(math.ceil(value * 196)) for value in FRACTIONS[1:])
ACTUAL_FRACTIONS = tuple(value / 196.0 for value in UNIT_COUNTS)
PROTECTED_PATHS = (
    "evidence/stage_d/d8_recovery_v2_20260904/formal_r2",
    "evidence/stage_d/d10_recovery_v2_20260906/formal_r2",
    "evidence/stage_e/e0_recovery_v2_20260906",
    "reports/stage_d_d9_recovery_v2_20260906",
    "reports/stage_d_d10_recovery_v2_20260906",
    "reports/stage_e_e0_20260906",
)
FORBIDDEN_SUFFIXES = {".png", ".pdf", ".svg"}


class StageEE1Error(RuntimeError):
    """E1 cannot proceed or pass under its frozen contract."""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: item for key, item in value.items() if key != "canonical_digest"
    })


def _signed(document: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(document)
    value["canonical_digest"] = _self_digest(value)
    return value


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except Exception as error:
        raise StageEE1Error(f"cannot read strict {context}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageEE1Error(f"{context} must be a JSON object")
    return value, raw


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest") != _self_digest(value):
        raise StageEE1Error(f"{context} canonical digest mismatch")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise StageEE1Error(f"immutable control differs: {path}")
        return
    _write_new_json(path, value)


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ("git", *args), cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise StageEE1Error(
            f"git {' '.join(args)} failed: {error.stderr.strip()}"
        ) from error


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    raw = path.read_bytes()
    result: dict[str, Any] = {
        "path": relative, "bytes": len(raw), "sha256": _sha(raw),
    }
    if path.suffix == ".json":
        value, _ = _strict_json(path, relative)
        result["canonical_digest"] = value.get("canonical_digest")
    else:
        result["canonical_digest"] = None
    return result


def _validate_config(repo: Path) -> dict[str, Any]:
    value, _ = _strict_json(repo / CONFIG_PATH, "E1 config")
    schema, _ = _strict_json(repo / CONFIG_SCHEMA_PATH, "E1 config schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: [str(part) for part in item.path],
    )
    if errors:
        where = ".".join(str(part) for part in errors[0].path) or "root"
        raise StageEE1Error(f"E1 config schema failure at {where}: {errors[0].message}")
    _validate_signed(value, "E1 config")
    if value["canonical_digest"] != EXPECTED_E1_CONFIG_DIGEST:
        raise StageEE1Error("E1 frozen config digest mismatch")
    if value["bindings"]["e0_config_canonical_digest"] != E0_CONFIG_DIGEST:
        raise StageEE1Error("E1 config E0 binding drifted")
    scope = value["scope"]
    if (
        tuple(row["variant_id"] for row in scope["variants"]) != VARIANTS
        or scope["sample_count"] != 500
        or scope["expected_new_units"] != 1000
        or scope["block"] != 0 or scope["head"] != 9
    ):
        raise StageEE1Error("E1 closed scope drifted")
    science = value["scientific_contract"]
    if (
        science["unit_count"] != 196
        or tuple(science["fractions"]) != FRACTIONS
        or science["tie_repeats"] != 5
        or science["tie_seed_comparison_identity"] != COMMON_SEED_IDENTITY
        or science["row_drop_allowed"] is not False
        or science["p_values_emitted"] is not False
        or science["winner_selection_allowed"] is not False
        or science["new_go_no_go_allowed"] is not False
    ):
        raise StageEE1Error("E1 scientific contract drifted")
    if value["budget"] != {
        "gpu_hours_max": 2.5, "cpu_hours_max": 8.0,
        "retained_bytes_max": 4294967296,
    }:
        raise StageEE1Error("E1 budget drifted")
    if value["authorization"] != {
        "requires": "E0_PASS", "current_stage": "E1_ONLY",
        "on_pass": "E2_ONLY", "e2_started": False,
        "e3_started": False, "e4_started": False,
        "stage_f_started": False,
    }:
        raise StageEE1Error("E1 authorization boundary drifted")
    return value


def e1_seed(sample_id: str, repeat: int) -> int:
    if type(repeat) is not int or not 0 <= repeat < 5:
        raise StageEE1Error("repeat must be an integer in [0,4]")
    material = (
        "stage-e-fidelity|20260906|" + sample_id
        + "|0|9|" + COMMON_SEED_IDENTITY + "|" + str(repeat)
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _array_digest(array: np.ndarray) -> str:
    source = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(source.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(source.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(source.tobytes(order="C"))
    return digest.hexdigest()


def aggregate_patch_map(source: Any) -> np.ndarray:
    value = np.asarray(source)
    if value.dtype != np.float32 or value.shape != (56, 56):
        raise StageEE1Error("source map must be signed float32 56x56")
    if not np.isfinite(value).all():
        raise StageEE1Error("source map is non-finite")
    result = value.astype(np.float64).reshape(14, 4, 14, 4).sum(axis=(1, 3), dtype=np.float64)
    if result.shape != (14, 14) or not np.isfinite(result).all():
        raise StageEE1Error("patch aggregation failed")
    before = float(value.sum(dtype=np.float64))
    after = float(result.sum(dtype=np.float64))
    tolerance = 1e-12 + 1e-12 * abs(before)
    if abs(after - before) > tolerance:
        raise StageEE1Error("signed mass changed during 56x56 to 14x14 aggregation")
    return result


def tie_priority(seed: int, unit_count: int = 196) -> np.ndarray:
    order = np.random.Generator(np.random.PCG64(seed)).permutation(unit_count)
    priority = np.empty(unit_count, dtype=np.int64)
    priority[order] = np.arange(unit_count, dtype=np.int64)
    return priority


def patch_ranking(signed_scores: Any, *, sigma: int, seed: int) -> np.ndarray:
    values = np.asarray(signed_scores, dtype=np.float64)
    if values.shape != (14, 14) or not np.isfinite(values).all() or sigma not in (-1, 1):
        raise StageEE1Error("invalid signed patch scores or target orientation")
    scores = np.maximum(float(sigma) * values.reshape(-1), 0.0)
    return np.lexsort((tie_priority(seed), -scores)).astype(np.int64)


def trapezoid_auc(responses: Sequence[float]) -> float:
    y = np.asarray(responses, dtype=np.float64)
    x = np.asarray(ACTUAL_FRACTIONS, dtype=np.float64)
    if y.shape != (21,) or not np.isfinite(y).all():
        raise StageEE1Error("AUC requires 21 finite responses")
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5, dtype=np.float64))


def _source_index(repo: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    reconciliation, _ = _strict_json(
        repo / D8_ROOT_PATH / "final/reconciliation.json", "D9 reconciliation"
    )
    rows = reconciliation.get("rows")
    if not isinstance(rows, list) or len(rows) != 1000:
        raise StageEE1Error("D9 reconciliation does not contain 1000 units")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("sample_id"), row.get("method_id"))
        if key in index or key[1] not in set(METHODS.values()):
            raise StageEE1Error("D9 reconciliation has duplicate or extra unit")
        if (
            row.get("terminal_state") != "succeeded"
            or row.get("failure_class") is not None
            or len(row.get("attempts", [])) != 1
        ):
            raise StageEE1Error("D9 source unit is not a unique success")
        index[key] = row
    if len(index) != 1000:
        raise StageEE1Error("D9 source index cardinality mismatch")
    return index, reconciliation


def _load_registry(repo: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry, raw = _strict_json(repo / REGISTRY_PATH, "D1 registry")
    rows = registry.get("rows")
    if (
        _sha(raw) != "9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b"
        or registry.get("canonical_digest") != "514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028"
        or not isinstance(rows, list) or len(rows) != 500
        or [row.get("selection_order") for row in rows] != list(range(500))
        or len({row.get("sample_id") for row in rows}) != 500
    ):
        raise StageEE1Error("D1 registry binding/order failed")
    return rows, registry


def _source_attempt(repo: Path, row: Mapping[str, Any]) -> Path:
    relative = row["attempts"][0]["attempt_path"]
    path = (repo / D8_ROOT_PATH / relative).resolve()
    root = (repo / D8_ROOT_PATH / "units").resolve()
    if root not in path.parents or path.name != "attempt-000001":
        raise StageEE1Error("unsafe D9 source attempt path")
    return path


def _build_plan(repo: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    registry, _ = _load_registry(repo)
    source_index, _ = _source_index(repo)
    units: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for row in registry:
        group_units: list[str] = []
        source_units: dict[str, str] = {}
        for variant in PATCH_VARIANTS:
            method = METHODS[variant]
            source = source_index[(row["sample_id"], method)]
            attempt = _source_attempt(repo, source)
            map_path = attempt / "L1/metric_ready_map.npy"
            result_path = attempt / "L0/result.json"
            unit = {
                "schema_version": 1, "document_type": "stage_e_e1_unit",
                "variant_id": variant, "method": method,
                "sample_id": row["sample_id"],
                "selection_order": row["selection_order"],
                "image_sha256": row["image_sha256"],
                "relative_path": row["relative_path"],
                "block": 0, "head": 9, "scale": "native_14x14_patch",
                "pair_identity": COMMON_SEED_IDENTITY,
                "source_unit_digest": source["unit_digest"],
                "source_attempt_inventory_digest": source["attempts"][0]["attempt_inventory_digest"],
                "source_map_path": map_path.relative_to(repo).as_posix(),
                "source_map_sha256": _sha_file(map_path),
                "source_result_path": result_path.relative_to(repo).as_posix(),
                "source_result_sha256": _sha_file(result_path),
            }
            unit["canonical_digest"] = _self_digest(unit)
            units.append(unit)
            group_units.append(unit["canonical_digest"])
            source_units[variant] = unit["source_unit_digest"]
        group = {
            "sample_id": row["sample_id"],
            "selection_order": row["selection_order"],
            "image_sha256": row["image_sha256"],
            "relative_path": row["relative_path"],
            "unit_digests": group_units, "source_unit_digests": source_units,
            "pair_identity": COMMON_SEED_IDENTITY,
        }
        group["canonical_digest"] = _self_digest(group)
        groups.append(group)
    document = {
        "schema_version": 1, "document_type": "stage_e_e1_typed_plan",
        "config_canonical_digest": config["canonical_digest"],
        "cohort": "stage_d_fidelity_v1_500",
        "ordered_sample_ids_sha256": config["scope"]["ordered_sample_ids_sha256"],
        "expected_image_groups": 500, "expected_new_units": 1000,
        "variants": list(VARIANTS), "groups": groups, "units": units,
    }
    return _signed(document)


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude)
    files: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in excluded or any(part.startswith(".") for part in Path(relative).parts):
                continue
            files.append({
                "path": relative, "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            })
    return {
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "canonical_digest": canonical_sha256({"files": files}),
        "files": files,
    }


def _upstream_snapshot(
    repo: Path, *, d9_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    d9_root = repo / D8_ROOT_PATH
    source_inventory, _ = _strict_json(
        d9_root / "final/source_inventory.json", "D9 source inventory"
    )
    _validate_signed(source_inventory, "D9 source inventory")
    if d9_inventory is None:
        upstream = dict(verify_upstream(repo))
        d9_inventory = upstream["d9_source_validation"]["root_inventory"]
    if d9_inventory.get("canonical_digest") != EXPECTED_D9_INVENTORY:
        raise StageEE1Error("D9 sealed-root inventory binding failed")
    d10 = _d10_inventory(repo)
    if d10["canonical_digest"] != EXPECTED_D10_INVENTORY:
        raise StageEE1Error("D10 output-root inventory binding failed")
    e0_root = repo / "evidence/stage_e/e0_recovery_v2_20260906"
    e0_inventory = _inventory(e0_root)
    protected_diff = _git(repo, "diff", "--name-only", E0_COMMIT, "--", *PROTECTED_PATHS).splitlines()
    if protected_diff:
        raise StageEE1Error(f"protected upstream tracked paths changed: {protected_diff}")
    return _signed({
        "schema_version": 1, "document_type": "stage_e_e1_protected_upstream_snapshot",
        "captured_at": _now(),
        "d9": {"root": D8_ROOT_PATH, "file_count": d9_inventory["file_count"], "total_bytes": d9_inventory["total_bytes"], "canonical_digest": d9_inventory["canonical_digest"], "source_inventory_canonical_digest":source_inventory["canonical_digest"]},
        "d10": {"root": D10_ROOT_PATH, **d10},
        "e0": {"root": "evidence/stage_e/e0_recovery_v2_20260906", **{key: e0_inventory[key] for key in ("file_count","total_bytes","canonical_digest")}},
        "protected_tracked_diff": protected_diff,
    })


def verify_preconditions(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _validate_config(repo)
    if _git(repo, "symbolic-ref", "--short", "HEAD") != "stage-d-d3-pass":
        raise StageEE1Error("wrong branch")
    head = _git(repo, "rev-parse", "HEAD")
    tracking = _git(repo, "rev-parse", "@{u}")
    remote = _git(repo, "rev-parse", "refs/remotes/origin/stage-d-d3-pass")
    archive = _git(repo, "rev-parse", "refs/archive/stage-e-e0-completion^{commit}")
    if {head, tracking, remote, archive} != {E0_COMMIT}:
        raise StageEE1Error(
            f"workspace commit binding failed: expected={E0_COMMIT} "
            f"head={head} tracking={tracking} remote={remote} archive={archive}"
        )
    status = _git(repo, "status", "--porcelain=v1")
    if status:
        allowed_suffixes = (
            "config/stage_e_e1.json", "config/stage_e_e1.schema.json",
            "config/stage_e_e1_output.schema.json", "src/vitnns/stage_e_e1.py",
            "tests/test_stage_e_e1.py", "evidence/stage_e/e1_recovery_v2_20260906/",
            "reports/stage_e_e1_20260906/",
        )
        unexpected = [
            line for line in status.splitlines()
            if not any(line[3:].endswith(suffix) for suffix in allowed_suffixes)
        ]
        if unexpected:
            raise StageEE1Error(f"unexpected worktree changes: {unexpected}")
    lfs = _git(repo, "lfs", "status")
    if "Git LFS objects to be committed:" not in lfs or "Git LFS objects not staged for commit:" not in lfs:
        raise StageEE1Error("cannot establish Git LFS status")
    e0, _ = _strict_json(repo / E0_COMPLETION_PATH, "E0 completion")
    try:
        validate_completion_document(e0, repo)
    except StageEE0Error as error:
        raise StageEE1Error(f"E0 completion validation failed: {error}") from error
    if e0.get("canonical_digest") != E0_COMPLETION_DIGEST:
        raise StageEE1Error("E0 completion digest mismatch")
    upstream = dict(verify_upstream(repo))
    if upstream.get("status") != "PASS":
        raise StageEE1Error("D9/D10 source validation did not PASS")
    snapshot = _upstream_snapshot(
        repo, d9_inventory=upstream["d9_source_validation"]["root_inventory"],
    )
    return {
        "status": "PASS", "validated_at": _now(), "head": head,
        "e0_completion_canonical_digest": e0["canonical_digest"],
        "d9_inventory": upstream["d9_source_validation"]["root_inventory"],
        "d10_inventory": upstream["d10_output_root_inventory"],
        "protected_snapshot": snapshot, "config": config,
    }


def initialize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    preconditions = verify_preconditions(repo)
    config = preconditions["config"]
    plan = _build_plan(repo, config)
    run_spec = _signed({
        "schema_version": 1, "document_type": "stage_e_e1_run_spec",
        "config_canonical_digest": config["canonical_digest"],
        "plan_canonical_digest": plan["canonical_digest"],
        "expected_image_groups": 500, "expected_new_units": 1000,
        "variants": list(VARIANTS), "unit_count": 196,
        "directions": ["insertion", "deletion"],
        "tie_repeats": list(range(5)), "fractions": list(FRACTIONS),
        "actual_unit_counts": list(UNIT_COUNTS),
        "actual_fractions": list(ACTUAL_FRACTIONS),
        "attempt_policy": {"initial_attempt":"attempt-000001","maximum_attempts":2,"automatic_retry":False,"terminal_recompute":False},
        "stage_boundary": {"current":"E1_ONLY","on_pass":"E2_ONLY","later_stages_started":False},
    })
    estimated_forward_calls = 500 * 2 * (2 + 5 * 2 * 21)
    preflight = _signed({
        "schema_version": 1, "document_type": "stage_e_e1_budget_preflight",
        "status": "PASS", "recorded_at": _now(),
        "basis": "D8 measured 249340 forward calls and 6661.53 GPU-seconds including attribution; E1 has 212000 forward-only calls and no backward calls",
        "estimated": {"forward_calls": estimated_forward_calls, "gpu_hours": 1.58, "cpu_hours": 4.0, "retained_bytes": 1073741824},
        "limits": config["budget"],
    })
    authorization = _signed({
        "schema_version": 1, "document_type": "stage_e_e1_execution_authorization",
        "status": "AUTHORIZED", "issued_at": _now(),
        "e0_pass_verified": True, "d9_d10_verified": True,
        "plan_canonical_digest": plan["canonical_digest"],
        "run_spec_canonical_digest": run_spec["canonical_digest"],
        "budget_preflight_canonical_digest": preflight["canonical_digest"],
        "authorized_scope": "E1_ONLY_500_GROUPS_1000_PATCH_UNITS",
        "d10_anchor_recompute": "FORBIDDEN_RECOMPUTE",
        "later_stages_authorized": False,
    })
    access = _signed({
        "schema_version": 1, "document_type": "stage_e_e1_outcome_access",
        "status": "PENDING_BEFORE_FIRST_E1_OUTCOME",
        "e1_started_at": _now(), "e0_pass_verified": True,
        "d10_outcome_known": True, "new_e1_outcome_accessed": False,
        "scope_source": "sealed_E0_only",
        "frozen_scope": {"variants":list(VARIANTS),"sample_count":500,"unit_count":196,"fractions":list(FRACTIONS),"tie_repeats":5,"bootstrap_draws":10000,"budget":config["budget"]},
        "upstream_reads": [
            _record(repo, E0_COMPLETION_PATH), _record(repo, REGISTRY_PATH),
            _record(repo, D8_ROOT_PATH + "/final/source_inventory.json"),
            _record(repo, D8_ROOT_PATH + "/final/reconciliation.json"),
            _record(repo, D10_ROWS_PATH),
            _record(repo, "reports/stage_d_d9_recovery_v2_20260906/d9_formal_validation.json"),
            _record(repo, "reports/stage_d_d10_recovery_v2_20260906/d10_formal_decision.json"),
        ],
        "protected_roots": [{"path":p,"mode":"READ_ONLY"} for p in PROTECTED_PATHS],
        "e2_started": False, "e3_started": False, "e4_started": False,
        "stage_f_started": False,
    })
    _write_or_verify(repo / PLAN_PATH, plan)
    _write_or_verify(repo / RUN_SPEC_PATH, run_spec)
    _write_or_verify(repo / PREFLIGHT_PATH, preflight)
    _write_or_verify(repo / AUTH_PATH, authorization)
    _write_or_verify(repo / UPSTREAM_BEFORE_PATH, preconditions["protected_snapshot"])
    _write_or_verify(repo / ACCESS_PENDING_PATH, access)
    completion_pending = _signed({
        "schema_version":1, "document_type":"stage_e_e1_completion_pending",
        "stage_id":"E1", "status":"PENDING",
        "started_at":access["e1_started_at"],
        "plan_canonical_digest":plan["canonical_digest"],
        "run_spec_canonical_digest":run_spec["canonical_digest"],
        "authorization_canonical_digest":authorization["canonical_digest"],
        "expected_image_groups":500, "expected_new_units":1000,
        "d10_anchor_recompute":"FORBIDDEN_RECOMPUTE",
        "next_authorization":"NONE_PENDING",
    })
    _write_or_verify(repo / COMPLETION_PENDING_PATH, completion_pending)
    validate_initialized(repo)
    return {
        "status": "PASS", "plan_canonical_digest": plan["canonical_digest"],
        "run_spec_canonical_digest": run_spec["canonical_digest"],
        "authorization_canonical_digest": authorization["canonical_digest"],
        "access_record": ACCESS_PENDING_PATH,
    }


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _validate_config(repo)
    plan, _ = _strict_json(repo / PLAN_PATH, "E1 plan")
    run_spec, _ = _strict_json(repo / RUN_SPEC_PATH, "E1 run spec")
    authorization, _ = _strict_json(repo / AUTH_PATH, "E1 authorization")
    preflight, _ = _strict_json(repo / PREFLIGHT_PATH, "E1 preflight")
    access, _ = _strict_json(repo / ACCESS_PENDING_PATH, "E1 pending access")
    pending, _ = _strict_json(repo / COMPLETION_PENDING_PATH, "E1 pending completion")
    for value, context in ((plan,"plan"),(run_spec,"run spec"),(authorization,"authorization"),(preflight,"preflight"),(access,"access"),(pending,"pending completion")):
        _validate_signed(value, f"E1 {context}")
    rebuilt = _build_plan(repo, config)
    if rebuilt != plan:
        raise StageEE1Error("E1 plan is not source-recomputable")
    if (
        len(plan["groups"]) != 500 or len(plan["units"]) != 1000
        or [row["selection_order"] for row in plan["groups"]] != list(range(500))
        or Counter(row["variant_id"] for row in plan["units"]) != Counter({PATCH_VARIANTS[0]:500,PATCH_VARIANTS[1]:500})
    ):
        raise StageEE1Error("E1 plan coverage failed")
    if authorization["status"] != "AUTHORIZED" or preflight["status"] != "PASS":
        raise StageEE1Error("E1 execution not authorized")
    if access["new_e1_outcome_accessed"] is not False or pending["status"] != "PENDING":
        raise StageEE1Error("E1 access/completion state is invalid before run")
    return {"status":"PASS","image_groups":500,"new_units":1000,"config_canonical_digest":config["canonical_digest"]}


def _target_semantic_digest(identity: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(identity, allow_nan=False))
    try:
        value["frozen_target"]["runtime_model_binding"]["sha256"] = "LIVE_RUNTIME_EXCLUDED"
    except (KeyError, TypeError) as error:
        raise StageEE1Error("target identity lacks runtime binding") from error
    return canonical_sha256(value)


def _load_source_pair(
    repo: Path, units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    results: dict[str, dict[str, Any]] = {}
    maps: dict[str, np.ndarray] = {}
    for unit in units:
        variant = unit["variant_id"]
        map_path = repo / unit["source_map_path"]
        result_path = repo / unit["source_result_path"]
        if (
            _sha_file(map_path) != unit["source_map_sha256"]
            or _sha_file(result_path) != unit["source_result_sha256"]
        ):
            raise StageEE1Error(f"sealed source bytes drifted for {variant}")
        result, _ = _strict_json(result_path, f"D9 {variant} result")
        source_map = np.load(map_path, allow_pickle=False)
        if source_map.dtype != np.float32 or source_map.shape != (56, 56):
            raise StageEE1Error(f"sealed source map contract failed for {variant}")
        if result.get("method_id") != METHODS[variant] or result.get("status") != "OK":
            raise StageEE1Error(f"sealed source result contract failed for {variant}")
        results[variant] = result
        maps[variant] = source_map
    if results[PATCH_VARIANTS[0]].get("target_identity") != results[PATCH_VARIANTS[1]].get("target_identity"):
        raise StageEE1Error("D9 paired methods do not share one target identity")
    return results, maps


def _load_model_and_context(repo: Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from .modeling import load_canonical_model
    from .recovery_v2 import load_recovery_v2_binding

    binding = load_recovery_v2_binding(
        repo / "config/recovery_v2_binding.json",
        schema_path=repo / "config/recovery_v2_binding.schema.json",
        repo_root=repo, verify_assets=False,
    )
    model = load_canonical_model(
        state_dict_path=binding.model_path, device=device, verify_hash=True,
        unfused_attention=True, recovery_binding=binding,
    )
    return model, binding, {
        "name": binding.model["name"],
        "path": Path(binding.model_path).relative_to(repo).as_posix(),
        "sha256": binding.model["state_dict_sha256"],
        "canonical_tensor_digest": binding.model["canonical_tensor_digest"],
    }


def _load_sample(binding: Any, row: Mapping[str, Any], model: Any, device: str) -> tuple[Any, Any, str]:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform_functional
    from .modeling import canonical_preprocess_spec
    from .stage_c_perturbations import tensor_sha256

    image_path = Path(binding.data_root) / row["relative_path"]
    if _sha_file(image_path) != row["image_sha256"]:
        raise StageEE1Error(f"live image digest mismatch: {row['sample_id']}")
    specification = canonical_preprocess_spec(model)
    expected = ((3,224,224),248,(224,224),"bicubic",(0.5,0.5,0.5),(0.5,0.5,0.5))
    observed = (specification.input_size,specification.resize_shorter_side,specification.center_crop,specification.interpolation,specification.mean,specification.std)
    if observed != expected:
        raise StageEE1Error("canonical preprocessing contract drifted")
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        resized = transform_functional.resize(
            rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True,
        )
        cropped = transform_functional.center_crop(resized, [224,224])
        canonical = transform_functional.pil_to_tensor(cropped).to(torch.float32).div(255.0).unsqueeze(0)
    normalized_cpu = (canonical - 0.5) / 0.5
    normalized = normalized_cpu.to(device)
    return canonical, normalized, tensor_sha256(normalized)


def _curve(
    *, evaluator: Any, response: Any, clean: Any, patch_scores: np.ndarray,
    sample_id: str, variant: str, direction: str, repeat: int,
) -> dict[str, Any]:
    import torch
    from .stage_c_perturbations import tensor_sha256

    seed = e1_seed(sample_id, repeat)
    ranking = patch_ranking(patch_scores, sigma=response.sigma, seed=seed)
    priority = tie_priority(seed)
    baseline = torch.zeros_like(clean)
    responses: list[float] = []
    intervention_hashes: list[str] = []
    for count in UNIT_COUNTS:
        selected = np.zeros(196, dtype=np.bool_)
        selected[ranking[:count]] = True
        pixel_mask = np.repeat(np.repeat(selected.reshape(14,14),16,axis=0),16,axis=1)
        mask = torch.from_numpy(pixel_mask.copy()).to(device=clean.device)[None,None].expand_as(clean)
        intervention = (
            torch.where(mask, clean, baseline)
            if direction == "insertion"
            else torch.where(mask, baseline, clean)
        )
        digest = tensor_sha256(intervention)
        scalar = evaluator.evaluate_prehashed(intervention, input_sha256=digest)
        normalized = response.normalize(scalar)
        if not isinstance(normalized, float) or not math.isfinite(normalized):
            raise StageEE1Error("non-finite normalized patch response")
        responses.append(normalized)
        intervention_hashes.append(digest)
    expected_endpoints = (0.0,1.0) if direction == "insertion" else (1.0,0.0)
    if (responses[0],responses[-1]) != expected_endpoints:
        raise StageEE1Error("patch response endpoints violate frozen identities")
    return {
        "variant_id":variant, "method":METHODS[variant],
        "sample_id":sample_id, "direction":direction, "repeat":repeat,
        "seed":seed, "seed_namespace":"stage-e-fidelity|20260906|",
        "seed_comparison_identity":COMMON_SEED_IDENTITY,
        "tie_priority_sha256":_array_digest(priority),
        "signed_patch_score_sha256":_array_digest(patch_scores),
        "ranking_sha256":_array_digest(ranking),
        "unit_counts":list(UNIT_COUNTS), "actual_fractions":list(ACTUAL_FRACTIONS),
        "intervention_sha256":intervention_hashes, "responses":responses,
        "auc":trapezoid_auc(responses), "response_clipped":False,
    }


def _attempt_paths(repo: Path, unit_digest: str) -> tuple[Path, Path, Path, Path]:
    root = repo / ROOT_PATH / "units" / unit_digest / "attempts/attempt-000001"
    return root, root / "result.json", root / "status.json", root / "manifest.json"


def _validate_attempt(repo: Path, unit: Mapping[str, Any]) -> dict[str, Any]:
    root, result_path, status_path, manifest_path = _attempt_paths(repo, unit["canonical_digest"])
    if not root.is_dir() or any(not path.is_file() for path in (result_path,status_path,manifest_path)):
        raise StageEE1Error(f"attempt is absent or incomplete: {root}")
    result, result_raw = _strict_json(result_path, "E1 attempt result")
    status, status_raw = _strict_json(status_path, "E1 attempt status")
    manifest, _ = _strict_json(manifest_path, "E1 attempt manifest")
    for value, name in ((result,"result"),(status,"status"),(manifest,"manifest")):
        _validate_signed(value, f"E1 attempt {name}")
    if (
        result.get("unit_digest") != unit["canonical_digest"]
        or result.get("variant_id") != unit["variant_id"]
        or result.get("sample_id") != unit["sample_id"]
        or result.get("validity_status") not in {"VALID","FAILED","NA"}
        or status.get("terminal") is not True
        or status.get("attempt_id") != "attempt-000001"
        or manifest.get("result_sha256") != _sha(result_raw)
        or manifest.get("status_sha256") != _sha(status_raw)
    ):
        raise StageEE1Error("E1 attempt control binding failed")
    if result["validity_status"] == "VALID":
        if (
            result.get("failure_class") is not None
            or result.get("failure_code") is not None
            or result.get("na_reason") is not None
            or result.get("attempt_id") != "attempt-000001"
            or result.get("retry_count") != 0
        ):
            raise StageEE1Error("valid E1 attempt has failure/NA/retry drift")
        curves = result.get("curves")
        if not isinstance(curves,list) or len(curves) != 10:
            raise StageEE1Error("valid E1 attempt lacks ten curves")
        source_map_path=repo/unit["source_map_path"]
        source_result_path=repo/unit["source_result_path"]
        if _sha_file(source_map_path) != unit["source_map_sha256"] or _sha_file(source_result_path) != unit["source_result_sha256"]:
            raise StageEE1Error("E1 attempt source bytes drifted")
        source_map=np.load(source_map_path,allow_pickle=False)
        patch_scores=aggregate_patch_map(source_map)
        source_result,_=_strict_json(source_result_path,"E1 attempt source result")
        sealed_identity=source_result.get("target_identity")
        target_response=result.get("target_response")
        if not isinstance(target_response,Mapping) or target_response.get("sigma") not in (-1,1):
            raise StageEE1Error("E1 attempt target response is invalid")
        target_identity=target_response.get("target_identity")
        if _target_semantic_digest(target_identity) != _target_semantic_digest(sealed_identity):
            raise StageEE1Error("E1 attempt target/source semantic binding failed")
        expected_hashes={
            "source_map_sha256":unit["source_map_sha256"],
            "source_result_sha256":unit["source_result_sha256"],
            "source_unit_digest":unit["source_unit_digest"],
            "image_sha256":unit["image_sha256"],
            "model_sha256":MODEL_SHA256,
            "model_canonical_tensor_digest":MODEL_TENSOR_DIGEST,
            "target_semantic_digest":_target_semantic_digest(target_identity),
            "signed_patch_score_sha256":_array_digest(patch_scores),
        }
        if result.get("source_hashes") != expected_hashes:
            raise StageEE1Error("E1 attempt source-hash binding failed")
        expected_pair={
            "sample_id":unit["sample_id"],"selection_order":unit["selection_order"],
            "image_sha256":unit["image_sha256"],"model_sha256":MODEL_SHA256,
            "target":"fixed_clean_attention_cls_head_direction",
            "target_semantic_digest":expected_hashes["target_semantic_digest"],
            "method":unit["method"],"variant_id":unit["variant_id"],
            "block":0,"head":9,"scale":"native_14x14_patch",
            "replacement":"normalized_zero_canonical_rgb_0_5",
            "fractions":list(FRACTIONS),
            "seeds":[e1_seed(unit["sample_id"],repeat) for repeat in range(5)],
            "seed_comparison_identity":COMMON_SEED_IDENTITY,
            "source_map_sha256":unit["source_map_sha256"],
            "source_result_sha256":unit["source_result_sha256"],
            "attempt_id":"attempt-000001",
        }
        if result.get("pair_binding") != expected_pair or result.get("pair_binding_digest") != canonical_sha256(expected_pair):
            raise StageEE1Error("E1 attempt pair binding failed")
        source_sum=float(source_map.sum(dtype=np.float64)); patch_sum=float(patch_scores.sum(dtype=np.float64))
        expected_map_qc={"source_shape":[56,56],"target_shape":[14,14],"dtype":"float64","source_signed_sum":source_sum,"patch_signed_sum":patch_sum,"absolute_error":abs(patch_sum-source_sum),"tolerance":1e-12+1e-12*abs(source_sum),"status":"PASS"}
        if result.get("map_aggregation_qc") != expected_map_qc:
            raise StageEE1Error("E1 attempt map aggregation QC failed")
        expected = {(direction,repeat) for direction in ("insertion","deletion") for repeat in range(5)}
        observed = {(curve.get("direction"),curve.get("repeat")) for curve in curves}
        if observed != expected:
            raise StageEE1Error("E1 curve coordinate coverage failed")
        for curve in curves:
            expected_seed=e1_seed(unit["sample_id"],curve["repeat"])
            expected_priority=tie_priority(expected_seed)
            expected_ranking=patch_ranking(patch_scores,sigma=target_response["sigma"],seed=expected_seed)
            if (
                curve.get("variant_id") != unit["variant_id"]
                or curve.get("method") != unit["method"]
                or curve.get("sample_id") != unit["sample_id"]
                or curve.get("seed") != expected_seed
                or curve.get("seed_comparison_identity") != COMMON_SEED_IDENTITY
                or curve.get("tie_priority_sha256") != _array_digest(expected_priority)
                or curve.get("signed_patch_score_sha256") != _array_digest(patch_scores)
                or curve.get("ranking_sha256") != _array_digest(expected_ranking)
                or curve.get("unit_counts") != list(UNIT_COUNTS)
                or curve.get("actual_fractions") != list(ACTUAL_FRACTIONS)
                or len(curve.get("responses",[])) != 21
                or len(curve.get("intervention_sha256",[])) != 21
                or curve.get("auc") != trapezoid_auc(curve["responses"])
            ):
                raise StageEE1Error("E1 curve source recomputation failed")
            if any(not isinstance(item,str) or len(item)!=64 for item in curve["intervention_sha256"]):
                raise StageEE1Error("E1 curve intervention digest failed")
        for direction in ("insertion","deletion"):
            values = np.asarray([curve["auc"] for curve in curves if curve["direction"]==direction],dtype=np.float64)
            if result[f"{direction}_repeat_aucs"] != values.tolist() or result[f"{direction}_auc_mean"] != float(values.mean()) or result[f"{direction}_auc_repeat_sd_ddof_0"] != float(values.std(ddof=0)):
                raise StageEE1Error("E1 repeat aggregation failed")
        if result.get("forward_calls") != 213 or result.get("unit_count") != 196 or result.get("unit_pixels") != [16,16]:
            raise StageEE1Error("E1 attempt execution accounting/patch geometry failed")
        if unit["variant_id"]==PATCH_VARIANTS[1]:
            matched=source_result["provenance_fragment"]["matched_contribution"]
            expected_matched={"cls_contribution":float(matched["cls_contribution"]),"cls_in_ranking":False,"native_patch_sum":patch_sum,"token_sum":float(matched["token_sum"]),"absolute_error":abs((patch_sum+float(matched["cls_contribution"]))-float(matched["token_sum"])),"tolerance":float(matched["conservation_tolerance"]),"status":"PASS"}
            if result.get("matched_conservation_qc") != expected_matched:
                raise StageEE1Error("E1 matched CLS/native conservation failed")
        elif result.get("matched_conservation_qc") is not None:
            raise StageEE1Error("NNS attempt contains matched-only QC")
    return {"result":result,"status":status,"manifest":manifest,"attempt_path":root}


def _publish_attempt(repo: Path, unit: Mapping[str, Any], result: Mapping[str, Any]) -> Path:
    root, result_path, status_path, manifest_path = _attempt_paths(repo, unit["canonical_digest"])
    if root.exists():
        raise StageEE1Error(f"refusing to overwrite terminal attempt: {root}")
    attempts_root = root.parent
    attempts_root.mkdir(parents=True, exist_ok=True)
    if any(path.name.startswith(".attempt-") for path in attempts_root.iterdir()):
        raise StageEE1Error(f"ambiguous unpublished attempt exists: {attempts_root}")
    temporary = Path(tempfile.mkdtemp(prefix=".attempt-000001-", dir=attempts_root))
    try:
        signed_result = _signed(result)
        result_raw = canonical_json_bytes(signed_result) + b"\n"
        result_temp = temporary / "result.json"
        result_temp.write_bytes(result_raw)
        status = _signed({
            "schema_version":1, "document_type":"stage_e_e1_attempt_status",
            "attempt_id":"attempt-000001", "attempt_number":1,
            "unit_digest":unit["canonical_digest"],
            "state":"succeeded" if signed_result["validity_status"]=="VALID" else "scientific_terminal",
            "terminal":True, "retry_count":0,
            "failure_class":signed_result["failure_class"],
            "failure_code":signed_result["failure_code"],
            "na_reason":signed_result["na_reason"],
            "published_at":_now(),
        })
        status_raw = canonical_json_bytes(status) + b"\n"
        (temporary / "status.json").write_bytes(status_raw)
        manifest = _signed({
            "schema_version":1, "document_type":"stage_e_e1_attempt_manifest",
            "attempt_id":"attempt-000001", "attempt_number":1,
            "unit_digest":unit["canonical_digest"],
            "result_sha256":_sha(result_raw), "result_bytes":len(result_raw),
            "status_sha256":_sha(status_raw), "status_bytes":len(status_raw),
            "source_map_sha256":unit["source_map_sha256"],
            "source_result_sha256":unit["source_result_sha256"],
            "immutable":True,
        })
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest)+b"\n")
        for path in temporary.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, root)
        directory_fd = os.open(attempts_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    _validate_attempt(repo, unit)
    return root


def _valid_result(
    *, unit: Mapping[str, Any], source_result: Mapping[str, Any],
    source_map: np.ndarray, patch_scores: np.ndarray, target_identity: Mapping[str, Any],
    response: Any, curves: list[dict[str, Any]], elapsed_seconds: float,
    cpu_seconds: float, forward_calls: int, peak_cuda_bytes: int,
) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for direction in ("insertion","deletion"):
        aucs = np.asarray([row["auc"] for row in curves if row["direction"]==direction],dtype=np.float64)
        aggregates[f"{direction}_repeat_aucs"] = aucs.tolist()
        aggregates[f"{direction}_auc_mean"] = float(aucs.mean())
        aggregates[f"{direction}_auc_repeat_sd_ddof_0"] = float(aucs.std(ddof=0))
    matched_qc = None
    if unit["variant_id"] == PATCH_VARIANTS[1]:
        matched = source_result["provenance_fragment"]["matched_contribution"]
        cls = float(matched["cls_contribution"])
        patch_sum = float(patch_scores.sum(dtype=np.float64))
        token_sum = float(matched["token_sum"])
        tolerance = float(matched["conservation_tolerance"])
        if abs((patch_sum + cls) - token_sum) > tolerance:
            raise StageEE1Error("matched native patch plus CLS conservation failed")
        matched_qc = {
            "cls_contribution":cls, "cls_in_ranking":False,
            "native_patch_sum":patch_sum, "token_sum":token_sum,
            "absolute_error":abs((patch_sum+cls)-token_sum),
            "tolerance":tolerance, "status":"PASS",
        }
    source_sum = float(source_map.sum(dtype=np.float64))
    patch_sum = float(patch_scores.sum(dtype=np.float64))
    source_hashes = {
        "source_map_sha256":unit["source_map_sha256"],
        "source_result_sha256":unit["source_result_sha256"],
        "source_unit_digest":unit["source_unit_digest"],
        "image_sha256":unit["image_sha256"], "model_sha256":MODEL_SHA256,
        "model_canonical_tensor_digest":MODEL_TENSOR_DIGEST,
        "target_semantic_digest":_target_semantic_digest(target_identity),
        "signed_patch_score_sha256":_array_digest(patch_scores),
    }
    pair_binding = {
        "sample_id":unit["sample_id"], "selection_order":unit["selection_order"],
        "image_sha256":unit["image_sha256"], "model_sha256":MODEL_SHA256,
        "target":"fixed_clean_attention_cls_head_direction",
        "target_semantic_digest":source_hashes["target_semantic_digest"],
        "method":unit["method"], "variant_id":unit["variant_id"],
        "block":0, "head":9, "scale":"native_14x14_patch",
        "replacement":"normalized_zero_canonical_rgb_0_5",
        "fractions":list(FRACTIONS),
        "seeds":[e1_seed(unit["sample_id"],repeat) for repeat in range(5)],
        "seed_comparison_identity":COMMON_SEED_IDENTITY,
        "source_map_sha256":unit["source_map_sha256"],
        "source_result_sha256":unit["source_result_sha256"],
        "attempt_id":"attempt-000001",
    }
    return {
        "schema_version":1, "document_type":"stage_e_e1_unit_result",
        "unit_digest":unit["canonical_digest"],
        "variant_id":unit["variant_id"], "sample_id":unit["sample_id"],
        "selection_order":unit["selection_order"],
        "cohort":"stage_d_fidelity_v1_500", "stratum":"overall",
        "validity_status":"VALID", "failure_class":None,
        "failure_code":None, "na_reason":None,
        "attempt_id":"attempt-000001", "retry_count":0,
        "method":unit["method"], "target":"fixed_clean_attention_cls_head_direction",
        "scale":"native_14x14_patch", "layer":0, "head":9,
        "unit_count":196, "unit_pixels":[16,16],
        "source_hashes":source_hashes, "pair_binding":pair_binding,
        "pair_binding_digest":canonical_sha256(pair_binding),
        "map_aggregation_qc":{"source_shape":[56,56],"target_shape":[14,14],"dtype":"float64","source_signed_sum":source_sum,"patch_signed_sum":patch_sum,"absolute_error":abs(patch_sum-source_sum),"tolerance":1e-12+1e-12*abs(source_sum),"status":"PASS"},
        "matched_conservation_qc":matched_qc,
        "target_response":response.to_dict(), "curves":curves, **aggregates,
        "elapsed_seconds":float(elapsed_seconds), "cpu_seconds":float(cpu_seconds),
        "gpu_hours":float(elapsed_seconds/3600.0), "forward_calls":int(forward_calls),
        "peak_cuda_memory_bytes":int(peak_cuda_bytes),
        "artifact_bytes":0,
    }


def run_shard(
    repo_root: Path | str = REPO_ROOT, *, shard_id: int, shard_count: int,
    device: str = "cuda:0", limit: int | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if type(shard_count) is not int or shard_count <= 0 or type(shard_id) is not int or not 0 <= shard_id < shard_count:
        raise StageEE1Error("invalid shard identity")
    validate_initialized(repo)
    plan, _ = _strict_json(repo / PLAN_PATH, "E1 plan")
    registry, _ = _load_registry(repo)
    units_by_sample: dict[str,list[dict[str,Any]]] = {}
    for unit in plan["units"]:
        units_by_sample.setdefault(unit["sample_id"],[]).append(unit)
    selected = [row for row in registry if row["selection_order"] % shard_count == shard_id]
    if limit is not None:
        selected = selected[:limit]
    import torch
    from .stage_c_fidelity import ValuePathScalarAdapter, resolve_target_response
    from .stage_c_value_path import resolve_fixed_clean_attention_head_direction
    model = binding = None
    shard_start = time.monotonic()
    cpu_start = time.process_time()
    completed = skipped = 0
    group_records: list[dict[str,Any]] = []
    try:
        model, binding, model_record = _load_model_and_context(repo, device)
        for row in selected:
            unit_pair = sorted(units_by_sample[row["sample_id"]], key=lambda item:item["variant_id"])
            existing = []
            for unit in unit_pair:
                attempt_root = _attempt_paths(repo,unit["canonical_digest"])[0]
                if attempt_root.exists():
                    existing.append(_validate_attempt(repo,unit))
            if existing:
                if len(existing) == 2:
                    skipped += 2
                    continue
            group_start = time.monotonic(); group_cpu_start = time.process_time()
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            source_results, source_maps = _load_source_pair(repo,unit_pair)
            _canonical, clean, normalized_sha = _load_sample(binding,row,model,device)
            if any(source_results[v]["clean_tensor_hashes"]["normalized_tensor_sha256"] != normalized_sha for v in PATCH_VARIANTS):
                raise StageEE1Error("live normalized input differs from sealed D9 binding")
            calls = 0
            handle = model.register_forward_pre_hook(lambda _module,_args: None)
            handle.remove()
            def count_hook(_module: Any, _args: Any) -> None:
                nonlocal calls
                calls += 1
            handle = model.register_forward_pre_hook(count_hook)
            target_started=time.monotonic(); target_cpu_started=time.process_time()
            try:
                target = resolve_fixed_clean_attention_head_direction(model,clean,0,9)
                evaluator = ValuePathScalarAdapter(model,clean,target)
                response = resolve_target_response(evaluator,clean,torch.zeros_like(clean))
                if response.na is not None:
                    raise StageEE1Error("frozen target unexpectedly became NA")
                target_elapsed=time.monotonic()-target_started
                target_cpu=time.process_time()-target_cpu_started
                target_calls=calls
                live_identity = json.loads(canonical_json_bytes(evaluator.target_identity).decode("utf-8"))
                sealed_identity = source_results[PATCH_VARIANTS[0]]["target_identity"]
                if _target_semantic_digest(live_identity) != _target_semantic_digest(sealed_identity):
                    raise StageEE1Error("live target differs semantically from sealed D9 target")
                unit_outputs: list[tuple[Mapping[str,Any],dict[str,Any]]] = []
                missing_units=[unit for unit in unit_pair if not _attempt_paths(repo,unit["canonical_digest"])[0].exists()]
                for missing_index,unit in enumerate(missing_units):
                    variant=unit["variant_id"]; start=time.monotonic(); cpu0=time.process_time(); calls0=calls
                    patch_scores=aggregate_patch_map(source_maps[variant])
                    curves=[]
                    for direction in ("insertion","deletion"):
                        for repeat in range(5):
                            curves.append(_curve(evaluator=evaluator,response=response,clean=clean,patch_scores=patch_scores,sample_id=row["sample_id"],variant=variant,direction=direction,repeat=repeat))
                    divisor=float(len(missing_units))
                    elapsed=time.monotonic()-start + target_elapsed/divisor
                    cpu_seconds=time.process_time()-cpu0 + target_cpu/divisor
                    target_share=target_calls//len(missing_units)+(1 if missing_index < target_calls%len(missing_units) else 0)
                    peak=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0
                    result=_valid_result(unit=unit,source_result=source_results[variant],source_map=source_maps[variant],patch_scores=patch_scores,target_identity=live_identity,response=response,curves=curves,elapsed_seconds=elapsed,cpu_seconds=cpu_seconds,forward_calls=calls-calls0+target_share,peak_cuda_bytes=peak)
                    unit_outputs.append((unit,result))
            finally:
                handle.remove()
            for unit,result in unit_outputs:
                path=_publish_attempt(repo,unit,result)
                result_doc,_=_strict_json(path/"result.json","published E1 result")
                result_doc["artifact_bytes"]=sum(p.stat().st_size for p in path.iterdir())
                # artifact_bytes is telemetry only; immutable result bytes remain authoritative.
                completed += 1
            group_records.append({"sample_id":row["sample_id"],"selection_order":row["selection_order"],"elapsed_seconds":time.monotonic()-group_start,"cpu_seconds":time.process_time()-group_cpu_start,"forward_calls":calls,"peak_cuda_memory_bytes":int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0})
            del clean, target, evaluator, response
        shard = _signed({
            "schema_version":1,"document_type":"stage_e_e1_shard_telemetry",
            "shard_id":shard_id,"shard_count":shard_count,"device":device,
            "gpu_name":torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "started_groups":len(selected),"completed_units":completed,"skipped_terminal_units":skipped,
            "elapsed_seconds":time.monotonic()-shard_start,"cpu_seconds":time.process_time()-cpu_start,
            "gpu_hours":(time.monotonic()-shard_start)/3600.0 if device.startswith("cuda") else 0.0,
            "forward_calls":sum(x["forward_calls"] for x in group_records),
            "peak_cuda_memory_bytes":max((x["peak_cuda_memory_bytes"] for x in group_records),default=0),
            "groups":group_records,"model":model_record,"completed_at":_now(),
        })
        telemetry_root=repo/ROOT_PATH/"telemetry"
        _write_or_verify(telemetry_root/f"shard-{shard_id:02d}-of-{shard_count:02d}.json",shard)
        return shard
    finally:
        if model is not None:
            del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _median(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values,dtype=np.float64))
    count = ordered.size
    if count == 0:
        raise StageEE1Error("cannot summarize an empty array")
    middle = count // 2
    return float(ordered[middle]) if count % 2 else float((ordered[middle-1]+ordered[middle])/2.0)


def _bootstrap(values: np.ndarray, variant_id: str) -> dict[str, Any]:
    source=np.asarray(values,dtype=np.float64)
    material="stage-e-bootstrap|20260906|E1|"+variant_id+"|overall"
    seed=int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8],"big")
    generator=np.random.Generator(np.random.PCG64(seed))
    medians=np.empty(10000,dtype=np.float64)
    for start in range(0,10000,1000):
        stop=min(start+1000,10000)
        indices=generator.integers(0,source.size,size=(stop-start,source.size))
        medians[start:stop]=np.median(source[indices],axis=1)
    interval=np.quantile(medians,[0.025,0.975],method="linear")
    return {"resamples":10000,"seed":seed,"prng":"numpy.random.Generator(numpy.random.PCG64(seed))","statistic":"arithmetic_even_median","quantile_method":"linear","ci_95":[float(interval[0]),float(interval[1])],"role":"DESCRIPTIVE_ONLY"}


def _summary(values: Sequence[float], variant_id: str) -> dict[str,Any]:
    source=np.asarray(values,dtype=np.float64)
    if source.shape != (500,) or not np.isfinite(source).all():
        raise StageEE1Error("summary requires all 500 finite image values")
    return {
        "n":500,"valid":500,"failed":0,"na":0,
        "positive_count":int(np.sum(source>0.0)),
        "mean":float(source.mean()),"median":_median(source),
        "sample_sd_ddof_1":float(source.std(ddof=1)),
        "paired_image_bootstrap_median_95_interval":_bootstrap(source,variant_id),
    }


def _table(table_id: str, rows: list[dict[str,Any]]) -> dict[str,Any]:
    columns=list(rows[0]) if rows else []
    if any(list(row) != columns for row in rows):
        raise StageEE1Error("output table rows do not share one ordered column set")
    return _signed({"schema_version":1,"document_type":"stage_e_e1_output_table","table_id":table_id,"columns":columns,"row_count":len(rows),"rows":rows})


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value,(dict,list)):
        return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
    if isinstance(value,bool):
        return "true" if value else "false"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name("."+path.name+".pending")
    if temporary.exists():
        raise StageEE1Error(f"ambiguous pending CSV: {temporary}")
    with temporary.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key:_csv_scalar(value) for key,value in row.items()} for row in rows)
        stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise StageEE1Error(f"immutable CSV differs: {path}")
        temporary.unlink()
    else:
        os.replace(temporary,path)


def _write_output_json(repo: Path, relative: str, value: Mapping[str,Any]) -> None:
    schema,_=_strict_json(repo/OUTPUT_SCHEMA_PATH,"E1 output schema")
    Draft202012Validator.check_schema(schema)
    errors=list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise StageEE1Error(f"E1 output schema failure: {errors[0].message}")
    _write_or_verify(repo/relative,value)


def _anchor_rows(repo: Path) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    source,raw=_strict_json(repo/D10_ROWS_PATH,"D10 per-sample anchor")
    if _sha(raw) != "8bb2e36dc2b4b3895bc5688b7ab5789719995add0c26cd48608d593486ba0cd2" or source.get("canonical_digest") != "4a89aeb89884219779699e15b49b017106bced1e60ab146dafac04d63ee9d9c0":
        raise StageEE1Error("D10 per-sample anchor binding failed")
    rows=[]
    for item in source["rows"]:
        rows.append({
            "variant_id":VARIANTS[0],"sample_id":item["sample_id"],"selection_order":item["selection_order"],
            "cohort":"stage_d_fidelity_v1_500","stratum":"overall","validity_status":"VALID",
            "source_hashes":{"d10_per_sample_sha256":_sha(raw),"d10_per_sample_canonical_digest":source["canonical_digest"],"d10_report_sha256":"7eac27b11b9df72a91c6e78cd0d46ea30f7d8701f0d41cbc29e9cb76695b5a08"},
            "method":"nns_vs_matched","target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","layer":0,"head":9,
            "failure_class":None,"failure_code":None,"na_reason":None,"attempt_id":"D10_READ_ONLY","retry_count":0,
            "insertion_repeat_aucs":None,"deletion_repeat_aucs":None,
            "insertion_auc_mean":None,"deletion_auc_mean":None,
            "insertion_auc_repeat_sd_ddof_0":None,"deletion_auc_repeat_sd_ddof_0":None,
            "nns_insertion_auc_mean":item["nns_insertion_auc"],"nns_deletion_auc_mean":item["nns_deletion_auc"],
            "matched_insertion_auc_mean":item["matched_insertion_auc"],"matched_deletion_auc_mean":item["matched_deletion_auc"],
            "delta_fid":item["delta_fid"],"d10_fine_4x4_delta_fid":item["delta_fid"],"delta_fid_patch_minus_fine":None,
            "elapsed_seconds":0.0,"gpu_hours":0.0,"cpu_seconds":0.0,"forward_calls":0,"peak_cuda_memory_bytes":0,"artifact_bytes":0,
            "recompute_status":"FORBIDDEN_RECOMPUTE","descriptive_only":True,
        })
    return rows,source


def _collect_patch_rows(repo: Path, plan: Mapping[str,Any]) -> tuple[list[dict[str,Any]],dict[str,dict[str,Any]]]:
    attempts: dict[str,dict[str,Any]]={}
    for unit in plan["units"]:
        attempts[unit["canonical_digest"]]=_validate_attempt(repo,unit)
    rows=[]
    by_sample: dict[str,dict[str,dict[str,Any]]]={}
    for unit in plan["units"]:
        observed=attempts[unit["canonical_digest"]]
        result=observed["result"]
        by_sample.setdefault(unit["sample_id"],{})[unit["variant_id"]]=result
    for group in plan["groups"]:
        pair=by_sample.get(group["sample_id"],{})
        if set(pair) != set(PATCH_VARIANTS):
            raise StageEE1Error("one image lacks both patch variants")
        nns,matched=(pair[PATCH_VARIANTS[0]],pair[PATCH_VARIANTS[1]])
        if nns["target_response"] != matched["target_response"]:
            raise StageEE1Error("paired methods do not use the same target response")
        for direction in ("insertion","deletion"):
            for repeat in range(5):
                left=next(x for x in nns["curves"] if x["direction"]==direction and x["repeat"]==repeat)
                right=next(x for x in matched["curves"] if x["direction"]==direction and x["repeat"]==repeat)
                if left["seed"] != right["seed"] or left["tie_priority_sha256"] != right["tie_priority_sha256"]:
                    raise StageEE1Error("paired methods do not share tie priority")
        delta=0.5*((nns["insertion_auc_mean"]-matched["insertion_auc_mean"])+(matched["deletion_auc_mean"]-nns["deletion_auc_mean"]))
        for variant,result in ((PATCH_VARIANTS[0],nns),(PATCH_VARIANTS[1],matched)):
            attempt_path=attempts[result["unit_digest"]]["attempt_path"]
            rows.append({
                "variant_id":variant,"sample_id":result["sample_id"],"selection_order":result["selection_order"],
                "cohort":"stage_d_fidelity_v1_500","stratum":"overall","validity_status":result["validity_status"],
                "source_hashes":result["source_hashes"],"method":result["method"],"target":result["target"],"scale":result["scale"],"layer":0,"head":9,
                "failure_class":result["failure_class"],"failure_code":result["failure_code"],"na_reason":result["na_reason"],"attempt_id":result["attempt_id"],"retry_count":result["retry_count"],
                "insertion_repeat_aucs":result["insertion_repeat_aucs"],"deletion_repeat_aucs":result["deletion_repeat_aucs"],
                "insertion_auc_mean":result["insertion_auc_mean"],"deletion_auc_mean":result["deletion_auc_mean"],
                "insertion_auc_repeat_sd_ddof_0":result["insertion_auc_repeat_sd_ddof_0"],"deletion_auc_repeat_sd_ddof_0":result["deletion_auc_repeat_sd_ddof_0"],
                "nns_insertion_auc_mean":nns["insertion_auc_mean"],"nns_deletion_auc_mean":nns["deletion_auc_mean"],
                "matched_insertion_auc_mean":matched["insertion_auc_mean"],"matched_deletion_auc_mean":matched["deletion_auc_mean"],
                "delta_fid":delta,"d10_fine_4x4_delta_fid":None,"delta_fid_patch_minus_fine":None,
                "elapsed_seconds":result["elapsed_seconds"],"gpu_hours":result["gpu_hours"],"cpu_seconds":result["cpu_seconds"],"forward_calls":result["forward_calls"],"peak_cuda_memory_bytes":result["peak_cuda_memory_bytes"],
                "artifact_bytes":sum(path.stat().st_size for path in attempt_path.iterdir()),
                "recompute_status":"NEW_PATCH_RESPONSE_COMPUTE","descriptive_only":True,
            })
    rows.sort(key=lambda row:(row["selection_order"],VARIANTS.index(row["variant_id"])))
    return rows,attempts


def _protected_after(repo: Path, before: Mapping[str,Any]) -> dict[str,Any]:
    after=_upstream_snapshot(repo)
    for name in ("d9","d10","e0"):
        if any(before[name].get(key) != after[name].get(key) for key in ("file_count","total_bytes","canonical_digest")):
            raise StageEE1Error(f"protected {name} root changed during E1")
    return after


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve()
    config=_validate_config(repo)
    plan,_=_strict_json(repo/PLAN_PATH,"E1 plan")
    rebuilt=_build_plan(repo,config)
    if plan != rebuilt:
        raise StageEE1Error("E1 plan drifted")
    rows,attempts=_collect_patch_rows(repo,plan)
    if len(rows) != 1000 or any(row["validity_status"] != "VALID" for row in rows):
        raise StageEE1Error("E1 new-compute terminal coverage is incomplete or invalid")
    anchor_rows,anchor_source=_anchor_rows(repo)
    anchor={row["sample_id"]:row for row in anchor_rows}
    for row in rows:
        row["d10_fine_4x4_delta_fid"]=anchor[row["sample_id"]]["delta_fid"]
        row["delta_fid_patch_minus_fine"]=row["delta_fid"]-row["d10_fine_4x4_delta_fid"]
    all_rows=sorted(anchor_rows+rows,key=lambda row:(row["selection_order"],VARIANTS.index(row["variant_id"])))
    if len(all_rows)!=1500:
        raise StageEE1Error("E1 output must retain 1500 rows across three variants")
    patch_by_sample={}
    for row in rows:
        patch_by_sample.setdefault(row["sample_id"],row["delta_fid"] )
    if len(patch_by_sample)!=500:
        raise StageEE1Error("E1 paired image delta coverage failed")
    patch_values=[patch_by_sample[group["sample_id"]] for group in plan["groups"]]
    fine_values=[anchor[group["sample_id"]]["delta_fid"] for group in plan["groups"]]
    delta_delta=[patch-fine for patch,fine in zip(patch_values,fine_values)]
    method_summaries={}
    for variant in PATCH_VARIANTS:
        selected=[row for row in rows if row["variant_id"]==variant]
        for direction in ("insertion","deletion"):
            values=[row[f"{direction}_auc_mean"] for row in selected]
            method_summaries[f"{variant}:{direction}_auc"]=_summary(values,f"{variant}_{direction}_auc")
    summary=_signed({
        "schema_version":1,"document_type":"stage_e_e1_overall_descriptive_summary",
        "stage_id":"E1","cohort":"stage_d_fidelity_v1_500","stratum":"overall",
        "descriptive_only":True,"p_values_emitted":False,"winner_selection_performed":False,"new_go_no_go_issued":False,
        "counts":{"expected_images":500,"anchor_rows":500,"new_compute_rows":1000,"valid":1000,"failed":0,"na":0},
        "d10_fine_anchor":{"recompute_status":"FORBIDDEN_RECOMPUTE","n":500,"positive_count":500,"mean":2.630671903027281,"median":2.3081996221714247,"sample_sd_ddof_1":1.4806950824072174,"paired_image_bootstrap_ci_95":[2.1110806126093067,2.445235411755993],"source_canonical_digest":anchor_source["canonical_digest"]},
        "delta_fid_patch":_summary(patch_values,"delta_fid_patch"),
        "delta_delta_fid_patch_minus_fine":_summary(delta_delta,"delta_delta_fid_patch_minus_fine"),
        "method_direction_descriptions":method_summaries,
        "interpretation_boundary":"Patch-scale results are post-D10 descriptive sensitivity values and do not replace the D10 primary or create a winner or GO/NO_GO decision.",
    })
    resources=[dict(row) for row in rows]
    telemetry=[]
    telemetry_root=repo/ROOT_PATH/"telemetry"
    for path in sorted(telemetry_root.glob("shard-*.json")):
        document,_=_strict_json(path,"E1 shard telemetry")
        _validate_signed(document,"E1 shard telemetry")
        telemetry.append(document)
    event,_=_strict_json(
        repo/REPORT_ROOT_PATH/"pre_outcome_infrastructure_event.json",
        "pre-outcome infrastructure event",
    )
    _validate_signed(event,"pre-outcome infrastructure event")
    charge=event["resource_accounting"]
    total={
        "elapsed_seconds":sum(float(item["elapsed_seconds"]) for item in telemetry)+float(charge["gpu_device_seconds_charged"]),
        "gpu_hours":sum(float(item["gpu_hours"]) for item in telemetry)+float(charge["gpu_hours_charged"]),
        "cpu_seconds":sum(float(item["cpu_seconds"]) for item in telemetry)+float(charge["cpu_seconds_charged"]),
        "forward_calls":sum(int(item["forward_calls"]) for item in telemetry),
        "peak_cuda_memory_bytes":max(int(item["peak_cuda_memory_bytes"]) for item in telemetry),
        "artifact_bytes":sum(row["artifact_bytes"] for row in resources),
        "gpu_invocation_count":len(telemetry)+1,
        "pre_outcome_failed_invocations":1,
        "successful_terminal_unit_attempts":1000,
        "retry_unit_attempts":0,
    }
    summary["resource_accounting"]={
        "global":total,
        "by_variant":{variant:{
            "elapsed_seconds":sum(row["elapsed_seconds"] for row in resources if row["variant_id"]==variant),
            "gpu_hours_scientific_unit_share":sum(row["gpu_hours"] for row in resources if row["variant_id"]==variant),
            "cpu_seconds_scientific_unit_share":sum(row["cpu_seconds"] for row in resources if row["variant_id"]==variant),
            "forward_calls":sum(row["forward_calls"] for row in resources if row["variant_id"]==variant),
        } for variant in PATCH_VARIANTS},
        "failed_pre_outcome_event":event,
    }
    summary["canonical_digest"]=_self_digest(summary)
    return {"config":config,"plan":plan,"rows":all_rows,"patch_rows":rows,"resource_rows":resources,"summary":summary,"attempts":attempts,"actual_cost":total}


FALSE_PASS_ATTACKS=(
    "e0_unverified","d10_anchor_only","map_aggregation_without_patch_responses","fewer_than_500_images","fewer_than_two_patch_methods","cell_3136_masquerades_as_patch_196","unpaired_seeds","missing_curves_auc_pair_bindings_or_attempts","failed_na_nonfinite_dropped","favorable_valid_subset_summary","d10_primary_recomputed_or_modified","protected_upstream_modified","budget_exceeded","tests_missing_or_failed","receipt_manifest_inventory_empty_or_placeholder","e1_pass_overauthorizes_later_stage",
)


def false_pass_reasons(preconditions: Mapping[str,bool]) -> tuple[str,...]:
    if set(preconditions) != set(FALSE_PASS_ATTACKS):
        raise StageEE1Error("false-PASS precondition set drifted")
    return tuple(name for name in FALSE_PASS_ATTACKS if preconditions[name] is not True)


def budget_reasons(actual: Mapping[str,Any], retained_bytes: int) -> tuple[str,...]:
    failures=[]
    if float(actual.get("gpu_hours",math.inf)) > 2.5:
        failures.append("gpu_hours")
    if float(actual.get("cpu_seconds",math.inf))/3600.0 > 8.0:
        failures.append("cpu_hours")
    if type(retained_bytes) is not int or retained_bytes > 4294967296:
        failures.append("retained_bytes")
    return tuple(failures)


def _publish_access_terminal(repo: Path, initial: Mapping[str,Any]) -> dict[str,Any]:
    pending=repo/ACCESS_PENDING_PATH
    if (repo/ACCESS_PATH).is_file():
        observed,_=_strict_json(repo/ACCESS_PATH,"terminal outcome access")
        _validate_signed(observed,"terminal outcome access")
        if observed.get("initial_pending_canonical_digest") != initial["canonical_digest"]:
            raise StageEE1Error("terminal outcome access does not bind initial pending record")
        return observed
    if not pending.is_file():
        raise StageEE1Error("outcome-access pending/terminal state is invalid")
    initial_raw=pending.read_bytes()
    terminal=_signed({
        "schema_version":1,"document_type":"stage_e_e1_outcome_access",
        "status":"TERMINAL","e1_started_at":initial["e1_started_at"],"completed_at":_now(),
        "e0_pass_verified":True,"d10_outcome_known":True,"new_e1_outcome_accessed":True,
        "first_outcome_access_after_pending_record":True,"initial_pending_sha256":_sha(initial_raw),
        "initial_pending_canonical_digest":initial["canonical_digest"],
        "scope_changed_after_outcome_access":False,"variants_added_or_removed":False,
        "failed_na_or_unfavorable_samples_dropped":False,"d10_primary_recomputed":False,
        "e2_started":False,"e3_started":False,"e4_started":False,"stage_f_started":False,
        "protected_roots":initial["protected_roots"],"upstream_reads":initial["upstream_reads"],
    })
    raw=canonical_json_bytes(terminal)+b"\n"
    temporary=pending.with_name(".outcome-access-terminal.tmp")
    if temporary.exists():
        raise StageEE1Error("ambiguous access terminal temporary file")
    _write_new_json(temporary,terminal)
    os.replace(temporary,pending)
    observed,_=_strict_json(pending,"terminal pending access")
    if observed != terminal or pending.read_bytes()!=raw:
        raise StageEE1Error("access record changed before atomic publication")
    os.replace(pending,repo/ACCESS_PATH)
    return terminal


def finalize(repo_root: Path | str = REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve()
    outputs=validate_outputs(repo)
    config=outputs["config"]
    before,_=_strict_json(repo/UPSTREAM_BEFORE_PATH,"protected upstream before")
    after=_protected_after(repo,before)
    actual=outputs["actual_cost"]
    root=repo/ROOT_PATH
    current_bytes=_inventory(root)["total_bytes"]
    if budget_reasons(actual,current_bytes):
        raise StageEE1Error(f"E1 budget exceeded: gpu={actual['gpu_hours']} cpu={actual['cpu_seconds']/3600.0} bytes={current_bytes}")
    patch_table=_table("patch_scale_sensitivity",outputs["rows"])
    resource_table=_table("resource_and_qc",outputs["resource_rows"])
    paper=root/"paper_data"
    _write_output_json(repo,ROOT_PATH+"/paper_data/patch_scale_sensitivity.json",patch_table)
    _write_csv(paper/"patch_scale_sensitivity.csv",outputs["rows"])
    _write_output_json(repo,ROOT_PATH+"/paper_data/resource_and_qc.json",resource_table)
    _write_csv(paper/"resource_and_qc.csv",outputs["resource_rows"])
    _write_or_verify(root/"overall_descriptive_summary.json",outputs["summary"])
    initial_access_path=(
        repo/ACCESS_PENDING_PATH if (repo/ACCESS_PENDING_PATH).is_file()
        else root/"control/outcome_access_initial.json"
    )
    access_initial,_=_strict_json(initial_access_path,"initial E1 outcome access")
    _write_or_verify(root/"control/outcome_access_initial.json",access_initial)
    access_terminal=_publish_access_terminal(repo,access_initial)
    provenance=_signed({
        "schema_version":1,"document_type":"stage_e_e1_provenance",
        "stage_id":"E1","source_files":access_initial["upstream_reads"],
        "pair_bindings":[row["result"]["pair_binding"] for row in outputs["attempts"].values()],
        "operations":["read sealed D10 fine anchor","float64 aligned 4x4 map aggregation","new native 16x16 patch interventions","five paired tie repeats","float64 AUC and image-level descriptive bootstrap"],
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()},
        "outcome_access":{"path":ACCESS_PATH,"sha256":_sha(canonical_json_bytes(access_terminal)+b"\n"),"canonical_digest":access_terminal["canonical_digest"]},
        "d9_d10_e0_unchanged":True,"d10_anchor_recomputed":False,
        "e2_e3_e4_stage_f_started":False,
    })
    _write_or_verify(root/"provenance.json",provenance)
    recipe=_signed({
        "schema_version":1,"document_type":"stage_e_figure_recipes",
        "numeric_source":"paper_data_only","renders_png":False,
        "figures":[{"figure_id":"e1_patch_scale_descriptive_future","table_path":"paper_data/patch_scale_sensitivity.json","x":"scale","y":"delta_fid","group":"variant_id","caption_contract":"Post-D10 descriptive sensitivity only; D10 fine anchor remains primary; no winner or new decision.","numeric_source":"paper_data_only","renders_png":False}]
    })
    _write_or_verify(root/"figure_recipes.json",recipe)
    false_audit=_signed({
        "schema_version":1,"document_type":"stage_e_e1_false_pass_audit","stage_id":"E1","status":"PASS",
        "checks":[{"attack":name,"result":"REJECTED","mechanism":{
            "e0_unverified":"terminal E0 source validation and exact archive ref/digest required",
            "d10_anchor_only":"1000 unique new-compute terminal units required",
            "map_aggregation_without_patch_responses":"each valid unit requires ten 21-point response curves",
            "fewer_than_500_images":"exact registry order 0..499 and 500 paired deltas required",
            "fewer_than_two_patch_methods":"exact two-method variant set required for every sample",
            "cell_3136_masquerades_as_patch_196":"unit count and ceil schedule fixed to 196 and full 16x16 masks",
            "unpaired_seeds":"cross-method seed and tie-priority equality validated per image direction repeat",
            "missing_curves_auc_pair_bindings_or_attempts":"source validator recomputes coordinates AUC hashes pair bindings and attempt controls",
            "failed_na_nonfinite_dropped":"all expected rows required and PASS requires 1000 VALID finite rows",
            "favorable_valid_subset_summary":"summary requires exactly all 500 paired values",
            "d10_primary_recomputed_or_modified":"anchor rows are exact bound copies and compute status is FORBIDDEN_RECOMPUTE",
            "protected_upstream_modified":"before and after D9 D10 E0 inventories plus Git diff gate",
            "budget_exceeded":"GPU CPU and retained-byte hard limits gate completion",
            "tests_missing_or_failed":"completion binds a PASS test report and required gates",
            "receipt_manifest_inventory_empty_or_placeholder":"nonempty manifests inventory and pending-to-terminal completion are source validated",
            "e1_pass_overauthorizes_later_stage":"completion next authorization is exactly E2_ONLY and all later started flags remain false",
        }[name]} for name in FALSE_PASS_ATTACKS],
        "mutation_test":"Each named condition is independently falsified by test_false_pass_guard_rejects_every_attack.",
        "pass_possible_without_all_conditions":False,
    })
    _write_or_verify(repo/REPORT_ROOT_PATH/"false_pass_audit.json",false_audit)
    validation=_signed({
        "schema_version":1,"document_type":"stage_e_e1_source_validation","stage_id":"E1","status":"PASS","validated_at":access_terminal["completed_at"],
        "checks":{"e0_d9_d10":"PASS","plan":"PASS_500_GROUPS_1000_UNITS","attempts":"PASS_1000_UNIQUE_TERMINAL","curves":"PASS_10000_CURVES_210000_POINTS","paired_seeds":"PASS","auc_delta_summary_recomputed":"PASS","d10_anchor_forbidden_recompute":"PASS","protected_upstream_unchanged":"PASS","no_render_artifacts":"PASS","stage_boundary":"PASS_E1_ONLY"},
        "counts":outputs["summary"]["counts"],"summary_canonical_digest":outputs["summary"]["canonical_digest"],"false_pass_audit_canonical_digest":false_audit["canonical_digest"],
    })
    _write_or_verify(repo/REPORT_ROOT_PATH/"validation_report.json",validation)
    test_report_path=repo/REPORT_ROOT_PATH/"test_results.json"
    if not test_report_path.is_file():
        raise StageEE1Error("test report must exist before finalization")
    tests,_=_strict_json(test_report_path,"E1 test report")
    _validate_signed(tests,"E1 test report")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageEE1Error("E1 tests are not PASS")
    manifest_files=[]
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json","whole_root_inventory.json","status.json"}:
            manifest_files.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":_sha_file(path)})
    if not manifest_files:
        raise StageEE1Error("payload manifest cannot be empty")
    manifest=_signed({"schema_version":1,"document_type":"stage_e_e1_payload_manifest","immutable":True,"artifacts":manifest_files,"payload_count":len(manifest_files)})
    _write_or_verify(root/"payload_manifest.json",manifest)
    retained_pre_status=_inventory(root)["total_bytes"]
    existing_status_path=root/"status.json"
    if existing_status_path.is_file():
        existing_status,_=_strict_json(existing_status_path,"existing E1 status")
        _validate_signed(existing_status,"existing E1 status")
        cost=existing_status["actual_cost"]
    else:
        cost={**actual,"cpu_hours":actual["cpu_seconds"]/3600.0,"retained_bytes_before_status_inventory_completion":retained_pre_status,"gpu_hours_max":2.5,"cpu_hours_max":8.0,"retained_bytes_max":4294967296,"status":"PASS_COST"}
    inventory=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/e1_completion.pending.json","completion/e1_completion.json"))
    inventory_doc=_signed({"schema_version":1,"document_type":"stage_e_e1_whole_root_inventory","root":ROOT_PATH,"scope":"all regular non-hidden E1 formal files except this self-referential inventory, status, and completion receipts",**inventory})
    _write_or_verify(root/"whole_root_inventory.json",inventory_doc)
    status=_signed({
        "schema_version":1,"document_type":"stage_e_e1_status","stage_id":"E1","status":"PASS","complete":True,"terminal":True,"immutable":True,
        "package_completeness":"PASS","workflow_integrity":"PASS","scientific_validity":"PASS","cost_status":"PASS_COST",
        "d10_decision_unchanged":True,"c10_history_unchanged":True,"d10_primary_recomputed":False,
        "e2_authorized":True,"e2_started":False,"e3_started":False,"e4_started":False,"stage_f_started":False,
        "actual_cost":cost,"whole_root_inventory":{"path":"whole_root_inventory.json","file_sha256":_sha_file(root/"whole_root_inventory.json"),"canonical_digest":inventory_doc["canonical_digest"],"file_count":inventory["file_count"],"total_bytes":inventory["total_bytes"],"scope":inventory_doc["scope"]},
    })
    _write_or_verify(root/"status.json",status)
    completion=_signed({
        "schema_version":1,"document_type":"stage_e_terminal_completion","stage_id":"E1","status":"PASS",
        "started_at":access_initial["e1_started_at"],"completed_at":_now(),
        "predecessor":{"path":E0_COMPLETION_PATH,"file_sha256":_sha_file(repo/E0_COMPLETION_PATH),"canonical_digest":E0_COMPLETION_DIGEST},
        "config_canonical_digest":config["canonical_digest"],"run_spec_canonical_digest":_strict_json(repo/RUN_SPEC_PATH,"E1 run spec")[0]["canonical_digest"],
        "expected_coordinates":[{"variant_id":v,"image_count":500} for v in VARIANTS],"observed_coordinates":[{"variant_id":v,"image_count":500} for v in VARIANTS],
        "variant_statuses":[{"variant_id":v,"status":"PASS","reason":"read-only D10 anchor" if v==VARIANTS[0] else "500 valid native-patch response units"} for v in VARIANTS],
        "package_completeness":"PASS","workflow_integrity":"PASS","scientific_validity":"PASS","cost_status":"PASS_COST",
        "source_validation":{"path":f"{REPORT_ROOT_PATH}/validation_report.json","file_sha256":_sha_file(repo/REPORT_ROOT_PATH/"validation_report.json"),"canonical_digest":validation["canonical_digest"],"test_report":_record(repo,REPORT_ROOT_PATH+"/test_results.json"),"false_pass_audit":_record(repo,REPORT_ROOT_PATH+"/false_pass_audit.json"),"payload_manifest":_record(repo,ROOT_PATH+"/payload_manifest.json"),"status":_record(repo,ROOT_PATH+"/status.json"),"whole_root_inventory":_record(repo,ROOT_PATH+"/whole_root_inventory.json")},
        "upstream_unchanged":True,"actual_cost":cost,"next_authorization":"E2_ONLY",
    })
    schema,_=_strict_json(repo/"config/stage_e_stage_completion.schema.json","Stage E completion schema")
    errors=list(Draft202012Validator(schema).iter_errors(completion))
    if errors:
        raise StageEE1Error(f"E1 completion schema failure: {errors[0].message}")
    pending=repo/COMPLETION_PENDING_PATH
    terminal=repo/COMPLETION_PATH
    temporary=pending.with_name(".e1-completion-terminal.tmp")
    _write_new_json(temporary,completion)
    os.replace(temporary,pending)
    observed,_=_strict_json(pending,"terminal pending completion")
    if observed != completion:
        raise StageEE1Error("completion changed before atomic publication")
    os.replace(pending,terminal)
    final_inventory=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/e1_completion.pending.json","completion/e1_completion.json"))
    if final_inventory != inventory:
        raise StageEE1Error("E1 root changed during terminal publication")
    return {"status":"PASS","completion":COMPLETION_PATH,"whole_root_inventory_digest":inventory_doc["canonical_digest"],"scientific_summary":outputs["summary"],"actual_cost":cost,"upstream_after":after}


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve()
    completion,_=_strict_json(repo/COMPLETION_PATH,"E1 completion")
    _validate_signed(completion,"E1 completion")
    schema,_=_strict_json(repo/"config/stage_e_stage_completion.schema.json","Stage E completion schema")
    errors=list(Draft202012Validator(schema).iter_errors(completion))
    if errors:
        raise StageEE1Error(f"E1 completion schema failure: {errors[0].message}")
    if (
        completion.get("status") != "PASS"
        or completion.get("next_authorization") != "E2_ONLY"
        or completion.get("upstream_unchanged") is not True
        or completion.get("predecessor",{}).get("canonical_digest") != E0_COMPLETION_DIGEST
    ):
        raise StageEE1Error("E1 completion terminal gate failed")
    expected_coordinates=[{"variant_id":variant,"image_count":500} for variant in VARIANTS]
    run_spec,_=_strict_json(repo/RUN_SPEC_PATH,"E1 run spec")
    _validate_signed(run_spec,"E1 run spec")
    if (
        completion.get("expected_coordinates") != expected_coordinates
        or completion.get("observed_coordinates") != expected_coordinates
        or completion.get("config_canonical_digest") != EXPECTED_E1_CONFIG_DIGEST
        or completion.get("run_spec_canonical_digest") != run_spec["canonical_digest"]
    ):
        raise StageEE1Error("E1 completion coordinate/config/run-spec binding drifted")
    outputs=validate_outputs(repo)
    summary,_=_strict_json(repo/ROOT_PATH/"overall_descriptive_summary.json","E1 summary")
    if outputs["summary"] != summary:
        raise StageEE1Error("E1 sealed summary is not source-recomputable")
    patch_table,_=_strict_json(repo/ROOT_PATH/"paper_data/patch_scale_sensitivity.json","E1 patch table")
    resource_table,_=_strict_json(repo/ROOT_PATH/"paper_data/resource_and_qc.json","E1 resource table")
    if patch_table != _table("patch_scale_sensitivity",outputs["rows"]) or resource_table != _table("resource_and_qc",outputs["resource_rows"]):
        raise StageEE1Error("E1 sealed output tables are not source-recomputable")
    for table,name in ((patch_table,"patch_scale_sensitivity"),(resource_table,"resource_and_qc")):
        schema_out,_=_strict_json(repo/OUTPUT_SCHEMA_PATH,"E1 output schema")
        if list(Draft202012Validator(schema_out).iter_errors(table)):
            raise StageEE1Error(f"E1 {name} output schema failed")
        csv_path=repo/ROOT_PATH/"paper_data"/f"{name}.csv"
        with csv_path.open(newline="",encoding="utf-8") as stream:
            csv_rows=list(csv.DictReader(stream))
        expected=[{key:_csv_scalar(value) for key,value in row.items()} for row in table["rows"]]
        if csv_rows != expected:
            raise StageEE1Error(f"E1 {name} CSV/JSON parity failed")
    status,_=_strict_json(repo/ROOT_PATH/"status.json","E1 status")
    inventory,_=_strict_json(repo/ROOT_PATH/"whole_root_inventory.json","E1 inventory")
    for value,name in ((status,"status"),(inventory,"inventory")):
        _validate_signed(value,f"E1 {name}")
    recomputed=_inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","status.json","completion/e1_completion.pending.json","completion/e1_completion.json"))
    if (
        any(inventory[key] != recomputed[key] for key in ("files","file_count","total_bytes"))
        or canonical_sha256({"files":inventory["files"]}) != recomputed["canonical_digest"]
    ):
        raise StageEE1Error("E1 whole-root inventory drifted")
    if (
        status["whole_root_inventory"]["canonical_digest"] != inventory["canonical_digest"]
        or status.get("status") != "PASS"
        or status.get("e2_authorized") is not True
        or any(status.get(key) is not False for key in ("e2_started","e3_started","e4_started","stage_f_started"))
    ):
        raise StageEE1Error("E1 status/inventory or stage-boundary drifted")
    if (
        status.get("actual_cost") != completion.get("actual_cost")
        or budget_reasons(completion["actual_cost"],inventory["total_bytes"])
    ):
        raise StageEE1Error("E1 completion cost binding or budget gate failed")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in (repo/ROOT_PATH).rglob("*") if path.is_file()):
        raise StageEE1Error("E1 contains a forbidden render artifact")
    for key,path in completion["source_validation"].items():
        if key in {"path","file_sha256","canonical_digest"}:
            continue
        expected=path
        observed=_record(repo,expected["path"])
        if expected != observed:
            raise StageEE1Error(f"completion {key} binding drifted")
    validation_record={key:completion["source_validation"][key] for key in ("path","file_sha256","canonical_digest")}
    observed_validation=_record(repo,validation_record["path"])
    if (
        validation_record["file_sha256"] != observed_validation["sha256"]
        or validation_record["canonical_digest"] != observed_validation["canonical_digest"]
    ):
        raise StageEE1Error("completion validation-report binding drifted")
    false_audit,_=_strict_json(repo/REPORT_ROOT_PATH/"false_pass_audit.json","E1 false-PASS audit")
    _validate_signed(false_audit,"E1 false-PASS audit")
    if (
        false_audit.get("status") != "PASS"
        or {row.get("attack") for row in false_audit.get("checks",[])} != set(FALSE_PASS_ATTACKS)
        or any(row.get("result") != "REJECTED" for row in false_audit.get("checks",[]))
    ):
        raise StageEE1Error("E1 false-PASS audit failed")
    tests,_=_strict_json(repo/REPORT_ROOT_PATH/"test_results.json","E1 required test report")
    _validate_signed(tests,"E1 required test report")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageEE1Error("E1 required tests failed")
    before,_=_strict_json(repo/UPSTREAM_BEFORE_PATH,"E1 protected upstream before")
    after=_protected_after(repo,before)
    recorded_after,_=_strict_json(repo/REPORT_ROOT_PATH/"protected_upstream_after.json","E1 protected upstream after")
    for name in ("d9","d10","e0"):
        if after[name] != recorded_after[name]:
            raise StageEE1Error(f"E1 recorded protected {name} after snapshot drifted")
    access,_=_strict_json(repo/ACCESS_PATH,"E1 terminal outcome access")
    _validate_signed(access,"E1 terminal outcome access")
    if (
        access.get("new_e1_outcome_accessed") is not True
        or access.get("scope_changed_after_outcome_access") is not False
        or access.get("d10_primary_recomputed") is not False
        or any(access.get(key) is not False for key in ("e2_started","e3_started","e4_started","stage_f_started"))
    ):
        raise StageEE1Error("E1 outcome-access terminal contract failed")
    return {"status":"PASS","completion_canonical_digest":completion["canonical_digest"],"row_count":len(outputs["rows"]),"new_compute_rows":len(outputs["patch_rows"]),"whole_root_inventory_digest":inventory["canonical_digest"],"summary":summary,"actual_cost":completion["actual_cost"],"next_authorization":completion["next_authorization"]}


def _main() -> int:
    parser=argparse.ArgumentParser(prog="python -m vitnns.stage_e_e1")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("verify-preconditions","initialize","validate-initialized","validate-outputs","finalize","validate-completion"):
        child=sub.add_parser(name); child.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    run=sub.add_parser("run-shard"); run.add_argument("--repo-root",type=Path,default=REPO_ROOT); run.add_argument("--shard-id",type=int,required=True); run.add_argument("--shard-count",type=int,required=True); run.add_argument("--device",default="cuda:0"); run.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.command=="verify-preconditions": value=verify_preconditions(args.repo_root)
    elif args.command=="initialize": value=initialize(args.repo_root)
    elif args.command=="validate-initialized": value=validate_initialized(args.repo_root)
    elif args.command=="run-shard": value=run_shard(args.repo_root,shard_id=args.shard_id,shard_count=args.shard_count,device=args.device,limit=args.limit)
    elif args.command=="validate-outputs":
        output=validate_outputs(args.repo_root); value={"status":"PASS","row_count":len(output["rows"]),"new_compute_row_count":len(output["patch_rows"]),"actual_cost":output["actual_cost"]}
    elif args.command=="finalize": value=finalize(args.repo_root)
    else: value=validate_completion(args.repo_root)
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(_main())
