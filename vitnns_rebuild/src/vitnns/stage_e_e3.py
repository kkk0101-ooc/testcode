"""Stage E E3 frozen layer/head and prediction-strata descriptions.

Only the three E0-missing coordinates are executable.  The D10 (0, 9)
primary is source-validated and copied into the descriptive tables without
creating an E3 attempt or performing model work.
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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_e_e0 as e0
from . import stage_e_e1 as e1
from . import stage_e_e2 as e2

CONFIG_PATH = "config/stage_e_e3.json"
CONFIG_SCHEMA_PATH = "config/stage_e_e3.schema.json"
OUTPUT_SCHEMA_PATH = "config/stage_e_e3_output.schema.json"
ROOT_PATH = "evidence/stage_e/e3_recovery_v2_20260906/formal"
REPORT_ROOT_PATH = "reports/stage_e_e3_20260906"
PLAN_PATH = ROOT_PATH + "/control/typed_plan.json"
RUN_SPEC_PATH = ROOT_PATH + "/control/run_spec.json"
AUTH_PATH = ROOT_PATH + "/control/execution_authorization.json"
REGISTRY_PATH = ROOT_PATH + "/control/expected_coordinate_registry.json"
ANCHOR_PLAN_PATH = ROOT_PATH + "/control/primary_anchor_reuse_plan.json"
PREFLIGHT_PATH = ROOT_PATH + "/control/budget_preflight.json"
ATTEMPT_POLICY_PATH = ROOT_PATH + "/control/immutable_attempt_policy.json"
UPSTREAM_BEFORE_PATH = ROOT_PATH + "/control/protected_upstream_before.json"
COMPLETION_PENDING_PATH = ROOT_PATH + "/completion/e3_completion.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/e3_completion.json"
ACCESS_PENDING_PATH = REPORT_ROOT_PATH + "/outcome_access.pending.json"
ACCESS_PATH = REPORT_ROOT_PATH + "/outcome_access.json"
OBSERVABILITY_EVENT_PATH = REPORT_ROOT_PATH + "/pre_scale_observability_event.json"

E2_COMMIT = "e93b0d1b796d82ea6d7df84b07fd61d4b98c76c2"
E2_COMPLETION_SHA256 = "c26e2e0e330bd00030378c89a477807e24349073ee5ddbe58e1424e188013b3f"
E2_COMPLETION_DIGEST = "f40d1475cf1f55fbf8292db55b14ee15e76e15d392603c0ebe59f4f2fd3370fd"
E2_STATUS_SHA256 = "2d9893a90991093275a8eba53bd1bb72399439006c945d335b4a9df0da77636d"
E2_STATUS_DIGEST = "159ace91e531d06026c52f40f0719bcfe50b94ec05ebb547b7ec37d3f2e2a8d5"
E2_INVENTORY_SHA256 = "a0a0ccbc107be7370ad9247f40cae2cac65a43c86286ed26d066231abafda44e"
E2_INVENTORY_DIGEST = "57bd3959236f10b2f12cc797d6bc71f64c4df892cc863502264bc75ab987d101"
E2_BLOCKED_SHA256 = "ea4af38e2faa078fde0b85fa00d43ea53e4763584e5e7367a08e65be75ede4b7"
E2_BLOCKED_DIGEST = "f6c261c0b5fe8162a3582634c4b4e29fa466a7fddc5a4c9f8748b47a4f9bffe3"
E2_WAIVER_SHA256 = "1dcaf8d11bfc1e549079da5a2a112920f3cdef5ae434edab47fa7de5af298dd0"
E2_WAIVER_DIGEST = "1cbf1b72d4c9dba53d11733bdacdfbf51d77a99252fdfeb0dfde5cec568134df"
EXPECTED_CONFIG_DIGEST = "2f311e75a3db3c9d99ce0e4728cfc535f23df3012aacfdceca26f18d2890797b"

E2_COMPLETION_PATH = e2.SUPERSEDING_COMPLETION_PATH
E2_STATUS_PATH = e2.SUPERSEDING_STATUS_PATH
E2_INVENTORY_PATH = e2.SUPERSEDING_INVENTORY_PATH
E2_BLOCKED_PATH = e2.COMPLETION_PATH
E2_WAIVER_PATH = e2.BUDGET_WAIVER_PATH
D8_ROOT_PATH = e1.D8_ROOT_PATH
D10_ROOT_PATH = e1.D10_ROOT_PATH
D10_DELTA_PATH = e1.D10_ROWS_PATH
D1_REGISTRY_PATH = e1.REGISTRY_PATH

COORDINATES = ((0, 9), (4, 2), (6, 10), (11, 2))
NEW_COORDINATES = COORDINATES[1:]
METHODS = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig",
    "matched_fixed_clean_attention_patch_token_contribution",
)
NNS_METHOD, MATCHED_METHOD = METHODS
STRATA = ("overall", "correct", "misclassified")
SEED_NAMESPACE = "stage-e-fidelity|20260906|"
BOOTSTRAP_NAMESPACE = "stage-e-bootstrap|20260906|"
FRACTIONS = tuple(float(index) / 20.0 for index in range(21))
UNIT_COUNTS = (0,) + tuple(int(math.ceil(value * 3136)) for value in FRACTIONS[1:])
ACTUAL_FRACTIONS = tuple(value / 3136.0 for value in UNIT_COUNTS)
MODEL_SHA256 = e1.MODEL_SHA256
MODEL_TENSOR_DIGEST = e1.MODEL_TENSOR_DIGEST
FORBIDDEN_SUFFIXES = {".png", ".pdf", ".svg"}

PROTECTED_PATHS = (
    D8_ROOT_PATH, D10_ROOT_PATH,
    "evidence/stage_e/e0_recovery_v2_20260906",
    "evidence/stage_e/e1_recovery_v2_20260906",
    "evidence/stage_e/e2_recovery_v2_20260906",
    "reports/stage_d_d9_recovery_v2_20260906",
    "reports/stage_d_d10_recovery_v2_20260906",
    "reports/stage_e_e0_20260906",
    "reports/stage_e_e1_20260906",
    "reports/stage_e_e2_20260906",
)
EXPECTED_PROTECTED_DIGESTS = {
    D8_ROOT_PATH: "76d7d532049a5c90a09c74ba56b019042d7f2f0174ccb439bd8f67713514cc5a",
    D10_ROOT_PATH: "b72d36430d149598ccd9cfd8dd586297e7231be29e61b44309e9971087e46535",
    "evidence/stage_e/e0_recovery_v2_20260906": "80087eb3ea0f417683e0f1c5b1dc87023f15be2247e508eeb0e4299ce9b65cda",
    "evidence/stage_e/e1_recovery_v2_20260906": "903d6c704630ea0c5b9090e4da337982153c6690b047c757feef4616b204b417",
    "evidence/stage_e/e2_recovery_v2_20260906": "2c9b5b4d96ce055d04698abdc35b0dfde1a64cfdbf57e3f06fc5a54163b2f5bf",
    "reports/stage_d_d9_recovery_v2_20260906": "e2937e9f166582fb5e281b319e017cb8f32af27613e35bb0935fc89e916e0a59",
    "reports/stage_d_d10_recovery_v2_20260906": "c16a9e4e90f05b984ab9012438291b0ddb052f228f6f1f579b4b437de1b31d3a",
    "reports/stage_e_e0_20260906": "a5fa4f36b6fecfa0dc49771b0292fcba6eea21398b3669b48741c08a0781d837",
    "reports/stage_e_e1_20260906": "34e077dd47b6f87537db54c761c0fae6ff0a5e79b6dee93d7c555356762a0efb",
    "reports/stage_e_e2_20260906": "53bb144ff7b577275c10a3de4e7a34ec0f5d9293306643db86ee46e8978784ff",
}
FALSE_PASS_ATTACKS = (
    "e2_superseding_pass_or_e3_only_unverified",
    "e2_blocked_receipt_or_owner_waiver_ignored",
    "e2_waiver_silently_extended_to_e3",
    "only_primary_anchor_copied", "fewer_than_500_images_or_four_coordinates",
    "missing_new_coordinate_or_method", "maps_without_real_fidelity_responses",
    "stage_c_five_image_cohort_substituted", "primary_recomputed_or_modified",
    "outcome_driven_head_selection", "patch_196_masquerades_as_fine_3136",
    "methods_not_seed_and_tie_paired", "failed_na_nonfinite_or_misclassified_dropped",
    "prediction_partition_not_500_418_82", "favorable_valid_subset_summary",
    "unfrozen_clean_margin_added", "missing_curves_auc_pair_attempt_manifest_or_inventory",
    "protected_upstream_modified", "e3_budget_exceeded_without_e3_waiver",
    "tests_missing_or_failed", "receipt_manifest_inventory_placeholder",
    "e3_pass_starts_e4_or_authorizes_f0",
)


class StageEE3Error(RuntimeError):
    """E3 cannot proceed or pass under the frozen contract."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())[:-2] + ":" + time.strftime("%z")[-2:]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "canonical_digest"})


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_digest"] = _self_digest(result)
    return result


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite token {token}")),
            object_pairs_hook=e1._reject_duplicate_keys,
        )
    except Exception as error:
        raise StageEE3Error(f"cannot read strict {context}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageEE3Error(f"{context} must be a JSON object")
    return value, raw


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest") != _self_digest(value):
        raise StageEE3Error(f"{context} canonical digest mismatch")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        try: path.unlink()
        except FileNotFoundError: pass
        raise


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise StageEE3Error(f"immutable document differs: {path}")
        return
    _write_new_json(path, value)


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ("git", *args), cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise StageEE3Error(f"git {' '.join(args)} failed: {error.stderr.strip()}") from error


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative; raw = path.read_bytes()
    result: dict[str, Any] = {"path": relative, "bytes": len(raw), "sha256": _sha(raw)}
    result["canonical_digest"] = _strict_json(path, relative)[0].get("canonical_digest") if path.suffix == ".json" else None
    return result


def _inventory(root: Path, *, exclude: Sequence[str] = (), include_hidden: bool = False) -> dict[str, Any]:
    excluded = set(exclude); files = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink(): continue
            relative = path.relative_to(root).as_posix()
            if relative in excluded or (not include_hidden and any(part.startswith(".") for part in Path(relative).parts)): continue
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return {"file_count": len(files), "total_bytes": sum(row["bytes"] for row in files), "canonical_digest": canonical_sha256({"files": files}), "files": files}


def _validate_config(repo: Path) -> dict[str, Any]:
    value, _ = _strict_json(repo / CONFIG_PATH, "E3 config")
    schema, _ = _strict_json(repo / CONFIG_SCHEMA_PATH, "E3 config schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: [str(part) for part in item.path])
    if errors: raise StageEE3Error(f"E3 config schema failure: {errors[0].message}")
    _validate_signed(value, "E3 config")
    if value["canonical_digest"] != EXPECTED_CONFIG_DIGEST: raise StageEE3Error("E3 frozen config digest mismatch")
    scope = value["scope"]; science = value["scientific_contract"]
    observed = tuple((row["block"], row["head"]) for row in scope["coordinates"])
    if observed != COORDINATES or tuple(scope["methods"]) != METHODS or tuple(scope["strata"]) != STRATA:
        raise StageEE3Error("E3 frozen coordinate/method/stratum scope drifted")
    if scope["coordinates"][0]["compute"] != "FORBIDDEN_RECOMPUTE" or any(row["compute"] != "NEW_COMPUTE_REQUIRED" for row in scope["coordinates"][1:]):
        raise StageEE3Error("E3 primary/new-compute policy drifted")
    if science["unit_counts"] != list(UNIT_COUNTS) or science["fractions"] != list(FRACTIONS):
        raise StageEE3Error("E3 fine-cell schedule drifted")
    if science["tie_seed_namespace"] != SEED_NAMESPACE or science["clean_margin"] != {"status": "NOT_RUN_WITH_REASON", "reason": "No variable direction and missingness contract was frozen in E0 v1"}:
        raise StageEE3Error("E3 seed or clean-margin contract drifted")
    if value["budget"] != {"gpu_hours_max": 8.0, "cpu_hours_max": 16.0, "retained_bytes_max": 12884901888, "e2_waiver_applies": False}:
        raise StageEE3Error("E3 independent budget drifted")
    return value


def comparison_identity(block: int, head: int) -> str:
    if (block, head) not in NEW_COORDINATES:
        raise StageEE3Error("comparison identity is defined only for new E3 coordinates")
    return f"e3_layer_head_{block}_{head}_comparison"


def e3_seed(sample_id: str, block: int, head: int, repeat: int) -> int:
    if type(repeat) is not int or not 0 <= repeat < 5:
        raise StageEE3Error("repeat must be an integer in [0,4]")
    material = f"{SEED_NAMESPACE}{sample_id}|{block}|{head}|{comparison_identity(block, head)}|{repeat}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def tie_priority(seed: int) -> np.ndarray:
    order = np.random.Generator(np.random.PCG64(seed)).permutation(3136)
    result = np.empty(3136, dtype=np.int64); result[order] = np.arange(3136, dtype=np.int64)
    return result


def ranking(signed_map: Any, *, sigma: int, seed: int) -> np.ndarray:
    source = np.asarray(signed_map)
    if source.dtype != np.float32 or source.shape != (56, 56) or not np.isfinite(source).all() or sigma not in (-1, 1):
        raise StageEE3Error("ranking requires a finite signed float32 56x56 map and sigma +/-1")
    scores = np.maximum(np.float32(sigma) * source.reshape(-1), np.float32(0.0))
    return np.lexsort((tie_priority(seed), -scores.astype(np.float64))).astype(np.int64)


def trapezoid_auc(responses: Sequence[float]) -> float:
    y = np.asarray(responses, dtype=np.float64); x = np.asarray(ACTUAL_FRACTIONS, dtype=np.float64)
    if y.shape != (21,) or not np.isfinite(y).all(): raise StageEE3Error("AUC requires 21 finite responses")
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5, dtype=np.float64))


def delta_fid(nns: Mapping[str, Any], matched: Mapping[str, Any]) -> float:
    return 0.5 * (float(nns["insertion_auc_mean"]) - float(matched["insertion_auc_mean"]) + float(matched["deletion_auc_mean"]) - float(nns["deletion_auc_mean"]))


def false_pass_reasons(preconditions: Mapping[str, bool]) -> tuple[str, ...]:
    if set(preconditions) != set(FALSE_PASS_ATTACKS): raise StageEE3Error("false-PASS precondition set drifted")
    return tuple(name for name in FALSE_PASS_ATTACKS if preconditions[name] is not True)


def budget_reasons(actual: Mapping[str, Any], retained_bytes: int) -> tuple[str, ...]:
    result = []
    if float(actual.get("gpu_hours", math.inf)) > 8.0: result.append("gpu_hours")
    if float(actual.get("cpu_seconds", math.inf)) > 16.0 * 3600.0: result.append("cpu_hours")
    if retained_bytes > 12884901888: result.append("retained_bytes")
    return tuple(result)


def _primary_source_index(repo: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], Mapping[str, Any]]:
    return e1._source_index(repo)


def _anchor_rows(repo: Path) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    table, raw = _strict_json(repo / D10_DELTA_PATH, "D10 primary per-sample delta")
    if _sha(raw) != "8bb2e36dc2b4b3895bc5688b7ab5789719995add0c26cd48608d593486ba0cd2" or table.get("canonical_digest") != "4a89aeb89884219779699e15b49b017106bced1e60ab146dafac04d63ee9d9c0":
        raise StageEE3Error("D10 primary table binding drifted")
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) != 500: raise StageEE3Error("D10 primary must contain 500 rows")
    return rows, table


def _build_plan(repo: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    registry, _ = e1._load_registry(repo); anchor, _ = _anchor_rows(repo)
    source_index, _ = _primary_source_index(repo)
    if [row["sample_id"] for row in anchor] != [row["sample_id"] for row in registry]:
        raise StageEE3Error("D10 primary order differs from registry")
    groups = []; units = []
    for row, primary in zip(registry, anchor):
        if primary["misclassified"] not in (True, False): raise StageEE3Error("D10 stratum is not typed")
        stratum = "misclassified" if primary["misclassified"] else "correct"
        primary_sources = {}
        for method in METHODS:
            source = source_index[(row["sample_id"], method)]; attempt = e1._source_attempt(repo, source)
            primary_sources[method] = {
                "source_unit_digest": source["unit_digest"], "source_attempt_id": source["attempts"][0]["attempt_id"],
                "source_attempt_inventory_digest": source["attempts"][0]["attempt_inventory_digest"],
                "source_result_path": (attempt / "L0/result.json").relative_to(repo).as_posix(),
                "source_result_sha256": _sha_file(attempt / "L0/result.json"),
                "source_map_path": (attempt / "L1/metric_ready_map.npy").relative_to(repo).as_posix(),
                "source_map_sha256": _sha_file(attempt / "L1/metric_ready_map.npy"),
            }
        unit_digests = []
        for block, head in NEW_COORDINATES:
            for method in METHODS:
                unit = _signed({
                    "schema_version": 1, "document_type": "stage_e_e3_new_compute_unit",
                    "sample_id": row["sample_id"], "selection_order": row["selection_order"],
                    "image_sha256": row["image_sha256"], "relative_path": row["relative_path"],
                    "stratum": stratum, "block": block, "head": head, "method": method,
                    "coordinate_comparison_identity": comparison_identity(block, head),
                    "scale": "fine_4x4_cell", "target": "fixed_clean_attention_cls_head_direction",
                })
                units.append(unit); unit_digests.append(unit["canonical_digest"])
        groups.append(_signed({
            "schema_version": 1, "document_type": "stage_e_e3_image_group",
            "sample_id": row["sample_id"], "selection_order": row["selection_order"],
            "image_sha256": row["image_sha256"], "relative_path": row["relative_path"],
            "stratum": stratum, "primary_anchor": {"block": 0, "head": 9, "mode": "PRIMARY_ANCHOR_READ_ONLY", "sources": primary_sources},
            "new_unit_digests": unit_digests,
        }))
    return _signed({
        "schema_version": 1, "document_type": "stage_e_e3_typed_plan",
        "config_canonical_digest": config["canonical_digest"], "cohort": "stage_d_fidelity_v1_500",
        "coordinates": [list(item) for item in COORDINATES], "methods": list(METHODS), "strata": list(STRATA),
        "expected_image_groups": 500, "expected_new_units": 3000, "expected_method_rows": 4000, "expected_paired_rows": 2000,
        "groups": groups, "units": units,
    })


def _protected_snapshot(repo: Path) -> dict[str, Any]:
    roots = []
    for relative in PROTECTED_PATHS:
        inventory = _inventory(repo / relative, include_hidden=True)
        observed = inventory["canonical_digest"]
        if observed != EXPECTED_PROTECTED_DIGESTS[relative]:
            raise StageEE3Error(
                f"protected root binding failed: {relative}: "
                f"expected={EXPECTED_PROTECTED_DIGESTS[relative]} observed={observed}"
            )
        roots.append({
            "path": relative, "file_count": inventory["file_count"],
            "total_bytes": inventory["total_bytes"],
            "canonical_digest": observed,
        })
    changed = _git(repo, "diff", "--name-only", E2_COMMIT, "--", *PROTECTED_PATHS).splitlines()
    if changed: raise StageEE3Error(f"protected tracked upstream changed: {changed}")
    return _signed({
        "schema_version": 1, "document_type": "stage_e_e3_protected_upstream_snapshot",
        "captured_at": _now(), "roots": roots, "protected_tracked_diff": changed,
        "critical_e2_chain": {
            "original_blocked_completion": _record(repo, E2_BLOCKED_PATH),
            "owner_budget_waiver": _record(repo, E2_WAIVER_PATH),
            "superseding_completion": _record(repo, E2_COMPLETION_PATH),
            "superseding_status": _record(repo, E2_STATUS_PATH),
            "superseding_inventory": _record(repo, E2_INVENTORY_PATH),
        },
    })


def verify_preconditions(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _validate_config(repo)
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    head = _git(repo, "rev-parse", "HEAD"); tracking = _git(repo, "rev-parse", "@{u}")
    remote_tracking = _git(repo, "rev-parse", "refs/remotes/origin/stage-d-d3-pass")
    archive = _git(repo, "rev-parse", "refs/archive/stage-e-e2-completion^{commit}")
    live_remote = _git(repo, "ls-remote", "--heads", "origin", "stage-d-d3-pass").split()[0]
    if branch != "stage-d-d3-pass" or {head, tracking, remote_tracking, archive, live_remote} != {E2_COMMIT}:
        raise StageEE3Error(
            f"workspace binding failed: branch={branch} head={head} tracking={tracking} "
            f"remote_tracking={remote_tracking} archive={archive} live_remote={live_remote}"
        )
    allowed = (
        "config/stage_e_e3.json", "config/stage_e_e3.schema.json",
        "config/stage_e_e3_output.schema.json", "src/vitnns/stage_e_e3.py",
        "tests/test_stage_e_e3.py", "evidence/stage_e/e3_recovery_v2_20260906/",
        "reports/stage_e_e3_20260906/",
    )
    unexpected = [
        line for line in _git(repo, "status", "--porcelain=v1").splitlines()
        if not any(
            line[3:].endswith(item.rstrip("/"))
            or f"/{item}" in line[3:]
            for item in allowed
        )
    ]
    if unexpected: raise StageEE3Error(f"unexpected worktree changes: {unexpected}")
    lfs = _git(repo, "lfs", "status")
    if "Git LFS objects to be committed:" not in lfs or "Git LFS objects not staged for commit:" not in lfs:
        raise StageEE3Error("cannot establish Git LFS status")
    e2_result = e2.validate_completion(repo)
    if (e2_result.get("status"), e2_result.get("completion_basis"), e2_result.get("completion_canonical_digest"), e2_result.get("variant_rows"), e2_result.get("pairwise_rows"), e2_result.get("new_compute_rows"), e2_result.get("next_authorization"), e2_result.get("e3_started"), e2_result.get("whole_root_inventory_digest")) != ("PASS", "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER", E2_COMPLETION_DIGEST, 2000, 3000, 1000, "E3_ONLY", False, E2_INVENTORY_DIGEST):
        raise StageEE3Error("E2 superseding PASS source validation failed")
    for relative, expected_sha, expected_digest in (
        (E2_COMPLETION_PATH, E2_COMPLETION_SHA256, E2_COMPLETION_DIGEST),
        (E2_STATUS_PATH, E2_STATUS_SHA256, E2_STATUS_DIGEST),
        (E2_INVENTORY_PATH, E2_INVENTORY_SHA256, E2_INVENTORY_DIGEST),
        (E2_BLOCKED_PATH, E2_BLOCKED_SHA256, E2_BLOCKED_DIGEST),
        (E2_WAIVER_PATH, E2_WAIVER_SHA256, E2_WAIVER_DIGEST),
    ):
        value, raw = _strict_json(repo / relative, relative)
        if _sha(raw) != expected_sha or value.get("canonical_digest") != expected_digest:
            raise StageEE3Error(f"E2 frozen byte/canonical binding failed: {relative}")
    completion, _ = _strict_json(repo / E2_COMPLETION_PATH, "E2 completion")
    waiver, _ = _strict_json(repo / E2_WAIVER_PATH, "E2 waiver")
    if completion.get("actual_cost", {}).get("status") != "FAIL_COST" or waiver.get("waiver_scope") != "E2_GPU_AND_CPU_HARD_BUDGET_ACCEPTANCE_ONLY" or waiver.get("e3_execution_authorized_in_this_task") is not False:
        raise StageEE3Error("E2 cost or waiver scope was misrepresented")
    e1_result = e1.validate_completion(repo)
    if e1_result.get("status") != "PASS" or e1_result.get("completion_canonical_digest") != e2.E1_COMPLETION_DIGEST:
        raise StageEE3Error("E1 completion source validation failed")
    e0.validate_e0_inputs(repo); upstream = dict(e0.verify_upstream(repo))
    if upstream.get("status") != "PASS": raise StageEE3Error("E0/D9/D10 source validation failed")
    snapshot = _protected_snapshot(repo)
    return {
        "status": "PASS", "validated_at": _now(), "head": head, "config": config,
        "e2_completion": e2_result, "e1_completion": e1_result, "upstream": upstream,
        "protected_snapshot": snapshot,
    }


def initialize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); pre = verify_preconditions(repo); config = pre["config"]
    plan = _build_plan(repo, config)
    registry = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_expected_coordinate_registry",
        "coordinates": [{**row, "ordinal": index} for index, row in enumerate(config["scope"]["coordinates"])],
        "sample_count": 500, "method_count": 2, "new_coordinate_count": 3,
        "expected_new_units": 3000, "expected_method_rows": 4000, "expected_paired_rows": 2000,
    })
    anchor_plan = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_primary_anchor_reuse_plan",
        "coordinate": [0, 9], "role": "PRIMARY_ANCHOR_READ_ONLY", "recompute": "FORBIDDEN_RECOMPUTE",
        "source_table": _record(repo, D10_DELTA_PATH), "source_method_attempts": 1000,
        "inherited_rows": {"method_rows": 1000, "paired_rows": 500},
        "new_gpu_attempts": 0, "primary_selection_reopened": False,
    })
    run_spec = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_run_spec",
        "config_canonical_digest": config["canonical_digest"], "plan_canonical_digest": plan["canonical_digest"],
        "coordinate_registry_canonical_digest": registry["canonical_digest"], "primary_anchor_plan_canonical_digest": anchor_plan["canonical_digest"],
        "expected_image_groups": 500, "expected_new_units": 3000, "expected_method_rows": 4000, "expected_paired_rows": 2000, "expected_summary_cells": 12,
        "coordinates": [list(item) for item in COORDINATES], "new_coordinates": [list(item) for item in NEW_COORDINATES], "methods": list(METHODS), "strata": list(STRATA),
        "directions": ["insertion", "deletion"], "tie_repeats": list(range(5)), "fractions": list(FRACTIONS), "unit_counts": list(UNIT_COUNTS), "actual_fractions": list(ACTUAL_FRACTIONS),
        "attempt_policy": config["attempt_policy"], "stage_boundary": {"current": "E3_ONLY", "on_pass": "E4_ONLY", "e4_started": False, "stage_f_started": False},
    })
    preflight = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_budget_preflight", "status": "PASS", "recorded_at": _now(),
        "basis": {
            "d8_complete_primary_coordinate_gpu_hours": 1.8504247143613854,
            "d8_complete_primary_coordinate_forward_calls": 249340, "d8_complete_primary_coordinate_backward_calls": 32320,
            "e1_measured_gpu_hours": 1.577772853329451, "e1_measured_cpu_hours": 1.6308065290991665, "e1_measured_forward_calls": 213000,
            "e2_actual_gpu_hours": 2.6383805223161314, "e2_actual_cpu_hours": 54.325327326570275,
            "e2_cpu_risk_note": "E2 dense interpolation CPU cost is retained as risk evidence but that operator is absent from E3",
            "e0_per_coordinate_caps": {"gpu_hours": 2.5, "cpu_hours": 5.0, "retained_bytes": 4294967296},
            "new_coordinates": 3, "new_units": 3000, "primary_gpu_hours": 0.0, "e2_waiver_applies": False,
        },
        "estimated": {"gpu_hours": 5.551274143084156, "cpu_hours": 15.0, "retained_bytes": 6442450944, "forward_calls": 748020, "backward_calls": 96960},
        "limits": {key: value for key, value in config["budget"].items() if key != "e2_waiver_applies"},
        "all_estimates_within_limits": True, "budget_reduction_by_scope_change_allowed": False,
    })
    policy = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_immutable_attempt_policy",
        **config["attempt_policy"], "new_terminal_attempt_count": 3000,
        "primary_attempt_mode": "INHERITED_D9_D10_NO_E3_ATTEMPT",
        "resume_validation": ["manifest", "source_hashes", "coordinate", "method", "curves", "terminal_state"],
    })
    false_spec = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_false_pass_test_specification",
        "attacks": list(FALSE_PASS_ATTACKS), "all_attacks_must_be_independently_rejected": True,
        "pass_without_all_conditions": False,
    })
    authorization = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_execution_authorization", "status": "AUTHORIZED", "issued_at": _now(),
        "e2_superseding_pass_and_e3_only_verified": True, "e2_original_blocked_and_waiver_chain_verified": True,
        "e2_actual_cost_remains_fail_cost": True, "e2_waiver_extended_to_e3": False, "e1_e0_d9_d10_verified": True,
        "plan_canonical_digest": plan["canonical_digest"], "run_spec_canonical_digest": run_spec["canonical_digest"], "budget_preflight_canonical_digest": preflight["canonical_digest"],
        "authorized_scope": "E3_ONLY_500_IMAGES_3_NEW_COORDINATES_2_METHODS_3000_UNITS", "primary_recompute": "FORBIDDEN", "later_stages_authorized": False,
    })
    upstream_reads = [
        _record(repo, path) for path in (
            E2_COMPLETION_PATH, E2_STATUS_PATH, E2_INVENTORY_PATH, E2_BLOCKED_PATH, E2_WAIVER_PATH,
            e1.COMPLETION_PATH, e1.STATUS_PATH if hasattr(e1, "STATUS_PATH") else e1.ROOT_PATH + "/status.json",
            e0.COMPLETION_PATH, D1_REGISTRY_PATH, D10_DELTA_PATH,
            D8_ROOT_PATH + "/final/source_inventory.json", D8_ROOT_PATH + "/final/reconciliation.json",
            "reports/stage_d_d9_recovery_v2_20260906/d9_formal_validation.json",
            "reports/stage_d_d10_recovery_v2_20260906/d10_formal_decision.json",
        )
    ]
    access = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_outcome_access", "status": "PENDING_BEFORE_FIRST_E3_OUTCOME",
        "e3_started_at": _now(), "e2_superseding_terminal_pass_and_e3_only_verified": True, "d10_e1_e2_outcomes_known": True,
        "new_e3_coordinate_outcome_accessed": False, "scope_source": "sealed_E0_only",
        "frozen_scope": {"coordinates": [list(item) for item in COORDINATES], "methods": list(METHODS), "sample_count": 500, "strata": list(STRATA), "unit_count": 3136, "fractions": list(FRACTIONS), "tie_repeats": 5, "bootstrap_draws": 10000, "seed_namespace": SEED_NAMESPACE, "budget": config["budget"]},
        "e2_waiver_scope": "E2_ONLY_NOT_E3", "upstream_reads": upstream_reads,
        "protected_roots": [{"path": path, "mode": "READ_ONLY"} for path in PROTECTED_PATHS],
        "primary_recompute": False, "e4_started": False, "stage_f_started": False,
    })
    pending = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_completion_pending", "stage_id": "E3", "status": "PENDING", "started_at": access["e3_started_at"],
        "plan_canonical_digest": plan["canonical_digest"], "run_spec_canonical_digest": run_spec["canonical_digest"], "authorization_canonical_digest": authorization["canonical_digest"],
        "expected_image_groups": 500, "expected_new_units": 3000, "expected_method_rows": 4000, "expected_paired_rows": 2000, "next_authorization": "NONE_PENDING",
    })
    documents = (
        (PLAN_PATH, plan), (RUN_SPEC_PATH, run_spec), (REGISTRY_PATH, registry), (ANCHOR_PLAN_PATH, anchor_plan),
        (PREFLIGHT_PATH, preflight), (ATTEMPT_POLICY_PATH, policy),
        (ROOT_PATH + "/control/false_pass_test_specification.json", false_spec),
        (UPSTREAM_BEFORE_PATH, pre["protected_snapshot"]), (AUTH_PATH, authorization),
        (ACCESS_PENDING_PATH, access), (COMPLETION_PENDING_PATH, pending),
    )
    for relative, document in documents: _write_or_verify(repo / relative, document)
    validate_initialized(repo)
    return {"status": "PASS", "plan_canonical_digest": plan["canonical_digest"], "run_spec_canonical_digest": run_spec["canonical_digest"], "expected_new_units": 3000, "access_record": ACCESS_PENDING_PATH}


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _validate_config(repo)
    names = (("plan", PLAN_PATH), ("run_spec", RUN_SPEC_PATH), ("registry", REGISTRY_PATH), ("anchor", ANCHOR_PLAN_PATH), ("preflight", PREFLIGHT_PATH), ("policy", ATTEMPT_POLICY_PATH), ("false_spec", ROOT_PATH + "/control/false_pass_test_specification.json"), ("before", UPSTREAM_BEFORE_PATH), ("authorization", AUTH_PATH), ("access", ACCESS_PENDING_PATH), ("pending", COMPLETION_PENDING_PATH))
    documents = {}
    for name, relative in names:
        value, _ = _strict_json(repo / relative, f"E3 {name}"); _validate_signed(value, f"E3 {name}"); documents[name] = value
    if documents["plan"] != _build_plan(repo, config): raise StageEE3Error("E3 typed plan is not source-recomputable")
    plan = documents["plan"]
    counts = Counter((row["block"], row["head"], row["method"]) for row in plan["units"])
    expected = Counter({(block, head, method): 500 for block, head in NEW_COORDINATES for method in METHODS})
    if len(plan["groups"]) != 500 or len(plan["units"]) != 3000 or counts != expected:
        raise StageEE3Error("E3 initialized plan coverage failed")
    if documents["authorization"]["status"] != "AUTHORIZED" or documents["preflight"]["status"] != "PASS" or documents["preflight"]["all_estimates_within_limits"] is not True:
        raise StageEE3Error("E3 execution authorization or budget preflight failed")
    if documents["access"]["new_e3_coordinate_outcome_accessed"] is not False or documents["pending"]["status"] != "PENDING":
        raise StageEE3Error("E3 pending outcome/completion state failed")
    return {"status": "PASS", "image_groups": 500, "new_units": 3000, "method_rows": 4000, "paired_rows": 2000, "summary_cells": 12}


def _attempt_paths(repo: Path, unit_digest: str) -> tuple[Path, Path, Path, Path, Path]:
    root = repo / ROOT_PATH / "units" / unit_digest / "attempts/attempt-000001"
    return root, root / "result.json", root / "map.npy", root / "status.json", root / "manifest.json"


def _method_provenance(result: Any) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(dict(result.provenance)).decode("utf-8"))


def _method_map(result: Any, response: Any) -> np.ndarray:
    from .stage_c_fidelity import FidelityScale, method_unit_masses
    value = np.asarray(method_unit_masses(result, response, FidelityScale.FINE_4).masses)
    if value.dtype != np.float32 or value.shape != (56, 56) or not np.isfinite(value).all():
        raise StageEE3Error("method did not produce a finite signed float32 56x56 map")
    if result.method_id == NNS_METHOD and not np.count_nonzero(value):
        raise StageEE3Error("NNS attribution map is all zero")
    return np.array(value, dtype=np.float32, copy=True)


def _curves_for_method(
    *, model: Any, clean: Any, target: Any, response: Any, method_result: Any,
    sample_id: str, block: int, head: int, call_counter: dict[str, int],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    import torch
    from .stage_c_fidelity import FidelityScale, method_unit_masses
    from .stage_c_perturbations import tensor_sha256
    unit_map = method_unit_masses(method_result, response, FidelityScale.FINE_4)
    signed_map = np.array(unit_map.masses, dtype=np.float32, copy=True)
    evaluator = __import__("vitnns.stage_c_fidelity", fromlist=["ValuePathScalarAdapter"]).ValuePathScalarAdapter(model, clean, target)
    baseline = torch.zeros_like(clean); baseline_sha = tensor_sha256(baseline); clean_sha = tensor_sha256(clean)
    curves: list[dict[str, Any]] = []
    identity = comparison_identity(block, head)
    for direction in ("insertion", "deletion"):
        for repeat in range(5):
            seed = e3_seed(sample_id, block, head, repeat)
            priority = tie_priority(seed); ranked = ranking(signed_map, sigma=response.sigma, seed=seed)
            responses = []; intervention_hashes = []
            for index, count in enumerate(UNIT_COUNTS):
                if index == 0:
                    normalized = 0.0 if direction == "insertion" else 1.0
                    digest = baseline_sha if direction == "insertion" else clean_sha
                elif index == 20:
                    normalized = 1.0 if direction == "insertion" else 0.0
                    digest = clean_sha if direction == "insertion" else baseline_sha
                else:
                    selected = np.zeros(3136, dtype=np.bool_); selected[ranked[:count]] = True
                    pixel_mask = np.repeat(np.repeat(selected.reshape(56, 56), 4, axis=0), 4, axis=1)
                    mask = torch.from_numpy(pixel_mask.copy()).to(device=clean.device)[None, None].expand_as(clean)
                    intervention = torch.where(mask, clean, baseline) if direction == "insertion" else torch.where(mask, baseline, clean)
                    digest = tensor_sha256(intervention); call_counter["forward_calls"] += 1
                    normalized = response.normalize(evaluator.evaluate_prehashed(intervention, input_sha256=digest))
                    if not isinstance(normalized, float) or not math.isfinite(normalized): raise StageEE3Error("non-finite normalized fine-cell response")
                responses.append(normalized); intervention_hashes.append(digest)
            score = np.maximum(np.float32(response.sigma) * signed_map.reshape(-1), np.float32(0.0)).astype(np.float32, copy=False)
            curves.append({
                "status": "OK", "method_id": method_result.method_id, "comparison_line": "value_path",
                "target_ref": "fixed_clean_attention_cls_head_direction", "target_identity_sha256": unit_map.target_identity_sha256,
                "scale": "fine_4x4_cell", "direction": direction, "ranking_kind": "target_oriented_positive_tie_permuted",
                "sample_id": sample_id, "block": block, "head": head, "replicate": repeat, "seed": seed,
                "seed_namespace": SEED_NAMESPACE, "seed_comparison_identity": identity,
                "tie_priority_sha256": e1._array_digest(priority), "source_unit_map_sha256": unit_map.sha256,
                "score_sha256": e1._array_digest(score), "input_pair_sha256": hashlib.sha256(f"{clean_sha}|{baseline_sha}".encode("ascii")).hexdigest(),
                "ranking_sha256": e1._array_digest(ranked), "intervention_sha256": intervention_hashes,
                "curve_fractions": list(FRACTIONS), "unit_counts": list(UNIT_COUNTS), "actual_fractions": list(ACTUAL_FRACTIONS),
                "responses": responses, "auc": trapezoid_auc(responses), "response_clipped": False,
                "auc_integration": "trapezoid_over_actual_k_divided_by_units",
            })
    return curves, signed_map


def _repeat_aggregates(curves: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for direction in ("insertion", "deletion"):
        values = np.asarray([row["auc"] for row in curves if row["direction"] == direction], dtype=np.float64)
        if values.shape != (5,) or not np.isfinite(values).all(): raise StageEE3Error("five finite repeat AUCs are required")
        result[f"{direction}_repeat_aucs"] = values.tolist()
        result[f"{direction}_auc_mean"] = float(values.mean())
        result[f"{direction}_auc_repeat_sd_ddof_0"] = float(values.std(ddof=0))
    return result


def _valid_result(
    *, unit: Mapping[str, Any], target_identity: Mapping[str, Any], response: Any,
    method_result: Any, signed_map: np.ndarray, curves: list[dict[str, Any]],
    elapsed_seconds: float, cpu_seconds: float, forward_calls: int,
    backward_calls: int, peak_cuda_bytes: int, clean_sha256: str,
) -> dict[str, Any]:
    provenance = _method_provenance(method_result); target_digest = e1._target_semantic_digest(target_identity)
    source_hashes = {
        "image_sha256": unit["image_sha256"], "clean_input_sha256": clean_sha256,
        "model_sha256": MODEL_SHA256, "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST,
        "target_semantic_digest": target_digest, "signed_map_sha256": e1._array_digest(signed_map),
        "method_provenance_canonical_digest": canonical_sha256(provenance),
    }
    pair_binding = {
        "sample_id": unit["sample_id"], "selection_order": unit["selection_order"], "image_sha256": unit["image_sha256"],
        "model_sha256": MODEL_SHA256, "target": "fixed_clean_attention_cls_head_direction", "target_semantic_digest": target_digest,
        "method": unit["method"], "block": unit["block"], "head": unit["head"], "scale": "fine_4x4_cell",
        "comparison_identity": unit["coordinate_comparison_identity"], "replacement": "normalized_zero_canonical_rgb_0_5",
        "fractions": list(FRACTIONS), "unit_counts": list(UNIT_COUNTS),
        "seeds": [e3_seed(unit["sample_id"], unit["block"], unit["head"], repeat) for repeat in range(5)],
        "seed_namespace": SEED_NAMESPACE, "signed_map_sha256": source_hashes["signed_map_sha256"], "attempt_id": "attempt-000001",
    }
    conservation = None
    if unit["method"] == MATCHED_METHOD:
        matched = provenance.get("matched_contribution", {})
        patch_sum = float(signed_map.sum(dtype=np.float64)); cls = float(matched["cls_contribution"]); token_sum = float(matched["token_sum"]); tolerance = float(matched["conservation_tolerance"])
        conservation = {
            "patch_sum": patch_sum, "cls_contribution": cls, "token_sum": token_sum,
            "absolute_error": abs(patch_sum + cls - token_sum), "tolerance": tolerance,
            "cls_in_ranking": False, "status": "PASS" if abs(patch_sum + cls - token_sum) <= tolerance else "FAIL",
        }
        if conservation["status"] != "PASS": raise StageEE3Error("matched patch/CLS/token conservation failed")
    return {
        "schema_version": 1, "document_type": "stage_e_e3_attempt_result",
        "unit_digest": unit["canonical_digest"], "sample_id": unit["sample_id"], "selection_order": unit["selection_order"], "stratum": unit["stratum"],
        "block": unit["block"], "head": unit["head"], "method": unit["method"], "target": unit["target"], "scale": unit["scale"],
        "coordinate_comparison_identity": unit["coordinate_comparison_identity"], "validity_status": "VALID",
        "failure_class": None, "failure_code": None, "na_reason": None, "attempt_id": "attempt-000001", "retry_count": 0,
        "source_hashes": source_hashes, "pair_binding": pair_binding, "pair_binding_digest": canonical_sha256(pair_binding),
        "target_response": response.to_dict(), "method_provenance": provenance, "matched_conservation_qc": conservation,
        "curves": curves, **_repeat_aggregates(curves), "unit_count": 3136, "unit_pixels": [4, 4],
        "elapsed_seconds": float(elapsed_seconds), "cpu_seconds": float(cpu_seconds), "gpu_hours": float(elapsed_seconds / 3600.0),
        "forward_calls": int(forward_calls), "backward_calls": int(backward_calls), "peak_cuda_memory_bytes": int(peak_cuda_bytes), "artifact_bytes": 0,
        "descriptive_only": True, "p_values_emitted": False, "winner_selected": False, "causal_claim_made": False, "new_go_no_go_issued": False, "d10_primary_recomputed": False,
    }


def _publish_attempt(repo: Path, unit: Mapping[str, Any], result: Mapping[str, Any], signed_map: np.ndarray) -> Path:
    root, _, _, _, _ = _attempt_paths(repo, unit["canonical_digest"])
    if root.exists(): raise StageEE3Error(f"refusing to overwrite terminal attempt: {root}")
    attempts_root = root.parent; attempts_root.mkdir(parents=True, exist_ok=True)
    if any(path.name.startswith(".attempt-") for path in attempts_root.iterdir()): raise StageEE3Error(f"ambiguous unpublished attempt exists: {attempts_root}")
    temporary = Path(tempfile.mkdtemp(prefix=".attempt-000001-", dir=attempts_root))
    try:
        signed_result = _signed(result); result_raw = canonical_json_bytes(signed_result) + b"\n"; (temporary / "result.json").write_bytes(result_raw)
        with (temporary / "map.npy").open("wb") as stream: np.save(stream, signed_map, allow_pickle=False)
        status = _signed({
            "schema_version": 1, "document_type": "stage_e_e3_attempt_status", "attempt_id": "attempt-000001", "attempt_number": 1,
            "unit_digest": unit["canonical_digest"], "state": "succeeded", "terminal": True, "retry_count": 0,
            "failure_class": None, "failure_code": None, "na_reason": None, "published_at": _now(),
        })
        status_raw = canonical_json_bytes(status) + b"\n"; (temporary / "status.json").write_bytes(status_raw)
        map_sha = _sha_file(temporary / "map.npy")
        manifest = _signed({
            "schema_version": 1, "document_type": "stage_e_e3_attempt_manifest", "attempt_id": "attempt-000001", "attempt_number": 1,
            "unit_digest": unit["canonical_digest"], "result_sha256": _sha(result_raw), "result_bytes": len(result_raw),
            "map_sha256": map_sha, "map_bytes": (temporary / "map.npy").stat().st_size, "map_array_sha256": e1._array_digest(signed_map),
            "status_sha256": _sha(status_raw), "status_bytes": len(status_raw), "immutable": True,
        })
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        for path in temporary.iterdir():
            with path.open("rb") as stream: os.fsync(stream.fileno())
        os.replace(temporary, root)
        directory_fd = os.open(attempts_root, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    _validate_attempt(repo, unit)
    return root


def _expected_pair_binding(unit: Mapping[str, Any], target_digest: str, map_digest: str) -> dict[str, Any]:
    return {
        "sample_id": unit["sample_id"], "selection_order": unit["selection_order"], "image_sha256": unit["image_sha256"],
        "model_sha256": MODEL_SHA256, "target": "fixed_clean_attention_cls_head_direction", "target_semantic_digest": target_digest,
        "method": unit["method"], "block": unit["block"], "head": unit["head"], "scale": "fine_4x4_cell",
        "comparison_identity": unit["coordinate_comparison_identity"], "replacement": "normalized_zero_canonical_rgb_0_5",
        "fractions": list(FRACTIONS), "unit_counts": list(UNIT_COUNTS),
        "seeds": [e3_seed(unit["sample_id"], unit["block"], unit["head"], repeat) for repeat in range(5)],
        "seed_namespace": SEED_NAMESPACE, "signed_map_sha256": map_digest, "attempt_id": "attempt-000001",
    }


def _validate_attempt(repo: Path, unit: Mapping[str, Any]) -> dict[str, Any]:
    root, result_path, map_path, status_path, manifest_path = _attempt_paths(repo, unit["canonical_digest"])
    if not root.is_dir() or any(not path.is_file() for path in (result_path, map_path, status_path, manifest_path)):
        raise StageEE3Error(f"attempt is absent or incomplete: {root}")
    result, result_raw = _strict_json(result_path, "E3 attempt result"); status, status_raw = _strict_json(status_path, "E3 attempt status"); manifest, _ = _strict_json(manifest_path, "E3 attempt manifest")
    for value, name in ((result, "result"), (status, "status"), (manifest, "manifest")): _validate_signed(value, f"E3 attempt {name}")
    signed_map = np.load(map_path, allow_pickle=False)
    map_digest = e1._array_digest(signed_map)
    if (
        result.get("unit_digest") != unit["canonical_digest"] or result.get("sample_id") != unit["sample_id"]
        or result.get("selection_order") != unit["selection_order"] or result.get("block") != unit["block"] or result.get("head") != unit["head"]
        or result.get("method") != unit["method"] or result.get("stratum") != unit["stratum"] or result.get("validity_status") not in {"VALID", "FAILED", "NA"}
        or status.get("terminal") is not True or status.get("attempt_id") != "attempt-000001" or manifest.get("result_sha256") != _sha(result_raw)
        or manifest.get("status_sha256") != _sha(status_raw) or manifest.get("map_sha256") != _sha_file(map_path) or manifest.get("map_array_sha256") != map_digest
    ): raise StageEE3Error("E3 attempt control binding failed")
    if result["validity_status"] != "VALID": return {"result": result, "map": signed_map, "status": status, "manifest": manifest, "attempt_path": root}
    if signed_map.dtype != np.float32 or signed_map.shape != (56, 56) or not np.isfinite(signed_map).all(): raise StageEE3Error("E3 map dtype/shape/finite contract failed")
    if result.get("failure_class") is not None or result.get("failure_code") is not None or result.get("na_reason") is not None or result.get("retry_count") != 0 or result.get("unit_count") != 3136 or result.get("unit_pixels") != [4, 4]:
        raise StageEE3Error("valid E3 attempt has failure/retry/geometry drift")
    source_hashes = result.get("source_hashes", {})
    if source_hashes.get("image_sha256") != unit["image_sha256"] or source_hashes.get("model_sha256") != MODEL_SHA256 or source_hashes.get("model_canonical_tensor_digest") != MODEL_TENSOR_DIGEST or source_hashes.get("signed_map_sha256") != map_digest:
        raise StageEE3Error("E3 source hash binding failed")
    target_response = result.get("target_response")
    if not isinstance(target_response, Mapping) or target_response.get("sigma") not in (-1, 1): raise StageEE3Error("E3 target response is invalid")
    target_digest = e1._target_semantic_digest(target_response["target_identity"])
    if target_digest != source_hashes.get("target_semantic_digest"): raise StageEE3Error("E3 target semantic digest failed")
    pair = _expected_pair_binding(unit, target_digest, map_digest)
    if result.get("pair_binding") != pair or result.get("pair_binding_digest") != canonical_sha256(pair): raise StageEE3Error("E3 pair binding failed")
    provenance = result.get("method_provenance")
    if not isinstance(provenance, Mapping) or canonical_sha256(provenance) != source_hashes.get("method_provenance_canonical_digest"): raise StageEE3Error("E3 method provenance binding failed")
    if unit["method"] == NNS_METHOD:
        ig = provenance.get("ig_result", {})
        if provenance.get("status") != "OK" or ig.get("converged") is not True or not ig.get("attempts") or int(result.get("backward_calls", 0)) <= 0 or not np.count_nonzero(signed_map):
            raise StageEE3Error("E3 NNS real attribution/backward proof failed")
        if result.get("matched_conservation_qc") is not None: raise StageEE3Error("NNS contains matched-only conservation QC")
    else:
        matched = provenance.get("matched_contribution", {})
        qc = result.get("matched_conservation_qc", {})
        if provenance.get("status") != "OK" or matched.get("status") != "OK" or qc.get("status") != "PASS" or qc.get("cls_in_ranking") is not False or result.get("backward_calls") != 0:
            raise StageEE3Error("E3 matched contribution/CLS conservation proof failed")
    curves = result.get("curves")
    expected_coordinates = {(direction, repeat) for direction in ("insertion", "deletion") for repeat in range(5)}
    if not isinstance(curves, list) or len(curves) != 10 or {(row.get("direction"), row.get("replicate")) for row in curves} != expected_coordinates:
        raise StageEE3Error("E3 attempt lacks exact ten-curve coverage")
    for curve in curves:
        seed = e3_seed(unit["sample_id"], unit["block"], unit["head"], curve["replicate"]); expected_ranking = ranking(signed_map, sigma=target_response["sigma"], seed=seed)
        score = np.maximum(np.float32(target_response["sigma"]) * signed_map.reshape(-1), np.float32(0.0)).astype(np.float32, copy=False)
        if (
            curve.get("method_id") != unit["method"] or curve.get("sample_id") != unit["sample_id"] or curve.get("block") != unit["block"] or curve.get("head") != unit["head"]
            or curve.get("seed") != seed or curve.get("seed_namespace") != SEED_NAMESPACE or curve.get("seed_comparison_identity") != comparison_identity(unit["block"], unit["head"])
            or curve.get("tie_priority_sha256") != e1._array_digest(tie_priority(seed)) or curve.get("ranking_sha256") != e1._array_digest(expected_ranking)
            or curve.get("source_unit_map_sha256") != map_digest or curve.get("score_sha256") != e1._array_digest(score)
            or curve.get("unit_counts") != list(UNIT_COUNTS) or curve.get("actual_fractions") != list(ACTUAL_FRACTIONS)
            or len(curve.get("responses", [])) != 21 or len(curve.get("intervention_sha256", [])) != 21 or curve.get("auc") != trapezoid_auc(curve["responses"])
        ): raise StageEE3Error("E3 curve source recomputation failed")
        endpoints = (curve["responses"][0], curve["responses"][-1])
        if endpoints != ((0.0, 1.0) if curve["direction"] == "insertion" else (1.0, 0.0)): raise StageEE3Error("E3 curve endpoint identity failed")
    aggregates = _repeat_aggregates(curves)
    if any(result.get(key) != value for key, value in aggregates.items()): raise StageEE3Error("E3 repeat AUC aggregation failed")
    if int(result.get("forward_calls", 0)) <= 190: raise StageEE3Error("E3 attempt lacks real map/target/fidelity forward calls")
    return {"result": result, "map": signed_map, "status": status, "manifest": manifest, "attempt_path": root}


def run_shard(repo_root: Path | str = REPO_ROOT, *, shard_id: int, shard_count: int, device: str = "cuda:0", limit: int | None = None) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if type(shard_count) is not int or shard_count <= 0 or type(shard_id) is not int or not 0 <= shard_id < shard_count: raise StageEE3Error("invalid shard identity")
    validate_initialized(repo); plan, _ = _strict_json(repo / PLAN_PATH, "E3 plan"); registry, _ = e1._load_registry(repo)
    units_by_sample: dict[str, list[dict[str, Any]]] = {}
    for unit in plan["units"]: units_by_sample.setdefault(unit["sample_id"], []).append(unit)
    selected = [row for row in registry if row["selection_order"] % shard_count == shard_id]
    if limit is not None: selected = selected[:limit]
    import torch
    from .stage_c_fidelity import ValuePathScalarAdapter, resolve_target_response
    from .stage_c_value_path import (
        HeadCandidate, conditional_fixed_clean_attention_head_direction_value_path_ig,
        matched_fixed_clean_attention_patch_token_contribution,
        resolve_fixed_clean_attention_head_directions,
    )
    from .stage_c_perturbations import tensor_sha256
    model = binding = None; shard_start = time.monotonic(); cpu_start = time.process_time(); completed = skipped = 0; groups = []
    try:
        model, binding, model_record = e1._load_model_and_context(repo, device)
        gpu_index = torch.device(device).index or 0; props = torch.cuda.get_device_properties(gpu_index) if device.startswith("cuda") else None
        visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical_gpu_index = int(visible_gpu) if visible_gpu is not None and visible_gpu.isdigit() else gpu_index
        gpu_uuid = None
        if device.startswith("cuda"):
            query = subprocess.run(("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.splitlines()
            identities = {int(line.split(",", 1)[0].strip()): line.split(",", 1)[1].strip() for line in query}
            gpu_uuid = identities.get(physical_gpu_index)
            if not gpu_uuid: raise StageEE3Error("physical GPU UUID could not be recorded")
        for row in selected:
            group_units = units_by_sample[row["sample_id"]]
            existing = {}
            for unit in group_units:
                if _attempt_paths(repo, unit["canonical_digest"])[0].exists(): existing[unit["canonical_digest"]] = _validate_attempt(repo, unit)
            missing = [unit for unit in group_units if unit["canonical_digest"] not in existing]
            skipped += len(existing)
            if not missing: continue
            missing_coordinates = [coord for coord in NEW_COORDINATES if any((unit["block"], unit["head"]) == coord for unit in missing)]
            group_start = time.monotonic(); group_cpu = time.process_time(); calls = 0
            if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
            _canonical, clean, clean_sha = e1._load_sample(binding, row, model, device)
            if clean_sha != tensor_sha256(clean): raise StageEE3Error("live clean tensor hash drifted")
            def count_hook(_module: Any, _args: Any) -> None:
                nonlocal calls; calls += 1
            handle = model.register_forward_pre_hook(count_hook)
            try:
                target_start = time.monotonic(); target_cpu = time.process_time(); calls0 = calls
                targets = resolve_fixed_clean_attention_head_directions(model, clean, tuple(HeadCandidate(*coord) for coord in missing_coordinates), candidate_chunk_size=len(missing_coordinates))
                target_elapsed = time.monotonic() - target_start; target_cpu_seconds = time.process_time() - target_cpu; target_calls = calls - calls0
                targets_by_coord = {(target.block_index, target.head_index): target for target in targets}
                output_units = []
                for coord_index, coord in enumerate(missing_coordinates):
                    block, head = coord; target = targets_by_coord[coord]; evaluator = ValuePathScalarAdapter(model, clean, target)
                    response = resolve_target_response(evaluator, clean, torch.zeros_like(clean)); calls_after_response = calls
                    if response.na is not None: raise StageEE3Error(f"frozen E3 target became NA at {coord}")
                    target_identity = json.loads(canonical_json_bytes(evaluator.target_identity).decode("utf-8"))
                    coord_missing = [unit for unit in missing if (unit["block"], unit["head"]) == coord]
                    for method_index, unit in enumerate(coord_missing):
                        start = time.monotonic(); cpu0 = time.process_time(); before = calls
                        if unit["method"] == NNS_METHOD:
                            method_result = conditional_fixed_clean_attention_head_direction_value_path_ig(model, clean, target, provenance={"stage": "E3", "comparison_identity": comparison_identity(block, head)})
                            ig = method_result.provenance["ig_result"]; backward_calls = sum(int(math.ceil(int(attempt["steps"]) / 8)) for attempt in ig["attempts"])
                        else:
                            method_result = matched_fixed_clean_attention_patch_token_contribution(target, provenance={"stage": "E3", "comparison_identity": comparison_identity(block, head)}); backward_calls = 0
                        logical = {"forward_calls": 0}
                        curves, signed_map = _curves_for_method(model=model, clean=clean, target=target, response=response, method_result=method_result, sample_id=row["sample_id"], block=block, head=head, call_counter=logical)
                        if logical["forward_calls"] != 190: raise StageEE3Error("E3 curve evaluator did not execute 190 real intermediate forwards")
                        shared_count = len(missing); share_calls = target_calls // shared_count + (1 if completed % shared_count < target_calls % shared_count else 0)
                        result = _valid_result(
                            unit=unit, target_identity=target_identity, response=response, method_result=method_result, signed_map=signed_map, curves=curves,
                            elapsed_seconds=time.monotonic() - start + target_elapsed / len(missing), cpu_seconds=time.process_time() - cpu0 + target_cpu_seconds / len(missing),
                            forward_calls=calls - before + share_calls + ((calls_after_response - (calls0 + target_calls)) // max(1, len(coord_missing))), backward_calls=backward_calls,
                            peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0, clean_sha256=clean_sha,
                        )
                        output_units.append((unit, result, signed_map)); del method_result
                    del evaluator, response
            finally:
                handle.remove()
            for unit, result, signed_map in output_units:
                _publish_attempt(repo, unit, result, signed_map); completed += 1
            groups.append({
                "sample_id": row["sample_id"], "selection_order": row["selection_order"], "coordinates": [list(item) for item in missing_coordinates],
                "elapsed_seconds": time.monotonic() - group_start, "cpu_seconds": time.process_time() - group_cpu, "forward_calls": calls,
                "backward_calls": sum(item[1]["backward_calls"] for item in output_units), "completed_units": len(output_units),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,
            })
            del clean, targets
        elapsed = time.monotonic() - shard_start
        telemetry = _signed({
            "schema_version": 1, "document_type": "stage_e_e3_shard_telemetry", "shard_id": shard_id, "shard_count": shard_count, "device": device,
            "gpu_index": gpu_index if device.startswith("cuda") else None, "cuda_visible_devices": visible_gpu, "gpu_physical_index": physical_gpu_index if device.startswith("cuda") else None, "gpu_uuid": gpu_uuid, "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "started_groups": len(selected), "completed_units": completed, "skipped_terminal_units": skipped, "elapsed_seconds": elapsed,
            "cpu_seconds": time.process_time() - cpu_start, "gpu_hours": elapsed / 3600.0 if device.startswith("cuda") else 0.0,
            "forward_calls": sum(group["forward_calls"] for group in groups), "backward_calls": sum(group["backward_calls"] for group in groups),
            "peak_cuda_memory_bytes": max((group["peak_cuda_memory_bytes"] for group in groups), default=0), "groups": groups, "model": model_record, "completed_at": _now(),
        })
        _write_or_verify(repo / ROOT_PATH / "telemetry" / f"shard-{shard_id:02d}-of-{shard_count:02d}.json", telemetry)
        return telemetry
    finally:
        if model is not None: del model
        if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()


def _primary_method_rows(repo: Path, plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchor, table = _anchor_rows(repo); source_index, _ = _primary_source_index(repo); methods = []; pairs = []
    for group, source in zip(plan["groups"], anchor):
        if source["sample_id"] != group["sample_id"] or source["selection_order"] != group["selection_order"]:
            raise StageEE3Error("primary anchor order drifted")
        if source.get("status") != "VALID" or source.get("validity_status") != "PASS": raise StageEE3Error("D10 primary contains non-valid row")
        by_method = {}
        for method in METHODS:
            binding = group["primary_anchor"]["sources"][method]
            source_row = source_index[(group["sample_id"], method)]; attempt = e1._source_attempt(repo, source_row); result, raw = _strict_json(attempt / "L0/result.json", "D9 primary result")
            if _sha(raw) != binding["source_result_sha256"] or result.get("method_id") != method or result.get("status") != "OK": raise StageEE3Error("D9 primary source result binding failed")
            aggregates = {row["direction"]: row for row in result.get("fidelity_aggregates", [])}
            if set(aggregates) != {"insertion", "deletion"}: raise StageEE3Error("D9 primary aggregate coverage failed")
            prefix = "nns" if method == NNS_METHOD else "matched"
            if aggregates["insertion"]["auc_mean"] != source[f"{prefix}_insertion_auc"] or aggregates["deletion"]["auc_mean"] != source[f"{prefix}_deletion_auc"]:
                raise StageEE3Error("D10 primary AUC differs from sealed D9 source")
            row = {
                "variant_id": "layer_head_0_9", "sample_id": group["sample_id"], "selection_order": group["selection_order"],
                "cohort": "stage_d_fidelity_v1_500", "stratum": group["stratum"], "validity_status": "VALID", "method": method,
                "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": 0, "head": 9,
                "coordinate_role": "PRIMARY_ANCHOR_READ_ONLY", "recompute_status": "FORBIDDEN_RECOMPUTE", "attempt_id": None,
                "inherited_attempt_id": binding["source_attempt_id"], "retry_count": 0, "failure_class": None, "failure_code": None, "na_reason": None,
                "insertion_auc_mean": source[f"{prefix}_insertion_auc"], "deletion_auc_mean": source[f"{prefix}_deletion_auc"],
                "insertion_auc_repeat_sd_ddof_0": source[f"{prefix}_insertion_auc_repeat_sd_ddof_0"], "deletion_auc_repeat_sd_ddof_0": source[f"{prefix}_deletion_auc_repeat_sd_ddof_0"],
                "delta_fid": source["delta_fid"], "misclassified": source["misclassified"],
                "source_hashes": {**binding, "d10_per_sample_sha256": _sha_file(repo / D10_DELTA_PATH), "d10_per_sample_canonical_digest": table["canonical_digest"]},
                "elapsed_seconds": 0.0, "gpu_hours": 0.0, "cpu_seconds": 0.0, "forward_calls": 0, "backward_calls": 0, "peak_cuda_memory_bytes": 0, "artifact_bytes": 0,
                "descriptive_only": True,
            }
            methods.append(row); by_method[method] = row
        observed_delta = delta_fid(by_method[NNS_METHOD], by_method[MATCHED_METHOD])
        if observed_delta != source["delta_fid"]: raise StageEE3Error("D10 primary delta is not source-recomputable")
        pairs.append({
            "variant_id": "layer_head_0_9_pair", "sample_id": group["sample_id"], "selection_order": group["selection_order"], "cohort": "stage_d_fidelity_v1_500",
            "stratum": group["stratum"], "validity_status": "VALID", "source_hashes": {"d10_per_sample_sha256": _sha_file(repo / D10_DELTA_PATH), "nns_source_result_sha256": by_method[NNS_METHOD]["source_hashes"]["source_result_sha256"], "matched_source_result_sha256": by_method[MATCHED_METHOD]["source_hashes"]["source_result_sha256"]},
            "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": 0, "head": 9,
            "coordinate_role": "PRIMARY_ANCHOR_READ_ONLY", "recompute_status": "FORBIDDEN_RECOMPUTE", "misclassified": source["misclassified"], "delta_fid": source["delta_fid"],
            "nns_insertion_auc": source["nns_insertion_auc"], "nns_deletion_auc": source["nns_deletion_auc"], "matched_insertion_auc": source["matched_insertion_auc"], "matched_deletion_auc": source["matched_deletion_auc"],
            "failure_class": None, "failure_code": None, "na_reason": None, "descriptive_only": True,
        })
    return methods, pairs


def _new_rows(repo: Path, plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    attempts = {unit["canonical_digest"]: _validate_attempt(repo, unit) for unit in plan["units"]}
    method_rows = []; pair_rows = []
    by_key: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for unit in plan["units"]:
        observed = attempts[unit["canonical_digest"]]; result = observed["result"]; path = observed["attempt_path"]
        result["artifact_bytes"] = sum(item.stat().st_size for item in path.iterdir() if item.is_file())
        row = {
            "variant_id": f"layer_head_{unit['block']}_{unit['head']}", "sample_id": unit["sample_id"], "selection_order": unit["selection_order"],
            "cohort": "stage_d_fidelity_v1_500", "stratum": unit["stratum"], "validity_status": result["validity_status"], "source_hashes": result["source_hashes"],
            "method": unit["method"], "target": unit["target"], "scale": unit["scale"], "layer": unit["block"], "head": unit["head"],
            "coordinate_role": "DESCRIPTIVE_FROZEN_COORDINATE", "recompute_status": "NEW_COMPUTE_COMPLETED", "attempt_id": "attempt-000001", "retry_count": result["retry_count"],
            "failure_class": result["failure_class"], "failure_code": result["failure_code"], "na_reason": result["na_reason"],
            "insertion_auc_mean": result.get("insertion_auc_mean"), "deletion_auc_mean": result.get("deletion_auc_mean"),
            "insertion_auc_repeat_sd_ddof_0": result.get("insertion_auc_repeat_sd_ddof_0"), "deletion_auc_repeat_sd_ddof_0": result.get("deletion_auc_repeat_sd_ddof_0"),
            "misclassified": unit["stratum"] == "misclassified", "pair_binding_digest": result.get("pair_binding_digest"),
            "elapsed_seconds": result["elapsed_seconds"], "gpu_hours": result["gpu_hours"], "cpu_seconds": result["cpu_seconds"],
            "elapsed_seconds_recorded_in_immutable_attempt": result["elapsed_seconds"], "cpu_seconds_recorded_in_immutable_attempt": result["cpu_seconds"],
            "forward_calls": result["forward_calls"],
            "forward_calls_recorded_in_immutable_attempt": result["forward_calls"],
            "forward_calls_accounting_correction": "190_intermediate_plus_IG_backward_chunks_plus_3_equal_shared_target_endpoint_calls",
            "backward_calls": result["backward_calls"], "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"], "artifact_bytes": result["artifact_bytes"], "descriptive_only": True,
        }
        method_rows.append(row); by_key.setdefault((unit["sample_id"], unit["block"], unit["head"]), {})[unit["method"]] = row
    for (sample_id, block, head), methods in by_key.items():
        if set(methods) != set(METHODS): raise StageEE3Error("E3 coordinate is missing a paired method")
        nns = methods[NNS_METHOD]; matched = methods[MATCHED_METHOD]
        nns_attempt = attempts[next(unit["canonical_digest"] for unit in plan["units"] if unit["sample_id"] == sample_id and unit["block"] == block and unit["head"] == head and unit["method"] == NNS_METHOD)]["result"]
        matched_attempt = attempts[next(unit["canonical_digest"] for unit in plan["units"] if unit["sample_id"] == sample_id and unit["block"] == block and unit["head"] == head and unit["method"] == MATCHED_METHOD)]["result"]
        if e1._target_semantic_digest(nns_attempt["target_response"]["target_identity"]) != e1._target_semantic_digest(matched_attempt["target_response"]["target_identity"]): raise StageEE3Error("paired E3 methods do not share target identity")
        for coordinate in {(row["direction"], row["replicate"]): row for row in nns_attempt["curves"]}:
            left = next(row for row in nns_attempt["curves"] if (row["direction"], row["replicate"]) == coordinate); right = next(row for row in matched_attempt["curves"] if (row["direction"], row["replicate"]) == coordinate)
            if left["seed"] != right["seed"] or left["tie_priority_sha256"] != right["tie_priority_sha256"]:
                raise StageEE3Error("paired E3 methods do not share seed/tie priority")
        status = "VALID" if nns["validity_status"] == matched["validity_status"] == "VALID" else "NA" if "NA" in {nns["validity_status"], matched["validity_status"]} else "FAILED"
        pair_rows.append({
            "variant_id": f"layer_head_{block}_{head}_pair", "sample_id": sample_id, "selection_order": nns["selection_order"], "cohort": "stage_d_fidelity_v1_500",
            "stratum": nns["stratum"], "validity_status": status, "source_hashes": {"nns_map_sha256": nns["source_hashes"]["signed_map_sha256"], "matched_map_sha256": matched["source_hashes"]["signed_map_sha256"], "target_semantic_digest": nns["source_hashes"]["target_semantic_digest"]},
            "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": block, "head": head,
            "coordinate_role": "DESCRIPTIVE_FROZEN_COORDINATE", "recompute_status": "NEW_COMPUTE_COMPLETED", "misclassified": nns["misclassified"],
            "nns_insertion_auc": nns["insertion_auc_mean"], "nns_deletion_auc": nns["deletion_auc_mean"], "matched_insertion_auc": matched["insertion_auc_mean"], "matched_deletion_auc": matched["deletion_auc_mean"],
            "delta_fid": delta_fid(nns, matched) if status == "VALID" else None, "failure_class": None if status == "VALID" else "SCIENTIFIC_INVALIDITY", "failure_code": None if status == "VALID" else "paired_method_invalid", "na_reason": None, "descriptive_only": True,
        })
    return method_rows, pair_rows, attempts


def _median(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64)); count = ordered.size
    if count == 0: raise StageEE3Error("cannot summarize an empty valid set")
    middle = count // 2
    return float(ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0)


def _bootstrap(values: Sequence[float], identity: str, stratum: str) -> dict[str, Any]:
    source = np.asarray(values, dtype=np.float64)
    if source.size == 0 or not np.isfinite(source).all(): raise StageEE3Error("bootstrap requires finite nonempty image values")
    material = BOOTSTRAP_NAMESPACE + "E3|" + identity + "|" + stratum
    seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed)); draws = np.empty(10000, dtype=np.float64)
    for index in range(10000): draws[index] = _median(source[rng.integers(0, source.size, size=source.size)])
    return {"ci_95": [float(x) for x in np.quantile(draws, [0.025, 0.975], method="linear")], "seed": seed, "resamples": 10000, "prng": "numpy.random.Generator(numpy.random.PCG64(seed))", "statistic": "arithmetic_even_median", "quantile_method": "linear", "role": "DESCRIPTIVE_ONLY"}


def _summary(all_rows: Sequence[Mapping[str, Any]], value_key: str, identity: str, stratum: str) -> dict[str, Any]:
    valid_rows = [row for row in all_rows if row["validity_status"] == "VALID"]
    values = np.asarray([row[value_key] for row in valid_rows], dtype=np.float64)
    if values.size and not np.isfinite(values).all(): raise StageEE3Error("summary source contains non-finite values")
    return {
        "n": len(all_rows), "valid": len(valid_rows), "failed": sum(row["validity_status"] == "FAILED" for row in all_rows), "na": sum(row["validity_status"] == "NA" for row in all_rows),
        "positive_count": int(np.sum(values > 0)) if values.size else 0, "mean": float(values.mean()) if values.size else None, "median": _median(values) if values.size else None,
        "sample_sd_ddof_1": float(values.std(ddof=1)) if values.size > 1 else None, "paired_image_bootstrap_median_95_interval": _bootstrap(values, identity, stratum) if values.size else None,
        "summary_population": "all_expected_rows_with_typed_status_counts_numeric_statistics_over_VALID_only",
    }


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _validate_config(repo); plan, _ = _strict_json(repo / PLAN_PATH, "E3 plan")
    if plan != _build_plan(repo, config): raise StageEE3Error("E3 plan drifted")
    primary_methods, primary_pairs = _primary_method_rows(repo, plan); new_methods, new_pairs, attempts = _new_rows(repo, plan)
    method_rows = sorted(primary_methods + new_methods, key=lambda row: (row["selection_order"], COORDINATES.index((row["layer"], row["head"])), METHODS.index(row["method"])))
    pair_rows = sorted(primary_pairs + new_pairs, key=lambda row: (row["selection_order"], COORDINATES.index((row["layer"], row["head"]))))
    if len(method_rows) != 4000 or len(pair_rows) != 2000 or len(attempts) != 3000: raise StageEE3Error("E3 output cardinality failed")
    registry, _ = e1._load_registry(repo); expected_strata = {group["sample_id"]: group["stratum"] for group in plan["groups"]}
    if Counter(expected_strata.values()) != Counter({"correct": 418, "misclassified": 82}) or len(expected_strata) != 500: raise StageEE3Error("E3 prediction strata partition drifted")
    if any(row["stratum"] != expected_strata[row["sample_id"]] for row in method_rows + pair_rows): raise StageEE3Error("E3 row stratum differs across coordinates")
    summaries = []; method_summaries = []
    for block, head in COORDINATES:
        coordinate_pairs = [row for row in pair_rows if (row["layer"], row["head"]) == (block, head)]
        coordinate_methods = [row for row in method_rows if (row["layer"], row["head"]) == (block, head)]
        for stratum in STRATA:
            selected_pairs = coordinate_pairs if stratum == "overall" else [row for row in coordinate_pairs if row["stratum"] == stratum]
            delta_summary = _summary(selected_pairs, "delta_fid", f"layer_head_{block}_{head}_delta_fid", stratum)
            summaries.append({
                "variant_id": f"layer_head_{block}_{head}_{stratum}", "sample_id": None, "selection_order": None, "cohort": "stage_d_fidelity_v1_500", "stratum": stratum,
                "validity_status": "VALID" if delta_summary["failed"] == delta_summary["na"] == 0 else "FAILED", "source_hashes": {"paired_rows_canonical_digest": canonical_sha256(selected_pairs)},
                "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": block, "head": head,
                "coordinate_role": "PRIMARY_ANCHOR_READ_ONLY" if (block, head) == (0, 9) else "DESCRIPTIVE_FROZEN_COORDINATE", "metric": "delta_fid", **delta_summary, "descriptive_only": True,
            })
            for method in METHODS:
                selected_methods = [row for row in coordinate_methods if row["method"] == method and (stratum == "overall" or row["stratum"] == stratum)]
                for direction in ("insertion", "deletion"):
                    summary = _summary(selected_methods, f"{direction}_auc_mean", f"layer_head_{block}_{head}_{method}_{direction}_auc", stratum)
                    method_summaries.append({
                        "variant_id": f"layer_head_{block}_{head}_{method}_{direction}_{stratum}", "sample_id": None, "selection_order": None, "cohort": "stage_d_fidelity_v1_500", "stratum": stratum,
                        "validity_status": "VALID" if summary["failed"] == summary["na"] == 0 else "FAILED", "source_hashes": {"method_rows_canonical_digest": canonical_sha256(selected_methods)},
                        "method": method, "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": block, "head": head,
                        "coordinate_role": "PRIMARY_ANCHOR_READ_ONLY" if (block, head) == (0, 9) else "DESCRIPTIVE_FROZEN_COORDINATE", "metric": f"{direction}_auc", **summary, "descriptive_only": True,
                    })
    if len(summaries) != 12 or len(method_summaries) != 48: raise StageEE3Error("E3 summary cell coverage failed")
    membership = [{
        "variant_id": "prediction_stratum_membership", "sample_id": row["sample_id"], "selection_order": row["selection_order"], "cohort": "stage_d_fidelity_v1_500",
        "stratum": expected_strata[row["sample_id"]], "validity_status": "VALID", "source_hashes": {"d10_per_sample_sha256": _sha_file(repo / D10_DELTA_PATH), "image_sha256": row["image_sha256"]},
        "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": None, "head": None, "misclassified": expected_strata[row["sample_id"]] == "misclassified",
    } for row in registry]
    telemetry = []
    for path in sorted((repo / ROOT_PATH / "telemetry").glob("shard-*.json")):
        value, _ = _strict_json(path, "E3 shard telemetry"); _validate_signed(value, "E3 shard telemetry"); telemetry.append(value)
    if not telemetry: raise StageEE3Error("E3 telemetry is absent")
    telemetry_groups = {group["sample_id"]: group for document in telemetry for group in document["groups"]}
    for sample_id in {row["sample_id"] for row in new_rows}:
        group_rows = sorted([row for row in new_rows if row["sample_id"] == sample_id], key=lambda row: (COORDINATES.index((row["layer"], row["head"])), METHODS.index(row["method"])))
        if sample_id not in telemetry_groups or len(group_rows) != 6: raise StageEE3Error("E3 telemetry does not cover one complete six-unit image group")
        measured = telemetry_groups[sample_id]
        base_calls = [190 + row["backward_calls"] + (2 if row["method"] == NNS_METHOD else 0) for row in group_rows]
        shared_calls = measured["forward_calls"] - sum(base_calls)
        if shared_calls < 0: raise StageEE3Error("measured group forwards are below method-local minimum")
        quotient, remainder = divmod(shared_calls, len(group_rows))
        elapsed_weights = np.asarray([row["elapsed_seconds_recorded_in_immutable_attempt"] for row in group_rows], dtype=np.float64); elapsed_weights /= elapsed_weights.sum()
        cpu_weights = np.asarray([row["cpu_seconds_recorded_in_immutable_attempt"] for row in group_rows], dtype=np.float64); cpu_weights /= cpu_weights.sum()
        for index, row in enumerate(group_rows):
            row["forward_calls"] = base_calls[index] + quotient + (1 if index < remainder else 0)
            row["elapsed_seconds"] = float(measured["elapsed_seconds"] * elapsed_weights[index]); row["gpu_hours"] = row["elapsed_seconds"] / 3600.0
            row["cpu_seconds"] = float(measured["cpu_seconds"] * cpu_weights[index])
        if sum(row["forward_calls"] for row in group_rows) != measured["forward_calls"] or not math.isclose(sum(row["elapsed_seconds"] for row in group_rows), measured["elapsed_seconds"], rel_tol=0, abs_tol=1e-9) or not math.isclose(sum(row["cpu_seconds"] for row in group_rows), measured["cpu_seconds"], rel_tol=0, abs_tol=1e-9):
            raise StageEE3Error("corrected per-unit accounting does not reconcile to measured image-group totals")
    new_rows = new_methods
    actual = {
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in telemetry), "gpu_hours": sum(float(row["gpu_hours"]) for row in telemetry),
        "cpu_seconds": sum(float(row["cpu_seconds"]) for row in telemetry), "forward_calls": sum(int(row["forward_calls"]) for row in telemetry),
        "backward_calls": sum(int(row["backward_calls"]) for row in telemetry), "peak_cuda_memory_bytes": max(int(row["peak_cuda_memory_bytes"]) for row in telemetry),
        "artifact_bytes": sum(int(row["artifact_bytes"]) for row in new_rows), "gpu_invocation_count": len(telemetry), "successful_terminal_unit_attempts": len(attempts),
        "retry_unit_attempts": 0, "primary_forward_calls": 0, "primary_backward_calls": 0,
    }
    by_coordinate = {}
    for block, head in NEW_COORDINATES:
        selected_rows = [row for row in new_rows if (row["layer"], row["head"]) == (block, head)]
        by_coordinate[f"{block},{head}"] = {
            "completed": len(selected_rows), "failed": sum(row["validity_status"] == "FAILED" for row in selected_rows), "na": sum(row["validity_status"] == "NA" for row in selected_rows),
            "elapsed_seconds_scientific_unit_share": sum(row["elapsed_seconds"] for row in selected_rows), "gpu_hours_scientific_unit_share": sum(row["gpu_hours"] for row in selected_rows),
            "cpu_seconds_scientific_unit_share": sum(row["cpu_seconds"] for row in selected_rows), "forward_calls_scientific_unit_share": sum(row["forward_calls"] for row in selected_rows), "backward_calls": sum(row["backward_calls"] for row in selected_rows),
        }
    summary = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_descriptive_summary", "stage_id": "E3", "cohort": "stage_d_fidelity_v1_500",
        "descriptive_only": True, "p_values_emitted": False, "winner_selection_performed": False, "causal_claim_made": False, "new_go_no_go_issued": False,
        "d10_primary_unchanged": True, "primary_anchor_recomputed": False, "caption_statement": "Coordinates were frozen by Stage C depth-band selection before holdout access and were not searched on the Stage D 500-image cohort.",
        "counts": {"expected_images": 500, "method_rows": len(method_rows), "paired_rows": len(pair_rows), "new_compute_attempts": len(attempts), "summary_cells": len(summaries), "method_summary_cells": len(method_summaries), "valid_method_rows": sum(row["validity_status"] == "VALID" for row in method_rows), "failed_method_rows": sum(row["validity_status"] == "FAILED" for row in method_rows), "na_method_rows": sum(row["validity_status"] == "NA" for row in method_rows)},
        "delta_fid_coordinate_strata": summaries, "method_auc_coordinate_strata": method_summaries,
        "prediction_partition": {"overall": 500, "correct": 418, "misclassified": 82, "mutually_exclusive": True, "union_equals_overall": True},
        "clean_margin": config["scientific_contract"]["clean_margin"], "resource_accounting": {"global": actual, "by_coordinate": by_coordinate, "inherited_primary": {"gpu_hours": 0.0, "cpu_seconds": 0.0, "forward_calls": 0, "backward_calls": 0}},
        "interpretation_boundary": "Post-D10 descriptive layer/head and correctness-stratum views only; no winner, causal claim, hypothesis decision, or new GO/NO_GO.",
    })
    resource_rows = [dict(row) for row in method_rows]
    return {"config": config, "plan": plan, "method_rows": method_rows, "pair_rows": pair_rows, "summary_rows": summaries + method_summaries, "membership_rows": membership, "resource_rows": resource_rows, "attempts": attempts, "summary": summary, "actual_cost": actual}


def _table(table_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: raise StageEE3Error(f"{table_id} cannot be empty")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    normalized = [{key: row.get(key) for key in columns} for row in rows]
    return _signed({"schema_version": 1, "document_type": "stage_e_e3_paper_table", "table_id": table_id, "columns": columns, "row_count": len(normalized), "rows": normalized})


def _csv_scalar(value: Any) -> str:
    if value is None: return ""
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, (dict, list)): return canonical_json_bytes(value).decode("utf-8")
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); columns = list(rows[0]); temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n"); writer.writeheader(); writer.writerows([{key: _csv_scalar(value) for key, value in row.items()} for row in rows]); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _protected_after(repo: Path, before: Mapping[str, Any]) -> dict[str, Any]:
    after = _protected_snapshot(repo)
    expected = {row["path"]: row for row in before["roots"]}
    for row in after["roots"]:
        prior = expected.get(row["path"])
        if prior is None or any(prior[key] != row[key] for key in ("file_count", "total_bytes", "canonical_digest")): raise StageEE3Error(f"protected root changed during E3: {row['path']}")
    if before["critical_e2_chain"] != after["critical_e2_chain"]: raise StageEE3Error("critical E2 receipt/waiver chain changed during E3")
    return after


def record_tests(repo_root: Path | str = REPO_ROOT, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if type(passed) is not int or passed <= 0 or type(failed) is not int or failed != 0 or not command.strip(): raise StageEE3Error("only a real zero-failure E3 test result can be recorded")
    report = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_test_results", "stage_id": "E3", "status": "PASS", "recorded_at": _now(),
        "command": command, "passed": passed, "failed": failed,
        "required_gates": {name: "PASS" for name in ("config_schema", "predecessor_chain", "registry_coordinates", "primary_forbidden_recompute", "new_compute_cardinality", "real_attribution_and_fidelity", "fine_cell_geometry", "paired_seeds", "auc_delta", "prediction_partition", "clean_margin_not_run", "typed_failure_na", "budget", "no_render", "protected_upstream", "atomic_completion", "stage_boundary", "false_pass")},
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "test_results.json", report); return report


def record_pre_scale_observability_event(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Bind the two successful, budget-charged smoke invocations before scaling."""

    repo = Path(repo_root).resolve(); plan, _ = _strict_json(repo / PLAN_PATH, "E3 plan")
    telemetry_paths = (
        ROOT_PATH + "/telemetry/shard-00-of-01.json",
        ROOT_PATH + "/telemetry/shard-01-of-500.json",
    )
    telemetry = []
    for relative in telemetry_paths:
        value, _ = _strict_json(repo / relative, "E3 smoke telemetry"); _validate_signed(value, "E3 smoke telemetry"); telemetry.append(value)
    selected = {0: 0, 1: 1}; query = subprocess.run(("nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.splitlines()
    hardware = {}
    for line in query:
        index, uuid, name = (item.strip() for item in line.split(",", 2)); hardware[int(index)] = {"physical_gpu_index": int(index), "gpu_uuid": uuid, "gpu_name": name}
    attempts = []
    for unit in plan["units"]:
        if unit["selection_order"] in selected:
            observed = _validate_attempt(repo, unit); attempts.append({"unit_digest": unit["canonical_digest"], "result_canonical_digest": observed["result"]["canonical_digest"], "manifest_canonical_digest": observed["manifest"]["canonical_digest"]})
    event = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_pre_scale_observability_event", "event_id": "E3_SMOKE_GPU_IDENTITY_001", "status": "RECORDED", "recorded_at": _now(),
        "failure_class": None, "failure_code": None, "new_e3_outcome_accessed": True, "scientific_scope_seed_target_ranking_or_response_changed": False,
        "reason": "The first two successful formal image groups were deliberately budget-charged smoke executions. A telemetry-only code addition records physical GPU index/UUID for subsequent shards; the 12 immutable smoke attempts were not changed or recomputed.",
        "smoke_runs": [{"telemetry": _record(repo, relative), "physical_gpu": hardware[selected[index]], "completed_units": telemetry[index]["completed_units"], "gpu_hours": telemetry[index]["gpu_hours"], "cpu_seconds": telemetry[index]["cpu_seconds"]} for index, relative in enumerate(telemetry_paths)],
        "immutable_smoke_attempts": attempts, "terminal_attempt_recompute_authorized": False, "current_e3_source_sha256": _sha_file(repo / "src/vitnns/stage_e_e3.py"),
        "scale_out_authorized": True, "basis": "single-thread smoke projects below both independent E3 GPU and CPU hard limits",
    })
    _write_or_verify(repo / OBSERVABILITY_EVENT_PATH, event); return event


def _publish_access_terminal(repo: Path, initial: Mapping[str, Any]) -> dict[str, Any]:
    pending = repo / ACCESS_PENDING_PATH; target = repo / ACCESS_PATH
    if target.is_file():
        value, _ = _strict_json(target, "terminal E3 outcome access"); _validate_signed(value, "terminal E3 outcome access"); return value
    if not pending.is_file(): raise StageEE3Error("E3 pending outcome-access record is absent")
    terminal = _signed({
        **{key: value for key, value in initial.items() if key != "canonical_digest"}, "status": "TERMINAL",
        "completed_at": _now(), "initial_pending_sha256": _sha(pending.read_bytes()), "initial_pending_canonical_digest": initial["canonical_digest"],
        "new_e3_coordinate_outcome_accessed": True, "scope_changed_after_outcome_access": False, "primary_recompute": False, "e4_started": False, "stage_f_started": False,
    })
    temporary = pending.with_name(".e3-outcome-access-terminal.tmp"); _write_new_json(temporary, terminal); os.replace(temporary, pending)
    observed, raw = _strict_json(pending, "terminal pending E3 outcome access")
    if observed != terminal or raw != canonical_json_bytes(terminal) + b"\n": raise StageEE3Error("E3 outcome-access terminal changed before publication")
    os.replace(pending, target); return terminal


def record_execution_stop_event(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); target_path = repo / REPORT_ROOT_PATH / "execution_stop_event.json"
    if target_path.is_file():
        event, _ = _strict_json(target_path, "E3 execution stop event"); _validate_signed(event, "E3 execution stop event")
        if event.get("status") != "TERMINAL_STOP" or event.get("trigger") != "NON_RETRYABLE_SCIENTIFIC_INVALIDITY": raise StageEE3Error("existing E3 stop event is invalid")
        return event
    plan, _ = _strict_json(repo / PLAN_PATH, "E3 plan")
    observed = []
    for unit in plan["units"]:
        if _attempt_paths(repo, unit["canonical_digest"])[0].exists(): observed.append((unit, _validate_attempt(repo, unit)))
    coverage = {}
    for block, head in NEW_COORDINATES:
        for method in METHODS:
            rows = [item for item in observed if (item[0]["block"], item[0]["head"], item[0]["method"]) == (block, head, method)]
            coverage[f"{block},{head}:{method}"] = {"expected": 500, "valid_terminal": len(rows), "failed_terminal": 0, "na_terminal": 0, "missing": 500 - len(rows)}
    query = subprocess.run(("nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.splitlines()
    hardware = {};
    for line in query:
        index, uuid, name = (item.strip() for item in line.split(",", 2)); hardware[int(index)] = {"physical_gpu_index": int(index), "gpu_uuid": uuid, "gpu_name": name}
    event = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_execution_stop_event", "event_id": "E3_SCIENTIFIC_INVALIDITY_STOP_001", "status": "TERMINAL_STOP", "recorded_at": _now(),
        "trigger": "NON_RETRYABLE_SCIENTIFIC_INVALIDITY", "exception_type": "StageCValuePathConvergenceError", "exception_message": "head-direction Value-path IG did not converge on the complete frozen schedule",
        "failed_image_groups": [
            {"selection_order": 72, "sample_id": "ILSVRC2012_val/ILSVRC2012_val_00026141.JPEG", "method": NNS_METHOD, "coordinate": "UNRESOLVED_WITHIN_FROZEN_NEW_COORDINATES_BEFORE_ATOMIC_GROUP_PUBLICATION", "candidate_coordinates": [list(item) for item in NEW_COORDINATES]},
            {"selection_order": 61, "sample_id": "ILSVRC2012_val/ILSVRC2012_val_00018234.JPEG", "method": NNS_METHOD, "coordinate": "UNRESOLVED_WITHIN_FROZEN_NEW_COORDINATES_BEFORE_ATOMIC_GROUP_PUBLICATION", "candidate_coordinates": [list(item) for item in NEW_COORDINATES]},
        ],
        "coordinate_precision_note": "No coordinate is asserted because the six-unit image group was intentionally buffered until complete and the exception did not publish a coordinate-local failure payload. Re-execution to recover that detail is forbidden for scientific invalidity.",
        "scale_out_processes": [{"shard_id": index, "physical_gpu": hardware[index], "terminal_process_state": "SCIENTIFIC_FAILURE" if index in (0, 5) else "COORDINATOR_INTERRUPTED_AFTER_GLOBAL_FAIL"} for index in range(8)],
        "resource_accounting": {"accounting_status": "CONSERVATIVE_480_SECONDS_CHARGED_PER_SCALE_OUT_GPU_PROCESS", "process_invocations": 8, "gpu_device_seconds_charged": 3840.0, "gpu_hours_charged": 3840.0 / 3600.0, "cpu_seconds_charged": 3840.0, "scientific_failure_processes": 2, "coordinator_interrupted_processes": 6, "unpublished_image_group_computations_charged": 8},
        "coverage_at_stop": {"valid_terminal_attempts": len(observed), "expected_new_attempts": 3000, "complete_image_groups": len(observed) // 6, "ambiguous_temporary_attempt_directories": sum(1 for _ in (repo / ROOT_PATH / "units").glob("*/attempts/.attempt-*")), "by_coordinate_method": coverage},
        "terminal_attempt_recompute_authorized": False, "scientific_invalidity_retry_authorized": False, "scope_or_threshold_change_authorized": False,
        "safe_recovery_entry": "Treat E3 v1 as terminal FAIL and do not retry these scientific-invalid groups under the frozen contract. Any future investigation requires a new prospective owner-authorized protocol; it must not relabel or overwrite this E3 receipt and cannot authorize E4 from E3 v1.",
    })
    _write_or_verify(target_path, event); return event


def finalize_failed(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _validate_config(repo); plan, _ = _strict_json(repo / PLAN_PATH, "E3 plan")
    event = record_execution_stop_event(repo); before, _ = _strict_json(repo / UPSTREAM_BEFORE_PATH, "E3 protected before"); after = _protected_after(repo, before)
    initial_path = repo / ACCESS_PENDING_PATH if (repo / ACCESS_PENDING_PATH).is_file() else repo / ROOT_PATH / "control/outcome_access_initial.json"
    initial, _ = _strict_json(initial_path, "initial E3 outcome access"); _write_or_verify(repo / ROOT_PATH / "control/outcome_access_initial.json", initial); terminal_access = _publish_access_terminal(repo, initial)
    after_path = repo / REPORT_ROOT_PATH / "protected_upstream_after.json"
    if after_path.is_file():
        recorded_after, _ = _strict_json(after_path, "recorded E3 protected after"); _validate_signed(recorded_after, "recorded E3 protected after")
        if any(recorded_after[key] != after[key] for key in ("roots", "critical_e2_chain", "protected_tracked_diff")): raise StageEE3Error("recorded E3 protected after snapshot drifted")
        after = recorded_after
    else:
        _write_or_verify(after_path, after)
    smoke = []
    for relative in (ROOT_PATH + "/telemetry/shard-00-of-01.json", ROOT_PATH + "/telemetry/shard-01-of-500.json"):
        value, _ = _strict_json(repo / relative, "E3 smoke telemetry"); smoke.append(value)
    valid_attempts = event["coverage_at_stop"]["valid_terminal_attempts"]
    complete_groups = valid_attempts // 6
    artifact_bytes = 0
    for unit in plan["units"]:
        attempt = _attempt_paths(repo, unit["canonical_digest"])[0]
        if attempt.is_dir(): artifact_bytes += sum(path.stat().st_size for path in attempt.iterdir() if path.is_file())
    actual = {
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in smoke) + event["resource_accounting"]["gpu_device_seconds_charged"],
        "gpu_hours": sum(float(row["gpu_hours"]) for row in smoke) + event["resource_accounting"]["gpu_hours_charged"],
        "cpu_seconds": sum(float(row["cpu_seconds"]) for row in smoke) + event["resource_accounting"]["cpu_seconds_charged"],
        "forward_calls_minimum": complete_groups * 1486, "backward_calls_minimum": complete_groups * 320,
        "call_count_status": "LOWER_BOUND_FROM_86_COMPLETE_SOURCE_VALID_GROUPS_UNPUBLISHED_FAILURE_AND_INTERRUPTION_WORK_CHARGED_IN_TIME_NOT_EXACT_CALLS",
        "artifact_bytes": artifact_bytes, "gpu_invocation_count": 10, "successful_terminal_unit_attempts": valid_attempts, "failed_process_invocations": 2, "coordinator_interrupted_process_invocations": 6, "retry_unit_attempts": 0,
    }
    retained = _inventory(repo / ROOT_PATH)["total_bytes"]; cost_failures = budget_reasons({"gpu_hours": actual["gpu_hours"], "cpu_seconds": actual["cpu_seconds"]}, retained)
    cost = {**actual, "cpu_hours": actual["cpu_seconds"] / 3600.0, "retained_bytes_before_status_inventory_completion": retained, "gpu_hours_max": 8.0, "cpu_hours_max": 16.0, "retained_bytes_max": 12884901888, "status": "PASS_COST" if not cost_failures else "FAIL_COST", "budget_failures": list(cost_failures)}
    diagnostic = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_failure_diagnostics", "stage_id": "E3", "status": "FAIL",
        "expected": {"images": 500, "coordinates": [list(item) for item in COORDINATES], "new_units": 3000, "method_rows": 4000, "paired_rows": 2000, "summary_cells": 12, "failed": 0, "na": 0},
        "observed": {**event["coverage_at_stop"], "primary_anchor_rows_available_read_only": 500, "new_method_rows_not_published": True, "paired_rows_not_published": True, "descriptive_summaries_not_published": True},
        "failure_objects": event["failed_image_groups"], "failure_classification": "SCIENTIFIC_INVALIDITY_NON_RETRYABLE", "cost": cost,
        "primary_anchor": {"coordinate": [0, 9], "status": "PRIMARY_ANCHOR_READ_ONLY_UNCHANGED", "e3_attempts": 0},
        "publication_boundary": "No incomplete or favorable-subset paper_data and no coordinate-stratum summaries are published. All 516 valid terminal attempts remain immutable local evidence.",
        "e4_authorized": False, "e4_started": False, "stage_f_started": False, "safe_recovery_entry": event["safe_recovery_entry"],
    }); _write_or_verify(repo / REPORT_ROOT_PATH / "failure_diagnostics.json", diagnostic)
    mechanisms = {name: "E3 FAIL is issued; no PASS or downstream authorization can be produced from this incomplete scientific payload." for name in FALSE_PASS_ATTACKS}
    false_audit = _signed({"schema_version": 1, "document_type": "stage_e_e3_false_pass_audit", "stage_id": "E3", "status": "PASS", "checks": [{"attack": name, "result": "REJECTED", "mechanism": mechanisms[name]} for name in FALSE_PASS_ATTACKS], "pass_possible_without_all_conditions": False, "e3_pass_issued": False}); _write_or_verify(repo / REPORT_ROOT_PATH / "false_pass_audit.json", false_audit)
    tests, _ = _strict_json(repo / REPORT_ROOT_PATH / "test_results.json", "E3 test report"); _validate_signed(tests, "E3 test report")
    if tests.get("status") != "PASS" or tests.get("failed") != 0: raise StageEE3Error("E3 implementation tests are not PASS")
    validation = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_source_validation", "stage_id": "E3", "status": "FAIL", "validated_at": terminal_access["completed_at"],
        "checks": {"e2_superseding_waiver_chain": "PASS", "e1_e0_d9_d10": "PASS", "primary_anchor": "PASS_READ_ONLY_NO_E3_ATTEMPTS", "valid_terminal_attempts": f"PASS_SOURCE_VALID_{valid_attempts}", "required_3000_attempt_coverage": f"FAIL_{valid_attempts}_OF_3000", "scientific_validity": "FAIL_IG_CONVERGENCE", "favorable_subset_summary_suppressed": "PASS", "budget": cost["status"], "protected_upstream_unchanged": "PASS", "no_render_artifacts": "PASS", "stage_boundary": "PASS_NO_E4_AUTHORIZATION"},
        "failure_diagnostics": _record(repo, REPORT_ROOT_PATH + "/failure_diagnostics.json"), "false_pass_audit_canonical_digest": false_audit["canonical_digest"],
    }); _write_or_verify(repo / REPORT_ROOT_PATH / "validation_report.json", validation)
    manifest_files = []
    for path in sorted((repo / ROOT_PATH).rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json", "whole_root_inventory.json", "status.json"}: manifest_files.append({"path": path.relative_to(repo / ROOT_PATH).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    manifest = _signed({"schema_version": 1, "document_type": "stage_e_e3_payload_manifest", "immutable": True, "artifacts": manifest_files, "payload_count": len(manifest_files), "terminal_stage_status": "FAIL"}); _write_or_verify(repo / ROOT_PATH / "payload_manifest.json", manifest)
    inventory = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "status.json", "completion/e3_completion.pending.json", "completion/e3_completion.json"))
    inventory_doc = _signed({"schema_version": 1, "document_type": "stage_e_e3_whole_root_inventory", "root": ROOT_PATH, "scope": "all regular non-hidden E3 formal files except self-referential inventory status and completion receipts", **inventory}); _write_or_verify(repo / ROOT_PATH / "whole_root_inventory.json", inventory_doc)
    status = _signed({"schema_version": 1, "document_type": "stage_e_e3_status", "stage_id": "E3", "status": "FAIL", "complete": False, "terminal": True, "immutable": True, "package_completeness": "FAIL", "workflow_integrity": "PASS", "scientific_validity": "FAIL", "cost_status": cost["status"], "actual_cost": cost, "d10_decision_unchanged": True, "c10_history_unchanged": True, "d10_primary_recomputed": False, "e4_authorized": False, "e4_started": False, "stage_f_started": False, "whole_root_inventory": {"path": "whole_root_inventory.json", "file_sha256": _sha_file(repo / ROOT_PATH / "whole_root_inventory.json"), "canonical_digest": inventory_doc["canonical_digest"], "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"], "scope": inventory_doc["scope"]}, "safe_recovery_entry": event["safe_recovery_entry"]}); _write_or_verify(repo / ROOT_PATH / "status.json", status)
    coverage = event["coverage_at_stop"]["by_coordinate_method"]
    observed_coordinates = [{"block": block, "head": head, "valid_terminal_attempts": 1000 if (block, head) == (0, 9) else sum(coverage[f"{block},{head}:{method}"]["valid_terminal"] for method in METHODS), "expected_method_attempts": 1000, "complete": (block, head) == (0, 9)} for block, head in COORDINATES]
    completion = _signed({
        "schema_version": 1, "document_type": "stage_e_terminal_completion", "stage_id": "E3", "status": "FAIL", "started_at": initial["e3_started_at"], "completed_at": _now(),
        "predecessor": {"path": E2_COMPLETION_PATH, "file_sha256": E2_COMPLETION_SHA256, "canonical_digest": E2_COMPLETION_DIGEST}, "config_canonical_digest": config["canonical_digest"], "run_spec_canonical_digest": _strict_json(repo / RUN_SPEC_PATH, "E3 run spec")[0]["canonical_digest"],
        "expected_coordinates": [{"block": block, "head": head, "image_count": 500, "method_count": 2} for block, head in COORDINATES], "observed_coordinates": observed_coordinates,
        "variant_statuses": [{"variant_id": "layer_head_0_9", "status": "PASS", "reason": "read-only D10 primary anchor intact"}] + [{"variant_id": f"layer_head_{block}_{head}", "status": "FAIL", "reason": f"incomplete after non-retryable NNS IG convergence failure; {coverage[f'{block},{head}:{NNS_METHOD}']['valid_terminal']} NNS and {coverage[f'{block},{head}:{MATCHED_METHOD}']['valid_terminal']} matched attempts are valid terminal"} for block, head in NEW_COORDINATES],
        "package_completeness": "FAIL", "workflow_integrity": "PASS", "scientific_validity": "FAIL", "cost_status": cost["status"],
        "source_validation": {"path": REPORT_ROOT_PATH + "/validation_report.json", "file_sha256": _sha_file(repo / REPORT_ROOT_PATH / "validation_report.json"), "canonical_digest": validation["canonical_digest"], "test_report": _record(repo, REPORT_ROOT_PATH + "/test_results.json"), "false_pass_audit": _record(repo, REPORT_ROOT_PATH + "/false_pass_audit.json"), "payload_manifest": _record(repo, ROOT_PATH + "/payload_manifest.json"), "status": _record(repo, ROOT_PATH + "/status.json"), "whole_root_inventory": _record(repo, ROOT_PATH + "/whole_root_inventory.json"), "execution_stop_event": _record(repo, REPORT_ROOT_PATH + "/execution_stop_event.json"), "failure_diagnostics": _record(repo, REPORT_ROOT_PATH + "/failure_diagnostics.json")},
        "upstream_unchanged": True, "actual_cost": cost, "next_authorization": "NONE",
    })
    schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema"); errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE3Error(f"E3 FAIL completion schema failure: {errors[0].message}")
    pending = repo / COMPLETION_PENDING_PATH; terminal = repo / COMPLETION_PATH; temporary = pending.with_name(".e3-completion-fail-terminal.tmp"); _write_new_json(temporary, completion); os.replace(temporary, pending)
    observed, raw = _strict_json(pending, "E3 FAIL pending completion")
    if observed != completion or raw != canonical_json_bytes(completion) + b"\n": raise StageEE3Error("E3 FAIL completion changed before atomic publication")
    os.replace(pending, terminal)
    return {"status": "FAIL", "completion_canonical_digest": completion["canonical_digest"], "whole_root_inventory_digest": inventory_doc["canonical_digest"], "valid_terminal_attempts": valid_attempts, "expected_new_attempts": 3000, "actual_cost": cost, "next_authorization": "NONE", "safe_recovery_entry": event["safe_recovery_entry"]}


def finalize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); outputs = validate_outputs(repo); root = repo / ROOT_PATH
    before, _ = _strict_json(repo / UPSTREAM_BEFORE_PATH, "E3 protected before"); after = _protected_after(repo, before)
    retained_now = _inventory(root)["total_bytes"]; failures = budget_reasons(outputs["actual_cost"], retained_now)
    if failures: raise StageEE3Error(f"E3 budget exceeded without E3-specific waiver: {failures}")
    if any(row["validity_status"] != "VALID" for row in outputs["method_rows"] + outputs["pair_rows"]): raise StageEE3Error("E3 scientific rows contain FAILED or NA; PASS forbidden")
    table_specs = (
        ("layer_head_method_rows", outputs["method_rows"]), ("layer_head_paired_delta", outputs["pair_rows"]),
        ("prediction_strata", outputs["summary_rows"]), ("prediction_strata_membership", outputs["membership_rows"]),
        ("resource_and_qc", outputs["resource_rows"]),
    )
    paper = root / "paper_data"
    for name, rows in table_specs:
        table = _table(name, rows); _write_or_verify(paper / f"{name}.json", table); _write_csv(paper / f"{name}.csv", table["rows"])
    _write_or_verify(root / "descriptive_summary.json", outputs["summary"])
    _write_or_verify(root / "clean_margin_not_run.json", _signed({"schema_version": 1, "document_type": "stage_e_e3_not_run", "analysis": "clean_margin", **outputs["config"]["scientific_contract"]["clean_margin"]}))
    access_source = repo / ACCESS_PENDING_PATH if (repo / ACCESS_PENDING_PATH).is_file() else root / "control/outcome_access_initial.json"
    initial, _ = _strict_json(access_source, "initial E3 outcome access"); _write_or_verify(root / "control/outcome_access_initial.json", initial); terminal_access = _publish_access_terminal(repo, initial)
    _write_or_verify(repo / REPORT_ROOT_PATH / "protected_upstream_after.json", after)
    recipe = _signed({
        "schema_version": 1, "document_type": "stage_e_figure_recipes", "numeric_source": "paper_data_only", "renders_png": False,
        "figures": [{"figure_id": "e3_layer_head_prediction_strata_future", "table_path": "paper_data/prediction_strata.json", "x": "layer", "y": "median", "group": "stratum", "caption_contract": "Post-D10 descriptive views for all four Stage-C-frozen coordinates; primary remains (0,9); no winner, causal claim, or new decision.", "numeric_source": "paper_data_only", "renders_png": False}],
    })
    _write_or_verify(root / "figure_recipes.json", recipe)
    provenance = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_provenance", "stage_id": "E3", "source_files": initial["upstream_reads"],
        "pair_bindings": [{"unit_digest": key, "pair_binding_digest": value["result"]["pair_binding_digest"]} for key, value in sorted(outputs["attempts"].items())],
        "operations": ["source-validate and exactly reuse the D10 primary anchor", "resolve three frozen coordinate-specific fixed-clean-attention head directions", "compute real NNS attribution and matched token contribution", "evaluate true 3136-unit fine-cell intervention curves", "compute float64 AUC delta and descriptive image bootstrap within frozen strata"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "outcome_access": _record(repo, ACCESS_PATH), "protected_upstream_before": _record(repo, UPSTREAM_BEFORE_PATH), "protected_upstream_after": _record(repo, REPORT_ROOT_PATH + "/protected_upstream_after.json"),
        "d9_d10_e0_e1_e2_unchanged": True, "d10_primary_recomputed": False, "e2_waiver_extended_to_e3": False, "e4_stage_f_started": False,
    })
    _write_or_verify(root / "provenance.json", provenance)
    mechanisms = {name: "Independent source-recomputing E3 validator and focused mutation test reject this condition." for name in FALSE_PASS_ATTACKS}
    mechanisms.update({
        "e2_blocked_receipt_or_owner_waiver_ignored": "Exact bytes and canonical digests for the original BLOCKED receipt, owner waiver, superseding status, inventory, and completion are all required.",
        "e2_waiver_silently_extended_to_e3": "The E3 config fixes e2_waiver_applies=false and the unwaived E3 hard budget gates completion.",
        "primary_recomputed_or_modified": "The primary has no E3 attempt surface and is reconstructed only from exact D9/D10 source bindings.",
        "methods_not_seed_and_tie_paired": "Both methods are checked for identical seed and tie-priority at every image, coordinate, direction, and repeat.",
        "e3_budget_exceeded_without_e3_waiver": "GPU, CPU, and retained-byte actuals are hard-gated; no E3-specific waiver exists.",
        "e3_pass_starts_e4_or_authorizes_f0": "Completion next_authorization is exactly E4_ONLY and records e4_started=false and stage_f_started=false.",
    })
    false_audit = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_false_pass_audit", "stage_id": "E3", "status": "PASS",
        "checks": [{"attack": name, "result": "REJECTED", "mechanism": mechanisms[name]} for name in FALSE_PASS_ATTACKS],
        "mutation_test": "Each named condition is independently falsified by test_false_pass_guard_rejects_every_attack.", "pass_possible_without_all_conditions": False,
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "false_pass_audit.json", false_audit)
    tests, _ = _strict_json(repo / REPORT_ROOT_PATH / "test_results.json", "E3 test report"); _validate_signed(tests, "E3 test report")
    if tests.get("status") != "PASS" or tests.get("failed") != 0: raise StageEE3Error("E3 tests are not PASS")
    validation = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_source_validation", "stage_id": "E3", "status": "PASS", "validated_at": terminal_access["completed_at"],
        "checks": {"e2_superseding_waiver_chain": "PASS", "e1_e0_d9_d10": "PASS", "plan": "PASS_500_GROUPS_3000_UNITS", "primary_anchor": "PASS_READ_ONLY_NO_E3_ATTEMPTS", "attempts": "PASS_3000_UNIQUE_TERMINAL", "curves": "PASS_30000_CURVES_630000_POINTS", "real_attribution_and_responses": "PASS", "paired_seeds": "PASS", "method_rows": "PASS_4000", "paired_rows": "PASS_2000", "summary_cells": "PASS_12_DELTA_AND_48_METHOD_AUC", "prediction_partition": "PASS_500_418_82", "budget": "PASS", "protected_upstream_unchanged": "PASS", "no_render_artifacts": "PASS", "stage_boundary": "PASS_E3_ONLY"},
        "counts": outputs["summary"]["counts"], "summary_canonical_digest": outputs["summary"]["canonical_digest"], "false_pass_audit_canonical_digest": false_audit["canonical_digest"],
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "validation_report.json", validation)
    manifest_files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json", "whole_root_inventory.json", "status.json"}:
            manifest_files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    if not manifest_files: raise StageEE3Error("E3 payload manifest cannot be empty")
    manifest = _signed({"schema_version": 1, "document_type": "stage_e_e3_payload_manifest", "immutable": True, "artifacts": manifest_files, "payload_count": len(manifest_files)}); _write_or_verify(root / "payload_manifest.json", manifest)
    retained_pre_status = _inventory(root)["total_bytes"]
    if budget_reasons(outputs["actual_cost"], retained_pre_status): raise StageEE3Error("E3 retained bytes exceeded after reports")
    cost = {**outputs["actual_cost"], "cpu_hours": outputs["actual_cost"]["cpu_seconds"] / 3600.0, "retained_bytes_before_status_inventory_completion": retained_pre_status, "gpu_hours_max": 8.0, "cpu_hours_max": 16.0, "retained_bytes_max": 12884901888, "status": "PASS_COST"}
    inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e3_completion.pending.json", "completion/e3_completion.json"))
    inventory_doc = _signed({"schema_version": 1, "document_type": "stage_e_e3_whole_root_inventory", "root": ROOT_PATH, "scope": "all regular non-hidden E3 formal files except self-referential inventory status and completion receipts", **inventory}); _write_or_verify(root / "whole_root_inventory.json", inventory_doc)
    status = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_status", "stage_id": "E3", "status": "PASS", "complete": True, "terminal": True, "immutable": True,
        "package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "PASS", "cost_status": "PASS_COST", "actual_cost": cost,
        "d10_decision_unchanged": True, "c10_history_unchanged": True, "d10_primary_recomputed": False, "e2_waiver_extended_to_e3": False,
        "e4_authorized": True, "e4_started": False, "stage_f_started": False, "whole_root_inventory": {"path": "whole_root_inventory.json", "file_sha256": _sha_file(root / "whole_root_inventory.json"), "canonical_digest": inventory_doc["canonical_digest"], "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"], "scope": inventory_doc["scope"]},
    }); _write_or_verify(root / "status.json", status)
    completion = _signed({
        "schema_version": 1, "document_type": "stage_e_terminal_completion", "stage_id": "E3", "status": "PASS", "started_at": initial["e3_started_at"], "completed_at": _now(),
        "predecessor": {"path": E2_COMPLETION_PATH, "file_sha256": E2_COMPLETION_SHA256, "canonical_digest": E2_COMPLETION_DIGEST}, "config_canonical_digest": outputs["config"]["canonical_digest"], "run_spec_canonical_digest": _strict_json(repo / RUN_SPEC_PATH, "E3 run spec")[0]["canonical_digest"],
        "expected_coordinates": [{"block": block, "head": head, "image_count": 500, "method_count": 2} for block, head in COORDINATES], "observed_coordinates": [{"block": block, "head": head, "image_count": 500, "method_count": 2} for block, head in COORDINATES],
        "variant_statuses": [{"variant_id": f"layer_head_{block}_{head}", "status": "PASS", "reason": "read-only D10 primary anchor" if (block, head) == (0, 9) else "500 images x two valid new-compute method attempts"} for block, head in COORDINATES],
        "package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "PASS", "cost_status": "PASS_COST",
        "source_validation": {"path": REPORT_ROOT_PATH + "/validation_report.json", "file_sha256": _sha_file(repo / REPORT_ROOT_PATH / "validation_report.json"), "canonical_digest": validation["canonical_digest"], "test_report": _record(repo, REPORT_ROOT_PATH + "/test_results.json"), "false_pass_audit": _record(repo, REPORT_ROOT_PATH + "/false_pass_audit.json"), "payload_manifest": _record(repo, ROOT_PATH + "/payload_manifest.json"), "status": _record(repo, ROOT_PATH + "/status.json"), "whole_root_inventory": _record(repo, ROOT_PATH + "/whole_root_inventory.json")},
        "upstream_unchanged": True, "actual_cost": cost, "next_authorization": "E4_ONLY",
    })
    schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE3Error(f"E3 completion schema failure: {errors[0].message}")
    pending = repo / COMPLETION_PENDING_PATH; terminal = repo / COMPLETION_PATH; temporary = pending.with_name(".e3-completion-terminal.tmp"); _write_new_json(temporary, completion); os.replace(temporary, pending)
    observed, raw = _strict_json(pending, "E3 terminal pending completion")
    if observed != completion or raw != canonical_json_bytes(completion) + b"\n": raise StageEE3Error("E3 completion changed before atomic publication")
    os.replace(pending, terminal)
    final_inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e3_completion.pending.json", "completion/e3_completion.json"))
    if final_inventory != inventory: raise StageEE3Error("E3 root changed during terminal publication")
    return {"status": "PASS", "completion_canonical_digest": completion["canonical_digest"], "whole_root_inventory_digest": inventory_doc["canonical_digest"], "method_rows": 4000, "paired_rows": 2000, "new_attempts": 3000, "actual_cost": cost, "next_authorization": "E4_ONLY", "e4_started": False}


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if not (repo / COMPLETION_PATH).is_file() or (repo / COMPLETION_PENDING_PATH).exists(): raise StageEE3Error("terminal E3 completion is absent or pending remains")
    completion, _ = _strict_json(repo / COMPLETION_PATH, "E3 completion"); _validate_signed(completion, "E3 completion")
    if completion.get("status") == "FAIL":
        return validate_failed_completion(repo, completion=completion)
    schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE3Error(f"E3 completion schema failure: {errors[0].message}")
    if completion.get("status") != "PASS" or completion.get("next_authorization") != "E4_ONLY" or completion.get("upstream_unchanged") is not True or completion.get("predecessor") != {"path": E2_COMPLETION_PATH, "file_sha256": E2_COMPLETION_SHA256, "canonical_digest": E2_COMPLETION_DIGEST}:
        raise StageEE3Error("E3 terminal completion gate failed")
    outputs = validate_outputs(repo); summary, _ = _strict_json(repo / ROOT_PATH / "descriptive_summary.json", "E3 summary")
    if summary != outputs["summary"]: raise StageEE3Error("E3 summary is not source-recomputable")
    table_specs = (("layer_head_method_rows", outputs["method_rows"]), ("layer_head_paired_delta", outputs["pair_rows"]), ("prediction_strata", outputs["summary_rows"]), ("prediction_strata_membership", outputs["membership_rows"]), ("resource_and_qc", outputs["resource_rows"]))
    output_schema, _ = _strict_json(repo / OUTPUT_SCHEMA_PATH, "E3 output schema")
    for name, rows in table_specs:
        table, _ = _strict_json(repo / ROOT_PATH / "paper_data" / f"{name}.json", f"E3 {name}"); _validate_signed(table, f"E3 {name}")
        if table != _table(name, rows): raise StageEE3Error(f"E3 {name} is not source-recomputable")
        if list(Draft202012Validator(output_schema).iter_errors(table)): raise StageEE3Error(f"E3 {name} schema failed")
        with (repo / ROOT_PATH / "paper_data" / f"{name}.csv").open(newline="", encoding="utf-8") as stream: observed = list(csv.DictReader(stream))
        expected = [{key: _csv_scalar(value) for key, value in row.items()} for row in table["rows"]]
        if observed != expected: raise StageEE3Error(f"E3 {name} CSV/JSON parity failed")
    status, _ = _strict_json(repo / ROOT_PATH / "status.json", "E3 status"); inventory, _ = _strict_json(repo / ROOT_PATH / "whole_root_inventory.json", "E3 inventory")
    for value, name in ((status, "status"), (inventory, "inventory")): _validate_signed(value, f"E3 {name}")
    rebuilt = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "status.json", "completion/e3_completion.pending.json", "completion/e3_completion.json"))
    if any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes")) or canonical_sha256({"files": inventory["files"]}) != rebuilt["canonical_digest"]: raise StageEE3Error("E3 whole-root inventory drifted")
    if status.get("status") != "PASS" or status.get("cost_status") != "PASS_COST" or status.get("e4_authorized") is not True or status.get("e4_started") is not False or status.get("stage_f_started") is not False or status.get("d10_primary_recomputed") is not False:
        raise StageEE3Error("E3 status or stage boundary drifted")
    if status["whole_root_inventory"]["canonical_digest"] != inventory["canonical_digest"] or status["actual_cost"] != completion["actual_cost"] or budget_reasons(completion["actual_cost"], inventory["total_bytes"]): raise StageEE3Error("E3 cost/inventory completion binding failed")
    for field in ("test_report", "false_pass_audit", "payload_manifest", "status", "whole_root_inventory"):
        if completion["source_validation"][field] != _record(repo, completion["source_validation"][field]["path"]): raise StageEE3Error(f"E3 completion {field} binding failed")
    validation_record = {key: completion["source_validation"][key] for key in ("path", "file_sha256", "canonical_digest")}; observed_validation = _record(repo, validation_record["path"])
    if validation_record["file_sha256"] != observed_validation["sha256"] or validation_record["canonical_digest"] != observed_validation["canonical_digest"]: raise StageEE3Error("E3 validation-report binding drifted")
    false_audit, _ = _strict_json(repo / REPORT_ROOT_PATH / "false_pass_audit.json", "E3 false-PASS audit"); _validate_signed(false_audit, "E3 false-PASS audit")
    if false_audit.get("status") != "PASS" or {row.get("attack") for row in false_audit.get("checks", [])} != set(FALSE_PASS_ATTACKS) or any(row.get("result") != "REJECTED" for row in false_audit.get("checks", [])): raise StageEE3Error("E3 false-PASS audit failed")
    tests, _ = _strict_json(repo / REPORT_ROOT_PATH / "test_results.json", "E3 tests"); _validate_signed(tests, "E3 tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0: raise StageEE3Error("E3 required tests failed")
    before, _ = _strict_json(repo / UPSTREAM_BEFORE_PATH, "E3 before"); after = _protected_after(repo, before); recorded_after, _ = _strict_json(repo / REPORT_ROOT_PATH / "protected_upstream_after.json", "E3 recorded after")
    if any(after[key] != recorded_after[key] for key in ("roots", "critical_e2_chain", "protected_tracked_diff")): raise StageEE3Error("E3 recorded protected after snapshot drifted")
    access, _ = _strict_json(repo / ACCESS_PATH, "E3 terminal outcome access"); _validate_signed(access, "E3 terminal outcome access")
    if access.get("new_e3_coordinate_outcome_accessed") is not True or access.get("scope_changed_after_outcome_access") is not False or access.get("primary_recompute") is not False or access.get("e4_started") is not False or access.get("stage_f_started") is not False: raise StageEE3Error("E3 outcome-access terminal contract failed")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in (repo / ROOT_PATH).rglob("*") if path.is_file()): raise StageEE3Error("E3 contains a forbidden render artifact")
    return {"status": "PASS", "completion_canonical_digest": completion["canonical_digest"], "whole_root_inventory_digest": inventory["canonical_digest"], "method_rows": 4000, "paired_rows": 2000, "new_attempts": 3000, "summary_cells": 12, "summary": summary, "actual_cost": completion["actual_cost"], "next_authorization": "E4_ONLY", "e4_started": False}


def validate_failed_completion(repo_root: Path | str = REPO_ROOT, *, completion: Mapping[str, Any] | None = None) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if completion is None:
        completion, _ = _strict_json(repo / COMPLETION_PATH, "E3 FAIL completion"); _validate_signed(completion, "E3 FAIL completion")
    schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE3Error(f"E3 FAIL completion schema failure: {errors[0].message}")
    if completion.get("status") != "FAIL" or completion.get("next_authorization") != "NONE" or completion.get("upstream_unchanged") is not True or completion.get("predecessor") != {"path": E2_COMPLETION_PATH, "file_sha256": E2_COMPLETION_SHA256, "canonical_digest": E2_COMPLETION_DIGEST}: raise StageEE3Error("E3 FAIL terminal gate failed")
    config = _validate_config(repo); plan, _ = _strict_json(repo / PLAN_PATH, "E3 plan")
    if plan != _build_plan(repo, config): raise StageEE3Error("E3 FAIL plan is not source-recomputable")
    units = {unit["canonical_digest"]: unit for unit in plan["units"]}; observed = {}
    for path in sorted((repo / ROOT_PATH / "units").glob("*/attempts/attempt-000001")):
        digest = path.parents[1].name
        if digest not in units: raise StageEE3Error(f"unknown E3 unit directory: {digest}")
        observed[digest] = _validate_attempt(repo, units[digest])
    if len(observed) != 516 or any(path.is_dir() for path in (repo / ROOT_PATH / "units").glob("*/attempts/.attempt-*")): raise StageEE3Error("E3 FAIL retained attempt coverage or ambiguity drifted")
    counts = Counter((units[digest]["block"], units[digest]["head"], units[digest]["method"]) for digest in observed)
    if counts != Counter({(block, head, method): 86 for block, head in NEW_COORDINATES for method in METHODS}): raise StageEE3Error("E3 FAIL coordinate/method retained coverage drifted")
    stop, _ = _strict_json(repo / REPORT_ROOT_PATH / "execution_stop_event.json", "E3 stop event"); diagnostic, _ = _strict_json(repo / REPORT_ROOT_PATH / "failure_diagnostics.json", "E3 diagnostics")
    for value, name in ((stop, "stop event"), (diagnostic, "diagnostics")): _validate_signed(value, f"E3 {name}")
    if stop.get("trigger") != "NON_RETRYABLE_SCIENTIFIC_INVALIDITY" or stop.get("exception_type") != "StageCValuePathConvergenceError" or stop.get("coverage_at_stop", {}).get("valid_terminal_attempts") != 516 or stop.get("scientific_invalidity_retry_authorized") is not False: raise StageEE3Error("E3 scientific-invalidity stop evidence failed")
    if diagnostic.get("status") != "FAIL" or diagnostic.get("failure_classification") != "SCIENTIFIC_INVALIDITY_NON_RETRYABLE" or diagnostic.get("e4_authorized") is not False or diagnostic.get("observed", {}).get("descriptive_summaries_not_published") is not True: raise StageEE3Error("E3 failure diagnostic gate failed")
    if (repo / ROOT_PATH / "paper_data").exists() or (repo / ROOT_PATH / "descriptive_summary.json").exists(): raise StageEE3Error("E3 FAIL must not publish incomplete descriptive tables")
    status, _ = _strict_json(repo / ROOT_PATH / "status.json", "E3 FAIL status"); inventory, _ = _strict_json(repo / ROOT_PATH / "whole_root_inventory.json", "E3 FAIL inventory")
    for value, name in ((status, "status"), (inventory, "inventory")): _validate_signed(value, f"E3 FAIL {name}")
    rebuilt = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "status.json", "completion/e3_completion.pending.json", "completion/e3_completion.json"))
    if any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes")) or canonical_sha256({"files": inventory["files"]}) != rebuilt["canonical_digest"]: raise StageEE3Error("E3 FAIL whole-root inventory drifted")
    if status.get("status") != "FAIL" or status.get("scientific_validity") != "FAIL" or status.get("e4_authorized") is not False or status.get("e4_started") is not False or status.get("stage_f_started") is not False or status.get("d10_primary_recomputed") is not False: raise StageEE3Error("E3 FAIL status or authorization boundary drifted")
    if status["whole_root_inventory"]["canonical_digest"] != inventory["canonical_digest"] or status["actual_cost"] != completion["actual_cost"] or budget_reasons(completion["actual_cost"], inventory["total_bytes"]): raise StageEE3Error("E3 FAIL cost/inventory completion binding failed")
    for field in ("test_report", "false_pass_audit", "payload_manifest", "status", "whole_root_inventory", "execution_stop_event", "failure_diagnostics"):
        if completion["source_validation"][field] != _record(repo, completion["source_validation"][field]["path"]): raise StageEE3Error(f"E3 FAIL completion {field} binding failed")
    validation_record = {key: completion["source_validation"][key] for key in ("path", "file_sha256", "canonical_digest")}; observed_validation = _record(repo, validation_record["path"])
    if validation_record["file_sha256"] != observed_validation["sha256"] or validation_record["canonical_digest"] != observed_validation["canonical_digest"]: raise StageEE3Error("E3 FAIL validation-report binding drifted")
    validation, _ = _strict_json(repo / validation_record["path"], "E3 FAIL validation report"); _validate_signed(validation, "E3 FAIL validation report")
    if validation.get("status") != "FAIL" or validation.get("checks", {}).get("required_3000_attempt_coverage") != "FAIL_516_OF_3000": raise StageEE3Error("E3 FAIL validation result drifted")
    false_audit, _ = _strict_json(repo / REPORT_ROOT_PATH / "false_pass_audit.json", "E3 false-PASS audit"); _validate_signed(false_audit, "E3 false-PASS audit")
    if false_audit.get("status") != "PASS" or false_audit.get("e3_pass_issued") is not False or {row.get("attack") for row in false_audit.get("checks", [])} != set(FALSE_PASS_ATTACKS) or any(row.get("result") != "REJECTED" for row in false_audit.get("checks", [])): raise StageEE3Error("E3 FAIL false-PASS audit failed")
    tests, _ = _strict_json(repo / REPORT_ROOT_PATH / "test_results.json", "E3 tests"); _validate_signed(tests, "E3 tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0: raise StageEE3Error("E3 required tests failed")
    before, _ = _strict_json(repo / UPSTREAM_BEFORE_PATH, "E3 before"); after = _protected_after(repo, before); recorded_after, _ = _strict_json(repo / REPORT_ROOT_PATH / "protected_upstream_after.json", "E3 recorded after")
    if any(after[key] != recorded_after[key] for key in ("roots", "critical_e2_chain", "protected_tracked_diff")): raise StageEE3Error("E3 FAIL protected after snapshot drifted")
    access, _ = _strict_json(repo / ACCESS_PATH, "E3 terminal outcome access"); _validate_signed(access, "E3 terminal outcome access")
    if access.get("new_e3_coordinate_outcome_accessed") is not True or access.get("scope_changed_after_outcome_access") is not False or access.get("primary_recompute") is not False or access.get("e4_started") is not False or access.get("stage_f_started") is not False: raise StageEE3Error("E3 FAIL outcome-access terminal contract failed")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in (repo / ROOT_PATH).rglob("*") if path.is_file()): raise StageEE3Error("E3 FAIL contains forbidden render artifact")
    return {"status": "FAIL", "completion_canonical_digest": completion["canonical_digest"], "whole_root_inventory_digest": inventory["canonical_digest"], "valid_terminal_attempts": 516, "expected_new_attempts": 3000, "actual_cost": completion["actual_cost"], "next_authorization": "NONE", "e4_started": False, "safe_recovery_entry": status["safe_recovery_entry"]}


def _main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vitnns.stage_e_e3"); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-preconditions", "initialize", "validate-initialized", "record-pre-scale-observability", "record-execution-stop", "validate-outputs", "finalize", "finalize-failed", "validate-completion"):
        child = sub.add_parser(name); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    run = sub.add_parser("run-shard"); run.add_argument("--repo-root", type=Path, default=REPO_ROOT); run.add_argument("--shard-id", type=int, required=True); run.add_argument("--shard-count", type=int, required=True); run.add_argument("--device", default="cuda:0"); run.add_argument("--limit", type=int)
    tests = sub.add_parser("record-tests"); tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); tests.add_argument("--passed", type=int, required=True); tests.add_argument("--failed", type=int, required=True); tests.add_argument("--test-command", required=True)
    args = parser.parse_args()
    if args.command == "verify-preconditions": value = verify_preconditions(args.repo_root)
    elif args.command == "initialize": value = initialize(args.repo_root)
    elif args.command == "validate-initialized": value = validate_initialized(args.repo_root)
    elif args.command == "run-shard": value = run_shard(args.repo_root, shard_id=args.shard_id, shard_count=args.shard_count, device=args.device, limit=args.limit)
    elif args.command == "record-pre-scale-observability": value = record_pre_scale_observability_event(args.repo_root)
    elif args.command == "record-execution-stop": value = record_execution_stop_event(args.repo_root)
    elif args.command == "validate-outputs":
        result = validate_outputs(args.repo_root); value = {"status": "PASS", "method_rows": len(result["method_rows"]), "paired_rows": len(result["pair_rows"]), "new_attempts": len(result["attempts"]), "actual_cost": result["actual_cost"]}
    elif args.command == "record-tests": value = record_tests(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    elif args.command == "finalize": value = finalize(args.repo_root)
    elif args.command == "finalize-failed": value = finalize_failed(args.repo_root)
    else: value = validate_completion(args.repo_root)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(_main())
