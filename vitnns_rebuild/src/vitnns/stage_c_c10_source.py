"""Read-only loader for the sealed, formal Stage C9 evidence snapshot.

This is deliberately a C10 boundary adapter, not a second C9 finalizer.  The
producer checkout is verified separately before C10 starts; this module binds
the evidence to that producer commit without consulting (or impersonating)
the live checkout.  It performs no model, image, CUDA, or artifact writes.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from . import stage_c_c9 as _c9
from .stage_c_artifacts import validate_execution_unit
from .stage_c_c10 import StageC10IntegrityError
from .stage_c_plan import ExecutionPlan, StageCEstimate
from .stage_c_provenance import canonical_sha256


C9_PRODUCER_COMMIT = "0d88fbb8b71c604d3dab94db19725e01b7519ae5"
C9_PRODUCER_TREE = "5ca1ee721420f114fbd2b74d5f71294f69a34610"
C9_EXECUTION_COMMIT = "c66e5224e87924253f10a9f5fead006fcc5b5ac7"
C9_EXECUTION_TREE = "83da3061d0540a12df29907f2eb6d214104b329a"
C9_SOURCE_SUBTREE_TREE = "82b4481e73be6ae36366aa71ce178077d12e38c8"
C9_AUTHORITY_PATH = "config/recovery_c9_launch_record_retry_20260903.json"
C9_AUTHORITY_RECORD_FILE_SHA256 = "8e466718d2efa1205bcce2906442fa2ff1efac6b957f6f8693f269c66771a9cc"
C9_AUTHORITY_RECORD_CANONICAL_DIGEST = "d2b24e8566e63ead545d81c1574b608e20aae41b1b151657af858706f78b12f3"
C9_REPORT_FILE_SHA256 = "ad164a87aa6dedba461e1b19d639d22d955a675d75e0c6e257af4d5846682763"
C9_REPORT_CANONICAL_DIGEST = "959bc874b04b007f487d05734cf5b85b1eaef75bb8c2ff2ded222c7cf9083248"
C9_EXECUTION_PLAN_FILE_SHA256 = "d96c9ca227dbec2aebb72942eb69bdae507639ab043161ce096e3d15643cc38c"
C9_EXECUTION_PLAN_CANONICAL_DIGEST = "d1c0a75b22f82f580f6a2ddbc73f399159d3ca95fc6ce42518dc8382922d1eb1"
C9_RUN_SPEC_FILE_SHA256 = "6f36f2e1064c0377e337f88fefe7e5184f1fe24ac58e03f85ca9262bfd62f74f"
C9_RUN_SPEC_CANONICAL_DIGEST = "935a7e006e6a88dc7cd583b967fd02a057029942ad7e1087195a5deaee4456b8"
C9_CONFIG_FILE_SHA256 = "0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669"
C9_CONFIG_CANONICAL_DIGEST = "a0a9312f607d80e134c32c35b1cf120c65af9ba8d3f48e28026291ed20afab30"
C9_PREFLIGHT_FILE_SHA256 = "299b6318ce9c04c5ccd4b233e27a57fb1c8ca3d6c1790043dbbf4d6e3e2c44db"
C9_PREFLIGHT_CANONICAL_DIGEST = "b1c258eceae20e0c9023df607ee28e4234a5e327660329bc92c0ea6b9714f8ec"
C9_AUTHORIZATION_FILE_SHA256 = "42450719c3474ff24098542ef09e46c4daf2cded9a3d44b899ad531942be7102"
C9_AUTHORIZATION_CANONICAL_DIGEST = "f6aa3e3af953277945a522a9bddec4d3b795309274ff83a0c85d39b4506ca55a"
C9_FAILURE_TABLE_FILE_SHA256 = "335c04b031ecc19d5af4c39b5593b348efa8d4aef33b9ef007c860b53a7b4c22"
C9_BATCH_STATUS_FILE_SHA256 = "cb14a44c91f0f1d059ace82a99ef391e29924b65f36ee7c457efd2c7092d936a"
C9_C8_HANDOFF_CANONICAL_DIGEST = "14f96a9693fd11acc482da6dc658676fcdf0788afda4d009e3da2b345dfbc638"
C9_POST_RUN_BUDGET_CANONICAL_DIGEST = "53fc5fc4e47a966b5a767463df2f9a096463f34e2eb3f03bda0a84692df475c5"
C9_STATUS_INDEX_DIGEST = "031e25cfd57cf439c9e8615e4dbd1c0b7f7b43e2e599af3d5236b67f85d8363b"
C9_MANIFEST_INDEX_DIGEST = "9700a232cbf45ec45f98078330e218aa6e6674a1387da5e11e6ad05cba0ec94f"
C9_ARTIFACT_INVENTORY_DIGEST = "17c71f0c692ad52fe09c4dd2011307e56b17026ca341e2dc708d488e4b5449e4"
C9_ROOT_INVENTORY_DIGEST = "6ab8ee024ce7d5bfa3c6a408ec118a9eeac7960d046d7001d8019fb8ab6650c8"
C9_ROOT_FILE_COUNT = 2_705
C9_ROOT_TOTAL_BYTES = 225_985_972
C9_ARTIFACT_COUNT = 2_056
C9_L0_COUNT = 640
C9_L1_COUNT = 1_280
C9_L2_COUNT = 136
C9_L3_COUNT = 0

FROZEN_C9_HASHES = MappingProxyType({
    "producer_commit": C9_PRODUCER_COMMIT,
    "producer_tree": C9_PRODUCER_TREE,
    "report_file_sha256": C9_REPORT_FILE_SHA256,
    "report_canonical_digest": C9_REPORT_CANONICAL_DIGEST,
    "execution_plan_file_sha256": C9_EXECUTION_PLAN_FILE_SHA256,
    "execution_plan_canonical_digest": C9_EXECUTION_PLAN_CANONICAL_DIGEST,
    "run_spec_file_sha256": C9_RUN_SPEC_FILE_SHA256,
    "run_spec_canonical_digest": C9_RUN_SPEC_CANONICAL_DIGEST,
    "config_file_sha256": C9_CONFIG_FILE_SHA256,
    "config_canonical_digest": C9_CONFIG_CANONICAL_DIGEST,
    "preflight_file_sha256": C9_PREFLIGHT_FILE_SHA256,
    "preflight_canonical_digest": C9_PREFLIGHT_CANONICAL_DIGEST,
    "authorization_file_sha256": C9_AUTHORIZATION_FILE_SHA256,
    "authorization_canonical_digest": C9_AUTHORIZATION_CANONICAL_DIGEST,
    "failure_table_file_sha256": C9_FAILURE_TABLE_FILE_SHA256,
    "batch_status_file_sha256": C9_BATCH_STATUS_FILE_SHA256,
    "c8_handoff_canonical_digest": C9_C8_HANDOFF_CANONICAL_DIGEST,
    "post_run_budget_canonical_digest": C9_POST_RUN_BUDGET_CANONICAL_DIGEST,
    "status_index_digest": C9_STATUS_INDEX_DIGEST,
    "manifest_index_digest": C9_MANIFEST_INDEX_DIGEST,
    "artifact_inventory_digest": C9_ARTIFACT_INVENTORY_DIGEST,
    "root_inventory_digest": C9_ROOT_INVENTORY_DIGEST,
})

_CONTROL_HASHES = MappingProxyType({
    "execution_plan.json": C9_EXECUTION_PLAN_FILE_SHA256,
    "preflight.json": C9_PREFLIGHT_FILE_SHA256,
    "c9_launch_authorization.json": C9_AUTHORIZATION_FILE_SHA256,
    "failure_table.json": C9_FAILURE_TABLE_FILE_SHA256,
    "batch_status.json": C9_BATCH_STATUS_FILE_SHA256,
})
_ALLOWED_ROOT_ENTRIES = frozenset({
    *_CONTROL_HASHES, _c9.FINAL_REPORT_FILENAME, "units",
    ".c9-checkpoint.lock", ".c9-lifecycle.lock",
    ".c9-artifact-activity.lock",
})
_EXPECTED_CODE = MappingProxyType({
    "commit": C9_EXECUTION_COMMIT, "dirty": False,
})
_L1_CLEAN_PATH = "L1/clean.npy"


@dataclass(frozen=True, slots=True)
class C9Snapshot(Mapping[str, Any]):
    """Fully validated C9 source material ready for C10 aggregation."""

    records: tuple[Mapping[str, Any], ...]
    report: Mapping[str, Any]
    bindings: Mapping[str, Any]
    hash_audit: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        if key not in {"records", "report", "bindings", "hash_audit"}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("records", "report", "bindings", "hash_audit"))

    def __len__(self) -> int:
        return 4


def _fail(message: str, error: BaseException | None = None) -> None:
    if error is None:
        raise StageC10IntegrityError(message)
    raise StageC10IntegrityError(message) from error


def _plain(value: Any) -> Any:
    method = getattr(value, "to_dict", None)
    value = method() if callable(method) else value
    return json.loads(json.dumps(value, allow_nan=False))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_read_bytes(path: Path, context: str) -> bytes:
    """Read one regular single-link file once through a no-follow fd."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("not a regular single-link file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if identity_before != identity_after:
                raise ValueError("file changed while being read")
            raw = b"".join(chunks)
            if len(raw) != before.st_size:
                raise ValueError("file size changed while being read")
            return raw
        finally:
            os.close(descriptor)
    except (OSError, ValueError) as error:
        _fail(f"{context} is missing, unsafe, or unstable: {path}", error)


def _strict_json_bytes(
    raw: bytes, path: Path, *, canonical: bool = False,
) -> dict[str, Any]:
    try:
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in items:
                if key in value:
                    raise ValueError(f"duplicate key {key!r}")
                value[key] = item
            return value

        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
        if not isinstance(value, dict):
            raise ValueError("top-level value is not an object")
        if canonical:
            expected = (
                json.dumps(
                    value, indent=2, sort_keys=True, ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            ).encode("utf-8")
            if raw != expected:
                raise ValueError("bytes are not canonical Stage C JSON")
        return value
    except StageC10IntegrityError:
        raise
    except Exception as error:
        _fail(f"invalid C9 JSON document {path}: {error}", error)


def _strict_json(path: Path, *, canonical: bool = False) -> dict[str, Any]:
    raw = _safe_read_bytes(path, f"JSON source {path.name}")
    return _strict_json_bytes(raw, path, canonical=canonical)


def _safe_root(value: Path | str) -> Path:
    root = Path(os.path.abspath(os.fspath(value)))
    current = Path(root.anchor)
    try:
        for component in root.parts[1:]:
            current /= component
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                _fail(f"C9 root contains a symlink component: {current}")
        if not stat.S_ISDIR(root.stat().st_mode):
            _fail("C9 root is not a directory")
    except (OSError, ValueError) as error:
        _fail(f"C9 root is missing or unsafe: {root}", error)
    return root


def _safe_regular(path: Path, context: str) -> Path:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("not a regular single-link file")
        if info.st_nlink != 1:
            raise ValueError("has multiple hard links")
    except (OSError, ValueError) as error:
        _fail(f"{context} is missing or unsafe: {path}", error)
    return path


def _inventory(root: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total = 0
    try:
        for directory, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            base = Path(directory)
            for name in [*directory_names, *file_names]:
                path = base / name
                info = path.lstat()
                relative = path.relative_to(root).as_posix()
                if stat.S_ISLNK(info.st_mode):
                    _fail(f"forbidden C9 symlink: {relative}")
                if not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode):
                    _fail(f"non-regular C9 entry: {relative}")
                if stat.S_ISREG(info.st_mode):
                    if info.st_nlink != 1:
                        _fail(f"C9 file has multiple hard links: {relative}")
                    records.append({
                        "path": relative, "bytes": info.st_size,
                        "sha256": _file_sha256(path),
                    })
                    total += info.st_size
    except OSError as error:
        _fail(f"cannot inventory C9 root {root}: {error}", error)
    records.sort(key=lambda item: item["path"])
    return records, total


def _digest_records(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256({"records": list(records)})


def _frozen_hash_audit(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed_controls = {
        name: _file_sha256(_safe_regular(root / name, f"C9 control {name}"))
        for name in _CONTROL_HASHES
    }
    if observed_controls != dict(_CONTROL_HASHES):
        drifted = sorted(
            name for name, expected in _CONTROL_HASHES.items()
            if observed_controls.get(name) != expected
        )
        _fail(f"C9 frozen control hash drifted: {', '.join(drifted)}")
    report_path = _safe_regular(
        root / _c9.FINAL_REPORT_FILENAME, "C9 final report"
    )
    if _file_sha256(report_path) != C9_REPORT_FILE_SHA256:
        _fail("C9 final report file hash drifted")
    entries = {path.name for path in root.iterdir()}
    if entries != _ALLOWED_ROOT_ENTRIES:
        _fail(
            "C9 root entry set drifted; "
            f"missing={sorted(_ALLOWED_ROOT_ENTRIES - entries)}, "
            f"extra={sorted(entries - _ALLOWED_ROOT_ENTRIES)}"
        )
    inventory, total = _inventory(root)
    digest = _digest_records(inventory)
    if (len(inventory), total, digest) != (
        C9_ROOT_FILE_COUNT, C9_ROOT_TOTAL_BYTES, C9_ROOT_INVENTORY_DIGEST,
    ):
        _fail(
            "C9 whole-root inventory drifted: "
            f"files={len(inventory)}, bytes={total}, digest={digest}"
        )
    return ({
        "control_file_sha256": observed_controls,
        "report_file_sha256": C9_REPORT_FILE_SHA256,
        "root_file_count": len(inventory),
        "root_total_bytes": total,
        "root_inventory_digest": digest,
    }, inventory)


def audit_c9_snapshot_hashes(c9_root: Path | str) -> dict[str, Any]:
    """Recompute the sealed report, controls, and whole-root inventory."""

    audit, _inventory_records = _frozen_hash_audit(_safe_root(c9_root))
    return audit


def _validate_controls(
    root: Path, plan: ExecutionPlan, config: Any,
) -> tuple[StageCEstimate, dict[str, Any], list[dict[str, Any]]]:
    try:
        scope = _c9.validate_c9_execution_plan(plan, config)
        if not scope.get("valid"):
            _fail("C10 was not given the exact frozen C9 execution plan")
        persisted_plan = _strict_json(root / "execution_plan.json")
        if persisted_plan != plan.to_dict():
            _fail("C9 persisted execution plan differs from the supplied plan")
        if plan.canonical_digest != C9_EXECUTION_PLAN_CANONICAL_DIGEST:
            _fail("C9 execution-plan canonical digest drifted")
        if canonical_sha256(config.to_dict()) != C9_CONFIG_CANONICAL_DIGEST:
            _fail("Stage C config differs from the frozen C9 config")
        if canonical_sha256(plan.run_spec.to_dict()) != C9_RUN_SPEC_CANONICAL_DIGEST:
            _fail("C9 RunSpec canonical digest drifted")
        estimate = StageCEstimate.from_dict(_strict_json(root / "preflight.json"))
        _c9._validate_c9_preflight(estimate, plan, require_issuer=False)
        if estimate.canonical_digest != C9_PREFLIGHT_CANONICAL_DIGEST:
            _fail("C9 preflight canonical digest drifted")
        authorization = _strict_json(root / _c9.AUTHORIZATION_FILENAME)
        _c9.validate_c9_launch_authorization(
            authorization, plan=plan, estimate=estimate, output_root=root,
        )
        if (
            authorization.get("canonical_digest")
            != C9_AUTHORIZATION_CANONICAL_DIGEST
            or authorization.get("c8_handoff_sha256")
            != C9_C8_HANDOFF_CANONICAL_DIGEST
            or authorization.get("code") != dict(_EXPECTED_CODE)
        ):
            _fail("C9 authorization frozen producer binding drifted")
        failure_table = _strict_json(root / "failure_table.json")
        rows = failure_table.get("rows")
        if not isinstance(rows, list) or len(rows) != _c9.C9_EXECUTION_UNITS:
            _fail("C9 failure table does not contain exactly 320 rows")
        checkpoint_rows = _c9._read_checkpoint_rows(root, plan)
        _checked_rows, batch = _c9._validate_checkpoint_controls(
            root, plan, allowed_lifecycle=("finalized",),
        )
        if not (
            batch.get("orchestration_complete") is True
            and batch.get("successful") is True
            and batch.get("active_claim_count") == 0
            and batch.get("validation_errors") == []
        ):
            _fail("C9 finalized batch status is not a clean PASS")
        layout_errors = _c9._validate_closed_layout(
            root, plan, require_all_units=True
        )
        if layout_errors:
            _fail("C9 closed layout is invalid: " + "; ".join(layout_errors))
        return estimate, authorization, checkpoint_rows
    except StageC10IntegrityError:
        raise
    except Exception as error:
        _fail(f"C9 control validation failed: {error}", error)


def _load_l1_clean(
    attempt: Path, manifest: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    matches = [
        item for item in manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("path") == _L1_CLEAN_PATH
    ]
    if len(matches) != 1:
        _fail("C9 manifest must bind exactly one L1/clean.npy artifact")
    binding = dict(matches[0])
    if (
        binding.get("level") != "L1"
        or binding.get("kind") != "metric_ready"
        or binding.get("format") != "npy"
        or binding.get("dtype") != "float32"
        or binding.get("shape") != [14, 14]
        or set(binding) != {
            "path", "level", "kind", "bytes", "sha256",
            "dtype", "shape", "format",
        }
    ):
        _fail("C9 L1/clean.npy manifest binding drifted")
    path = _safe_regular(attempt / _L1_CLEAN_PATH, "C9 L1 clean array")
    raw = _safe_read_bytes(path, "C9 L1 clean array")
    if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        _fail("C9 L1/clean.npy bytes differ from its manifest")
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as error:
        _fail(f"C9 L1/clean.npy cannot be loaded safely: {error}", error)
    if array.dtype != np.dtype("float32") or array.shape != (14, 14):
        _fail("C9 L1/clean.npy dtype/shape drifted")
    if not bool(np.isfinite(array).all()):
        _fail("C9 L1/clean.npy contains non-finite values")
    array = np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(14, 14)
    return array, binding


def load_c9_snapshot(
    c9_root: Path | str, plan: ExecutionPlan, config: Any,
) -> C9Snapshot:
    """Validate and load the exact sealed C9 snapshot for C10.

    The caller must supply the exact :class:`ExecutionPlan` and frozen config.
    Only ``L0/result.json``, ``L0/provenance.json`` and the manifest-verified
    ``L1/clean.npy`` payload are returned or used as C10 metric inputs.  The
    existing artifact validator may inspect retained L2 structure/hashes; L2
    and L3 never enter C10 numerical calculations.
    """

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    if not hasattr(config, "to_dict"):
        raise TypeError("config must be a loaded StageCConfig")
    # Validate the tracked recovery authority without requiring the live C10
    # checkout to impersonate the sealed C9 execution checkout.
    try:
        import subprocess
        producer_tree = subprocess.run(
            ["git", "rev-parse", f"{C9_PRODUCER_COMMIT}^{{tree}}"],
            cwd=_c9.REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        execution_tree = subprocess.run(
            ["git", "rev-parse", f"{C9_EXECUTION_COMMIT}^{{tree}}"],
            cwd=_c9.REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        prefix = subprocess.run(
            ["git", "rev-parse", "--show-prefix"], cwd=_c9.REPO_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        producer_source_tree = subprocess.run(
            ["git", "rev-parse", f"{C9_PRODUCER_COMMIT}:{prefix}src/vitnns"],
            cwd=_c9.REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception as error:
        _fail(f"C9 recovery commit attestation cannot be verified: {error}", error)
    if (
        producer_tree != C9_PRODUCER_TREE
        or execution_tree != C9_EXECUTION_TREE
        or producer_source_tree != C9_SOURCE_SUBTREE_TREE
    ):
        _fail("C9 recovery producer, execution, or source tree drifted")
    try:
        from .recovery_c9_launch import load_recovery_c9_launch_record
        recovery_authority = load_recovery_c9_launch_record(
            _c9.REPO_ROOT / C9_AUTHORITY_PATH, repo_root=_c9.REPO_ROOT,
            require_tracked=True, full_c8_validation=False,
            validate_runtime=False,
        )
    except Exception as error:
        _fail(f"C9 recovery authority cannot be validated: {error}", error)
    if (
        recovery_authority.file_sha256 != C9_AUTHORITY_RECORD_FILE_SHA256
        or recovery_authority.canonical_digest
        != C9_AUTHORITY_RECORD_CANONICAL_DIGEST
        or recovery_authority.producer_commit != C9_PRODUCER_COMMIT
        or recovery_authority.launch_commit != C9_EXECUTION_COMMIT
    ):
        _fail("C9 recovery authority identity drifted")
    root = _safe_root(c9_root)
    hash_audit, _root_inventory = _frozen_hash_audit(root)
    estimate, authorization, checkpoint_rows = _validate_controls(
        root, plan, config
    )
    report = _strict_json(
        root / _c9.FINAL_REPORT_FILENAME, canonical=True
    )
    try:
        _c9.validate_c9_final_report(
            report, plan=plan, output_root=root, estimate=estimate,
            authorization=authorization,
        )
    except Exception as error:
        _fail(f"C9 final report validation failed: {error}", error)
    if (
        report.get("canonical_digest") != C9_REPORT_CANONICAL_DIGEST
        or report.get("status") != "PASS"
        or report.get("integrity_valid") is not True
        or report.get("orchestration_complete") is not True
        or report.get("successful") is not True
        or report.get("terminal_unit_count") != 320
        or report.get("state_counts") != {"succeeded": 320}
        or report.get("primary_holdout_valid_count") != 15
        or report.get("primary_holdout_complete") is not True
        or report.get("primary_holdout_failures") != []
        or report.get("errors") != []
        or report.get("post_c9_budget", {}).get("canonical_digest")
        != C9_POST_RUN_BUDGET_CANONICAL_DIGEST
    ):
        _fail("C9 final report is not the frozen 320/320 PASS report")
    if report.get("rows") != checkpoint_rows:
        _fail("C9 report rows differ from the finalized failure table")

    expected_bindings = {
        "configuration": {
            "execution_plan_sha256": plan.canonical_digest,
            "run_spec_sha256": canonical_sha256(plan.run_spec.to_dict()),
        },
        "resources": {
            "preflight": {
                "plan_digest": plan.canonical_digest,
                "canonical_digest": estimate.canonical_digest,
            }
        },
    }
    caps = dict(estimate.caps["effective"])
    records: list[dict[str, Any]] = []
    status_index: list[dict[str, Any]] = []
    manifest_index: list[dict[str, Any]] = []
    artifact_index: list[dict[str, Any]] = []
    level_counts = {level: 0 for level in ("L0", "L1", "L2", "L3")}
    seen_attempts: set[str] = set()

    for index, (unit, row) in enumerate(zip(plan.units, report["rows"])):
        digest = unit.key.canonical_digest
        if (
            row.get("state") != "succeeded"
            or row.get("terminal") is not True
            or row.get("complete") is not True
            or row.get("failure") is not None
        ):
            _fail(f"C9 row {index} is failed, missing, incomplete, or tampered")
        selected = row.get("selected_attempt")
        expected = root / "units" / digest / "attempts" / "attempt-000001"
        if selected != str(expected) or selected in seen_attempts:
            _fail(f"C9 row {index} selected_attempt is not exact and unique")
        seen_attempts.add(selected)
        attempt = Path(selected)
        try:
            checked = validate_execution_unit(
                attempt, expected_unit_identity=unit.key.to_dict(),
                expected_unit_digest=digest,
                expected_provenance_bindings=expected_bindings,
                observed_git_state=dict(_EXPECTED_CODE), caps=caps,
                require_runner_telemetry=True,
                trusted_recovery_context=recovery_authority.repair_record,
            )
        except Exception as error:
            _fail(f"C9 unit {digest} validation raised: {error}", error)
        if not checked.resumable:
            _fail(
                f"C9 unit {digest} is not a valid complete attempt: "
                + "; ".join(checked.errors)
            )
        if (
            checked.l0_l1_bytes != row.get("l0_l1_bytes")
            or checked.l2_bytes != row.get("l2_bytes")
            or checked.total_bytes != row.get("total_bytes")
        ):
            _fail(f"C9 row {index} artifact byte accounting drifted")
        status_path = _safe_regular(attempt / "status.json", "C9 unit status")
        manifest_path = _safe_regular(
            attempt / "manifest.json", "C9 unit manifest"
        )
        result_path = _safe_regular(
            attempt / "L0/result.json", "C9 unit result"
        )
        provenance_path = _safe_regular(
            attempt / "L0/provenance.json", "C9 unit provenance"
        )
        status_raw = _safe_read_bytes(status_path, "C9 unit status")
        manifest_raw = _safe_read_bytes(manifest_path, "C9 unit manifest")
        status_hash = hashlib.sha256(status_raw).hexdigest()
        manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
        if (
            status_hash != row.get("selected_status_sha256")
            or manifest_hash != checked.manifest_sha256
        ):
            _fail(f"C9 row {index} status/manifest hash binding drifted")
        status = _strict_json_bytes(status_raw, status_path, canonical=True)
        manifest = _strict_json_bytes(manifest_raw, manifest_path, canonical=True)
        if status.get("manifest_sha256") != manifest_hash:
            _fail(f"C9 unit {digest} status does not bind manifest bytes")
        result_raw = _safe_read_bytes(result_path, "C9 unit result")
        provenance_raw = _safe_read_bytes(provenance_path, "C9 unit provenance")
        result = _strict_json_bytes(result_raw, result_path, canonical=True)
        provenance = _strict_json_bytes(provenance_raw, provenance_path, canonical=True)
        provenance_configuration = provenance.get("bindings", {}).get(
            "configuration"
        )
        if (
            not isinstance(provenance_configuration, Mapping)
            or provenance_configuration.get("sha256") != C9_CONFIG_FILE_SHA256
            or provenance_configuration.get("run_spec_sha256")
            != C9_RUN_SPEC_CANONICAL_DIGEST
            or provenance_configuration.get("execution_plan_sha256")
            != C9_EXECUTION_PLAN_CANONICAL_DIGEST
        ):
            _fail(f"C9 unit {digest} frozen config/RunSpec provenance drifted")
        inventory_by_path = {
            item.get("path"): item for item in manifest.get("artifacts", [])
            if isinstance(item, Mapping)
        }
        result_hash = hashlib.sha256(result_raw).hexdigest()
        provenance_hash = hashlib.sha256(provenance_raw).hexdigest()
        if (
            inventory_by_path.get("L0/result.json", {}).get("sha256")
            != result_hash
            or inventory_by_path.get("L0/provenance.json", {}).get("sha256")
            != provenance_hash
            or manifest.get("provenance_sha256") != provenance_hash
        ):
            _fail(f"C9 unit {digest} L0 manifest binding drifted")
        if (
            result.get("unit_identity") != unit.key.to_dict()
            or result.get("state") != "completed"
            or result.get("failure") is not None
            or provenance.get("unit_identity") != unit.key.to_dict()
        ):
            _fail(f"C9 unit {digest} L0 identity/state drifted")
        scalars = result.get("scalars")
        if not isinstance(scalars, Mapping):
            _fail(f"C9 unit {digest} result has no scalar mapping")
        clean, clean_binding = _load_l1_clean(attempt, manifest)
        artifacts = manifest.get("artifacts", [])
        for item in artifacts:
            level = item.get("level")
            if level not in level_counts:
                _fail(f"C9 unit {digest} has an unknown artifact level")
            level_counts[level] += 1
            artifact_index.append({
                "unit_identity_sha256": digest,
                "path": item["path"], "level": level,
                "bytes": item["bytes"], "sha256": item["sha256"],
            })
        status_index.append({
            "unit_identity_sha256": digest, "status_sha256": status_hash,
        })
        manifest_index.append({
            "unit_identity_sha256": digest,
            "manifest_sha256": manifest_hash,
        })
        records.append({
            "plan_index": index, "unit_key": unit.key.to_dict(),
            "state": "succeeded", "trusted_by_batch_validator": True,
            "attempt_path": attempt.relative_to(root).as_posix(),
            "hashes": {
                "status_sha256": status_hash,
                "selected_status_sha256": row["selected_status_sha256"],
                "manifest_sha256": manifest_hash,
                "provenance_sha256": provenance_hash,
                "result_sha256": result_hash,
                "unit_identity_sha256": digest,
            },
            "scalars": _plain(scalars), "provenance": provenance,
            "resources": _plain(result.get("resources", {})),
            "runner_telemetry": _plain(status.get("runner_telemetry", {})),
            "artifact_inventory": _plain(artifacts),
            "work_item_counts": _plain(unit.work_item_counts),
            "audit": unit.sample.audit, "validation_errors": [],
            "l1_clean_array": clean,
            "l1_clean_binding": _plain(clean_binding),
        })

    observed_indexes = {
        "status_index_digest": _digest_records(status_index),
        "manifest_index_digest": _digest_records(manifest_index),
        "artifact_inventory_digest": _digest_records(artifact_index),
    }
    expected_indexes = {
        "status_index_digest": C9_STATUS_INDEX_DIGEST,
        "manifest_index_digest": C9_MANIFEST_INDEX_DIGEST,
        "artifact_inventory_digest": C9_ARTIFACT_INVENTORY_DIGEST,
    }
    if observed_indexes != expected_indexes:
        _fail(f"C9 ordered hash index drifted: {observed_indexes}")
    if len(artifact_index) != C9_ARTIFACT_COUNT or level_counts != {
        "L0": C9_L0_COUNT, "L1": C9_L1_COUNT,
        "L2": C9_L2_COUNT, "L3": C9_L3_COUNT,
    }:
        _fail("C9 artifact inventory counts drifted")

    bindings = {
        **_plain(report["bindings"]),
        "producer_commit": C9_PRODUCER_COMMIT,
        "authority_execution_commit": C9_EXECUTION_COMMIT,
        "recovery_authority_path": C9_AUTHORITY_PATH,
        "recovery_authority_file_sha256": C9_AUTHORITY_RECORD_FILE_SHA256,
        "recovery_authority_canonical_digest": (
            C9_AUTHORITY_RECORD_CANONICAL_DIGEST
        ),
        "report_file_sha256": C9_REPORT_FILE_SHA256,
        "report_canonical_digest": C9_REPORT_CANONICAL_DIGEST,
        "status_index_digest": C9_STATUS_INDEX_DIGEST,
        "manifest_index_digest": C9_MANIFEST_INDEX_DIGEST,
        "artifact_inventory_digest": C9_ARTIFACT_INVENTORY_DIGEST,
        "root_inventory_digest": C9_ROOT_INVENTORY_DIGEST,
    }
    hash_audit.update(observed_indexes)
    hash_audit["artifact_count"] = len(artifact_index)
    hash_audit["artifact_level_counts"] = level_counts
    hash_audit["execution_unit_count"] = len(records)
    final_hash_audit, _final_inventory = _frozen_hash_audit(root)
    if final_hash_audit != {
        key: hash_audit[key] for key in final_hash_audit
    }:
        _fail("C9 snapshot changed while C10 was loading it")
    # Deep-freeze JSON containers.  L1 arrays are backed by immutable bytes,
    # so their write flag cannot be re-enabled by a downstream caller.
    return C9Snapshot(
        records=tuple(_freeze(item) for item in records),
        report=_freeze(copy.deepcopy(report)), bindings=_freeze(bindings),
        hash_audit=_freeze(hash_audit),
    )


def validate_c9_snapshot(
    c9_root: Path | str, plan: ExecutionPlan, config: Any,
) -> C9Snapshot:
    """Compatibility spelling for validating and loading the snapshot."""

    return load_c9_snapshot(c9_root, plan, config)


load_c9_source = load_c9_snapshot


__all__ = [
    "C9Snapshot", "C9_PRODUCER_COMMIT", "C9_PRODUCER_TREE",
    "C9_EXECUTION_COMMIT", "C9_EXECUTION_TREE", "C9_SOURCE_SUBTREE_TREE",
    "C9_AUTHORITY_PATH", "C9_AUTHORITY_RECORD_FILE_SHA256",
    "C9_AUTHORITY_RECORD_CANONICAL_DIGEST", "C9_REPORT_FILE_SHA256",
    "C9_REPORT_CANONICAL_DIGEST", "C9_EXECUTION_PLAN_FILE_SHA256",
    "C9_EXECUTION_PLAN_CANONICAL_DIGEST", "C9_RUN_SPEC_FILE_SHA256",
    "C9_RUN_SPEC_CANONICAL_DIGEST", "C9_PREFLIGHT_FILE_SHA256",
    "C9_CONFIG_FILE_SHA256", "C9_CONFIG_CANONICAL_DIGEST",
    "C9_PREFLIGHT_CANONICAL_DIGEST", "C9_AUTHORIZATION_FILE_SHA256",
    "C9_AUTHORIZATION_CANONICAL_DIGEST",
    "C9_FAILURE_TABLE_FILE_SHA256", "C9_BATCH_STATUS_FILE_SHA256",
    "C9_C8_HANDOFF_CANONICAL_DIGEST",
    "C9_POST_RUN_BUDGET_CANONICAL_DIGEST", "C9_STATUS_INDEX_DIGEST",
    "C9_MANIFEST_INDEX_DIGEST", "C9_ARTIFACT_INVENTORY_DIGEST",
    "C9_ROOT_INVENTORY_DIGEST", "FROZEN_C9_HASHES",
    "StageC10IntegrityError", "audit_c9_snapshot_hashes",
    "load_c9_snapshot", "load_c9_source", "validate_c9_snapshot",
]
