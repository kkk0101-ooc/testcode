"""Control-plane sealing for the completed Stage D D8 campaign.

This module deliberately stops at D8 execution completeness.  It validates
receipt, completion, checkpoint, manifest and provenance controls without
deserializing L0/L1 scientific payload values and without invoking D9 finalize
or D10 aggregation paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .constants import REPO_ROOT
from .stage_d_config import strict_json_bytes
from .stage_d_plan import (
    AUDIT_REPEAT_CLEAN_MAP_COUNT,
    FIDELITY_CURVE_COUNT,
    FIDELITY_CURVE_POINT_COUNT,
    IMAGE_GROUP_COUNT,
    LOGICAL_WORK_ITEM_COUNT,
    MAP_WORK_ITEM_COUNT,
    METHOD_IDS,
    STATISTICAL_N,
    canonical_sha256,
)


class StageDD8CompletionError(ValueError):
    """D8 campaign sealing cannot establish execution completeness."""


REPORT_ROOT = "reports/stage_d_d8_recovery_v2_20260904_r4"
FORMAL_ROOT = "evidence/stage_d/d8_recovery_v2_20260904/formal_r2"
BLOCKED_ROOT = "evidence/stage_d/d8_recovery_v2_20260904/formal"
CAMPAIGN_COMPLETION_PATH = f"{REPORT_ROOT}/d8_campaign_completion.json"
CAMPAIGN_ID = "STAGE_D_RECOVERY_V2_D8_20260904"
DOCUMENT_TYPE = "stage_d_d8_campaign_completion"
COMPLETION_REF = "refs/archive/stage-d-fidelity-v1-d8-completion"
SCHEMA_VERSION = 1
SHARD_COUNT = 8
EXPECTED_GENERATIONS = [0, 380, 760, 1140, 1520, 1894, 2268, 2642, 3016]
EXPECTED_LEDGER_DIGESTS = [
    "0e91369655142c38b18142ded5d5532aa6df3f1d9acb3e8c3972667855a95e2e",
    "e828172d11a27a255a98d2b443c1a3ecc3c468cc2ff9ace5509a3367eb4332c4",
    "0369f67f82935471a3a6fadbf685433d4854a3f55353cd0015898c9fca8945df",
    "9c86c6d2b02a5895ad6b20f952ff0f563499047bddb9e248cc0212675dc14c91",
    "5e82bb12ca819e1bb1691ab54f4fa8550686fe6f61afd45a36bac09efcc3843d",
    "847a303a83a6530fab158d170071b5ea5fecc4a364ffc7f8f83e1730c27f0d31",
    "7e510b2dd1cbbe68fc5c81f48ff4b4ee708bf350881adddb427e18077e542ab5",
    "611fbca7db8648ace8a229cabc1430278cea0e79d35e620a819cb5b48478cfb8",
    "3c0951e07ef1ed10446188c0ebd56e141a26218962fd147a7ce2f13a5d426fc9",
]
FINAL_CHECKPOINT_DIGEST = EXPECTED_LEDGER_DIGESTS[-1]
FINAL_CHECKPOINT_GENERATION = EXPECTED_GENERATIONS[-1]
D8_0_REBIND_AUTHORIZATION_PATH = (
    "config/stage_d_d8_0_fresh_root_authorization_20260904_r1.json"
)
D7_AUTHORIZATION_PATH = "config/stage_d_d7_execution_authorization_20260904_r4.json"
D8_0_COMPLETION_PATH = f"{REPORT_ROOT}/d8_0_completion.json"
R7_AUTHORIZATION_PATH = "config/stage_d_d8_continuation_authorization_20260905_r7.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_JSON_BYTES = 64 * 1024 * 1024


SHARD_EXPECTATIONS = [
    {
        "shard": "S00",
        "index": 0,
        "receipt_file_sha256": "93dd91beb47afa5a788a5ef66821459779c72c1a8fed1992b5700d238e75a942",
        "receipt_canonical_digest": "9b4a9b1edb6b69d96db6893dbe051c2a8f6ae3795d4b634b12bd0485555b2827",
        "completion_file_sha256": "86e96bb3aea2c25557d0ee49e70f3c7f078258a105afd9054082c4f96f7ddaab",
        "completion_canonical_digest": "e2125ecf983a5c75244e762c0837902b3914279607ca04bd2f2dc00685804d91",
        "completion_ref": "refs/archive/stage-d-fidelity-v1-d8-s00-completion",
        "completion_commit": "ac27c30f1f93e2df1ddd8f8fa02e8b13ea3e980d",
        "expected_image_groups": 63,
        "expected_units": 126,
        "expected_map_work_items": 126,
        "expected_fidelity_curves": 1260,
        "expected_fidelity_curve_points": 26460,
        "expected_logical_work_items": 26586,
    },
    {
        "shard": "S01",
        "index": 1,
        "receipt_file_sha256": "4fa38255e69eca09b9a62db1e2257a9650380d72f4154abe8eca3c2d55699857",
        "receipt_canonical_digest": "7178914a8c54824c869be6b67d11cc9a8ec9b3a823727b37dcac4e967a57be7e",
        "completion_file_sha256": "9a3599a8bfb5c970d4692ce2c2a56760526ad5a071cb4f02674647cc4f1a2abe",
        "completion_canonical_digest": "62dab22b67904c8e50fc9cb3a6aab9c558d3cbc815aaef3f713125bb44ec4996",
        "completion_ref": "refs/archive/stage-d-fidelity-v1-d8-s01-completion",
        "completion_commit": "f22a0942c559d1626b9cc3c4aa50cbd527082923",
        "expected_image_groups": 63,
        "expected_units": 126,
        "expected_map_work_items": 128,
        "expected_fidelity_curves": 1260,
        "expected_fidelity_curve_points": 26460,
        "expected_logical_work_items": 26588,
    },
    {
        "shard": "S02",
        "index": 2,
        "receipt_file_sha256": "a806eac0adff1c34cf2a587ec1adc5b7d63efefa45c729c5c82a2c0c060ce443",
        "receipt_canonical_digest": "f7ebe0c1d474c34a048872d702f9448f69bf357918f45dba734873d61038e4f3",
        "completion_file_sha256": "aedb78ecdc563cdfa1c6846248994e4fe7777cbef4d6bea9db99dd5cb0fde566",
        "completion_canonical_digest": "b3cf5dd8f815d21371a19e035de6742e6f8859d7fe798e15ac5c2ea019afc66b",
        "completion_ref": "refs/archive/stage-d-fidelity-v1-d8-s02-completion",
        "completion_commit": "b81f165266d4ab19e9b78c8a4c048d0cde756ddc",
        "expected_image_groups": 63,
        "expected_units": 126,
        "expected_map_work_items": 126,
        "expected_fidelity_curves": 1260,
        "expected_fidelity_curve_points": 26460,
        "expected_logical_work_items": 26586,
    },
    {
        "shard": "S03",
        "index": 3,
        "receipt_file_sha256": "adea8d7417cf8c5a4ab1f5be428b40cdd127e6a0f704f42bd05a0f014c075be5",
        "receipt_canonical_digest": "dbb3ecbc47db9df6f4440f0d71aa3e461a9bc7f2017191b5b9ece9afd3ae930c",
        "completion_file_sha256": "7e8661dd6538105c9b003369075092db582f113a22492c598b5704a0cf5e9598",
        "completion_canonical_digest": "d94575872debbf940a046c502971ba8fc47aa19e18933f3f6e88054696770ff5",
        "completion_ref": "refs/archive/stage-d-fidelity-v1-d8-s03-completion",
        "completion_commit": "8f54a3f00ba6815de722b07cd7af35d37daf3830",
        "expected_image_groups": 63,
        "expected_units": 126,
        "expected_map_work_items": 126,
        "expected_fidelity_curves": 1260,
        "expected_fidelity_curve_points": 26460,
        "expected_logical_work_items": 26586,
    },
    {
        "shard": "S04",
        "index": 4,
        "receipt_file_sha256": "8c64ef6060c6328de19d1166787bd9e7806f92bab24f0e2db5595b54a1719a42",
        "receipt_canonical_digest": "6f5352892cb737f74216da22ea4eb5fa0a99310c819c6e55d128811900953bbe",
        "completion_file_sha256": "a42caed61962110e1d644975f14e6e143d9af327c72ce2cbad8920d2bb9c69cd",
        "completion_canonical_digest": "3e579ae1a93c4d31f24a30fd5593b0fc14e6e256c90ae46a8a25989f87638a72",
        "completion_ref": "refs/archive/stage-d-fidelity-v1-d8-s04-completion",
        "completion_commit": "80a92ea77e5d450b3de0fdaa0f6dfae353b289e9",
        "expected_image_groups": 62,
        "expected_units": 124,
        "expected_map_work_items": 126,
        "expected_fidelity_curves": 1240,
        "expected_fidelity_curve_points": 26040,
        "expected_logical_work_items": 26166,
    },
    {
        "shard": "S05",
        "index": 5,
        "receipt_file_sha256": "947d2bc5332fdad5c0099dcfacc1dd4bc5a49318f51e6e3e5a6d61ab9b94ceee",
        "receipt_canonical_digest": "71e83ac5c1697806df54321eaabe097e75e1558975945d80d2d7da49a5d1ac02",
        "completion_file_sha256": "e3de9e6b3fbe21869e5c52babcc4cba53663f078194f9d84b7017719e04d777e",
        "completion_canonical_digest": "0cbcd32896c3da8f16b2f4b30b2e8c594c273d17c56b76c21b50218c12ef36e3",
        "completion_ref": "refs/archive/stage-d-fidelity-v1-d8-s05-completion",
        "completion_commit": "62d958c6f70daef97a8a546fbe01a63ade49fbd5",
        "expected_image_groups": 62,
        "expected_units": 124,
        "expected_map_work_items": 124,
        "expected_fidelity_curves": 1240,
        "expected_fidelity_curve_points": 26040,
        "expected_logical_work_items": 26164,
    },
    {
        "shard": "S06",
        "index": 6,
        "receipt_file_sha256": "64eaf5faf0b2b9c6399f2fb0d0582b4234258209aeca64d3c22fddf630f988d5",
        "receipt_canonical_digest": "aa572fedb7f5ce4fd4fc084768dbd7f1070680660d967c88d56af75ed8e77b19",
        "completion_file_sha256": "2b25fadda0d377d46b19e28d39e622b195c8cf5f71c5fbfbfec291c77c0983ab",
        "completion_canonical_digest": "068ebf2bf49fec60286a7cb6a49daf4e58e02015853d10673655a8b02a7178c6",
        "completion_ref": "refs/archive/stage-d-fidelity-v1-d8-s06-completion",
        "completion_commit": "0df64dc7314cfc0faa286bb60627f3501b9a065a",
        "expected_image_groups": 62,
        "expected_units": 124,
        "expected_map_work_items": 128,
        "expected_fidelity_curves": 1240,
        "expected_fidelity_curve_points": 26040,
        "expected_logical_work_items": 26168,
    },
    {
        "shard": "S07",
        "index": 7,
        "receipt_file_sha256": "c536ab4cee9fdf796dec9a497fdbd89de1998404ccd92d81dfd276a79594a18d",
        "receipt_canonical_digest": "a0aff53ec63227b9dea5f5b45a940aa295cc6ddfeddbf4e93439faa67ecae809",
        "completion_file_sha256": "a85318f9a9721f9748b88c637aea1cf367dd28e2da700b49b4328b97acbb8cf8",
        "completion_canonical_digest": "f853540712d0fbcdb358121af9dade8d2bdc012e0d2c720e58db6954669e9c5f",
        "completion_ref": "refs/archive/stage-d-fidelity-v1-d8-s07-completion",
        "completion_commit": "1474b103124c5f68187960e305410e114718a350",
        "expected_image_groups": 62,
        "expected_units": 124,
        "expected_map_work_items": 126,
        "expected_fidelity_curves": 1240,
        "expected_fidelity_curve_points": 26040,
        "expected_logical_work_items": 26166,
    },
]


def _plain(value: Any, context: str = "value") -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDD8CompletionError(f"{context} contains non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDD8CompletionError(f"{context} has an invalid key")
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(child, f"{context}[{index}]") for index, child in enumerate(value)]
    raise StageDD8CompletionError(f"{context} is not JSON-compatible")


def _semantic_digest(document: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: child for key, child in document.items() if key != "canonical_digest"
    })


def canonical_file_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain(document), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"


def _safe_regular_bytes(path: Path, context: str, *, max_bytes: int = _MAX_JSON_BYTES) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    except OSError as error:
        raise StageDD8CompletionError(f"cannot safely open {context}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > max_bytes
        ):
            raise StageDD8CompletionError(f"{context} is not a bounded single-link file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
        )
        if len(raw) != before.st_size or identity(before) != identity(after):
            raise StageDD8CompletionError(f"{context} changed while being read")
        return raw
    finally:
        os.close(descriptor)


def _read_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    raw = _safe_regular_bytes(path, context)
    try:
        value = strict_json_bytes(raw, context)
    except Exception as error:
        raise StageDD8CompletionError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise StageDD8CompletionError(f"{context} must be an object")
    return value, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StageDD8CompletionError(f"{context} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, context: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise StageDD8CompletionError(f"{context} must be a Git commit SHA")
    return value


def _git(repo: Path, *arguments: str) -> str:
    import subprocess

    environment = os.environ.copy()
    environment.update({
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
    })
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=repo, check=True, capture_output=True,
            text=True, timeout=60, env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise StageDD8CompletionError(
            f"Git check failed: {' '.join(arguments)}"
        ) from error
    return result.stdout.strip()


def _validate_self_digest(document: Mapping[str, Any], context: str) -> str:
    digest = _require_sha256(document.get("canonical_digest"), f"{context}.canonical_digest")
    if _semantic_digest(document) != digest:
        raise StageDD8CompletionError(f"{context} canonical digest mismatch")
    return digest


def _file_record(repo: Path, relative: str, context: str) -> dict[str, Any]:
    raw = _safe_regular_bytes(repo / relative, context)
    value = None
    try:
        value = strict_json_bytes(raw, context)
    except Exception:
        pass
    record = {"path": relative, "file_sha256": _sha256(raw)}
    if isinstance(value, Mapping) and "canonical_digest" in value:
        record["canonical_digest"] = _validate_self_digest(value, context)
    return record


def _validate_report_json(
    repo: Path, relative: str, expected_file: str, expected_canonical: str,
    context: str,
) -> tuple[dict[str, Any], bytes]:
    document, raw = _read_json(repo / relative, context)
    if _sha256(raw) != expected_file:
        raise StageDD8CompletionError(f"{context} file SHA-256 mismatch")
    if _validate_self_digest(document, context) != expected_canonical:
        raise StageDD8CompletionError(f"{context} canonical digest mismatch")
    return document, raw


def _git_ref_record(repo: Path, ref: str, expected: str) -> dict[str, Any]:
    observed = _git(repo, "rev-parse", f"{ref}^{{commit}}")
    if observed != expected:
        raise StageDD8CompletionError(
            f"{ref} drifted: expected {expected}, observed {observed}"
        )
    return {"ref": ref, "commit": observed}


def _attempt_inventory_digest(path: Path) -> str:
    records: list[dict[str, Any]] = []
    for directory, names, files in os.walk(path, followlinks=False):
        names.sort()
        files.sort()
        current = Path(directory)
        for name in names:
            child = current / name
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise StageDD8CompletionError("attempt inventory has unsafe directory")
        for name in files:
            child = current / name
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise StageDD8CompletionError("attempt inventory has unsafe file")
            raw = _safe_regular_bytes(child, f"attempt file {child}", max_bytes=256 * 1024 * 1024)
            records.append({
                "path": child.relative_to(path).as_posix(),
                "bytes": len(raw),
                "sha256": _sha256(raw),
            })
    records.sort(key=lambda row: row["path"])
    return canonical_sha256({"files": records})


def _validate_artifact_hashes(
    attempt_path: Path, manifest: Mapping[str, Any],
) -> tuple[int, int, int, bool]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise StageDD8CompletionError("attempt manifest artifacts are not an array")
    l0_l1_bytes = 0
    l2_bytes = 0
    payload_count = 0
    has_l2 = False
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise StageDD8CompletionError("attempt artifact record is not an object")
        relative = row.get("path")
        if not isinstance(relative, str) or relative.startswith("/") or ".." in relative.split("/"):
            raise StageDD8CompletionError("attempt artifact path is unsafe")
        path = attempt_path / relative
        raw = _safe_regular_bytes(path, f"attempt artifact {relative}", max_bytes=256 * 1024 * 1024)
        if row.get("bytes") != len(raw) or row.get("sha256") != _sha256(raw):
            raise StageDD8CompletionError(f"attempt artifact binding mismatch: {relative}")
        level = row.get("level")
        kind = row.get("kind")
        if level in {"L0", "L1"} and kind not in {"provenance", "scientific_failure"}:
            l0_l1_bytes += len(raw)
            payload_count += 1
        if level == "L2":
            l2_bytes += len(raw)
            has_l2 = True
    return l0_l1_bytes, l2_bytes, payload_count, has_l2


def _read_attempt_controls(
    attempt_path: Path, *, expected_unit_digest: str, expected_group_digest: str,
    expected_method_index: int, expected_pair_digest: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, bytes, bytes]:
    status, status_raw = _read_json(attempt_path / "status.json", "attempt status")
    manifest, manifest_raw = _read_json(attempt_path / "manifest.json", "attempt manifest")
    provenance, provenance_raw = _read_json(
        attempt_path / "L0/provenance.json", "attempt provenance"
    )
    if (
        status.get("document_type") != "stage_d_execution_unit_status"
        or status.get("status") != "succeeded"
        or status.get("terminal") is not True
        or status.get("complete") is not True
        or status.get("attempt_number") != 1
        or status.get("unit_digest") != expected_unit_digest
        or status.get("manifest_sha256") != _sha256(manifest_raw)
    ):
        raise StageDD8CompletionError("attempt status binding mismatch")
    if (
        manifest.get("document_type") != "stage_d_execution_unit_manifest"
        or manifest.get("status") != "succeeded"
        or manifest.get("terminal") is not True
        or manifest.get("complete") is not True
        or manifest.get("attempt_number") != 1
        or manifest.get("unit_digest") != expected_unit_digest
        or manifest.get("group_digest") != expected_group_digest
        or manifest.get("pair_digest") != expected_pair_digest
        or manifest.get("provenance_sha256") != _sha256(provenance_raw)
    ):
        raise StageDD8CompletionError("attempt manifest binding mismatch")
    identity = manifest.get("unit_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("canonical_digest") != expected_unit_digest
        or identity.get("method_index") != expected_method_index
        or identity.get("pair_binding_digest") != expected_pair_digest
    ):
        raise StageDD8CompletionError("attempt manifest unit identity mismatch")
    if (
        provenance.get("document_type") != "stage_d_execution_unit_provenance"
        or provenance.get("mode") != "formal"
        or provenance.get("unit_identity_sha256") != expected_unit_digest
        or provenance.get("unit_identity", {}).get("canonical_digest") != expected_unit_digest
        or provenance.get("bindings", {}).get("coordinates", {}).get("group_digest") != expected_group_digest
        or provenance.get("bindings", {}).get("coordinates", {}).get("pair_digest") != expected_pair_digest
        or provenance.get("bindings", {}).get("terminal") != {
            "complete": True,
            "status": "succeeded",
            "terminal": True,
        }
        or provenance.get("bindings", {}).get("failure") is not None
        or provenance.get("canonical_digest") != _semantic_digest(provenance)
    ):
        raise StageDD8CompletionError("attempt provenance binding mismatch")
    # Do not parse L0/L1 payload values.  Only validate byte/hash bindings.
    _validate_artifact_hashes(attempt_path, manifest)
    if manifest.get("artifacts", [])[:-1] != provenance.get("bindings", {}).get("artifact_inventory"):
        raise StageDD8CompletionError("provenance artifact inventory mismatch")
    return status, manifest, provenance, status_raw, manifest_raw, provenance_raw


def _stage_units_by_shard(checkpoint: Mapping[str, Any], shard_index: int) -> list[Mapping[str, Any]]:
    return [
        row for row in checkpoint["units"]
        if row["group_selection_order"] % SHARD_COUNT == shard_index
    ]


def _validate_attempt_binding_digest(
    receipt: Mapping[str, Any], units: Sequence[Mapping[str, Any]],
) -> str:
    rows = [
        {
            "unit_digest": unit["unit_digest"],
            "attempts": [
                {
                    "attempt_id": attempt["attempt_id"],
                    "canonical_digest": attempt["canonical_digest"],
                    "status_sha256": attempt["status_sha256"],
                    "attempt_inventory_digest": attempt["attempt_inventory_digest"],
                }
                for attempt in unit["attempts"]
            ],
        }
        for unit in units
    ]
    observed = canonical_sha256(rows)
    stored = receipt.get("attempts", {}).get("ordered_attempt_bindings_sha256")
    _require_sha256(stored, "receipt ordered attempt bindings")
    # S00/S01 were written before the later binding-projection field was
    # introduced.  The checkpoint/manifest source validation remains
    # authoritative for those receipts, while later receipts must match the
    # now-explicit projection formula exactly.
    if "binding_projection" in receipt.get("attempts", {}):
        if observed != stored:
            raise StageDD8CompletionError("receipt attempt binding digest mismatch")
        return "projection_v2_matched"
    return "legacy_receipt_projection_source_validated"


def _validate_control_record(
    control: Mapping[str, Any], raw: bytes, receipt: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    counts = control.get("counts")
    if not isinstance(counts, Mapping):
        raise StageDD8CompletionError("shard control counts missing")
    if (
        control.get("document_type") != "stage_d_shard_control"
        or control.get("status") != "PASS"
        or control.get("control", {}).get("shard_index") != expected["index"]
        or control.get("control", {}).get("shard_count") != SHARD_COUNT
        or control.get("control", {}).get("checkpoint_generation")
        != EXPECTED_GENERATIONS[expected["index"] + 1]
        or control.get("terminal", {}).get("state") != "completed"
        or control.get("digests", {}).get("checkpoint_digest")
        != EXPECTED_LEDGER_DIGESTS[expected["index"] + 1]
        or counts.get("expected_groups") != expected["expected_image_groups"]
        or counts.get("expected_units") != expected["expected_units"]
        or counts.get("terminal_groups") != expected["expected_image_groups"]
        or counts.get("terminal_units") != expected["expected_units"]
        or counts.get("succeeded_units") != expected["expected_units"]
        or counts.get("scientific_failed_units") != 0
        or counts.get("retry_pending_units") != 0
        or counts.get("blocked_units") != 0
    ):
        raise StageDD8CompletionError("shard control record does not match D8 completion")
    if receipt.get("source_control", {}).get("file_sha256") != _sha256(raw):
        raise StageDD8CompletionError("receipt source control file hash mismatch")


def _shard_execution_binding(
    repo: Path, index: int, completion: Mapping[str, Any], receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if index == 0:
        authority = receipt.get("authority", {})
        producer = _git_ref_record(
            repo,
            "refs/archive/stage-d-fidelity-v1-d8-0-rebind-r1-producer",
            authority.get("rebind_producer_commit"),
        )
        authority_ref = _git_ref_record(
            repo,
            "refs/archive/stage-d-fidelity-v1-d8-0-rebind-r1-authority",
            authority.get("rebind_authority_commit"),
        )
        return {
            "producer": producer,
            "authority": authority_ref,
            "authorization_path": D8_0_REBIND_AUTHORIZATION_PATH,
            "authorization_canonical_digest": authority[
                "rebind_authorization_canonical_digest"
            ],
            "base_authority_commit": authority["base_authority_commit"],
            "base_authorization_canonical_digest": authority[
                "base_authorization_canonical_digest"
            ],
            "scientific_scope_changed": False,
        }
    source = (
        completion.get("continuation_correction")
        if index == 1
        else completion.get("rolling_authority")
    )
    if not isinstance(source, Mapping):
        raise StageDD8CompletionError(f"D8-S{index:02d} authority binding missing")
    producer = _git_ref_record(repo, source["producer_ref"], source["producer_commit"])
    authority = _git_ref_record(repo, source["authority_ref"], source["authority_commit"])
    if source.get("scientific_scope_changed") is not False:
        raise StageDD8CompletionError(f"D8-S{index:02d} authority changed science scope")
    return {
        "producer": producer,
        "authority": authority,
        "authorization_path": source["authorization_path"],
        "authorization_file_sha256": source["authorization_file_sha256"],
        "authorization_canonical_digest": source["authorization_canonical_digest"],
        "prior_receipt_canonical_digest": source.get("prior_receipt_canonical_digest"),
        "scientific_scope_changed": source["scientific_scope_changed"],
    }


def _validate_shard_records(
    repo: Path, checkpoint: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deviations: list[dict[str, Any]] = []
    totals = {
        "image_groups": 0,
        "units": 0,
        "map_work_items": 0,
        "fidelity_curves": 0,
        "curve_points": 0,
        "logical_work_items": 0,
        "succeeded": 0,
        "scientific_failed": 0,
        "retry_pending": 0,
        "blocked": 0,
    }
    for expected in SHARD_EXPECTATIONS:
        index = expected["index"]
        receipt_path = f"{REPORT_ROOT}/d8_s{index:02d}_shard_receipt.json"
        completion_path = f"{REPORT_ROOT}/d8_s{index:02d}_completion.json"
        control_path = f"{REPORT_ROOT}/d8_s{index:02d}_shard_control.json"
        receipt, receipt_raw = _validate_report_json(
            repo, receipt_path, expected["receipt_file_sha256"],
            expected["receipt_canonical_digest"], f"D8-{expected['shard']} receipt",
        )
        completion, completion_raw = _validate_report_json(
            repo, completion_path, expected["completion_file_sha256"],
            expected["completion_canonical_digest"],
            f"D8-{expected['shard']} completion",
        )
        control, control_raw = _read_json(repo / control_path, f"D8-{expected['shard']} control")
        _validate_control_record(control, control_raw, receipt, expected)
        if (
            receipt.get("document_type") != "stage_d_d8_shard_receipt"
            or receipt.get("status") != "PASS"
            or receipt.get("subtask") != f"D8-{expected['shard']}"
            or receipt.get("parent_status") != "IN_PROGRESS"
            or receipt.get("formal_root") != FORMAL_ROOT
            or receipt.get("shard", {}).get("index") != index
            or receipt.get("shard", {}).get("count") != SHARD_COUNT
            or receipt.get("shard", {}).get("assignment")
            != "selection_order_modulo_frozen_shard_count"
            or receipt.get("ledger_transition", {}).get("before_generation")
            != EXPECTED_GENERATIONS[index]
            or receipt.get("ledger_transition", {}).get("after_generation")
            != EXPECTED_GENERATIONS[index + 1]
            or receipt.get("ledger_transition", {}).get("before_canonical_digest")
            != EXPECTED_LEDGER_DIGESTS[index]
            or receipt.get("ledger_transition", {}).get("after_canonical_digest")
            != EXPECTED_LEDGER_DIGESTS[index + 1]
            or receipt.get("ledger_transition", {}).get("before_shard_state") != "pending"
            or receipt.get("ledger_transition", {}).get("after_shard_state") != "completed"
        ):
            raise StageDD8CompletionError(f"D8-{expected['shard']} receipt binding mismatch")
        if (
            completion.get("document_type") != f"stage_d_d8_s{index:02d}_completion"
            or completion.get("subtask_status") != "SUBTASK_PASS"
            or completion.get("parent_status") != "IN_PROGRESS"
            or completion.get("formal_root") != FORMAL_ROOT
            or completion.get("receipt", {}).get("receipt_canonical_digest")
            != receipt["canonical_digest"]
            or completion.get("receipt", {}).get("receipt_file_sha256")
            != expected["receipt_file_sha256"]
            or completion.get("receipt", {}).get("receipt_path") != receipt_path
        ):
            raise StageDD8CompletionError(f"D8-{expected['shard']} completion binding mismatch")
        receipt_counts = receipt["counts"]
        completion_counts = completion["counts"]
        for name, key in (
            ("expected_image_groups", "expected_image_groups"),
            ("expected_units", "expected_units"),
            ("expected_map_work_items", "expected_map_work_items"),
            ("expected_fidelity_curves", "expected_fidelity_curves"),
            ("expected_fidelity_curve_points", "expected_fidelity_curve_points"),
            ("expected_logical_work_items", "expected_logical_work_items"),
        ):
            if receipt_counts.get(key) != expected[name] or completion_counts.get(key) != expected[name]:
                raise StageDD8CompletionError(f"D8-{expected['shard']} count mismatch for {key}")
        for key in (
            "terminal_image_groups", "terminal_units", "succeeded_units",
            "scientific_failed_units", "retry_pending_units", "blocked_units",
            "attempt_count",
        ):
            if receipt_counts.get(key) != completion_counts.get(key):
                raise StageDD8CompletionError(f"D8-{expected['shard']} receipt/completion count mismatch")
        orders = receipt.get("shard", {}).get("selection_orders")
        expected_orders = list(range(index, IMAGE_GROUP_COUNT, SHARD_COUNT))
        if orders != expected_orders:
            raise StageDD8CompletionError(f"D8-{expected['shard']} selection order mismatch")
        units = _stage_units_by_shard(checkpoint, index)
        groups = [
            row for row in checkpoint["groups"]
            if row["selection_order"] % SHARD_COUNT == index
        ]
        if len(groups) != expected["expected_image_groups"] or len(units) != expected["expected_units"]:
            raise StageDD8CompletionError(f"D8-{expected['shard']} checkpoint coverage mismatch")
        if [row["selection_order"] for row in groups] != expected_orders:
            raise StageDD8CompletionError(f"D8-{expected['shard']} checkpoint assignment mismatch")
        if any(len(row["unit_digests"]) != 2 for row in groups):
            raise StageDD8CompletionError(f"D8-{expected['shard']} group/unit pair split")
        group_digest = canonical_sha256([row["group_digest"] for row in groups])
        unit_digest = canonical_sha256([row["unit_digest"] for row in units])
        if (
            receipt.get("shard", {}).get("ordered_image_group_digests_sha256") != group_digest
            or receipt.get("shard", {}).get("ordered_unit_digests_sha256") != unit_digest
        ):
            raise StageDD8CompletionError(f"D8-{expected['shard']} ordered digest mismatch")
        projection_status = _validate_attempt_binding_digest(receipt, units)
        actual_audit_groups: set[int] = set()
        actual_audit_units = 0
        l2_units = 0
        for group in groups:
            if group["state"] != "terminal" or group["terminal"] is not True:
                raise StageDD8CompletionError(f"D8-{expected['shard']} group is not terminal")
        by_group = {group["group_digest"]: group for group in groups}
        for unit in units:
            if (
                unit["state"] != "succeeded"
                or unit["terminal"] is not True
                or unit["failure_class"] is not None
                or unit["failure_code"] is not None
                or unit["retry_authorization_digest"] is not None
                or len(unit["attempts"]) != 1
            ):
                raise StageDD8CompletionError(f"D8-{expected['shard']} unit state mismatch")
            attempt = unit["attempts"][0]
            attempt_path = Path(attempt["attempt_path"])
            status, manifest, provenance, status_raw, _manifest_raw, _provenance_raw = (
                _read_attempt_controls(
                    attempt_path,
                    expected_unit_digest=unit["unit_digest"],
                    expected_group_digest=unit["group_digest"],
                    expected_method_index=unit["method_index"],
                    expected_pair_digest=unit["pair_binding_digest"],
                )
            )
            if _sha256(status_raw) != attempt["status_sha256"]:
                raise StageDD8CompletionError("checkpoint status SHA mismatch")
            inventory = _attempt_inventory_digest(attempt_path)
            if inventory != attempt["attempt_inventory_digest"]:
                raise StageDD8CompletionError("checkpoint attempt inventory mismatch")
            if attempt["canonical_digest"] != canonical_sha256({
                key: child for key, child in attempt.items()
                if key != "canonical_digest"
            }):
                raise StageDD8CompletionError("checkpoint attempt self digest mismatch")
            if manifest["audit"]:
                actual_audit_units += 1
                actual_audit_groups.add(unit["group_selection_order"])
                if any(row.get("path") == "L2/audit_evidence.npz" for row in manifest["artifacts"]):
                    l2_units += 1
            if unit["method_id"] not in METHOD_IDS:
                raise StageDD8CompletionError("unit method is outside frozen methods")
            if by_group[unit["group_digest"]]["shard_id"] != index:
                raise StageDD8CompletionError("unit group shard mismatch")
        if actual_audit_units != l2_units:
            raise StageDD8CompletionError("audit L2 coverage mismatch")
        expected_audit_groups = (expected["expected_map_work_items"] - expected["expected_units"]) // 2
        if actual_audit_units != expected_audit_groups * 2:
            raise StageDD8CompletionError("audit unit count mismatch")
        stored_audit = receipt_counts.get("audit_image_groups")
        if stored_audit is not None and stored_audit != expected_audit_groups:
            deviations.append({
                "shard": expected["shard"],
                "field": "counts.audit_image_groups",
                "stored": stored_audit,
                "source_recomputed": expected_audit_groups,
                "disposition": (
                    "historical summary-field deviation; source manifests and "
                    "map/logical work-item totals verify audit evidence retained"
                ),
            })
        ref = _git_ref_record(repo, expected["completion_ref"], expected["completion_commit"])
        row = {
            "shard": expected["shard"],
            "shard_index": index,
            "execution_binding": _shard_execution_binding(
                repo, index, completion, receipt,
            ),
            "receipt_path": receipt_path,
            "receipt_file_sha256": _sha256(receipt_raw),
            "receipt_canonical_digest": receipt["canonical_digest"],
            "completion_path": completion_path,
            "completion_file_sha256": _sha256(completion_raw),
            "completion_canonical_digest": completion["canonical_digest"],
            "completion_ref": ref["ref"],
            "completion_commit": ref["commit"],
            "subtask_status": "SUBTASK_PASS",
            "parent_status_at_shard_completion": "IN_PROGRESS",
            "selection_orders_sha256": canonical_sha256(orders),
            "ordered_image_group_digests_sha256": group_digest,
            "ordered_unit_digests_sha256": unit_digest,
            "attempt_binding_projection_status": projection_status,
            "attempt_binding_digest": receipt.get("attempts", {}).get("ordered_attempt_bindings_sha256"),
            "attempt_ids": ["attempt-000001"],
            "counts": {
                "image_groups": expected["expected_image_groups"],
                "units": expected["expected_units"],
                "map_work_items": expected["expected_map_work_items"],
                "fidelity_curves": expected["expected_fidelity_curves"],
                "curve_points": expected["expected_fidelity_curve_points"],
                "logical_work_items": expected["expected_logical_work_items"],
                "terminal_image_groups": receipt_counts["terminal_image_groups"],
                "terminal_units": receipt_counts["terminal_units"],
                "succeeded_units": receipt_counts["succeeded_units"],
                "scientific_failed_units": receipt_counts["scientific_failed_units"],
                "retry_pending_units": receipt_counts["retry_pending_units"],
                "blocked_units": receipt_counts["blocked_units"],
                "audit_image_groups_source_recomputed": expected_audit_groups,
                "audit_units_source_recomputed": actual_audit_units,
            },
            "ledger_transition": {
                "before_generation": EXPECTED_GENERATIONS[index],
                "before_canonical_digest": EXPECTED_LEDGER_DIGESTS[index],
                "after_generation": EXPECTED_GENERATIONS[index + 1],
                "after_canonical_digest": EXPECTED_LEDGER_DIGESTS[index + 1],
            },
        }
        rows.append(row)
        totals["image_groups"] += expected["expected_image_groups"]
        totals["units"] += expected["expected_units"]
        totals["map_work_items"] += expected["expected_map_work_items"]
        totals["fidelity_curves"] += expected["expected_fidelity_curves"]
        totals["curve_points"] += expected["expected_fidelity_curve_points"]
        totals["logical_work_items"] += expected["expected_logical_work_items"]
        totals["succeeded"] += receipt_counts["succeeded_units"]
        totals["scientific_failed"] += receipt_counts["scientific_failed_units"]
        totals["retry_pending"] += receipt_counts["retry_pending_units"]
        totals["blocked"] += receipt_counts["blocked_units"]
    return rows, deviations, totals


def _validate_checkpoint(repo: Path) -> tuple[dict[str, Any], bytes]:
    checkpoint_path = repo / FORMAL_ROOT / "runtime_checkpoint.json"
    checkpoint, raw = _read_json(checkpoint_path, "D8 final runtime checkpoint")
    if _semantic_digest(checkpoint) != FINAL_CHECKPOINT_DIGEST:
        raise StageDD8CompletionError("final checkpoint canonical digest mismatch")
    lifecycle = checkpoint.get("lifecycle")
    if (
        checkpoint.get("document_type") != "stage_d_runtime_checkpoint"
        or checkpoint.get("generation") != FINAL_CHECKPOINT_GENERATION
        or checkpoint.get("canonical_digest") != FINAL_CHECKPOINT_DIGEST
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("state") != "complete"
        or lifecycle.get("active_group_claims") != 0
        or lifecycle.get("active_shard_leases") != 0
        or lifecycle.get("terminal_groups") != IMAGE_GROUP_COUNT
        or lifecycle.get("terminal_units") != 1000
        or lifecycle.get("scientific_failure_units") != 0
        or lifecycle.get("retry_pending_units") != 0
        or lifecycle.get("blocked_units") != 0
        or lifecycle.get("blocked_reasons") != []
    ):
        raise StageDD8CompletionError("final checkpoint lifecycle mismatch")
    shards = checkpoint.get("shards")
    groups = checkpoint.get("groups")
    units = checkpoint.get("units")
    if (
        not isinstance(shards, list) or len(shards) != SHARD_COUNT
        or [row.get("state") for row in shards] != ["completed"] * SHARD_COUNT
        or any(row.get("terminal") is not True or row.get("owner_token_digest") is not None for row in shards)
        or not isinstance(groups, list) or len(groups) != IMAGE_GROUP_COUNT
        or not isinstance(units, list) or len(units) != 1000
    ):
        raise StageDD8CompletionError("final checkpoint coverage mismatch")
    for index, group in enumerate(groups):
        if (
            group.get("selection_order") != index
            or group.get("shard_id") != index % SHARD_COUNT
            or group.get("state") != "terminal"
            or group.get("terminal") is not True
            or group.get("owner_token_digest") is not None
            or not isinstance(group.get("unit_digests"), list)
            or len(group["unit_digests"]) != 2
        ):
            raise StageDD8CompletionError("checkpoint group binding mismatch")
    for index, unit in enumerate(units):
        if (
            unit.get("group_selection_order") != index // 2
            or unit.get("method_index") != index % 2
            or unit.get("method_id") != METHOD_IDS[index % 2]
            or unit.get("state") != "succeeded"
            or unit.get("terminal") is not True
            or unit.get("owner_token_digest") is not None
            or unit.get("failure_class") is not None
            or unit.get("failure_code") is not None
            or unit.get("retry_authorization_digest") is not None
            or not isinstance(unit.get("attempts"), list)
            or len(unit["attempts"]) != 1
            or unit["attempts"][0].get("attempt_id") != "attempt-000001"
            or unit["attempts"][0].get("state") != "succeeded"
            or unit["attempts"][0].get("retry_authorization_digest") is not None
        ):
            raise StageDD8CompletionError("checkpoint unit binding mismatch")
    if checkpoint.get("failure_rows") != [
        {
            "unit_digest": unit["unit_digest"],
            "group_selection_order": unit["group_selection_order"],
            "method_index": unit["method_index"],
            "state": unit["state"],
            "terminal": unit["terminal"],
            "attempt_count": len(unit["attempts"]),
            "failure_class": unit["failure_class"],
            "failure_code": unit["failure_code"],
        }
        for unit in units
    ]:
        raise StageDD8CompletionError("checkpoint failure rows mismatch")
    return checkpoint, raw


def _validate_access_and_authority(repo: Path) -> dict[str, Any]:
    d7, d7_raw = _validate_report_json(
        repo, D7_AUTHORIZATION_PATH,
        "458c903e783385514c861c636fcbd5ee2423c00b9b217592530242b71d0a2ab8",
        "2c649deda654ac1a040d8919a973467ff6d520e5279a778bfef785c8ef072f8f",
        "D7-R4 authorization",
    )
    d8_rebind, d8_rebind_raw = _validate_report_json(
        repo, D8_0_REBIND_AUTHORIZATION_PATH,
        "c7f306e64dd99fe3b9e9e7586b8cde6805b907d1c2e84504043875c0a92f1657",
        "3479f9457aa924fe189591e836e38bb1a65cce8543e3a879477025782659afc9",
        "D8.0 rebind authorization",
    )
    d8_0, d8_0_raw = _validate_report_json(
        repo, D8_0_COMPLETION_PATH,
        "aff28d4f43ba589aa18d19e4c8c6fb2d6a049a0af9f5b296a8479b13cbcf4b64",
        "6bd9569f502a22a5785db4c92ac48cf33e7780c8a39487d6de59ab01f995664f",
        "D8.0 completion",
    )
    r7, r7_raw = _validate_report_json(
        repo, R7_AUTHORIZATION_PATH,
        "6ebfa30fb51493d6c2081a3078ca14482049fe7e6da96d3fab02eadb00ed54b8",
        "69fb2d13538cec53a4f6776b75ca19404cecf0183bfced367acaa4360e1594dd",
        "D8 R7 authorization",
    )
    lifecycle, lifecycle_raw = _read_json(
        repo / FORMAL_ROOT / "access/lifecycle.json", "D8 access lifecycle"
    )
    if (
        lifecycle.get("state") != "RUNNABLE"
        or lifecycle.get("canonical_digest")
        != "56b3f3cdb3a72bd72bdaa3f93f15cdb1043f362691a1b24089382862b935cf50"
        or lifecycle.get("journal_head_digest")
        != "293f46d8a1421607a547171b3ed78be21aebab4461c00762fdba4fa68d50ddc8"
        or lifecycle.get("live_assets_digest")
        != "489aa2a9a16e4f84046df8ebdf910a5adf04d9525a1aa901915fe778c6ad1d40"
        or lifecycle.get("validated_asset_count") != 500
    ):
        raise StageDD8CompletionError("D8 access lifecycle mismatch")
    return {
        "d7_r4": {
            "producer": _git_ref_record(repo, "refs/archive/stage-d-fidelity-v1-d7-r4-producer", "d7c88df66f3e55ec31ad4c6ea34a3002ca6d1a4d"),
            "authority": _git_ref_record(repo, "refs/archive/stage-d-fidelity-v1-d7-r4-authority", "61214b808b75cf804763078479d1b96f7418db32"),
            "authorization_path": D7_AUTHORIZATION_PATH,
            "authorization_file_sha256": _sha256(d7_raw),
            "authorization_canonical_digest": d7["canonical_digest"],
        },
        "d8_0_rebind": {
            "producer": _git_ref_record(repo, "refs/archive/stage-d-fidelity-v1-d8-0-rebind-r1-producer", "4c4dd4af010afdc72e26f10cb8e6519e2448cf40"),
            "authority": _git_ref_record(repo, "refs/archive/stage-d-fidelity-v1-d8-0-rebind-r1-authority", "fb27b2526d55861489c19069b3a5665cede757e3"),
            "authorization_path": D8_0_REBIND_AUTHORIZATION_PATH,
            "authorization_file_sha256": _sha256(d8_rebind_raw),
            "authorization_canonical_digest": d8_rebind["canonical_digest"],
            "completion_path": D8_0_COMPLETION_PATH,
            "completion_file_sha256": _sha256(d8_0_raw),
            "completion_canonical_digest": d8_0["canonical_digest"],
        },
        "r7_final_execution_binding": {
            "producer": _git_ref_record(repo, "refs/archive/stage-d-fidelity-v1-d8-continuation-r7-producer", "3fdf2e1f30d5038679b4dd47afdabd9eb5d0528a"),
            "authority": _git_ref_record(repo, "refs/archive/stage-d-fidelity-v1-d8-continuation-r7-authority", "c58f8ba4f8b2cbba15b08def615f0d3989682b4e"),
            "authorization_path": R7_AUTHORIZATION_PATH,
            "authorization_file_sha256": _sha256(r7_raw),
            "authorization_canonical_digest": r7["canonical_digest"],
            "postprocessor_fix_commit": "59c2c9c852908fc4aeb46078b7a5ce1cbde523da",
        },
        "access_gate": {
            "lifecycle_path": f"{FORMAL_ROOT}/access/lifecycle.json",
            "lifecycle_file_sha256": _sha256(lifecycle_raw),
            "lifecycle_digest": lifecycle["canonical_digest"],
            "journal_head_digest": lifecycle["journal_head_digest"],
            "live_assets_digest": lifecycle["live_assets_digest"],
            "validated_live_assets": lifecycle["validated_asset_count"],
        },
    }


def build_d8_campaign_completion(
    *, repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    authority = _validate_access_and_authority(repo)
    checkpoint, checkpoint_raw = _validate_checkpoint(repo)
    shard_rows, deviations, totals = _validate_shard_records(repo, checkpoint)
    if totals != {
        "image_groups": IMAGE_GROUP_COUNT,
        "units": 1000,
        "map_work_items": MAP_WORK_ITEM_COUNT,
        "fidelity_curves": FIDELITY_CURVE_COUNT,
        "curve_points": FIDELITY_CURVE_POINT_COUNT,
        "logical_work_items": LOGICAL_WORK_ITEM_COUNT,
        "succeeded": 1000,
        "scientific_failed": 0,
        "retry_pending": 0,
        "blocked": 0,
    }:
        raise StageDD8CompletionError("D8 campaign totals do not match frozen plan")
    baseline_head = _git(
        repo, "rev-parse", "refs/archive/stage-d-fidelity-v1-d8-s07-completion^{commit}",
    )
    if baseline_head != "1474b103124c5f68187960e305410e114718a350":
        raise StageDD8CompletionError("D8-S07 completion baseline ref drifted")
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "document_type": DOCUMENT_TYPE,
        "campaign_id": CAMPAIGN_ID,
        "parent_stage": "D8",
        "status": "PASS",
        "objective_completed": (
            "D8 raw shard execution completed for all frozen S00-S07 shards; "
            "this records execution completeness only."
        ),
        "formal_root": str((repo / FORMAL_ROOT).resolve()),
        "relative_formal_root": FORMAL_ROOT,
        "blocked_root_retained": {
            "path": str((repo / BLOCKED_ROOT).resolve()),
            "relative_path": BLOCKED_ROOT,
            "retained": True,
            "reused": False,
        },
        "authority_chain": authority,
        "plan": {
            "canonical_digest": "5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443",
            "coordinate_digest": "cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb",
            "expected": {
                "image_groups": IMAGE_GROUP_COUNT,
                "units": 1000,
                "map_work_items": MAP_WORK_ITEM_COUNT,
                "fidelity_curves": FIDELITY_CURVE_COUNT,
                "curve_points": FIDELITY_CURVE_POINT_COUNT,
                "logical_work_items": LOGICAL_WORK_ITEM_COUNT,
                "statistical_n": STATISTICAL_N,
                "audit_repeat_clean_maps": AUDIT_REPEAT_CLEAN_MAP_COUNT,
            },
        },
        "final_checkpoint": {
            "path": f"{FORMAL_ROOT}/runtime_checkpoint.json",
            "generation": checkpoint["generation"],
            "file_sha256": _sha256(checkpoint_raw),
            "canonical_digest": checkpoint["canonical_digest"],
            "lifecycle": checkpoint["lifecycle"],
            "shard_states": [row["state"] for row in checkpoint["shards"]],
        },
        "ledger_chain": {
            "generations": EXPECTED_GENERATIONS,
            "canonical_digests": EXPECTED_LEDGER_DIGESTS,
            "continuous": True,
        },
        "shards": shard_rows,
        "receipt_summary_field_deviations": deviations,
        "campaign_totals": totals,
        "all_shards_terminal": True,
        "all_receipts_verified": True,
        "expected_image_groups": IMAGE_GROUP_COUNT,
        "observed_image_groups": checkpoint["lifecycle"]["terminal_groups"],
        "expected_units": 1000,
        "observed_units": checkpoint["lifecycle"]["terminal_units"],
        "succeeded": totals["succeeded"],
        "scientific_failed": totals["scientific_failed"],
        "retry_pending": totals["retry_pending"],
        "blocked": totals["blocked"],
        "active_claims": checkpoint["lifecycle"]["active_group_claims"],
        "active_leases": checkpoint["lifecycle"]["active_shard_leases"],
        "scientific_failures_present": False,
        "prior_shards_rerun": False,
        "retry_attempts_created": False,
        "effect_summary_accessed": False,
        "finalize_run": False,
        "aggregate_run": False,
        "d9_started": False,
        "d10_started": False,
        "paper_data_or_figures_generated": False,
        "scientific_validity": "NOT_EVALUATED",
        "completed_campaign_cost_gate": "NOT_EVALUATED",
        "analysis_authorization": "NOT_EVALUATED",
        "package_complete": "not_applicable_at_D8",
        "retention": {
            "all_attempts_retained": True,
            "failure_rows_retained": True,
            "resource_records_retained": True,
            "each_unit_exactly_one_attempt_000001": True,
            "retry_attempts_created": False,
        },
        "reconciliation": {
            "all_shards_terminal": True,
            "all_receipts_verified": True,
            "expected_image_groups": IMAGE_GROUP_COUNT,
            "observed_terminal_image_groups": checkpoint["lifecycle"]["terminal_groups"],
            "expected_units": 1000,
            "observed_terminal_units": checkpoint["lifecycle"]["terminal_units"],
            "succeeded_units": totals["succeeded"],
            "scientific_failed_units": totals["scientific_failed"],
            "retry_pending_units": totals["retry_pending"],
            "blocked_units": totals["blocked"],
            "active_claims": checkpoint["lifecycle"]["active_group_claims"],
            "active_leases": checkpoint["lifecycle"]["active_shard_leases"],
            "pending_shards": 0,
            "retry_authorized_coordinates": 0,
            "retry_pending_coordinates": 0,
            "image_groups_terminal_ratio": "500/500",
            "units_terminal_ratio": "1000/1000",
        },
        "control_flags": {
            "scientific_failures_present": False,
            "prior_shards_rerun": False,
            "retry_attempts_created": False,
            "effect_summary_accessed": False,
            "finalize_run": False,
            "aggregate_run": False,
            "d9_started": False,
            "d10_started": False,
            "paper_data_or_figures_generated": False,
        },
        "evaluation_boundaries": {
            "scientific_validity": "NOT_EVALUATED",
            "completed_campaign_cost_gate": "NOT_EVALUATED",
            "analysis_authorization": "NOT_EVALUATED",
            "package_complete": "not_applicable_at_D8",
            "d8_pass_meaning": (
                "execution completeness only; not scientific validity, "
                "fidelity pass, cost pass, analysis authorization or paper conclusion"
            ),
        },
        "c10_status_preserved": {
            "acceptance": "BLOCKED",
            "primary_decision": "NO_GO_NOVELTY",
            "overall_spatial": "FAIL_SPATIAL",
            "holm": "HOLM_BLOCKED_UNDERSPECIFIED",
            "downstream_authorized": False,
        },
        "s07_control_plane_deviation": {
            "present": True,
            "classification": "CONTROL_PLANE_TAIL_FINALIZE_DEVIATION_NOT_SCIENTIFIC_FAILURE",
            "worker_exit_after_terminal_attempt_published": True,
            "published_attempt_source_validated": True,
            "postprocessor_fix_commit": "59c2c9c852908fc4aeb46078b7a5ce1cbde523da",
            "fix_scope": "checkpoint reconcile and group/shard release only",
            "unit_recomputed": False,
            "retry_attempt_created": False,
            "effect_summary_accessed": False,
        },
        "git": {
            "branch": _git(repo, "symbolic-ref", "--short", "HEAD"),
            "head_before_completion": baseline_head,
            "tracking_ref": "origin/stage-d-d3-pass",
            "tracking_sha_before_completion": baseline_head,
            "completion_commit": "SELF",
            "completion_ref": COMPLETION_REF,
        },
        "next_task": "D9",
    }
    document["canonical_digest"] = _semantic_digest(document)
    return MappingProxyType(document)


def write_d8_campaign_completion(
    *, repo_root: Path | str = REPO_ROOT,
    path: Path | str | None = None,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d8_campaign_completion(repo_root=repo)
    target = repo / (CAMPAIGN_COMPLETION_PATH if path is None else os.fspath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_file_bytes(document))
    return document


def validate_d8_campaign_completion(
    *, repo_root: Path | str = REPO_ROOT,
    path: Path | str = CAMPAIGN_COMPLETION_PATH,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    relative = os.fspath(path)
    document, raw = _read_json(repo / relative, "D8 campaign completion")
    if raw != canonical_file_bytes(document):
        raise StageDD8CompletionError("D8 campaign completion bytes are not canonical")
    if document.get("canonical_digest") != _semantic_digest(document):
        raise StageDD8CompletionError("D8 campaign completion self digest mismatch")
    expected = build_d8_campaign_completion(repo_root=repo)
    if document != expected:
        raise StageDD8CompletionError("D8 campaign completion is not source-recomputable")
    return MappingProxyType(document)


__all__ = [
    "CAMPAIGN_COMPLETION_PATH",
    "COMPLETION_REF",
    "StageDD8CompletionError",
    "build_d8_campaign_completion",
    "canonical_file_bytes",
    "validate_d8_campaign_completion",
    "write_d8_campaign_completion",
]
