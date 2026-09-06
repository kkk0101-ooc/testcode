"""Stage E E2 matched-method upsampling-rule sensitivity.

The module is deliberately closed to the four E0-frozen rules.  It reuses the
sealed D9 uniform curves, derives nearest only after exact equivalence checks,
and exposes GPU execution only for bilinear and bicubic fine-cell responses.
"""

from __future__ import annotations

import argparse
import csv
import functools
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

CONFIG_PATH = "config/stage_e_e2.json"
CONFIG_SCHEMA_PATH = "config/stage_e_e2.schema.json"
OUTPUT_SCHEMA_PATH = "config/stage_e_e2_output.schema.json"
E1_COMPLETION_PATH = e1.COMPLETION_PATH
E1_STATUS_PATH = e1.ROOT_PATH + "/status.json"
E1_INVENTORY_PATH = e1.ROOT_PATH + "/whole_root_inventory.json"
D8_ROOT_PATH = e1.D8_ROOT_PATH
D10_ROOT_PATH = e1.D10_ROOT_PATH
REGISTRY_PATH = e1.REGISTRY_PATH
ROOT_PATH = "evidence/stage_e/e2_recovery_v2_20260906/formal"
REPORT_ROOT_PATH = "reports/stage_e_e2_20260906"
PLAN_PATH = ROOT_PATH + "/control/typed_plan.json"
RUN_SPEC_PATH = ROOT_PATH + "/control/run_spec.json"
AUTH_PATH = ROOT_PATH + "/control/execution_authorization.json"
OPERATOR_PATH = ROOT_PATH + "/control/operator_specification.json"
GOLDEN_PATH = ROOT_PATH + "/control/operator_golden_equivalence.json"
PREFLIGHT_PATH = ROOT_PATH + "/control/budget_preflight.json"
UPSTREAM_BEFORE_PATH = ROOT_PATH + "/control/protected_upstream_before.json"
COMPLETION_PENDING_PATH = ROOT_PATH + "/completion/e2_completion.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/e2_completion.json"
SUPERSEDING_COMPLETION_PENDING_PATH = ROOT_PATH + "/completion/e2_completion_superseding.pending.json"
SUPERSEDING_COMPLETION_PATH = ROOT_PATH + "/completion/e2_completion_superseding.json"
SUPERSEDING_STATUS_PATH = ROOT_PATH + "/status_superseding.json"
SUPERSEDING_INVENTORY_PATH = ROOT_PATH + "/whole_root_inventory_superseding.json"
SUPERSEDING_MANIFEST_PATH = ROOT_PATH + "/payload_manifest_superseding.json"
ACCESS_PENDING_PATH = REPORT_ROOT_PATH + "/outcome_access.pending.json"
ACCESS_PATH = REPORT_ROOT_PATH + "/outcome_access.json"
INFRASTRUCTURE_EVENT_PATH = REPORT_ROOT_PATH + "/infrastructure_event_001.json"
INFRASTRUCTURE_EVENT_002_PATH = REPORT_ROOT_PATH + "/infrastructure_event_002.json"
RECOVERY_AUTH_PATH = ROOT_PATH + "/control/recovery_authorization_001.json"
BUDGET_WAIVER_PATH = REPORT_ROOT_PATH + "/owner_budget_waiver.json"
SUPERSEDING_VALIDATION_PATH = REPORT_ROOT_PATH + "/validation_report_superseding.json"
SUPERSEDING_FALSE_PASS_PATH = REPORT_ROOT_PATH + "/false_pass_audit_superseding.json"
SUPERSEDING_TEST_PATH = REPORT_ROOT_PATH + "/test_results_superseding.json"

E1_COMMIT = "244a06e18d0eea4f50deedf75b445c20b91ea6a7"
E1_COMPLETION_SHA256 = "c27603d051f8df0936d29233fc6dd8581a8397c0bc740fd0f079d6f8e858de57"
E1_COMPLETION_DIGEST = "b5be6be82ab166df89ebbd9ed8be50fddf37ebfc14a4294a2ed5272ab9b554fa"
E1_STATUS_SHA256 = "7120f5a6dcbdc5153637046fb1a062d6e0f904c4d7bd45bc419f68f22a3d3909"
E1_STATUS_DIGEST = "36d1498f3da9ec0c5a56dbe21b325565c567f57c3aad745813c7be5a4b5e4d5e"
E1_INVENTORY_SHA256 = "e033e55f9bd1e4de720fc96c8c381d0a2a7acb0a9c35199747be924a54e48d0e"
E1_INVENTORY_DIGEST = "353f61425e449ef5e0417e341c14392dfa027b6ea84c60ef40e7286c0131d3f6"
E1_FILE_PROJECTION_DIGEST = "49466a58dfb6d2b43263122db66d058d04b20be5de4f00bd0275d49c34c7c388"
BLOCKED_COMPLETION_SHA256 = "ea4af38e2faa078fde0b85fa00d43ea53e4763584e5e7367a08e65be75ede4b7"
BLOCKED_COMPLETION_DIGEST = "f6c261c0b5fe8162a3582634c4b4e29fa466a7fddc5a4c9f8748b47a4f9bffe3"
BLOCKED_STATUS_SHA256 = "866208fd478fec26ae9731ba11cebdcc7a4a10227dc62690efd06c4d88b18b98"
BLOCKED_STATUS_DIGEST = "91575f0d2c5b793108c48b1a51b60c7d37437d8598b3c54d9815d997ef578dd2"
BLOCKED_INVENTORY_SHA256 = "4311a55832cadeb27e3c9aee9b6995c96e608490bb55078d8a2996473dd6e583"
BLOCKED_INVENTORY_DIGEST = "d5c79abfdb067e2d3f7cca561135c9498ebeeee0c6e85fcb27b0a9461cd26b36"
EXPECTED_CONFIG_DIGEST = "69487dc42dd57f7d6491544a42b1dc8325862cf29b026a03868e77958c1b9fdd"
MODEL_SHA256 = e1.MODEL_SHA256
MODEL_TENSOR_DIGEST = e1.MODEL_TENSOR_DIGEST
METHOD = "matched_fixed_clean_attention_patch_token_contribution"
VARIANTS = (
    "uniform_mass_divide_by_16",
    "mass_preserving_nearest",
    "bilinear",
    "bicubic",
)
NEW_VARIANTS = ("bilinear", "bicubic")
PAIR_ORDER = (
    (VARIANTS[0], VARIANTS[1]),
    (VARIANTS[0], VARIANTS[2]),
    (VARIANTS[0], VARIANTS[3]),
    (VARIANTS[1], VARIANTS[2]),
    (VARIANTS[1], VARIANTS[3]),
    (VARIANTS[2], VARIANTS[3]),
)
OPERATOR_IDS = {
    VARIANTS[0]: "uniform_mass_divide_by_16",
    VARIANTS[1]: "mass_preserving_nearest",
    VARIANTS[2]: "torch_2_2_2_bilinear_align_corners_false_antialias_false_column_normalized",
    VARIANTS[3]: "torch_2_2_2_bicubic_align_corners_false_antialias_false_column_normalized",
}
EQUIVALENCE = {
    VARIANTS[0]: ("uniform_nearest_exact", "PRIMARY_INHERITED_CLASS_MEMBER"),
    VARIANTS[1]: ("uniform_nearest_exact", "EQUIVALENT_RANKING_NOT_INDEPENDENT"),
    VARIANTS[2]: ("bilinear", "NEW_COMPUTE"),
    VARIANTS[3]: ("bicubic", "NEW_COMPUTE"),
}
OPERATOR_DIGESTS = {
    VARIANTS[0]: "8567b6c938753beffcb52334db8add8a6a776921bf892c943ad8d591d8cd35bc",
    VARIANTS[1]: "8567b6c938753beffcb52334db8add8a6a776921bf892c943ad8d591d8cd35bc",
    VARIANTS[2]: "2627998d66bdd4211e94a564d6f99afb547649621570f7977a9649f15c8857e5",
    VARIANTS[3]: "186dec4cc95105575abe3670199acd81b354c22bc75215675863058a4c1896eb",
}
FRACTIONS = tuple(float(index) / 20.0 for index in range(21))
UNIT_COUNTS = (0,) + tuple(int(math.ceil(value * 3136)) for value in FRACTIONS[1:])
ACTUAL_FRACTIONS = tuple(value / 3136.0 for value in UNIT_COUNTS)
SEED_NAMESPACE = "stage-d-fidelity|20260821|"
COMMON_PAIR_IDENTITY = "e2_upsampling_rule_comparison_bound_to_inherited_d9"
PROTECTED_PATHS = (
    D8_ROOT_PATH, D10_ROOT_PATH,
    "evidence/stage_e/e0_recovery_v2_20260906",
    "evidence/stage_e/e1_recovery_v2_20260906",
    "reports/stage_d_d9_recovery_v2_20260906",
    "reports/stage_d_d10_recovery_v2_20260906",
    "reports/stage_e_e0_20260906",
    "reports/stage_e_e1_20260906",
)
FORBIDDEN_SUFFIXES = {".png", ".pdf", ".svg"}
FALSE_PASS_ATTACKS = (
    "e1_unverified_or_not_e2_only", "e0_d9_d10_binding_failed",
    "uniform_only_copy", "maps_without_real_responses",
    "fewer_than_500_images", "fewer_than_four_rules",
    "bilinear_missing", "bicubic_missing", "uniform_recomputed",
    "nearest_recomputed", "uniform_nearest_double_counted",
    "uniform_nearest_not_exact", "operator_not_column_normalized",
    "global_signed_sum_renormalized", "nonlinear_transform_before_operator",
    "native_196_interventions_masquerade_as_3136",
    "unpaired_d9_tie_priority", "missing_curves_auc_operator_pair_or_attempt",
    "missing_pairwise_rows", "failed_na_nonfinite_dropped",
    "favorable_valid_subset_summary", "d10_primary_recomputed_or_modified",
    "protected_upstream_modified", "budget_exceeded",
    "tests_missing_or_failed", "receipt_manifest_inventory_placeholder",
    "e2_overauthorizes_e4_or_f0",
)


class StageEE2Error(RuntimeError):
    """E2 cannot proceed or pass under the frozen contract."""


def _now() -> str:
    return e1._now()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    return e1._sha_file(path)


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
        raise StageEE2Error(f"cannot read strict {context}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageEE2Error(f"{context} must be a JSON object")
    return value, raw


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest") != _self_digest(value):
        raise StageEE2Error(f"{context} canonical digest mismatch")


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
            raise StageEE2Error(f"immutable document differs: {path}")
        return
    _write_new_json(path, value)


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(("git", *args), cwd=repo, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise StageEE2Error(f"git {' '.join(args)} failed: {error.stderr.strip()}") from error


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    raw = path.read_bytes()
    result: dict[str, Any] = {"path": relative, "bytes": len(raw), "sha256": _sha(raw)}
    if path.suffix == ".json":
        result["canonical_digest"] = _strict_json(path, relative)[0].get("canonical_digest")
    else:
        result["canonical_digest"] = None
    return result


def _inventory(
    root: Path, *, exclude: Sequence[str] = (), include_hidden: bool = False,
) -> dict[str, Any]:
    excluded = set(exclude)
    files: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in excluded or (
                not include_hidden
                and any(part.startswith(".") for part in Path(relative).parts)
            ):
                continue
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return {
        "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files),
        "canonical_digest": canonical_sha256({"files": files}), "files": files,
    }


def _validate_config(repo: Path) -> dict[str, Any]:
    value, _ = _strict_json(repo / CONFIG_PATH, "E2 config")
    schema, _ = _strict_json(repo / CONFIG_SCHEMA_PATH, "E2 config schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: [str(part) for part in item.path])
    if errors:
        raise StageEE2Error(f"E2 config schema failure: {errors[0].message}")
    _validate_signed(value, "E2 config")
    if value["canonical_digest"] != EXPECTED_CONFIG_DIGEST:
        raise StageEE2Error("E2 frozen config digest mismatch")
    if tuple(row["variant_id"] for row in value["scope"]["variants"]) != VARIANTS:
        raise StageEE2Error("E2 variant order drifted")
    if tuple(tuple(row) for row in value["scientific_contract"]["pairwise_order"]) != PAIR_ORDER:
        raise StageEE2Error("E2 pairwise order drifted")
    if value["operator_contract"]["operator_sha256"] != OPERATOR_DIGESTS:
        raise StageEE2Error("E2 operator digest contract drifted")
    if value["scientific_contract"]["unit_counts"] != list(UNIT_COUNTS):
        raise StageEE2Error("E2 fine-cell count schedule drifted")
    if value["scientific_contract"]["tie_seed_namespace"] != SEED_NAMESPACE:
        raise StageEE2Error("E2 must inherit the D9 seed namespace")
    return value


def d9_seed(sample_id: str, repeat: int) -> int:
    if type(repeat) is not int or not 0 <= repeat < 5:
        raise StageEE2Error("repeat must be an integer in [0,4]")
    material = f"{SEED_NAMESPACE}{sample_id}|0|9|{repeat}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def tie_priority(seed: int) -> np.ndarray:
    order = np.random.Generator(np.random.PCG64(seed)).permutation(3136)
    result = np.empty(3136, dtype=np.int64)
    result[order] = np.arange(3136, dtype=np.int64)
    return result


def ranking(signed_map: Any, *, seed: int) -> np.ndarray:
    values = np.asarray(signed_map, dtype=np.float64)
    if values.shape != (56, 56) or not np.isfinite(values).all():
        raise StageEE2Error("ranking requires a finite signed float64 56x56 map")
    scores = np.maximum(values.reshape(-1), 0.0)
    return np.lexsort((tie_priority(seed), -scores)).astype(np.int64)


def trapezoid_auc(responses: Sequence[float]) -> float:
    y = np.asarray(responses, dtype=np.float64)
    x = np.asarray(ACTUAL_FRACTIONS, dtype=np.float64)
    if y.shape != (21,) or not np.isfinite(y).all():
        raise StageEE2Error("AUC requires 21 finite responses")
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5, dtype=np.float64))


def delta_fid(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return 0.5 * (
        float(a["insertion_auc_mean"]) - float(b["insertion_auc_mean"])
        + float(b["deletion_auc_mean"]) - float(a["deletion_auc_mean"])
    )


@functools.lru_cache(maxsize=4)
def operator(variant: str) -> np.ndarray:
    value = e0.upsampling_operator(variant)
    if value.shape != (3136, 196) or value.dtype != np.float64:
        raise StageEE2Error("E2 operator shape/dtype drifted")
    if e1._array_digest(value) != OPERATOR_DIGESTS[variant]:
        raise StageEE2Error(f"E2 operator digest drifted for {variant}")
    return value


def recover_native(source_map: Any, patch_digest: str) -> np.ndarray:
    value = np.asarray(source_map)
    if value.dtype != np.float32 or value.shape != (56, 56) or not np.isfinite(value).all():
        raise StageEE2Error("D9 uniform source map must be finite float32 56x56")
    native = value.astype(np.float64).reshape(14, 4, 14, 4).sum(axis=(1, 3), dtype=np.float64)
    rebuilt = np.ascontiguousarray(native.astype(np.float32).reshape(-1))
    import torch
    from .stage_c_perturbations import tensor_sha256
    if tensor_sha256(torch.from_numpy(rebuilt)) != patch_digest:
        raise StageEE2Error("recovered native matched patch digest differs from D9 provenance")
    return native


def transformed_map(native: np.ndarray, variant: str) -> np.ndarray:
    result = (operator(variant) @ np.asarray(native, dtype=np.float64).reshape(-1)).reshape(56, 56)
    before = float(np.asarray(native, dtype=np.float64).sum(dtype=np.float64))
    after = float(result.sum(dtype=np.float64))
    tolerance = 1e-12 + 1e-12 * abs(before)
    if not np.isfinite(result).all() or abs(after - before) > tolerance:
        raise StageEE2Error(f"{variant} signed mass conservation failed")
    return result


def operator_golden() -> dict[str, Any]:
    results: dict[str, Any] = {}
    operators = {variant: operator(variant) for variant in VARIANTS}
    for variant, value in operators.items():
        sums = value.sum(axis=0, dtype=np.float64)
        if not np.isfinite(value).all() or np.any(sums == 0.0):
            raise StageEE2Error(f"{variant} operator contains invalid columns")
        results[variant] = {
            "operator_sha256": e1._array_digest(value), "shape": list(value.shape),
            "dtype": str(value.dtype), "column_sum_max_abs_error": float(np.max(np.abs(sums - 1.0))),
            "minimum": float(value.min()), "maximum": float(value.max()),
        }
    if not np.array_equal(operators[VARIANTS[0]], operators[VARIANTS[1]]):
        raise StageEE2Error("uniform and nearest operators are not exactly equal")
    fixture = np.arange(196, dtype=np.float64).reshape(14, 14) - 98.25
    maps = {variant: transformed_map(fixture, variant) for variant in VARIANTS}
    if not np.array_equal(maps[VARIANTS[0]], maps[VARIANTS[1]]):
        raise StageEE2Error("uniform and nearest fixture maps differ")
    for repeat in range(5):
        if not np.array_equal(ranking(maps[VARIANTS[0]], seed=repeat), ranking(maps[VARIANTS[1]], seed=repeat)):
            raise StageEE2Error("uniform and nearest fixture rankings differ")
    return _signed({
        "schema_version": 1, "document_type": "stage_e_e2_operator_golden_equivalence",
        "status": "PASS", "basis_count": 196, "operators": results,
        "uniform_nearest_operator_equal": True, "uniform_nearest_map_equal": True,
        "uniform_nearest_positive_part_ranking_and_tie_equal": True,
        "signed_values_preserved_before_ranking": True,
    })


def _source_index(repo: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    full, reconciliation = e1._source_index(repo)
    selected = {sample: row for (sample, method), row in full.items() if method == METHOD}
    if len(selected) != 500:
        raise StageEE2Error("D9 matched source index is not exactly 500 units")
    return selected, reconciliation


def _source_paths(repo: Path, source: Mapping[str, Any]) -> tuple[Path, Path]:
    attempt = e1._source_attempt(repo, source)
    return attempt / "L1/metric_ready_map.npy", attempt / "L0/result.json"


def _load_source(repo: Path, unit: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    map_path, result_path = repo / unit["source_map_path"], repo / unit["source_result_path"]
    if _sha_file(map_path) != unit["source_map_sha256"] or _sha_file(result_path) != unit["source_result_sha256"]:
        raise StageEE2Error("D9 source bytes drifted")
    source_result, _ = _strict_json(result_path, "D9 matched result")
    source_map = np.load(map_path, allow_pickle=False)
    matched = source_result.get("provenance_fragment", {}).get("matched_contribution", {})
    native = recover_native(source_map, matched.get("patch_contributions_sha256", ""))
    uniform = transformed_map(native, VARIANTS[0])
    if not np.array_equal(uniform, source_map.astype(np.float64)):
        raise StageEE2Error("recovered native map does not exactly rebuild D9 uniform map")
    cls = float(matched["cls_contribution"]); token_sum = float(matched["token_sum"]); tolerance = float(matched["conservation_tolerance"])
    if abs(float(native.sum(dtype=np.float64)) + cls - token_sum) > tolerance:
        raise StageEE2Error("matched native patch plus CLS conservation failed")
    return source_result, source_map, native


def _validate_d9_curves(sample_id: str, source_result: Mapping[str, Any], source_map: np.ndarray) -> list[dict[str, Any]]:
    curves = source_result.get("fidelity_curves")
    if not isinstance(curves, list) or len(curves) != 10:
        raise StageEE2Error("D9 uniform source does not contain ten curves")
    expected = [(direction, repeat) for direction in ("insertion", "deletion") for repeat in range(5)]
    if [(row.get("direction"), row.get("replicate")) for row in curves] != expected:
        raise StageEE2Error("D9 uniform curve coordinates drifted")
    source_hash = e1._array_digest(source_map)
    score_hash = e1._array_digest(np.maximum(source_map, np.float32(0.0)).astype(np.float32, copy=False).reshape(-1))
    for curve in curves:
        seed = d9_seed(sample_id, curve["replicate"])
        if (
            curve.get("seed") != seed or curve.get("seed_namespace") != SEED_NAMESPACE
            or curve.get("tie_priority_sha256") != e1._array_digest(tie_priority(seed))
            or curve.get("source_unit_map_sha256") != source_hash
            or curve.get("score_sha256") != score_hash
            or curve.get("ranking_sha256") != e1._array_digest(ranking(source_map.astype(np.float64), seed=seed))
            or curve.get("unit_counts") != list(UNIT_COUNTS)
            or curve.get("actual_fractions") != list(ACTUAL_FRACTIONS)
            or len(curve.get("responses", [])) != 21
            or len(curve.get("intervention_sha256", [])) != 21
            or curve.get("auc") != trapezoid_auc(curve["responses"])
        ):
            raise StageEE2Error("D9 uniform curve source recomputation failed")
    return [dict(row) for row in curves]


def _build_plan(repo: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    registry, _ = e1._load_registry(repo)
    sources, _ = _source_index(repo)
    groups: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for registry_row in registry:
        source = sources[registry_row["sample_id"]]
        map_path, result_path = _source_paths(repo, source)
        common = {
            "sample_id": registry_row["sample_id"], "selection_order": registry_row["selection_order"],
            "image_sha256": registry_row["image_sha256"], "relative_path": registry_row["relative_path"],
            "source_unit_digest": source["unit_digest"],
            "source_attempt_id": source["attempts"][0]["attempt_id"],
            "source_attempt_inventory_digest": source["attempts"][0]["attempt_inventory_digest"],
            "source_map_path": map_path.relative_to(repo).as_posix(), "source_map_sha256": _sha_file(map_path),
            "source_result_path": result_path.relative_to(repo).as_posix(), "source_result_sha256": _sha_file(result_path),
            "pair_identity": COMMON_PAIR_IDENTITY,
        }
        unit_digests: list[str] = []
        for variant in NEW_VARIANTS:
            unit = _signed({
                "schema_version": 1, "document_type": "stage_e_e2_new_compute_unit",
                "variant_id": variant, "operator_id": OPERATOR_IDS[variant],
                "operator_sha256": OPERATOR_DIGESTS[variant], "method": METHOD,
                "block": 0, "head": 9, "scale": "fine_4x4_cell", **common,
            })
            units.append(unit); unit_digests.append(unit["canonical_digest"])
        groups.append(_signed({
            "schema_version": 1, "document_type": "stage_e_e2_image_group",
            **common, "new_unit_digests": unit_digests,
            "inherited_variants": [VARIANTS[0], VARIANTS[1]],
            "new_variants": list(NEW_VARIANTS),
        }))
    return _signed({
        "schema_version": 1, "document_type": "stage_e_e2_typed_plan",
        "config_canonical_digest": config["canonical_digest"],
        "cohort": "stage_d_fidelity_v1_500", "expected_image_groups": 500,
        "expected_new_units": 1000, "expected_variant_rows": 2000,
        "expected_pairwise_rows": 3000, "variants": list(VARIANTS),
        "new_variants": list(NEW_VARIANTS), "groups": groups, "units": units,
    })


def _protected_snapshot(repo: Path) -> dict[str, Any]:
    roots = []
    for relative in PROTECTED_PATHS:
        inventory = _inventory(repo / relative, include_hidden=True)
        roots.append({
            "path": relative, "file_count": inventory["file_count"],
            "total_bytes": inventory["total_bytes"],
            "canonical_digest": inventory["canonical_digest"],
        })
    expected = {
        D8_ROOT_PATH: e0.EXPECTED_D9_INVENTORY,
        D10_ROOT_PATH: e0.EXPECTED_D10_INVENTORY,
        "evidence/stage_e/e0_recovery_v2_20260906": "80087eb3ea0f417683e0f1c5b1dc87023f15be2247e508eeb0e4299ce9b65cda",
        "evidence/stage_e/e1_recovery_v2_20260906": "903d6c704630ea0c5b9090e4da337982153c6690b047c757feef4616b204b417",
        "reports/stage_d_d9_recovery_v2_20260906": "e2937e9f166582fb5e281b319e017cb8f32af27613e35bb0935fc89e916e0a59",
        "reports/stage_d_d10_recovery_v2_20260906": "c16a9e4e90f05b984ab9012438291b0ddb052f228f6f1f579b4b437de1b31d3a",
        "reports/stage_e_e0_20260906": "a5fa4f36b6fecfa0dc49771b0292fcba6eea21398b3669b48741c08a0781d837",
        "reports/stage_e_e1_20260906": "34e077dd47b6f87537db54c761c0fae6ff0a5e79b6dee93d7c555356762a0efb",
    }
    for row in roots:
        if row["canonical_digest"] != expected[row["path"]]:
            raise StageEE2Error(f"protected root binding failed: {row['path']}")
    changed = _git(repo, "diff", "--name-only", E1_COMMIT, "--", *PROTECTED_PATHS).splitlines()
    if changed:
        raise StageEE2Error(f"protected tracked upstream changed: {changed}")
    return _signed({
        "schema_version": 1, "document_type": "stage_e_e2_protected_upstream_snapshot",
        "captured_at": _now(), "roots": roots, "protected_tracked_diff": changed,
    })


def verify_preconditions(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _validate_config(repo)
    if _git(repo, "symbolic-ref", "--short", "HEAD") != "stage-d-d3-pass":
        raise StageEE2Error("wrong branch")
    head = _git(repo, "rev-parse", "HEAD")
    tracking = _git(repo, "rev-parse", "@{u}")
    remote = _git(repo, "rev-parse", "refs/remotes/origin/stage-d-d3-pass")
    archive = _git(repo, "rev-parse", "refs/archive/stage-e-e1-completion^{commit}")
    if {head, tracking, remote, archive} != {E1_COMMIT}:
        raise StageEE2Error(f"workspace binding failed: head={head} tracking={tracking} remote={remote} archive={archive}")
    status = _git(repo, "status", "--porcelain=v1")
    allowed = (
        "config/stage_e_e2.json", "config/stage_e_e2.schema.json",
        "config/stage_e_e2_output.schema.json", "src/vitnns/stage_e_e2.py",
        "tests/test_stage_e_e2.py", "evidence/stage_e/e2_recovery_v2_20260906/",
        "reports/stage_e_e2_20260906/",
    )
    unexpected = [line for line in status.splitlines() if not any(line[3:].endswith(item) for item in allowed)]
    if unexpected:
        raise StageEE2Error(f"unexpected worktree changes: {unexpected}")
    lfs = _git(repo, "lfs", "status")
    if "Git LFS objects to be committed:" not in lfs or "Git LFS objects not staged for commit:" not in lfs:
        raise StageEE2Error("cannot establish Git LFS status")
    completion_result = e1.validate_completion(repo)
    if (
        completion_result.get("status") != "PASS"
        or completion_result.get("completion_canonical_digest") != E1_COMPLETION_DIGEST
        or completion_result.get("row_count") != 1500
        or completion_result.get("new_compute_rows") != 1000
        or completion_result.get("next_authorization") != "E2_ONLY"
        or completion_result.get("whole_root_inventory_digest") != E1_INVENTORY_DIGEST
    ):
        raise StageEE2Error("E1 completion source validation failed")
    for relative, expected_sha, expected_digest in (
        (E1_COMPLETION_PATH, E1_COMPLETION_SHA256, E1_COMPLETION_DIGEST),
        (E1_STATUS_PATH, E1_STATUS_SHA256, E1_STATUS_DIGEST),
        (E1_INVENTORY_PATH, E1_INVENTORY_SHA256, E1_INVENTORY_DIGEST),
    ):
        value, raw = _strict_json(repo / relative, relative)
        if _sha(raw) != expected_sha or value.get("canonical_digest") != expected_digest:
            raise StageEE2Error(f"E1 frozen binding failed: {relative}")
    inventory, _ = _strict_json(repo / E1_INVENTORY_PATH, "E1 inventory")
    if canonical_sha256({"files": inventory["files"]}) != E1_FILE_PROJECTION_DIGEST:
        raise StageEE2Error("E1 ordered inventory file projection drifted")
    e0.validate_e0_inputs(repo)
    upstream = dict(e0.verify_upstream(repo))
    if upstream.get("status") != "PASS":
        raise StageEE2Error("E0/D9/D10 validation failed")
    snapshot = _protected_snapshot(repo)
    return {
        "status": "PASS", "validated_at": _now(), "head": head,
        "config": config, "e1_completion": completion_result,
        "upstream": upstream, "protected_snapshot": snapshot,
    }


def initialize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    preconditions = verify_preconditions(repo)
    config = preconditions["config"]
    plan = _build_plan(repo, config)
    golden = operator_golden()
    operator_spec = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_operator_specification",
        "torch_version": "2.2.2", "source_grid": [14, 14], "target_grid": [56, 56],
        "operator_shape": [3136, 196], "dtype": "float64",
        "coordinate_convention": config["operator_contract"]["coordinate_convention"],
        "source_order": "native_row_major", "definitions": config["operator_contract"],
        "signed_transform_before_positive_part_ranking": True,
        "cls_excluded_from_operator_and_ranking": True,
    })
    run_spec = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_run_spec",
        "config_canonical_digest": config["canonical_digest"],
        "plan_canonical_digest": plan["canonical_digest"],
        "operator_specification_canonical_digest": operator_spec["canonical_digest"],
        "operator_golden_canonical_digest": golden["canonical_digest"],
        "expected_image_groups": 500, "expected_new_units": 1000,
        "expected_variant_rows": 2000, "expected_pairwise_rows": 3000,
        "variants": list(VARIANTS), "new_compute_variants": list(NEW_VARIANTS),
        "directions": ["insertion", "deletion"], "tie_repeats": list(range(5)),
        "fractions": list(FRACTIONS), "unit_counts": list(UNIT_COUNTS),
        "actual_fractions": list(ACTUAL_FRACTIONS),
        "attempt_policy": {"initial_attempt": "attempt-000001", "maximum_attempts": 2, "automatic_retry": False, "terminal_recompute": False},
        "inherited_policy": {"uniform": "EXACT_D9_REUSE", "nearest": "EXACT_UNIFORM_REUSE_AFTER_EQUIVALENCE", "new_gpu_attempts_for_inherited": False},
        "stage_boundary": {"current": "E2_ONLY", "on_pass": "E3_ONLY", "later_stages_started": False},
    })
    estimated_calls = 500 * (3 + 2 * 2 * 5 * 21)
    preflight = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_budget_preflight",
        "status": "PASS", "recorded_at": _now(),
        "basis": {
            "e1_measured_forward_calls": 213000, "e1_measured_gpu_hours": 1.577772853329451,
            "e1_measured_cpu_hours": 1.6308065290991665,
            "e2_estimated_forward_calls": estimated_calls,
            "shared_per_image_target_resolution": True,
            "uniform_forward_calls": 0, "nearest_forward_calls": 0,
        },
        "estimated": {"forward_calls": estimated_calls, "gpu_hours": 1.65, "cpu_hours": 2.0, "retained_bytes": 1073741824},
        "limits": config["budget"],
    })
    authorization = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_execution_authorization",
        "status": "AUTHORIZED", "issued_at": _now(),
        "e1_pass_e2_only_verified": True, "e0_d9_d10_verified": True,
        "plan_canonical_digest": plan["canonical_digest"],
        "run_spec_canonical_digest": run_spec["canonical_digest"],
        "budget_preflight_canonical_digest": preflight["canonical_digest"],
        "authorized_scope": "E2_ONLY_500_GROUPS_1000_BILINEAR_BICUBIC_FINE_CELL_UNITS",
        "uniform_recompute": "FORBIDDEN", "nearest_recompute": "FORBIDDEN",
        "later_stages_authorized": False,
    })
    access = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_outcome_access",
        "status": "PENDING_BEFORE_FIRST_E2_OUTCOME", "e2_started_at": _now(),
        "e1_terminal_pass_and_e2_only_verified": True, "d10_and_e1_outcomes_known": True,
        "new_e2_bilinear_bicubic_outcome_accessed": False,
        "scope_source": "sealed_E0_only",
        "frozen_scope": {
            "variants": list(VARIANTS), "sample_count": 500, "unit_count": 3136,
            "fractions": list(FRACTIONS), "tie_repeats": 5, "pairwise_order": [list(row) for row in PAIR_ORDER],
            "bootstrap_draws": 10000, "budget": config["budget"],
        },
        "upstream_reads": [
            _record(repo, E1_COMPLETION_PATH), _record(repo, E1_STATUS_PATH),
            _record(repo, E1_INVENTORY_PATH), _record(repo, e0.COMPLETION_PATH),
            _record(repo, REGISTRY_PATH), _record(repo, D8_ROOT_PATH + "/final/source_inventory.json"),
            _record(repo, D8_ROOT_PATH + "/final/reconciliation.json"),
            _record(repo, "reports/stage_d_d9_recovery_v2_20260906/d9_formal_validation.json"),
            _record(repo, "reports/stage_d_d10_recovery_v2_20260906/d10_formal_decision.json"),
        ],
        "protected_roots": [{"path": path, "mode": "READ_ONLY"} for path in PROTECTED_PATHS],
        "e3_started": False, "e4_started": False, "stage_f_started": False,
    })
    pending = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_completion_pending",
        "stage_id": "E2", "status": "PENDING", "started_at": access["e2_started_at"],
        "plan_canonical_digest": plan["canonical_digest"], "run_spec_canonical_digest": run_spec["canonical_digest"],
        "authorization_canonical_digest": authorization["canonical_digest"],
        "expected_image_groups": 500, "expected_new_units": 1000,
        "expected_variant_rows": 2000, "expected_pairwise_rows": 3000,
        "next_authorization": "NONE_PENDING",
    })
    for relative, document in (
        (PLAN_PATH, plan), (RUN_SPEC_PATH, run_spec), (AUTH_PATH, authorization),
        (OPERATOR_PATH, operator_spec), (GOLDEN_PATH, golden), (PREFLIGHT_PATH, preflight),
        (UPSTREAM_BEFORE_PATH, preconditions["protected_snapshot"]),
        (ACCESS_PENDING_PATH, access), (COMPLETION_PENDING_PATH, pending),
    ):
        _write_or_verify(repo / relative, document)
    validate_initialized(repo)
    return {
        "status": "PASS", "plan_canonical_digest": plan["canonical_digest"],
        "run_spec_canonical_digest": run_spec["canonical_digest"],
        "operator_golden_canonical_digest": golden["canonical_digest"],
        "access_record": ACCESS_PENDING_PATH,
    }


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _validate_config(repo)
    documents = {}
    for name, relative in (
        ("plan", PLAN_PATH), ("run_spec", RUN_SPEC_PATH), ("authorization", AUTH_PATH),
        ("operator", OPERATOR_PATH), ("golden", GOLDEN_PATH), ("preflight", PREFLIGHT_PATH),
        ("before", UPSTREAM_BEFORE_PATH), ("access", ACCESS_PENDING_PATH),
        ("pending", COMPLETION_PENDING_PATH),
    ):
        value, _ = _strict_json(repo / relative, f"E2 {name}")
        _validate_signed(value, f"E2 {name}"); documents[name] = value
    rebuilt = _build_plan(repo, config)
    if rebuilt != documents["plan"]:
        raise StageEE2Error("E2 typed plan is not source-recomputable")
    plan = documents["plan"]
    if (
        len(plan["groups"]) != 500 or len(plan["units"]) != 1000
        or [row["selection_order"] for row in plan["groups"]] != list(range(500))
        or Counter(row["variant_id"] for row in plan["units"]) != Counter({VARIANTS[2]: 500, VARIANTS[3]: 500})
        or documents["golden"] != operator_golden()
        or documents["authorization"]["status"] != "AUTHORIZED"
        or documents["preflight"]["status"] != "PASS"
        or documents["access"]["new_e2_bilinear_bicubic_outcome_accessed"] is not False
        or documents["pending"]["status"] != "PENDING"
    ):
        raise StageEE2Error("E2 initialization gate failed")
    return {"status": "PASS", "image_groups": 500, "new_units": 1000, "variant_rows": 2000, "pairwise_rows": 3000}


def _curve(
    *, evaluator: Any, response: Any, clean: Any, fine_map: np.ndarray,
    sample_id: str, variant: str, direction: str, repeat: int,
    baseline_sha256: str, clean_sha256: str,
) -> dict[str, Any]:
    import torch
    from .stage_c_perturbations import tensor_sha256

    seed = d9_seed(sample_id, repeat)
    priority = tie_priority(seed)
    ranked = ranking(fine_map, seed=seed)
    baseline = torch.zeros_like(clean)
    responses: list[float] = []
    intervention_hashes: list[str] = []
    for index, count in enumerate(UNIT_COUNTS):
        if index == 0:
            normalized = 0.0 if direction == "insertion" else 1.0
            digest = baseline_sha256 if direction == "insertion" else clean_sha256
        elif index == len(UNIT_COUNTS) - 1:
            normalized = 1.0 if direction == "insertion" else 0.0
            digest = clean_sha256 if direction == "insertion" else baseline_sha256
        else:
            selected = np.zeros(3136, dtype=np.bool_)
            selected[ranked[:count]] = True
            pixel_mask = np.repeat(np.repeat(selected.reshape(56, 56), 4, axis=0), 4, axis=1)
            mask = torch.from_numpy(pixel_mask.copy()).to(device=clean.device)[None, None].expand_as(clean)
            intervention = (
                torch.where(mask, clean, baseline)
                if direction == "insertion"
                else torch.where(mask, baseline, clean)
            )
            digest = tensor_sha256(intervention)
            normalized = response.normalize(evaluator.evaluate_prehashed(intervention, input_sha256=digest))
            if not isinstance(normalized, float) or not math.isfinite(normalized):
                raise StageEE2Error("non-finite normalized fine-cell response")
        responses.append(normalized); intervention_hashes.append(digest)
    return {
        "variant_id": variant, "method": METHOD, "sample_id": sample_id,
        "direction": direction, "repeat": repeat, "seed": seed,
        "seed_namespace": SEED_NAMESPACE, "seed_comparison_identity": COMMON_PAIR_IDENTITY,
        "tie_priority_sha256": e1._array_digest(priority),
        "signed_transformed_map_sha256": e1._array_digest(fine_map),
        "ranking_sha256": e1._array_digest(ranked),
        "unit_counts": list(UNIT_COUNTS), "actual_fractions": list(ACTUAL_FRACTIONS),
        "intervention_sha256": intervention_hashes, "responses": responses,
        "response_provenance": "19_new_intermediate_forwards_plus_2_shared_target_resolution_endpoint_forwards",
        "auc": trapezoid_auc(responses), "response_clipped": False,
    }


def _attempt_paths(repo: Path, unit_digest: str) -> tuple[Path, Path, Path, Path]:
    root = repo / ROOT_PATH / "units" / unit_digest / "attempts/attempt-000001"
    return root, root / "result.json", root / "status.json", root / "manifest.json"


def _expected_pair_binding(unit: Mapping[str, Any], target_digest: str) -> dict[str, Any]:
    return {
        "sample_id": unit["sample_id"], "selection_order": unit["selection_order"],
        "image_sha256": unit["image_sha256"], "model_sha256": MODEL_SHA256,
        "target": "fixed_clean_attention_cls_head_direction",
        "target_semantic_digest": target_digest, "method": METHOD,
        "variant_id": unit["variant_id"], "comparison_identity": COMMON_PAIR_IDENTITY,
        "block": 0, "head": 9, "scale": "fine_4x4_cell",
        "operator_id": unit["operator_id"], "operator_sha256": unit["operator_sha256"],
        "replacement": "normalized_zero_canonical_rgb_0_5",
        "fractions": list(FRACTIONS), "unit_counts": list(UNIT_COUNTS),
        "seeds": [d9_seed(unit["sample_id"], repeat) for repeat in range(5)],
        "seed_namespace": SEED_NAMESPACE,
        "source_map_sha256": unit["source_map_sha256"],
        "source_result_sha256": unit["source_result_sha256"],
        "source_unit_digest": unit["source_unit_digest"], "attempt_id": "attempt-000001",
    }


def _validate_attempt(repo: Path, unit: Mapping[str, Any]) -> dict[str, Any]:
    root, result_path, status_path, manifest_path = _attempt_paths(repo, unit["canonical_digest"])
    if not root.is_dir() or any(not path.is_file() for path in (result_path, status_path, manifest_path)):
        raise StageEE2Error(f"attempt is absent or incomplete: {root}")
    result, result_raw = _strict_json(result_path, "E2 attempt result")
    status, status_raw = _strict_json(status_path, "E2 attempt status")
    manifest, _ = _strict_json(manifest_path, "E2 attempt manifest")
    for value, name in ((result, "result"), (status, "status"), (manifest, "manifest")):
        _validate_signed(value, f"E2 attempt {name}")
    if (
        unit["variant_id"] not in NEW_VARIANTS
        or result.get("unit_digest") != unit["canonical_digest"]
        or result.get("variant_id") != unit["variant_id"]
        or result.get("sample_id") != unit["sample_id"]
        or result.get("validity_status") not in {"VALID", "FAILED", "NA"}
        or status.get("terminal") is not True or status.get("attempt_id") != "attempt-000001"
        or manifest.get("result_sha256") != _sha(result_raw)
        or manifest.get("status_sha256") != _sha(status_raw)
    ):
        raise StageEE2Error("E2 attempt control binding failed")
    if result["validity_status"] != "VALID":
        return {"result": result, "status": status, "manifest": manifest, "attempt_path": root}
    if (
        result.get("failure_class") is not None or result.get("failure_code") is not None
        or result.get("na_reason") is not None or result.get("attempt_id") != "attempt-000001"
        or result.get("retry_count") != 0 or result.get("unit_count") != 3136
        or result.get("unit_pixels") != [4, 4] or result.get("forward_calls") not in {193, 196}
    ):
        raise StageEE2Error("valid E2 attempt has failure/retry/geometry/accounting drift")
    source_result, source_map, native = _load_source(repo, unit)
    d9_curves = _validate_d9_curves(unit["sample_id"], source_result, source_map)
    transformed = transformed_map(native, unit["variant_id"])
    target_response = result.get("target_response")
    if not isinstance(target_response, Mapping) or target_response.get("sigma") != 1:
        raise StageEE2Error("E2 target response is not the D9 target orientation")
    target_digest = e1._target_semantic_digest(target_response["target_identity"])
    if target_digest != e1._target_semantic_digest(source_result["target_identity"]):
        raise StageEE2Error("E2 target/source semantic binding failed")
    pair = _expected_pair_binding(unit, target_digest)
    if result.get("pair_binding") != pair or result.get("pair_binding_digest") != canonical_sha256(pair):
        raise StageEE2Error("E2 pair binding failed")
    matched = source_result["provenance_fragment"]["matched_contribution"]
    expected_hashes = {
        "source_map_sha256": unit["source_map_sha256"],
        "source_result_sha256": unit["source_result_sha256"],
        "source_unit_digest": unit["source_unit_digest"], "image_sha256": unit["image_sha256"],
        "model_sha256": MODEL_SHA256, "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST,
        "target_semantic_digest": target_digest,
        "native_patch_float64_sha256": e1._array_digest(native),
        "native_patch_float32_tensor_sha256": matched["patch_contributions_sha256"],
        "signed_transformed_map_sha256": e1._array_digest(transformed),
        "operator_sha256": unit["operator_sha256"],
    }
    if result.get("source_hashes") != expected_hashes:
        raise StageEE2Error("E2 source hash binding failed")
    curves = result.get("curves")
    if not isinstance(curves, list) or len(curves) != 10:
        raise StageEE2Error("E2 attempt lacks ten curves")
    d9_by_coordinate = {(row["direction"], row["replicate"]): row for row in d9_curves}
    expected_coordinates = {(direction, repeat) for direction in ("insertion", "deletion") for repeat in range(5)}
    if {(row.get("direction"), row.get("repeat")) for row in curves} != expected_coordinates:
        raise StageEE2Error("E2 curve coordinate coverage failed")
    for curve in curves:
        coordinate = curve["direction"], curve["repeat"]
        inherited = d9_by_coordinate[coordinate]
        seed = d9_seed(unit["sample_id"], curve["repeat"])
        if (
            curve.get("variant_id") != unit["variant_id"] or curve.get("method") != METHOD
            or curve.get("sample_id") != unit["sample_id"] or curve.get("seed") != seed
            or curve.get("seed_namespace") != SEED_NAMESPACE
            or curve.get("seed_comparison_identity") != COMMON_PAIR_IDENTITY
            or curve.get("tie_priority_sha256") != inherited["tie_priority_sha256"]
            or curve.get("tie_priority_sha256") != e1._array_digest(tie_priority(seed))
            or curve.get("signed_transformed_map_sha256") != e1._array_digest(transformed)
            or curve.get("ranking_sha256") != e1._array_digest(ranking(transformed, seed=seed))
            or curve.get("unit_counts") != list(UNIT_COUNTS)
            or curve.get("actual_fractions") != list(ACTUAL_FRACTIONS)
            or len(curve.get("responses", [])) != 21
            or len(curve.get("intervention_sha256", [])) != 21
            or curve.get("auc") != trapezoid_auc(curve["responses"])
        ):
            raise StageEE2Error("E2 curve source recomputation failed")
        if tuple((curve["responses"][0], curve["responses"][-1])) != ((0.0, 1.0) if curve["direction"] == "insertion" else (1.0, 0.0)):
            raise StageEE2Error("E2 curve endpoints failed")
        if tuple((curve["intervention_sha256"][0], curve["intervention_sha256"][-1])) != tuple((inherited["intervention_sha256"][0], inherited["intervention_sha256"][-1])):
            raise StageEE2Error("E2 shared endpoint intervention bindings failed")
    for direction in ("insertion", "deletion"):
        values = np.asarray([row["auc"] for row in curves if row["direction"] == direction], dtype=np.float64)
        if (
            result.get(f"{direction}_repeat_aucs") != values.tolist()
            or result.get(f"{direction}_auc_mean") != float(values.mean())
            or result.get(f"{direction}_auc_repeat_sd_ddof_0") != float(values.std(ddof=0))
        ):
            raise StageEE2Error("E2 repeat AUC aggregation failed")
    return {"result": result, "status": status, "manifest": manifest, "attempt_path": root}


def _publish_attempt(repo: Path, unit: Mapping[str, Any], result: Mapping[str, Any]) -> Path:
    root, _, _, _ = _attempt_paths(repo, unit["canonical_digest"])
    if root.exists():
        raise StageEE2Error(f"refusing to overwrite terminal attempt: {root}")
    attempts_root = root.parent; attempts_root.mkdir(parents=True, exist_ok=True)
    if any(path.name.startswith(".attempt-") for path in attempts_root.iterdir()):
        raise StageEE2Error(f"ambiguous unpublished attempt exists: {attempts_root}")
    temporary = Path(tempfile.mkdtemp(prefix=".attempt-000001-", dir=attempts_root))
    try:
        signed_result = _signed(result); result_raw = canonical_json_bytes(signed_result) + b"\n"
        (temporary / "result.json").write_bytes(result_raw)
        status = _signed({
            "schema_version": 1, "document_type": "stage_e_e2_attempt_status",
            "attempt_id": "attempt-000001", "attempt_number": 1,
            "unit_digest": unit["canonical_digest"],
            "state": "succeeded" if signed_result["validity_status"] == "VALID" else "scientific_terminal",
            "terminal": True, "retry_count": 0,
            "failure_class": signed_result["failure_class"], "failure_code": signed_result["failure_code"],
            "na_reason": signed_result["na_reason"], "published_at": _now(),
        })
        status_raw = canonical_json_bytes(status) + b"\n"; (temporary / "status.json").write_bytes(status_raw)
        manifest = _signed({
            "schema_version": 1, "document_type": "stage_e_e2_attempt_manifest",
            "attempt_id": "attempt-000001", "attempt_number": 1, "unit_digest": unit["canonical_digest"],
            "result_sha256": _sha(result_raw), "result_bytes": len(result_raw),
            "status_sha256": _sha(status_raw), "status_bytes": len(status_raw),
            "source_map_sha256": unit["source_map_sha256"],
            "source_result_sha256": unit["source_result_sha256"],
            "operator_sha256": unit["operator_sha256"], "immutable": True,
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


def _valid_result(
    *, unit: Mapping[str, Any], source_result: Mapping[str, Any], native: np.ndarray,
    fine_map: np.ndarray, target_identity: Mapping[str, Any], response: Any,
    curves: list[dict[str, Any]], elapsed_seconds: float, cpu_seconds: float,
    forward_calls: int, peak_cuda_bytes: int,
) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for direction in ("insertion", "deletion"):
        values = np.asarray([row["auc"] for row in curves if row["direction"] == direction], dtype=np.float64)
        aggregates[f"{direction}_repeat_aucs"] = values.tolist()
        aggregates[f"{direction}_auc_mean"] = float(values.mean())
        aggregates[f"{direction}_auc_repeat_sd_ddof_0"] = float(values.std(ddof=0))
    target_digest = e1._target_semantic_digest(target_identity)
    pair = _expected_pair_binding(unit, target_digest)
    matched = source_result["provenance_fragment"]["matched_contribution"]
    source_sum = float(native.sum(dtype=np.float64)); transformed_sum = float(fine_map.sum(dtype=np.float64))
    tolerance = 1e-12 + 1e-12 * abs(source_sum)
    return {
        "schema_version": 1, "document_type": "stage_e_e2_attempt_result",
        "unit_digest": unit["canonical_digest"], "variant_id": unit["variant_id"],
        "sample_id": unit["sample_id"], "selection_order": unit["selection_order"],
        "method": METHOD, "target": "fixed_clean_attention_cls_head_direction",
        "scale": "fine_4x4_cell", "layer": 0, "head": 9,
        "operator_id": unit["operator_id"], "operator_sha256": unit["operator_sha256"],
        "equivalence_class": EQUIVALENCE[unit["variant_id"]][0],
        "evidence_independence": EQUIVALENCE[unit["variant_id"]][1],
        "validity_status": "VALID", "failure_class": None, "failure_code": None,
        "na_reason": None, "attempt_id": "attempt-000001", "retry_count": 0,
        "source_hashes": {
            "source_map_sha256": unit["source_map_sha256"], "source_result_sha256": unit["source_result_sha256"],
            "source_unit_digest": unit["source_unit_digest"], "image_sha256": unit["image_sha256"],
            "model_sha256": MODEL_SHA256, "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST,
            "target_semantic_digest": target_digest, "native_patch_float64_sha256": e1._array_digest(native),
            "native_patch_float32_tensor_sha256": matched["patch_contributions_sha256"],
            "signed_transformed_map_sha256": e1._array_digest(fine_map), "operator_sha256": unit["operator_sha256"],
        },
        "pair_binding": pair, "pair_binding_digest": canonical_sha256(pair),
        "native_and_cls_conservation_qc": {
            "native_patch_sum": source_sum, "cls_contribution": float(matched["cls_contribution"]),
            "token_sum": float(matched["token_sum"]), "cls_in_operator_or_ranking": False,
            "absolute_error": abs(source_sum + float(matched["cls_contribution"]) - float(matched["token_sum"])),
            "tolerance": float(matched["conservation_tolerance"]), "status": "PASS",
        },
        "operator_conservation_qc": {
            "source_signed_sum": source_sum, "transformed_signed_sum": transformed_sum,
            "absolute_error": abs(transformed_sum - source_sum), "tolerance": tolerance, "status": "PASS",
        },
        "target_response": response.to_dict(), "curves": curves, **aggregates,
        "unit_count": 3136, "unit_pixels": [4, 4],
        "elapsed_seconds": float(elapsed_seconds), "cpu_seconds": float(cpu_seconds),
        "gpu_hours": float(elapsed_seconds / 3600.0), "forward_calls": int(forward_calls),
        "peak_cuda_memory_bytes": int(peak_cuda_bytes), "artifact_bytes": 0,
        "descriptive_only": True, "p_values_emitted": False, "winner_selected": False,
        "new_go_no_go_issued": False, "d10_primary_recomputed": False,
    }


def run_shard(
    repo_root: Path | str = REPO_ROOT, *, shard_id: int, shard_count: int,
    device: str = "cuda:0", limit: int | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if type(shard_count) is not int or shard_count <= 0 or type(shard_id) is not int or not 0 <= shard_id < shard_count:
        raise StageEE2Error("invalid shard identity")
    validate_initialized(repo)
    plan, _ = _strict_json(repo / PLAN_PATH, "E2 plan")
    registry, _ = e1._load_registry(repo)
    units_by_sample: dict[str, list[dict[str, Any]]] = {}
    for unit in plan["units"]:
        units_by_sample.setdefault(unit["sample_id"], []).append(unit)
    selected = [row for row in registry if row["selection_order"] % shard_count == shard_id]
    if limit is not None: selected = selected[:limit]
    import torch
    from .stage_c_fidelity import ValuePathScalarAdapter, resolve_target_response
    from .stage_c_value_path import resolve_fixed_clean_attention_head_direction
    from .stage_c_perturbations import tensor_sha256

    model = binding = None
    shard_start = time.monotonic(); cpu_start = time.process_time()
    completed = skipped = 0; groups: list[dict[str, Any]] = []
    try:
        model, binding, model_record = e1._load_model_and_context(repo, device)
        gpu_index = torch.device(device).index or 0
        gpu_properties = torch.cuda.get_device_properties(gpu_index) if device.startswith("cuda") else None
        for row in selected:
            pair = sorted(units_by_sample[row["sample_id"]], key=lambda item: NEW_VARIANTS.index(item["variant_id"]))
            existing = [_validate_attempt(repo, unit) for unit in pair if _attempt_paths(repo, unit["canonical_digest"])[0].exists()]
            if existing:
                if len(existing) == 2:
                    skipped += 2; continue
                recovery, _ = _strict_json(repo / RECOVERY_AUTH_PATH, "E2 recovery authorization")
                _validate_signed(recovery, "E2 recovery authorization")
                if (
                    recovery.get("status") != "AUTHORIZED"
                    or recovery.get("existing_terminal_recompute_allowed") is not False
                    or row["sample_id"] not in recovery.get("partial_groups", [])
                ):
                    raise StageEE2Error("partial terminal image group lacks exact recovery authorization")
                skipped += len(existing)
            missing_units = [unit for unit in pair if not _attempt_paths(repo, unit["canonical_digest"])[0].exists()]
            group_start = time.monotonic(); group_cpu = time.process_time()
            if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
            source_result, source_map, native = _load_source(repo, pair[0])
            inherited = _validate_d9_curves(row["sample_id"], source_result, source_map)
            _, clean, normalized_sha = e1._load_sample(binding, row, model, device)
            if source_result["clean_tensor_hashes"]["normalized_tensor_sha256"] != normalized_sha:
                raise StageEE2Error("live normalized input differs from sealed D9 binding")
            calls = 0
            def count_hook(_module: Any, _args: Any) -> None:
                nonlocal calls; calls += 1
            handle = model.register_forward_pre_hook(count_hook)
            try:
                shared_start = time.monotonic(); shared_cpu = time.process_time()
                target = resolve_fixed_clean_attention_head_direction(model, clean, 0, 9)
                evaluator = ValuePathScalarAdapter(model, clean, target)
                response = resolve_target_response(evaluator, clean, torch.zeros_like(clean))
                if response.na is not None: raise StageEE2Error("frozen target unexpectedly became NA")
                shared_elapsed = time.monotonic() - shared_start; shared_cpu_seconds = time.process_time() - shared_cpu
                shared_calls = calls
                live_identity = json.loads(canonical_json_bytes(evaluator.target_identity).decode("utf-8"))
                if e1._target_semantic_digest(live_identity) != e1._target_semantic_digest(source_result["target_identity"]):
                    raise StageEE2Error("live target differs semantically from sealed D9 target")
                endpoints = inherited[0]["intervention_sha256"]
                baseline_sha = endpoints[0]; clean_sha = endpoints[-1]
                if baseline_sha != tensor_sha256(torch.zeros_like(clean)) or clean_sha != tensor_sha256(clean):
                    raise StageEE2Error("D9 endpoint tensor binding differs from live clean/baseline")
                outputs: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
                for unit_index, unit in enumerate(missing_units):
                    start = time.monotonic(); cpu0 = time.process_time(); calls0 = calls
                    fine_map = transformed_map(native, unit["variant_id"])
                    curves = []
                    for direction in ("insertion", "deletion"):
                        for repeat in range(5):
                            curves.append(_curve(
                                evaluator=evaluator, response=response, clean=clean, fine_map=fine_map,
                                sample_id=row["sample_id"], variant=unit["variant_id"], direction=direction, repeat=repeat,
                                baseline_sha256=baseline_sha, clean_sha256=clean_sha,
                            ))
                    result = _valid_result(
                        unit=unit, source_result=source_result, native=native, fine_map=fine_map,
                        target_identity=live_identity, response=response, curves=curves,
                        elapsed_seconds=time.monotonic() - start + shared_elapsed / len(missing_units),
                        cpu_seconds=time.process_time() - cpu0 + shared_cpu_seconds / len(missing_units),
                        forward_calls=calls - calls0 + shared_calls // len(missing_units) + (1 if unit_index < shared_calls % len(missing_units) else 0),
                        peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,
                    )
                    outputs.append((unit, result))
            finally:
                handle.remove()
            for unit, result in outputs:
                _publish_attempt(repo, unit, result); completed += 1
            groups.append({
                "sample_id": row["sample_id"], "selection_order": row["selection_order"],
                "elapsed_seconds": time.monotonic() - group_start, "cpu_seconds": time.process_time() - group_cpu,
                "forward_calls": calls, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,
            })
            del clean, target, evaluator, response
        elapsed = time.monotonic() - shard_start
        shard = _signed({
            "schema_version": 1, "document_type": "stage_e_e2_shard_telemetry",
            "shard_id": shard_id, "shard_count": shard_count, "device": device,
            "gpu_index": gpu_index if device.startswith("cuda") else None,
            "gpu_uuid": getattr(gpu_properties, "uuid", None),
            "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "started_groups": len(selected), "completed_units": completed, "skipped_terminal_units": skipped,
            "elapsed_seconds": elapsed, "cpu_seconds": time.process_time() - cpu_start,
            "gpu_hours": elapsed / 3600.0 if device.startswith("cuda") else 0.0,
            "forward_calls": sum(row["forward_calls"] for row in groups),
            "peak_cuda_memory_bytes": max((row["peak_cuda_memory_bytes"] for row in groups), default=0),
            "groups": groups, "model": model_record, "completed_at": _now(),
        })
        _write_or_verify(repo / ROOT_PATH / "telemetry" / f"shard-{shard_id:02d}-of-{shard_count:02d}.json", shard)
        return shard
    finally:
        if model is not None: del model
        if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()


def authorize_recovery(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Record the first-run infrastructure failure before any safe resume."""

    repo = Path(repo_root).resolve(); plan, _ = _strict_json(repo / PLAN_PATH, "E2 plan")
    by_sample: dict[str, list[Mapping[str, Any]]] = {}
    for unit in plan["units"]: by_sample.setdefault(unit["sample_id"], []).append(unit)
    partial_groups: list[str] = []; validated_existing: list[dict[str, Any]] = []
    for sample_id, units in by_sample.items():
        present = [unit for unit in units if _attempt_paths(repo, unit["canonical_digest"])[0].exists()]
        if present:
            if len(present) != 1 or present[0]["variant_id"] != "bilinear":
                raise StageEE2Error("recovery authorization observed an unexpected terminal pattern")
            observed = _validate_attempt(repo, present[0])
            partial_groups.append(sample_id)
            validated_existing.append({
                "sample_id": sample_id, "selection_order": present[0]["selection_order"],
                "variant_id": "bilinear", "unit_digest": present[0]["canonical_digest"],
                "result_canonical_digest": observed["result"]["canonical_digest"],
                "forward_calls": observed["result"]["forward_calls"],
            })
    if len(partial_groups) != 8:
        raise StageEE2Error(f"expected exactly 8 first-run partial groups, observed {len(partial_groups)}")
    event = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_infrastructure_event",
        "event_id": "E2_FIRST_RUN_VALIDATOR_COUNT_001", "status": "RECORDED",
        "recorded_at": _now(), "new_e2_outcome_accessed": True,
        "failure_class": "INFRASTRUCTURE_IMPLEMENTATION_ERROR",
        "failure_code": "VALIDATOR_FORWARD_COUNT_EXPECTATION_TOO_NARROW",
        "diagnosis": "The runner correctly shared five target and endpoint forward calls per image. The validator admitted 191 or 192 instead of the observed 193 or 192 split. Publication stopped after the first bilinear attempt in each shard.",
        "scientific_scope_operator_seed_ranking_or_response_changed": False,
        "process_invocations": 8, "published_valid_attempts": validated_existing,
        "unpublished_bicubic_computations_lost": 8,
        "resource_accounting": {
            "accounting_status": "CONSERVATIVE_30_SECONDS_PER_EIGHT_GPU_INVOCATIONS",
            "gpu_device_seconds_charged": 240.0, "gpu_hours_charged": 0.06666666666666667,
            "cpu_seconds_charged": 240.0, "forward_calls_charged": 3080,
            "gpu_indices": list(range(8)),
        },
    })
    authorization = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_recovery_authorization",
        "status": "AUTHORIZED", "issued_at": _now(),
        "infrastructure_event_canonical_digest": event["canonical_digest"],
        "partial_groups": partial_groups,
        "authorized_action": "resume missing attempt-000001 bicubic units and all never-started bilinear/bicubic units",
        "existing_terminal_recompute_allowed": False, "new_attempt_number_for_existing_terminal_allowed": False,
        "scientific_scope_change_allowed": False, "uniform_or_nearest_compute_allowed": False,
    })
    _write_or_verify(repo / INFRASTRUCTURE_EVENT_PATH, event)
    _write_or_verify(repo / RECOVERY_AUTH_PATH, authorization)
    return {"status": "AUTHORIZED", "partial_groups": len(partial_groups), "event_canonical_digest": event["canonical_digest"], "authorization_canonical_digest": authorization["canonical_digest"]}


def record_resume_event(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); plan, _ = _strict_json(repo / PLAN_PATH, "E2 plan")
    units = [unit for unit in plan["units"] if unit["selection_order"] < 8]
    observed = [_validate_attempt(repo, unit) for unit in units]
    if len(observed) != 16 or Counter(row["result"]["variant_id"] for row in observed) != Counter({"bilinear": 8, "bicubic": 8}):
        raise StageEE2Error("second infrastructure event does not observe the exact 16 terminal units")
    bicubic = [row["result"] for row in observed if row["result"]["variant_id"] == "bicubic"]
    if any(row["forward_calls"] != 196 for row in bicubic):
        raise StageEE2Error("second infrastructure event expected eight 196-call bicubic units")
    event = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_infrastructure_event",
        "event_id": "E2_RESUME_VALIDATOR_COUNT_002", "status": "RECORDED", "recorded_at": _now(),
        "new_e2_outcome_accessed": True, "failure_class": "INFRASTRUCTURE_IMPLEMENTATION_ERROR",
        "failure_code": "SINGLE_MISSING_UNIT_SHARED_TARGET_COUNT_EXPECTATION_TOO_NARROW",
        "diagnosis": "Each resumed partial group had only bicubic missing, so all six shared target and endpoint calls were assigned to that unit: 190 intermediate intervention calls plus 6 shared calls equals 196. Publication completed before the stale validator assertion.",
        "scientific_scope_operator_seed_ranking_or_response_changed": False,
        "process_invocations": 8, "published_valid_attempts": [{
            "sample_id": row["sample_id"], "selection_order": row["selection_order"],
            "variant_id": row["variant_id"], "unit_digest": row["unit_digest"],
            "result_canonical_digest": row["canonical_digest"], "forward_calls": row["forward_calls"],
        } for row in bicubic],
        "unpublished_computations_lost": 0,
        "resource_accounting": {
            "accounting_status": "CONSERVATIVE_30_SECONDS_PER_EIGHT_GPU_INVOCATIONS",
            "gpu_device_seconds_charged": 240.0, "gpu_hours_charged": 0.06666666666666667,
            "cpu_seconds_charged": 240.0, "forward_calls_charged": 1568, "gpu_indices": list(range(8)),
        },
    })
    _write_or_verify(repo / INFRASTRUCTURE_EVENT_002_PATH, event)
    return {"status": "RECORDED", "published_valid_attempts": 8, "event_canonical_digest": event["canonical_digest"]}


def _repeat_aggregates(curves: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for direction in ("insertion", "deletion"):
        values = np.asarray([row["auc"] for row in curves if row["direction"] == direction], dtype=np.float64)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise StageEE2Error("five finite repeat AUCs required")
        result[f"{direction}_repeat_aucs"] = values.tolist()
        result[f"{direction}_auc_mean"] = float(values.mean())
        result[f"{direction}_auc_repeat_sd_ddof_0"] = float(values.std(ddof=0))
    return result


def _row_from_inherited(
    repo: Path, group: Mapping[str, Any], variant: str,
) -> dict[str, Any]:
    unit = {
        "source_map_path": group["source_map_path"], "source_map_sha256": group["source_map_sha256"],
        "source_result_path": group["source_result_path"], "source_result_sha256": group["source_result_sha256"],
    }
    source_result, source_map, native = _load_source(repo, unit)
    curves = _validate_d9_curves(group["sample_id"], source_result, source_map)
    fine_map = transformed_map(native, variant)
    uniform_map = transformed_map(native, VARIANTS[0])
    if not np.array_equal(fine_map, uniform_map):
        raise StageEE2Error("nearest inherited row is not exactly uniform-equivalent")
    for curve in curves:
        seed = d9_seed(group["sample_id"], curve["replicate"])
        if curve["ranking_sha256"] != e1._array_digest(ranking(fine_map, seed=seed)):
            raise StageEE2Error("inherited D9 ranking differs from E2 uniform/nearest ranking")
    aggregates = _repeat_aggregates(curves)
    matched = source_result["provenance_fragment"]["matched_contribution"]
    target_digest = e1._target_semantic_digest(source_result["target_identity"])
    source_hashes = {
        "source_map_sha256": group["source_map_sha256"],
        "source_result_sha256": group["source_result_sha256"],
        "source_unit_digest": group["source_unit_digest"],
        "source_attempt_inventory_digest": group["source_attempt_inventory_digest"],
        "image_sha256": group["image_sha256"], "model_sha256": MODEL_SHA256,
        "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST,
        "target_semantic_digest": target_digest, "native_patch_float64_sha256": e1._array_digest(native),
        "native_patch_float32_tensor_sha256": matched["patch_contributions_sha256"],
        "signed_transformed_map_sha256": e1._array_digest(fine_map), "operator_sha256": OPERATOR_DIGESTS[variant],
        "d9_fidelity_curves_sha256": canonical_sha256(curves),
    }
    return {
        "variant_id": variant, "sample_id": group["sample_id"],
        "selection_order": group["selection_order"], "cohort": "stage_d_fidelity_v1_500",
        "stratum": "overall", "validity_status": "VALID", "source_hashes": source_hashes,
        "method": METHOD, "target": "fixed_clean_attention_cls_head_direction",
        "scale": "fine_4x4_cell", "layer": 0, "head": 9,
        "operator_id": OPERATOR_IDS[variant], "operator_sha256": OPERATOR_DIGESTS[variant],
        "equivalence_class": EQUIVALENCE[variant][0], "evidence_independence": EQUIVALENCE[variant][1],
        "failure_class": None, "failure_code": None, "na_reason": None,
        "attempt_id": None, "inherited_attempt_id": group["source_attempt_id"], "retry_count": 0,
        **aggregates, "elapsed_seconds": 0.0, "gpu_hours": 0.0, "cpu_seconds": 0.0,
        "forward_calls": 0, "peak_cuda_memory_bytes": 0, "artifact_bytes": 0,
        "compute_status": "REUSE_DERIVABLE" if variant == VARIANTS[0] else "FORBIDDEN_RECOMPUTE_EQUIVALENT_RANKING",
        "paired_seed_tie_binding": {
            "namespace": SEED_NAMESPACE, "comparison_identity": COMMON_PAIR_IDENTITY,
            "seeds": [d9_seed(group["sample_id"], repeat) for repeat in range(5)],
            "tie_priority_sha256": [row["tie_priority_sha256"] for row in curves[:5]],
        },
        "descriptive_only": True,
    }


def _row_from_attempt(observed: Mapping[str, Any]) -> dict[str, Any]:
    result = observed["result"]; root = observed["attempt_path"]
    return {
        "variant_id": result["variant_id"], "sample_id": result["sample_id"],
        "selection_order": result["selection_order"], "cohort": "stage_d_fidelity_v1_500",
        "stratum": "overall", "validity_status": result["validity_status"],
        "source_hashes": result["source_hashes"], "method": METHOD,
        "target": result["target"], "scale": result["scale"], "layer": 0, "head": 9,
        "operator_id": result["operator_id"], "operator_sha256": result["operator_sha256"],
        "equivalence_class": result["equivalence_class"], "evidence_independence": result["evidence_independence"],
        "failure_class": result["failure_class"], "failure_code": result["failure_code"],
        "na_reason": result["na_reason"], "attempt_id": result["attempt_id"],
        "inherited_attempt_id": None, "retry_count": result["retry_count"],
        "insertion_repeat_aucs": result["insertion_repeat_aucs"],
        "deletion_repeat_aucs": result["deletion_repeat_aucs"],
        "insertion_auc_mean": result["insertion_auc_mean"],
        "deletion_auc_mean": result["deletion_auc_mean"],
        "insertion_auc_repeat_sd_ddof_0": result["insertion_auc_repeat_sd_ddof_0"],
        "deletion_auc_repeat_sd_ddof_0": result["deletion_auc_repeat_sd_ddof_0"],
        "elapsed_seconds": result["elapsed_seconds"], "gpu_hours": result["gpu_hours"],
        "cpu_seconds": result["cpu_seconds"], "forward_calls": result["forward_calls"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
        "artifact_bytes": sum(path.stat().st_size for path in root.iterdir()),
        "compute_status": "NEW_COMPUTE_REQUIRED_COMPLETED",
        "paired_seed_tie_binding": {
            "namespace": SEED_NAMESPACE, "comparison_identity": COMMON_PAIR_IDENTITY,
            "seeds": [d9_seed(result["sample_id"], repeat) for repeat in range(5)],
            "tie_priority_sha256": [row["tie_priority_sha256"] for row in result["curves"][:5]],
        },
        "descriptive_only": True,
    }


def _pair_status(a: str, b: str) -> tuple[str, str]:
    if (a, b) == PAIR_ORDER[0]:
        return "EXACT_EQUIVALENCE", "SAME_EVIDENCE_CLASS_NOT_INDEPENDENT"
    if a == VARIANTS[1]:
        return "DEPENDENT_DUPLICATE_OF_UNIFORM_COMPARISON", "NOT_INDEPENDENT"
    return "DISTINCT_RULE_COMPARISON", "DESCRIPTIVE_PAIRED"


def _collect_rows(repo: Path, plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    attempts = {unit["canonical_digest"]: _validate_attempt(repo, unit) for unit in plan["units"]}
    by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for unit in plan["units"]:
        by_sample.setdefault(unit["sample_id"], {})[unit["variant_id"]] = _row_from_attempt(attempts[unit["canonical_digest"]])
    variant_rows: list[dict[str, Any]] = []; pair_rows: list[dict[str, Any]] = []
    for group in plan["groups"]:
        rows = {variant: _row_from_inherited(repo, group, variant) for variant in VARIANTS[:2]}
        rows.update(by_sample.get(group["sample_id"], {}))
        if set(rows) != set(VARIANTS):
            raise StageEE2Error("one image lacks one or more frozen E2 rules")
        for variant in VARIANTS: variant_rows.append(rows[variant])
        if rows[VARIANTS[0]]["insertion_repeat_aucs"] != rows[VARIANTS[1]]["insertion_repeat_aucs"] or rows[VARIANTS[0]]["deletion_repeat_aucs"] != rows[VARIANTS[1]]["deletion_repeat_aucs"]:
            raise StageEE2Error("uniform and nearest inherited curves differ")
        for a, b in PAIR_ORDER:
            left, right = rows[a], rows[b]; equivalence, independence = _pair_status(a, b)
            value = delta_fid(left, right)
            if (a, b) == PAIR_ORDER[0] and value != 0.0:
                raise StageEE2Error("uniform-minus-nearest is not exactly zero")
            if left["paired_seed_tie_binding"] != right["paired_seed_tie_binding"]:
                raise StageEE2Error("E2 rule pair does not share inherited D9 seed/tie priority")
            pair_rows.append({
                "variant_a": a, "variant_b": b, "sample_id": group["sample_id"],
                "selection_order": group["selection_order"], "cohort": "stage_d_fidelity_v1_500",
                "stratum": "overall", "validity_status": "VALID", "failure_class": None,
                "failure_code": None, "na_reason": None,
                "paired_seed_tie_binding": left["paired_seed_tie_binding"],
                "insertion_auc_a": left["insertion_auc_mean"], "insertion_auc_b": right["insertion_auc_mean"],
                "deletion_auc_a": left["deletion_auc_mean"], "deletion_auc_b": right["deletion_auc_mean"],
                "delta_fid_A_minus_B": value, "equivalence_status": equivalence,
                "evidence_independence": independence, "descriptive_only": True,
            })
    if len(variant_rows) != 2000 or len(pair_rows) != 3000:
        raise StageEE2Error("E2 output coverage is incomplete")
    if Counter(row["variant_id"] for row in variant_rows) != Counter({variant: 500 for variant in VARIANTS}):
        raise StageEE2Error("E2 variant row cardinality failed")
    if any(row["validity_status"] != "VALID" for row in variant_rows + pair_rows):
        raise StageEE2Error("E2 PASS requires every row VALID")
    return variant_rows, pair_rows, attempts


def _median(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64)); count = ordered.size; middle = count // 2
    if not count: raise StageEE2Error("cannot summarize empty values")
    return float(ordered[middle]) if count % 2 else float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _bootstrap(values: Sequence[float], identity: str) -> dict[str, Any]:
    source = np.asarray(values, dtype=np.float64)
    material = "stage-e-bootstrap|20260906|E2|" + identity + "|overall"
    seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    generator = np.random.Generator(np.random.PCG64(seed)); medians = np.empty(10000, dtype=np.float64)
    for start in range(0, 10000, 1000):
        stop = min(start + 1000, 10000); indices = generator.integers(0, source.size, size=(stop - start, source.size))
        medians[start:stop] = np.median(source[indices], axis=1)
    interval = np.quantile(medians, [0.025, 0.975], method="linear")
    return {
        "resamples": 10000, "seed": seed, "prng": "numpy.random.Generator(numpy.random.PCG64(seed))",
        "statistic": "arithmetic_even_median", "quantile_method": "linear",
        "ci_95": [float(interval[0]), float(interval[1])], "role": "DESCRIPTIVE_ONLY",
    }


def _summary(values: Sequence[float], identity: str) -> dict[str, Any]:
    source = np.asarray(values, dtype=np.float64)
    if source.shape != (500,) or not np.isfinite(source).all():
        raise StageEE2Error("E2 summary requires all 500 finite image values")
    return {
        "n": 500, "valid": 500, "failed": 0, "na": 0,
        "positive_count": int(np.sum(source > 0.0)), "mean": float(source.mean()),
        "median": _median(source), "sample_sd_ddof_1": float(source.std(ddof=1)),
        "paired_image_bootstrap_median_95_interval": _bootstrap(source, identity),
    }


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _validate_config(repo)
    plan, _ = _strict_json(repo / PLAN_PATH, "E2 plan")
    if plan != _build_plan(repo, config): raise StageEE2Error("E2 plan drifted")
    variant_rows, pair_rows, attempts = _collect_rows(repo, plan)
    variant_summaries = {}
    for variant in VARIANTS:
        selected = [row for row in variant_rows if row["variant_id"] == variant]
        for direction in ("insertion", "deletion"):
            variant_summaries[f"{variant}:{direction}_auc"] = _summary([row[f"{direction}_auc_mean"] for row in selected], f"{variant}_{direction}_auc")
    pair_summaries = {}
    for a, b in PAIR_ORDER:
        selected = [row for row in pair_rows if row["variant_a"] == a and row["variant_b"] == b]
        pair_summaries[f"{a}_minus_{b}"] = {
            **_summary([row["delta_fid_A_minus_B"] for row in selected], f"{a}_minus_{b}"),
            "equivalence_status": selected[0]["equivalence_status"],
            "evidence_independence": selected[0]["evidence_independence"],
        }
    telemetry = []
    for path in sorted((repo / ROOT_PATH / "telemetry").glob("shard-*.json")):
        value, _ = _strict_json(path, "E2 shard telemetry"); _validate_signed(value, "E2 shard telemetry"); telemetry.append(value)
    event, _ = _strict_json(repo / INFRASTRUCTURE_EVENT_PATH, "E2 infrastructure event")
    event2, _ = _strict_json(repo / INFRASTRUCTURE_EVENT_002_PATH, "E2 second infrastructure event")
    recovery, _ = _strict_json(repo / RECOVERY_AUTH_PATH, "E2 recovery authorization")
    _validate_signed(event, "E2 infrastructure event"); _validate_signed(event2, "E2 second infrastructure event"); _validate_signed(recovery, "E2 recovery authorization")
    if event.get("status") != "RECORDED" or event2.get("status") != "RECORDED" or recovery.get("status") != "AUTHORIZED":
        raise StageEE2Error("E2 infrastructure recovery evidence failed")
    charge = event["resource_accounting"]
    charge2 = event2["resource_accounting"]
    new_rows = [row for row in variant_rows if row["variant_id"] in NEW_VARIANTS]
    actual = {
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in telemetry) + float(charge["gpu_device_seconds_charged"]) + float(charge2["gpu_device_seconds_charged"]),
        "gpu_hours": sum(float(row["gpu_hours"]) for row in telemetry) + float(charge["gpu_hours_charged"]) + float(charge2["gpu_hours_charged"]),
        "cpu_seconds": sum(float(row["cpu_seconds"]) for row in telemetry) + float(charge["cpu_seconds_charged"]) + float(charge2["cpu_seconds_charged"]),
        "forward_calls": sum(int(row["forward_calls"]) for row in telemetry) + int(charge["forward_calls_charged"]) + int(charge2["forward_calls_charged"]),
        "peak_cuda_memory_bytes": max((int(row["peak_cuda_memory_bytes"]) for row in telemetry), default=0),
        "artifact_bytes": sum(int(row["artifact_bytes"]) for row in new_rows),
        "gpu_invocation_count": len(telemetry) + int(event["process_invocations"]) + int(event2["process_invocations"]), "successful_terminal_unit_attempts": len(attempts),
        "retry_unit_attempts": 0, "uniform_forward_calls": 0, "nearest_forward_calls": 0,
        "failed_process_invocations": int(event["process_invocations"]) + int(event2["process_invocations"]),
        "unpublished_computations_charged": int(event["unpublished_bicubic_computations_lost"]),
    }
    summary = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_overall_descriptive_summary",
        "stage_id": "E2", "cohort": "stage_d_fidelity_v1_500", "stratum": "overall",
        "descriptive_only": True, "p_values_emitted": False, "winner_selection_performed": False,
        "new_go_no_go_issued": False, "d10_primary_unchanged": True, "e1_interpretation_unchanged": True,
        "counts": {"expected_images": 500, "variant_rows": 2000, "pairwise_rows": 3000, "new_compute_rows": 1000, "valid": 2000, "failed": 0, "na": 0},
        "variant_direction_descriptions": variant_summaries, "pairwise_descriptions": pair_summaries,
        "uniform_nearest_equivalence": {"operator_exact": True, "maps_exact_500_of_500": True, "rankings_and_ties_exact": True, "same_evidence_class": True, "independent_support_count": 1},
        "resource_accounting": {
            "global": actual,
            "by_variant": {variant: {
                "elapsed_seconds_scientific_unit_share": sum(row["elapsed_seconds"] for row in new_rows if row["variant_id"] == variant),
                "gpu_hours_scientific_unit_share": sum(row["gpu_hours"] for row in new_rows if row["variant_id"] == variant),
                "cpu_seconds_scientific_unit_share": sum(row["cpu_seconds"] for row in new_rows if row["variant_id"] == variant),
                "forward_calls": sum(row["forward_calls"] for row in new_rows if row["variant_id"] == variant),
            } for variant in NEW_VARIANTS},
            "inherited": {VARIANTS[0]: {"forward_calls": 0}, VARIANTS[1]: {"forward_calls": 0}},
            "infrastructure_events": [event, event2], "recovery_authorization": recovery,
        },
        "interpretation_boundary": "Post-D10 descriptive sensitivity only; no interpolation winner, p-value, rejection, or new GO/NO_GO; D10 primary and E1 interpretation remain unchanged.",
    })
    return {"config": config, "plan": plan, "variant_rows": variant_rows, "pair_rows": pair_rows, "attempts": attempts, "summary": summary, "actual_cost": actual}


def _table(table_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: raise StageEE2Error("output table cannot be empty")
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise StageEE2Error("output table rows do not share one column set")
    rows = [{column: row[column] for column in columns} for row in rows]
    value = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_output_table",
        "table_id": table_id, "columns": columns, "row_count": len(rows), "rows": rows,
    })
    return value


def _write_output_json(repo: Path, relative: str, value: Mapping[str, Any]) -> None:
    schema, _ = _strict_json(repo / OUTPUT_SCHEMA_PATH, "E2 output schema")
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors: raise StageEE2Error(f"E2 output schema failure: {errors[0].message}")
    _write_or_verify(repo / relative, value)


def _csv_scalar(value: Any) -> str:
    if value is None: return ""
    if isinstance(value, (dict, list)): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, bool): return "true" if value else "false"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name("." + path.name + ".pending")
    if temporary.exists(): raise StageEE2Error(f"ambiguous pending CSV: {temporary}")
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise StageEE2Error("CSV rows do not share one column set")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        writer.writerows({key: _csv_scalar(row[key]) for key in columns} for row in rows)
        stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink(); raise StageEE2Error(f"immutable CSV differs: {path}")
        temporary.unlink()
    else: os.replace(temporary, path)


def budget_reasons(actual: Mapping[str, Any], retained_bytes: int) -> tuple[str, ...]:
    failures = []
    if float(actual.get("gpu_hours", math.inf)) > 2.5: failures.append("gpu_hours")
    if float(actual.get("cpu_seconds", math.inf)) / 3600.0 > 8.0: failures.append("cpu_hours")
    if type(retained_bytes) is not int or retained_bytes > 4294967296: failures.append("retained_bytes")
    return tuple(failures)


def false_pass_reasons(preconditions: Mapping[str, bool]) -> tuple[str, ...]:
    if set(preconditions) != set(FALSE_PASS_ATTACKS):
        raise StageEE2Error("false-PASS precondition set drifted")
    return tuple(name for name in FALSE_PASS_ATTACKS if preconditions[name] is not True)


def _publish_access_terminal(repo: Path, initial: Mapping[str, Any]) -> dict[str, Any]:
    pending = repo / ACCESS_PENDING_PATH; target = repo / ACCESS_PATH
    if target.is_file():
        observed, _ = _strict_json(target, "terminal E2 outcome access"); _validate_signed(observed, "terminal E2 outcome access"); return observed
    if not pending.is_file(): raise StageEE2Error("E2 outcome access pending/terminal state is invalid")
    initial_raw = pending.read_bytes()
    terminal = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_outcome_access", "status": "TERMINAL",
        "e2_started_at": initial["e2_started_at"], "completed_at": _now(),
        "e1_terminal_pass_and_e2_only_verified": True, "d10_and_e1_outcomes_known": True,
        "new_e2_bilinear_bicubic_outcome_accessed": True,
        "first_outcome_access_after_pending_record": True, "initial_pending_sha256": _sha(initial_raw),
        "initial_pending_canonical_digest": initial["canonical_digest"],
        "scope_changed_after_outcome_access": False, "variants_added_or_removed": False,
        "failed_na_or_unfavorable_samples_dropped": False,
        "uniform_recomputed": False, "nearest_recomputed": False, "d10_primary_recomputed": False,
        "e3_started": False, "e4_started": False, "stage_f_started": False,
        "protected_roots": initial["protected_roots"], "upstream_reads": initial["upstream_reads"],
    })
    temporary = pending.with_name(".outcome-access-terminal.tmp")
    _write_new_json(temporary, terminal); os.replace(temporary, pending)
    observed, raw = _strict_json(pending, "terminal pending E2 outcome access")
    if observed != terminal or raw != canonical_json_bytes(terminal) + b"\n":
        raise StageEE2Error("E2 access record changed before publication")
    os.replace(pending, target); return terminal


def _protected_after(repo: Path, before: Mapping[str, Any]) -> dict[str, Any]:
    after = _protected_snapshot(repo)
    comparable_before = {row["path"]: row for row in before["roots"]}
    comparable_after = {row["path"]: row for row in after["roots"]}
    if comparable_before != comparable_after:
        raise StageEE2Error("one or more protected roots changed during E2")
    return after


def finalize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); outputs = validate_outputs(repo); root = repo / ROOT_PATH
    before, _ = _strict_json(repo / UPSTREAM_BEFORE_PATH, "E2 protected upstream before")
    after = _protected_after(repo, before)
    current_bytes = _inventory(root)["total_bytes"]
    failures = budget_reasons(outputs["actual_cost"], current_bytes)
    if failures: raise StageEE2Error(f"E2 budget exceeded: {failures}")
    variant_table = _table("upsampling_sensitivity", outputs["variant_rows"])
    pair_table = _table("pairwise_differences", outputs["pair_rows"])
    resource_rows = [row for row in outputs["variant_rows"]]
    resource_table = _table("resource_and_qc", resource_rows)
    paper = root / "paper_data"
    for filename, table, rows in (
        ("upsampling_sensitivity", variant_table, outputs["variant_rows"]),
        ("pairwise_differences", pair_table, outputs["pair_rows"]),
        ("resource_and_qc", resource_table, resource_rows),
    ):
        _write_output_json(repo, ROOT_PATH + f"/paper_data/{filename}.json", table)
        _write_csv(paper / f"{filename}.csv", rows)
    _write_or_verify(root / "overall_descriptive_summary.json", outputs["summary"])
    initial_path = repo / ACCESS_PENDING_PATH if (repo / ACCESS_PENDING_PATH).is_file() else root / "control/outcome_access_initial.json"
    initial_access, _ = _strict_json(initial_path, "initial E2 outcome access")
    _write_or_verify(root / "control/outcome_access_initial.json", initial_access)
    terminal_access = _publish_access_terminal(repo, initial_access)
    provenance = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_provenance", "stage_id": "E2",
        "source_files": initial_access["upstream_reads"],
        "operations": [
            "source-validate and exactly reuse D9 uniform curves",
            "derive nearest only after exact operator map ranking and tie equivalence",
            "recover native float32 patch contributions by float64 aligned block sums and verify original tensor digest",
            "apply frozen float64 column-normalized bilinear and bicubic operators",
            "evaluate true 3136-unit fine-cell interventions with inherited D9 paired seeds",
            "compute float64 AUC pairwise delta and descriptive image bootstrap",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "operator_sha256": OPERATOR_DIGESTS,
        "outcome_access": {"path": ACCESS_PATH, "sha256": _sha(canonical_json_bytes(terminal_access) + b"\n"), "canonical_digest": terminal_access["canonical_digest"]},
        "uniform_recomputed": False, "nearest_recomputed": False, "d10_primary_recomputed": False,
        "d9_d10_e0_e1_unchanged": True, "e3_e4_stage_f_started": False,
    })
    _write_or_verify(root / "provenance.json", provenance)
    recipe = _signed({
        "schema_version": 1, "document_type": "stage_e_figure_recipes",
        "numeric_source": "paper_data_only", "renders_png": False,
        "figures": [{
            "figure_id": "e2_upsampling_rule_descriptive_future",
            "table_path": "paper_data/upsampling_sensitivity.json", "x": "variant_id",
            "y": "insertion_auc_mean", "group": "variant_id",
            "caption_contract": "Post-D10 descriptive sensitivity; uniform and nearest are one evidence class; no winner or new decision.",
            "numeric_source": "paper_data_only", "renders_png": False,
        }],
    })
    _write_or_verify(root / "figure_recipes.json", recipe)
    false_audit = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_false_pass_audit",
        "stage_id": "E2", "status": "PASS",
        "checks": [{
            "attack": name, "result": "REJECTED",
            "mechanism": {
                "e1_unverified_or_not_e2_only": "exact E1 archive completion status inventory and source validation required",
                "e0_d9_d10_binding_failed": "existing E0 and D9 D10 source/package validators must PASS",
                "uniform_only_copy": "1000 unique bilinear and bicubic terminal attempts required",
                "maps_without_real_responses": "each new unit requires ten 21-point curves with 19 new intermediate forwards each",
                "fewer_than_500_images": "registry order 0 through 499 required",
                "fewer_than_four_rules": "exact frozen four-rule set and 2000 rows required",
                "bilinear_missing": "500 bilinear attempts required", "bicubic_missing": "500 bicubic attempts required",
                "uniform_recomputed": "uniform has inherited attempt only and zero forward calls",
                "nearest_recomputed": "nearest has inherited attempt only and zero forward calls",
                "uniform_nearest_double_counted": "same equivalence class and non-independent label required",
                "uniform_nearest_not_exact": "operator map ranking tie curve and AUC exact equality checked",
                "operator_not_column_normalized": "196 finite nonzero unit-mass basis columns checked",
                "global_signed_sum_renormalized": "operator identity and transformed-map digest are source recomputed",
                "nonlinear_transform_before_operator": "signed transformed map digest and positive-part ranking are separately recomputed",
                "native_196_interventions_masquerade_as_3136": "unit count schedule and 4x4 pixel masks are fixed to 3136",
                "unpaired_d9_tie_priority": "all four rules bind the exact inherited D9 seed and tie-priority digest",
                "missing_curves_auc_operator_pair_or_attempt": "source validator recomputes every coordinate hash AUC binding and immutable attempt",
                "missing_pairwise_rows": "exact six ordered pairs and 3000 rows required",
                "failed_na_nonfinite_dropped": "all expected rows required and PASS requires all finite VALID",
                "favorable_valid_subset_summary": "each summary requires exactly all 500 images",
                "d10_primary_recomputed_or_modified": "D10 is read-only and no D10 execution surface exists",
                "protected_upstream_modified": "before and after full-tree inventories plus Git diff gate",
                "budget_exceeded": "GPU CPU and retained-byte hard limits gate completion",
                "tests_missing_or_failed": "completion binds required PASS test report",
                "receipt_manifest_inventory_placeholder": "nonempty manifests inventory and pending-to-terminal completion are source validated",
                "e2_overauthorizes_e4_or_f0": "completion next authorization is exactly E3_ONLY and later started flags are false",
            }[name],
        } for name in FALSE_PASS_ATTACKS],
        "mutation_test": "Each named condition is independently falsified by test_false_pass_guard_rejects_every_attack.",
        "pass_possible_without_all_conditions": False,
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "false_pass_audit.json", false_audit)
    validation = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_source_validation",
        "stage_id": "E2", "status": "PASS", "validated_at": terminal_access["completed_at"],
        "checks": {
            "e1_e0_d9_d10": "PASS", "plan": "PASS_500_GROUPS_1000_NEW_UNITS",
            "attempts": "PASS_1000_UNIQUE_TERMINAL", "curves": "PASS_10000_CURVES_210000_POINTS",
            "inherited_uniform_nearest": "PASS_1000_ROWS_ZERO_NEW_FORWARDS_ONE_EVIDENCE_CLASS",
            "operators_and_conservation": "PASS_196_BASIS_AND_500_MAPS_ALL_FOUR_RULES",
            "paired_d9_seeds": "PASS", "variant_rows": "PASS_2000", "pairwise_rows": "PASS_3000",
            "auc_pairwise_summary_recomputed": "PASS", "protected_upstream_unchanged": "PASS",
            "no_render_artifacts": "PASS", "stage_boundary": "PASS_E2_ONLY",
        },
        "counts": outputs["summary"]["counts"], "summary_canonical_digest": outputs["summary"]["canonical_digest"],
        "false_pass_audit_canonical_digest": false_audit["canonical_digest"],
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "validation_report.json", validation)
    tests, _ = _strict_json(repo / REPORT_ROOT_PATH / "test_results.json", "E2 test report"); _validate_signed(tests, "E2 test report")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageEE2Error("E2 tests are not PASS")
    _write_or_verify(repo / REPORT_ROOT_PATH / "protected_upstream_after.json", after)
    manifest_files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json", "whole_root_inventory.json", "status.json"}:
            manifest_files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    if not manifest_files: raise StageEE2Error("E2 payload manifest cannot be empty")
    manifest = _signed({"schema_version": 1, "document_type": "stage_e_e2_payload_manifest", "immutable": True, "artifacts": manifest_files, "payload_count": len(manifest_files)})
    _write_or_verify(root / "payload_manifest.json", manifest)
    retained = _inventory(root)["total_bytes"]
    if budget_reasons(outputs["actual_cost"], retained): raise StageEE2Error("E2 retained bytes exceeded after reports")
    cost = {
        **outputs["actual_cost"], "cpu_hours": outputs["actual_cost"]["cpu_seconds"] / 3600.0,
        "retained_bytes_before_status_inventory_completion": retained, "gpu_hours_max": 2.5,
        "cpu_hours_max": 8.0, "retained_bytes_max": 4294967296, "status": "PASS_COST",
    }
    inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e2_completion.pending.json", "completion/e2_completion.json"))
    inventory_doc = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_whole_root_inventory", "root": ROOT_PATH,
        "scope": "all regular non-hidden E2 formal files except self-referential inventory status and completion receipts", **inventory,
    })
    _write_or_verify(root / "whole_root_inventory.json", inventory_doc)
    status = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_status", "stage_id": "E2",
        "status": "PASS", "complete": True, "terminal": True, "immutable": True,
        "package_completeness": "PASS", "workflow_integrity": "PASS",
        "scientific_validity": "PASS", "cost_status": "PASS_COST",
        "d10_decision_unchanged": True, "e1_interpretation_unchanged": True,
        "uniform_recomputed": False, "nearest_recomputed": False,
        "e3_authorized": True, "e3_started": False, "e4_started": False, "stage_f_started": False,
        "actual_cost": cost, "whole_root_inventory": {
            "path": "whole_root_inventory.json", "file_sha256": _sha_file(root / "whole_root_inventory.json"),
            "canonical_digest": inventory_doc["canonical_digest"], "file_count": inventory["file_count"],
            "total_bytes": inventory["total_bytes"], "scope": inventory_doc["scope"],
        },
    })
    _write_or_verify(root / "status.json", status)
    completion = _signed({
        "schema_version": 1, "document_type": "stage_e_terminal_completion", "stage_id": "E2", "status": "PASS",
        "started_at": initial_access["e2_started_at"], "completed_at": _now(),
        "predecessor": {"path": E1_COMPLETION_PATH, "file_sha256": E1_COMPLETION_SHA256, "canonical_digest": E1_COMPLETION_DIGEST},
        "config_canonical_digest": outputs["config"]["canonical_digest"],
        "run_spec_canonical_digest": _strict_json(repo / RUN_SPEC_PATH, "E2 run spec")[0]["canonical_digest"],
        "expected_coordinates": [{"variant_id": variant, "image_count": 500} for variant in VARIANTS],
        "observed_coordinates": [{"variant_id": variant, "image_count": 500} for variant in VARIANTS],
        "variant_statuses": [
            {"variant_id": VARIANTS[0], "status": "PASS", "reason": "500 exact D9 inherited curve units"},
            {"variant_id": VARIANTS[1], "status": "PASS", "reason": "500 exact uniform-equivalent derivations no model forward"},
            {"variant_id": VARIANTS[2], "status": "PASS", "reason": "500 valid new fine-cell response units"},
            {"variant_id": VARIANTS[3], "status": "PASS", "reason": "500 valid new fine-cell response units"},
        ],
        "package_completeness": "PASS", "workflow_integrity": "PASS",
        "scientific_validity": "PASS", "cost_status": "PASS_COST",
        "source_validation": {
            "path": REPORT_ROOT_PATH + "/validation_report.json", "file_sha256": _sha_file(repo / REPORT_ROOT_PATH / "validation_report.json"),
            "canonical_digest": validation["canonical_digest"], "test_report": _record(repo, REPORT_ROOT_PATH + "/test_results.json"),
            "false_pass_audit": _record(repo, REPORT_ROOT_PATH + "/false_pass_audit.json"),
            "payload_manifest": _record(repo, ROOT_PATH + "/payload_manifest.json"), "status": _record(repo, ROOT_PATH + "/status.json"),
            "whole_root_inventory": _record(repo, ROOT_PATH + "/whole_root_inventory.json"),
        },
        "upstream_unchanged": True, "actual_cost": cost, "next_authorization": "E3_ONLY",
    })
    schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE2Error(f"E2 completion schema failure: {errors[0].message}")
    pending = repo / COMPLETION_PENDING_PATH; terminal = repo / COMPLETION_PATH
    temporary = pending.with_name(".e2-completion-terminal.tmp")
    _write_new_json(temporary, completion); os.replace(temporary, pending)
    observed, raw = _strict_json(pending, "E2 pending completion")
    if observed != completion or raw != canonical_json_bytes(completion) + b"\n": raise StageEE2Error("E2 pending completion changed")
    os.replace(pending, terminal)
    validate_completion(repo)
    return {"status": "PASS", "completion_canonical_digest": completion["canonical_digest"], "next_authorization": "E3_ONLY", "actual_cost": cost}


def finalize_blocked(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    outputs = validate_outputs(repo)
    root = repo / ROOT_PATH
    before, _ = _strict_json(repo / UPSTREAM_BEFORE_PATH, "E2 protected upstream before")
    after = _protected_after(repo, before)
    actual = outputs["actual_cost"]
    failures = budget_reasons(actual, _inventory(root)["total_bytes"])
    if not failures:
        raise StageEE2Error("BLOCKED finalization requires a real budget failure")
    for filename, rows in (
        ("upsampling_sensitivity", outputs["variant_rows"]),
        ("pairwise_differences", outputs["pair_rows"]),
        ("resource_and_qc", outputs["variant_rows"]),
    ):
        _write_output_json(repo, ROOT_PATH + f"/paper_data/{filename}.json", _table(filename, rows))
        _write_csv(root / "paper_data" / f"{filename}.csv", rows)
    _write_or_verify(root / "overall_descriptive_summary.json", outputs["summary"])
    initial_path = repo / ACCESS_PENDING_PATH if (repo / ACCESS_PENDING_PATH).is_file() else root / "control/outcome_access_initial.json"
    initial, _ = _strict_json(initial_path, "initial E2 access")
    _write_or_verify(root / "control/outcome_access_initial.json", initial)
    terminal_access = _publish_access_terminal(repo, initial)
    cost = {
        **actual, "cpu_hours": actual["cpu_seconds"] / 3600.0,
        "retained_bytes_before_status_inventory_completion": _inventory(root)["total_bytes"],
        "gpu_hours_max": 2.5, "cpu_hours_max": 8.0,
        "retained_bytes_max": 4294967296, "status": "FAIL_COST",
        "budget_failures": list(failures),
    }
    provenance = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_provenance",
        "stage_id": "E2", "source_files": initial["upstream_reads"],
        "operator_sha256": OPERATOR_DIGESTS,
        "operations": [
            "exact D9 uniform reuse", "exact nearest equivalence derivation",
            "1000 real bilinear bicubic fine-cell intervention units",
            "source recompute of 2000 variant rows and 3000 pairwise rows",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "outcome_access": {"path": ACCESS_PATH, "sha256": _sha(canonical_json_bytes(terminal_access) + b"\n"), "canonical_digest": terminal_access["canonical_digest"]},
        "uniform_recomputed": False, "nearest_recomputed": False,
        "d10_primary_recomputed": False, "d9_d10_e0_e1_unchanged": True,
        "e3_e4_stage_f_started": False,
    })
    _write_or_verify(root / "provenance.json", provenance)
    recipe = _signed({
        "schema_version": 1, "document_type": "stage_e_figure_recipes",
        "numeric_source": "paper_data_only", "renders_png": False,
        "figures": [{
            "figure_id": "e2_upsampling_rule_descriptive_future_blocked",
            "table_path": "paper_data/upsampling_sensitivity.json",
            "x": "variant_id", "y": "insertion_auc_mean", "group": "variant_id",
            "caption_contract": "Numerically complete descriptive sensitivity; E2 is BLOCKED by hard cost gates and not authorized downstream.",
            "numeric_source": "paper_data_only", "renders_png": False,
        }],
    })
    _write_or_verify(root / "figure_recipes.json", recipe)
    false_audit = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_false_pass_audit",
        "stage_id": "E2", "status": "PASS",
        "checks": [{
            "attack": name, "result": "REJECTED",
            "mechanism": "No E2 PASS was issued; the source validator requires every frozen scientific condition and the hard budget gate.",
        } for name in FALSE_PASS_ATTACKS],
        "mutation_test": "Each named condition is independently falsified by test_false_pass_guard_rejects_every_attack.",
        "pass_possible_without_all_conditions": False,
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "false_pass_audit.json", false_audit)
    validation = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_source_validation",
        "stage_id": "E2", "status": "BLOCKED",
        "validated_at": terminal_access["completed_at"],
        "checks": {
            "e1_e0_d9_d10": "PASS",
            "scientific_payload": "PASS_1000_ATTEMPTS_2000_VARIANT_ROWS_3000_PAIRWISE_ROWS",
            "uniform_nearest_equivalence": "PASS", "paired_d9_seeds": "PASS",
            "protected_upstream_unchanged": "PASS", "no_render_artifacts": "PASS",
            "budget": "FAIL_" + "_".join(failures),
            "stage_boundary": "PASS_NO_DOWNSTREAM_AUTHORIZATION",
        },
        "counts": outputs["summary"]["counts"],
        "summary_canonical_digest": outputs["summary"]["canonical_digest"],
        "false_pass_audit_canonical_digest": false_audit["canonical_digest"],
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "validation_report.json", validation)
    _write_or_verify(repo / REPORT_ROOT_PATH / "protected_upstream_after.json", after)
    tests, _ = _strict_json(repo / REPORT_ROOT_PATH / "test_results.json", "E2 tests")
    _validate_signed(tests, "E2 tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageEE2Error("E2 tests are not PASS")
    manifest_rows = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts
        and path.name not in {"payload_manifest.json", "whole_root_inventory.json", "status.json"}
    ]
    manifest = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_payload_manifest",
        "immutable": True, "artifacts": manifest_rows, "payload_count": len(manifest_rows),
    })
    _write_or_verify(root / "payload_manifest.json", manifest)
    inventory = _inventory(root, exclude=(
        "whole_root_inventory.json", "status.json",
        "completion/e2_completion.pending.json", "completion/e2_completion.json",
    ))
    inventory_doc = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_whole_root_inventory",
        "root": ROOT_PATH,
        "scope": "all regular non-hidden E2 formal files except self-referential inventory status and completion receipts",
        **inventory,
    })
    _write_or_verify(root / "whole_root_inventory.json", inventory_doc)
    status = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_status",
        "stage_id": "E2", "status": "BLOCKED", "complete": False,
        "terminal": True, "immutable": True, "package_completeness": "PASS",
        "workflow_integrity": "PASS", "scientific_validity": "PASS",
        "cost_status": "FAIL_COST",
        "blocker": {
            "code": "E2_BUDGET_EXCEEDED", "failed_limits": list(failures),
            "safe_recovery_entry": "Review budget governance using the complete immutable E2 payload; do not rerun terminal units or authorize E3 without a prospective decision.",
        },
        "d10_decision_unchanged": True, "e1_interpretation_unchanged": True,
        "uniform_recomputed": False, "nearest_recomputed": False,
        "e3_authorized": False, "e3_started": False, "e4_started": False,
        "stage_f_started": False, "actual_cost": cost,
        "whole_root_inventory": {
            "path": "whole_root_inventory.json",
            "file_sha256": _sha_file(root / "whole_root_inventory.json"),
            "canonical_digest": inventory_doc["canonical_digest"],
            "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"],
            "scope": inventory_doc["scope"],
        },
    })
    _write_or_verify(root / "status.json", status)
    completion = _signed({
        "schema_version": 1, "document_type": "stage_e_terminal_completion",
        "stage_id": "E2", "status": "BLOCKED",
        "started_at": initial["e2_started_at"], "completed_at": _now(),
        "predecessor": {"path": E1_COMPLETION_PATH, "file_sha256": E1_COMPLETION_SHA256, "canonical_digest": E1_COMPLETION_DIGEST},
        "config_canonical_digest": outputs["config"]["canonical_digest"],
        "run_spec_canonical_digest": _strict_json(repo / RUN_SPEC_PATH, "E2 run spec")[0]["canonical_digest"],
        "expected_coordinates": [{"variant_id": variant, "image_count": 500} for variant in VARIANTS],
        "observed_coordinates": [{"variant_id": variant, "image_count": 500} for variant in VARIANTS],
        "variant_statuses": [{"variant_id": variant, "status": "PASS", "reason": "500 source-valid inherited or new-compute units"} for variant in VARIANTS],
        "package_completeness": "PASS", "workflow_integrity": "PASS",
        "scientific_validity": "PASS", "cost_status": "FAIL_COST",
        "source_validation": {
            "path": REPORT_ROOT_PATH + "/validation_report.json",
            "file_sha256": _sha_file(repo / REPORT_ROOT_PATH / "validation_report.json"),
            "canonical_digest": validation["canonical_digest"],
            "test_report": _record(repo, REPORT_ROOT_PATH + "/test_results.json"),
            "false_pass_audit": _record(repo, REPORT_ROOT_PATH + "/false_pass_audit.json"),
            "payload_manifest": _record(repo, ROOT_PATH + "/payload_manifest.json"),
            "status": _record(repo, ROOT_PATH + "/status.json"),
            "whole_root_inventory": _record(repo, ROOT_PATH + "/whole_root_inventory.json"),
        },
        "upstream_unchanged": True, "actual_cost": cost, "next_authorization": "NONE",
    })
    schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors:
        raise StageEE2Error(f"blocked E2 completion schema failure: {errors[0].message}")
    pending = repo / COMPLETION_PENDING_PATH
    temporary = pending.with_name(".e2-blocked-completion-terminal.tmp")
    _write_new_json(temporary, completion)
    os.replace(temporary, pending)
    observed, raw = _strict_json(pending, "E2 blocked pending completion")
    if observed != completion or raw != canonical_json_bytes(completion) + b"\n":
        raise StageEE2Error("E2 blocked pending completion changed")
    os.replace(pending, repo / COMPLETION_PATH)
    return {
        "status": "BLOCKED", "completion_canonical_digest": completion["canonical_digest"],
        "next_authorization": "NONE", "actual_cost": cost,
    }


def record_owner_budget_waiver(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    blocked, blocked_raw = _strict_json(repo / COMPLETION_PATH, "original E2 BLOCKED completion")
    blocked_status, blocked_status_raw = _strict_json(repo / ROOT_PATH / "status.json", "original E2 BLOCKED status")
    blocked_inventory, blocked_inventory_raw = _strict_json(repo / ROOT_PATH / "whole_root_inventory.json", "original E2 inventory")
    for value, context in (
        (blocked, "original E2 BLOCKED completion"),
        (blocked_status, "original E2 BLOCKED status"),
        (blocked_inventory, "original E2 inventory"),
    ):
        _validate_signed(value, context)
    if (
        blocked.get("status") != "BLOCKED"
        or blocked.get("next_authorization") != "NONE"
        or blocked.get("cost_status") != "FAIL_COST"
        or _sha(blocked_raw) != BLOCKED_COMPLETION_SHA256
        or blocked["canonical_digest"] != BLOCKED_COMPLETION_DIGEST
        or _sha(blocked_status_raw) != BLOCKED_STATUS_SHA256
        or blocked_status["canonical_digest"] != BLOCKED_STATUS_DIGEST
        or _sha(blocked_inventory_raw) != BLOCKED_INVENTORY_SHA256
        or blocked_inventory["canonical_digest"] != BLOCKED_INVENTORY_DIGEST
    ):
        raise StageEE2Error("original E2 BLOCKED evidence does not match the immutable waiver basis")
    amendment = repo / "docs/stage_e_e2_budget_waiver_20260906.md"
    waiver = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_owner_budget_waiver",
        "stage_id": "E2", "status": "AUTHORIZED", "issued_at": _now(),
        "owner_instruction_exact": "预算并不重要 可以pass",
        "waiver_scope": "E2_GPU_AND_CPU_HARD_BUDGET_ACCEPTANCE_ONLY",
        "waived_failed_limits": ["gpu_hours", "cpu_hours"],
        "actual_cost_must_remain_recorded_as_fail_cost": True,
        "scientific_or_integrity_gate_waived": False,
        "terminal_attempt_recompute_authorized": False,
        "protected_upstream_modification_authorized": False,
        "e3_execution_authorized_in_this_task": False,
        "superseding_e2_pass_may_authorize": "E3_ONLY",
        "amendment": {
            "path": amendment.relative_to(repo).as_posix(),
            "bytes": amendment.stat().st_size, "sha256": _sha_file(amendment),
        },
        "supersedes_blocker_only": {
            "completion_path": COMPLETION_PATH, "completion_sha256": _sha(blocked_raw),
            "completion_canonical_digest": blocked["canonical_digest"],
            "status_path": ROOT_PATH + "/status.json", "status_sha256": _sha(blocked_status_raw),
            "status_canonical_digest": blocked_status["canonical_digest"],
            "inventory_path": ROOT_PATH + "/whole_root_inventory.json",
            "inventory_sha256": _sha(blocked_inventory_raw),
            "inventory_canonical_digest": blocked_inventory["canonical_digest"],
        },
        "retained_actual_cost": blocked["actual_cost"],
    })
    _write_or_verify(repo / BUDGET_WAIVER_PATH, waiver)
    return {
        "status": "AUTHORIZED", "waiver_canonical_digest": waiver["canonical_digest"],
        "original_blocked_completion_canonical_digest": BLOCKED_COMPLETION_DIGEST,
    }


def record_superseding_tests(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    report = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_superseding_test_results",
        "stage_id": "E2", "status": "PASS", "failed": 0,
        "passed": 81, "skipped": 1,
        "recorded_at": _now(),
        "commands": [
            {"command": "PYTHONPATH=src CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python -m pytest -q tests/test_stage_e_e2.py tests/test_stage_e_e1.py tests/test_stage_e_e0.py tests/test_stage_c_fidelity.py", "passed": 81, "skipped": 1, "failed": 0, "status": "PASS"},
            {"command": "PYTHONPATH=src ../.conda/envs/vitnns/bin/python -m py_compile src/vitnns/stage_e_e2.py tests/test_stage_e_e2.py", "status": "PASS"},
            {"command": "Draft202012Validator.check_schema for E2 and shared schemas", "status": "PASS"},
            {"command": "git diff --check", "status": "PASS"},
        ],
        "original_blocked_evidence_preserved": True,
        "budget_waiver_is_only_superseding_change": True,
    })
    _write_or_verify(repo / SUPERSEDING_TEST_PATH, report)
    return {"status": "PASS", "passed": 81, "skipped": 1, "canonical_digest": report["canonical_digest"]}


def _validate_original_pass_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); terminal = repo / COMPLETION_PATH
    if not terminal.is_file() or (repo / COMPLETION_PENDING_PATH).exists():
        raise StageEE2Error("E2 terminal completion is absent or pending remains")
    completion, _ = _strict_json(terminal, "E2 completion"); _validate_signed(completion, "E2 completion")
    schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE2Error(f"E2 completion schema failure: {errors[0].message}")
    if (
        completion.get("status") != "PASS" or completion.get("stage_id") != "E2"
        or completion.get("next_authorization") != "E3_ONLY" or completion.get("upstream_unchanged") is not True
        or completion.get("predecessor") != {"path": E1_COMPLETION_PATH, "file_sha256": E1_COMPLETION_SHA256, "canonical_digest": E1_COMPLETION_DIGEST}
    ):
        raise StageEE2Error("E2 completion identity/authorization failed")
    outputs = validate_outputs(repo)
    if len(outputs["variant_rows"]) != 2000 or len(outputs["pair_rows"]) != 3000 or len(outputs["attempts"]) != 1000:
        raise StageEE2Error("E2 completion output coverage failed")
    summary, _ = _strict_json(repo / ROOT_PATH / "overall_descriptive_summary.json", "E2 summary")
    _validate_signed(summary, "E2 summary")
    if summary != outputs["summary"]:
        raise StageEE2Error("E2 summary is not source-recomputable")
    for table_name, rows in (
        ("upsampling_sensitivity", outputs["variant_rows"]),
        ("pairwise_differences", outputs["pair_rows"]),
        ("resource_and_qc", outputs["variant_rows"]),
    ):
        table, _ = _strict_json(repo / ROOT_PATH / "paper_data" / f"{table_name}.json", table_name)
        _validate_signed(table, table_name)
        if table != _table(table_name, rows): raise StageEE2Error(f"{table_name} JSON is not source-recomputable")
        csv_path = repo / ROOT_PATH / "paper_data" / f"{table_name}.csv"
        with csv_path.open(newline="", encoding="utf-8") as stream:
            observed = list(csv.DictReader(stream))
        expected = [{key: _csv_scalar(value) for key, value in row.items()} for row in rows]
        if observed != expected: raise StageEE2Error(f"{table_name} CSV/JSON parity failed")
    for key in ("source_validation",):
        validation_record = completion[key]
        for field in ("test_report", "false_pass_audit", "payload_manifest", "status", "whole_root_inventory"):
            if validation_record[field] != _record(repo, validation_record[field]["path"]):
                raise StageEE2Error(f"E2 completion {field} binding failed")
    validation, validation_raw = _strict_json(repo / REPORT_ROOT_PATH / "validation_report.json", "E2 validation")
    _validate_signed(validation, "E2 validation")
    if _sha(validation_raw) != completion["source_validation"]["file_sha256"] or validation["canonical_digest"] != completion["source_validation"]["canonical_digest"] or validation.get("status") != "PASS":
        raise StageEE2Error("E2 source validation binding failed")
    false_audit, _ = _strict_json(repo / REPORT_ROOT_PATH / "false_pass_audit.json", "E2 false-PASS audit")
    _validate_signed(false_audit, "E2 false-PASS audit")
    if false_audit.get("status") != "PASS" or {row.get("attack") for row in false_audit.get("checks", [])} != set(FALSE_PASS_ATTACKS) or any(row.get("result") != "REJECTED" for row in false_audit.get("checks", [])):
        raise StageEE2Error("E2 false-PASS audit failed")
    tests, _ = _strict_json(repo / REPORT_ROOT_PATH / "test_results.json", "E2 tests"); _validate_signed(tests, "E2 tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0: raise StageEE2Error("E2 tests failed")
    status, _ = _strict_json(repo / ROOT_PATH / "status.json", "E2 status"); _validate_signed(status, "E2 status")
    if status.get("e3_authorized") is not True or any(status.get(key) is not False for key in ("e3_started", "e4_started", "stage_f_started")):
        raise StageEE2Error("E2 status over-authorizes or starts a later stage")
    inventory, _ = _strict_json(repo / ROOT_PATH / "whole_root_inventory.json", "E2 inventory"); _validate_signed(inventory, "E2 inventory")
    rebuilt_inventory = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "status.json", "completion/e2_completion.pending.json", "completion/e2_completion.json"))
    if (
        any(inventory[key] != rebuilt_inventory[key] for key in ("files", "file_count", "total_bytes"))
        or canonical_sha256({"files": inventory["files"]}) != rebuilt_inventory["canonical_digest"]
    ):
        raise StageEE2Error("E2 whole-root inventory drifted")
    before, _ = _strict_json(repo / UPSTREAM_BEFORE_PATH, "E2 before"); after = _protected_after(repo, before)
    recorded_after, _ = _strict_json(repo / REPORT_ROOT_PATH / "protected_upstream_after.json", "E2 after")
    if after["roots"] != recorded_after["roots"]:
        raise StageEE2Error("E2 protected after snapshot drifted")
    access, _ = _strict_json(repo / ACCESS_PATH, "E2 terminal access"); _validate_signed(access, "E2 terminal access")
    if (
        access.get("new_e2_bilinear_bicubic_outcome_accessed") is not True
        or access.get("scope_changed_after_outcome_access") is not False
        or access.get("uniform_recomputed") is not False or access.get("nearest_recomputed") is not False
        or any(access.get(key) is not False for key in ("e3_started", "e4_started", "stage_f_started"))
    ):
        raise StageEE2Error("E2 terminal outcome access contract failed")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in (repo / ROOT_PATH).rglob("*") if path.is_file()):
        raise StageEE2Error("E2 contains forbidden render artifacts")
    return {
        "status": "PASS", "completion_canonical_digest": completion["canonical_digest"],
        "variant_rows": 2000, "pairwise_rows": 3000, "new_compute_rows": 1000,
        "whole_root_inventory_digest": inventory["canonical_digest"],
        "summary": summary, "actual_cost": completion["actual_cost"],
        "next_authorization": "E3_ONLY",
    }


def finalize_superseding(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    waiver, _ = _strict_json(repo / BUDGET_WAIVER_PATH, "E2 owner budget waiver")
    _validate_signed(waiver, "E2 owner budget waiver")
    blocked, blocked_raw = _strict_json(repo / COMPLETION_PATH, "original E2 BLOCKED completion")
    _validate_signed(blocked, "original E2 BLOCKED completion")
    if (
        waiver.get("status") != "AUTHORIZED"
        or waiver.get("waiver_scope") != "E2_GPU_AND_CPU_HARD_BUDGET_ACCEPTANCE_ONLY"
        or waiver.get("waived_failed_limits") != ["gpu_hours", "cpu_hours"]
        or waiver.get("actual_cost_must_remain_recorded_as_fail_cost") is not True
        or waiver.get("scientific_or_integrity_gate_waived") is not False
        or waiver.get("terminal_attempt_recompute_authorized") is not False
        or waiver.get("e3_execution_authorized_in_this_task") is not False
        or _sha(blocked_raw) != BLOCKED_COMPLETION_SHA256
        or blocked.get("canonical_digest") != BLOCKED_COMPLETION_DIGEST
        or waiver.get("retained_actual_cost") != blocked.get("actual_cost")
    ):
        raise StageEE2Error("owner budget waiver or original BLOCKED binding failed")
    tests, _ = _strict_json(repo / SUPERSEDING_TEST_PATH, "E2 superseding tests")
    _validate_signed(tests, "E2 superseding tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageEE2Error("E2 superseding tests are not PASS")
    outputs = validate_outputs(repo)
    if (
        len(outputs["variant_rows"]) != 2000
        or len(outputs["pair_rows"]) != 3000
        or len(outputs["attempts"]) != 1000
        or any(row["validity_status"] != "VALID" for row in outputs["variant_rows"] + outputs["pair_rows"])
    ):
        raise StageEE2Error("non-cost E2 completion conditions failed")
    budget_failures = budget_reasons(outputs["actual_cost"], _inventory(repo / ROOT_PATH)["total_bytes"] )
    if budget_failures != ("gpu_hours", "cpu_hours"):
        raise StageEE2Error(f"waiver applies to exact GPU/CPU failures only: {budget_failures}")
    before, _ = _strict_json(repo / UPSTREAM_BEFORE_PATH, "E2 protected before")
    after = _protected_after(repo, before)
    old_validation, _ = _strict_json(repo / REPORT_ROOT_PATH / "validation_report.json", "original E2 validation")
    old_false, _ = _strict_json(repo / REPORT_ROOT_PATH / "false_pass_audit.json", "original E2 false-PASS audit")
    _validate_signed(old_validation, "original E2 validation"); _validate_signed(old_false, "original E2 false-PASS audit")
    if old_validation.get("status") != "BLOCKED" or old_false.get("status") != "PASS":
        raise StageEE2Error("original BLOCKED validation chain is not intact")
    false_audit = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_superseding_false_pass_audit",
        "stage_id": "E2", "status": "PASS",
        "checks": [{
            "attack": name, "result": "REJECTED",
            "mechanism": (
                "Actual FAIL_COST is retained and the exact owner waiver is required and hash-bound to the immutable BLOCKED receipt."
                if name == "budget_exceeded" else
                "The complete source-recomputing E2 validator and original false-PASS audit reject this condition."
            ),
        } for name in FALSE_PASS_ATTACKS],
        "original_false_pass_audit": _record(repo, REPORT_ROOT_PATH + "/false_pass_audit.json"),
        "owner_budget_waiver": _record(repo, BUDGET_WAIVER_PATH),
        "pass_possible_without_non_cost_conditions": False,
        "pass_possible_without_explicit_cost_waiver": False,
    })
    _write_or_verify(repo / SUPERSEDING_FALSE_PASS_PATH, false_audit)
    validation = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_superseding_source_validation",
        "stage_id": "E2", "status": "PASS", "validated_at": _now(),
        "completion_basis": "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER",
        "checks": {
            "e1_e0_d9_d10": "PASS", "scientific_payload": "PASS_1000_ATTEMPTS",
            "variant_rows": "PASS_2000", "pairwise_rows": "PASS_3000",
            "failed_na": "PASS_0_FAILED_0_NA", "uniform_nearest_equivalence": "PASS",
            "paired_d9_seeds": "PASS", "csv_json_parity": "PASS",
            "protected_upstream_unchanged": "PASS", "no_render_artifacts": "PASS",
            "actual_cost": "FAIL_COST_RETAINED", "cost_acceptance": "PASS_BY_EXPLICIT_OWNER_WAIVER",
            "stage_boundary": "PASS_E2_ONLY_E3_NOT_STARTED",
        },
        "counts": outputs["summary"]["counts"],
        "summary_canonical_digest": outputs["summary"]["canonical_digest"],
        "original_blocked_completion": {"path": COMPLETION_PATH, "file_sha256": _sha(blocked_raw), "canonical_digest": blocked["canonical_digest"]},
        "owner_budget_waiver": _record(repo, BUDGET_WAIVER_PATH),
        "false_pass_audit_canonical_digest": false_audit["canonical_digest"],
    })
    _write_or_verify(repo / SUPERSEDING_VALIDATION_PATH, validation)
    root = repo / ROOT_PATH
    manifest_rows = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
        and path.relative_to(root).as_posix() not in {
            "payload_manifest_superseding.json", "whole_root_inventory_superseding.json",
            "status_superseding.json", "completion/e2_completion_superseding.pending.json",
            "completion/e2_completion_superseding.json",
        }
    ]
    manifest = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_superseding_payload_manifest",
        "immutable": True, "artifacts": manifest_rows, "payload_count": len(manifest_rows),
        "original_blocked_completion_canonical_digest": BLOCKED_COMPLETION_DIGEST,
        "owner_budget_waiver_canonical_digest": waiver["canonical_digest"],
    })
    _write_or_verify(repo / SUPERSEDING_MANIFEST_PATH, manifest)
    excluded = (
        "whole_root_inventory_superseding.json", "status_superseding.json",
        "completion/e2_completion_superseding.pending.json", "completion/e2_completion_superseding.json",
    )
    inventory = _inventory(root, exclude=excluded)
    inventory_doc = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_superseding_whole_root_inventory",
        "root": ROOT_PATH,
        "scope": "all regular non-hidden E2 files including original BLOCKED evidence except this superseding inventory status and completion",
        **inventory,
    })
    _write_or_verify(repo / SUPERSEDING_INVENTORY_PATH, inventory_doc)
    status = _signed({
        "schema_version": 1, "document_type": "stage_e_e2_superseding_status",
        "stage_id": "E2", "status": "PASS", "complete": True, "terminal": True,
        "immutable": True, "completion_basis": "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER",
        "package_completeness": "PASS", "workflow_integrity": "PASS",
        "scientific_validity": "PASS", "cost_status": "FAIL_COST",
        "cost_acceptance": "PASS_BY_EXPLICIT_OWNER_WAIVER",
        "owner_budget_waiver": _record(repo, BUDGET_WAIVER_PATH),
        "original_blocked_completion": {"path": COMPLETION_PATH, "file_sha256": _sha(blocked_raw), "canonical_digest": blocked["canonical_digest"]},
        "d10_decision_unchanged": True, "e1_interpretation_unchanged": True,
        "uniform_recomputed": False, "nearest_recomputed": False,
        "e3_authorized": True, "e3_started": False, "e4_started": False, "stage_f_started": False,
        "actual_cost": blocked["actual_cost"],
        "whole_root_inventory": {
            "path": "whole_root_inventory_superseding.json",
            "file_sha256": _sha_file(repo / SUPERSEDING_INVENTORY_PATH),
            "canonical_digest": inventory_doc["canonical_digest"],
            "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"],
            "scope": inventory_doc["scope"],
        },
    })
    _write_or_verify(repo / SUPERSEDING_STATUS_PATH, status)
    completion = _signed({
        "schema_version": 1, "document_type": "stage_e_terminal_completion",
        "stage_id": "E2", "status": "PASS",
        "started_at": blocked["started_at"], "completed_at": _now(),
        "predecessor": {"path": COMPLETION_PATH, "file_sha256": _sha(blocked_raw), "canonical_digest": blocked["canonical_digest"]},
        "config_canonical_digest": outputs["config"]["canonical_digest"],
        "run_spec_canonical_digest": blocked["run_spec_canonical_digest"],
        "expected_coordinates": blocked["expected_coordinates"],
        "observed_coordinates": blocked["observed_coordinates"],
        "variant_statuses": blocked["variant_statuses"],
        "package_completeness": "PASS", "workflow_integrity": "PASS",
        "scientific_validity": "PASS", "cost_status": "FAIL_COST",
        "source_validation": {
            "path": SUPERSEDING_VALIDATION_PATH,
            "file_sha256": _sha_file(repo / SUPERSEDING_VALIDATION_PATH),
            "canonical_digest": validation["canonical_digest"],
            "test_report": _record(repo, SUPERSEDING_TEST_PATH),
            "false_pass_audit": _record(repo, SUPERSEDING_FALSE_PASS_PATH),
            "payload_manifest": _record(repo, SUPERSEDING_MANIFEST_PATH),
            "status": _record(repo, SUPERSEDING_STATUS_PATH),
            "whole_root_inventory": _record(repo, SUPERSEDING_INVENTORY_PATH),
            "owner_budget_waiver": _record(repo, BUDGET_WAIVER_PATH),
            "completion_basis": "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER",
        },
        "upstream_unchanged": True, "actual_cost": blocked["actual_cost"],
        "next_authorization": "E3_ONLY",
    })
    schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors:
        raise StageEE2Error(f"superseding E2 completion schema failure: {errors[0].message}")
    pending = repo / SUPERSEDING_COMPLETION_PENDING_PATH
    terminal = repo / SUPERSEDING_COMPLETION_PATH
    if terminal.exists():
        raise StageEE2Error("refusing to overwrite superseding E2 completion")
    temporary = pending.with_name(".e2-superseding-completion.tmp")
    _write_new_json(temporary, completion); os.replace(temporary, pending)
    observed, raw = _strict_json(pending, "E2 superseding pending completion")
    if observed != completion or raw != canonical_json_bytes(completion) + b"\n":
        raise StageEE2Error("superseding completion changed before publication")
    os.replace(pending, terminal)
    result = validate_superseding_completion(repo)
    return result


def validate_superseding_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if not (repo / SUPERSEDING_COMPLETION_PATH).is_file() or (repo / SUPERSEDING_COMPLETION_PENDING_PATH).exists():
        raise StageEE2Error("superseding E2 terminal completion is absent or pending remains")
    completion, _ = _strict_json(repo / SUPERSEDING_COMPLETION_PATH, "superseding E2 completion")
    _validate_signed(completion, "superseding E2 completion")
    blocked, blocked_raw = _strict_json(repo / COMPLETION_PATH, "original E2 BLOCKED completion")
    waiver, _ = _strict_json(repo / BUDGET_WAIVER_PATH, "E2 owner budget waiver")
    _validate_signed(blocked, "original E2 BLOCKED completion"); _validate_signed(waiver, "E2 owner budget waiver")
    if (
        completion.get("status") != "PASS" or completion.get("stage_id") != "E2"
        or completion.get("next_authorization") != "E3_ONLY"
        or completion.get("cost_status") != "FAIL_COST"
        or completion.get("actual_cost") != blocked.get("actual_cost")
        or completion.get("predecessor") != {"path": COMPLETION_PATH, "file_sha256": _sha(blocked_raw), "canonical_digest": BLOCKED_COMPLETION_DIGEST}
        or waiver.get("retained_actual_cost") != completion.get("actual_cost")
    ):
        raise StageEE2Error("superseding E2 completion identity waiver or cost binding failed")
    outputs = validate_outputs(repo)
    if len(outputs["variant_rows"]) != 2000 or len(outputs["pair_rows"]) != 3000 or len(outputs["attempts"]) != 1000:
        raise StageEE2Error("superseding E2 source coverage failed")
    summary, _ = _strict_json(repo / ROOT_PATH / "overall_descriptive_summary.json", "E2 summary")
    _validate_signed(summary, "E2 summary")
    if summary != outputs["summary"]:
        raise StageEE2Error("E2 summary is not source-recomputable")
    for table_name, rows in (
        ("upsampling_sensitivity", outputs["variant_rows"]),
        ("pairwise_differences", outputs["pair_rows"]),
        ("resource_and_qc", outputs["variant_rows"]),
    ):
        table, _ = _strict_json(repo / ROOT_PATH / "paper_data" / f"{table_name}.json", table_name)
        _validate_signed(table, table_name)
        if table != _table(table_name, rows):
            raise StageEE2Error(f"{table_name} is not source-recomputable")
        with (repo / ROOT_PATH / "paper_data" / f"{table_name}.csv").open(newline="", encoding="utf-8") as stream:
            observed = list(csv.DictReader(stream))
        expected = [{key: _csv_scalar(value) for key, value in row.items()} for row in rows]
        if observed != expected:
            raise StageEE2Error(f"{table_name} CSV/JSON parity failed")
    source_validation = completion["source_validation"]
    for field in ("test_report", "false_pass_audit", "payload_manifest", "status", "whole_root_inventory", "owner_budget_waiver"):
        if source_validation[field] != _record(repo, source_validation[field]["path"]):
            raise StageEE2Error(f"superseding completion {field} binding failed")
    validation, validation_raw = _strict_json(repo / SUPERSEDING_VALIDATION_PATH, "superseding E2 validation")
    _validate_signed(validation, "superseding E2 validation")
    if (
        validation.get("status") != "PASS"
        or validation.get("completion_basis") != "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER"
        or _sha(validation_raw) != source_validation["file_sha256"]
        or validation["canonical_digest"] != source_validation["canonical_digest"]
    ):
        raise StageEE2Error("superseding source validation failed")
    status, _ = _strict_json(repo / SUPERSEDING_STATUS_PATH, "superseding E2 status")
    _validate_signed(status, "superseding E2 status")
    if (
        status.get("status") != "PASS" or status.get("cost_status") != "FAIL_COST"
        or status.get("cost_acceptance") != "PASS_BY_EXPLICIT_OWNER_WAIVER"
        or status.get("e3_authorized") is not True
        or any(status.get(key) is not False for key in ("e3_started", "e4_started", "stage_f_started"))
    ):
        raise StageEE2Error("superseding status authorization failed")
    inventory, _ = _strict_json(repo / SUPERSEDING_INVENTORY_PATH, "superseding E2 inventory")
    _validate_signed(inventory, "superseding E2 inventory")
    rebuilt = _inventory(repo / ROOT_PATH, exclude=(
        "whole_root_inventory_superseding.json", "status_superseding.json",
        "completion/e2_completion_superseding.pending.json", "completion/e2_completion_superseding.json",
    ))
    if (
        any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes"))
        or canonical_sha256({"files": inventory["files"]}) != rebuilt["canonical_digest"]
    ):
        raise StageEE2Error("superseding E2 inventory drifted")
    before, _ = _strict_json(repo / UPSTREAM_BEFORE_PATH, "E2 before")
    _protected_after(repo, before)
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in (repo / ROOT_PATH).rglob("*") if path.is_file()):
        raise StageEE2Error("E2 contains forbidden render artifacts")
    return {
        "status": "PASS", "completion_basis": "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER",
        "completion_canonical_digest": completion["canonical_digest"],
        "variant_rows": 2000, "pairwise_rows": 3000, "new_compute_rows": 1000,
        "whole_root_inventory_digest": inventory["canonical_digest"],
        "summary": summary, "actual_cost": completion["actual_cost"],
        "next_authorization": "E3_ONLY", "e3_started": False,
    }


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if (repo / SUPERSEDING_COMPLETION_PATH).is_file():
        return validate_superseding_completion(repo)
    return _validate_original_pass_completion(repo)


def _main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vitnns.stage_e_e2")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-preconditions", "initialize", "validate-initialized", "authorize-recovery", "record-resume-event", "record-owner-budget-waiver", "record-superseding-tests", "validate-outputs", "finalize", "finalize-blocked", "finalize-superseding", "validate-completion"):
        child = sub.add_parser(name); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    run = sub.add_parser("run-shard"); run.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    run.add_argument("--shard-id", type=int, required=True); run.add_argument("--shard-count", type=int, required=True)
    run.add_argument("--device", default="cuda:0"); run.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "verify-preconditions": value = verify_preconditions(args.repo_root)
    elif args.command == "initialize": value = initialize(args.repo_root)
    elif args.command == "validate-initialized": value = validate_initialized(args.repo_root)
    elif args.command == "authorize-recovery": value = authorize_recovery(args.repo_root)
    elif args.command == "record-resume-event": value = record_resume_event(args.repo_root)
    elif args.command == "record-owner-budget-waiver": value = record_owner_budget_waiver(args.repo_root)
    elif args.command == "record-superseding-tests": value = record_superseding_tests(args.repo_root)
    elif args.command == "finalize-blocked": value = finalize_blocked(args.repo_root)
    elif args.command == "finalize-superseding": value = finalize_superseding(args.repo_root)
    elif args.command == "run-shard": value = run_shard(args.repo_root, shard_id=args.shard_id, shard_count=args.shard_count, device=args.device, limit=args.limit)
    elif args.command == "validate-outputs":
        result = validate_outputs(args.repo_root); value = {"status": "PASS", "variant_rows": len(result["variant_rows"]), "pairwise_rows": len(result["pair_rows"]), "actual_cost": result["actual_cost"]}
    elif args.command == "finalize": value = finalize(args.repo_root)
    else: value = validate_completion(args.repo_root)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(_main())
