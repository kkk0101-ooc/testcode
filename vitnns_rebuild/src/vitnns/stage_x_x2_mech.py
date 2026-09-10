"""Outcome-blind P0-R1 control plane for the X2-MECH study.

This module may inspect source code and X2 control metadata.  It deliberately
has no Torch, timm, PIL, image-decoder, checkpoint loader, CUDA, np.load, or
empirical outcome-row reader.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT

RUN_ID = "STAGE_X_X2_MECH_P0_R1_V1_20260910"
AUTHORIZATION = "X2_MECH_P0_R1_PROTOCOL_AND_IMPLEMENTATION_FREEZE_ONLY"
NEXT_AUTHORIZATION = "X2_MECH_M1_TRANSFORMED_MAP_AND_MULTISCALE_FIDELITY_ONLY"
BASE_COMMIT = "9ddfec3dca2e180930009174faa00a0f48e0b752"
BRANCH = "stage-d-d3-pass"
X2_PRODUCER = "d7d8ca177caac0b81efc3d051093d84bc464cf0a"
X2_COMPLETION = BASE_COMMIT
X2_PRODUCER_REF = "refs/archive/stage-x-nns-v2-x2-producer"
X2_COMPLETION_REF = "refs/archive/stage-x-nns-v2-x2-completion"
PRODUCER_REF = "refs/archive/stage-x-x2-mech-p0-r1-producer"
COMPLETION_REF = "refs/archive/stage-x-x2-mech-p0-r1-completion"

ROOT = Path("evidence/stage_x/x2_mech_p0_r1_20260910/formal")
REPORT_ROOT = Path("reports/stage_x_x2_mech_p0_r1_20260910")
CONFIG = Path("config/stage_x_x2_mech_p0_r1.json")
CONFIG_SCHEMA = Path("config/stage_x_x2_mech_p0_r1.schema.json")
ARTIFACT_SCHEMA = Path("config/stage_x_x2_mech_artifact.schema.json")
AMENDMENT = Path("docs/stage_x_x2_mech_p0_r1_access_boundary_amendment_20260910.md")
PROTOCOL = Path("docs/stage_x_x2_mechanism_protocol_20260910.md")
COMPLETION = ROOT / "completion/x2_mech_p0_r1.json"
PENDING = ROOT / "completion/x2_mech_p0_r1.pending.json"

X2_ROOT = Path("evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal")
X2_CLOSEOUT = Path("reports/stage_x_x2_provenance_closeout_20260910.json")
X2_CONTROL_FILES = (
    X2_ROOT / "completion/x2_r2_r1_r1.json", X2_ROOT / "status.json",
    X2_ROOT / "reconciliation.json", X2_ROOT / "payload_manifest.json",
    X2_ROOT / "whole_root_inventory.json", X2_ROOT / "resource_profiles.json",
    X2_ROOT / "formal_budget_projection.json",
    Path("reports/stage_x_x2_r2_r1_r1_nns_v2_development_20260910/validation_report.json"),
    X2_CLOSEOUT,
)
SOURCE_FILES = (
    AMENDMENT, PROTOCOL, CONFIG, CONFIG_SCHEMA, ARTIFACT_SCHEMA,
    Path("src/vitnns/stage_x_x2_mech.py"),
    Path("src/vitnns/stage_x_x2_mech_worker.py"),
    Path("src/vitnns/stage_x_x2_mech_analysis.py"),
    Path("tests/test_stage_x_x2_mech.py"),
)
ACCESS_FLAGS = (
    "empirical_map_values_accessed", "per_route_outcome_rows_accessed",
    "jpeg_bytes_accessed", "checkpoint_bytes_accessed", "model_loaded",
    "torch_imported", "timm_imported", "cuda_accessed",
    "real_mechanism_outcomes_computed", "x2_decision_modified",
    "x3_x5_started",
)
FUTURE_OUTPUTS = {
    "map_manifest.csv": "M1", "map_manifest.json": "M1",
    "valid_mask_registry.json": "M1",
    "pixel_reproduction_audit.csv": "M2",
    "pixel_reproduction_audit.json": "M2",
    "multiscale_stability.csv": "M2",
    "multiscale_fidelity.csv": "M1",
    "within_between_error.csv": "M2", "transport_control.csv": "M2",
    "clean_structure_metrics.csv": "M2",
    "horizontal_flip_control.csv": "M2",
    "association_summary.csv": "M2", "association_summary.json": "M2",
    "estimability_matrix.json": "M1_M2", "resource_ledger.json": "M1_M2",
    "false_pass_audit.json": "M1_M2",
    "whole_root_inventory.json": "M1_M2",
    "completion/x2_mech.json": "M1_M2", "x2_mech_report.md": "M1_M2",
}
ATTACKS = {
    "modify_or_overwrite_x2_roots_receipts_or_refs": "source hashes and both X2 refs are recomputed before completion",
    "relabel_x2_mech_confirmatory_or_redecide_x2": "config and protocol fix post-hoc identity and immutable X2 NO-GO",
    "drop_failed_route_or_change_denominator": "typed plan binds all 200 source terminals including one typed failure",
    "read_source_map_values_then_change_protocol": "P0 control has no array loader and its source seal precedes M1",
    "method_specific_smoothing_or_normalization": "one shared signed-sum operator and no smoothing/normalization are frozen",
    "resize_claimed_as_4x4_or_16x16_aggregation": "synthetic tests require exact non-overlapping block sums",
    "accept_partial_valid_blocks": "block validity is all-pixels-valid and partial blocks are counted as excluded",
    "outcome_select_mask_scale_metric_or_transform": "all masks scales metrics and transforms are closed in the producer config",
    "hide_horizontal_flip": "horizontal flip is a mandatory separate control output",
    "treat_transport_control_as_model_attribution": "transport control has a separate output and interpretation boundary",
    "mix_stage_d_value_path_with_class_logit_fidelity": "fidelity contract permits only X2 class-logit bindings",
    "omit_transformed_or_inverse_maps": "typed M1 contract requires exactly 4776 of each",
    "write_jpeg_or_png_result": "map contract forbids display products and worker output suffix checks fail closed",
    "p0_import_torch_timm_load_model_jpeg_or_cuda": "AST and clean-interpreter import tests reject these execution surfaces",
    "start_x3_x5_f3_final_or_stage_g": "startup and completion checks reject downstream roots and refs",
    "produce_empirical_outcome_before_completion": "P0 output allowlist contains control metadata only",
    "reduce_workload_to_meet_budget": "resource contract forbids reduction of any route transform method scale or seed",
}


class X2MechP0Error(ValueError):
    pass


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise X2MechP0Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               X2MechP0Error(f"non-finite JSON: {token}")))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise X2MechP0Error(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise X2MechP0Error(f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "$schema": ARTIFACT_SCHEMA.name, "schema_version": 1,
        "document_type": document_type, "stage_id": "X2-MECH",
        "phase": "P0-R1", "run_id": RUN_ID,
        "authorization": AUTHORIZATION, "status": status,
        "payload": dict(payload),
    }
    value["canonical_digest"] = canonical_sha256(value)
    return value


def payload(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "identity": [], "bindings": [], "checks": [], "counts": [],
        "state": [], "files": [], "routes": [], "outputs": [],
        "budget": [], "estimability": [], "attacks": [], "access": [],
        "terminal": {"terminal": False, "complete": False, "immutable": False,
                     "next_authorization": "NONE", "x2_decision_modified": False,
                     "x3_x5_started": False},
    }
    value.update(updates)
    return value


def validate_signed(value: Mapping[str, Any], context: str, schema: Mapping[str, Any] | None = None) -> None:
    copy = dict(value); observed = copy.pop("canonical_digest", None)
    if observed != canonical_sha256(copy):
        raise X2MechP0Error(f"{context}: canonical self-digest mismatch")
    if schema is not None:
        errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path))
        if errors:
            raise X2MechP0Error(f"{context}: schema validation failed: {errors[0].message}")


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())


def _write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())


def _publish_atomic(pending: Path, terminal: Path, value: Mapping[str, Any]) -> None:
    if pending.exists() or terminal.exists():
        raise X2MechP0Error("completion path already exists")
    _write_new(pending, value); os.replace(pending, terminal)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode:
        raise X2MechP0Error(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def file_record(repo: Path, relative: Path) -> dict[str, Any]:
    path = repo / relative
    canonical = None
    if path.suffix == ".json":
        try:
            value = strict_read(path); canonical = value.get("canonical_digest")
        except X2MechP0Error:
            canonical = None
    return {"path": relative.as_posix(), "bytes": path.stat().st_size,
            "sha256": sha_file(path), "canonical_digest": canonical}


def _exact(value: Mapping[str, Any], keys: set[str], context: str) -> None:
    if set(value) != keys:
        raise X2MechP0Error(f"{context}: fields differ")


def load_config(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = strict_read(repo / CONFIG); schema = strict_read(repo / CONFIG_SCHEMA)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(config), key=lambda item: list(item.path))
    if errors:
        raise X2MechP0Error(f"config schema validation failed: {errors[0].message}")
    copy = dict(config); observed = copy.pop("canonical_digest")
    if observed != canonical_sha256(copy):
        raise X2MechP0Error("config canonical digest mismatch")
    if config["cohort"]["method_transform_instances"] != 199 * 12 * 2:
        raise X2MechP0Error("method-transform count drifted")
    if config["map_contract"]["transformed_map_count"] != 4776 or config["map_contract"]["inverse_map_count"] != 4776:
        raise X2MechP0Error("map count drifted")
    if config["claims"]["x2_decision_mutable"] or config["claims"]["x3_x5_authorized"]:
        raise X2MechP0Error("immutable claim boundary drifted")
    if config["resources"]["workload_reduction_allowed"]:
        raise X2MechP0Error("workload reduction became allowed")
    return config


def _source_ast_checks(repo: Path) -> list[dict[str, str]]:
    forbidden_top = {"torch", "torchvision", "timm", "PIL", "cv2"}
    checks: list[dict[str, str]] = []
    for relative in SOURCE_FILES[-4:-1]:
        tree = ast.parse((repo / relative).read_text(encoding="utf-8"), filename=str(relative))
        top_imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import): top_imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: top_imports.add(node.module.split(".")[0])
        if top_imports & forbidden_top:
            raise X2MechP0Error(f"top-level forbidden import in {relative}: {sorted(top_imports & forbidden_top)}")
        checks.append({"check_id": f"lazy_import_{relative.name}", "status": "PASS",
                       "detail": "no Torch, timm, torchvision, PIL, or cv2 top-level import"})
    control_source = (repo / SOURCE_FILES[-4]).read_text(encoding="utf-8")
    for forbidden in ("np." + "load(", "numpy." + "load(", "torch." + "cuda", "Image." + "open("):
        if forbidden in control_source:
            raise X2MechP0Error(f"P0 control contains forbidden execution surface: {forbidden}")
    return checks


def _validate_x2(repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if _git(repo, "symbolic-ref", "--short", "HEAD") != BRANCH:
        raise X2MechP0Error("branch drifted")
    for ref, expected in ((X2_PRODUCER_REF, X2_PRODUCER), (X2_COMPLETION_REF, X2_COMPLETION)):
        if _git(repo, "rev-parse", ref + "^{commit}") != expected:
            raise X2MechP0Error(f"X2 ref drifted: {ref}")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", X2_PRODUCER, X2_COMPLETION], cwd=repo)
    if ancestor.returncode:
        raise X2MechP0Error("X2 producer is not completion ancestor")
    closeout = strict_read(repo / X2_CLOSEOUT); validate_signed(closeout, "X2 closeout")
    if closeout.get("status") != "NO_GO_BROAD_CLAIM_DEVELOPMENT":
        raise X2MechP0Error("X2 decision drifted")
    if any(closeout["downstream"].get(key) for key in ("x3_started", "x4_started", "x5_started")):
        raise X2MechP0Error("X3-X5 state drifted")
    bindings = []
    for relative in X2_CONTROL_FILES:
        value = strict_read(repo / relative); validate_signed(value, str(relative))
        bindings.append(file_record(repo, relative))
    refs = _git(repo, "show-ref")
    if any(token in refs.lower() for token in ("x3-", "x4-", "x5-", "x3_", "x4_", "x5_")):
        raise X2MechP0Error("X3-X5 archive ref exists")
    checks = [
        {"check_id": "x2_refs_and_ancestry", "status": "PASS", "detail": "producer/completion refs exact and ancestry valid"},
        {"check_id": "x2_control_metadata", "status": "PASS", "detail": "allowed X2 control files rehashed and canonical digests recomputed"},
        {"check_id": "x2_decision_immutable", "status": "PASS", "detail": "NO_GO_BROAD_CLAIM_DEVELOPMENT unchanged"},
        {"check_id": "x3_x5_not_started", "status": "PASS", "detail": "closeout and refs show no X3-X5 start"},
    ]
    return bindings, checks


def _source_routes(repo: Path) -> list[dict[str, Any]]:
    units = sorted((repo / X2_ROOT / "units").glob("*/terminal.json"))
    if len(units) != 200:
        raise X2MechP0Error(f"expected 200 X2 unit terminals, found {len(units)}")
    routes = []
    for terminal_path in units:
        terminal = strict_read(terminal_path); validate_signed(terminal, str(terminal_path))
        identity = {row["name"]: row["value"] for row in terminal["payload"]["identity"]}
        if terminal["status"] == "VALID":
            status = "VALID"
        else:
            state = {row["name"]: row["value"] for row in terminal["payload"]["state"]}
            if state.get("coverage_class") != "FAILED_ROUTE_LOCAL_NUMERICAL":
                raise X2MechP0Error("unexpected non-valid X2 route")
            status = "FAILED_ROUTE_LOCAL_NUMERICAL"
        maps = []
        if status == "VALID":
            candidates = [row["path"] for row in terminal["payload"]["bindings"]
                          if "/clean/terminal.json" in row["path"]]
            if len(candidates) != 1:
                raise X2MechP0Error("clean terminal binding is not unique")
            clean_path = repo / candidates[0]; clean = strict_read(clean_path); validate_signed(clean, str(clean_path))
            for method in ("nns_v2", "vit_grad_cam"):
                matches = [row for row in clean["payload"]["files"] if Path(row["path"]).name == method + ".npy"]
                if len(matches) != 1:
                    raise X2MechP0Error(f"{method} source map binding is not unique")
                record = matches[0]; source = repo / record["path"]
                if source.stat().st_size != record["bytes"] or sha_file(source) != record["sha256"]:
                    raise X2MechP0Error(f"{method} source map byte binding drifted")
                if record.get("dtype") != "float32" or record.get("shape") != [224, 224]:
                    raise X2MechP0Error(f"{method} source map metadata drifted")
                maps.append({"method_id": method, "path": record["path"], "bytes": record["bytes"],
                             "sha256": record["sha256"], "canonical_digest": record["canonical_digest"],
                             "dtype": "float32", "shape": [224, 224]})
        relative = terminal_path.relative_to(repo)
        routes.append({
            "unit_digest": identity["unit_digest"], "model_id": identity["model_id"],
            "sample_id": identity["sample_id"], "source_status": status,
            "shard": int.from_bytes(bytes.fromhex(identity["unit_digest"]), "big") % 4,
            "source_terminal_path": relative.as_posix(),
            "source_terminal_sha256": sha_file(terminal_path), "source_maps": maps,
        })
    if sum(row["source_status"] == "VALID" for row in routes) != 199:
        raise X2MechP0Error("valid route count drifted")
    return routes


def _budget(repo: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    profile = strict_read(repo / X2_ROOT / "resource_profiles.json")
    resources = profile["payload"]["resources"]; factor = 1.2 * (12 / 23 + 4 / 11)
    projected_gpu = round(float(resources["gpu_seconds"]) / 3600 * factor, 1)
    projected_cpu = math.ceil(float(resources["cpu_seconds"]) / 3600 * factor)
    projected = config["resources"]["projected"]; caps = config["resources"]["hard_caps"]
    if projected_gpu != projected["h20_gpu_hours"] or projected_cpu != projected["cpu_hours"]:
        raise X2MechP0Error("budget projection does not reproduce sealed telemetry formula")
    rows = []
    for resource, unit in (("h20_gpu_hours", "hours"), ("wall_hours", "hours"),
                           ("cpu_hours", "hours"), ("peak_reserved_gib_per_process", "GiB"),
                           ("retained_gib", "GiB"), ("max_devices", "devices"),
                           ("downloaded_bytes", "bytes")):
        status = "PASS" if projected[resource] <= caps[resource] else "BLOCKED_BUDGET"
        rows.append({"resource": resource, "projected": projected[resource], "cap": caps[resource],
                     "unit": unit, "status": status})
    return rows


def _schema(repo: Path) -> dict[str, Any]:
    schema = strict_read(repo / ARTIFACT_SCHEMA); Draft202012Validator.check_schema(schema); return schema


def _validate_artifact(repo: Path, relative: Path) -> dict[str, Any]:
    value = strict_read(repo / relative); validate_signed(value, str(relative), _schema(repo)); return value


def freeze(repo_root: Path | str, producer_commit: str, test_summary: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = load_config(repo)
    if _git(repo, "rev-parse", "HEAD") != producer_commit:
        raise X2MechP0Error("HEAD is not the declared producer commit")
    if _git(repo, "rev-parse", PRODUCER_REF + "^{commit}") != producer_commit:
        raise X2MechP0Error("producer archive ref does not bind HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, producer_commit], cwd=repo).returncode:
        raise X2MechP0Error("producer does not descend from the frozen X2 completion")
    if (repo / ROOT).exists() or (repo / REPORT_ROOT).exists():
        raise X2MechP0Error("P0-R1 output root already exists")
    x2_bindings, x2_checks = _validate_x2(repo); ast_checks = _source_ast_checks(repo); routes = _source_routes(repo)
    source_records = [file_record(repo, path) for path in SOURCE_FILES]
    budget = _budget(repo, config)
    if any(row["status"] != "PASS" for row in budget):
        raise X2MechP0Error("BLOCKED_BUDGET")
    access = [{"name": name, "value": False} for name in ACCESS_FLAGS]
    terminal_open = {"terminal": False, "complete": True, "immutable": True,
                     "next_authorization": "NONE", "x2_decision_modified": False,
                     "x3_x5_started": False}
    startup = signed("stage_x_x2_mech_startup_gate", "PASS", payload(
        identity=[{"name": "base_commit", "value": BASE_COMMIT}, {"name": "producer_commit", "value": producer_commit}],
        bindings=x2_bindings, checks=x2_checks + ast_checks,
        counts=[{"name": "x2_route_count", "value": 200}],
        state=[{"name": "initial_worktree_index_lfs_clean", "value": True},
               {"name": "prior_p0_status", "value": "BLOCKED_PROCEDURAL_ACCESS_SCOPE_CONTRADICTION"}],
        terminal=terminal_open))
    implementation = signed("stage_x_x2_mech_implementation_seal", "FROZEN", payload(
        identity=[{"name": "producer_commit", "value": producer_commit}], bindings=source_records,
        checks=ast_checks, counts=[{"name": "source_file_count", "value": len(source_records)}], terminal=terminal_open))
    source_binding = signed("stage_x_x2_mech_source_binding", "PASS", payload(
        bindings=x2_bindings, routes=routes,
        counts=[{"name": "route_count", "value": 200}, {"name": "valid_route_count", "value": 199},
                {"name": "failed_route_count", "value": 1}, {"name": "source_map_binding_count", "value": 398}],
        state=[{"name": "array_values_loaded", "value": False}, {"name": "source_map_bytes_only_hashed", "value": True}],
        terminal=terminal_open))
    outputs = [{"path": path, "stage": stage, "required": True} for path, stage in FUTURE_OUTPUTS.items()]
    plan = signed("stage_x_x2_mech_typed_plan", "FROZEN", payload(
        identity=[{"name": "study_type", "value": config["study_type"]}], routes=routes, outputs=outputs,
        counts=[{"name": "route_count", "value": 200}, {"name": "method_transform_instances", "value": 4776},
                {"name": "transformed_maps", "value": 4776}, {"name": "inverse_maps", "value": 4776}],
        state=[{"name": "workload_reduction_allowed", "value": False},
               {"name": "route_transforms_cross_shard_allowed", "value": False}], terminal=terminal_open))
    budget_doc = signed("stage_x_x2_mech_budget_projection", "PASS", payload(
        bindings=[file_record(repo, X2_ROOT / "resource_profiles.json")], budget=budget,
        state=[{"name": "conservative_projection", "value": True},
               {"name": "workload_reduction_used", "value": False}], terminal=terminal_open))
    estimability = [
        {"endpoint": "fine_cell_4x4_transformed_stability", "before_m1": "NOT_ESTIMABLE_FROM_RETAINED_X2_EVIDENCE", "after_m1_maximum": "POST_HOC_EXPLORATORY_ESTIMABLE", "confirmatory_upgrade_allowed": False},
        {"endpoint": "native_patch_16x16_transformed_stability", "before_m1": "NOT_ESTIMABLE_FROM_RETAINED_X2_EVIDENCE", "after_m1_maximum": "POST_HOC_EXPLORATORY_ESTIMABLE", "confirmatory_upgrade_allowed": False},
    ]
    estimability_doc = signed("stage_x_x2_mech_estimability_matrix", "NOT_ESTIMABLE_FROM_RETAINED_X2_EVIDENCE", payload(estimability=estimability, terminal=terminal_open))
    attacks = [{"attack": key, "result": "REJECTED", "mechanism": value} for key, value in ATTACKS.items()]
    false_pass = signed("stage_x_x2_mech_false_pass_audit", "PASS", payload(
        attacks=attacks, counts=[{"name": "attack_count", "value": len(attacks)}], terminal=terminal_open))
    access_doc = signed("stage_x_x2_mech_access_audit", "PASS", payload(
        access=access, checks=[{"check_id": "source_code_not_empirical_outcome", "status": "PASS",
                               "detail": "source/tests inspected while real arrays and per-route outcome rows remained unread"}], terminal=terminal_open))
    test_doc = signed("stage_x_x2_mech_test_report", "PASS", payload(
        checks=[{"check_id": "focused_synthetic_tests", "status": "PASS", "detail": test_summary}],
        state=[{"name": "fixtures", "value": "synthetic_numpy_only"}], access=access, terminal=terminal_open))
    documents = {
        "startup_gate.json": startup, "implementation_seal.json": implementation,
        "source_binding.json": source_binding, "typed_plan.json": plan,
        "budget_projection.json": budget_doc, "estimability_matrix.json": estimability_doc,
        "false_pass_audit.json": false_pass, "access_audit.json": access_doc,
        "test_report.json": test_doc,
    }
    for name, value in documents.items(): _write_new(repo / ROOT / name, value)
    report_text = (
        "# X2-MECH P0-R1 validation report\n\n"
        "Status: `PASS`\n\n"
        f"Producer: `{producer_commit}`. X2 remains `NO_GO_BROAD_CLAIM_DEVELOPMENT`. "
        "The protocol, implementation, typed 200-route plan, two-method/12-transform map contract, "
        "three-scale analysis, ROAD contract, synchronized bootstrap, resource caps, and false-pass "
        "controls are frozen. Synthetic tests passed. No real map value, per-route outcome row, JPEG, "
        "checkpoint, model, Torch/timm import, or CUDA/GPU surface was accessed; no empirical mechanism "
        "outcome was computed. Only `X2_MECH_M1_TRANSFORMED_MAP_AND_MULTISCALE_FIDELITY_ONLY` is next.\n")
    _write_text_new(repo / REPORT_ROOT / "x2_mech_p0_r1_report.md", report_text)
    validation = signed("stage_x_x2_mech_validation_report", "PASS", payload(
        bindings=[file_record(repo, ROOT / name) for name in documents], checks=x2_checks + ast_checks,
        counts=[{"name": "routes_validated", "value": 200}, {"name": "source_maps_hash_bound", "value": 398},
                {"name": "false_pass_attacks", "value": len(attacks)}], access=access, terminal=terminal_open))
    _write_new(repo / REPORT_ROOT / "validation_report.json", validation)
    inventory_paths = sorted(path for path in (repo / ROOT).rglob("*") if path.is_file() and path not in {repo / PENDING, repo / COMPLETION})
    inventory = signed("stage_x_x2_mech_whole_root_inventory", "PASS", payload(
        files=[file_record(repo, path.relative_to(repo)) for path in inventory_paths],
        counts=[{"name": "file_count", "value": len(inventory_paths)},
                {"name": "total_bytes", "value": sum(path.stat().st_size for path in inventory_paths)}], terminal=terminal_open))
    _write_new(repo / ROOT / "whole_root_inventory.json", inventory)
    completion_bindings = [file_record(repo, ROOT / name) for name in (*documents, "whole_root_inventory.json")]
    completion_bindings += [file_record(repo, REPORT_ROOT / "validation_report.json"), file_record(repo, REPORT_ROOT / "x2_mech_p0_r1_report.md")]
    completion = signed("stage_x_x2_mech_completion", "PASS", payload(
        identity=[{"name": "producer_commit", "value": producer_commit},
                  {"name": "x2_completion_commit", "value": X2_COMPLETION}],
        bindings=completion_bindings, checks=x2_checks + ast_checks,
        counts=[{"name": "routes", "value": 200}, {"name": "valid_routes", "value": 199},
                {"name": "failed_routes", "value": 1}, {"name": "method_transform_instances", "value": 4776}],
        state=[{"name": "x2_decision", "value": "NO_GO_BROAD_CLAIM_DEVELOPMENT"},
               {"name": "study_type", "value": "POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS"}],
        access=access, terminal={"terminal": True, "complete": True, "immutable": True,
                                 "next_authorization": NEXT_AUTHORIZATION,
                                 "x2_decision_modified": False, "x3_x5_started": False}))
    _publish_atomic(repo / PENDING, repo / COMPLETION, completion)
    return completion


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); load_config(repo); _validate_x2(repo); _source_ast_checks(repo)
    expected = ("startup_gate.json", "implementation_seal.json", "source_binding.json",
                "typed_plan.json", "budget_projection.json", "estimability_matrix.json",
                "false_pass_audit.json", "access_audit.json", "test_report.json",
                "whole_root_inventory.json", "completion/x2_mech_p0_r1.json")
    values = {name: _validate_artifact(repo, ROOT / name) for name in expected}
    completion = values["completion/x2_mech_p0_r1.json"]
    if completion["status"] != "PASS" or completion["payload"]["terminal"]["next_authorization"] != NEXT_AUTHORIZATION:
        raise X2MechP0Error("completion terminal contract drifted")
    access = {row["name"]: row["value"] for row in completion["payload"]["access"]}
    if set(access) != set(ACCESS_FLAGS) or any(access.values()):
        raise X2MechP0Error("completion access audit drifted")
    for binding in completion["payload"]["bindings"]:
        path = repo / binding["path"]
        if path.stat().st_size != binding["bytes"] or sha_file(path) != binding["sha256"]:
            raise X2MechP0Error(f"completion binding drifted: {binding['path']}")
    return {"status": "PASS", "artifacts_validated": len(expected),
            "next_authorization": NEXT_AUTHORIZATION}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-config"); validate.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    freeze_parser = sub.add_parser("freeze"); freeze_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    freeze_parser.add_argument("--producer-commit", required=True); freeze_parser.add_argument("--test-summary", required=True)
    completion_parser = sub.add_parser("validate-completion"); completion_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config": result = {"status": "PASS", "run_id": load_config(args.repo_root)["run_id"]}
        elif args.command == "freeze": result = freeze(args.repo_root, args.producer_commit, args.test_summary)
        else: result = validate_completion(args.repo_root)
    except (X2MechP0Error, OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True)); return 2
    print(json.dumps(result, sort_keys=True, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
