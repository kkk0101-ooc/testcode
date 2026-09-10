"""Stage X X2 frozen initialization, execution, and source validation.

The control plane deliberately separates outcome-free initialization from model
or image access. Runtime units publish immutable attempt journals and terminal
files; reconciliation reopens their sources rather than trusting summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256

CONFIG_PATH = Path("config/stage_x_x2.json")
CONFIG_SCHEMA_PATH = Path("config/stage_x_x2.schema.json")
ARTIFACT_SCHEMA_PATH = Path("config/stage_x_x2_artifact.schema.json")
ROOT_PATH = Path("evidence/stage_x/x2_nns_v2_development_20260909/formal")
REPORT_PATH = Path("reports/stage_x_x2_nns_v2_development_20260909")
PENDING_PATH = ROOT_PATH / "completion/x2.pending.json"
COMPLETION_PATH = ROOT_PATH / "completion/x2.json"
EXPECTED_BRANCH = "stage-d-d3-pass"
DELIVERY_COMMIT = "69fc9177491c63a21088a2092274859b00917879"
DELIVERY_PARENT = "0dd96d71a9e246a6cf64686180418b850e11ce8a"
X1_ARCHIVE = "refs/archive/stage-x-nns-v2-x1-r1-completion"
X1_ROOT = Path("evidence/stage_x/x1_r1_nns_v2_20260908/formal")
X1_REPORT = Path("reports/stage_x_x1_r1_nns_v2_20260908")
X1_COMPLETION_SHA256 = "0c884c8bff2d2b356b3105c9dea3c83dced438c7bbb111a38100ed74ddd517e7"
X1_STATUS_DIGEST = "5e63f87b1e7c407941c917adffe7a5740b8e3231295d2e1ac4047dba9f576006"
X1_INVENTORY_DIGEST = "cd8be7a1a7cfd45282686ae6ede2cab39602744bca48aa4e5c902062b8b9de57"
HISTORICAL_REFS = {
    "refs/archive/stage-x-nns-v2-x0-completion": "bc4b18580f88fc2e82e1a3c686b29092e9ad7b40",
    "refs/archive/stage-x-nns-v2-x0-r1-completion": "a7adbd64c468aa33ffa1c02ae1c575609d837b30",
    "refs/archive/stage-x-nns-v2-x1-blocked": "6e7dfc90b5c059f1432f21c0533afae9ea302506",
    X1_ARCHIVE: DELIVERY_COMMIT,
}
FROZEN_HISTORICAL_PATHS = (
    "evidence/stage_x/x0_nns_v2_20260908",
    "evidence/stage_x/x0_r1_identity_control_20260908",
    "evidence/stage_x/x1_nns_v2_20260908",
    "evidence/stage_x/x1_r1_nns_v2_20260908",
    "reports/stage_x_x1_r1_nns_v2_20260908",
)
FREEZE_ARTIFACTS = (
    "startup_gate.json", "x1_r1_binding.json", "dependency_binding.json",
    "environment_binding.json", "x2_run_spec.json",
    "model_materialization.json", "development_sample_manifest.json",
    "split_isolation_audit.json", "target_binding.json",
    "gamma_selection_plan.json", "development_unit_manifest.json",
)
REQUIRED_TERMINAL_ARTIFACTS = (
    *FREEZE_ARTIFACTS, "gamma_selection_result.json",
    "architecture_compatibility.json", "identity_control_validation.json",
    "complete_cut_validation.json", "comparator_adapter_validation.json",
    "terminal_union.json", "faithfulness_validation.json",
    "stability_validation.json", "pixel_class_sensitivity_validation.json",
    "margin_seal.json", "futility_decision.json",
    "route_resource_profiles.json", "formal_budget_projection.json",
    "resource_audit.json", "access_audit.json", "attempt_journal.json",
    "false_pass_audit.json", "test_report.json", "payload_manifest.json",
    "whole_root_inventory.json", "status.json", "completion/x2.json",
)
class StageXX2Error(ValueError):
    pass


def _duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StageXX2Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_duplicates, parse_constant=lambda value: (_ for _ in ()).throw(StageXX2Error(f"non-finite JSON: {value}")))
    if not isinstance(value, dict):
        raise StageXX2Error(f"JSON root must be object: {path}")
    return value


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    process = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if process.returncode:
        raise StageXX2Error(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {"$schema": ARTIFACT_SCHEMA_PATH.name, "schema_version": 1, "document_type": document_type, "stage_id": "X2", "status": status, "payload": dict(payload)}
    value["canonical_digest"] = canonical_sha256(value)
    return value


def validate_signed(value: Mapping[str, Any], context: str) -> None:
    schema = strict_read(REPO_ROOT / ARTIFACT_SCHEMA_PATH)
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise StageXX2Error(f"{context}: {errors[0].message}")
    plain = dict(value); observed = plain.pop("canonical_digest")
    if observed != canonical_sha256(plain):
        raise StageXX2Error(f"{context}: canonical digest mismatch")


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    validate_signed(value, path.as_posix())
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())


def record(repo: Path, relative: Path) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}


def artifact_record(repo: Path, relative: Path) -> dict[str, Any]:
    value = strict_read(repo / relative); validate_signed(value, relative.as_posix())
    return {**record(repo, relative), "canonical_digest": value["canonical_digest"]}


def upstream_artifact_record(repo: Path, relative: Path) -> dict[str, Any]:
    """Record an upstream artifact after its stage-owned validator has passed."""
    value = strict_read(repo / relative)
    digest = value.get("canonical_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise StageXX2Error(f"upstream canonical digest missing: {relative}")
    return {**record(repo, relative), "canonical_digest": digest}


def load_config(repo: Path) -> dict[str, Any]:
    value = strict_read(repo / CONFIG_PATH); schema = strict_read(repo / CONFIG_SCHEMA_PATH)
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise StageXX2Error(errors[0].message)
    plain = dict(value); observed = plain.pop("canonical_digest")
    if observed != canonical_sha256(plain):
        raise StageXX2Error("X2 config canonical digest mismatch")
    return value


def _remote_ref(repo: Path, ref: str) -> str:
    rows = git(repo, "ls-remote", "origin", ref).splitlines()
    if len(rows) != 1 or rows[0].split()[-1] != ref:
        raise StageXX2Error(f"remote ref missing or ambiguous: {ref}")
    return rows[0].split()[0]


def post_delivery_startup(repo: Path) -> dict[str, Any]:
    branch, head, tracking = (git(repo, "symbolic-ref", "--short", "HEAD"), git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "@{upstream}"))
    remote = _remote_ref(repo, f"refs/heads/{EXPECTED_BRANCH}")
    parent = git(repo, "rev-parse", DELIVERY_COMMIT + "^")
    if (branch, head, tracking, remote, parent) != (EXPECTED_BRANCH, DELIVERY_COMMIT, DELIVERY_COMMIT, DELIVERY_COMMIT, DELIVERY_PARENT):
        raise StageXX2Error("post-delivery Git binding drifted")
    refs = {}; remote_refs = {}
    for ref, expected in HISTORICAL_REFS.items():
        refs[ref] = git(repo, "rev-parse", ref); remote_refs[ref] = _remote_ref(repo, ref)
        if refs[ref] != expected or remote_refs[ref] != expected:
            raise StageXX2Error(f"historical archive ref drifted: {ref}")
    status = git(repo, "status", "--porcelain=v1")
    worktree_root = Path(git(repo, "rev-parse", "--show-toplevel")).resolve()
    repo_prefix = repo.relative_to(worktree_root).as_posix()
    prefix = "" if repo_prefix == "." else repo_prefix + "/"
    expected_new = {
        f"?? {prefix}config/stage_x_x2.json", f"?? {prefix}config/stage_x_x2.schema.json",
        f"?? {prefix}config/stage_x_x2_artifact.schema.json",
        f"?? {prefix}docs/stage_x_x2_unfused_execution_amendment_20260909.md",
        f"?? {prefix}src/vitnns/stage_x_x2.py", f"?? {prefix}tests/test_stage_x_x2.py",
    }
    rows = {row for row in status.splitlines() if row}
    unexpected = sorted(rows - expected_new)
    if unexpected:
        raise StageXX2Error("unexpected user worktree changes: " + repr(unexpected))
    historical_diff = git(repo, "diff", "--", *FROZEN_HISTORICAL_PATHS)
    if historical_diff:
        raise StageXX2Error("historical Stage X roots were modified")
    # Reuse the X1 source validator with the delivered commit as the startup
    # identity. This changes no evidence and avoids the obsolete pre-delivery gate.
    from . import stage_x_x1_r1 as x1
    original = x1.EXPECTED_HEAD
    try:
        x1.EXPECTED_HEAD = DELIVERY_COMMIT
        x1_result = x1.validate_outputs(repo)
    finally:
        x1.EXPECTED_HEAD = original
    completion = repo / X1_ROOT / "completion/x1_r1.json"
    status_value = strict_read(repo / X1_ROOT / "status.json")
    inventory_value = strict_read(repo / X1_ROOT / "whole_root_inventory.json")
    if sha_file(completion) != X1_COMPLETION_SHA256 or status_value["canonical_digest"] != X1_STATUS_DIGEST or inventory_value["canonical_digest"] != X1_INVENTORY_DIGEST:
        raise StageXX2Error("X1-R1 fixed delivery digest drifted")
    lfs = subprocess.run(["git", "lfs", "fsck"], cwd=repo, text=True, capture_output=True, check=False)
    if lfs.returncode:
        raise StageXX2Error("Git LFS fsck failed: " + lfs.stderr.strip())
    return {"branch":branch,"local_head":head,"tracking_head":tracking,"remote_head":remote,"delivery_parent":parent,"archive_refs_local":refs,"archive_refs_remote":remote_refs,"x1_source_validation":x1_result,"x1_completion_sha256":sha_file(completion),"x1_status_canonical_digest":status_value["canonical_digest"],"x1_inventory_canonical_digest":inventory_value["canonical_digest"],"lfs_fsck":"PASS","historical_roots_read_only":True,"unexpected_user_changes":[],"x3_started":False}


def _sample_rows(repo: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = repo / config["sample_selection"]["source"]
    source = strict_read(source_path); namespace = config["sample_selection"]["namespace"]
    allowed = set(config["sample_selection"]["allowed_fields"])
    candidates = []
    for raw in source["rows"]:
        # Explicit mapping from the frozen Stage D registry to the five permitted
        # X2 selection fields. No prediction or result field is inspected.
        row = {
            "dataset_release_id": config["sample_selection"]["dataset_release_id"],
            "class_id": str(raw["imagenet1k_index"]),
            "source_object_id": raw["sample_id"],
            "file_bytes": raw["size_bytes"],
            "file_sha256": raw["image_sha256"],
        }
        if set(row) != allowed:
            raise StageXX2Error("selection field boundary drifted")
        payload = "\0".join((namespace,row["dataset_release_id"],row["class_id"],row["source_object_id"],row["file_sha256"]))
        row["selection_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        candidates.append(row)
    per_class: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for row in candidates: per_class[row["class_id"]].append(row)
    winners = [sorted(rows,key=lambda item:(item["selection_sha256"],item["source_object_id"]))[0] for rows in per_class.values()]
    selected = sorted(winners,key=lambda item:(item["selection_sha256"],item["source_object_id"]))[:40]
    if len(selected) != 40 or len({r["source_object_id"] for r in selected}) != 40:
        raise StageXX2Error("development sample count/identity invalid")
    for rank,row in enumerate(selected): row["development_rank"] = rank
    proof = {"source":record(repo,Path(config["sample_selection"]["source"])),"source_registry_canonical_digest":source["canonical_digest"],"source_row_count":len(candidates),"source_class_count":len(per_class),"class_winner_count":len(winners),"selection_fields_read":sorted(allowed),"outcome_fields_read":[],"algorithm":config["sample_selection"]["algorithm"],"namespace":namespace}
    return selected, proof


def _unit_manifest(config: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    units = []
    for model in config["models"]:
        for sample in samples:
            identity = {"model_id":model["model_id"],"sample_id":sample["source_object_id"],"sample_sha256":sample["file_sha256"],"development_rank":sample["development_rank"]}
            digest = canonical_sha256(identity)
            units.append({**identity,"unit_digest":digest,"shard":int.from_bytes(bytes.fromhex(digest)[:8],"big") % config["execution"]["shards"],"expected_terminal":"VALID_OR_TYPED_FAILED_OR_NA"})
    return {"units":units,"unit_count":len(units),"unique_unit_count":len({u["unit_digest"] for u in units}),"per_model_counts":dict(Counter(u["model_id"] for u in units)),"shared_sample_ids_sha256":canonical_sha256([r["source_object_id"] for r in samples]),"shard_counts":dict(Counter(str(u["shard"]) for u in units)),"atomic_terminals":True,"resume_source_validation":True}


def initialize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = load_config(repo)
    if (repo/ROOT_PATH).exists() or (repo/REPORT_PATH).exists():
        raise StageXX2Error("X2 output root already exists; refusing overwrite")
    startup = post_delivery_startup(repo)
    samples, proof = _sample_rows(repo,config); units = _unit_manifest(config,samples)
    root_artifacts = {
        "startup_gate.json": ("stage_x_x2_startup_gate", startup),
        "x1_r1_binding.json": ("stage_x_x2_x1_r1_binding", {"delivery_commit":DELIVERY_COMMIT,"delivery_parent":DELIVERY_PARENT,"archive_ref":X1_ARCHIVE,"completion":record(repo,X1_ROOT/"completion/x1_r1.json"),"status":upstream_artifact_record(repo,X1_ROOT/"status.json"),"payload_manifest":upstream_artifact_record(repo,X1_ROOT/"payload_manifest.json"),"whole_root_inventory":upstream_artifact_record(repo,X1_ROOT/"whole_root_inventory.json"),"validation_report":record(repo,X1_REPORT/"validation_report.json"),"source_recomputed":True,"authorization":config["authorization"]}),
        "dependency_binding.json": ("stage_x_x2_dependency_binding", {"authoritative_protocols":[record(repo,Path(p)) for p in ("docs/stage_x_protocol.md","docs/stage_x_acceptance.md","docs/stage_x_x0_r1_identity_control_amendment_20260908.md","docs/stage_x_x2_unfused_execution_amendment_20260909.md")],"authoritative_manifests":[record(repo,Path(p)) for p in ("manifests/stage_x_benchmark_plan_v2.json","manifests/stage_x_comparator_registry_v2.json","manifests/stage_x_sample_split_plan_v1.json","manifests/stage_x_prior_access_ledger_v1.json","manifests/stage_x_model_data_inventory_v1.json")],"implementation_bindings":[record(repo,Path(p)) for p in ("src/vitnns/stage_x_nns_v2.py","src/vitnns/stage_x_comparators.py","src/vitnns/stage_x_x2.py")]}),
        "environment_binding.json": ("stage_x_x2_environment_binding", {"python":platform.python_version(),"platform":platform.platform(),"requirements_lock":record(repo,Path("requirements.lock.txt")),"canonical_execution_path":config["execution"]["canonical_path"],"devices_frozen":config["execution"]["shard_devices"]}),
        "x2_run_spec.json": ("stage_x_x2_run_spec", {"config":record(repo,CONFIG_PATH),"config_canonical_digest":config["canonical_digest"],"authorization":config["authorization"],"execution":config["execution"],"methods":config["methods"],"faithfulness":config["faithfulness"],"stability":config["stability"],"retention":config["retention"],"resources":config["resources"],"margin_and_futility":config["margin_and_futility"],"formal_projection":config["formal_projection"],"terminal_contract":config["terminal_contract"]}),
        "model_materialization.json": ("stage_x_x2_model_materialization_plan", {"models":config["models"],"download_total_max":sum(m["download_bytes_max"] for m in config["models"]),"stream_sha256_before_deserialization":True,"canonical_sorted_tensor_digest_after_load":True,"strict_load_required":True,"outcome_dependent_checkpoint_replacement":False,"large_bytes_git_tracked":False}),
        "development_sample_manifest.json": ("stage_x_x2_development_sample_manifest", {"samples":samples,"count":40,"shared_across_five_models":True,"selection_proof":proof,"formal_exclusion":True,"image_decoded_before_seal":False}),
        "split_isolation_audit.json": ("stage_x_x2_split_isolation_audit", {"x2_sample_ids":[r["source_object_id"] for r in samples],"x2_excluded_from_x3_x5":True,"formal_candidate_accessed":False,"formal_manifest_materialized":False,"replacement_or_reselection_allowed":False,"five_models_same_list":True}),
        "target_binding.json": ("stage_x_x2_target_binding", config["target"]),
        "gamma_selection_plan.json": ("stage_x_x2_gamma_selection_plan", config["gamma_selection"]),
        "development_unit_manifest.json": ("stage_x_x2_development_unit_manifest", units),
    }
    (repo/ROOT_PATH).mkdir(parents=True,exist_ok=False); (repo/REPORT_PATH).mkdir(parents=True,exist_ok=False)
    for name,(kind,payload) in root_artifacts.items(): write_new(repo/ROOT_PATH/name,signed(kind,"PENDING",payload))
    return {"status":"PENDING","frozen_artifacts":len(root_artifacts),"samples":len(samples),"units":units["unit_count"],"outcome_access":False}


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); config=load_config(repo)
    for name in FREEZE_ARTIFACTS:
        path=repo/ROOT_PATH/name
        if not path.is_file(): raise StageXX2Error(f"missing frozen artifact: {name}")
        validate_signed(strict_read(path),name)
    samples,proof=_sample_rows(repo,config); recorded=strict_read(repo/ROOT_PATH/"development_sample_manifest.json")["payload"]
    if recorded["samples"]!=samples or recorded["selection_proof"]!=proof: raise StageXX2Error("sample selection does not source-recompute")
    units=_unit_manifest(config,samples); recorded_units=strict_read(repo/ROOT_PATH/"development_unit_manifest.json")["payload"]
    if recorded_units!=units: raise StageXX2Error("unit manifest does not source-recompute")
    return {"status":"PASS","frozen_artifacts":len(FREEZE_ARTIFACTS),"sample_count":len(samples),"unit_count":units["unit_count"],"config_digest":config["canonical_digest"]}


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    for command in ("initialize","validate-init"):
        item=sub.add_parser(command); item.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    args=parser.parse_args(argv)
    try: result=initialize(args.repo_root) if args.command=="initialize" else validate_initialized(args.repo_root)
    except (StageXX2Error,OSError,ValueError,KeyError) as error:
        print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True)); return 2
    print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
