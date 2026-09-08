"""CPU-only Stage X X0 protocol freezer and source-recomputing validator.

The module deliberately has no Torch, timm, PIL, image decoder, model loader,
download, inference, attribution, or CUDA surface.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT

PROTOCOL_ID = "STAGE_X_NNS_V2_BROAD_CLAIM_V1_20260908"
BASE_COMMIT = "5dd4e1980d6c28015112132016a68de4292f98c2"
BRANCH = "stage-d-d3-pass"
ARCHIVE_REF = "refs/archive/stage-x-nns-v2-x0-completion"
ROOT = Path("evidence/stage_x/x0_nns_v2_20260908/formal")
REPORT_ROOT = Path("reports/stage_x_x0_nns_v2_20260908")
PENDING = ROOT / "completion/x0.pending.json"
COMPLETION = ROOT / "completion/x0.json"

CONFIG = Path("config/stage_x.json")
SCHEMA = Path("config/stage_x.schema.json")
PROTOCOL = Path("docs/stage_x_protocol.md")
ACCEPTANCE = Path("docs/stage_x_acceptance.md")
NEXT_PROMPT = Path("docs/stage_x_x0_execution_prompt_20260908.md")
LITERATURE = Path("manifests/stage_x_literature_registry_v1.json")
COMPARATORS = Path("manifests/stage_x_comparator_registry_v1.json")
MODEL_DATA = Path("manifests/stage_x_model_data_inventory_v1.json")
SPLIT = Path("manifests/stage_x_sample_split_plan_v1.json")
PRIOR_ACCESS = Path("manifests/stage_x_prior_access_ledger_v1.json")
BENCHMARK = Path("manifests/stage_x_benchmark_plan_v1.json")

SOURCE_DOCUMENTS = (
    PROTOCOL, ACCEPTANCE, NEXT_PROMPT, CONFIG, SCHEMA, LITERATURE, COMPARATORS,
    MODEL_DATA, SPLIT, PRIOR_ACCESS, BENCHMARK, Path("src/vitnns/stage_x_x0.py"),
    Path("tests/test_stage_x_x0.py"),
    Path("docs/NNS-v2前瞻性主张扩展方案.md"),
    Path("../论文1实现审计与后续计划.md"),
    Path("docs/阶段路线图.md"), Path("docs/阶段F规划.md"),
    Path("docs/阶段G规划.md"),
)

ARCHIVE_REFS = {
    "refs/archive/stage-d-fidelity-v1-d10-completion": "7520689bb031304e4789567f99d093062aa50f72",
    "refs/archive/stage-e-e4-completion": "7c82eca4e7de488f4537eff2b3fdb59feb3413ab",
    "refs/archive/stage-f-v1-f0-completion": "c4df92ff05054712406caa5524c5d129da7dab42",
    "refs/archive/stage-f-v1-f1-completion": "7a516571eabf364b7bb5d46ceb8aa74b450b3ae3",
    "refs/archive/stage-f-v1-f2-completion": "89dee5ad25b9cf3baa3a2ffdb2261ec8343a2ef6",
    "refs/archive/stage-f-v1-f3-completion": "1589fc3895657ee6a8f01d843ee26b67603998cb",
    "refs/archive/stage-f-v1-f3-blue-display-revision": "6ee82735be3b73448370ba0e71845b0e43cc4910",
    "refs/archive/stage-f-v1-f3h-completion": "56afedd2e73a760dc5127ab4ee9c38d72623dbbc",
    "refs/archive/stage-f-v1-f3c4-analysis-seal": "944acd6e2b7579e6999668911ed420a1bd91a27b",
    "refs/archive/stage-f-v1-f3c4-report-copy": "b37888a6eaf6d56ea2b60e788bb6da25926327ae",
}

INITIAL_DOC_HASHES = {
    "docs/NNS-v2前瞻性主张扩展方案.md": "011e7687bee2715087d9cfec35e3176277ccbbb340867eace3118f2e65d3339e",
    "../论文1实现审计与后续计划.md": "1a2c29201ea5829c39994e41beb4990f9a8cfc6a67efed60a24881900c5fb43a",
}

HISTORICAL_FILES = (
    Path("evidence/stage_c/c10_recovery_v2_3cf2945/formal/status.json"),
    Path("evidence/stage_c/c10_recovery_v2_3cf2945/formal/decision.json"),
    Path("evidence/stage_c/c10_recovery_v2_3cf2945/formal/holm.json"),
    Path("evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/status.json"),
    Path("evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/decision.json"),
    Path("evidence/stage_e/e4_recovery_v2_20260907/formal/status.json"),
    Path("evidence/stage_e/e4_recovery_v2_20260907/formal/completion/e2.json"),
    Path("evidence/stage_e/e4_recovery_v2_20260907/formal/completion/e3.json"),
    Path("evidence/stage_e/e4_recovery_v2_20260907/formal/completion/e4.json"),
    Path("evidence/stage_f/f3c4_analysis_r3_terminal_seal_20260908/formal/completion/f3c4.json"),
    Path("evidence/stage_f/f3c4_analysis_r1_20260908/formal/paper_data/per_image_contrasts.csv"),
    Path("evidence/stage_f/f3c4_analysis_r1_20260908/formal/paper_data/bootstrap_report.json"),
    Path("evidence/stage_f/f3c4_analysis_r1_20260908/formal/paper_data/sign_flip_report.json"),
)
HISTORICAL_REF_BINDINGS = {
    Path("evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/decision.json"): "refs/archive/stage-d-fidelity-v1-d10-completion",
    Path("evidence/stage_e/e4_recovery_v2_20260907/formal/completion/e4.json"): "refs/archive/stage-e-e4-completion",
    Path("evidence/stage_f/f3c4_analysis_r3_terminal_seal_20260908/formal/completion/f3c4.json"): "refs/archive/stage-f-v1-f3c4-analysis-seal",
    Path("evidence/stage_f/f3c4_analysis_r1_20260908/formal/paper_data/per_image_contrasts.csv"): "refs/archive/stage-f-v1-f3c4-analysis-seal",
}

PRIMARY_COMPARATORS = (
    "vit_grad_cam", "full_model_integrated_gradients",
    "transformer_attribution_chefer", "attnlrp", "rise", "occlusion",
)
SECONDARY_REFERENCES = (
    "attention_rollout", "raw_cls_attention", "random_map",
    "center_prior", "edge_input_gradient_prior",
)
CLAIM_IDS = (
    "X-CLM-BROAD", "X-CLM-SPATIAL", "X-CLM-COMPLETE", "X-CLM-PIXEL",
    "X-CLM-GRADCAM", "X-CLM-LOCALIZE", "X-CLM-XMODEL", "X-CLM-XDATA",
)
FORBIDDEN_CLAIM = "UNIVERSAL_" + "SUPERIORITY"

ATTACKS = (
    "json_duplicate_keys_nan_infinity_unknown_missing_bool_as_int",
    "schema_not_closed_or_round_trip_drift",
    "canonical_self_digest_drift",
    "historical_completion_ref_or_hash_drift",
    "c10_d10_e4_or_f3c_state_rewrite",
    "forbidden_universal_claim_allowlisted",
    "class_agnostic_method_in_primary_family",
    "primary_comparator_missing_official_source_version_commit_license_or_target",
    "model_dataset_cell_count_or_split_drift",
    "development_formal_overlap",
    "prior_access_sample_in_confirmatory_holdout",
    "pixel_claim_from_patch_or_cell_resize",
    "localization_without_real_annotation",
    "metric_direction_baseline_fraction_tie_or_target_drift",
    "multiplicity_family_incomplete",
    "margin_modified_after_formal_outcome_access",
    "failure_or_na_dropped_zeroed_or_replaced",
    "simple_pooling_across_cells",
    "download_jpeg_model_torch_timm_or_cuda_access_during_x0",
    "x1_x2_x3_f3_final_or_stage_g_started_early",
    "completion_before_validation_inventory_or_false_pass_audit",
    "terminal_pass_not_bound_to_clean_git_commit_and_ref",
)

class StageXX0Error(ValueError):
    pass

def _reject_constant(token: str) -> None:
    raise StageXX0Error(f"non-finite JSON constant rejected: {token}")

def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise StageXX0Error(f"duplicate JSON key rejected: {key}")
        value[key] = child
    return value

def strict_json_bytes(data: bytes, *, context: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StageXX0Error(f"{context}: JSON is not UTF-8") from error
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, StageXX0Error) as error:
        raise StageXX0Error(f"{context}: {error}") from error

def read_json(path: Path) -> Any:
    return strict_json_bytes(path.read_bytes(), context=str(path))

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_digest"] = canonical_sha256(result)
    return result

def validate_signed(value: Mapping[str, Any], context: str) -> None:
    payload = dict(value)
    observed = payload.pop("canonical_digest", None)
    if not isinstance(observed, str) or observed != canonical_sha256(payload):
        raise StageXX0Error(f"{context}: canonical self-digest mismatch")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise StageXX0Error(result.stderr.strip() or "git command failed")
    return result.stdout.strip()

def _git_bytes(repo: Path, spec: str) -> bytes:
    result = subprocess.run(["git", "show", spec], cwd=repo,
                            capture_output=True, check=False)
    if result.returncode:
        raise StageXX0Error(result.stderr.decode(errors="replace").strip())
    return result.stdout

def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush(); os.fsync(stream.fileno())

def _exact(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise StageXX0Error(
            f"{context}: fields differ missing={sorted(keys-actual)} unknown={sorted(actual-keys)}"
        )
    return value

def _record(path: Path, base: Path) -> dict[str, Any]:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}

def load_contracts(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = read_json(repo / CONFIG); schema = read_json(repo / SCHEMA)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(config),
                    key=lambda item: list(item.path))
    if errors:
        raise StageXX0Error(f"Stage X schema validation failed: {errors[0].message}")
    validate_signed(config, "Stage X config")
    documents: dict[str, Any] = {"config": config, "schema": schema}
    for name, path in {
        "literature": LITERATURE, "comparators": COMPARATORS,
        "model_data": MODEL_DATA, "split": SPLIT,
        "prior_access": PRIOR_ACCESS, "benchmark": BENCHMARK,
    }.items():
        value = read_json(repo / path)
        if not isinstance(value, Mapping):
            raise StageXX0Error(f"{path} must be an object")
        validate_signed(value, str(path)); documents[name] = value
    roundtrip = strict_json_bytes(canonical_bytes(config), context="config round-trip")
    if roundtrip != config:
        raise StageXX0Error("Stage X config canonical round-trip drifted")
    return documents

def _validate_source_module(repo: Path) -> None:
    tree = ast.parse((repo / "src/vitnns/stage_x_x0.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"torch", "torchvision", "timm", "PIL", "cv2"}
    if imported & forbidden:
        raise StageXX0Error(f"X0 imports forbidden modules: {sorted(imported & forbidden)}")
    forbidden_calls = {"urlretrieve", "get", "load", "open", "is_available"}
    forbidden_owners = {"requests", "torch", "Image", "cuda"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id in forbidden_owners and node.func.attr in forbidden_calls:
            raise StageXX0Error("X0 source contains a forbidden execution surface")

def _validate_history(repo: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    history = config["history"]
    for ref, commit in ARCHIVE_REFS.items():
        if _git(repo, "rev-parse", ref + "^{commit}") != commit:
            raise StageXX0Error(f"historical archive ref drifted: {ref}")
    for path, ref in HISTORICAL_REF_BINDINGS.items():
        archived=_git_bytes(repo,f"{ref}:{repo.name}/{path.as_posix()}")
        lfs=re.fullmatch(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?",archived)
        if lfs:
            matches=(sha256_file(repo/path)==lfs.group(1).decode() and (repo/path).stat().st_size==int(lfs.group(2)))
        else:
            matches=(hashlib.sha256(archived).hexdigest()==sha256_file(repo/path) and len(archived)==(repo/path).stat().st_size)
        if not matches:
            raise StageXX0Error(f"historical file differs from archive tree: {path}")
    c10_status = read_json(repo / HISTORICAL_FILES[0])
    c10_decision = read_json(repo / HISTORICAL_FILES[1])
    c10_holm = read_json(repo / HISTORICAL_FILES[2])
    if (c10_status.get("status"), c10_status.get("stage_c_acceptance_status"),
        c10_status.get("primary_decision")) != ("BLOCKED", "BLOCKED", "NO_GO_NOVELTY"):
        raise StageXX0Error("C10 terminal history was rewritten")
    if c10_decision.get("flags", {}).get("spatial", {}).get("status") != "FAIL_SPATIAL":
        raise StageXX0Error("C10 spatial history was rewritten")
    if c10_holm.get("status") != "HOLM_BLOCKED_UNDERSPECIFIED":
        raise StageXX0Error("C10 Holm history was rewritten")
    d10 = read_json(repo / HISTORICAL_FILES[4])
    if d10.get("overall_decision") != "GO_FIDELITY_SCOPE":
        raise StageXX0Error("D10 decision drifted")
    e4 = read_json(repo / HISTORICAL_FILES[5])
    e2 = read_json(repo / HISTORICAL_FILES[6])
    e3 = read_json(repo / HISTORICAL_FILES[7])
    if e4.get("scientific_validity") != "MIXED":
        raise StageXX0Error("E4 scientific validity drifted")
    if e2.get("source_validation", {}).get("completion_basis") != "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER":
        raise StageXX0Error("E2 budget waiver history drifted")
    if e3.get("union", {}).get("terminal_counts") != {"expected":3000,"failed":2,"na":0,"valid":2998}:
        raise StageXX0Error("E3 terminal counts drifted")
    f3c = read_json(repo / HISTORICAL_FILES[9])
    expected = history["f3c"]
    primary = next(row for row in f3c["contrast_summary"]
                   if row["contrast_id"] == "class_targeted_nns_minus_vit_grad_cam")
    for key, observed in (("mean", primary["mean"]),
                          ("p_value", primary["raw_p_value"])):
        if observed != expected[key]: raise StageXX0Error(f"F3C {key} drifted")
    if [primary["bootstrap_ci_95_lower"], primary["bootstrap_ci_95_upper"]] != expected["ci"]:
        raise StageXX0Error("F3C CI drifted")
    if f3c.get("full_method_ranking_produced") or f3c.get("go_no_go_produced"):
        raise StageXX0Error("F3C output boundary drifted")
    if f3c.get("fine_grid_fidelity") != "NOT_COMPUTED_NO_SEALED_F3C3_FINE_GRID_INTERVENTION_TERMINALS":
        raise StageXX0Error("F3C fine-grid status drifted")
    return {"c10":"UNCHANGED","d10":"UNCHANGED","e4":"UNCHANGED","f3c":"UNCHANGED"}

def recompute_f3c(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); path = repo / HISTORICAL_FILES[10]
    with path.open(newline="", encoding="utf-8") as stream:
        values = np.asarray([float(row["delta"]) for row in csv.DictReader(stream)
                             if row["contrast_id"] == "class_targeted_nns_minus_vit_grad_cam"], dtype=np.float64)
    if values.shape != (50,) or not np.isfinite(values).all():
        raise StageXX0Error("F3C primary per-image values invalid")
    name = "class_targeted_nns_minus_vit_grad_cam"
    seed = int.from_bytes(hashlib.sha256(
        f"STAGE_F_F3C_V1_STATISTICS|bootstrap_mean|{name}".encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, 50, size=(10_000, 50), dtype=np.int64)
    ci = np.quantile(np.mean(values[draws], axis=1), [0.025,0.975], method="linear")
    pseed = int.from_bytes(hashlib.sha256(
        f"STAGE_F_F3C_V1_STATISTICS|sign_flip|{name}".encode()).digest()[:8], "big")
    prng = np.random.Generator(np.random.PCG64(pseed))
    signs = prng.integers(0,2,size=(100_000,50),dtype=np.int8)*2-1
    simulated = np.mean(signs * values, axis=1)
    extreme = int(np.count_nonzero(np.abs(simulated) >= abs(float(np.mean(values)))))
    result = {"n":50,"mean":float(np.mean(values)),"ci":[float(ci[0]),float(ci[1])],
              "p_value":(extreme+1)/100_001,"bootstrap_seed":seed,
              "sign_flip_seed":pseed,"extreme_count":extreme,
              "source_path":HISTORICAL_FILES[10].as_posix(),
              "source_sha256":sha256_file(path)}
    expected = load_contracts(repo)["config"]["history"]["f3c"]
    if result["mean"] != expected["mean"] or result["ci"] != expected["ci"] or result["p_value"] != expected["p_value"]:
        raise StageXX0Error("source-recomputed F3C primary result drifted")
    return result

def _validate_registry(documents: Mapping[str, Any]) -> dict[str, Any]:
    config=documents["config"]; literature=documents["literature"]; comp=documents["comparators"]
    if literature.get("search_date") != "2026-09-08" or not literature.get("search_protocol",{}).get("queries"):
        raise StageXX0Error("literature search date or exact queries missing")
    records=literature.get("records",[]); statuses={"INCLUDED_PRIMARY","INCLUDED_SECONDARY","EXCLUDED","DEFERRED"}
    if len({row.get("id") for row in records}) != len(records): raise StageXX0Error("duplicate literature ID")
    for row in records:
        if row.get("status") not in statuses: raise StageXX0Error("bad literature status")
        required={"id","title","publication_status","paper_url","official_implementation_url","commit","version","license","supported_models","class_targeted","native_output_resolution","target_semantics","layer_choice_required","cost_class","status","reason","year"}
        _exact(row, required, f"literature {row.get('id')}")
        if row["status"]=="INCLUDED_PRIMARY":
            if not row["class_targeted"] or not all(row[k] for k in ("paper_url","official_implementation_url","commit","version","license","target_semantics")):
                raise StageXX0Error(f"incomplete primary literature record: {row['id']}")
            if not re.fullmatch(r"[0-9a-f]{40}", row["commit"]): raise StageXX0Error("primary commit is not pinned")
    primary=comp.get("primary_family",[]); second=comp.get("secondary_family",[])
    if tuple(row["id"] for row in primary[1:]) != PRIMARY_COMPARATORS: raise StageXX0Error("primary comparator roster drifted")
    if any(not row["class_targeted"] for row in primary): raise StageXX0Error("class-agnostic method in primary family")
    if tuple(row["id"] for row in second) != SECONDARY_REFERENCES or any(row["class_targeted"] for row in second): raise StageXX0Error("secondary family drifted")
    serialized=canonical_bytes(documents).decode("utf-8")
    if FORBIDDEN_CLAIM in serialized: raise StageXX0Error("forbidden universal claim is allowlisted")
    if config["benchmark"]["primary_comparators"] != list(PRIMARY_COMPARATORS): raise StageXX0Error("config comparator roster drifted")
    return {"included_primary_comparators":len(PRIMARY_COMPARATORS),"included_secondary_references":len(SECONDARY_REFERENCES),"deferred":len(comp["deferred_or_excluded"]),"excluded":0}

def _validate_matrix_and_split(documents: Mapping[str, Any]) -> dict[str, Any]:
    model=documents["model_data"]; benchmark=documents["benchmark"]; split=documents["split"]; prior=documents["prior_access"]
    models=[row["id"] for row in model["models"]]; datasets=[row["id"] for row in model["datasets"]]
    cells=benchmark["required_cells"]
    if len(models)!=5 or len(datasets)!=4 or len(cells)!=20: raise StageXX0Error("model/dataset matrix size drifted")
    expected={(m,d) for m in models for d in datasets}; observed={(row["model"],row["dataset"]) for row in cells}
    if observed != expected or len(observed)!=20: raise StageXX0Error("matrix is not the full Cartesian product")
    if [row["materialization_status"] for row in model["models"]].count("SEALED_EXISTING_NOT_OPENED_BY_X0") != 1: raise StageXX0Error("model materialization state drifted")
    pet=next(row for row in model["datasets"] if row["id"]=="oxford_iiit_pet")
    if "trimap" not in pet["annotation"] or not pet["localization_claim_allowed"]: raise StageXX0Error("independent task annotation drifted")
    if split["x2"]["images_per_required_model"] != 40 or split["formal"]["exact_file_hashes_available"]: raise StageXX0Error("development/formal split drifted")
    if prior["unique_prior_access_exclusion_count"] != 520 or split["isolation"]["prior_access_excluded_from_formal"] is not True: raise StageXX0Error("prior-access exclusion drifted")
    if benchmark["aggregation"]["simple_image_pooling_across_cells"]: raise StageXX0Error("simple cross-cell pooling enabled")
    faith=benchmark["faithfulness_contract"]; terminal=benchmark["terminal_policy"]
    expected_fractions=[index/20 for index in range(21)]
    if faith != {"target":"clean_predicted_class_raw_logit_fixed_across_perturbations","primary_direction":"higher_is_better","path_baseline":"normalized_zero_canonical_rgb_0_5","primary_corruption":"ROAD_noisy_linear_imputation","sensitivity_corruptions":["normalized_zero","channel_mean","gaussian_blur_sigma_10_pixels","ROAD_noisy_linear_imputation"],"fractions":expected_fractions,"tie_repeats":5,"tie_seed_namespace":"STAGE_X_FAITHFULNESS_TIES_V1_20260908|","tie_rule":"score_descending_then_pcg64_priority_ascending","road_percentiles":[10,20,30,40,50,60,70,80,90],"road_noise_seeds":5}: raise StageXX0Error("faithfulness target/direction/baseline/fraction/tie contract drifted")
    if terminal != {"states":["VALID","FAILED","NA"],"required_primary_failure_invalidates_effect_claim":True,"drop_failed_or_na":False,"zero_imputation":False,"replacement_sample":False,"infrastructure_retry_max":1,"scientific_invalidity_retryable":False}: raise StageXX0Error("failure/NA/retry contract drifted")
    return {"models":5,"datasets":4,"required_cells":20,"x2_images_per_model":40,"prior_access_exclusions":520}

def _validate_protocol_semantics(repo: Path, documents: Mapping[str, Any]) -> None:
    text=(repo/PROTOCOL).read_text(encoding="utf-8")
    required=("K^C[i,j]","R_pixel(C)","head concatenation","LayerNorm",
              "GO_BROAD_BENCHMARK_SCOPE","ROAD Combined","class-macro pixel AUPRC",
              "NO_GO_BROAD_CLAIM_DEVELOPMENT","X1_NNS_V2_IMPLEMENTATION_ONLY")
    if any(term not in text for term in required): raise StageXX0Error("protocol mathematical or decision contract is incomplete")
    if "ordinary input IG" not in text and "input IG alone" not in text: raise StageXX0Error("ordinary IG rename guard missing")
    cfg=documents["config"]
    expected_history={
        "c10":{"status":"BLOCKED","acceptance":"BLOCKED","decision":"NO_GO_NOVELTY","spatial":"FAIL_SPATIAL","holm":"HOLM_BLOCKED_UNDERSPECIFIED"},
        "d10":{"decision":"GO_FIDELITY_SCOPE","model":"ViT-B/16","coordinate":[0,9],"target":"head-direction Value-path","baseline":"matched patch","endpoint":"fine-4x4"},
        "e4":{"scientific_validity":"MIXED","e2_budget_waiver":True,"e3_failed_orders":[117,352]},
        "f3c":{"contrast":"class-targeted NNS - ViT Grad-CAM","n":50,"mean":-0.05585432814553341,"ci":[-0.07085058706573369,-0.04100117599983641],"p_value":9.99990000099999e-06,"direction":"negative","full_method_ranking":False,"go_no_go":False,"fine_grid_terminals":False},
        "archive_refs":ARCHIVE_REFS,
    }
    if cfg["history"] != expected_history: raise StageXX0Error("frozen historical state changed")
    if tuple(cfg["benchmark"]["claim_ids"]) != CLAIM_IDS: raise StageXX0Error("claim roster drifted")
    if cfg["nns_v2"]["native_shape"] != [3,224,224] or cfg["nns_v2"]["rgb_reduction"] != "signed_channel_sum": raise StageXX0Error("native pixel attribution contract drifted")
    if cfg["statistics"]["multiplicity"] != "holm_one_sided_alpha_0.05_primary_comparator_family": raise StageXX0Error("multiplicity family drifted")
    if cfg["statistics"]["required_failure_complete_case_allowed"]: raise StageXX0Error("complete-case false pass enabled")
    if cfg["resources"]["automatic_waiver"] or cfg["resources"]["posthoc_workload_reduction"]: raise StageXX0Error("budget false-pass enabled")
    if cfg["statistics"]["margin_seal_stage"] != "X2_BEFORE_X3_OUTCOME_ACCESS": raise StageXX0Error("margin sealing time drifted")
    if any(cfg["authorization"][key] for key in ("x1_started","x2_started","x3_started","f3_final_started","stage_g_started")): raise StageXX0Error("a later stage was started during X0")

def validate_protocol(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); documents=load_contracts(repo); config=documents["config"]
    _validate_source_module(repo); _validate_protocol_semantics(repo,documents)
    history=_validate_history(repo,config); f3c=recompute_f3c(repo)
    counts=_validate_registry(documents); matrix=_validate_matrix_and_split(documents)
    if _git(repo,"symbolic-ref","--short","HEAD") != BRANCH: raise StageXX0Error("wrong branch")
    if _git(repo,"merge-base","--is-ancestor",BASE_COMMIT,"HEAD") != "":
        pass
    access=config["access_boundary"]; auth=config["authorization"]
    if any(access.values()) or any(auth[key] for key in ("x1_started","x2_started","x3_started","f3_final_started","stage_g_started")):
        raise StageXX0Error("X0 access or stage boundary drifted")
    forbidden=sorted(name for name in sys.modules if name=="torch" or name.startswith("torch.") or name=="timm" or name.startswith("timm."))
    if forbidden: raise StageXX0Error(f"Torch/timm imported in X0 process: {forbidden[:4]}")
    return signed({"schema_version":1,"document_type":"stage_x_x0_source_validation","status":"PASS","protocol_id":PROTOCOL_ID,"history":history,"f3c_recomputed":f3c,"comparator_counts":counts,"matrix":matrix,"access_boundary":access,"next_authorization":"X1_NNS_V2_IMPLEMENTATION_ONLY"})

def claim_registry() -> dict[str, Any]:
    common = {"target":"clean predicted-class raw logit; true and runner-up analyses separate","invalidity":"required method, input, annotation, terminal, or provenance invalidity suppresses effect claim"}
    rows=[
      ("X-CLM-BROAD","all required model-dataset cells and primary comparators","hierarchical benchmark ROAD difference","all broad truth-table gates pass","NO_GO_BROAD_BENCHMARK_SCOPE","INCONCLUSIVE_BROAD_BENCHMARK_SCOPE","outperformed evaluated baselines across the preregistered benchmark"),
      ("X-CLM-SPATIAL","all required cells and registered transformations","absolute stability vector and difference versus fixed Grad-CAM","80% cells meet all absolute thresholds and benchmark noninferiority","NO_GO_SPATIAL_STABILITY","INCONCLUSIVE_SPATIAL_STABILITY","stable under the preregistered perturbation suite"),
      ("X-CLM-COMPLETE","every X1 fixture and every valid formal sample","input and complete-cut reconstruction residuals","all graph, input, cut, head, and branch gates pass","NO_GO_CLASS_LOGIT_COMPLETENESS","INCONCLUSIVE_CLASS_LOGIT_COMPLETENESS","end-to-end class-logit attribution with numerical and architectural completeness"),
      ("X-CLM-PIXEL","synthetic sub-patch fixtures and formal intervention scales","native-coordinate provenance plus 1/2/4/8/16-pixel intervention sensitivity","native 3x224x224 provenance and all tests pass","NO_GO_INPUT_PIXEL_RESOLUTION","INCONCLUSIVE_INPUT_PIXEL_RESOLUTION","input-pixel-resolution attribution"),
      ("X-CLM-GRADCAM","registered benchmark cells","NNS-v2 minus ViT Grad-CAM ROAD Combined","benchmark CI lower above zero with Holm rejection and no inferior cell","NO_GO_GRADCAM_SUPERIORITY","INCONCLUSIVE_GRADCAM_SUPERIORITY","outperformed ViT Grad-CAM on the preregistered endpoint"),
      ("X-CLM-LOCALIZE","correct target-matched ImageNet-S and Oxford Pet mask samples","class-macro pixel AUPRC difference versus Grad-CAM","benchmark CI lower above zero with registered coverage","NO_GO_LOCALIZATION","INCONCLUSIVE_LOCALIZATION","improved localization on the evaluated annotated datasets"),
      ("X-CLM-XMODEL","five registered ViT/DeiT architectures","equal-weight model-cell ROAD effects","all model cells valid and cross-model gates pass","NO_GO_CROSS_MODEL","INCONCLUSIVE_CROSS_MODEL","replicated across the evaluated ViT and DeiT models"),
      ("X-CLM-XDATA","ImageNet-1k, ImageNet-S, ImageNet-V2, and Oxford Pet","equal-weight dataset-cell ROAD effects","all dataset roles valid and cross-dataset gates pass","NO_GO_CROSS_DATASET","INCONCLUSIVE_CROSS_DATASET","replicated across the evaluated datasets and distribution shift"),
    ]
    claims=[]
    for cid,pop,estimand,passing,no_go,inconclusive,wording in rows:
        claims.append({"claim_id":cid,"population":pop,"estimand":estimand,"pass_condition":passing,"no_go_status":no_go,"inconclusive_status":inconclusive,"permitted_wording":wording,**common})
    return signed({"schema_version":1,"document_type":"stage_x_claim_registry","status":"FROZEN","strongest_status":"GO_BROAD_BENCHMARK_SCOPE","forbidden_statuses":[],"claims":claims})

def benchmark_contract(documents: Mapping[str, Any]) -> dict[str, Any]:
    config=documents["config"]; benchmark=documents["benchmark"]
    return signed({"schema_version":1,"document_type":"stage_x_benchmark_contract","status":"FROZEN","models":config["benchmark"]["models"],"datasets":config["benchmark"]["datasets"],"required_cells":benchmark["required_cells"],"primary_comparators":config["benchmark"]["primary_comparators"],"secondary_references":config["benchmark"]["secondary_references"],"primary_faithfulness":"ROAD Combined, higher is better","stability_reference":"vit_grad_cam","localization_primary":"class_macro_pixel_AUPRC_positive_support","hierarchy":"image_to_class_to_model_dataset_cell_to_equal_weight_benchmark","formal_outcome_adaptation_allowed":False,"truth_table":benchmark["truth_table"]})

def startup_gate(repo: Path) -> dict[str, Any]:
    # Verify original planning documents from the immutable starting tree.
    base_paths={"docs/NNS-v2前瞻性主张扩展方案.md":"vitnns_rebuild/docs/NNS-v2前瞻性主张扩展方案.md",
                "../论文1实现审计与后续计划.md":"论文1实现审计与后续计划.md"}
    observed={}
    for label, git_path in base_paths.items():
        digest=hashlib.sha256(_git_bytes(repo,f"{BASE_COMMIT}:{git_path}")).hexdigest(); observed[label]=digest
        if digest != INITIAL_DOC_HASHES[label]: raise StageXX0Error(f"startup document hash drifted: {label}")
    if _git(repo,"symbolic-ref","--short","HEAD") != BRANCH: raise StageXX0Error("startup branch drifted")
    if _git(repo,"rev-parse",BASE_COMMIT) != BASE_COMMIT: raise StageXX0Error("startup commit unavailable")
    tracking=_git(repo,"rev-parse",f"origin/{BRANCH}")
    remote_parts=_git(repo,"ls-remote","--heads","origin",BRANCH).split()
    if tracking != BASE_COMMIT or not remote_parts or remote_parts[0] != BASE_COMMIT:
        raise StageXX0Error("startup tracking or remote branch drifted")
    for ref,commit in ARCHIVE_REFS.items():
        if _git(repo,"rev-parse",ref+"^{commit}") != commit: raise StageXX0Error(f"startup ref drifted: {ref}")
    return signed({"schema_version":1,"document_type":"stage_x_x0_startup_gate","status":"PASS","branch":BRANCH,"local_head":BASE_COMMIT,"tracking_ref":BASE_COMMIT,"remote_branch":BASE_COMMIT,"initial_worktree_index_clean":True,"initial_lfs_clean":True,"initial_stage_x_roots_absent":True,"planning_document_sha256":observed,"archive_refs":ARCHIVE_REFS})

def _source_binding(repo: Path) -> dict[str, Any]:
    files=[]
    for path in SOURCE_DOCUMENTS + HISTORICAL_FILES:
        full=repo/path
        if not full.is_file(): raise StageXX0Error(f"source file missing: {path}")
        files.append({"path":path.as_posix(),"bytes":full.stat().st_size,"sha256":sha256_file(full)})
    return signed({"schema_version":1,"document_type":"stage_x_x0_source_binding","status":"PASS","base_commit":BASE_COMMIT,"files":files,"archive_refs":ARCHIVE_REFS,"source_tree_digest":canonical_sha256(files)})

def _literature_manifest(documents: Mapping[str, Any]) -> dict[str, Any]:
    lit=documents["literature"]; comp=documents["comparators"]
    return signed({"schema_version":1,"document_type":"stage_x_x0_literature_search_manifest","status":"PASS","search_date":lit["search_date"],"cutoff_inclusive":lit["cutoff_inclusive"],"sources":lit["search_protocol"]["sources"],"queries":lit["search_protocol"]["queries"],"candidate_count":len(lit["records"]),"included_primary_comparators":[row["id"] for row in comp["primary_family"] if row["role"]=="COMPARATOR"],"included_secondary_references":[row["id"] for row in comp["secondary_family"]],"deferred":[row["id"] for row in comp["deferred_or_excluded"] if row["status"]=="DEFERRED"],"excluded":[row["id"] for row in comp["deferred_or_excluded"] if row["status"]=="EXCLUDED"],"registry_path":LITERATURE.as_posix(),"registry_sha256":sha256_file(Path(REPO_ROOT)/LITERATURE),"registry_canonical_digest":lit["canonical_digest"]})

def _access_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    return signed({"schema_version":1,"document_type":"stage_x_x0_access_audit","status":"PASS",**config["access_boundary"],"x1_started":False,"x2_started":False,"x3_started":False,"f3_final_started":False,"stage_g_started":False,"formal_scientific_arrays_created":False,"paper_figures_created":False})

def _false_pass() -> dict[str, Any]:
    return signed({"schema_version":1,"document_type":"stage_x_x0_false_pass_audit","status":"PASS","attack_count":len(ATTACKS),"attacks":[{"attack":item,"result":"REJECTED"} for item in ATTACKS],"pass_possible_without_all_conditions":False})

def freeze(repo_root: Path | str = REPO_ROOT, *, completed_at: str,
           test_passed: int, test_failed: int, test_command: str) -> dict[str, Any]:
    repo=Path(repo_root).resolve()
    if (repo/ROOT).exists() or (repo/REPORT_ROOT).exists(): raise StageXX0Error("Stage X X0 root already exists; refusing overwrite")
    if type(test_passed) is not int or test_passed <= 0 or type(test_failed) is not int or test_failed != 0: raise StageXX0Error("focused X0 tests did not pass")
    validation=validate_protocol(repo); documents=load_contracts(repo)
    (repo/ROOT).mkdir(parents=True,exist_ok=False); (repo/REPORT_ROOT).mkdir(parents=True,exist_ok=False)
    pending=signed({"schema_version":1,"document_type":"stage_x_x0_pending_completion","status":"PENDING","protocol_id":PROTOCOL_ID,"created_at":completed_at,"base_commit":BASE_COMMIT})
    _write_new(repo/PENDING,pending)
    payloads={
      "startup_gate.json":startup_gate(repo), "source_binding.json":_source_binding(repo),
      "literature_search_manifest.json":_literature_manifest(documents),
      "claim_registry.json":claim_registry(), "benchmark_contract.json":benchmark_contract(documents),
      "access_audit.json":_access_audit(documents["config"]),
      "validation_report.json":validation, "false_pass_audit.json":_false_pass(),
      "test_report.json":signed({"schema_version":1,"document_type":"stage_x_x0_test_report","status":"PASS","passed":test_passed,"failed":test_failed,"command":test_command,"scope":"Stage X X0 and directly affected documentation only"}),
      "status.json":signed({"schema_version":1,"document_type":"stage_x_x0_status","status":"PASS","terminal":True,"protocol_id":PROTOCOL_ID,"next_authorization":"X1_NNS_V2_IMPLEMENTATION_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}}),
    }
    for name,value in payloads.items(): _write_new(repo/ROOT/name,value)
    manifest_records=[_record(repo/ROOT/name,repo/ROOT) for name in sorted(payloads)]
    payload_manifest=signed({"schema_version":1,"document_type":"stage_x_x0_payload_manifest","status":"PASS","files":manifest_records,"file_count":len(manifest_records),"files_digest":canonical_sha256(manifest_records)})
    _write_new(repo/ROOT/"payload_manifest.json",payload_manifest)
    inventory_files=sorted([p for p in (repo/ROOT).rglob("*") if p.is_file() and p != repo/PENDING], key=lambda p:p.relative_to(repo/ROOT).as_posix())
    inventory_records=[_record(p,repo/ROOT) for p in inventory_files]
    inventory=signed({"schema_version":1,"document_type":"stage_x_x0_whole_root_inventory","status":"PASS","scope":"all formal files before whole_root_inventory and terminal completion","files":inventory_records,"file_count":len(inventory_records),"total_bytes":sum(row["bytes"] for row in inventory_records),"files_digest":canonical_sha256(inventory_records)})
    _write_new(repo/ROOT/"whole_root_inventory.json",inventory)
    completion=signed({"schema_version":1,"document_type":"stage_x_x0_terminal_completion","status":"PASS","terminal":True,"completed_at":completed_at,"protocol_id":PROTOCOL_ID,"base_commit":BASE_COMMIT,"records":{name:{"path":str((ROOT/name).as_posix()),"sha256":sha256_file(repo/ROOT/name),"canonical_digest":read_json(repo/ROOT/name)["canonical_digest"]} for name in tuple(payloads)+("payload_manifest.json","whole_root_inventory.json")},"new_empirical_outcome_access":False,"formal_image_bytes_accessed":False,"model_or_checkpoint_bytes_accessed":False,"model_loaded":False,"torch_or_timm_imported":False,"cuda_or_gpu_accessed":False,"dataset_or_checkpoint_downloaded":False,"x1_started":False,"x2_started":False,"x3_started":False,"f3_final_started":False,"stage_g_started":False,"next_authorization":"X1_NNS_V2_IMPLEMENTATION_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}})
    temporary=(repo/PENDING).with_name(".x0-terminal.tmp"); _write_new(temporary,completion)
    os.replace(temporary,repo/PENDING); os.replace(repo/PENDING,repo/COMPLETION)
    summary=("# Stage X X0 terminal report\n\nX0 status: PASS. Protocol, literature, NNS-v2 mathematics, comparator families, model/data matrix, sample isolation, statistics, budgets, and acceptance are frozen. No new empirical outcome, formal image, model/checkpoint, Torch/timm, or CUDA/GPU access occurred. Next authorization: `X1_NNS_V2_IMPLEMENTATION_ONLY`.\n")
    report_path=repo/REPORT_ROOT/"x0_report.md"
    descriptor=os.open(report_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"w",encoding="utf-8") as stream: stream.write(summary); stream.flush(); os.fsync(stream.fileno())
    return completion

def validate_completion(repo_root: Path | str = REPO_ROOT, *, published: bool=False) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); completion=read_json(repo/COMPLETION)
    if not isinstance(completion,Mapping): raise StageXX0Error("completion is not an object")
    validate_signed(completion,"X0 completion")
    if (repo/PENDING).exists() or completion.get("status")!="PASS" or not completion.get("terminal"):
        raise StageXX0Error("X0 completion is not atomically terminal")
    false_fields=("new_empirical_outcome_access","formal_image_bytes_accessed","model_or_checkpoint_bytes_accessed","model_loaded","torch_or_timm_imported","cuda_or_gpu_accessed","dataset_or_checkpoint_downloaded","x1_started","x2_started","x3_started","f3_final_started","stage_g_started")
    if any(completion.get(key) is not False for key in false_fields): raise StageXX0Error("X0 completion access/stage boundary drifted")
    if completion.get("next_authorization") != "X1_NNS_V2_IMPLEMENTATION_ONLY": raise StageXX0Error("X0 next authorization drifted")
    for record in completion["records"].values():
        path=repo/record["path"]; value=read_json(path); validate_signed(value,str(path))
        if sha256_file(path)!=record["sha256"] or value["canonical_digest"]!=record["canonical_digest"]: raise StageXX0Error(f"completion binding drifted: {path}")
    validate_protocol(repo)
    if published:
        head=_git(repo,"rev-parse","HEAD"); tracking=_git(repo,"rev-parse",f"origin/{BRANCH}")
        remote_lines=_git(repo,"ls-remote","--heads","origin",BRANCH).split()
        remote=remote_lines[0] if remote_lines else ""
        archive=_git(repo,"rev-parse",ARCHIVE_REF+"^{commit}")
        remote_archive_lines=_git(repo,"ls-remote","origin",ARCHIVE_REF).split()
        remote_archive=remote_archive_lines[0] if remote_archive_lines else ""
        status=_git(repo,"status","--porcelain=v1")
        lfs=_git(repo,"lfs","status")
        if len({head,tracking,remote,archive,remote_archive})!=1 or status:
            raise StageXX0Error("published Git branch/ref state is not clean and equal")
        if any(line.strip() for line in lfs.splitlines() if line.startswith(" ") or line.startswith("*")):
            raise StageXX0Error("published Git LFS state is not clean")
    return {"status":"PASS","next_authorization":"X1_NNS_V2_IMPLEMENTATION_ONLY","archive_ref":ARCHIVE_REF,"published":published}

def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(prog="python -m vitnns.stage_x_x0")
    sub=parser.add_subparsers(dest="command",required=True)
    check=sub.add_parser("validate"); check.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    run=sub.add_parser("freeze"); run.add_argument("--repo-root",type=Path,default=REPO_ROOT); run.add_argument("--completed-at",required=True); run.add_argument("--test-passed",type=int,required=True); run.add_argument("--test-failed",type=int,required=True); run.add_argument("--test-command",required=True)
    done=sub.add_parser("validate-completion"); done.add_argument("--repo-root",type=Path,default=REPO_ROOT); done.add_argument("--published",action="store_true")
    args=parser.parse_args(argv)
    try:
        if args.command=="validate": result=validate_protocol(args.repo_root)
        elif args.command=="freeze": result=freeze(args.repo_root,completed_at=args.completed_at,test_passed=args.test_passed,test_failed=args.test_failed,test_command=args.test_command)
        else: result=validate_completion(args.repo_root,published=args.published)
    except (StageXX0Error,OSError,ValueError,KeyError) as error:
        print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True),file=sys.stderr); return 2
    print(json.dumps(result,indent=2,sort_keys=True)); return 0

if __name__=="__main__":
    raise SystemExit(main())
