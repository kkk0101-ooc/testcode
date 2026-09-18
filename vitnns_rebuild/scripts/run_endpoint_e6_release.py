#!/usr/bin/env python3
"""Execute frozen ENDPOINT-E6 statistics and paper release, CPU only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e6_release import (  # noqa: E402
    aggregate_decision, bootstrap_median_ci, decide_family, exact_two_sided_sign,
    holm_adjust, normalized_auc, spearman,
)

SPEC_PATH = Path("config/nns_vit_endpoint/endpoint_e6_release_run_spec.json")
SOURCE_CODE = [
    "config/nns_vit_endpoint/endpoint_e6_release_run_spec.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE6执行合同_20260918.md",
    "src/vitnns/endpoint_e6_release.py",
    "scripts/run_endpoint_e6_release.py",
    "scripts/validate_endpoint_e6_release.py",
    "tests/test_endpoint_e6_release.py",
]
REFERENCE_PATHS = {
    "cohort_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
    "statistical_analysis_plan_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json",
    "endpoint_e4_r2_artifact_manifest_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/artifact_manifest.json",
    "endpoint_e4_r3_artifact_manifest_sha256": "reports/nns_vit_endpoint_e4_intervention_r3_20260918/artifact_manifest.json",
    "endpoint_e4_r3_combined_index_sha256": "reports/nns_vit_endpoint_e4_intervention_r3_20260918/combined_result_index.json",
    "endpoint_e4_r3_terminal_sha256": "reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json",
    "endpoint_e4_r3_validation_sha256": "reports/nns_vit_endpoint_e4_intervention_r3_20260918/validation_report.json",
    "endpoint_e5_artifact_manifest_sha256": "reports/nns_vit_endpoint_e5_spatial_r1_20260918/artifact_manifest.json",
    "endpoint_e5_coverage_sha256": "reports/nns_vit_endpoint_e5_spatial_r1_20260918/formal_coverage_audit.json",
    "endpoint_e5_task_manifest_sha256": "reports/nns_vit_endpoint_e5_spatial_r1_20260918/task_manifest.json",
    "endpoint_e5_terminal_sha256": "reports/nns_vit_endpoint_e5_spatial_r1_20260918/terminal_receipt.json",
    "endpoint_e5_validation_sha256": "reports/nns_vit_endpoint_e5_spatial_r1_20260918/validation_report.json",
    "endpoint_ep1_r11_terminal_sha256": "reports/nns_vit_endpoint_ep1_r11_20260918/terminal_receipt.json",
    "endpoint_ep1_r11_validation_sha256": "reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json",
    "endpoint_e2_r4_terminal_sha256": "reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json",
    "endpoint_e2_r4_validation_sha256": "reports/nns_vit_endpoint_e2_atlas_r4_20260918/validation_report.json",
    "endpoint_e3_r2_terminal_sha256": "reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json",
    "endpoint_e3_r2_validation_sha256": "reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json",
    "endpoint_e4_r1_terminal_sha256": "reports/nns_vit_endpoint_e4_intervention_r1_20260918/terminal_receipt.json",
    "endpoint_e4_r2_terminal_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/terminal_receipt.json",
    "unified_plan_sha256": "docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md",
}
E4_R3 = Path("reports/nns_vit_endpoint_e4_intervention_r3_20260918")
E5 = Path("reports/nns_vit_endpoint_e5_spatial_r1_20260918")
PASS = "ENDPOINT_E6_PASS_STATISTICS_AND_PAPER_RELEASE_COMPLETE"
NEXT = "NONE_ENDPOINT_PRIMARY_TRACK_COMPLETE"
ALLOWED_STATUS = {"INVALID", "INCONCLUSIVE", "SUPPORTED", "NOT_SUPPORTED"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): value.update(chunk)
    return value.hexdigest()


def load(path: Path, audit: list[dict[str, Any]] | None = None) -> Any:
    resolved = path if path.is_absolute() else REPO / path
    if audit is not None:
        audit.append({"path": resolved.relative_to(REPO).as_posix(), "bytes": resolved.stat().st_size,
                      "sha256": file_sha256(resolved)})
    return json.loads(resolved.read_text())


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        os.write(descriptor, payload); os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, value.encode()); os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_spec(spec: dict[str, Any]) -> None:
    if spec["authorization_token"] != "ENDPOINT_E6_EXPLICIT_USER_REQUEST_20260918" or spec["stage"] != "ENDPOINT-E6":
        raise ValueError("E6_AUTHORIZATION")
    expected_execution = {"cpu_only": True, "cuda_visible_devices": "", "torch_timm_pil_import_allowed": False,
        "image_decode_allowed": False, "model_load_allowed": False, "new_empirical_compute_allowed": False,
        "source_recomputing_statistics": True, "output_root_must_not_exist": True}
    if spec["execution"] != expected_execution or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU_ONLY_BOUNDARY")
    if set(sys.modules).intersection({"torch", "timm", "PIL"}):
        raise ValueError("FORBIDDEN_IMPORT")
    if spec["statistics"] != {
        "unit": "image", "fixed_image_count": 400, "models": ["vit_b16", "deit_s16"],
        "references": ["normalized_zero", "blur_rgb"],
        "c4a_component_families": ["heads", "routing_value", "residual_branches"],
        "c4b_baselines": ["total_gxi", "raw_gradient", "regional_taylor", "random_regions"],
        "family_count": 16, "test_count": 28, "bootstrap_replicates": 10000,
        "bootstrap_namespace": "NNS_VIT_ENDPOINT_EP0_R1_20260918", "bootstrap_quantile_method": "linear",
        "sign_test_zero_threshold": 1e-12, "holm_alpha": .05,
        "decision_order": ["INVALID", "INCONCLUSIVE", "SUPPORTED", "NOT_SUPPORTED"]}:
        raise ValueError("E6_STATISTICAL_CONTRACT")


def verify_source_hashes(spec: dict[str, Any], audit: list[dict[str, Any]]) -> None:
    if set(spec["predecessors"]) != set(REFERENCE_PATHS):
        raise ValueError("PREDECESSOR_KEYS")
    for key, relative in REFERENCE_PATHS.items():
        path = REPO / relative; observed = file_sha256(path)
        audit.append({"path": relative, "bytes": path.stat().st_size, "sha256": observed})
        if observed != spec["predecessors"][key]: raise ValueError("PREDECESSOR_HASH:" + key)


def git_state() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    upstream = subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=REPO, text=True).strip()
    remote_line = subprocess.check_output(["git", "ls-remote", "origin", f"refs/heads/{branch}"], cwd=REPO, text=True).strip()
    remote = remote_line.split()[0] if remote_line else None
    if not head == upstream == remote: raise ValueError("PUBLISHED_HEAD_GATE")
    for relative in SOURCE_CODE:
        blob = subprocess.check_output(["git", "show", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO)
        if hashlib.sha256(blob).hexdigest() != file_sha256(REPO / relative): raise ValueError("SOURCE_NOT_PUBLISHED:" + relative)
    return {"branch": branch, "head": head, "upstream": upstream, "remote": remote}


def load_e4(audit: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    combined = load(E4_R3 / "combined_result_index.json", audit)
    r3_manifest = {row["path"]: row for row in load(E4_R3 / "artifact_manifest.json", audit)["files"]}
    r2_root = Path("reports/nns_vit_endpoint_e4_intervention_r2_20260918")
    r2_manifest = {row["path"]: row for row in load(r2_root / "artifact_manifest.json", audit)["files"]}
    rows = {}
    for record in combined["records"]:
        path = Path(record["source_result_path"]); document = load(path, audit)
        if path.is_relative_to(E4_R3): manifest_record = r3_manifest.get(path.relative_to(E4_R3).as_posix())
        elif path.is_relative_to(r2_root): manifest_record = r2_manifest.get(path.relative_to(r2_root).as_posix())
        else: raise ValueError("E4_RESULT_PATH")
        source_path = REPO / path
        if manifest_record is None or source_path.stat().st_size != manifest_record["bytes"] or file_sha256(source_path) != manifest_record["sha256"]:
            raise ValueError("E4_RESULT_MANIFEST")
        signed = dict(document); observed = signed.pop("canonical_digest")
        if digest(signed) != observed or observed != record["source_result_digest"]: raise ValueError("E4_RESULT_DIGEST")
        rows[record["task_id"]] = document["result"]
    if len(rows) != 800: raise ValueError("E4_COVERAGE")
    return rows


def load_e5(audit: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    manifest = load(E5 / "artifact_manifest.json", audit)
    records = {row["path"]: row for row in manifest["files"]}
    rows = {}
    for model in ("vit_b16", "deit_s16"):
        for order in range(400):
            relative = f"formal_results/attempts/{model}__{order:03d}/attempt-01.json"
            record = records.get(relative); path = E5 / relative
            if record is None or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
                raise ValueError("E5_RESULT_MANIFEST")
            document = load(path, audit); signed = dict(document); observed = signed.pop("canonical_digest")
            if digest(signed) != observed: raise ValueError("E5_RESULT_DIGEST")
            rows[document["task_id"]] = document["result"]
    if len(rows) != 800: raise ValueError("E5_COVERAGE")
    return rows


def recompute_c4a(result: dict[str, Any], reference: str, family: str) -> dict[str, Any]:
    components = [row for row in result["finite_utility"][reference]["components"] if row["family"] == family]
    metric = spearman([row["m"] for row in components], [abs(row["effect"]) for row in components])
    secondary = {
        "a_vs_effect": spearman([row["a"] for row in components], [row["effect"] for row in components]),
        "p_ref_vs_effect": spearman([row["p_ref"] for row in components], [row["effect"] for row in components]),
        "abs_p_ref_vs_absolute_effect": spearman([abs(row["p_ref"]) for row in components], [abs(row["effect"]) for row in components]),
        "output_norm_vs_absolute_effect": spearman([row["output_norm"] for row in components], [abs(row["effect"]) for row in components]),
        "gradient_norm_vs_absolute_effect": spearman([row["gradient_norm"] for row in components], [abs(row["effect"]) for row in components]),
        "abs_gradient_activation_vs_absolute_effect": spearman([abs(row["gradient_activation"]) for row in components], [abs(row["effect"]) for row in components]),
    }
    return {"primary": metric, "secondary": secondary, "component_count": len(components)}


def recompute_c4b(result: dict[str, Any], reference: str) -> dict[str, Any]:
    value = result["references"][reference]; heads = result["heads"]; curves = value["curves"]
    deterministic = {}
    for method in ("endpoint_component", "total_gxi", "raw_gradient", "regional_taylor"):
        scores = []
        for head in heads:
            matches = [row for row in curves if row["method"] == method and row["head"] == head]
            if len(matches) != 1: raise ValueError("C4B_DETERMINISTIC_CURVE")
            score = normalized_auc([point["isolated_effect"] for point in matches[0]["points"]]); scores.append(score)
        deterministic[method] = float(np.mean(scores))
    random_heads = []
    for head in heads:
        repeats = sorted((row for row in curves if row["method"] == "random_regions" and row["head"] == head), key=lambda row: row["repeat"])
        if len(repeats) != 20: raise ValueError("C4B_RANDOM_CURVE")
        mean_signed = [float(np.mean([row["points"][index]["isolated_effect"] for row in repeats])) for index in range(4)]
        random_heads.append(normalized_auc(mean_signed))
    scores = {**deterministic, "random_regions": float(np.mean(random_heads))}
    paired = {baseline: scores["endpoint_component"] - scores[baseline] for baseline in ("total_gxi", "raw_gradient", "regional_taylor", "random_regions")}
    return {"scores": scores, "paired": paired}


def build_inputs(e4: dict[str, dict[str, Any]], e5: dict[str, dict[str, Any]], plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    family_values = {row["family_id"]: [] for row in plan["families"]}
    c4b_columns: dict[str, dict[str, list[float]]] = {row["family_id"]: {test_id: [] for test_id in row["tests"]} for row in plan["families"] if row["claim"] == "C4b"}
    rows = []
    for model in ("vit_b16", "deit_s16"):
        for order in range(400):
            task_id = f"{model}__{order:03d}"; left=e4[task_id]; right=e5[task_id]
            if (left["sample_id"], left["split_order"], left["class_id"], left["image_sha256"]) != (right["sample_id"], right["split_order"], right["class_id"], right["image_sha256"]):
                raise ValueError("E4_E5_IDENTITY")
            for reference in ("normalized_zero", "blur_rgb"):
                c4b = recompute_c4b(right, reference)
                item = {"task_id":task_id,"model_id":model,"split_order":order,"sample_id":left["sample_id"],"reference_id":reference,"c4a":{},"c4b":c4b}
                for family in ("heads", "routing_value", "residual_branches"):
                    computed = recompute_c4a(left, reference, family); item["c4a"][family]=computed
                    if computed["primary"]["reason"] is None:
                        family_values[f"C4a.{model}.{reference}.{family}"].append(computed["primary"]["value"])
                family_id=f"C4b.{model}.{reference}.head_regions"
                for baseline,value in c4b["paired"].items(): c4b_columns[family_id][f"{family_id}.vs_{baseline}"].append(value)
                rows.append(item)
    if len(rows) != 1600: raise ValueError("PER_IMAGE_INPUT_COVERAGE")
    return rows, family_values, c4b_columns


def infer(plan: dict[str, Any], family_values: dict[str, list[float]], c4b_columns: dict[str, dict[str, list[float]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    plan_tests={row["test_id"]:row for row in plan["tests"]}; family_rows=[]; test_rows=[]
    for family in plan["families"]:
        family_id=family["family_id"]
        columns = {family_id: family_values[family_id]} if family["claim"]=="C4a" else c4b_columns[family_id]
        valid_sets={len(value) for value in columns.values()}; exact=len(selected)==400
        if len(valid_sets)!=1: raise ValueError("PAIRED_COMMON_SET")
        bootstrap=bootstrap_median_ci(columns,namespace="NNS_VIT_ENDPOINT_EP0_R1_20260918",family_id=family_id)
        tests=[]
        for test_id,values in columns.items():
            sign=exact_two_sided_sign(values); ci=bootstrap["results"][test_id]
            row={"test_id":test_id,"family_id":family_id,"claim":family["claim"],"model_id":family["model"],"reference_id":family["replacement"],
                 "component_family":family["component_family"],"baseline":test_id.split(".vs_",1)[1] if ".vs_" in test_id else None,
                 "estimate":ci["estimate"],"ci_lower":ci["ci_lower"],"ci_upper":ci["ci_upper"],"ci_width":ci["ci_upper"]-ci["ci_lower"],
                 "bootstrap_seed":bootstrap["seed"],"bootstrap_replicates":bootstrap["replicates"],"min_nonzero_n":plan_tests[test_id]["min_nonzero_n"],**sign}
            tests.append(row);test_rows.append(row)
        family_rows.append({"family_id":family_id,"claim":family["claim"],"model_id":family["model"],"reference_id":family["replacement"],
            "component_family":family["component_family"],"n_valid":next(iter(valid_sets)),"n_valid_min":family["n_valid_min"],"exact_coverage":exact,
            "delta_min":family["delta_min"],"ci_width_max":family["ci_width_max"],"tests":[row["test_id"] for row in tests]})
    holm=holm_adjust({row["test_id"]:row["raw_p_value"] for row in test_rows},plan["holm"]["test_ids"]);by_id={row["test_id"]:row for row in holm}
    for row in test_rows: row.update({"holm_adjusted_p_value":by_id[row["test_id"]]["adjusted_p_value"],"holm_reject":by_id[row["test_id"]]["reject_alpha_0_05"]})
    for family in family_rows:
        tests=[row for row in test_rows if row["family_id"]==family["family_id"]];status,reasons=decide_family(family,tests)
        family["status"]=status;family["decision_reasons"]=reasons;family["test_results"]=tests
    return family_rows,test_rows,holm


def render_figures(root: Path, families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output=[]
    for claim in ("C4a","C4b"):
        selected=[row for row in families if row["claim"]==claim]; tests=[test for row in selected for test in row["test_results"]]
        labels=[row["test_id"].replace(f"{claim}.","") for row in tests]
        low=min([0.0,*[row["ci_lower"] for row in tests]]);high=max([0.0,*[row["ci_upper"] for row in tests]])
        margin=max((high-low)*.08,.02);low-=margin;high+=margin;width=1200;left=570;right=1150;row_h=34;height=100+row_h*len(tests)
        def xpos(value: float) -> float: return left+(float(value)-low)/(high-low)*(right-left)
        title="C4a: component mass vs finite effect" if claim=="C4a" else "C4b: Endpoint region score minus mandatory baseline"
        parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',f'<text x="20" y="30" font-family="sans-serif" font-size="20">{title}</text>',
               f'<line x1="{xpos(0):.2f}" y1="55" x2="{xpos(0):.2f}" y2="{height-35}" stroke="#222" stroke-width="1"/>']
        for index,(label,row) in enumerate(zip(labels,tests)):
            y=65+index*row_h;color="#2878b5" if row["holm_reject"] and row["estimate"]>0 else "#888888"
            parts.extend([f'<text x="15" y="{y+5}" font-family="monospace" font-size="12">{label}</text>',
                f'<line x1="{xpos(row["ci_lower"]):.2f}" y1="{y}" x2="{xpos(row["ci_upper"]):.2f}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{xpos(row["estimate"]):.2f}" cy="{y}" r="5" fill="{color}"/>'])
        parts.append(f'<text x="{left}" y="{height-10}" font-family="sans-serif" font-size="13">Median image-level statistic (95% percentile bootstrap CI); zero line shown</text></svg>')
        path=root/f"figure_{claim.lower()}_forest.svg";write_text_new(path,"\n".join(parts)+"\n")
        output.append({"path":path.relative_to(root).as_posix(),"sha256":file_sha256(path),"bytes":path.stat().st_size})
    return output


def markdown_tables(families: list[dict[str, Any]]) -> tuple[str,str]:
    family_lines=["| Family | Estimate(s) | 95% CI(s) | Holm p(s) | Decision |","|---|---:|---:|---:|---|"]
    for row in families:
        tests=row["test_results"]; family_lines.append("| "+row["family_id"]+" | "+", ".join(f"{t['estimate']:.4f}" for t in tests)+" | "+", ".join(f"[{t['ci_lower']:.4f}, {t['ci_upper']:.4f}]" for t in tests)+" | "+", ".join(f"{t['holm_adjusted_p_value']:.4g}" for t in tests)+" | "+row["status"]+" |")
    return "\n".join(family_lines), "\n".join(family_lines).replace("Decision","判定").replace("Estimate(s)","估计值").replace("95% CI(s)","95% 区间")


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--repo-root",type=Path,default=REPO);args=parser.parse_args()
    if args.repo_root.resolve()!=REPO: raise ValueError("REPO_ROOT")
    started=time.monotonic(); audit=[]; spec=load(SPEC_PATH,audit);validate_spec(spec);verify_source_hashes(spec,audit);state=git_state()
    root=REPO/spec["outputs"]["formal_root"]
    if root.exists(): raise ValueError("OUTPUT_ROOT_EXISTS")
    root.mkdir(parents=True)
    plan=load(Path(spec["inputs"]["statistical_plan"]),audit)
    try:
        e4=load_e4(audit);e5=load_e5(audit);per_image,family_values,c4b_columns=build_inputs(e4,e5,plan)
        families,tests,holm=infer(plan,family_values,c4b_columns)
        if len(families)!=16 or len(tests)!=len(holm)!=28 or any(row["status"] not in ALLOWED_STATUS for row in families): raise ValueError("INFERENCE_COVERAGE")
        aggregate={claim:aggregate_decision([row["status"] for row in families if row["claim"]==claim]) for claim in ("C4a","C4b")}
        write_new(root/"per_image_statistics.json",{"stage":"ENDPOINT-E6","unit":"image","row_count":len(per_image),"rows_digest":digest(per_image),"rows":per_image})
        write_new(root/"bootstrap_report.json",{"stage":"ENDPOINT-E6","method":plan["bootstrap"],"family_count":16,"families":[{"family_id":r["family_id"],"tests":[{k:t[k] for k in ("test_id","estimate","ci_lower","ci_upper","ci_width","bootstrap_seed","bootstrap_replicates")} for t in r["test_results"]]} for r in families]})
        write_new(root/"sign_test_report.json",{"stage":"ENDPOINT-E6","method":plan["sign_test"],"test_count":28,"tests":[{k:r[k] for k in ("test_id","n_total","n_nonzero","n_positive","n_negative","n_tie","raw_p_value")} for r in tests]})
        write_new(root/"holm_report.json",{"stage":"ENDPOINT-E6","method":plan["holm"],"test_count":28,"tests":holm})
        write_new(root/"family_decisions.json",{"stage":"ENDPOINT-E6","decision_order":plan["decision_order"],"family_count":16,"families":families,"aggregate_claims":aggregate})
        identity_correctness={"Identity":{"status":"ESTABLISHED_IMPLEMENTATION_IDENTITY","statement":"For the same scalar target at the input endpoint, the base Endpoint map is Gradient multiplied by Input.","source":"EP1-R11"},
          "Correctness":{"status":"SUPPORTED","closure":{"positive_checks":76800,"negative_controls":28800,"maximum_relative_l2":1.4144569831842091e-05,"source":"ENDPOINT-E3-R2"},"local_jvp":{"checks":6144,"maximum_absolute_error":2.648076753197781e-05,"maximum_error_over_absolute_dot_mass":6.948755071487438e-06,"source":"ENDPOINT-E4-R3"}},
          "Utility":{"C4a":aggregate["C4a"],"C4b":aggregate["C4b"],"source":"ENDPOINT-E6 frozen inference"},
          "Path_IG":{"status":"EP2_R1_PATH_NUMERICAL_CALIBRATION_FAILED_IMMUTABLE","statement":"No cross-model Endpoint-Path fidelity, stability, or cost-quality conclusion is released."}}
        write_new(root/"claim_decisions.json",identity_correctness)
        figure_records=render_figures(root,families);table_en,table_zh=markdown_tables(families)
        en=rf"""# Endpoint Scanner E6 results release\n\n## Identity, correctness, and utility\n\nFor a fixed scalar target at the input endpoint, the base Endpoint map is $x \odot \nabla_x s(x)$, the same numerical object as Gradient × Input. The registered implementation scanned 400 shared evaluation images for each of ViT-B/16 and DeiT-S/16. Structural cuts passed 76,800 closure checks and 28,800 omission/duplication controls; the maximum relative L2 error was 1.41446e-5. Under the E4-R3 condition-scaled contract, 6,144 analytic JVP checks passed.\n\nThe confirmatory finite-intervention decisions are **C4a {aggregate['C4a']}** and **C4b {aggregate['C4b']}**. C4a measures image-level associations between component absolute mass and finite activation-replacement magnitude. C4b measures the image-level spatial-score gain over each mandatory baseline. The table reports all 16 registered families and all 28 globally Holm-adjusted tests.\n\n{table_en}\n\n## Interpretation boundary\n\nThe utility conclusions above follow the registered two-model, two-reference evaluation. Endpoint base attribution is not a new formula distinct from Gradient × Input. Structural closure and local JVP agreement establish correctness; finite replacement and spatial intervention determine utility. Path IG calibration failed for ViT-B/16 under the frozen candidate range, so the release contains no cross-model Endpoint–Path claim.\n"""
        zh=rf"""# Endpoint Scanner E6 结果发布\n\n## Identity、Correctness 与 Utility\n\n对同一输入端标量目标，基础 Endpoint 图为 $x \odot \nabla_x s(x)$，数值上就是 Gradient × Input。注册实现对 ViT-B/16 和 DeiT-S/16 各自扫描同一组 400 张 evaluation 图。结构切面通过 76,800 项闭合检查和 28,800 项遗漏/重复负例，最大 relative L2 为 1.41446e-5；E4-R3 条件尺度合同下的 6,144 项解析 JVP 检查全部通过。\n\n确认性有限干预判定为：**C4a {aggregate['C4a']}**、**C4b {aggregate['C4b']}**。C4a 检验成分 absolute mass 与有限 activation-replacement 幅度的图像级关联；C4b 检验 Endpoint 空间排序相对每个 mandatory baseline 的图像级增量。下表完整报告 16 个注册 family 和统一 Holm 校正的 28 个检验。\n\n{table_zh}\n\n## 解释边界\n\n上述效用结论限定在注册的双模型、双 reference 评价中。基础 Endpoint 归因不是不同于 Gradient × Input 的新公式。结构闭合和局部 JVP 支持 correctness；有限 replacement 和空间干预决定 utility。Path IG 在 ViT-B/16 上未通过冻结候选范围的数值校准，因此本文不发布跨模型 Endpoint–Path 结论。\n"""
        write_text_new(REPO/spec["outputs"]["paper_release_en"],en);write_text_new(REPO/spec["outputs"]["paper_release_zh"],zh)
        evidence="# Endpoint Scanner E6 claim-evidence matrix\n\n| Claim | Evidence | Decision | Boundary |\n|---|---|---|---|\n"+"\n".join([
          "| Base Endpoint identity | EP1-R11 implementation/regression | ESTABLISHED_IMPLEMENTATION_IDENTITY | Same scalar target and normalized input coordinates |",
          "| Structured cut correctness | E3-R2: 76,800 positive closure checks; 28,800 negative controls | SUPPORTED | Correctness, not utility |",
          "| Local directional correctness | E4-R3: 6,144 condition-scaled analytic JVP checks | SUPPORTED | Local differential result |",
          f"| C4a finite-replacement association | E6 12 families and 12 Holm tests | {aggregate['C4a']} | All registered families required |",
          f"| C4b spatial information gain | E6 4 families and 16 mandatory-baseline Holm tests | {aggregate['C4b']} | All four baselines in every family required |",
          "| Endpoint-Path comparison | EP2-R1 | NOT RELEASED | ViT-B/16 calibration failed under the frozen candidate range |",
        ])+"\n"
        write_text_new(REPO/spec["outputs"]["evidence_matrix"],evidence)
        defensive="""# Endpoint Scanner E6 defensive-writing audit\n\n| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |\n|---|---|---|---|---|---|---|---|---|\n| E6 interpretation boundary | Endpoint base attribution is not a new formula distinct from Gradient × Input. | D14 | Minor | Yes | It could read as novelty prebuttal if repeated. | Clear once; distracting if repeated. | Preserve once as the identity result. | The base Endpoint map equals Gradient × Input for the same scalar target. |\n| E6 Path boundary | Path IG calibration failed for ViT-B/16 under the frozen candidate range, so the release contains no cross-model Endpoint–Path claim. | CLEAN | None | Yes | It states the direct failed gate and consequence. | The scope follows the registered evidence. | Preserve once. | No change. |\n\n### Paper-level defensive-writing score\n\n2/10. The release leads with measured results and keeps one concise boundary per claim.\n\n### Highest-priority removals\n\n- Do not repeat the Gradient × Input novelty boundary outside Identity and the interpretation boundary.\n- Do not excuse any NOT_SUPPORTED family or infer causes without a registered diagnostic.\n- Do not convert E3/E4 correctness into utility language.\n\n### Necessary caveats to preserve\n\n- Endpoint base attribution is Gradient × Input for the same target.\n- E3 closure and E4-R3 JVP are correctness evidence.\n- Path IG cross-model claims remain blocked by the immutable EP2-R1 calibration failure.\n\n### Repeated phrase audit\n\nThe release uses no reviewer-facing prebuttal phrase and states each material boundary at most twice across the result and interpretation paragraphs.\n"""
        write_text_new(REPO/spec["outputs"]["defensive_writing_audit"],defensive)
        read_unique={row["path"]:row for row in audit};write_new(root/"source_read_audit.json",{"stage":"ENDPOINT-E6","cpu_only":True,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"forbidden_imports_loaded":sorted(set(sys.modules).intersection({"torch","timm","PIL"})),"image_files_read":0,"model_files_read":0,"read_count":len(read_unique),"reads":sorted(read_unique.values(),key=lambda row:row["path"])})
        release_files=[spec["outputs"][key] for key in ("paper_release_en","paper_release_zh","evidence_matrix","defensive_writing_audit")];write_new(root/"reproduction_manifest.json",{"command":"CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python scripts/run_endpoint_e6_release.py","producer_commit":state["head"],"source_code":[{"path":p,"sha256":file_sha256(REPO/p)} for p in SOURCE_CODE],"inputs":[{"path":p,"sha256":spec["predecessors"][k]} for k,p in REFERENCE_PATHS.items()],"paper_release_files":[{"path":p,"bytes":(REPO/p).stat().st_size,"sha256":file_sha256(REPO/p)} for p in release_files],"figures":figure_records})
        elapsed=time.monotonic()-started;report={"status":PASS,"stage":"ENDPOINT-E6","producer_commit":state["head"],"cpu_only":True,"gpu_hours":0,"new_empirical_compute":False,"image_decode_count":0,"model_load_count":0,"evaluation_images":400,"image_model_tasks_per_source":800,"per_image_reference_rows":1600,"family_count":16,"test_count":28,"holm_family_count":28,"bootstrap_replicates_per_family":10000,"family_status_counts":{s:sum(r["status"]==s for r in families) for s in sorted(ALLOWED_STATUS)},"aggregate_claims":aggregate,"identity_correctness_utility":identity_correctness,"source_read_count":len(read_unique),"wall_seconds":elapsed,"peak_cpu_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"paper_release_files":release_files,"figure_files":[r["path"] for r in figure_records],"path_track_status":"EP2_R1_PATH_NUMERICAL_CALIBRATION_FAILED_IMMUTABLE","next_authorization":NEXT}
        write_new(root/"validation_report.json",report)
        files=[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":file_sha256(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.name not in {"artifact_manifest.json","terminal_receipt.json"}]
        manifest={"files":files,"canonical_digest":digest(files)};write_new(root/"artifact_manifest.json",manifest)
        terminal={"status":PASS,"stage":"ENDPOINT-E6","producer_commit":state["head"],"evaluation_image_count":400,"family_count":16,"test_count":28,"C4a":aggregate["C4a"],"C4b":aggregate["C4b"],"validation_digest":digest(report),"manifest_digest":manifest["canonical_digest"],"next_authorization":NEXT}
        write_new(root/"terminal_receipt.pending.json",terminal);os.replace(root/"terminal_receipt.pending.json",root/"terminal_receipt.json");print(json.dumps(terminal,sort_keys=True));return 0
    except Exception as error:
        failure={"status":"ENDPOINT_E6_FAIL_CLOSED","stage":"ENDPOINT-E6","error_type":type(error).__name__,"error":str(error),"next_authorization":"NONE"}
        if root.exists() and not (root/"terminal_receipt.json").exists(): write_new(root/"failure_receipt.json",failure)
        raise


if __name__=="__main__": raise SystemExit(main())
