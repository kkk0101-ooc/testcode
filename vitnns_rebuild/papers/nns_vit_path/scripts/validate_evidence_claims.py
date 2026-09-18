#!/usr/bin/env python3
"""Recompute manuscript headline facts from frozen machine-readable evidence."""

import json
import math
import statistics
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
OUT = PAPER / "qc/evidence_validation_report.json"
checks = []


def load(path: str):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def check(name: str, observed, expected, tol: float = 0.0):
    if isinstance(expected, float):
        passed = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=tol)
    else:
        passed = observed == expected
    checks.append({"check": name, "observed": observed, "expected": expected, "pass": passed})


d10 = load("evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/analysis.json")
summary = d10["summary"]
check("stage_d_n", summary["n"], 500)
check("stage_d_positive_count", summary["positive_count"], 500)
check("stage_d_mean", summary["mean"], 2.630671903027281, 1e-15)
check("stage_d_median", summary["median"], 2.3081996221714247, 1e-15)
check("stage_d_ci_lower", summary["bootstrap_ci_lower"], 2.1110806126093067, 1e-15)
check("stage_d_ci_upper", summary["bootstrap_ci_upper"], 2.445235411755993, 1e-15)
check("c10_overall_decision", d10["c10_reference"]["overall_decision"], "NO_GO_NOVELTY")
check("c10_spatial_status", d10["c10_reference"]["spatial_status"], "FAIL_SPATIAL")

patch_rows = load("evidence/stage_e/e4_recovery_v2_20260907/formal/paper_data/patch_scale_sensitivity.json")["rows"]
native = [r["delta_fid"] for r in patch_rows if r.get("variant_id") == "nns_native_14x14_patch" and r.get("validity_status") == "VALID"]
check("native_patch_valid_n", len(native), 500)
check("native_patch_median", statistics.median(native), 0.0)
check("native_patch_zero_count", sum(value == 0.0 for value in native), 400)
check("native_patch_max_abs", max(map(abs, native)), 0.001379888674958342, 1e-18)

coordinate_rows = load("evidence/stage_e/e4_recovery_v2_20260907/formal/paper_data/layer_head_descriptive.json")["rows"]
for layer, head, n, median in ((0, 9, 500, 2.3081996221714247), (4, 2, 500, 0.07336550959657523), (6, 10, 500, 0.038911060704660244), (11, 2, 498, 0.056172238568077015)):
    values = [r["delta_fid"] for r in coordinate_rows if (r["layer"], r["head"]) == (layer, head) and r["validity_status"] == "VALID"]
    check(f"coordinate_{layer}_{head}_n", len(values), n)
    check(f"coordinate_{layer}_{head}_median", statistics.median(values), median, 1e-15)

contrasts = {r["contrast_id"]: r for r in load("evidence/stage_f/f3c4_analysis_r1_20260908/formal/paper_data/contrast_summary.json")["rows"]}
for key, mean in (("class_targeted_nns_minus_vit_grad_cam", -0.05585432814553341), ("class_targeted_nns_minus_full_model_ig", 0.005579104576659729), ("class_targeted_nns_minus_attention_rollout", 0.03943182220568996), ("class_targeted_nns_minus_raw_cls_attention", 0.011264721309726382)):
    check(f"contrast_{key}_n", contrasts[key]["n"], 50)
    check(f"contrast_{key}_mean", contrasts[key]["mean"], mean, 1e-15)

identity = load("evidence/stage_x/x1_r1_nns_v2_20260908/formal/identity_control_validation.json")
check("complete_total_identity_max_abs", identity["payload"]["max_abs_error"], 0.0)
cuts = load("evidence/stage_x/x1_r1_nns_v2_20260908/formal/complete_cut_validation.json")["payload"]
check("complete_cut_count", cuts["passed_cuts"], 135)
check("complete_component_count", cuts["component_count"], 38107)
check("complete_cut_max_pixel_error", max(record["pixel_max_abs_error"] for record in cuts["per_cut"].values()), 1.564621925354004e-06, 1e-18)

y4 = {r["hypothesis_id"]: r for r in load("evidence/stage_y/y4_r5_mechanism_analysis_20260914/formal/hypothesis_test_table.json")["payload"]["records"]}
for key, estimate, outcome in (("B4", 0.13910861073660347, "SUPPORTED"), ("B5", -0.24922637255981336, "SUPPORTED")):
    check(f"stage_y_{key}_estimate", y4[key]["estimate"], estimate, 1e-15)
    check(f"stage_y_{key}_outcome", y4[key]["outcome"], outcome)

r10 = load("reports/nns_vit_unified_paper_figures_r10_r1_20260917/module_compute_receipt.json")
check("r10_second_residual_max_abs", r10["second_residual_additivity_max_abs"], 3.3306690738754696e-16, 1e-30)
check("r10_classifier_order", r10["selected_orders"]["classifier"], 4096)
check("r10_classifier_completeness_abs", r10["classifier_attempts"][-1]["completeness_abs"], 0.004874336722195949, 1e-15)

s5 = load("evidence/nns_vit_patch_depth/s5_r1_posthoc_exploratory_20260916/endpoint_decisions.json")["payload"]["endpoints"]
for key, estimate, n in (("P1", 0.06260031101967031, 235), ("P2", 0.07493962390638756, 224), ("D1", 0.03631419994377305, 235), ("D2", 0.04615150420306633, 3424)):
    check(f"stage_s_{key}_estimate", s5[key]["estimate"], estimate, 1e-15)
    check(f"stage_s_{key}_n_valid", s5[key]["n_valid"], n)

errors = [item for item in checks if not item["pass"]]
report = {"status": "PASS" if not errors else "FAIL", "check_count": len(checks), "errors": errors, "checks": checks}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": report["status"], "check_count": len(checks), "error_count": len(errors)}, ensure_ascii=False))
raise SystemExit(0 if not errors else 1)
