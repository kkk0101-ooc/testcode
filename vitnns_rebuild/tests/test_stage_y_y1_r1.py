from __future__ import annotations
import ast,json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
import importlib.util
SPEC=importlib.util.spec_from_file_location("y1r1",ROOT/"scripts/validate_stage_y_y1_r1.py");y1r1=importlib.util.module_from_spec(SPEC);assert SPEC.loader;SPEC.loader.exec_module(y1r1)

def test_config_exact_waiver_scope_and_no_future_or_scientific_waiver():
 d=json.loads((ROOT/y1r1.CONFIG).read_text());assert d["waiver_scope"]=="HISTORICAL_PHASE_LEVEL_TELEMETRY_ONLY";assert d["affected_attempts"]==[x.name for x in y1r1.ROOTS[:2]]
 for k in ("overall_resource_telemetry_waived","future_telemetry_waived","scientific_gate_waived","method_or_scope_changed","rerun_authorized","formal_cohort_accessed"):assert d[k] is False
 assert d["cumulative_resource_accounting_required"] is d["conservative_maximum_projection_required"] is True

def test_original_y1_binding_is_immutable():
 d=y1r1.original();assert d["completion"]["sha256"]==y1r1.EXPECTED_SHA["completion"];assert d["report"]["sha256"]==y1r1.EXPECTED_SHA["report"]

def test_cpu_only_static_surface():
 d=y1r1.source_isolation();assert d["cpu_only"] and d["model_checkpoint_image_cuda_access"]==0
 tree=ast.parse((ROOT/"scripts/validate_stage_y_y1_r1.py").read_text());names={a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names};assert not names&y1r1.FORBIDDEN_IMPORTS

def test_three_workloads_rehash_and_recompute_all_closures():
 rows=y1r1.load_workloads();assert len(rows)==3;assert all(x["array_file_count"]==1048 for x in rows);assert all(len(x["source_recomputed_closure"])==4 for x in rows);assert all(y["status"]=="PASS" for x in rows for y in x["source_recomputed_closure"]);assert rows[0]["phase_telemetry_status"]==rows[1]["phase_telemetry_status"]=="NOT_RECOVERABLE_HISTORICAL_PHASE_TELEMETRY";assert rows[2]["phase_telemetry_status"]=="COMPLETE"

def test_cumulative_lower_bound_and_conservative_budget_exact():
 workloads=y1r1.load_workloads();c,b=y1r1.recompute(workloads);expected=json.loads((ROOT/y1r1.CONFIG).read_text());assert c["accounting_name"]=="MEASURED_RETAINED_ATTEMPT_LOWER_BOUND" and c["not_complete_historical_total"] is True
 for k,v in expected["expected_cumulative_measured_lower_bound"].items():assert c[k]==pytest.approx(v,abs=1e-12)
 assert b["safety_multiplier"]==1.5;assert b["y2"]["proposed_caps"]==expected["expected_proposed_caps"]["y2"];assert b["y3"]["proposed_caps"]==expected["expected_proposed_caps"]["y3"]

def test_false_pass_mutations_all_rejected():
 d=y1r1.mutations();assert d["attack_count"]==d["rejected_count"]==17;assert all(x["observed"]=="REJECTED" and x["reasons"]==[x["attack"]] for x in d["checks"] )

def test_unrecoverable_marker_is_not_numeric_or_null():
 d=json.loads((ROOT/y1r1.CONFIG).read_text());assert d["unavailable_value"]=="NOT_RECOVERABLE_HISTORICAL_PHASE_TELEMETRY";assert d["unavailable_value"] not in (0,None,False)
