from __future__ import annotations
import ast,importlib.util,json,subprocess,sys
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator
from vitnns import stage_y_y2 as y2
from vitnns import stage_y_y2_r2 as r2
REPO=Path(__file__).resolve().parents[1]

def _validator():
 s=importlib.util.spec_from_file_location("r2_validator_test",REPO/"scripts/validate_stage_y_y2_r2.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_run_spec_and_artifact_schemas_are_valid_and_closed():
 spec=json.loads((REPO/r2.RUN_SPEC).read_text());schema=json.loads((REPO/"config/stage_y/y2_r2_run_spec.schema.json").read_text());artifact=json.loads((REPO/"config/stage_y/y2_r2_artifact.schema.json").read_text());Draft202012Validator.check_schema(schema);Draft202012Validator.check_schema(artifact);Draft202012Validator(schema).validate(spec);assert artifact["properties"]["payload"]["additionalProperties"] is False and artifact["properties"]["payload"]["properties"]

def test_exact_source_partition_is_84_plus_3_plus_13():
 terms=r2.terminal_by_order(REPO);valid={o for o,(_,d) in terms.items() if d["status"]=="VALID"};completed={o for o,(_,d) in terms.items() if d["status"]=="BLOCKED_BUDGET" and len(d["payload"]["array_records"])==262};missing={o for o,(_,d) in terms.items() if d["status"]=="BLOCKED_BUDGET" and not d["payload"]["array_records"]};assert len(valid)==84 and completed==set(r2.COMPLETED_BUDGET_ORDERS) and missing==set(r2.MISSING_ORDERS) and len(valid|completed|missing)==100

def test_missing_units_have_no_scientific_work_and_exact_failure_code():
 rows=r2.validate_missing_sources(REPO);assert len(rows)==13
 for row in rows:
  source=json.loads((REPO/row["source_terminal"]["path"]).read_text());assert source["payload"]["failure_code"]=="per-shard resource cap already exhausted";assert source["payload"]["scientific_record"] is None and source["payload"]["array_records"]==[] and source["payload"]["quadrature_attempts"]==[]

def test_completed_budget_units_have_complete_science_candidate_state():
 terms=r2.terminal_by_order(REPO)
 for order,steps in ((73,1024),(80,1024),(86,512)):
  q=terms[order][1]["payload"];assert len(q["array_records"])==262 and q["quadrature_attempts"][-1]["steps"]==steps and q["quadrature_attempts"][-1]["all_contracts_pass"] is True and q["quadrature_attempts"][-1]["closure_status"]=="PASS"

def test_array_contract_has_exact_262_names():
 assert len(r2.EXPECTED_ARRAY_NAMES)==262 and len(set(r2.EXPECTED_ARRAY_NAMES))==262
 assert r2.expected_axes("input_attribution",(3,224,224))==["rgb","pixel_y","pixel_x"]

def test_worker_contains_no_budget_stop_or_relabel():
 source=(REPO/"src/vitnns/stage_y_y2_r2_worker.py").read_text();tree=ast.parse(source);assert "BLOCKED_BUDGET" not in source and "cap_h20" not in source and "cap_cpu" not in source and "cap_worker" not in source;assert "CUBLAS_WORKSPACE_CONFIG must equal :4096:8 before Torch initialization" in source;guard=min(n.lineno for n in ast.walk(tree) if isinstance(n,ast.If) and "CUBLAS_WORKSPACE_CONFIG" in ast.unparse(n.test));execute=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="execute");imp=min(n.lineno for n in ast.walk(execute) if isinstance(n,ast.Import) and any(a.name=="torch" for a in n.names));assert guard<imp

def test_worker_only_uses_missing_manifest_and_single_attempt():
 source=(REPO/"src/vitnns/stage_y_y2_r2_worker.py").read_text();assert "attempt-001.json" in source and '"retry_index":1' in source;assert "for order in manifest" in source and "dynamic_rebalancing" not in source

def test_missing_modulo_sharding_exact_for_one_to_eight_devices():
 for k in range(1,9):
  parts=[[o for i,o in enumerate(r2.MISSING_ORDERS) if i%k==s] for s in range(k)];assert sorted(o for p in parts for o in p)==list(r2.MISSING_ORDERS) and len({o for p in parts for o in p})==13

def test_resource_semantics_are_reporting_only():
 spec=json.loads((REPO/r2.RUN_SPEC).read_text());assert spec["resources"]["semantics"]=="REPORTING_ONLY" and spec["resources"]["old_caps_are_stop_gates"] is False and spec["resources"]["download_authorized"] is False

def test_rebind_does_not_copy_arrays_or_run_gpu():
 source=(REPO/"scripts/run_stage_y_y2_r2.py").read_text();section=source.split("def build_rebinds",1)[1].split("def _gpu_rows",1)[0];assert "shutil" not in section and "torch" not in section and "nvidia-smi" not in section

def test_atomic_terminal_primitive_uses_fsync_and_replace():
 source=(REPO/"src/vitnns/stage_y_y2.py").read_text();section=source.split("def publish_terminal",1)[1].split("def whole_root_inventory",1)[0];assert "os.fsync" in section and "os.replace" in section

def test_false_pass_audit_rejects_all_attacks():
 result=_validator().false_pass_audit(REPO,producer=True);assert result["status"]=="PASS" and result["attack_count"]==result["rejected_count"]>=22

def test_validator_is_source_recomputing_and_no_statistics():
 source=(REPO/"scripts/validate_stage_y_y2_r2.py").read_text();tree=ast.parse(source);calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)];assert "validate_array_record" in source and "validate_closure" in source and not any(name.endswith("bootstrap") or name.endswith("multipletests") or name.endswith("holm") for name in calls)
