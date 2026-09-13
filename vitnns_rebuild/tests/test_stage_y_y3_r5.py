from __future__ import annotations
import importlib.util,os,subprocess
from pathlib import Path
from jsonschema import ValidationError
import pytest
from vitnns import stage_y_y3 as y3
from vitnns import stage_y_y3_r5 as r5
REPO=Path(__file__).resolve().parents[1]
def runner():
 s=importlib.util.spec_from_file_location("run_r5",REPO/"scripts/run_stage_y_y3_r5.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def validator():
 s=importlib.util.spec_from_file_location("validate_r5",REPO/"scripts/validate_stage_y_y3_r5.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_exact_recovery_roster():assert r5.NEW_INDICES==(141,145,149,153,157) and all(x%4==1 for x in r5.NEW_INDICES)
def test_source_has_155_valid_and_five_budget_terminals():
 docs=[y3.read(p) for p in (REPO/r5.SOURCE_ROOT).glob("shards/shard-*/units/unit-*/completion/unit.json")];assert len(docs)==160;assert sum(x["status"]=="VALID" for x in docs)==155;assert sorted(x["payload"]["unit_index"] for x in docs if x["status"]=="BLOCKED_BUDGET")==list(r5.NEW_INDICES)
def test_run_spec_fixed_gpu1_and_global_gate():
 s=r5.read(REPO/r5.SPEC);assert s["schedule"]["physical_indices"]==[1] and s["schedule"]["logical_shard_id"]==1;assert s["budget_gate"]["divide_by_worker_count"] is False and s["new_compute_count"]==5 and s["arrays_copied"] is False
def test_budget_projection_within_unchanged_caps():
 _,maximum,projected=runner().budget_admission(REPO);assert projected["gpu_seconds"]<102.4*3600 and projected["cpu_seconds"]<349*3600 and projected["retained_bytes"]<9*2**30;assert maximum["gpu_active_seconds"]>0
def test_schema_closed_and_spec_valid():
 r=runner();r.validate_schema_files(REPO)
 def walk(v):
  if isinstance(v,dict):
   if v.get("type")=="object":assert v.get("additionalProperties") is False
   for x in v.values():walk(x)
  elif isinstance(v,list):
   for x in v:walk(x)
 walk(r5.artifact_schema())
def test_false_pass_audit():
 a=validator().false_pass_audit();assert a["attack_count"]==a["rejected_count"]>=20
def test_cublas_guard():
 env=dict(os.environ);env.pop("CUBLAS_WORKSPACE_CONFIG",None);env["PYTHONPATH"]=str(REPO/"src");p=subprocess.run([str(y3.PYTHON),"-c","import vitnns.stage_y_y3_r5_worker"],cwd=REPO,env=env,text=True,capture_output=True);assert p.returncode!=0 and "before Torch import" in p.stderr
def test_probe_compiles_without_nul():
 source=runner().probe_script();assert "\0" not in source;compile(source,"<r5-probe>","exec")
def test_worker_uses_global_maximum_reserve():
 source=(REPO/"src/vitnns/stage_y_y3_r5_worker.py").read_text();assert "combined_gpu+maximum" in source and "/k" not in source and "/ 4" not in source
def test_no_r4_files_in_producer_set():assert all(not str(x).startswith(str(r5.SOURCE_ROOT)) for x in r5.PRODUCER_FILES)
def test_atomic_completion(tmp_path):
 p=tmp_path/"completion/pending.json";r5.write(p,"pending",{"started_unix_ns":1,"next_authorization":"NONE_WHILE_PENDING"},"PENDING");value=r5.signed("failure",{"phase":"x","failure_type":"BLOCKED_BUDGET","message":"x","next_authorization":"NONE"},"BLOCKED_BUDGET");r5.publish_terminal(p,p.with_name("terminal.json"),value);assert not p.exists()
def test_schema_rejects_extra_payload():
 d=r5.signed("pending",{"started_unix_ns":1,"next_authorization":"NONE_WHILE_PENDING"},"PENDING")
 d["payload"]["extra"]=1
 d["canonical_digest"]=r5.canonical({k:v for k,v in d.items() if k!="canonical_digest"})
 with pytest.raises(ValidationError):r5.validate_document(d)
