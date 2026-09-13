from __future__ import annotations
import importlib.util,json
from pathlib import Path
from vitnns import stage_y_y3_r6 as r6
REPO=Path(__file__).resolve().parents[1]
def validator():
 s=importlib.util.spec_from_file_location("v6",REPO/"scripts/validate_stage_y_y3_r6.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_zero_compute_contract():
 s=r6.read(REPO/r6.SPEC);assert s["gpu_execution"] is False and s["new_empirical_attempt_count"]==0 and s["arrays_copied"] is False
def test_source_counts():assert r6.read(REPO/r6.SPEC)["r4_valid_rebind_count"]==155 and r6.read(REPO/r6.SPEC)["r5_valid_rebind_count"]==5
def test_keyed_comparison():assert r6.read(REPO/r6.SPEC)["comparison_key"]=="unit_index"
def test_r5_sources_are_five_valid():
 import glob
 docs=[json.load(open(p)) for p in glob.glob(str(REPO/r6.r5.ROOT/"shards/shard-01/units/unit-*/completion/unit.json"))];assert len(docs)==5 and all(x["status"]=="VALID" for x in docs)
def test_schema_closed():
 def walk(v):
  if isinstance(v,dict):
   if v.get("type")=="object":assert v.get("additionalProperties") is False
   for x in v.values():walk(x)
  elif isinstance(v,list):
   for x in v:walk(x)
 walk(r6.artifact_schema())
def test_false_pass_audit():a=validator().false_pass_audit();assert a["attack_count"]==a["rejected_count"]>=15
def test_no_worker_file():assert not (REPO/"src/vitnns/stage_y_y3_r6_worker.py").exists()
def test_producer_files_exclude_source_roots():assert all(not str(x).startswith("evidence/") for x in r6.PRODUCER_FILES)
