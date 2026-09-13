#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,py_compile,tempfile
from pathlib import Path
from vitnns import stage_y_y4_r4 as y
def tests(repo):
 class Results:
  def __init__(self):self.rows=[]
  def pytest_runtest_logreport(self,report):
   if report.when=="call" and report.passed:self.rows.append(report.nodeid)
 result=Results()
 import pytest
 rc=pytest.main([str(repo/"tests/test_stage_y_y4_r4.py"),str(repo/"tests/test_stage_y_y4_r3.py"),str(repo/"tests/test_stage_y_y4_r1.py"),"-q","-p","no:cacheprovider","--import-mode=importlib"],plugins=[result]);y.require(rc==0,"focused tests failed",y.StartupError);return len(result.rows)
def preflight(repo,output):
 gate=y.startup(repo,True);y.write_schema_sources(repo);y.validate_schema_sources(repo);count=tests(repo)
 with tempfile.TemporaryDirectory(prefix="y4-r4-pyc-") as tmp:
  for i,p in enumerate(y.PRODUCER_FILES):
   if p.suffix==".py":py_compile.compile(str(repo/p),cfile=f"{tmp}/{i}.pyc",doraise=True)
 payload={"startup_commit":y.STARTUP_COMMIT,"checks":["Y4 through Y4-R3 immutable","sealed R3 cut retained","registered zero-mass NA propagation","missing intervention endpoint-level NA","invalid endpoint retained in Holm","complete descriptive tables","producer and validator access/resource ledgers","all prior outcome-blind gates inherited"],"test_count":count,"false_pass_attack_count":len(y.FALSE_PASS_ATTACKS),"startup_refs_digest":gate["startup_refs_digest"],"formal_outcome_access":False};output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(y.canonical_bytes(payload)+b"\n");return {"status":"PASS",**payload}
def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("preflight","seal-producer","analyze","seal","validate","validate-publication"));p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--output",type=Path);p.add_argument("--preflight",type=Path);a=p.parse_args();repo=a.repo_root.resolve();os.environ.setdefault("CUDA_VISIBLE_DEVICES","");y.runtime_guard()
 try:
  if a.command=="preflight":out=preflight(repo,a.output)
  elif a.command=="seal-producer":out=y.seal_producer(repo,json.loads(a.preflight.read_text()))
  else:
   from scripts import validate_stage_y_y4_r4 as v
   out=v.produce(repo) if a.command=="analyze" else v.seal(repo) if a.command=="seal" else v.validate_completion(repo,a.command=="validate-publication")
 except Exception as e:print(json.dumps({"status":getattr(e,"code","BLOCKED_INFRASTRUCTURE"),"error":str(e),"next_authorization":"NONE"},sort_keys=True));return 2
 print(json.dumps(out,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
