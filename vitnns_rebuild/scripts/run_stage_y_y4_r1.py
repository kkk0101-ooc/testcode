#!/usr/bin/env python3
"""Y4-R1 CPU-only producer and publication orchestrator."""
from __future__ import annotations
import argparse,json,os,py_compile,tempfile,time
from pathlib import Path
from vitnns import stage_y_y4_r1 as y

def focused_tests(repo:Path)->list[str]:
    import pytest
    class Results:
        def __init__(self):self.rows=[]
        def pytest_runtest_logreport(self,report):
            if report.when=="call" and report.passed:self.rows.append(report.nodeid)
    result=Results();rc=pytest.main([str(repo/"tests/test_stage_y_y4_r1.py"),"-q","-p","no:cacheprovider","--import-mode=importlib"],plugins=[result]);y.require(rc==0,"focused tests failed",y.StartupError);return result.rows

def preflight(repo:Path,output:Path)->dict:
    gate=y.startup(repo,True);y.write_schema_sources(repo);y.validate_schema_sources(repo);rows=focused_tests(repo)
    with tempfile.TemporaryDirectory(prefix="y4-r1-pyc-") as tmp:
        for i,p in enumerate(y.PRODUCER_FILES):
            if p.suffix==".py":py_compile.compile(str(repo/p),cfile=f"{tmp}/{i}.pyc",doraise=True)
    payload={"startup_commit":y.STARTUP_COMMIT,"checks":["local/tracking/remote clean baseline","Git LFS clean/fsck external startup check","immutable Stage Y refs snapshot","Y0/Y2/Y3 raw and canonical identity","historical publication bound to historical ref","fresh Y4-R1 namespace","CPU-only import/access guard","closed schemas","synthetic metrics/bootstrap/sign-flip/Holm/claims","false-pass mutations"],"test_count":len(rows),"false_pass_attack_count":len(y.FALSE_PASS_ATTACKS),"startup_refs_digest":gate["startup_refs_digest"],"formal_outcome_access":False}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(y.canonical_bytes(payload)+b"\n");return {"status":"PASS",**payload}

def seal_producer(repo:Path,preflight_path:Path)->dict:
    payload=json.loads(preflight_path.read_text());return y.seal_producer(repo,payload)

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("preflight","seal-producer","analyze","seal","validate","validate-publication"));p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--output",type=Path);p.add_argument("--preflight",type=Path);a=p.parse_args();repo=a.repo_root.resolve();os.environ.setdefault("CUDA_VISIBLE_DEVICES","");y.runtime_guard()
    try:
        if a.command=="preflight":result=preflight(repo,a.output)
        elif a.command=="seal-producer":result=seal_producer(repo,a.preflight)
        else:
            from scripts import validate_stage_y_y4_r1 as v
            result=v.produce(repo) if a.command=="analyze" else v.seal(repo) if a.command=="seal" else v.validate_completion(repo,a.command=="validate-publication")
    except Exception as e:
        print(json.dumps({"status":getattr(e,"code","BLOCKED_INFRASTRUCTURE"),"error":str(e),"next_authorization":"NONE"},sort_keys=True));return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
