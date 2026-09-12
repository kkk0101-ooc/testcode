#!/usr/bin/env python3
"""Independent metadata-only recovery orchestration."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from vitnns import stage_y_y2_r2_r2 as r


def tests(repo):
    import pytest
    source=repo/"tests/test_stage_y_y2_r2_r2.py"
    class Results:
        def __init__(self):self.rows=[]
        def pytest_runtest_logreport(self,report):
            if report.when=="call" and report.passed:self.rows.append({"name":report.nodeid,"status":"PASS"})
    result=Results()
    rc=pytest.main([str(source),"-q","-p","no:cacheprovider","--import-mode=importlib"],plugins=[result])
    r.require(rc==0,"focused tests failed")
    return result.rows


def preflight(repo,output):
    r.runtime_guard();start=time.monotonic();cpu=time.process_time()
    r.startup(repo,True);r.validate_schema_sources(repo);r.validate_cohort(repo)
    rows=tests(repo)
    import py_compile,tempfile
    with tempfile.TemporaryDirectory(prefix="y2-ledger-compile-") as tmp:
        for i,path in enumerate(r.FILES):
            if path.suffix==".py":py_compile.compile(str(repo/path),cfile=f"{tmp}/{i}.pyc",doraise=True)
    files=[r.record(repo/p,repo) for p in r.FILES+r.DEPENDENCIES]
    payload={"tests":rows,"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.monotonic()-start,"measurement_window":"producer startup, metadata/raw-byte checks, synthetic tests, compilation","startup_commit":r.BASE,"source_files":files,"files_digest":r.canonical({"files":files})}
    r.write(output,"preflight",payload)
    return {"status":"PASS","test_count":len(rows),"preflight_sha256":r.sha(output)}


def producer_seal(repo,preflight_path):
    r.runtime_guard();head=r.git(repo,"rev-parse","HEAD")
    remote={line.split()[1]:line.split()[0] for line in r.git(repo,"ls-remote","origin").splitlines()}
    r.require(head==r.git(repo,"rev-parse","@{upstream}")==remote.get("refs/heads/stage-d-d3-pass")==r.git(repo,"rev-parse",r.PRODUCER)==remote.get(r.PRODUCER),"producer publication not closed")
    r.require(not r.git(repo,"status","--porcelain"),"producer worktree dirty")
    pre=r.read(preflight_path);r.validate_document(pre)
    files=[r.record(repo/p,repo) for p in r.FILES+r.DEPENDENCIES]
    r.require(files==pre["payload"]["source_files"] and r.canonical({"files":files})==pre["payload"]["files_digest"],"preflight source drift")
    r.check_refs(repo);r.validate_schema_sources(repo);history=r.validate_history(repo)
    root=repo/r.ROOT;r.require(not root.exists(),"root already exists")
    r.write(root/"completion/y2_r2_r2.pending.json","pending",{"started_unix_ns":time.time_ns(),"next_authorization":"NONE_WHILE_PENDING"},"PENDING")
    r.write(root/"producer_implementation_seal.json","implementation",{"producer_commit":head,"producer_ref":r.PRODUCER,"source_files":files,"files_digest":r.canonical({"files":files})})
    r.write(root/"producer_preflight.json","preflight",pre["payload"])
    amendment_files=[r.record(repo/p,repo) for p in (r.AMENDMENT,r.SPEC)]
    r.write(root/"prospective_amendment_binding.json","source_snapshot",{"source_files":amendment_files,"files_digest":r.canonical({"files":amendment_files})})
    r.write(root/"historical_immutable_bindings.json","bindings",history)
    r.write(root/"historical_source_snapshot.json","source_snapshot",r.source_snapshots(repo))
    r.verify_implementation(repo)
    return {"status":"PASS","producer_commit":head}


def audit_from_preflight(repo):
    doc=r.read(repo/r.ROOT/"producer_preflight.json");r.validate_document(doc)
    checks=[row for row in doc["payload"]["tests"] if "false_pass" in row["name"]]
    r.require(len(checks)>=24,"insufficient executable negative tests")
    return {"checks":checks,"rejected_count":len(checks)}


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("preflight","seal-producer","collect","seal","validate","validate-publication"));p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--preflight",type=Path);p.add_argument("--output",type=Path);a=p.parse_args();repo=a.repo_root.resolve()
    r.runtime_guard()
    try:
        if a.command=="preflight":result=preflight(repo,a.output)
        elif a.command=="seal-producer":result=producer_seal(repo,a.preflight)
        elif a.command=="collect":result=r.collect(repo)
        elif a.command=="seal":
            doc=r.seal(repo,audit_from_preflight(repo));result={"status":doc["status"],"valid_clean_count":doc["payload"]["valid_clean_count"]}
        else:result=r.validate_completion(repo,a.command=="validate-publication")
    except Exception as error:
        if (repo/r.ROOT/"completion/y2_r2_r2.pending.json").exists():r.seal_failure(repo,a.command,error)
        print(json.dumps({"status":getattr(error,"code","BLOCKED_PROVENANCE"),"error":str(error),"next_authorization":"NONE"}),flush=True);return 2
    print(json.dumps(result,indent=2,sort_keys=True),flush=True);return 0

if __name__=="__main__":raise SystemExit(main())
