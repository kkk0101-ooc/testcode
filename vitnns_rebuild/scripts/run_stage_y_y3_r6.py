#!/usr/bin/env python3
"""Producer lifecycle for the CPU-only Y3-R6 validator recovery."""
from __future__ import annotations
import argparse,importlib.util,json,os,subprocess,time
from pathlib import Path
from jsonschema import Draft202012Validator
from vitnns import stage_y_y3 as y3
from vitnns import stage_y_y3_r5 as r5
from vitnns import stage_y_y3_r6 as r6
from vitnns import stage_y_y2_r2_r2 as y2c
def schemas(repo):
    spec=r6.read(repo/r6.SPEC);schema=r6.read(repo/r6.SPEC_SCHEMA);artifact=r6.read(repo/r6.ARTIFACT_SCHEMA);Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(spec);Draft202012Validator.check_schema(artifact);r6.require(artifact==r6.artifact_schema(),"R6 artifact schema drift",r6.StartupError)
    def walk(v):
        if isinstance(v,dict):
            if v.get("type")=="object":r6.require(v.get("additionalProperties") is False,"open nested schema",r6.StartupError)
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(schema);walk(artifact)
def validator(repo):
    s=importlib.util.spec_from_file_location("validate_r6",repo/"scripts/validate_stage_y_y3_r6.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def preflight(repo):
    refs=y3.remote_refs(repo);r6.require(y3.git(repo,"symbolic-ref","--short","HEAD")==y3.BRANCH,"branch drift",r6.StartupError);r6.require(y3.git(repo,"rev-parse","HEAD")==y3.git(repo,"rev-parse","@{upstream}")==refs.get("refs/heads/"+y3.BRANCH)==r6.STARTUP_COMMIT,"R6 startup baseline drift",r6.StartupError);r6.require(refs.get(r6.R4_REF)==r6.R4_COMMIT and y3.git(repo,"rev-parse",r6.R4_REF)==r6.R4_COMMIT,"R4 blocked ref drift",r6.StartupError);r6.require(refs.get(r6.R5_REF)==r6.R5_COMMIT and y3.git(repo,"rev-parse",r6.R5_REF)==r6.R5_COMMIT,"R5 blocked ref drift",r6.StartupError);y2c.git_binding(repo,r5.SOURCE_ROOT,r6.R4_COMMIT);y2c.git_binding(repo,r5.SOURCE_REPORT_ROOT,r6.R4_COMMIT);y2c.git_binding(repo,r5.ROOT,r6.R5_COMMIT);y2c.git_binding(repo,r5.REPORT_ROOT,r6.R5_COMMIT)
    for ref in (r6.PRODUCER_REF,r6.PASS_REF,r6.BLOCKED_REF):r6.require(ref not in refs and not y3.git(repo,"for-each-ref","--format=%(refname)",ref),"R6 ref exists: "+ref,r6.StartupError)
    status=y3.git(repo,"status","--porcelain").splitlines();allowed={str(x) for x in r6.PRODUCER_FILES};unexpected=[x[3:] for x in status if not any(x[3:]==p or x[3:].endswith("/"+p) for p in allowed)];r6.require(not unexpected,"unexpected worktree paths: "+repr(unexpected),r6.StartupError);y3.lfs_clean(repo);r6.require(not (repo/r6.ROOT).exists() and not (repo/r6.REPORT_ROOT).exists(),"R6 roots exist",r6.StartupError);schemas(repo)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1");tests=subprocess.run([str(y3.PYTHON),"-m","pytest","-q","tests/test_stage_y_y3_r6.py","--basetemp","/tmp/stage-y3-r6-pytest"],cwd=repo,env=env,text=True,capture_output=True);r6.require(tests.returncode==0,"R6 tests failed: "+tests.stdout+tests.stderr,r6.StartupError);audit=validator(repo).false_pass_audit();root=repo/r6.ROOT;root.mkdir(parents=True);r6.write(root/"completion/y3_r6.pending.json","pending",{"started_unix_ns":time.time_ns(),"next_authorization":"NONE_WHILE_PENDING"},"PENDING");checks=["R4/R5 immutable Git and LFS bindings","R4/R5 blocked refs","R5 completion identity","validator-only zero compute","unit-index keyed reconciliation","closed schemas","focused tests","false-pass audit"];r6.write(root/"producer_preflight.json","producer_preflight",{"startup_commit":r6.STARTUP_COMMIT,"checks":checks,"focused_test_count":8,"false_pass_attack_count":audit["attack_count"],"source_count":160,"new_empirical_attempt_count":0,"gpu_execution":False,"jpeg_decoded":False,"model_loaded":False,"torch_imported":False});r6.write(root/"producer_false_pass_audit.json","false_pass_audit",audit)
    for name,path,commit in (("r4_source",r5.SOURCE_ROOT/"completion/y3.json",r6.R4_COMMIT),("r5_source",r5.ROOT/"completion/y3_r5.json",r6.R5_COMMIT)):
        f=r6.file_record(repo/path,repo);r6.write(root/(name+"_binding.json"),"binding",{"name":name,"files":[f],"files_digest":r6.canonical({"files":[f]}),"canonical_digest_bound":r6.read(repo/path)["canonical_digest"],"commit":commit})
    return {"status":"PASS","tests":8,"false_pass_attacks":audit["attack_count"],"gpu_execution":False}
def finalize(repo):schemas(repo);r6.require(y3.git(repo,"rev-parse","HEAD")==y3.git(repo,"rev-parse","@{upstream}")==r6.STARTUP_COMMIT,"R6 final baseline drift",r6.StartupError);y2c.git_binding(repo,r5.SOURCE_ROOT,r6.R4_COMMIT);y2c.git_binding(repo,r5.ROOT,r6.R5_COMMIT);return {"status":"PASS"}
def seal_impl(repo,commit):rows=r6.source_files(repo);return r6.write(repo/r6.ROOT/r6.IMPLEMENTATION_SEAL,"implementation_seal",{"producer_commit":commit,"producer_ref":r6.PRODUCER_REF,"source_files":rows,"files_digest":r6.canonical({"files":rows})})
def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("write-schemas","preflight","finalize-preflight","write-producer-seal","reconcile","seal","validate-completion","validate-publication"));p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--producer-commit");a=p.parse_args();repo=a.repo_root.resolve()
    try:
        if a.command=="write-schemas":r6.write_schemas(repo);out={"status":"PASS"}
        elif a.command=="preflight":out=preflight(repo)
        elif a.command=="finalize-preflight":out=finalize(repo)
        elif a.command=="write-producer-seal":out=seal_impl(repo,a.producer_commit or y3.git(repo,"rev-parse","HEAD"))
        else:
            v=validator(repo);out=v.reconcile(repo,a.command=="seal") if a.command in ("reconcile","seal") else v.validate_completion(repo,a.command=="validate-publication")
    except Exception as e:print(json.dumps({"status":getattr(e,"code","BLOCKED_PROVENANCE"),"error":str(e),"next_authorization":"NONE"},sort_keys=True));return 2
    print(json.dumps(out,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
