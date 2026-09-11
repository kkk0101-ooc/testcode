#!/usr/bin/env python3
"""Independent source, array, schema, and publication reconciliation."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from jsonschema import Draft202012Validator
from vitnns.constants import REPO_ROOT
from vitnns import stage_y_y2 as y2
from vitnns import stage_y_y2_r2 as r2


def now(): return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
def remote(repo,ref):
    rows=y2.git(repo,"ls-remote","origin",ref).split()
    r2.require(len(rows)==2 and rows[1]==ref,"remote ref missing: "+ref)
    return rows[0]

def validate_refs(repo,remote_check=False):
    for ref,commit in r2.EXPECTED_REFS.items():
        r2.require(y2.git(repo,"rev-parse",ref)==commit,"immutable archive ref drift: "+ref)
        if remote_check:r2.require(remote(repo,ref)==commit,"remote archive ref drift: "+ref)

def validate_no_later_stages(repo):
    for parent in (repo/"evidence/stage_y",repo/"reports"):
        r2.require(not any(p.is_dir() and (p.name.startswith(("y3","y4","stage_y_y3","stage_y_y4"))) for p in parent.iterdir()),"later stage root exists")


def validate_schemas(repo):
    from vitnns.stage_y_y2_r2_schema import artifact_schema
    schema=r2.read_json(repo/r2.ARTIFACT_SCHEMA)
    r2.require(schema==artifact_schema(),"serialized artifact schema drift")
    Draft202012Validator.check_schema(schema)
    spec=r2.read_json(repo/r2.RUN_SPEC);run_schema=r2.read_json(repo/r2.RUN_SCHEMA)
    Draft202012Validator.check_schema(run_schema);Draft202012Validator(run_schema).validate(spec)
    old=r2.read_json(repo/y2.ORIGINAL_RUN_SPEC)
    expected={k:old["execution"][k] for k in ("target","baseline","quadrature_schedule","alpha_chunk","source_patch_jvp_chunk","component_vjp_chunk","deletion_forward_batch","first_all_contract_pass_selected")}
    expected.update(condition="clean",blocks=old["coverage"]["blocks"],heads=old["coverage"]["heads"],cuts=old["coverage"]["cuts"])
    r2.require(spec["scientific_contract"]==expected,"scientific contract drift")
    r2.require(spec["source_classes"]=={"Y2_R1_VALID_REBIND":84,"Y2_R1_COMPLETED_BUDGET_REBIND":list(r2.COMPLETED_BUDGET_ORDERS),"Y2_R2_COMPUTED":list(r2.MISSING_ORDERS)},"source partition drift")
    for relative in r2.DEPENDENCIES:
        original=subprocess.check_output(["git","show",f"{r2.Y2_R1_COMMIT}:{repo.name}/{relative}"],cwd=repo)
        r2.require(hashlib.sha256(original).hexdigest()==r2.sha_file(repo/relative),"frozen science dependency drift: "+str(relative))


def validate_predecessors(repo,*,raw_release,arrays=False):
    validate_refs(repo);validate_no_later_stages(repo)
    for relative,commit in r2.PROTECTED:
        r2.require(subprocess.run(["git","diff","--exit-code",commit,"--",str(relative)],cwd=repo).returncode==0,"protected path drift: "+str(relative))
    for root in (r2.Y2_ROOT,r2.Y2_R1_ROOT,r2.Y2_R2_ROOT):
        base=repo/root;inventory=r2.read_json(base/"whole_root_inventory.json");r2.validate_canonical(inventory)
        q=inventory["payload"];records=q["files"]
        r2.require(q["files_digest"]==r2.canonical_digest({"files":records}),"predecessor inventory digest drift")
        actual={p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p.relative_to(base).as_posix() not in q["excluded"]}
        r2.require(actual=={r["path"] for r in records},"predecessor inventory closure drift")
        for row in records:
            path=base/row["path"]
            r2.require(not path.is_symlink() and path.stat().st_size==row["bytes"] and r2.sha_file(path)==row["sha256"],"immutable raw file drift: "+str(path))
    c0=r2.read_json(repo/r2.Y2_ROOT/"completion/y2.json");c1=r2.read_json(repo/r2.Y2_R1_ROOT/"completion/y2_r1.json");c2=r2.read_json(repo/r2.Y2_R2_ROOT/"completion/y2_r2.json")
    for doc in (c0,c1,c2):r2.validate_canonical(doc);r2.require(doc["payload"]["next_authorization"]=="NONE","old authorization drift")
    r2.require(c0["status"]=="BLOCKED_INFRASTRUCTURE" and c1["status"]=="BLOCKED_BUDGET" and c2["status"]=="BLOCKED_STARTUP_STATE","old terminal status drift")
    old_terms=[r2.read_json(p) for p in (repo/r2.Y2_ROOT).glob("shards/shard-*/units/unit-*/completion/unit.json")]
    r2.require(len(old_terms)==100 and {t["status"] for t in old_terms}=={"BLOCKED_INFRASTRUCTURE"},"original failures drift")
    codes={t["payload"]["failure_code"] for t in old_terms}
    r2.require(len(codes)==1 and "CUBLAS_WORKSPACE_CONFIG" in next(iter(codes)),"original failure root drift")
    for t in old_terms:
        r2.validate_canonical(t);q=t["payload"]
        r2.require(q["target_identity"]=={} and q["quadrature_attempts"]==[] and q["array_records"]==[] and q["scientific_record"] is None,"original failed unit science drift")
    terms=r2.terminal_by_order(repo);valid=sorted(o for o,(_,d) in terms.items() if d["status"]=="VALID")
    complete=sorted(o for o,(_,d) in terms.items() if d["status"]=="BLOCKED_BUDGET" and len(d["payload"]["array_records"])==262)
    r2.require(len(valid)==84 and complete==list(r2.COMPLETED_BUDGET_ORDERS),"84+3 source roster drift")
    r2.validate_missing_sources(repo)
    cohort=r2.read_json(repo/r2.Y2_ROOT/"cohort_registry.json");audit=r2.read_json(repo/r2.Y2_ROOT/"audit_subset_registry.json");perturb=r2.read_json(repo/r2.Y2_ROOT/"perturbation_subset_registry.json")
    for doc in (cohort,audit,perturb):r2.validate_canonical(doc)
    r2.require(cohort["canonical_digest"]==r2.COHORT_DIGEST and cohort["payload"]["selection_digest"]==r2.SELECTION_DIGEST,"cohort selection digest drift")
    rows=cohort["payload"]["records"];r2.require(len(rows)==100 and [r["selection_order"] for r in rows]==list(range(100)) and len({r["class_id"] for r in rows})==100,"cohort order/classes drift")
    if raw_release:
        _,excluded=y2.exclusion_union(repo);candidates,_=y2.enumerate_candidates(repo);selected,proof=y2.select_cohort(candidates,excluded)
        r2.require(selected==rows and proof["selection_digest"]==r2.SELECTION_DIGEST,"raw-release selection drift")
        r2.require(y2.subset_registry(selected,y2.AUDIT_NAMESPACE,10,"audit_10")==audit["payload"] and y2.subset_registry(selected,y2.PERTURBATION_NAMESPACE,20,"perturbation_20")==perturb["payload"],"raw subset drift")
    for order in valid+complete:
        path,terminal=terms[order]
        r2.validate_source_binding(repo,terminal,order)
        r2.validate_attempts(repo,terminal,path.parent.parent)
    _historical_resources(repo,r2.Y2_ROOT);_historical_resources(repo,r2.Y2_R1_ROOT)
    if arrays:
        for o in valid:r2.validate_valid_source(repo,o)
        for o in complete:r2.validate_completed_budget_source(repo,o)
    return {"status":"PASS","counts":[84,3,13]}


def run_focused_tests(repo,mutation_only=False):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(repo/"src"))
    command=[str(r2.PYTHON),"-m","pytest","-q","-p","no:cacheprovider","tests/test_stage_y_y2_r2.py"]
    if mutation_only: command += ["-k","false_pass_mutation"]
    p=subprocess.run(command,cwd=repo,env=env,text=True,capture_output=True)
    r2.require(p.returncode==0,"focused tests failed: "+(p.stdout+p.stderr)[-8000:])
    import re
    match=re.search(r"(\d+) passed",p.stdout);r2.require(match is not None,"focused test count missing")
    return int(match.group(1))


def validate_legacy_stage_commands(repo):
    commands=[[str(r2.PYTHON),"scripts/validate_stage_y_y0.py","--repo",str(repo),"validate-completion"],[str(r2.PYTHON),"scripts/validate_stage_y_y1_r1.py","validate"]]
    names=[]
    for command in commands:
        p=subprocess.run(command,cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(repo/"src")),text=True,capture_output=True)
        r2.require(p.returncode==0,"legacy validator failure: "+(p.stdout+p.stderr)[-3000:]);names.append(command[1]+" PASS")
    names.append("Y2/Y2-R1 source-recomputed metadata and raw inventory; no pre-freeze scientific array reads")
    return names


def false_pass_audit(repo):
    count=run_focused_tests(repo,mutation_only=True)
    worker=(repo/"src/vitnns/stage_y_y2_r2_worker.py").read_text();runner=(repo/"scripts/run_stage_y_y2_r2.py").read_text()
    checks=[("executable_false_pass_mutations_"+str(count),count>=22),
        ("resource_stop_or_relabel","BLOCKED_BUDGET" not in worker and "cap_" not in worker),
        ("determinism_disabled","use_deterministic_algorithms(False" not in worker and "warn_only=True" not in worker),
        ("later_stage_execution",all(x not in runner+worker for x in ("stage_y_y3","stage_y_y4","bootstrap(","holm("))),
        ("copy_arrays_as_rebind","shutil.copy" not in runner),
        ("scientific_contract_drift",r2.read_json(repo/r2.RUN_SPEC)["scientific_contract"]["quadrature_schedule"]==list(r2.LADDER))]
    r2.require(all(ok for _,ok in checks),"static false-pass check accepted")
    return {"attack_count":count+len(checks)-1,"rejected_count":count+len(checks)-1,"checks":[{"attack":name,"expected":"REJECT","observed":"REJECT","pass":True} for name,_ in checks]}


def validate_shard_manifests(repo):
    root=repo/r2.ROOT;gate=r2.read_json(root/"startup_gate.json");r2.validate_signed(gate);q=gate["payload"];k=q["locked_worker_count"]
    r2.require(len(q["devices"])==k and len({d["gpu_uuid"] for d in q["devices"]})==k,"device uniqueness drift")
    r2.require([d["physical_index"] for d in q["devices"]]==sorted({d["physical_index"] for d in q["devices"]}),"device order drift")
    for d in q["devices"]:
        r2.require(r2.canonical_digest(d["environment_identity"])==d["environment_digest"]==q["environment_digest"] and d["numeric_probe_digest"]==q["numeric_probe_digest"],"cross-device environment/probe drift")
        r2.require(d["environment_identity"]["python_executable_realpath"]==os.path.realpath(r2.PYTHON),"startup executable drift")
    paths=sorted(root.glob("shards/shard-*/shard_manifest.json"))
    r2.require(len(paths)==k,"shard count drift")
    for shard,path in enumerate(paths):
        doc=r2.read_json(path);r2.validate_signed(doc);p=doc["payload"]
        r2.require(path==root/f"shards/shard-{shard:02d}/shard_manifest.json" and p["shard_id"]==shard and p["selection_orders"]==r2.shard_orders(shard,k) and p["unit_count"]==len(p["selection_orders"]),"missing-only deterministic sharding drift")
    return gate


def validate_computed(repo,path,gate):
    t=r2.read_json(path);r2.validate_signed(t);q=t["payload"];order=q["selection_order"];shard=q["shard_id"];unit=path.parent.parent
    r2.require(t["status"]=="VALID" and order in r2.shard_orders(shard,gate["payload"]["locked_worker_count"]),"computed unit owner/status drift")
    r2.require(path==repo/r2.ROOT/f"shards/shard-{shard:02d}/units/unit-{order:03d}/completion/unit.json","computed path drift")
    r2.require(q["predecessor_terminal"]==r2.file_record(r2.terminal_by_order(repo)[order][0],repo),"computed predecessor drift")
    r2.validate_source_binding(repo,t,order);r2.validate_attempts(repo,t,unit)
    arrays=r2.load_validated_arrays(repo,t,unit);closure=r2.validate_quadrature_arrays(t,arrays);r2.validate_scientific_result(repo,t,unit,closure)
    inv=r2.read_json(r2.checked_record(repo,q["array_inventory"],unit));r2.validate_signed(inv)
    r2.require(inv["payload"]["array_records"]==q["array_records"] and inv["payload"]["records_digest"]==r2.canonical_digest(q["array_records"]),"computed array inventory drift")
    resource=r2.read_json(r2.checked_record(repo,q["resource_record"],unit));r2.validate_signed(resource)
    r2.require(resource["payload"]==q["resource_telemetry"],"computed resource link drift")
    expected={unit/"completion/unit.json",unit/"scientific_result.json",unit/"array_inventory.json",unit/"resource_record.json",*(repo/r["path"] for r in q["array_records"]+q["attempt_records"])}
    r2.require({p for p in unit.rglob("*") if p.is_file()}==expected,"unit payload is not closed")
    worker=r2.read_json(unit.parents[1]/"worker_identity.json")["payload"]
    for field in ("physical_index","gpu_uuid","pci_bus_id","environment_digest"):
        r2.require(worker[field]==gate["payload"]["devices"][shard][field],"worker startup binding drift")
    return {"selection_order":order,"source_object_id":q["source_object_id"],"class_id":r2.cohort_by_order(repo)[order]["class_id"],"source_type":"Y2_R2_COMPUTED","status":"VALID","terminal":r2.file_record(path,repo),"resource_telemetry":q["resource_telemetry"],"validation_digest":r2.canonical_digest({"closure":closure,"arrays":q["array_records"],"attempts":q["attempt_records"]})}


def validate_unified(records):
    r2.require(len(records)==100 and [r["selection_order"] for r in records]==list(range(100)),"unified denominator/order drift")
    r2.require(all(r["status"]=="VALID" for r in records),"unified invalid unit")
    counts={s:sum(r["source_type"]==s for r in records) for s in r2.SOURCE_TYPES}
    r2.require(counts==dict(zip(r2.SOURCE_TYPES,(84,3,13))),"84+3+13 partition drift")
    r2.require([r["selection_order"] for r in records if r["source_type"]==r2.SOURCE_TYPES[1]]==list(r2.COMPLETED_BUDGET_ORDERS),"completed candidate replacement")
    r2.require([r["selection_order"] for r in records if r["source_type"]==r2.SOURCE_TYPES[2]]==list(r2.MISSING_ORDERS),"missing roster replacement")
    return counts


def _historical_resources(repo,relative):
    base=repo/relative;rows=[r2.read_json(p)["payload"]["resource_telemetry"] for p in sorted(base.glob("shards/shard-*/units/unit-*/completion/unit.json"))]
    shards=[r2.read_json(p)["payload"] for p in sorted(base.glob("shards/shard-*/completion/shard.json"))]
    rows += [s["fixed_initialization_resource"] for s in shards]
    retained=sum(p.stat().st_size for p in (base/"shards").rglob("*") if p.is_file())
    total=r2.resource_sum(rows,retained_bytes=retained);reported=r2.read_json(base/"reconciliation.json")["payload"]["resource_totals"]
    for key in ("h20_gpu_hours","cpu_hours","retained_storage_gib"):
        r2.require(abs(total[key]-reported[key])<=max(1e-10,abs(reported[key])*1e-12),"historical resource sum drift")
    return total


def resource_ledger(repo,computed):
    root=repo/r2.ROOT;old=_historical_resources(repo,r2.Y2_ROOT);r1=_historical_resources(repo,r2.Y2_R1_ROOT)
    blocked_bytes=sum(p.stat().st_size for p in (repo/r2.Y2_R2_ROOT).rglob("*") if p.is_file())
    blocked=r2.resource_sum([],retained_bytes=blocked_bytes)
    shards=[r2.read_json(p)["payload"] for p in sorted(root.glob("shards/shard-*/completion/shard.json"))]
    extra=[r2.read_json(root/name)["payload"] for name in ("rebind_validation_resource.json","reconciliation_resource.json")]
    extra.append(r2.read_json(root/"producer_preflight.json")["payload"]["resource_telemetry"])
    rows=[r["resource_telemetry"] for r in computed]+[s["fixed_initialization"] for s in shards]+extra
    retained=sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and (p.relative_to(root).parts[0] in ("shards","rebind_units") or p.name in ("producer_implementation_seal.json","prospective_amendment_binding.json","original_y2_immutable_binding.json","y2_r1_immutable_binding.json","y2_r2_immutable_binding.json","producer_preflight.json","valid_rebind_manifest.json","completed_budget_rebind_manifest.json","missing_work_manifest.json","startup_gate.json","rebind_validation_resource.json","reconciliation_resource.json")))
    current=r2.resource_sum(rows,retained_bytes=retained);stages={"Y2":old,"Y2_R1":r1,"Y2_R2_BLOCKED":blocked,"Y2_R2_R1":current}
    cumulative={k:(max(v[k] for v in stages.values()) if k=="peak_memory_bytes" else sum(v[k] for v in stages.values())) for k in current}
    worker_walls=[s["worker_wall_seconds"] for s in shards]
    begin=r2.read_json(root/"execution_start.json")["payload"]["started_unix_ns"]
    end=max(s["completed_unix_ns"] for s in shards)
    prior={"Y2":r2.read_json(repo/r2.Y2_ROOT/"reconciliation.json")["payload"].get("overall_stage_wall_seconds",0),"Y2_R1":r2.read_json(repo/r2.Y2_R1_ROOT/"reconciliation.json")["payload"]["overall_stage_wall_seconds"],"Y2_R2_BLOCKED":None}
    return {"stage_resources":stages,"cumulative_resources":cumulative,"worker_wall_seconds":worker_walls,"critical_path_wall_seconds":max(worker_walls),"overall_stage_wall_seconds":(end-begin)/1e9,"prior_stage_wall_seconds":prior,"reporting_only":True,"retained_bytes_basis":"new immutable worker/rebind/source payload bytes; final metadata bytes additionally in whole_root_inventory","historical_blocked_cpu_measurement":"Y2-R2 had no GPU work; CPU and wall telemetry were not retained. Cumulative CPU is a measured lower bound, not zero-cost historical recovery."}


def reconcile(repo,write=False):
    start=time.monotonic();cpu=time.process_time();root=repo/r2.ROOT
    r2.validate_implementation(repo);validate_schemas(repo);validate_predecessors(repo,raw_release=True,arrays=False)
    valid_doc=r2.read_json(root/"valid_rebind_manifest.json");complete_doc=r2.read_json(root/"completed_budget_rebind_manifest.json")
    valid_orders=sorted(o for o,(_,d) in r2.terminal_by_order(repo).items() if d["status"]=="VALID")
    records=[]
    for doc,orders,fn in ((valid_doc,valid_orders,r2.validate_valid_source),(complete_doc,list(r2.COMPLETED_BUDGET_ORDERS),r2.validate_completed_budget_source)):
        r2.validate_signed(doc);expected=[fn(repo,o) for o in orders]
        r2.require(doc["payload"]["selection_orders"]==orders and doc["payload"]["records"]==expected and doc["payload"]["records_digest"]==r2.canonical_digest(expected),"rebind source recomputation drift")
        for record in expected:
            terminal=r2.read_json(root/f"rebind_units/unit-{record['selection_order']:03d}/completion/unit.json");r2.validate_signed(terminal)
            r2.require(terminal["status"]=="VALID" and terminal["payload"]==record,"rebind terminal drift")
        records.extend(expected)
    missing=r2.read_json(root/"missing_work_manifest.json");r2.validate_signed(missing)
    r2.require(missing["payload"]["records"]==r2.validate_missing_sources(repo),"missing provenance drift")
    gate=validate_shard_manifests(repo);computed=[validate_computed(repo,p,gate) for p in sorted(root.glob("shards/shard-*/units/unit-*/completion/unit.json"))]
    records=sorted(records+computed,key=lambda r:r["selection_order"]);counts=validate_unified(records)
    for shard in range(gate["payload"]["locked_worker_count"]):
        doc=r2.read_json(root/f"shards/shard-{shard:02d}/completion/shard.json");r2.validate_signed(doc)
        r2.require(doc["status"]=="PASS" and doc["payload"]["shard_id"]==shard and doc["payload"]["unit_count"]==doc["payload"]["terminal_counts"]["VALID"]==len(r2.shard_orders(shard,gate["payload"]["locked_worker_count"])) and sum(doc["payload"]["terminal_counts"].values())==doc["payload"]["unit_count"],"shard terminal count drift")
    payload={"denominator":100,"valid_clean_count":100,"source_counts":counts,"selection_orders":list(range(100)),"records":records,"records_digest":r2.canonical_digest(records),"formal_outcome_access":False,"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"later_stages_started":[],"next_authorization":r2.NEXT}
    doc=r2.signed("reconciliation","PASS",payload)
    if write:
        r2.write_json_new(root/"reconciliation_resource.json",r2.signed("verification_resource","PASS",{"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.monotonic()-start,"gpu_active_seconds":0,"peak_reserved_gpu_bytes":0,"retained_bytes":0,"download_bytes":0}))
        ledger=resource_ledger(repo,computed)
        overruns=[]
        for stage,values in {**ledger["stage_resources"],"cumulative":ledger["cumulative_resources"]}.items():
            for field,cap in (("h20_gpu_hours",64.0),("cpu_hours",218.0),("retained_storage_gib",6.0),("peak_memory_bytes",21*1024**3)):
                if values[field]>cap:overruns.append({"stage":stage,"field":field,"observed":values[field],"former_ceiling":cap,"classification":"RESOURCE_OVERRUN_REPORTED","scientific_validity_effect":"NONE"})
        for name,kind,p in (("unified_100_unit_manifest","unified_manifest",payload),("resource_ledger","resource_ledger",ledger),("cumulative_resource_ledger","resource_ledger",ledger),("resource_overrun_report","overruns",{"resource_overruns":overruns,"reporting_only":True}),("access_audit","access_audit",{"external_accesses":[],"download_bytes":0,"clean_array_validation_performed":True,"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"later_stages_started":[]}),("false_pass_audit","false_pass_audit",false_pass_audit(repo))):
            r2.write_json_new(root/(name+".json"),r2.signed(kind,"PASS",p))
        r2.write_json_new(root/"reconciliation.json",doc)
    return doc


def validate_root_schemas(repo):
    for path in sorted((repo/r2.ROOT).rglob("*.json")):
        r2.validate_signed(r2.read_json(path))


def _write_report(repo,text):
    path=repo/r2.REPORT_ROOT/"y2_r2_r1_report.md";path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
    return r2.file_record(path,repo)


def seal(repo):
    root=repo/r2.ROOT;rec=reconcile(repo,False)
    r2.require(rec==r2.read_json(root/"reconciliation.json"),"seal reconciliation drift")
    validate_root_schemas(repo)
    computed=[r for r in rec["payload"]["records"] if r["source_type"]==r2.SOURCE_TYPES[2]]
    ledger=resource_ledger(repo,computed)
    r2.require(ledger==r2.read_json(root/"resource_ledger.json")["payload"],"resource ledger drift")
    validation={"checks":["all original source bytes and cohort identities recomputed","84 VALID and 3 completed-budget arrays recomputed without GPU replay or copying","13 new attempts independently validated","100 exact orders and all VALID","closed schemas and typed immutable terminals","frozen producer dependency identities","resources reporting-only","no statistics or later stages"],"record_count":100,"records_digest":rec["payload"]["records_digest"]}
    r2.write_json_new(root/"source_recomputing_validation_report.json",r2.signed("validation_report","PASS",validation))
    report=_write_report(repo,"# Stage Y2-R2-R1 completion\n\nPASS: 100/100 VALID, comprising 84 original VALID rebinds, 3 independently validated completed-budget rebinds, and 13 new computations. Original roots and refs remain unchanged. Resource overruns are reporting-only.\n\nThe three completed-budget units retain their original BLOCKED_BUDGET terminals. Their discarded intervention scalar payloads are not claimed as recovered; completed execution is established by immutable control flow and call accounting.\n\nMeasured cumulative resources: `"+json.dumps(ledger["cumulative_resources"],sort_keys=True)+"`. Historical blocked Y2-R2 CPU/wall telemetry was not retained, so cumulative CPU is a measured lower bound. Full separate ledgers and overrun records are in the formal root.\n\nNo bootstrap, Holm, claim decision, Y3, or Y4 executed. Next authorization: `"+r2.NEXT+"`.\n")
    files=[r2.file_record(p,repo) for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix()!="completion/y2_r2_r1.pending.json"]
    r2.write_json_new(root/"payload_manifest.json",r2.signed("payload_manifest","PASS",{"files":files,"file_count":len(files),"files_digest":r2.canonical_digest({"files":files})}))
    r2.write_json_new(root/"whole_root_inventory.json",r2.whole_root_inventory(root,"PASS"))
    completion=r2.signed("completion","PASS",{"completed_at":now(),"denominator":100,"valid_clean_count":100,"source_counts":rec["payload"]["source_counts"],"formal_outcome_access":False,"next_authorization":r2.NEXT,"reconciliation":r2.file_record(root/"reconciliation.json",repo),"validation_report":r2.file_record(root/"source_recomputing_validation_report.json",repo),"whole_root_inventory":r2.file_record(root/"whole_root_inventory.json",repo),"report":report,"failure":None})
    r2.publish_terminal(root/"completion/y2_r2_r1.pending.json",root/"completion/y2_r2_r1.json",completion)
    return {"status":"PASS","valid_clean_count":100,"next_authorization":r2.NEXT}


def seal_failure(repo,phase,error):
    root=repo/r2.ROOT;status=getattr(error,"code","BLOCKED_PROVENANCE")
    if status not in ("FAILED_NUMERICAL","BLOCKED_STARTUP_STATE","BLOCKED_SOURCE_DRIFT","BLOCKED_ACCESS","BLOCKED_PROVENANCE","BLOCKED_INFRASTRUCTURE","BLOCKED_SCIENTIFIC_VALIDITY"):status="BLOCKED_PROVENANCE"
    if not (root/"failure.json").exists():r2.write_json_new(root/"failure.json",r2.signed("failure",status,{"phase":phase,"failure_type":type(error).__name__,"message":str(error),"next_authorization":"NONE"}))
    report=_write_report(repo,"# Stage Y2-R2-R1 blocked\n\n"+status+" during "+phase+".\n\n"+str(error)+"\n\nEvidence is preserved. No next authorization.\n")
    r2.write_json_new(root/"whole_root_inventory.json",r2.whole_root_inventory(root,status))
    completion=r2.signed("completion",status,{"completed_at":now(),"denominator":100,"valid_clean_count":0,"source_counts":dict.fromkeys(r2.SOURCE_TYPES,0),"formal_outcome_access":False,"next_authorization":"NONE","reconciliation":None,"validation_report":None,"whole_root_inventory":r2.file_record(root/"whole_root_inventory.json",repo),"report":report,"failure":r2.file_record(root/"failure.json",repo)})
    r2.publish_terminal(root/"completion/y2_r2_r1.pending.json",root/"completion/y2_r2_r1.json",completion)


def validate(repo,publication=False):
    root=repo/r2.ROOT;completion=r2.read_json(root/"completion/y2_r2_r1.json");r2.validate_signed(completion)
    r2.require(completion["status"]=="PASS" and completion["payload"]["valid_clean_count"]==100 and completion["payload"]["next_authorization"]==r2.NEXT,"completion PASS semantics drift")
    r2.require(not list(root.rglob("*.pending.json")) and not list(root.rglob("*.tmp")),"unsealed pending/temp file")
    validate_root_schemas(repo);rec=reconcile(repo,False)
    r2.require(rec==r2.read_json(root/"reconciliation.json"),"recomputed reconciliation drift")
    for key in ("reconciliation","validation_report","whole_root_inventory","report"):r2.checked_record(repo,completion["payload"][key])
    r2.require(r2.whole_root_inventory(root,"PASS")==r2.read_json(root/"whole_root_inventory.json"),"whole inventory drift")
    manifest=r2.read_json(root/"payload_manifest.json");files=manifest["payload"]["files"]
    for row in files:r2.checked_record(repo,row)
    r2.require(manifest["payload"]["files_digest"]==r2.canonical_digest({"files":files}),"payload manifest digest drift")
    ledger=resource_ledger(repo,[r for r in rec["payload"]["records"] if r["source_type"]==r2.SOURCE_TYPES[2]])
    r2.require(ledger==r2.read_json(root/"resource_ledger.json")["payload"]==r2.read_json(root/"cumulative_resource_ledger.json")["payload"],"ledger recomputation drift")
    false_pass_audit(repo)
    if publication:
        head=y2.git(repo,"rev-parse","HEAD")
        r2.require(head==y2.git(repo,"rev-parse","@{upstream}")==remote(repo,"refs/heads/stage-d-d3-pass")==y2.git(repo,"rev-parse",r2.COMPLETION_REF)==remote(repo,r2.COMPLETION_REF),"completion publication drift")
        r2.require(remote(repo,r2.PRODUCER_REF)==r2.validate_implementation(repo),"producer remote drift")
        validate_refs(repo,True);r2.require(not y2.git(repo,"status","--porcelain"),"publication worktree dirty")
        lfs=y2.git(repo,"lfs","status");r2.require("(LFS:" not in lfs and "(Git:" not in lfs,"publication LFS pending")
    return {"status":"PASS","valid_clean_count":100,"source_counts":rec["payload"]["source_counts"],"completion_sha256":r2.sha_file(root/"completion/y2_r2_r1.json"),"completion_canonical_digest":completion["canonical_digest"],"inventory_sha256":r2.sha_file(root/"whole_root_inventory.json"),"publication":publication,"next_authorization":r2.NEXT}


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("focused-tests","false-pass-audit","validate","validate-publication"));p.add_argument("--repo-root",type=Path,default=REPO_ROOT);a=p.parse_args();repo=a.repo_root.resolve()
    try:
        if a.command=="focused-tests":result={"test_count":run_focused_tests(repo)}
        elif a.command=="false-pass-audit":result=false_pass_audit(repo)
        else:result=validate(repo,a.command=="validate-publication")
    except Exception as error:print(json.dumps({"status":getattr(error,"code","BLOCKED_PROVENANCE"),"error":str(error)}));return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
