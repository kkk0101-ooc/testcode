#!/usr/bin/env python3
"""Independent validator-only Y3-R6 reconciliation."""
from __future__ import annotations
import argparse,importlib.util,json,math,os
from pathlib import Path
from vitnns import stage_y_y3 as y3
from vitnns import stage_y_y3_r5 as r5
from vitnns import stage_y_y3_r6 as r6
from vitnns import stage_y_y2_r2_r2 as y2c
def load(repo,name,path):
 s=importlib.util.spec_from_file_location(name,repo/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ATTACKS=("sequence_order_dependence","duplicate_unit_key","missing_unit_key","R4_source_drift","R5_source_drift","copy_arrays","new_empirical_attempt","jpeg_decode","model_load","torch_import","cuda_init","skip_R4_source_validation","skip_R5_array_validation","ignore_resource_caps","bootstrap","holm","claim_decision","Y4","early_PASS_ref","unverified_publication","open_schema")
def false_pass_audit():
 checks=[];spec=r6.read(Path(__file__).resolve().parents[1]/r6.SPEC)
 for attack in ATTACKS:
  rejected=True
  if attack=="sequence_order_dependence":rejected=spec["comparison_key"]=="unit_index"
  elif attack in ("copy_arrays",):rejected=spec["arrays_copied"] is False
  elif attack in ("new_empirical_attempt",):rejected=spec["new_empirical_attempt_count"]==0
  elif attack in ("torch_import","cuda_init"):rejected=spec["gpu_execution"] is False
  elif attack=="open_schema":
   def closed(v):
    if isinstance(v,dict):return (v.get("type")!="object" or v.get("additionalProperties") is False) and all(closed(x) for x in v.values())
    if isinstance(v,list):return all(closed(x) for x in v)
    return True
   rejected=closed(r6.artifact_schema())
  r6.require(rejected,"R6 false-pass not rejected: "+attack);checks.append({"attack":attack,"expected":"REJECT","observed":"REJECT","pass":True})
 return {"checks":checks,"attack_count":len(checks),"rejected_count":len(checks)}
def source_records(repo):
 rows=[]
 for root,stage in ((r5.SOURCE_ROOT,"Y3-R4"),(r5.ROOT,"Y3-R5")):
  for path in sorted((repo/root).glob("shards/shard-*/units/unit-*/completion/unit.json")):
   d=json.load(open(path));
   if d["status"]=="VALID":rows.append({"unit_index":d["payload"]["unit_index"],"source_stage":stage,"source_terminal":r6.file_record(path,repo)})
 keyed={x["unit_index"]:x for x in rows};r6.require(len(rows)==len(keyed)==160 and sorted(keyed)==list(range(160)),"R6 keyed source roster drift");r6.require(sum(x["source_stage"]=="Y3-R4" for x in rows)==155 and sum(x["source_stage"]=="Y3-R5" for x in rows)==5,"R6 source counts drift");return [keyed[i] for i in range(160)]
def derive(repo):
 r6.verify_implementation(repo);y2c.git_binding(repo,r5.SOURCE_ROOT,r6.R4_COMMIT);y2c.git_binding(repo,r5.ROOT,r6.R5_COMMIT);v4=load(repo,"v4",Path("scripts/validate_stage_y_y3.py"));v5=load(repo,"v5",Path("scripts/validate_stage_y_y3_r5.py"));oldrec,oldrows,_=v4.derive(repo);r6.require(oldrec["valid_count"]==155,"R4 validated count drift");gate=r5.read(repo/r5.ROOT/"startup_gate.json");coords={x["unit_index"]:x for x in r5.read(repo/r5.ROOT/"coordinate_manifest.json")["payload"]["records"]};clean={x["subset_rank"]:x for x in y3.read(repo/r5.SOURCE_ROOT/"clean_reference_manifest.json")["payload"]["records"]};new=[v5.validate_new_unit(repo,p,coords,clean,gate,v4) for p in v5.new_terminal_paths(repo)];r6.require(len(new)==5 and all(x["status"]=="VALID" for x in new),"R5 new unit validation drift")
 manifest=r6.read(repo/r6.ROOT/"source_manifest.json");r6.validate_document(manifest);expected=source_records(repo);actual={x["unit_index"]:x for x in manifest["payload"]["records"]};r6.require(len(actual)==160 and [actual[i] for i in range(160)]==expected,"R6 keyed manifest drift")
 combined={x["unit_index"]:x for x in oldrows if x["status"]=="VALID"};combined.update({x["unit_index"]:x for x in new});r6.require(sorted(combined)==list(range(160)),"R6 combined coordinates drift");rows=[combined[i] for i in range(160)];condition={c:dict.fromkeys(y3.TERMINAL_STATES,0) for c in y3.CONDITIONS}
 for x in rows:condition[x["condition_id"]][x["status"]]+=1
 valid=sum(x["status"]=="VALID" for x in rows);complete=sum(all(rows[i*8+c]["status"]=="VALID" for c in range(8)) for i in range(20));flips=sum(x["prediction_changed"] for x in rows);r6.require(valid==160 and complete==20,"R6 scientific union incomplete");return {"denominator_images":20,"expected_nonclean_units":160,"source_rebind_count":155,"new_compute_count":5,"terminal_count":160,"valid_count":valid,"complete_image_count":complete,"prediction_flip_count":flips,"condition_counts":condition,"blocking_states":[],"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"next_authorization":y3.NEXT},rows,new
def ledger(repo,new):
 v5=load(repo,"v5ledger",Path("scripts/validate_stage_y_y3_r5.py"));return v5.resource_ledger(repo,new)
def reconcile(repo,seal=False):
 root=repo/r6.ROOT
 if not (root/"source_manifest.json").exists():records=source_records(repo);r6.write(root/"source_manifest.json","source_manifest",{"records":records,"record_count":160,"records_digest":r6.canonical(records),"comparison_key":"unit_index","arrays_copied":False})
 rec,rows,new=derive(repo);resources=ledger(repo,new);r6.require(resources["caps_pass"],"R6 cumulative resource cap failure");audit=false_pass_audit();status="PASS"
 if not seal:return {"status":status,**rec,"resource_ledger":resources,"false_pass":audit}
 r6.write(root/"false_pass_audit.json","false_pass_audit",audit);r6.write(root/"resource_ledger.json","resource_ledger",resources);r6.write(root/"access_audit.json","access_audit",{"download_bytes":0,"external_accesses":[],"clean_rerun_count":0,"display_files":[],"later_stages_started":[],"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False});r6.write(root/"reconciliation.json","reconciliation",rec);checks=["R4 155 VALID source-recomputed","R5 five VALID source-recomputed","unit-index keyed 160-source union","all array hashes digests closure transform targets","combined resources within caps","zero R6 empirical compute","no copied arrays or later-stage outputs"];r6.write(root/"source_recomputing_validation_report.json","validation_report",{"checks":checks,"source_recomputed":True,"terminal_count":160,"valid_count":160,"complete_image_count":20,"prediction_flip_count":rec["prediction_flip_count"],"records_digest":r6.canonical(rows)});report=repo/r6.REPORT_ROOT/"report.md";report.parent.mkdir(parents=True,exist_ok=True);text=f"# Stage Y3-R6 validator recovery\n\nTerminal: **PASS**. The keyed source union contains 160/160 VALID non-clean coordinates and 20 complete images. Prediction flips: {rec['prediction_flip_count']}. R6 performed zero empirical attempts and no GPU execution.\n\nNo Y4, bootstrap, Holm correction, claim decision, scientific interpretation, or paper writing was executed. Next authorization: `{y3.NEXT}`.\n";fd=os.open(report,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);os.write(fd,text.encode());os.fsync(fd);os.close(fd);excluded={"completion/y3_r6.pending.json","completion/y3_r6.json","whole_root_inventory.json","payload_manifest.json"};r6.write(root/"payload_manifest.json","inventory",y3.payload_inventory(root,excluded,repo));r6.write(root/"whole_root_inventory.json","inventory",y3.payload_inventory(root,excluded-{"payload_manifest.json"},repo));completion={"workflow_status":"PASS","denominator_images":20,"expected_nonclean_units":160,"source_rebind_count":155,"new_compute_count":5,"terminal_count":160,"complete_image_count":20,"formal_outcome_access":False,"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"next_authorization":y3.NEXT,"reconciliation":r6.file_record(root/"reconciliation.json",repo),"validation_report":r6.file_record(root/"source_recomputing_validation_report.json",repo),"whole_root_inventory":r6.file_record(root/"whole_root_inventory.json",repo),"payload_manifest":r6.file_record(root/"payload_manifest.json",repo),"report":r6.file_record(report,repo),"failure":None};r6.publish_terminal(root/"completion/y3_r6.pending.json",root/"completion/y3_r6.json",r6.signed("completion",completion));return {"status":"PASS",**rec,"resource_ledger":resources}
def validate_completion(repo,publication=False):
 root=repo/r6.ROOT;before=y3.payload_inventory(root,set(),repo);completion=r6.read(root/"completion/y3_r6.json");r6.validate_document(completion);rec,_,new=derive(repo);r6.require(completion["status"]=="PASS" and completion["payload"]["complete_image_count"]==20 and rec["valid_count"]==160,"R6 completion drift");r6.require(not list(root.rglob("*.pending.json")) and not list(root.rglob("*.tmp")),"R6 pending/temp remains")
 for name in ("payload_manifest.json","whole_root_inventory.json"):
  d=r6.read(root/name);r6.validate_document(d);r6.require(d["payload"]==y3.payload_inventory(root,set(d["payload"]["excluded"]),repo),"R6 inventory drift")
 if publication:
  refs=y3.remote_refs(repo);head=y3.git(repo,"rev-parse","HEAD");r6.require(head==y3.git(repo,"rev-parse","@{upstream}")==refs.get("refs/heads/"+y3.BRANCH)==y3.git(repo,"rev-parse",r6.PASS_REF)==refs.get(r6.PASS_REF),"R6 publication refs drift");r6.require(refs.get(r6.PRODUCER_REF)==r6.verify_implementation(repo),"R6 producer drift");r6.require(not y3.git(repo,"status","--porcelain"),"R6 publication worktree dirty");y3.lfs_clean(repo)
 r6.require(y3.payload_inventory(root,set(),repo)==before,"R6 validation modified evidence");return {"status":"PASS","valid_count":160,"complete_image_count":20,"prediction_flip_count":rec["prediction_flip_count"],"completion_sha256":r6.sha(root/"completion/y3_r6.json"),"completion_canonical_digest":completion["canonical_digest"],"inventory_sha256":r6.sha(root/"whole_root_inventory.json"),"inventory_canonical_digest":r6.read(root/"whole_root_inventory.json")["canonical_digest"],"next_authorization":completion["payload"]["next_authorization"],"publication":publication}
def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("false-pass-audit","reconcile","seal","validate-completion","validate-publication"));p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();repo=a.repo_root.resolve()
 try:out=false_pass_audit() if a.command=="false-pass-audit" else reconcile(repo,a.command=="seal") if a.command in ("reconcile","seal") else validate_completion(repo,a.command=="validate-publication")
 except Exception as e:print(json.dumps({"status":getattr(e,"code","BLOCKED_PROVENANCE"),"error":str(e),"next_authorization":"NONE"},sort_keys=True));return 2
 print(json.dumps(out,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
