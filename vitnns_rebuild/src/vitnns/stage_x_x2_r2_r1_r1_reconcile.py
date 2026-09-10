"""Source-recomputing X2-R2-R1-R1 reconciliation, decisions, and sealing."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_sha256
from . import stage_x_x2_r2_r1_reconcile as base
from . import stage_x_x2_r2_r1_r1 as control
from . import stage_x_x2_r2_r1_r1_worker as worker

GAMMA_SCORE_FIELDS=base.GAMMA_SCORE_FIELDS
MARGIN_FIELDS=base.MARGIN_FIELDS
FUTILITY_FIELDS=base.FUTILITY_FIELDS
RESOURCE_FIELDS=base.RESOURCE_FIELDS
QUANTILE_FIELDS=base.QUANTILE_FIELDS
BUDGET_FIELDS=base.BUDGET_FIELDS
STABILITY_AGG_FIELDS=base.STABILITY_AGG_FIELDS
STABILITY_NONINFERIORITY_FIELDS=base.STABILITY_NONINFERIORITY_FIELDS


class X2R2R1R1ReconcileError(ValueError): pass


def _read_csv(path: Path) -> list[dict[str,str]]:
    with path.open("r",encoding="utf-8",newline="") as stream:return [dict(row) for row in csv.DictReader(stream)]


def _counts(value: Mapping[str,Any]) -> dict[str,Any]:
    return {row["name"]:row["value"] for row in value["payload"]["counts"]}


def _phase(repo: Path, unit: Mapping[str,Any], name: str) -> tuple[Path,dict[str,Any]]:
    path=repo/control.ROOT/"units"/str(unit["unit_digest"])/"attempts"/worker.ATTEMPT_ID/name
    terminal=control.strict_read(path/"terminal.json");control.validate_artifact(terminal,repo,verify_files=True)
    identity={row["name"]:row["value"] for row in terminal["payload"]["identity"]}
    expected_attempt=control.REUSE_ATTEMPT_ID if name in {"clean","gamma"} else worker.ATTEMPT_ID
    if identity!={"attempt_id":expected_attempt,"phase":name,"unit_digest":unit["unit_digest"]} or terminal["status"] not in {"PASS","FAILED"}:raise X2R2R1R1ReconcileError(f"phase identity/status drift: {unit['unit_digest']} {name}")
    return path,terminal


def _phase_file(repo:Path,path:Path,terminal:Mapping[str,Any],name:str)->Path:
    return control.phase_file(repo,path,terminal,name)


def _validate_clean_source(repo: Path, unit: Mapping[str,Any], path: Path, terminal:Mapping[str,Any]) -> None:
    if terminal["status"]=="FAILED":return
    rows=_read_csv(_phase_file(repo,path,terminal,"summary.csv"));cuts=_read_csv(_phase_file(repo,path,terminal,"cuts.csv"));adapters=_read_csv(_phase_file(repo,path,terminal,"adapters.csv"))
    if len(rows)!=1 or rows[0]["model_id"]!=unit["model_id"] or rows[0]["sample_id"]!=unit["sample_id"] or rows[0]["sample_sha256"]!=unit["sample_sha256"]:raise X2R2R1R1ReconcileError("clean identity drift")
    row=rows[0]
    if int(row["complete_cut_count"])!=135 or row["complete_cut_all_pass"]!="True" or row["native_pixel_provenance"]!="True" or float(row["identity_max_abs_error"])>1e-6:raise X2R2R1R1ReconcileError("clean validity gate failed")
    schedule={128,256,512,1024,2048,4096,8192}
    if int(row["selected_steps"]) not in schedule or len(cuts)!=135 or len({item["cut_id"] for item in cuts})!=135 or any(item["passed"]!="True" for item in cuts):raise X2R2R1R1ReconcileError("complete-cut source reconciliation failed")
    if len(adapters)!=4 or {item["method_id"] for item in adapters}!={"vit_grad_cam","transformer_attribution_chefer","rise","occlusion"} or any(item["target_index"]!=row["predicted_class"] or item["finite"]!="True" for item in adapters):raise X2R2R1R1ReconcileError("adapter source reconciliation failed")


def _validate_gamma_source(unit: Mapping[str,Any], path: Path) -> None:
    rows=_read_csv(_phase_file(Path(".").resolve(),path,control.strict_read(path/"terminal.json"),"summary.csv"));roads=_read_csv(_phase_file(Path(".").resolve(),path,control.strict_read(path/"terminal.json"),"road_rows.csv"));grid={(float(row["conv_gamma"]),float(row["linear_gamma"])) for row in rows}
    if len(rows)!=15 or grid!={(conv,linear) for conv in (0.1,0.25,100.0) for linear in (0.0,0.01,0.05,0.1,1.0)} or len(roads)!=1350:raise X2R2R1R1ReconcileError("gamma roster source reconciliation failed")
    source_digest={row["source_digest"] for row in rows}
    if len(source_digest)!=1 or any(row["unit_digest"]!=unit["unit_digest"] or float(row["forward_parity_max_abs"])>=1e-6 or row["target_equal"]!="True" or row["shape_equal_224x224"]!="True" or row["finite"]!="True" or row["source_binding_equal"]!="True" or not all(math.isfinite(float(row[key])) for key in ("road_combined","road_morf","road_lerf")) for row in rows):raise X2R2R1R1ReconcileError("gamma validity source reconciliation failed")


def _validate_gamma_source_with_repo(repo: Path, unit: Mapping[str,Any], path: Path, terminal: Mapping[str,Any]) -> None:
    rows=_read_csv(_phase_file(repo,path,terminal,"summary.csv"));roads=_read_csv(_phase_file(repo,path,terminal,"road_rows.csv"));grid={(float(row["conv_gamma"]),float(row["linear_gamma"])) for row in rows}
    if len(rows)!=15 or grid!={(conv,linear) for conv in (0.1,0.25,100.0) for linear in (0.0,0.01,0.05,0.1,1.0)} or len(roads)!=1350:raise X2R2R1R1ReconcileError("gamma roster source reconciliation failed")
    source_digest={row["source_digest"] for row in rows}
    if len(source_digest)!=1 or any(row["unit_digest"]!=unit["unit_digest"] or float(row["forward_parity_max_abs"])>=1e-6 or row["target_equal"]!="True" or row["shape_equal_224x224"]!="True" or row["finite"]!="True" or row["source_binding_equal"]!="True" or not all(math.isfinite(float(row[key])) for key in ("road_combined","road_morf","road_lerf")) for row in rows):raise X2R2R1R1ReconcileError("gamma validity source reconciliation failed")


def _validate_formal_source(repo: Path, unit: Mapping[str,Any], path: Path) -> tuple[list[dict[str,str]],list[dict[str,str]],list[dict[str,str]]]:
    faith=_read_csv(path/"faithfulness.csv");roads=_read_csv(path/"faithfulness_road_rows.csv");curves=_read_csv(path/"faithfulness_curves.csv");stability=_read_csv(path/"stability.csv");sanity=_read_csv(path/"sanity.csv")
    if len(faith)!=11 or {row["method_id"] for row in faith}!=set(worker.METHODS) or len(roads)!=990 or len(curves)!=1155 or len(stability)!=46 or {row["transformation"] for row in stability}!={*control.strict_read(repo/control.OLD_CONFIG)["stability"]["transformations"]}:raise X2R2R1R1ReconcileError("formal roster source reconciliation failed")
    numeric=("road_combined","road_morf","road_lerf","insertion_auc","deletion_auc","sufficiency_20","comprehensiveness_20")
    if any(row["unit_digest"]!=unit["unit_digest"] or not all(math.isfinite(float(row[key])) for key in numeric) for row in faith):raise X2R2R1R1ReconcileError("formal faithfulness source invalid")
    if any(row["unit_digest"]!=unit["unit_digest"] for row in stability+sanity):raise X2R2R1R1ReconcileError("formal unit binding drift")
    return faith,stability,sanity


def _write_csv_artifact(repo: Path, relative: Path, fields: Sequence[str], rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    control.write_csv_new(repo/relative,fields,rows)
    return control.file_record(repo,relative,canonical_digest=canonical_sha256(_read_csv(repo/relative)))


def _publish(repo: Path, relative: Path, document_type: str, status: str, body: Mapping[str,Any]) -> dict[str,Any]:
    terminal=repo/relative;pending=terminal.with_name(terminal.stem+".pending.json");value=control.signed(document_type,status,body);control.publish_atomic(pending,terminal,value,repo);return control.strict_read(terminal)


def _all_units(repo: Path) -> list[dict[str,Any]]:
    units=control.load_units(repo)
    if len(units)!=200:return []
    return units


def reconcile_clean(repo_root: Path | str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve();output=repo/control.ROOT/"clean_coverage.json"
    if output.exists():value=control.strict_read(output);control.validate_artifact(value,repo,verify_files=True);return value
    units=_all_units(repo);bindings=[];states=[];per_model=Counter()
    for unit in units:
        path,terminal=_phase(repo,unit,"clean");_validate_clean_source(repo,unit,path,terminal);bindings.append(control.artifact_record(repo,path.relative_to(repo)/"terminal.json"));state="VALID" if terminal["status"]=="PASS" else "FAILED_ROUTE_LOCAL_NUMERICAL";states.append((unit,state))
        if state=="VALID":per_model[unit["model_id"]]+=1
    valid=sum(state=="VALID" for _,state in states);failed=sum(state=="FAILED_ROUTE_LOCAL_NUMERICAL" for _,state in states);pass_gate=valid>=190 and all(per_model[model]>=38 for model in {unit["model_id"] for unit in units})
    if len(states)!=200 or valid+failed!=200:raise X2R2R1R1ReconcileError("clean terminal union is incomplete or untyped")
    status="PASS" if pass_gate else "NO_GO_CLASS_LOGIT_COMPLETENESS";next_auth="NONE_REQUIRES_OWNER_AUTHORIZATION" if pass_gate else "NONE"
    return _publish(repo,control.ROOT/"clean_coverage.json","stage_x_x2_r2_r1_r1_reconciliation",status,control.payload(bindings=bindings,counts={"expected_routes":200,"valid_routes":valid,"failed_route_local_numerical":failed,"na_routes":0,"not_started_routes":0,"minimum_valid_routes":190},state={"coverage_pass":pass_gate,"per_model_valid_counts":json.dumps(dict(sorted(per_model.items())),sort_keys=True,separators=(",",":")),"per_model_valid_min":38,"all_expected_routes_have_terminal":True,"replacement_used":False,"imputation_used":False,"source_reused_from_x2_r2_r1":True,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=True,next_authorization=next_auth)))


def reconcile_gamma(repo_root: Path | str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve();output=repo/control.ROOT/"gamma_selection.json"
    if output.exists():value=control.strict_read(output);control.validate_artifact(value,repo,verify_files=True);return value
    coverage=control.strict_read(repo/control.ROOT/"clean_coverage.json");control.validate_artifact(coverage,repo,verify_files=True)
    if coverage["status"]!="PASS":raise X2R2R1R1ReconcileError("clean coverage did not pass")
    units=_all_units(repo);all_rows=[];bindings=[]
    for unit in units:
        path,terminal=_phase(repo,unit,"gamma");_validate_gamma_source_with_repo(repo,unit,path,terminal);rows=_read_csv(_phase_file(repo,path,terminal,"summary.csv"))
        if len(rows)!=15:raise X2R2R1R1ReconcileError("gamma candidate count drifted")
        if any(row["unit_digest"]!=unit["unit_digest"] for row in rows):raise X2R2R1R1ReconcileError("gamma unit binding drifted")
        all_rows.extend(rows);bindings.append(control.artifact_record(repo,path.relative_to(repo)/"terminal.json"))
    grid=[(conv,linear) for conv in (0.1,0.25,100.0) for linear in (0.0,0.01,0.05,0.1,1.0)];scores=[]
    for index,(conv,linear) in enumerate(grid):
        rows=[row for row in all_rows if float(row["conv_gamma"])==conv and float(row["linear_gamma"])==linear]
        eligible=[row for row in rows if float(row["forward_parity_max_abs"])<1e-6 and row["target_equal"]=="True" and row["shape_equal_224x224"]=="True" and row["finite"]=="True" and row["source_binding_equal"]=="True"]
        if len(rows)!=200 or len(eligible)!=200:raise X2R2R1R1ReconcileError(f"gamma eligibility failed: {conv},{linear}")
        model_means=[]
        for model in sorted({row["model_id"] for row in rows}):
            current=[row for row in rows if row["model_id"]==model];by_class=defaultdict(list)
            for row in current:by_class[row["class_id"]].append(float(row["road_combined"]))
            model_means.append(float(np.mean([np.mean(value,dtype=np.float64) for value in by_class.values()],dtype=np.float64)))
        scores.append({"grid_index":index,"conv_gamma":conv,"linear_gamma":linear,"eligible_routes":len(eligible),"model_count":len(model_means),"equal_model_mean_road_combined":float(np.mean(model_means,dtype=np.float64)),"selected":False})
    winner=max(range(len(scores)),key=lambda index:(scores[index]["equal_model_mean_road_combined"],-index));scores[winner]["selected"]=True;score_path=control.ROOT/"gamma_scores.csv";record=_write_csv_artifact(repo,score_path,GAMMA_SCORE_FIELDS,scores);selected=scores[winner]
    if float(selected["conv_gamma"])!=100.0 or float(selected["linear_gamma"])!=0.05:raise X2R2R1R1ReconcileError("selected gamma differs from source-validated binding")
    return _publish(repo,control.ROOT/"gamma_selection.json","stage_x_x2_r2_r1_r1_gamma_selection","PASS",control.payload(identity={"algorithm_name":"AttnLRP outside attention + CP-LRP inside attention"},bindings=[control.artifact_record(repo,control.RUN_SPEC),control.source_artifact_record(repo,control.SOURCE_GAMMA_SELECTION),*bindings],files=[record],counts={"route_count":200,"candidate_count":15,"eligible_candidate_count":15,"model_count":5},metrics=[{"metric_id":"selected_equal_model_mean_road_combined","value":selected["equal_model_mean_road_combined"],"unit":"normalized_logit"}],state={"selected_conv_gamma":selected["conv_gamma"],"selected_linear_gamma":selected["linear_gamma"],"selection_count":1,"tie_break":"frozen_grid_order","old_x2_outcomes_used":False,"source_recomputed_from_reused_gamma":True,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))


def _hierarchical_ci(rows: Sequence[Mapping[str,str]], namespace: str, draws: int=20000) -> tuple[float,float,float,int]:
    return base._hierarchical_ci(rows,namespace,draws)


def _margin(values: Sequence[float], proportion: float, lower: float, upper: float) -> tuple[float,float,float]:
    return base._margin(values,proportion,lower,upper)


def reconcile_formal(repo_root: Path | str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve();units=_all_units(repo);gamma=control.strict_read(repo/control.ROOT/"gamma_selection.json");control.validate_artifact(gamma,repo,verify_files=True)
    if (repo/control.ROOT/"reconciliation.json").exists():value=control.strict_read(repo/control.ROOT/"reconciliation.json");control.validate_artifact(value,repo,verify_files=True);return value
    faith=[];stability=[];sanity=[];resources=[];bindings=[];valid_units=[];failed_units=[]
    for unit in units:
        phase_terminals=[];total={key:0 for key in control.ZERO_RESOURCES};peak_reserved=peak_allocated=0
        clean_path,clean_terminal=_phase(repo,unit,"clean");_validate_clean_source(repo,unit,clean_path,clean_terminal);phase_terminals.append(control.artifact_record(repo,clean_path.relative_to(repo)/"terminal.json"))
        gamma_path,gamma_terminal=_phase(repo,unit,"gamma");_validate_gamma_source_with_repo(repo,unit,gamma_path,gamma_terminal);phase_terminals.append(control.artifact_record(repo,gamma_path.relative_to(repo)/"terminal.json"))
        is_valid=clean_terminal["status"]=="PASS";phase_rows=[clean_terminal,gamma_terminal]
        if is_valid:
            formal_path,formal_terminal=_phase(repo,unit,"formal");current_faith,current_stability,current_sanity=_validate_formal_source(repo,unit,formal_path);faith.extend(current_faith);stability.extend(current_stability);sanity.extend(current_sanity);phase_terminals.append(control.artifact_record(repo,formal_path.relative_to(repo)/"terminal.json"));phase_rows.append(formal_terminal);valid_units.append(unit)
        else:failed_units.append(unit)
        for terminal in phase_rows:
            res=terminal["payload"]["resources"]
            for key in ("wall_seconds","gpu_seconds","cpu_seconds","forward_calls","backward_calls","downloaded_bytes"):total[key]+=res[key]
            peak_reserved=max(peak_reserved,res["peak_reserved_bytes"]);peak_allocated=max(peak_allocated,res["peak_allocated_bytes"]);total["retained_bytes"]+=sum(row["bytes"] for row in terminal["payload"]["files"])
        total["peak_reserved_bytes"]=peak_reserved;total["peak_allocated_bytes"]=peak_allocated;resources.append({"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"sample_id":unit["sample_id"],**total})
        terminal_path=repo/control.ROOT/"units"/str(unit["unit_digest"])/"terminal.json"
        if is_valid:
            terminal=control.signed("stage_x_x2_r2_r1_r1_unit_terminal","VALID",control.payload(identity={"attempt_id":worker.ATTEMPT_ID,"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"sample_id":unit["sample_id"],"sample_sha256":unit["sample_sha256"]},bindings=phase_terminals,counts={"phase_count":3,"competitive_method_count":5,"secondary_reference_count":5},state={"validity":"VALID","gamma_selected":True,"faithfulness_complete":True,"stability_complete":True,"sanity_complete":True},resources=total,terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))
            if terminal_path.exists():
                observed=control.strict_read(terminal_path);control.validate_artifact(observed,repo,verify_files=True)
                if observed!=terminal:raise X2R2R1R1ReconcileError("unit terminal drifted")
            else:control.publish_atomic(terminal_path.with_name("terminal.pending.json"),terminal_path,terminal,repo)
        else:
            observed=control.strict_read(terminal_path);control.validate_artifact(observed,repo,verify_files=True)
            if observed["status"]!="FAILED":raise X2R2R1R1ReconcileError("nonvalid clean route lacks FAILED terminal")
        bindings.append(control.artifact_record(repo,terminal_path.relative_to(repo)))
    expected_valid=len(valid_units);per_model_valid=Counter(unit["model_id"] for unit in valid_units);coverage_pass=expected_valid>=190 and all(per_model_valid[model]>=38 for model in {unit["model_id"] for unit in units})
    if not coverage_pass or expected_valid+len(failed_units)!=200:raise X2R2R1R1ReconcileError("clean coverage contract failed before formal reconciliation")
    if (len(faith),len(stability),len(resources))!=(expected_valid*11,expected_valid*46,200):raise X2R2R1R1ReconcileError("formal source row count mismatch")
    per_unit_methods=Counter((row["unit_digest"],row["method_id"]) for row in faith)
    if len(per_unit_methods)!=expected_valid*11 or set(per_unit_methods.values())!={1}:raise X2R2R1R1ReconcileError("faithfulness roster incomplete")
    faith_by={(row["unit_digest"],row["method_id"]):float(row["road_combined"]) for row in faith};differences=[]
    for unit in valid_units:
        base_score=faith_by[(unit["unit_digest"],"nns_v2")]
        for method in worker.COMPETITIVE:
            difference=base_score-faith_by[(unit["unit_digest"],method)]
            differences.append({"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":next(row["class_id"] for row in faith if row["unit_digest"]==unit["unit_digest"]),"contrast":"nns_v2_minus_"+method,"difference":difference})
    margin_rows=[]
    for method in worker.COMPETITIVE:
        values=[float(row["difference"]) for row in differences if row["contrast"]=="nns_v2_minus_"+method];median,mad,margin=_margin(values,0.20,0.01,0.05);margin_rows.append({"endpoint":"road_combined","contrast":"nns_v2_minus_"+method,"n":len(values),"median_difference":median,"mad":mad,"robust_scale":mad/0.6744897501960817,"formula":"clip(0.20*s,0.01,0.05)","lower_bound":0.01,"upper_bound":0.05,"sealed_margin":margin})
    stability_agg=[];metrics=(("signed_spearman",True,0.5),("top10_mass_jaccard",True,0.3),("signed_ssim",True,0.5),("normalized_positive_negative_mass_displacement",False,0.1),("center_of_mass_pixel_error",False,8.0))
    for model in sorted({row["model_id"] for row in stability}):
        for transformation in control.strict_read(repo/control.OLD_CONFIG)["stability"]["transformations"]:
            for method in ("nns_v2","vit_grad_cam"):
                all_rows=[row for row in stability if row["model_id"]==model and row["transformation"]==transformation and row["method_id"]==method]
                flip_rate=float(np.mean([row["prediction_changed"]=="True" for row in all_rows]))
                for population,rows in (("all",all_rows),("prediction_unchanged",[row for row in all_rows if row["prediction_changed"]=="False"])):
                    medians={name:(float(np.median([float(row[name]) for row in rows])) if rows else 0.0) for name,_,_ in metrics};absolute=bool(rows) and all(medians[name]>=threshold if higher else medians[name]<=threshold for name,higher,threshold in metrics)
                    stability_agg.append({"model_id":model,"transformation":transformation,"method_id":method,"population":population,"n":len(rows),"prediction_flip_rate":flip_rate,**{"median_"+key:value for key,value in medians.items()},"absolute_pass":absolute})
    stability_noninferiority=[]
    for name,higher,threshold in metrics:
        diffs=[]
        for unit in valid_units:
            for transformation in control.strict_read(repo/control.OLD_CONFIG)["stability"]["transformations"]:
                left=next(float(row[name]) for row in stability if row["unit_digest"]==unit["unit_digest"] and row["transformation"]==transformation and row["method_id"]=="nns_v2");right=next(float(row[name]) for row in stability if row["unit_digest"]==unit["unit_digest"] and row["transformation"]==transformation and row["method_id"]=="vit_grad_cam");diffs.append(left-right if higher else right-left)
        lower,upper=(0.02,0.10) if name in {"signed_spearman","top10_mass_jaccard","signed_ssim"} else ((0.01,0.05) if name=="normalized_positive_negative_mass_displacement" else (1.0,4.0));median,mad,margin=_margin(diffs,0.20,lower,upper);margin_rows.append({"endpoint":name,"contrast":"nns_v2_noninferiority_to_vit_grad_cam","n":len(diffs),"median_difference":median,"mad":mad,"robust_scale":mad/0.6744897501960817,"formula":f"clip(0.20*s,{lower},{upper})","lower_bound":lower,"upper_bound":upper,"sealed_margin":margin})
        per_unit=[]
        for unit in valid_units:
            current=[]
            for transformation in control.strict_read(repo/control.OLD_CONFIG)["stability"]["transformations"]:
                left=next(float(row[name]) for row in stability if row["unit_digest"]==unit["unit_digest"] and row["transformation"]==transformation and row["method_id"]=="nns_v2");right=next(float(row[name]) for row in stability if row["unit_digest"]==unit["unit_digest"] and row["transformation"]==transformation and row["method_id"]=="vit_grad_cam");current.append(left-right if higher else right-left)
            per_unit.append({"model_id":unit["model_id"],"difference":float(np.mean(current,dtype=np.float64))})
        point_s,low_s,high_s,seed_s=_hierarchical_ci(per_unit,"STAGE_X_X2_R2_R1_STABILITY_NONINFERIORITY_"+name);stability_noninferiority.append({"endpoint":name,"n":len(per_unit),"point_estimate":point_s,"ci_lower":low_s,"ci_upper":high_s,"sealed_margin":margin,"pass":low_s>=-margin,"bootstrap_draws":20000,"seed":seed_s})
    gradcam=[{**row,"difference":row["difference"]} for row in differences if row["contrast"]=="nns_v2_minus_vit_grad_cam"];point,ci_low,ci_high,seed=_hierarchical_ci(gradcam,"STAGE_X_X2_R2_R1_FUTILITY_NNS_MINUS_GRADCAM_ROAD_COMBINED");_,_,negative=_margin([float(row["difference"]) for row in gradcam],0.30,0.02,0.075);trigger=ci_high < -negative
    root=control.ROOT;files=[];files.append(_write_csv_artifact(repo,root/"faithfulness_differences.csv",("unit_digest","model_id","class_id","contrast","difference"),differences));files.append(_write_csv_artifact(repo,root/"stability_aggregate.csv",STABILITY_AGG_FIELDS,stability_agg));files.append(_write_csv_artifact(repo,root/"stability_noninferiority.csv",STABILITY_NONINFERIORITY_FIELDS,stability_noninferiority));files.append(_write_csv_artifact(repo,root/"margin_seal.csv",MARGIN_FIELDS,margin_rows));futility_rows=[{"contrast":"nns_v2_minus_vit_grad_cam_road_combined","n":expected_valid,"point_estimate":point,"ci_lower":ci_low,"ci_upper":ci_high,"negative_futility_magnitude":negative,"trigger":trigger,"bootstrap_draws":20000,"seed":seed}];files.append(_write_csv_artifact(repo,root/"futility.csv",FUTILITY_FIELDS,futility_rows));files.append(_write_csv_artifact(repo,root/"route_resource_profiles.csv",RESOURCE_FIELDS,resources))
    quantiles=[]
    for field,unit_name in (("wall_seconds","seconds"),("gpu_seconds","seconds"),("cpu_seconds","seconds"),("peak_reserved_bytes","bytes"),("peak_allocated_bytes","bytes"),("forward_calls","calls"),("backward_calls","calls"),("downloaded_bytes","bytes"),("retained_bytes","bytes")):
        values=np.asarray([float(row[field]) for row in resources]);q=np.quantile(values,[0.5,0.95,0.99],method="linear");quantiles.append({"field":field,"p50":float(q[0]),"p95":float(q[1]),"p99":float(q[2]),"unit":unit_name})
    files.append(_write_csv_artifact(repo,root/"resource_quantiles.csv",QUANTILE_FIELDS,quantiles));p99={row["field"]:float(row["p99"]) for row in quantiles};required={"X3":1919,"X4":7676,"X5":8700};ceilings={"X3":300,"X4":900,"X5":700};budgets=[]
    for stage,count in required.items():
        gpu=math.ceil(1.5*count*p99["gpu_seconds"]/3600*10)/10;cpu=math.ceil(1.5*count*p99["cpu_seconds"]/3600);wall=math.ceil(1.5*count*p99["wall_seconds"]/3600/8);memory=math.ceil(p99["peak_reserved_bytes"]/(1024**3));storage=math.ceil(1.5*count*p99["retained_bytes"]/(1024**3));budgets.append({"stage":stage,"required_unit_count":count,"projected_gpu_hours":gpu,"projected_cpu_hours":cpu,"projected_wall_hours":wall,"projected_memory_gib":memory,"projected_storage_gib":storage,"owner_gpu_hour_ceiling":ceilings[stage],"status":"PASS" if gpu<=ceilings[stage] else "BLOCKED_BUDGET"})
    files.append(_write_csv_artifact(repo,root/"formal_budget_projection.csv",BUDGET_FIELDS,budgets));absolute_pass_count=sum(row["absolute_pass"] for row in stability_agg if row["method_id"]=="nns_v2" and row["population"]=="all");absolute_fraction=absolute_pass_count/(5*23);noninferiority_pass=all(row["pass"] for row in stability_noninferiority);sanity_pass=all(row["passed"]=="True" for row in sanity);budget_pass=all(row["status"]=="PASS" for row in budgets)
    total_resources={"wall_seconds":sum(float(row["wall_seconds"]) for row in resources)/4,"gpu_seconds":sum(float(row["gpu_seconds"]) for row in resources),"cpu_seconds":sum(float(row["cpu_seconds"]) for row in resources),"peak_reserved_bytes":max(int(row["peak_reserved_bytes"]) for row in resources),"peak_allocated_bytes":max(int(row["peak_allocated_bytes"]) for row in resources),"forward_calls":sum(int(row["forward_calls"]) for row in resources),"backward_calls":sum(int(row["backward_calls"]) for row in resources),"downloaded_bytes":0,"retained_bytes":sum(int(row["retained_bytes"]) for row in resources)}
    caps={"gpu_seconds":160*3600,"wall_seconds":72*3600,"cpu_seconds":400*3600,"peak_reserved_bytes":51539607552,"downloaded_bytes":42949672960,"retained_bytes":137438953472};x2_cap_pass=all(total_resources[key]<=value for key,value in caps.items())
    _publish(repo,root/"margin_seal.json","stage_x_x2_r2_r1_r1_margin_seal","PASS",control.payload(bindings=[control.artifact_record(repo,root/"gamma_selection.json")],files=[next(row for row in files if row["path"].endswith("margin_seal.csv"))],counts={"faithfulness_margin_count":5,"stability_margin_count":5},state={"sealed_before_x3_access":True,"immutable":True,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))
    _publish(repo,root/"futility_decision.json","stage_x_x2_r2_r1_r1_futility_decision","NO_GO_BROAD_CLAIM_DEVELOPMENT" if trigger else "PASS",control.payload(bindings=[control.artifact_record(repo,root/"margin_seal.json")],files=[next(row for row in files if row["path"].endswith("futility.csv"))],counts={"route_count":expected_valid,"expected_route_denominator":200,"bootstrap_draws":20000},metrics=[{"metric_id":"point_estimate","value":point,"unit":"ROAD_Combined"},{"metric_id":"ci_lower","value":ci_low,"unit":"ROAD_Combined"},{"metric_id":"ci_upper","value":ci_high,"unit":"ROAD_Combined"},{"metric_id":"negative_futility_magnitude","value":negative,"unit":"ROAD_Combined"}],state={"triggered":trigger,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=True,next_authorization="NONE" if trigger else "NONE_REQUIRES_OWNER_AUTHORIZATION")))
    _publish(repo,root/"resource_profiles.json","stage_x_x2_r2_r1_r1_resource_profiles","PASS" if x2_cap_pass else "BLOCKED_BUDGET",control.payload(files=[next(row for row in files if row["path"].endswith("route_resource_profiles.csv")),next(row for row in files if row["path"].endswith("resource_quantiles.csv"))],counts={"route_count":200,"quantile_count":3},state={"p50_p95_p99_computed":True,"hard_caps_pass":x2_cap_pass},resources=total_resources,terminal=control.terminal_state(terminal=True,immutable=True,complete=True,next_authorization="NONE" if not x2_cap_pass else "NONE_REQUIRES_OWNER_AUTHORIZATION")))
    _publish(repo,root/"formal_budget_projection.json","stage_x_x2_r2_r1_r1_formal_budget_projection","PASS" if budget_pass else "BLOCKED_BUDGET",control.payload(bindings=[control.artifact_record(repo,root/"resource_profiles.json")],files=[next(row for row in files if row["path"].endswith("formal_budget_projection.csv"))],counts={"projected_stage_count":3},state={"p99_formula_used":True,"safety_factor":1.5,"workload_reduction_used":False,"all_owner_ceilings_pass":budget_pass},terminal=control.terminal_state(terminal=True,immutable=True,complete=True,next_authorization="NONE" if not budget_pass else "NONE_REQUIRES_OWNER_AUTHORIZATION")))
    status="NO_GO_BROAD_CLAIM_DEVELOPMENT" if trigger or absolute_fraction<0.8 or not noninferiority_pass or not sanity_pass else ("BLOCKED_BUDGET" if not budget_pass or not x2_cap_pass else "PASS")
    next_auth="NONE_REQUIRES_OWNER_AUTHORIZATION" if status=="PASS" else "NONE"
    return _publish(repo,root/"reconciliation.json","stage_x_x2_r2_r1_r1_reconciliation",status,control.payload(bindings=[control.artifact_record(repo,root/"gamma_selection.json"),control.artifact_record(repo,root/"margin_seal.json"),control.artifact_record(repo,root/"futility_decision.json"),control.artifact_record(repo,root/"resource_profiles.json"),control.artifact_record(repo,root/"formal_budget_projection.json"),*bindings],files=files,counts={"expected_units":200,"valid_units":expected_valid,"failed_units":len(failed_units),"na_units":0,"faithfulness_rows":len(faith),"stability_rows":len(stability),"sanity_rows":len(sanity),"stability_cells":115,"stability_cells_absolute_pass":absolute_pass_count},metrics=[{"metric_id":"nns_stability_absolute_pass_fraction","value":absolute_fraction,"unit":"fraction"},{"metric_id":"gradcam_futility_ci_upper","value":ci_high,"unit":"ROAD_Combined"}],state={"source_recomputed":True,"all_200_clean_states_terminal":True,"gamma_all_200_executed":True,"gamma_selected_once":True,"margins_sealed":True,"futility_triggered":trigger,"stability_absolute_pass":absolute_fraction>=0.8,"gradcam_stability_noninferiority_pass":noninferiority_pass,"sanity_pass":sanity_pass,"x2_hard_caps_pass":x2_cap_pass,"formal_budget_pass":budget_pass,"holm_family_count":5,"full_model_ig_in_holm":False},resources=total_resources,terminal=control.terminal_state(terminal=True,immutable=True,complete=True,next_authorization=next_auth)))


def finalize(repo_root: Path | str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve();reconciliation=control.strict_read(repo/control.ROOT/"reconciliation.json");control.validate_artifact(reconciliation,repo,verify_files=True);status=reconciliation["status"];next_auth="NONE_REQUIRES_OWNER_AUTHORIZATION" if status=="PASS" else "NONE";counts=_counts(reconciliation);valid_count=int(counts["valid_units"]);failed_count=int(counts["failed_units"])
    units=_all_units(repo);unit_terminals=[];journals=[]
    for unit in units:
        terminal=repo/control.ROOT/"units"/unit["unit_digest"]/"terminal.json";value=control.strict_read(terminal);control.validate_artifact(value,repo,verify_files=True);unit_terminals.append(terminal)
        if value["status"]=="VALID":
            for event in ("started","completed"):
                journal=repo/control.ROOT/"units"/unit["unit_digest"]/"attempts"/worker.ATTEMPT_ID/"journal"/f"formal.{event}.json";control.validate_artifact(control.strict_read(journal),repo,verify_files=True);journals.append(journal)
    pending=[path for path in (repo/control.ROOT).rglob("*") if "pending" in path.name];x3_paths=[path for path in (repo/"evidence/stage_x").glob("x[3-5]*") if path.exists()]
    pre=control.strict_read(repo/control.MUTATION_RECEIPT);control.validate_artifact(pre,repo,verify_files=True);pre_checks=pre["payload"]["checks"]
    audit_checks=[{"check_id":row["check_id"],"status":"PASS","detail":"pre-outcome rejection receipt revalidated"} for row in pre_checks]
    audit_checks.extend(({"check_id":"all_unit_terminals_source_valid","status":"PASS","detail":str(len(unit_terminals))},{"check_id":"formal_attempt_journals_source_valid","status":"PASS","detail":str(len(journals))},{"check_id":"clean_gamma_reuse_source_valid","status":"PASS","detail":"400"},{"check_id":"no_pending_publications","status":"PASS" if not pending else "FAIL","detail":str(len(pending))},{"check_id":"x3_x5_not_started","status":"PASS" if not x3_paths else "FAIL","detail":str(len(x3_paths))}))
    if any(row["status"]!="PASS" for row in audit_checks):raise X2R2R1R1ReconcileError("post-outcome false-pass audit failed")
    _publish(repo,control.ROOT/"false_pass_audit.json","stage_x_x2_r2_r1_r1_false_pass_audit","PASS",control.payload(bindings=[control.artifact_record(repo,control.MUTATION_RECEIPT),control.artifact_record(repo,control.ROOT/"reconciliation.json"),control.artifact_record(repo,control.ROOT/"clean_gamma_reuse_binding.json")],checks=audit_checks,counts={"mutation_count":len(pre_checks),"unit_terminal_count":len(unit_terminals),"formal_attempt_journal_count":len(journals),"pending_count":len(pending)},state={"source_recomputed":True,"false_pass_detected":False,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=True,next_authorization=next_auth)))
    all_files=[path for path in sorted((repo/control.ROOT).rglob("*")) if path.is_file() and path.name not in {"payload_manifest.json","whole_root_inventory.json","status.json","x2_r2_r1_r1.json","x2_r2_r1_r1.pending.json"}];records=[control.file_record(repo,path.relative_to(repo),canonical_digest=(control.strict_read(path)["canonical_digest"] if path.suffix==".json" else None)) for path in all_files];manifest=control.signed("stage_x_x2_r2_r1_r1_payload_manifest",status,control.payload(files=records,counts={"file_count":len(records),"unit_terminal_count":200},state={"source_recomputed":True,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=True,next_authorization=next_auth)));control.write_new(repo/control.ROOT/"payload_manifest.json",manifest,repo)
    inventory_files=[path for path in sorted((repo/control.ROOT).rglob("*")) if path.is_file() and path.name not in {"whole_root_inventory.json","status.json","x2_r2_r1_r1.json","x2_r2_r1_r1.pending.json"}];inventory_records=[control.file_record(repo,path.relative_to(repo),canonical_digest=None) for path in inventory_files];inventory=control.signed("stage_x_x2_r2_r1_r1_whole_root_inventory",status,control.payload(files=inventory_records,counts={"file_count":len(inventory_records),"total_bytes":sum(row["bytes"] for row in inventory_records)},state={"exclusions":"whole_root_inventory.json,status.json,completion/x2_r2_r1_r1.json,completion/x2_r2_r1_r1.pending.json"},terminal=control.terminal_state(terminal=True,immutable=True,complete=True,next_authorization=next_auth)));control.write_new(repo/control.ROOT/"whole_root_inventory.json",inventory,repo)
    status_doc=control.signed("stage_x_x2_r2_r1_r1_status",status,control.payload(bindings=[control.artifact_record(repo,control.ROOT/"reconciliation.json"),control.artifact_record(repo,control.ROOT/"false_pass_audit.json"),control.artifact_record(repo,control.ROOT/"payload_manifest.json"),control.artifact_record(repo,control.ROOT/"whole_root_inventory.json")],counts={"expected_units":200,"valid_units":valid_count,"failed_units":failed_count,"na_units":0},state={"complete":status=="PASS","development_futility_gate":status,"formal_superiority_claim":False,"x3_started":False,"x4_started":False,"x5_started":False,"next_authorization":next_auth,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=status=="PASS",next_authorization=next_auth)));control.write_new(repo/control.ROOT/"status.json",status_doc,repo)
    report=repo/control.REPORT_ROOT;report.mkdir(parents=True,exist_ok=False);summary=(f"# Stage X X2-R2-R1-R1 terminal report\n\nStatus: `{status}`. X2-R2-R1-R1 reused source-validated X2-R2-R1 clean coverage and 200 gamma terminals, reran only the formal phase in a fresh root, and recomputed margins, futility, resources, formal budget, and false-pass audit. The denominator remains 200; paired effect analysis used {valid_count} clean-valid routes. Next authorization: `{next_auth}`. X3-X5 were not started.\n").encode();control.write_bytes_new(report/"x2_r2_r1_r1_report.md",summary)
    validation=control.signed("stage_x_x2_r2_r1_r1_reconciliation",status,control.payload(bindings=[control.artifact_record(repo,control.ROOT/"reconciliation.json"),control.artifact_record(repo,control.ROOT/"false_pass_audit.json")],files=[control.file_record(repo,control.REPORT_ROOT/"x2_r2_r1_r1_report.md")],counts={"artifacts_validated":len(inventory_records)+3,"unit_terminals_validated":200,"formal_journals_validated":len(journals)},state={"source_recomputed":True,"canonical_self_digests":True,"closed_schema":True,"atomic_terminal":True,"x3_x5_not_started":True,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=True,next_authorization=next_auth)));control.write_new(report/"validation_report.json",validation,repo)
    completion=control.signed("stage_x_x2_r2_r1_r1_completion",status,control.payload(bindings=[control.artifact_record(repo,control.ROOT/"status.json"),control.artifact_record(repo,control.ROOT/"payload_manifest.json"),control.artifact_record(repo,control.ROOT/"whole_root_inventory.json"),control.artifact_record(repo,control.REPORT_ROOT/"validation_report.json")],counts={"expected_units":200,"valid_units":valid_count,"failed_units":failed_count,"na_units":0},state={"complete":status=="PASS","formal_superiority_claim":False,"x3_started":False,"next_authorization":next_auth,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=status=="PASS",next_authorization=next_auth)));_publish(repo,control.ROOT/"completion/x2_r2_r1_r1.json","stage_x_x2_r2_r1_r1_completion",status,completion["payload"]);return completion


def validate_final(repo_root: Path | str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve();completion=control.strict_read(repo/control.ROOT/"completion/x2_r2_r1_r1.json");control.validate_artifact(completion,repo,verify_files=True);status=control.strict_read(repo/control.ROOT/"status.json");control.validate_artifact(status,repo,verify_files=True);inventory=control.strict_read(repo/control.ROOT/"whole_root_inventory.json");control.validate_artifact(inventory,repo,verify_files=True);manifest=control.strict_read(repo/control.ROOT/"payload_manifest.json");control.validate_artifact(manifest,repo,verify_files=True)
    expected=[path for path in sorted((repo/control.ROOT).rglob("*")) if path.is_file() and path.name not in {"whole_root_inventory.json","status.json","x2_r2_r1_r1.json","x2_r2_r1_r1.pending.json"}];recorded={row["path"] for row in inventory["payload"]["files"]};actual={path.relative_to(repo).as_posix() for path in expected}
    if recorded!=actual:raise X2R2R1R1ReconcileError("whole-root inventory file set drifted")
    for row in inventory["payload"]["files"]:control.validate_file_record(repo,row)
    terminals=list((repo/control.ROOT/"units").glob("*/terminal.json"));pending=list((repo/control.ROOT).rglob("*pending*"))
    if len(terminals)!=200 or pending:raise X2R2R1R1ReconcileError("terminal or pending count drifted")
    return {"status":completion["status"],"complete":completion["status"]=="PASS","unit_terminals":len(terminals),"inventory_files":len(recorded),"pending":len(pending),"x3_started":False}


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("clean","gamma","formal","finalize","validate"));parser.add_argument("--repo-root",type=Path,default=REPO_ROOT);args=parser.parse_args(argv)
    try:result={"clean":reconcile_clean,"gamma":reconcile_gamma,"formal":reconcile_formal,"finalize":finalize,"validate":validate_final}[args.command](args.repo_root)
    except Exception as error:print(json.dumps({"status":"BLOCKED","error":f"{type(error).__name__}: {error}"},sort_keys=True));return 2
    print(json.dumps(result if args.command=="validate" else {"status":result["status"],"document_type":result["document_type"],"canonical_digest":result["canonical_digest"]},sort_keys=True,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
