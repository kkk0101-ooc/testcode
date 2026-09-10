"""Deterministic, resumable X2-R2 GPU worker."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_x_x2_r2 as control

ATTEMPT_ID = "attempt-000001"
METHODS = ("nns_v2","vit_grad_cam","transformer_attribution_chefer","attnlrp","rise","occlusion","attention_rollout","raw_cls_attention","random_map","center_prior","edge_input_gradient_prior")
COMPETITIVE = ("vit_grad_cam","transformer_attribution_chefer","attnlrp","rise","occlusion")
SECONDARY = ("attention_rollout","raw_cls_attention","random_map","center_prior","edge_input_gradient_prior")
CLEAN_SUMMARY_FIELDS = ("unit_digest","model_id","sample_id","sample_sha256","checkpoint_sha256","canonical_rgb_sha256","clean_tensor_sha256","predicted_class","true_class","runner_up_class","predicted_logit","true_logit","runner_up_logit","clean_margin","clean_logits_sha256","selected_steps","input_delta","input_attribution_sum","input_absolute_error","identity_max_abs_error","complete_cut_count","complete_cut_all_pass","native_pixel_provenance")
CUT_FIELDS = ("unit_digest","cut_id","component_count","pixel_max_abs_error","total_abs_error","passed")
ADAPTER_FIELDS = ("unit_digest","method_id","target_index","class_targeted","shape","finite","source_digest","target_semantics")
GAMMA_SUMMARY_FIELDS = ("unit_digest","model_id","class_id","conv_gamma","linear_gamma","forward_parity_max_abs","target_equal","shape_equal_224x224","finite","source_binding_equal","source_digest","road_combined","road_morf","road_lerf","elapsed_seconds")
GAMMA_ROAD_FIELDS = ("unit_digest","conv_gamma","linear_gamma","percentile","seed_index","direction","pcg64_seed","normalized_response")
FAITH_FIELDS = ("unit_digest","model_id","class_id","method_id","road_combined","road_morf","road_lerf","insertion_auc","deletion_auc","sufficiency_20","comprehensiveness_20")
FAITH_ROAD_FIELDS = ("unit_digest","method_id","percentile","seed_index","direction","pcg64_seed","normalized_response")
FAITH_CURVE_FIELDS = ("unit_digest","method_id","repeat","seed","fraction","actual_fraction","insertion","deletion")
STABILITY_FIELDS = ("unit_digest","model_id","class_id","method_id","transformation","signed_spearman","top10_mass_jaccard","signed_ssim","normalized_positive_negative_mass_displacement","center_of_mass_pixel_error","prediction_changed","transformed_target_logit")
SANITY_FIELDS = ("unit_digest","model_id","class_id","check_id","target_class","value","reference","passed")
CLEAN_FAILURE_FIELDS = ("unit_digest","steps","input_abs_error","identity_max_abs_error","all_cuts_pass","passed")


class X2R2WorkerError(ValueError):
    pass


def _records(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {row["name"]: row["value"] for row in value["payload"][name]}


def _model_spec(repo: Path, model_id: str) -> Mapping[str, Any]:
    source=control.strict_read(repo/control.OLD_CONFIG)
    matches=[row for row in source["models"] if row["model_id"]==model_id]
    if len(matches)!=1: raise X2R2WorkerError("model identity not frozen")
    return matches[0]


def _load_model(repo: Path, spec: Mapping[str, Any], device: str) -> Any:
    model,_=control._load_state(repo,spec,fused=False)
    return model.to(device).eval()


def _load_image(repo: Path, sample: Mapping[str, Any], spec: Mapping[str, Any], device: str) -> tuple[Any,dict[str,Any]]:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform
    from .stage_c_perturbations import tensor_sha256
    path=repo/"assets/data"/str(sample["source_object_id"])
    if path.stat().st_size!=int(sample["file_bytes"]) or control.sha_file(path)!=sample["file_sha256"]:
        raise X2R2WorkerError("frozen sample bytes drifted")
    preprocess=spec["preprocess"]
    with Image.open(path) as image:
        rgb=image.convert(str(preprocess["color_mode"]));resized=transform.resize(rgb,int(preprocess["resize_shorter_side"]),interpolation=InterpolationMode.BICUBIC,antialias=bool(preprocess["antialias"]));cropped=transform.center_crop(resized,list(preprocess["center_crop"]));canonical=transform.pil_to_tensor(cropped).float().div(255).unsqueeze(0)
    mean=torch.tensor(preprocess["mean"]).reshape(1,3,1,1);std=torch.tensor(preprocess["std"]).reshape(1,3,1,1);clean=((canonical-mean)/std).to(device)
    return clean,{"canonical_rgb_sha256":tensor_sha256(canonical),"clean_tensor_sha256":tensor_sha256(clean)}


def _phase_dir(repo: Path, unit: Mapping[str, Any], phase: str) -> Path:
    return repo/control.ROOT/"units"/str(unit["unit_digest"])/"attempts"/ATTEMPT_ID/phase


def _journal(repo: Path, unit: Mapping[str, Any], phase: str, event: str, checkpoint_sha: str) -> Path:
    return repo/control.ROOT/"units"/str(unit["unit_digest"])/"attempts"/ATTEMPT_ID/"journal"/f"{phase}.{event}.json"


def _journal_value(repo: Path, unit: Mapping[str, Any], phase: str, event: str, checkpoint_sha: str) -> dict[str, Any]:
    return control.signed("stage_x_x2_r2_attempt_journal","PENDING" if event=="started" else "PASS",control.payload(
        identity={"attempt_id":ATTEMPT_ID,"event":event,"phase":phase,"unit_digest":unit["unit_digest"]},
        bindings=[control.artifact_record(repo,control.RUN_SPEC),control.artifact_record(repo,control.ROOT/"implementation_seal.json"),control.artifact_record(repo,control.ROOT/"model_materialization.json"),control.artifact_record(repo,control.ROOT/"development_unit_manifest.json"),control.artifact_record(repo,control.ROOT/"development_sample_manifest.json")],
        state={"sample_sha256":unit["sample_sha256"],"checkpoint_sha256":checkpoint_sha,"resume_source_validated":True},
        terminal=control.terminal_state(terminal=event=="completed",immutable=True,complete=event=="completed"),
    ))


def _start_journal(repo: Path, unit: Mapping[str, Any], phase: str, checkpoint_sha: str) -> None:
    path=_journal(repo,unit,phase,"started",checkpoint_sha)
    expected=_journal_value(repo,unit,phase,"started",checkpoint_sha)
    if path.exists():
        observed=control.strict_read(path);control.validate_artifact(observed,repo,verify_files=True)
        if observed!=expected: raise X2R2WorkerError("attempt start journal binding drifted")
    else: control.write_new(path,expected,repo)


def _complete_journal(repo: Path, unit: Mapping[str, Any], phase: str, checkpoint_sha: str) -> None:
    path=_journal(repo,unit,phase,"completed",checkpoint_sha)
    if not path.exists(): control.write_new(path,_journal_value(repo,unit,phase,"completed",checkpoint_sha),repo)


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    output=io.StringIO(newline="");writer=csv.DictWriter(output,fieldnames=list(fields),lineterminator="\n",extrasaction="raise");writer.writeheader()
    for row in rows:
        if set(row)!=set(fields): raise X2R2WorkerError("CSV field mismatch")
        writer.writerow(row)
    return output.getvalue().encode()


def _write_temp_bytes(path: Path, data: bytes) -> None:
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(descriptor,"wb") as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())


def _save_array(path: Path, value: Any) -> None:
    array=np.ascontiguousarray(value,dtype=np.float32)
    if not np.isfinite(array).all():raise X2R2WorkerError("scientific array is non-finite")
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(descriptor,"wb") as stream:np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())


def _file_record_from_temp(repo: Path, temp: Path, final: Path, *, array: bool=False) -> dict[str, Any]:
    relative=final.relative_to(repo)
    if array:
        from .stage_f_f3c1_kernels import array_digest
        value=np.load(temp,allow_pickle=False)
        return {"kind":"array","path":relative.as_posix(),"bytes":temp.stat().st_size,"sha256":control.sha_file(temp),"canonical_digest":array_digest(value),"dtype":str(value.dtype),"shape":list(value.shape)}
    return {"kind":"file","path":relative.as_posix(),"bytes":temp.stat().st_size,"sha256":control.sha_file(temp),"canonical_digest":None}


def _publish_phase(repo: Path, unit: Mapping[str, Any], phase: str, rows: Sequence[tuple[str,Sequence[str],Sequence[Mapping[str,Any]]]], arrays: Mapping[str,Any], resources: Mapping[str,Any], counts: Mapping[str,Any], state: Mapping[str,Any], *, status: str="PASS") -> dict[str,Any]:
    final=_phase_dir(repo,unit,phase)
    if final.exists():
        terminal=control.strict_read(final/"terminal.json");control.validate_artifact(terminal,repo,verify_files=True);return terminal
    final.parent.mkdir(parents=True,exist_ok=True); temporary=Path(tempfile.mkdtemp(prefix=f".{phase}.",dir=final.parent)); files=[]
    try:
        for name,fields,data in rows:
            path=temporary/name;_write_temp_bytes(path,_csv_bytes(fields,data));files.append(_file_record_from_temp(repo,path,final/name))
        for name,value in sorted(arrays.items()):
            path=temporary/name;_save_array(path,value);files.append(_file_record_from_temp(repo,path,final/name,array=True))
        pending=control.signed("stage_x_x2_r2_phase_manifest","PENDING",control.payload(identity={"attempt_id":ATTEMPT_ID,"phase":phase,"unit_digest":unit["unit_digest"]},bindings=[control.artifact_record(repo,control.RUN_SPEC),control.artifact_record(repo,control.ROOT/"implementation_seal.json")],files=files,counts=counts,state=state,resources=resources,terminal=control.terminal_state()))
        actual_resources=dict(resources);actual_resources["retained_bytes"]=sum(int(row["bytes"]) for row in files)
        terminal=control.signed("stage_x_x2_r2_phase_manifest",status,control.payload(identity={"attempt_id":ATTEMPT_ID,"phase":phase,"unit_digest":unit["unit_digest"]},bindings=[control.artifact_record(repo,control.RUN_SPEC),control.artifact_record(repo,control.ROOT/"implementation_seal.json")],files=files,counts=counts,state=state,resources=actual_resources,terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))
        control.validate_artifact(pending,repo);control.validate_artifact(terminal,repo)
        _write_temp_bytes(temporary/"terminal.pending.json",canonical_json_bytes(pending)+b"\n");_write_temp_bytes(temporary/"terminal.json.atomic",canonical_json_bytes(terminal)+b"\n");os.replace(temporary/"terminal.json.atomic",temporary/"terminal.json");(temporary/"terminal.pending.json").unlink();os.replace(temporary,final);return terminal
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def _resource(started: float, cpu_started: float, device: str, forward_calls: int, backward_calls: int, retained_bytes: int=0) -> dict[str,Any]:
    import torch
    elapsed=time.monotonic()-started
    return {"wall_seconds":elapsed,"gpu_seconds":elapsed,"cpu_seconds":time.process_time()-cpu_started,"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),"forward_calls":forward_calls,"backward_calls":backward_calls,"downloaded_bytes":0,"retained_bytes":retained_bytes}


def _summary_map(value: Mapping[str, Any]) -> np.ndarray:
    from .stage_x_x2_metrics import pixel_map
    return pixel_map(value["attribution"]).numpy().astype(np.float32,copy=False)


def run_clean_unit(repo: Path, model: Any, spec: Mapping[str, Any], sample: Mapping[str, Any], unit: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch
    from . import stage_x_comparators as comparators
    from . import stage_x_nns_v2 as nns
    final=_phase_dir(repo,unit,"clean");checkpoint_sha=control.sha_file(repo/control.MODEL_LOCAL_PATHS[str(spec["id"])]);
    if final.exists():
        terminal=control.strict_read(final/"terminal.json");control.validate_artifact(terminal,repo,verify_files=True);return {"status":"REUSED_OR_EXISTING","unit":unit["unit_digest"]}
    _start_journal(repo,unit,"clean",checkpoint_sha)
    started=time.monotonic();cpu_started=time.process_time();torch.cuda.reset_peak_memory_stats(device);forward_calls=0
    handle=model.register_forward_pre_hook(lambda _m,_a: None)
    try:
        image,image_identity=_load_image(repo,sample,spec,device);baseline=torch.zeros_like(image)
        result=nns.explain(model,image,true_class=int(sample["class_id"]),baseline=baseline,steps=tuple(control.strict_read(repo/control.OLD_CONFIG)["execution"]["steps"]),alpha_chunk=4,component_chunk=8)
        identity=comparators.full_model_ig_control(model,image,baseline,result.target.predicted_class,result.steps);identity_check=comparators.validate_identity(result,identity)
        arrays={"nns_v2_rgb.npy":result.attribution.numpy(),"nns_v2.npy":result.signed_rgb_map.numpy(),"full_model_ig_rgb.npy":identity["attribution"].numpy(),"full_model_ig.npy":identity["signed_rgb_map"].numpy()}
        adapter_rows=[]
        functions={"vit_grad_cam":lambda:comparators.vit_grad_cam(model,image,result.target.predicted_class),"transformer_attribution_chefer":lambda:comparators.chefer(model,image,result.target.predicted_class),"rise":lambda:comparators.rise(model,image,result.target.predicted_class,namespace=control.RUN_ID+"|rise|"+str(unit["unit_digest"])+"|"),"occlusion":lambda:comparators.occlusion(model,image,result.target.predicted_class)}
        for name,function in functions.items():
            value=function(); array=_summary_map(value);arrays[name+".npy"]=array;source=canonical_sha256({key:item for key,item in value.items() if key not in {"attribution","elapsed_seconds"}})
            adapter_rows.append({"unit_digest":unit["unit_digest"],"method_id":name,"target_index":value["target_index"],"class_targeted":True,"shape":"x".join(map(str,array.shape)),"finite":bool(np.isfinite(array).all()),"source_digest":source,"target_semantics":value["target_semantics"]})
        secondaries=(("attention_rollout",comparators.attention_rollout(model,image)),("raw_cls_attention",comparators.raw_cls_attention(model,image)),("random_map",comparators.deterministic_random_map(224,224,str(unit["unit_digest"]))),("center_prior",comparators.center_prior(224,224)))
        for name,value in secondaries:
            arrays[name+".npy"]=_summary_map(value)
        edge=comparators.edge_input_gradient_prior(model,image);arrays["edge_input_gradient_prior.npy"]=_summary_map({"attribution":edge["edge_attribution"]});arrays["input_gradient_diagnostic.npy"]=_summary_map({"attribution":edge["input_gradient_attribution"]})
        target=result.target; identity_max_abs=float((identity["attribution"]-result.attribution).abs().max())
        clean_row={"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"sample_id":unit["sample_id"],"sample_sha256":unit["sample_sha256"],"checkpoint_sha256":checkpoint_sha,**image_identity,"predicted_class":target.predicted_class,"true_class":target.true_class,"runner_up_class":target.runner_up_class,"predicted_logit":target.predicted_logit,"true_logit":target.true_logit,"runner_up_logit":target.runner_up_logit,"clean_margin":target.clean_margin,"clean_logits_sha256":target.clean_logits_sha256,"selected_steps":result.steps,"input_delta":result.delta,"input_attribution_sum":result.input_sum,"input_absolute_error":result.input_abs_error,"identity_max_abs_error":identity_max_abs,"complete_cut_count":len(result.cuts),"complete_cut_all_pass":all(row.passed for row in result.cuts.values()),"native_pixel_provenance":True}
        cut_rows=[{"unit_digest":unit["unit_digest"],"cut_id":key,"component_count":value.component_count,"pixel_max_abs_error":value.pixel_max_abs_error,"total_abs_error":value.total_abs_error,"passed":value.passed} for key,value in sorted(result.cuts.items())]
        chunks=sum(math.ceil(int(row["steps"])/4) for row in result.attempts);forward_calls=139+2*chunks
        backward_calls=3+19*chunks
        resources=_resource(started,cpu_started,device,forward_calls,backward_calls)
        terminal=_publish_phase(repo,unit,"clean",[("summary.csv",CLEAN_SUMMARY_FIELDS,[clean_row]),("cuts.csv",CUT_FIELDS,cut_rows),("adapters.csv",ADAPTER_FIELDS,adapter_rows)],arrays,resources,{"complete_cut_count":len(cut_rows),"competitive_without_attnlrp_count":4,"secondary_reference_count":5,"array_count":len(arrays)},{"status":"VALID","identity_control_pass":True,"complete_cuts_pass":True,"native_pixel_provenance":True,"clean_phase_only":True})
        _complete_journal(repo,unit,"clean",checkpoint_sha);return {"status":"VALID","unit":unit["unit_digest"],"steps":result.steps,"wall_seconds":resources["wall_seconds"]}
    finally:
        handle.remove()


def _read_one_csv(path: Path) -> dict[str,str]:
    with path.open("r",encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=1:raise X2R2WorkerError(f"expected one CSV row: {path}")
    return rows[0]


def _phase_file(repo: Path, phase: Path, name: str) -> Path:
    local=phase/name
    if local.is_file():return local
    terminal=control.strict_read(phase/"terminal.json")
    matches=[repo/row["path"] for row in terminal["payload"]["files"] if Path(row["path"]).name==name]
    if len(matches)!=1:raise X2R2WorkerError(f"phase file not uniquely bound: {name}")
    return matches[0]


def run_gamma_unit(repo: Path, model: Any, spec: Mapping[str, Any], sample: Mapping[str, Any], unit: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch
    from . import stage_x_comparators as comparators
    from . import stage_x_nns_v2 as nns
    from . import stage_x_x2_metrics as metrics
    clean=_phase_dir(repo,unit,"clean");clean_terminal=control.strict_read(clean/"terminal.json");control.validate_artifact(clean_terminal,repo,verify_files=True)
    checkpoint_sha=control.sha_file(repo/control.MODEL_LOCAL_PATHS[str(spec["id"])]);image,_=_load_image(repo,sample,spec,device)
    if clean_terminal["status"]=="PASS":summary=_read_one_csv(_phase_file(repo,clean,"summary.csv"));target=int(summary["predicted_class"])
    else:target=nns.resolve_target(model,image,true_class=int(sample["class_id"])).predicted_class
    _start_journal(repo,unit,"gamma",checkpoint_sha);final=_phase_dir(repo,unit,"gamma")
    if final.exists():
        terminal=control.strict_read(final/"terminal.json");control.validate_artifact(terminal,repo,verify_files=True);_complete_journal(repo,unit,"gamma",checkpoint_sha);return {"status":"EXISTING","unit":unit["unit_digest"]}
    started=time.monotonic();cpu_started=time.process_time();torch.cuda.reset_peak_memory_stats(device);rows=[];road_rows=[];arrays={};forward_calls=0;backward_calls=0;source_digest=None
    for conv in comparators.ATTNLRP_CONV_GAMMAS:
        for linear in comparators.ATTNLRP_LINEAR_GAMMAS:
            began=time.monotonic();value=comparators.attnlrp(model,image,target,conv_gamma=conv,linear_gamma=linear);array=_summary_map(value);name=f"attnlrp_c{conv:g}_l{linear:g}.npy";arrays[name]=array
            current_source=canonical_sha256(value["official_sources"]);source_digest=current_source if source_digest is None else source_digest
            road=metrics.road_combined(model,image,value["attribution"],target,str(unit["unit_digest"])+f"|gamma|{conv:g}|{linear:g}")
            rows.append({"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":sample["class_id"],"conv_gamma":conv,"linear_gamma":linear,"forward_parity_max_abs":value["forward_parity_max_abs"],"target_equal":value["target_index"]==target,"shape_equal_224x224":list(array.shape)==[224,224],"finite":bool(np.isfinite(array).all()),"source_binding_equal":current_source==source_digest,"source_digest":current_source,"road_combined":road["road_combined"],"road_morf":road["road_morf"],"road_lerf":road["road_lerf"],"elapsed_seconds":time.monotonic()-began})
            road_rows.extend({"unit_digest":unit["unit_digest"],"conv_gamma":conv,"linear_gamma":linear,**item} for item in road["rows"]);forward_calls+=2+road["forward_calls"];backward_calls+=1
    resources=_resource(started,cpu_started,device,forward_calls,backward_calls)
    _publish_phase(repo,unit,"gamma",[("summary.csv",GAMMA_SUMMARY_FIELDS,rows),("road_rows.csv",GAMMA_ROAD_FIELDS,road_rows)],arrays,resources,{"candidate_count":15,"road_row_count":len(road_rows),"array_count":15},{"status":"VALID","all_candidates_retained":True,"all_forward_lt_1e_6":all(float(row["forward_parity_max_abs"])<1e-6 for row in rows),"all_target_shape_finite_source_pass":all(row["target_equal"] and row["shape_equal_224x224"] and row["finite"] and row["source_binding_equal"] for row in rows)})
    _complete_journal(repo,unit,"gamma",checkpoint_sha);return {"status":"VALID","unit":unit["unit_digest"],"wall_seconds":resources["wall_seconds"]}


def _selected_gamma(repo: Path) -> tuple[float,float]:
    path=repo/control.ROOT/"gamma_selection.json";value=control.strict_read(path);control.validate_artifact(value,repo,verify_files=True);state=_records(value,"state")
    return float(state["selected_conv_gamma"]),float(state["selected_linear_gamma"])


def _load_map(path: Path) -> np.ndarray:
    value=np.load(path,allow_pickle=False)
    if value.dtype!=np.float32 or list(value.shape)!=[224,224] or not np.isfinite(value).all():raise X2R2WorkerError(f"map drift: {path}")
    return value


def _delete_support(image: Any, score: np.ndarray, *, positive: bool) -> Any:
    import torch
    flat=score.reshape(-1); count=max(1,int(math.ceil(0.2*flat.size)));eligible=np.flatnonzero(flat>0 if positive else flat<0)
    if not len(eligible):return image.clone()
    chosen=eligible[np.argsort(flat[eligible])[-count:] if positive else np.argsort(flat[eligible])[:count]];mask=torch.ones(flat.size,device=image.device,dtype=image.dtype);mask[torch.as_tensor(chosen,device=image.device)]=0;return image*mask.reshape(1,1,*score.shape)


def _randomize_module(module: Any, seed: int, device: str) -> None:
    import torch
    devices=[int(device.split(":",1)[1])] if device.startswith("cuda:") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices: torch.cuda.manual_seed_all(seed)
        for child in module.modules():
            if hasattr(child,"reset_parameters"):
                child.reset_parameters()


def run_formal_unit(repo: Path, model: Any, spec: Mapping[str, Any], sample: Mapping[str, Any], unit: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch
    from . import stage_x_comparators as comparators
    from . import stage_x_nns_v2 as nns
    from . import stage_x_x2_metrics as metrics
    clean=_phase_dir(repo,unit,"clean");gamma=_phase_dir(repo,unit,"gamma")
    for base in (clean,gamma):control.validate_artifact(control.strict_read(base/"terminal.json"),repo,verify_files=True)
    summary=_read_one_csv(_phase_file(repo,clean,"summary.csv"));target=int(summary["predicted_class"]);selected_steps=int(summary["selected_steps"]);checkpoint_sha=summary["checkpoint_sha256"];_start_journal(repo,unit,"formal",checkpoint_sha);final=_phase_dir(repo,unit,"formal")
    if final.exists():
        terminal=control.strict_read(final/"terminal.json");control.validate_artifact(terminal,repo,verify_files=True);_complete_journal(repo,unit,"formal",checkpoint_sha);return {"status":"EXISTING","unit":unit["unit_digest"]}
    started=time.monotonic();cpu_started=time.process_time();torch.cuda.reset_peak_memory_stats(device);image,_=_load_image(repo,sample,spec,device);conv,linear=_selected_gamma(repo);gamma_name=f"attnlrp_c{conv:g}_l{linear:g}.npy"
    maps={name:_load_map(_phase_file(repo,clean,name+".npy")) for name in METHODS if name not in {"attnlrp"}};sealed_gamma_map=_load_map(gamma/gamma_name);selected_gamma=comparators.attnlrp(model,image,target,conv_gamma=conv,linear_gamma=linear);selected_gamma_map=_summary_map(selected_gamma)
    if not np.allclose(selected_gamma_map,sealed_gamma_map,atol=1e-6,rtol=1e-6):raise X2R2WorkerError("sealed gamma recomputation drifted")
    maps["attnlrp"]=selected_gamma_map
    faith=[];road_rows=[];curve_rows=[];forward_calls=0
    for method in METHODS:
        attribution=torch.from_numpy(maps[method]).to(device);identity=str(unit["unit_digest"])+"|formal|"+method;road=metrics.road_combined(model,image,attribution,target,identity);curve=metrics.insertion_deletion(model,image,attribution,target,identity)
        faith.append({"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":sample["class_id"],"method_id":method,"road_combined":road["road_combined"],"road_morf":road["road_morf"],"road_lerf":road["road_lerf"],**curve["means"]})
        road_rows.extend({"unit_digest":unit["unit_digest"],"method_id":method,**row} for row in road["rows"]);
        for record in curve["repeats"]:
            curve_rows.extend({"unit_digest":unit["unit_digest"],"method_id":method,"repeat":record["repeat"],"seed":record["seed"],"fraction":metrics.FRACTIONS[index],"actual_fraction":record["actual_fractions"][index],"insertion":record["insertion"][index],"deletion":record["deletion"][index]} for index in range(len(metrics.FRACTIONS)))
        forward_calls+=road["forward_calls"]+curve["forward_calls"]
    clean_nns=maps["nns_v2"];stability=[];backward_calls=1;forward_calls+=1
    for transformation in control.strict_read(repo/control.OLD_CONFIG)["stability"]["transformations"]:
        changed,metadata=metrics.transform_image(image,transformation,str(unit["unit_digest"]),spec["preprocess"]);
        with torch.no_grad():changed_logits=model(changed);prediction_changed=int(changed_logits.argmax())!=target;transformed_logit=float(changed_logits[0,target])
        nns_map=nns.input_ig(model,changed,torch.zeros_like(changed),target,selected_steps,4).sum(0);grad=comparators.vit_grad_cam(model,changed,target)["attribution"]
        for method,value in (("nns_v2",nns_map),("vit_grad_cam",grad)):
            restored,mask=metrics.inverse_map(value,metadata);scores=metrics.map_metrics(maps[method],restored,mask)
            stability.append({"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":sample["class_id"],"method_id":method,"transformation":transformation,**scores,"prediction_changed":prediction_changed,"transformed_target_logit":transformed_logit})
        forward_calls+=1+math.ceil(selected_steps/4)+1;backward_calls+=math.ceil(selected_steps/4)+1
    baseline=torch.zeros_like(image);runner=int(summary["runner_up_class"]);runner_map=nns.input_ig(model,image,baseline,runner,selected_steps,4).sum(0).numpy();true=int(summary["true_class"]);true_map=nns.input_ig(model,image,baseline,true,selected_steps,4).sum(0).numpy() if true!=target else clean_nns
    from scipy.stats import spearmanr
    pred_runner=float(spearmanr(clean_nns.reshape(-1),runner_map.reshape(-1)).statistic);pred_true=float(spearmanr(clean_nns.reshape(-1),true_map.reshape(-1)).statistic) if true!=target else 1.0
    positive=_delete_support(image,clean_nns,positive=True);negative=_delete_support(image,clean_nns,positive=False)
    with torch.no_grad():clean_logit=float(model(image)[0,target]);positive_logit=float(model(positive)[0,target]);negative_logit=float(model(negative)[0,target])
    sanity=[{"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":sample["class_id"],"check_id":"predicted_runner_up_target_swap_spearman","target_class":runner,"value":pred_runner,"reference":1.0,"passed":math.isfinite(pred_runner)},{"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":sample["class_id"],"check_id":"predicted_true_target_swap_spearman","target_class":true,"value":pred_true,"reference":1.0,"passed":math.isfinite(pred_true)},{"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":sample["class_id"],"check_id":"positive_support_deletion_delta","target_class":target,"value":clean_logit-positive_logit,"reference":0.0,"passed":math.isfinite(clean_logit-positive_logit)},{"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":sample["class_id"],"check_id":"negative_contribution_deletion_delta","target_class":target,"value":negative_logit-clean_logit,"reference":0.0,"passed":math.isfinite(negative_logit-clean_logit)}]
    randomization_stages=[("head",model.head),("final_norm",model.norm),*((f"block_{index}",model.blocks[index]) for index in range(len(model.blocks)-1,-1,-1)),("patch_embed",model.patch_embed)]
    for index,(name,module) in enumerate(randomization_stages):
        seed=int.from_bytes(bytes.fromhex(str(unit["unit_digest"]))[:8],"big")+index;_randomize_module(module,seed,device);randomized=nns.input_ig(model,image,baseline,target,selected_steps,4).sum(0).numpy();correlation=float(spearmanr(clean_nns.reshape(-1),randomized.reshape(-1)).statistic)
        sanity.append({"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":sample["class_id"],"check_id":"progressive_parameter_randomization_"+name,"target_class":target,"value":correlation,"reference":1.0,"passed":math.isfinite(correlation)});forward_calls+=math.ceil(selected_steps/4);backward_calls+=math.ceil(selected_steps/4)
    forward_calls+=2*math.ceil(selected_steps/4)+3;backward_calls+=2*math.ceil(selected_steps/4);resources=_resource(started,cpu_started,device,forward_calls,backward_calls)
    _publish_phase(repo,unit,"formal",[("faithfulness.csv",FAITH_FIELDS,faith),("faithfulness_road_rows.csv",FAITH_ROAD_FIELDS,road_rows),("faithfulness_curves.csv",FAITH_CURVE_FIELDS,curve_rows),("stability.csv",STABILITY_FIELDS,stability),("sanity.csv",SANITY_FIELDS,sanity)],{},resources,{"faithfulness_method_count":len(faith),"road_row_count":len(road_rows),"curve_row_count":len(curve_rows),"stability_row_count":len(stability),"sanity_row_count":len(sanity)},{"status":"VALID","selected_conv_gamma":conv,"selected_linear_gamma":linear,"fixed_clean_target":target,"selected_steps":selected_steps,"all_outputs_finite":True})
    _complete_journal(repo,unit,"formal",checkpoint_sha);return {"status":"VALID","unit":unit["unit_digest"],"wall_seconds":resources["wall_seconds"]}


def run_phase(repo: Path, phase: str, shard: int, device: str, limit: int | None) -> dict[str,Any]:
    for required in (control.RUN_SPEC,control.ROOT/"implementation_seal.json",control.ROOT/"model_materialization.json"):
        control.validate_artifact(control.strict_read(repo/required),repo,verify_files=True)
    units=[row for row in control.load_units(repo) if int(row["shard"])==shard];samples={row["source_object_id"]:row for row in control.load_samples(repo)}
    if phase=="clean":
        new_ids={row["unit_digest"] for row in control.r1.read_csv(repo/control.NEW_CSV,control.NEW_FIELDS)};units=[row for row in units if row["unit_digest"] in new_ids]
    elif phase=="formal":
        units=[row for row in units if control.strict_read(_phase_dir(repo,row,"clean")/"terminal.json")["status"]=="PASS"]
    if limit is not None:units=units[:limit]
    results=[];model=None;loaded=None
    for unit in units:
        stage_failure=repo/control.ROOT/"stage_failure.json"
        if stage_failure.exists():
            stopped=control.strict_read(stage_failure);control.validate_artifact(stopped,repo,verify_files=True);return {"status":stopped["status"],"phase":phase,"shard":shard,"global_stop_before_unit":unit["unit_digest"],"results":results}
        if loaded!=unit["model_id"]:
            if model is not None:del model
            spec=_model_spec(repo,str(unit["model_id"]));model=_load_model(repo,spec,device);loaded=unit["model_id"]
        function={"clean":run_clean_unit,"gamma":run_gamma_unit,"formal":run_formal_unit}[phase]
        unit_started=time.monotonic();unit_cpu_started=time.process_time()
        try:result=function(repo,model,spec,samples[str(unit["sample_id"])],unit,device)
        except Exception as error:
            from .stage_x_nns_v2 import NNSV2ConvergenceError, NNSV2Error, NNSV2MemoryPlanError
            from .stage_x_comparators import ComparatorError
            if phase=="clean" and isinstance(error,NNSV2ConvergenceError):
                attempts=[{"unit_digest":unit["unit_digest"],"steps":row["steps"],"input_abs_error":row["input_abs_error"],"identity_max_abs_error":row["identity_max_abs_error"],"all_cuts_pass":row["all_cuts_pass"],"passed":row["pass"]} for row in error.attempts]
                chunks=sum(math.ceil(int(row["steps"])/4) for row in attempts);resources=_resource(unit_started,unit_cpu_started,device,139+2*chunks,3+19*chunks);_publish_phase(repo,unit,"clean",[("convergence_attempts.csv",CLEAN_FAILURE_FIELDS,attempts)],{},resources,{"attempt_count":len(attempts),"complete_cut_count":135},{"status":"FAILED_ROUTE_LOCAL_NUMERICAL","failure_code":"NO_GO_CLASS_LOGIT_COMPLETENESS","schedule_exhausted":True,"coverage_tolerated_class":True},status="FAILED")
                failed=control.signed("stage_x_x2_r2_unit_terminal","FAILED",control.payload(identity={"attempt_id":ATTEMPT_ID,"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"sample_id":unit["sample_id"],"sample_sha256":unit["sample_sha256"]},bindings=[control.artifact_record(repo,_phase_dir(repo,unit,"clean").relative_to(repo)/"terminal.json")],counts={"convergence_attempts":len(attempts)},state={"failure_code":"NO_GO_CLASS_LOGIT_COMPLETENESS","coverage_class":"FAILED_ROUTE_LOCAL_NUMERICAL","actual_execution_attempted":True},terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))
                terminal_path=repo/control.ROOT/"units"/unit["unit_digest"]/"terminal.json";control.publish_atomic(terminal_path.with_name("terminal.pending.json"),terminal_path,failed,repo);_complete_journal(repo,unit,"clean",control.sha_file(repo/control.MODEL_LOCAL_PATHS[str(spec["id"])]));results.append({"status":"FAILED_ROUTE_LOCAL_NUMERICAL","unit":unit["unit_digest"]});continue
            scientific=isinstance(error,(NNSV2ConvergenceError,NNSV2Error,ComparatorError)) and not isinstance(error,NNSV2MemoryPlanError);status="NO_GO_CLASS_LOGIT_COMPLETENESS" if isinstance(error,NNSV2ConvergenceError) else ("NO_GO_METHOD_VALIDITY" if scientific else ("BLOCKED_DETERMINISTIC_MEMORY_PLAN" if isinstance(error,NNSV2MemoryPlanError) else "BLOCKED_INFRASTRUCTURE"));unit_status="FAILED" if scientific else "NA";checkpoint_sha=control.sha_file(repo/control.MODEL_LOCAL_PATHS[str(spec["id"])]);terminal=control.signed("stage_x_x2_r2_unit_terminal",unit_status,control.payload(identity={"attempt_id":ATTEMPT_ID,"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"sample_id":unit["sample_id"],"sample_sha256":unit["sample_sha256"]},counts={"completed_phase_count":len(results)},state={"failure_code":status,"failure_type":type(error).__name__,"failure_message":str(error),"actual_execution_attempted":True},terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))
            terminal_path=repo/control.ROOT/"units"/str(unit["unit_digest"])/"terminal.json"
            if not terminal_path.exists():control.publish_atomic(terminal_path.with_name("terminal.pending.json"),terminal_path,terminal,repo)
            stage_path=repo/control.ROOT/"stage_failure.json";stage=control.signed("stage_x_x2_r2_status",status,control.payload(bindings=[control.artifact_record(repo,terminal_path.relative_to(repo))],counts={"executed_failed_units":1},state={"failure_code":status,"global_stop":True,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))
            if not stage_path.exists():control.publish_atomic(stage_path.with_name("stage_failure.pending.json"),stage_path,stage,repo)
            return {"status":status,"phase":phase,"shard":shard,"failed_unit":unit["unit_digest"],"error":f"{type(error).__name__}: {error}","results":results}
        results.append(result)
        if phase=="formal":
            del model; model=None; loaded=None
    return {"status":"PASS","phase":phase,"shard":shard,"results":results}


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("clean","gamma","formal"));parser.add_argument("--repo-root",type=Path,default=REPO_ROOT);parser.add_argument("--shard",type=int,required=True,choices=range(4));parser.add_argument("--device",required=True);parser.add_argument("--limit",type=int);args=parser.parse_args(argv)
    result=run_phase(args.repo_root.resolve(),args.phase,args.shard,args.device,args.limit);print(json.dumps(result,sort_keys=True,indent=2));return 0 if result["status"]=="PASS" else 2


if __name__=="__main__":raise SystemExit(main())
