"""Resumable GPU worker for Stage X X2 development units."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_x_x2 as control

ATTEMPT_ID="attempt-000001"
IMPLEMENTATION_FILES=(
    "src/vitnns/stage_x_x2.py","src/vitnns/stage_x_x2_worker.py",
    "src/vitnns/stage_x_x2_metrics.py","src/vitnns/stage_x_nns_v2.py",
    "src/vitnns/stage_x_comparators.py","config/stage_x_x2.json",
    "config/stage_x_x2.schema.json","config/stage_x_x2_artifact.schema.json",
    "docs/stage_x_x2_unfused_execution_amendment_20260909.md",
    "tests/test_stage_x_x2.py","tests/test_stage_x_x2_metrics.py",
)
MODEL_LOCAL_PATHS={
    "vit_s16":Path("assets/models/stage_x_x2/vit_small_patch16_224.augreg_in21k_ft_in1k.npz"),
    "vit_b16":Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"),
    "deit_s16":Path("assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth"),
    "deit_b16":Path("assets/models/stage_x_x2/deit_base_patch16_224.fb_in1k.pth"),
    "vit_b32":Path("assets/models/stage_x_x2/vit_base_patch32_224.augreg_in21k_ft_in1k.npz"),
}


class X2WorkerError(ValueError): pass


def _write_bytes_new(path:Path,value:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True); fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream: stream.write(value);stream.flush();os.fsync(stream.fileno())


def _write_json_new(path:Path,value:Mapping[str,Any])->None:
    result=dict(value);result["canonical_digest"]=canonical_sha256(result)
    _write_bytes_new(path,canonical_json_bytes(result)+b"\n")


def _validate_plain_signed(value:Mapping[str,Any],context:str)->None:
    plain=dict(value); observed=plain.pop("canonical_digest",None)
    if observed!=canonical_sha256(plain): raise X2WorkerError(f"{context}: canonical digest mismatch")


def _transition(path:Path,value:Mapping[str,Any])->None:
    control.validate_signed(value,path.as_posix());previous=control.strict_read(path);control.validate_signed(previous,path.as_posix())
    if previous["status"]!="PENDING" or value["status"]=="PENDING":raise X2WorkerError(f"invalid pending transition: {path}")
    temporary=path.with_name(path.name+".terminal.pending")
    if temporary.exists():raise X2WorkerError(f"stale pending transition: {temporary}")
    control.write_new(temporary,value);os.replace(temporary,path)


def seal_implementation(repo:Path)->dict[str,Any]:
    path=repo/control.ROOT_PATH/"implementation_seal.json"
    if path.exists():
        value=control.strict_read(path);control.validate_signed(value,path.as_posix());return {"status":"PASS","existing":True,"digest":value["canonical_digest"]}
    for relative in IMPLEMENTATION_FILES:
        if not (repo/relative).is_file(): raise X2WorkerError(f"implementation source missing: {relative}")
    payload={"files":[control.record(repo,Path(p)) for p in IMPLEMENTATION_FILES],"sealed_before_x2_development_outcome_access":True,"development_unit_or_outcome_file_exists":any((repo/control.ROOT_PATH/"units").rglob("*")) if (repo/control.ROOT_PATH/"units").exists() else False,"source_changes_after_seal_allowed":False}
    if payload["development_unit_or_outcome_file_exists"]: raise X2WorkerError("cannot seal implementation after outcome access")
    dependency=repo/control.ROOT_PATH/"dependency_binding.json"
    dependency_payload=control.strict_read(dependency)["payload"]
    dependency_payload["implementation_bindings"]=[control.record(repo,Path(p)) for p in IMPLEMENTATION_FILES]
    _transition(dependency,control.signed("stage_x_x2_dependency_binding","PASS",dependency_payload))
    control.write_new(path,control.signed("stage_x_x2_implementation_seal","PASS",payload))
    return {"status":"PASS","existing":False,"digest":control.strict_read(path)["canonical_digest"]}


def validate_seal(repo:Path)->dict[str,Any]:
    path=repo/control.ROOT_PATH/"implementation_seal.json";value=control.strict_read(path);control.validate_signed(value,path.as_posix())
    for row in value["payload"]["files"]:
        if control.record(repo,Path(row["path"]))!=row: raise X2WorkerError(f"sealed implementation drifted: {row['path']}")
    return value


def _load_model(repo:Path,spec:Mapping[str,Any],device:str):
    import torch
    import timm
    from timm.layers import set_fused_attn
    path=repo/MODEL_LOCAL_PATHS[spec["id"]];set_fused_attn(False);model=timm.create_model(spec["model_id"],pretrained=False).eval();kind=spec["source_kind"]
    if kind=="official_timm_0.9.16_npz":model.load_pretrained(str(path));state=model.state_dict()
    else:
        with path.open("rb") as stream:loaded=torch.load(stream,map_location="cpu",weights_only=True)
        state=loaded["model"] if kind=="official_timm_0.9.16_pth" else loaded
        result=model.load_state_dict(state,strict=True)
        if tuple(result.missing_keys) or tuple(result.unexpected_keys):raise X2WorkerError("strict model load drifted")
    model=model.to(device).eval()
    for block in model.blocks:block.attn.fused_attn=False
    return model


def _load_image(repo:Path,sample:Mapping[str,Any],spec:Mapping[str,Any],device:str):
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    path=repo/"assets/data"/sample["source_object_id"]
    if path.stat().st_size!=sample["file_bytes"] or control.sha_file(path)!=sample["file_sha256"]: raise X2WorkerError("sample bytes drifted")
    p=spec["preprocess"]
    with Image.open(path) as source:
        rgb=source.convert(p["color_mode"]); resized=TF.resize(rgb,p["resize_shorter_side"],interpolation=InterpolationMode.BICUBIC,antialias=p["antialias"]);cropped=TF.center_crop(resized,p["center_crop"]);canonical=TF.pil_to_tensor(cropped).float().div(255).unsqueeze(0)
    mean=torch.tensor(p["mean"]).reshape(1,3,1,1);std=torch.tensor(p["std"]).reshape(1,3,1,1);clean=((canonical-mean)/std).to(device)
    from .stage_c_perturbations import tensor_sha256
    return clean,{"jpeg_path":path.relative_to(repo).as_posix(),"jpeg_sha256":sample["file_sha256"],"canonical_rgb_sha256":tensor_sha256(canonical),"clean_tensor_sha256":tensor_sha256(clean),"shape":list(clean.shape),"dtype":str(clean.dtype)}


def _array_meta(path:Path)->dict[str,Any]:
    from .stage_f_f3c1_kernels import array_digest
    value=np.load(path,allow_pickle=False)
    return {"path":path.name,"bytes":path.stat().st_size,"sha256":control.sha_file(path),"dtype":str(value.dtype),"shape":list(value.shape),"array_digest":array_digest(value)}


def _publish_failure(root:Path,unit:Mapping[str,Any],code:str,details:Mapping[str,Any],resources:Mapping[str,Any])->None:
    if root.exists(): raise X2WorkerError("unit terminal exists")
    root.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=".failed-",dir=root.parent))
    try:
        result={"schema_version":1,"document_type":"stage_x_x2_development_unit_terminal","status":"FAILED","unit_identity":dict(unit),"unit_digest":unit["unit_digest"],"attempt_id":ATTEMPT_ID,"failure_code":code,"failure_details":dict(details),"resources":dict(resources),"arrays":[],"terminal":True,"immutable":True};_write_json_new(tmp/"result.json",result);_write_json_new(tmp/"status.json",{"schema_version":1,"document_type":"stage_x_x2_unit_status","status":"FAILED","unit_digest":unit["unit_digest"],"attempt_id":ATTEMPT_ID,"terminal":True,"immutable":True});os.replace(tmp,root)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def _publish_clean(root:Path,result:Mapping[str,Any],arrays:Mapping[str,np.ndarray])->None:
    if root.exists(): raise X2WorkerError("clean attempt exists")
    root.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=".clean-",dir=root.parent))
    try:
        _write_json_new(tmp/"result.json",result); records=[]
        from .stage_f_f3c1_kernels import array_digest
        for name,value in sorted(arrays.items()):
            path=tmp/name;array=np.ascontiguousarray(value,dtype=np.float32);path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
            with os.fdopen(fd,"wb") as stream:np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
            records.append({"path":name,"bytes":path.stat().st_size,"sha256":control.sha_file(path),"dtype":"float32","shape":list(array.shape),"array_digest":array_digest(array)})
        _write_json_new(tmp/"manifest.json",{"schema_version":1,"document_type":"stage_x_x2_clean_attempt_manifest","status":"VALID","unit_digest":result["unit_digest"],"attempt_id":ATTEMPT_ID,"result":{"path":"result.json","bytes":(tmp/"result.json").stat().st_size,"sha256":control.sha_file(tmp/"result.json")},"arrays":records,"immutable":True});os.replace(tmp,root)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def _summary_map(result:Mapping[str,Any])->np.ndarray:
    from .stage_x_x2_metrics import pixel_map
    return pixel_map(result["attribution"]).numpy().astype(np.float32,copy=False)


def run_clean_unit(repo:Path,model:Any,spec:Mapping[str,Any],sample:Mapping[str,Any],unit:Mapping[str,Any],device:str)->dict[str,Any]:
    import torch
    from . import stage_x_comparators as comparators
    from . import stage_x_nns_v2 as nns
    from . import stage_x_x2_metrics as metrics
    clean_root=repo/control.ROOT_PATH/"units"/unit["unit_digest"]/"attempts"/ATTEMPT_ID/"clean"
    terminal_root=clean_root.parent/"terminal"
    if clean_root.exists() or terminal_root.exists(): return {"unit":unit["unit_digest"],"status":"EXISTING"}
    started=time.monotonic();cpu_started=time.process_time();torch.cuda.reset_peak_memory_stats(device)
    try:
        image,image_identity=_load_image(repo,sample,spec,device);baseline=torch.zeros_like(image)
        result=nns.explain(model,image,true_class=int(sample["class_id"]),baseline=baseline,steps=tuple(control.load_config(repo)["execution"]["steps"]),alpha_chunk=4,component_chunk=8)
        identity=comparators.full_model_ig_control(model,image,baseline,result.target.predicted_class,result.steps);identity_check=comparators.validate_identity(result,identity)
        arrays={"nns_v2.npy":result.signed_rgb_map.numpy(),"full_model_ig.npy":identity["signed_rgb_map"].numpy()}
        adapters={};method_functions={"vit_grad_cam":lambda:comparators.vit_grad_cam(model,image,result.target.predicted_class),"transformer_attribution_chefer":lambda:comparators.chefer(model,image,result.target.predicted_class),"rise":lambda:comparators.rise(model,image,result.target.predicted_class,namespace="STAGE_X_X2_RISE_V1|"+unit["unit_digest"]+"|"),"occlusion":lambda:comparators.occlusion(model,image,result.target.predicted_class)}
        for name,function in method_functions.items():
            began=time.monotonic();value=function();arrays[name+".npy"]=_summary_map(value);adapters[name]={key:item for key,item in value.items() if key!="attribution"};adapters[name]["elapsed_seconds"]=time.monotonic()-began
        gamma=[]
        for conv in comparators.ATTNLRP_CONV_GAMMAS:
            for linear in comparators.ATTNLRP_LINEAR_GAMMAS:
                began=time.monotonic();value=comparators.attnlrp(model,image,result.target.predicted_class,conv_gamma=conv,linear_gamma=linear);name=f"attnlrp_c{conv:g}_l{linear:g}.npy";arrays[name]=_summary_map(value);road=metrics.road_combined(model,image,value["attribution"],result.target.predicted_class,unit["unit_digest"]+f"|c{conv:g}|l{linear:g}")
                gamma.append({"conv_gamma":conv,"linear_gamma":linear,"forward_parity_max_abs":value["forward_parity_max_abs"],"target_index":value["target_index"],"shape":list(value["attribution"].shape),"finite":bool(torch.isfinite(value["attribution"]).all()),"source_binding":value["official_sources"],"road":road,"elapsed_seconds":time.monotonic()-began})
        secondary={}
        for name,value in (("attention_rollout",comparators.attention_rollout(model,image)),("raw_cls_attention",comparators.raw_cls_attention(model,image)),("random_map",comparators.deterministic_random_map(224,224,unit["unit_digest"])),("center_prior",comparators.center_prior(224,224))): arrays[name+".npy"]=_summary_map(value);secondary[name]={key:item for key,item in value.items() if key!="attribution"}
        edge=comparators.edge_input_gradient_prior(model,image);arrays["edge_prior.npy"]=_summary_map({"attribution":edge["edge_attribution"]});arrays["input_gradient_prior.npy"]=_summary_map({"attribution":edge["input_gradient_attribution"]});secondary["edge_input_gradient_prior"]={key:item for key,item in edge.items() if key not in {"edge_attribution","input_gradient_attribution"}}
        elapsed=time.monotonic()-started;resources={"wall_seconds":elapsed,"gpu_seconds":elapsed,"cpu_seconds":time.process_time()-cpu_started,"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),"forward_calls_recorded":None,"backward_calls_recorded":None,"downloaded_bytes":0,"retained_bytes":0}
        payload={"schema_version":1,"document_type":"stage_x_x2_clean_attempt","status":"VALID","unit_identity":dict(unit),"unit_digest":unit["unit_digest"],"attempt_id":ATTEMPT_ID,"model_id":spec["model_id"],"checkpoint_sha256":control.sha_file(repo/MODEL_LOCAL_PATHS[spec["id"]]),"image_identity":image_identity,"target":result.target.to_dict(),"baseline":"normalized_zero","path":"straight","quadrature":{"schedule":control.load_config(repo)["execution"]["steps"],"selected_steps":result.steps,"alpha_chunk":4,"attempts":[dict(x) for x in result.attempts]},"input_completeness":{"delta":result.delta,"attribution_sum":result.input_sum,"absolute_error":result.input_abs_error,"status":"PASS"},"identity_control":identity_check,"complete_cuts":{"count":len(result.cuts),"all_pass":all(x.passed for x in result.cuts.values()),"per_cut":{key:{"component_count":value.component_count,"pixel_max_abs_error":value.pixel_max_abs_error,"total_abs_error":value.total_abs_error,"passed":value.passed} for key,value in result.cuts.items()}},"adapters":adapters,"gamma_candidates":gamma,"secondary_references":secondary,"pixel_provenance":{"nns_native_shape":list(result.attribution.shape),"signed_rgb_shape":list(result.signed_rgb_map.shape),"resize_or_interpolation":False},"resources":resources,"clean_phase_only":True,"faithfulness_except_gamma":False,"stability":False,"terminal":False}
        _publish_clean(clean_root,payload,arrays);return {"unit":unit["unit_digest"],"status":"VALID","steps":result.steps,"elapsed":elapsed}
    except nns.NNSV2ConvergenceError as error:
        resources={"wall_seconds":time.monotonic()-started,"gpu_seconds":time.monotonic()-started,"cpu_seconds":time.process_time()-cpu_started,"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device))};_publish_failure(terminal_root,unit,error.status,{"message":str(error),"attempts":[dict(x) for x in error.attempts]},resources);return {"unit":unit["unit_digest"],"status":"FAILED","code":error.status}
    except nns.NNSV2MemoryPlanError as error:
        resources={"wall_seconds":time.monotonic()-started,"gpu_seconds":time.monotonic()-started,"cpu_seconds":time.process_time()-cpu_started,"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device))};_publish_failure(terminal_root,unit,error.status,{"message":str(error)},resources);return {"unit":unit["unit_digest"],"status":"FAILED","code":error.status}


def run_clean(repo:Path,shard:int,device:str,limit:int|None)->dict[str,Any]:
    validate_seal(repo);config=control.load_config(repo);materialized=control.strict_read(repo/control.ROOT_PATH/"model_materialization.json")
    if materialized["status"]!="PASS":raise X2WorkerError("models are not materialized")
    manifest=control.strict_read(repo/control.ROOT_PATH/"development_unit_manifest.json")["payload"];sample_doc=control.strict_read(repo/control.ROOT_PATH/"development_sample_manifest.json")["payload"]
    if shard not in range(config["execution"]["shards"]):raise X2WorkerError("shard outside frozen range")
    samples={r["source_object_id"]:r for r in sample_doc["samples"]};units=[u for u in manifest["units"] if u["shard"]==shard]
    if limit is not None:units=units[:limit]
    specs={x["model_id"]:x for x in config["models"]};model=None;loaded=None;results=[]
    for unit in units:
        if loaded!=unit["model_id"]:
            if model is not None:del model
            spec=specs[unit["model_id"]];model=_load_model(repo,spec,device);loaded=unit["model_id"]
        results.append(run_clean_unit(repo,model,spec,samples[unit["sample_id"]],unit,device))
        if results[-1].get("status")=="FAILED":break
    return {"status":"PASS","shard":shard,"results":results}


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    seal=sub.add_parser("seal");seal.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    run=sub.add_parser("run-clean");run.add_argument("--repo-root",type=Path,default=REPO_ROOT);run.add_argument("--shard",required=True,type=int);run.add_argument("--device",default="cuda:0");run.add_argument("--limit",type=int)
    args=parser.parse_args(argv)
    try:result=seal_implementation(args.repo_root.resolve()) if args.command=="seal" else run_clean(args.repo_root.resolve(),args.shard,args.device,args.limit)
    except (X2WorkerError,control.StageXX2Error,OSError,RuntimeError,ValueError,KeyError) as error:print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True));return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
