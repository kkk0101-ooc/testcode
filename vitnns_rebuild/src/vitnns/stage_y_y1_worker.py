"""Single-device empirical worker for the frozen Stage Y Y1 pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import REPO_ROOT
from . import stage_y_y1 as y1

ROOT_PATH = Path("evidence/stage_y/y1_validation_pilot_20260910/formal")
REPORT_PATH = Path("reports/stage_y_y1_20260910")
RUN_SPEC = Path("config/stage_y/y1_run_spec.json")
MODEL_PATH = Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth")
SAMPLE_PATH = Path("assets/data/ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG")
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
MODEL_LOADER_IDENTITY = "F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES"
SAMPLE_SHA256 = "c597ef7e061560542edbd77d2cbca6107cc44ff5e86de487a015eb47ae4620a8"
CLEAN_SHA256 = "f901fab0e43fbbc725aff3157ef7a91604cb30b4fdd92d2cb2f1aea49f092484"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(y1.StageYY1Error(token)))


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(y1.canonical_bytes(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())


def write_array_new(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value, dtype="<f4")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    started=time.monotonic();cpu_started=time.process_time()
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": y1.sha_file(path), "canonical_digest": y1.array_digest(array), "dtype": str(array.dtype), "shape": list(array.shape), "axes": _axes(path.name, array.shape),"serialization_fsync_wall_seconds":time.monotonic()-started,"serialization_fsync_cpu_seconds":time.process_time()-cpu_started}


def _axes(name: str, shape: Sequence[int]) -> list[str]:
    if tuple(shape) == (3,224,224): return ["rgb","pixel_y","pixel_x"]
    if tuple(shape) == (56,56): return ["cell_y","cell_x"]
    if tuple(shape) == (12,56,56): return ["head","cell_y","cell_x"]
    if tuple(shape) == (196,196): return ["source_patch","destination_patch"]
    if tuple(shape) == (12,196,196): return ["head","source_patch","destination_patch"]
    if tuple(shape) == (196,): return ["source_patch"]
    if tuple(shape) == (12,196): return ["head","source_patch"]
    return [f"axis_{index}" for index in range(len(shape))]


def load_model_and_image(repo: Path, device: str):
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf
    from .modeling import canonical_preprocess_spec, load_canonical_model
    from .stage_c_perturbations import tensor_sha256

    if y1.sha_file(repo / MODEL_PATH) != MODEL_SHA256 or y1.sha_file(repo / SAMPLE_PATH) != SAMPLE_SHA256:
        raise y1.StageYY1Error("sealed model/sample identity drifted")
    checkpoint_started = time.monotonic(); checkpoint_cpu = time.process_time()
    model = load_canonical_model(repo / MODEL_PATH, device=device, unfused_attention=True, expected_state_dict_sha256=MODEL_SHA256, expected_model_name=MODEL_NAME, expected_model_identity=MODEL_LOADER_IDENTITY, expected_canonical_tensor_digest=MODEL_TENSOR_DIGEST)
    checkpoint_cost = {"wall_seconds": time.monotonic()-checkpoint_started, "cpu_seconds": time.process_time()-checkpoint_cpu}
    specification = canonical_preprocess_spec(model)
    observed = (specification.input_size,specification.resize_shorter_side,specification.center_crop,specification.interpolation,specification.mean,specification.std)
    if observed != ((3,224,224),248,(224,224),"bicubic",(0.5,0.5,0.5),(0.5,0.5,0.5)):
        raise y1.StageYY1Error("preprocessing contract drifted")
    preprocess_started=time.monotonic(); preprocess_cpu=time.process_time()
    with Image.open(repo/SAMPLE_PATH) as source:
        rgb=source.convert("RGB"); resized=tf.resize(rgb,248,interpolation=InterpolationMode.BICUBIC,antialias=True); cropped=tf.center_crop(resized,[224,224]); clean=((tf.pil_to_tensor(cropped).float().div(255).unsqueeze(0)-0.5)/0.5).to(device)
    preprocessing_cost={"wall_seconds":time.monotonic()-preprocess_started,"cpu_seconds":time.process_time()-preprocess_cpu}
    if tensor_sha256(clean)!=CLEAN_SHA256: raise y1.StageYY1Error("preprocessed sample identity drifted")
    return model,clean,specification,checkpoint_cost,preprocessing_cost


def _head_maps(arrays: Mapping[str,np.ndarray]) -> dict[int,np.ndarray]:
    return {block: arrays[f"b{block:02d}_first_heads_cell"] for block in range(12)}


def _attention_maps(arrays: Mapping[str,np.ndarray]) -> dict[int,np.ndarray]:
    return {block: arrays[f"b{block:02d}_first_attention_cell"] for block in range(12)}


def _sum_calls(*rows: Mapping[str,int]) -> dict[str,int]:
    keys=set().union(*(row.keys() for row in rows)); return {key:sum(int(row.get(key,0)) for row in rows) for key in sorted(keys)}


def _condition_repeat(repo: Path, model: Any, image: Any, condition: str, repeat: int, target: int, clean_logit: float, baseline_logit: float, spec: Mapping[str,Any]) -> tuple[dict[str,Any],dict[str,np.ndarray]]:
    import torch
    from .stage_x_nns_v2 import input_ig
    source_id="ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"
    preprocessing_started=time.monotonic();preprocessing_cpu=time.process_time();transformed,metadata=y1.transform_image(image,condition,source_id);preprocessing_phase={"wall_seconds":time.monotonic()-preprocessing_started,"cpu_seconds":time.process_time()-preprocessing_cpu}
    started=time.monotonic();cpu_started=time.process_time();torch.cuda.reset_peak_memory_stats(transformed.device);gpu_started=torch.cuda.Event(enable_timing=True);gpu_ended=torch.cuda.Event(enable_timing=True);gpu_started.record()
    forward_started=time.monotonic();forward_cpu=time.process_time()
    with torch.no_grad(): logits=model(transformed);transformed_prediction=int(logits.argmax(1));transformed_target_logit=float(logits[0,target])
    target_forward_phase={"wall_seconds":time.monotonic()-forward_started,"cpu_seconds":time.process_time()-forward_cpu,"forward_calls":1}
    baseline_started=time.monotonic();baseline_cpu=time.process_time()
    with torch.no_grad(): baseline=float(model(torch.zeros_like(transformed))[0,target])
    baseline_forward_phase={"wall_seconds":time.monotonic()-baseline_started,"cpu_seconds":time.process_time()-baseline_cpu,"forward_calls":1}
    calls={"forward_calls":2,"backward_calls":0,"jvp_calls":0,"jvp_direction_lanes":0,"vjp_calls":0,"quadrature_alpha_batches":0}
    attempts=[]; selected=None; selected_identity=None; integration=None; arrays=None; closure=None; identity_error=None
    for steps in spec["quadrature"]["attempts"]:
        attempt_started=time.monotonic(); candidate=input_ig(model,transformed,torch.zeros_like(transformed),target,steps,4); calls["forward_calls"]+=math.ceil(steps/4);calls["backward_calls"]+=math.ceil(steps/4)
        delta=transformed_target_logit-baseline; error=abs(float(candidate.double().sum())-delta); threshold=1e-4+0.005*abs(delta); input_pass=error<=threshold
        candidate_integration=None;candidate_arrays=None;candidate_closure=None;candidate_identity_error=None
        extraction_wall=0.0;extraction_cpu=0.0
        if input_pass:
            candidate_integration=y1.integrate_routes(model,transformed,torch.zeros_like(transformed),target,steps,alpha_chunk=4,source_patch_chunk=int(spec["planner"]["source_patch_jvp_chunk"]),component_vjp_chunk=int(spec["planner"]["component_vjp_chunk"]));calls=_sum_calls(calls,candidate_integration.calls)
            candidate_identity_error=float((candidate_integration.attribution-candidate).abs().max());extraction_started=time.monotonic();extraction_cpu_started=time.process_time();candidate_arrays=y1.materialize_arrays(candidate_integration);candidate_closure=y1.validate_closure(candidate_arrays,delta);extraction_wall=time.monotonic()-extraction_started;extraction_cpu=time.process_time()-extraction_cpu_started
        passed=input_pass and candidate_closure is not None and candidate_closure["status"]=="PASS" and candidate_identity_error is not None and candidate_identity_error<=1e-5
        attempts.append({"steps":steps,"input_sum":float(candidate.double().sum()),"input_delta":delta,"input_abs_error":error,"threshold":threshold,"input_pass":input_pass,"all_contracts_pass":passed,"closure_status":None if candidate_closure is None else candidate_closure["status"],"identity_max_abs_error":candidate_identity_error,"wall_seconds":time.monotonic()-attempt_started,"array_digest":y1.array_digest(candidate)})
        if passed:
            selected=steps;selected_identity=candidate;integration=candidate_integration;arrays=candidate_arrays;closure=candidate_closure;identity_error=candidate_identity_error;break
    if selected is None: raise y1.StageYY1NumericalError("input IG nonconverged at 8192")
    assert integration is not None and arrays is not None and closure is not None and identity_error is not None
    extraction_phase={"wall_seconds":extraction_wall,"cpu_seconds":extraction_cpu,"arrays":len(arrays),"patch_matrices":sum(name.endswith("_matrix") for name in arrays),"cls_vectors":sum(name.endswith("_cls") for name in arrays)}
    ablation_started=time.monotonic();ablation_cpu=time.process_time();ablations,ablation_calls=y1.run_head_ablations(model,transformed,target,clean_logit,baseline_logit,_head_maps(arrays),source_id);ablation_phase={"wall_seconds":time.monotonic()-ablation_started,"cpu_seconds":time.process_time()-ablation_cpu,"calls":ablation_calls};calls=_sum_calls(calls,ablation_calls)
    deletion_started=time.monotonic();deletion_cpu=time.process_time();deletion,deletion_calls=y1.run_deletion_operator(model,transformed,target,clean_logit,baseline_logit,_attention_maps(arrays),source_id,int(spec["planner"]["deletion_forward_batch"]));deletion_phase={"wall_seconds":time.monotonic()-deletion_started,"cpu_seconds":time.process_time()-deletion_cpu,"calls":deletion_calls};calls=_sum_calls(calls,deletion_calls)
    gpu_ended.record();torch.cuda.synchronize();gpu_active=gpu_started.elapsed_time(gpu_ended)/1000.0
    wall=time.monotonic()-started;cpu=time.process_time()-cpu_started;peak_alloc=int(torch.cuda.max_memory_allocated(transformed.device));peak_reserved=int(torch.cuda.max_memory_reserved(transformed.device))
    record={"condition":condition,"repeat":repeat,"status":"PASS","fixed_target_class":target,"transformed_predicted_class":transformed_prediction,"prediction_changed":transformed_prediction!=target,"target_logit":transformed_target_logit,"baseline_logit":baseline,"delta":transformed_target_logit-baseline,"transform_metadata":metadata,"quadrature_attempts":attempts,"selected_steps":selected,"identity_max_abs_error":identity_error,"closure":closure,"ablations":ablations,"deletion_operator":deletion,"calls":calls,"phase_telemetry":{"condition_preprocessing":preprocessing_phase,"target_forward":target_forward_phase,"baseline_forward":baseline_forward_phase,"selected_grouped_cut_integration":integration.phase_telemetry,"matrix_cls_extraction":extraction_phase,"single_and_multi_head_ablations":ablation_phase,"hierarchical_deletion":deletion_phase},"peak_allocated_gpu_bytes":peak_alloc,"peak_reserved_gpu_bytes":peak_reserved,"gpu_active_seconds":gpu_active,"wall_seconds":wall,"cpu_seconds":cpu}
    return record,arrays


def execute(repo_root: Path|str=REPO_ROOT,device: str="cuda:0") -> dict[str,Any]:
    import torch
    import torchvision
    import timm
    repo=Path(repo_root).resolve(); formal=repo/ROOT_PATH; reports=repo/REPORT_PATH
    resumable = formal.exists() and (formal/"completion/y1.pending.json").is_file() and not (formal/"worker_result.json").exists() and not (formal/"arrays").exists()
    if (formal.exists() or reports.exists()) and not resumable: raise y1.StageYY1Error("Y1 output root already exists")
    if not device.startswith("cuda") or not torch.cuda.is_available(): raise y1.StageYY1Error("Y1 empirical pilot requires exactly one CUDA device")
    if torch.cuda.device_count()!=1: raise y1.StageYY1Error("CUDA_VISIBLE_DEVICES must expose exactly one device")
    spec=read_json(repo/RUN_SPEC); started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if not resumable:
        formal.mkdir(parents=True);reports.mkdir(parents=True);write_json_new(formal/"completion/y1.pending.json",y1.signed("stage_y_y1_pending_completion","PENDING",{"started_at":started_at,"run_spec_sha256":y1.sha_file(repo/RUN_SPEC),"terminal_published":False}))
    else:
        reports.mkdir(parents=True,exist_ok=True)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_device(torch.device(device))
    model,clean,preprocessing,fixed_model,fixed_preprocessing=load_model_and_image(repo,device);model.eval();baseline=torch.zeros_like(clean)
    with torch.no_grad(): logits=model(clean);target=int(logits.argmax(1));clean_logit=float(logits[0,target]);baseline_logit=float(model(baseline)[0,target])
    if target!=713: raise y1.StageYY1Error("clean target drifted from sealed X1-R1 class 713")
    records=[];array_records=[];retained_per=[]
    for condition in spec["conditions"]:
        for repeat in spec["repeats"]:
            record,arrays=_condition_repeat(repo,model,clean,condition,repeat,target,clean_logit,baseline_logit,spec)
            condition_bytes=0
            for name,value in sorted(arrays.items()):
                relative=ROOT_PATH/"arrays"/condition/f"repeat-{repeat}"/f"{name}.npy";item=write_array_new(repo/relative,value);item["path"]=relative.as_posix();array_records.append(item);condition_bytes+=item["bytes"]
            record["output_bytes_by_schema_type"]={}
            for item in array_records[-len(arrays):]: record["output_bytes_by_schema_type"][str(tuple(item["shape"]))]=record["output_bytes_by_schema_type"].get(str(tuple(item["shape"])),0)+item["bytes"]
            record["phase_telemetry"]["metric_serialization_and_fsync"]={"wall_seconds":sum(item["serialization_fsync_wall_seconds"] for item in array_records[-len(arrays):]),"cpu_seconds":sum(item["serialization_fsync_cpu_seconds"] for item in array_records[-len(arrays):]),"files":len(arrays)}
            record["retained_bytes"]=condition_bytes;retained_per.append(condition_bytes);records.append(record)
    repeatability=[]
    for condition in spec["conditions"]:
        pair=[(record,arrays) for record,arrays in []]
        rows=[record for record in records if record["condition"]==condition]
        digests=[]
        for repeat in spec["repeats"]:
            prefix=(repo/ROOT_PATH/"arrays"/condition/f"repeat-{repeat}")
            digests.append({path.name:y1.sha_file(path) for path in sorted(prefix.glob("*.npy"))})
        same=digests[0]==digests[1] and rows[0]["selected_steps"]==rows[1]["selected_steps"]
        repeatability.append({"condition":condition,"status":"PASS" if same else "FAILED_NUMERICAL","byte_identical_arrays":same,"selected_steps_match":rows[0]["selected_steps"]==rows[1]["selected_steps"]})
    if any(row["status"]!="PASS" for row in repeatability): raise y1.StageYY1NumericalError("repeatability contract failed")
    total_wall=sum(row["wall_seconds"] for row in records)+fixed_model["wall_seconds"]+fixed_preprocessing["wall_seconds"];total_cpu=sum(row["cpu_seconds"] for row in records)+fixed_model["cpu_seconds"]+fixed_preprocessing["cpu_seconds"];total_gpu=sum(row["gpu_active_seconds"] for row in records);peak=max(row["peak_reserved_gpu_bytes"] for row in records);retained=sum(retained_per)
    caps={"h20_gpu_hours":total_gpu/3600,"wall_hours":total_wall/3600,"cpu_hours":total_cpu/3600,"devices":1,"peak_reserved_gpu_gib_per_process":peak/1024**3,"retained_storage_gib":retained/1024**3,"download_bytes":0}
    hard=spec["hard_caps"];cap_pass=all((caps[key]<=hard[key] for key in hard))
    if not cap_pass: raise y1.StageYY1BudgetError("Y1 hard cap exceeded")
    fixed={"gpu_active_seconds":0.0,"wall_seconds":fixed_model["wall_seconds"]+fixed_preprocessing["wall_seconds"],"cpu_seconds":fixed_model["cpu_seconds"]+fixed_preprocessing["cpu_seconds"]}
    budget=y1.project_budget(fixed,records,max(retained_per),sum((repo/RUN_SPEC).stat().st_size for _ in [0]));budget["observed_peak_reserved_gib"]=peak/1024**3;budget["proposed_peak_reserved_gib"]=y1.ceil_to(1.5*peak/1024**3,1.0);budget["formal_outcome_access_occurred"]=False;budget["allowed_device_schedule"]="single H20 sequential"
    if budget["proposed_peak_reserved_gib"]>80 or budget["status"]!="PASS": raise y1.StageYY1BudgetError("budget projection exceeds owner ceiling")
    summary={"status":"PASS","records":records,"repeatability":repeatability,"array_records":array_records,"fixed_cost":fixed,"caps_observed":caps,"caps_status":"PASS","budget":budget,"device":{"index":0,"name":torch.cuda.get_device_name(0),"uuid":torch.cuda.get_device_properties(0).uuid if hasattr(torch.cuda.get_device_properties(0),"uuid") else "reported_by_nvidia_smi_in_environment_binding"},"environment":{"python":platform.python_version(),"torch":torch.__version__,"torchvision":torchvision.__version__,"timm":timm.__version__,"cuda":torch.version.cuda}}
    write_json_new(formal/"worker_result.json",y1.signed("stage_y_y1_worker_result","PASS",summary))
    return {"status":"PASS","records":len(records),"retained_bytes":retained,"wall_seconds":total_wall,"h20_gpu_hours":total_gpu/3600}


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser();parser.add_argument("--repo-root",type=Path,default=REPO_ROOT);parser.add_argument("--device",default="cuda:0");args=parser.parse_args(argv)
    try: result=execute(args.repo_root,args.device)
    except Exception as error:
        print(json.dumps({"status":getattr(error,"code","BLOCKED"),"error":str(error)},sort_keys=True));return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__": raise SystemExit(main())
