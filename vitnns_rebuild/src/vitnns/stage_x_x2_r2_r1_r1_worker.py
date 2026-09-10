"""Formal-only X2-R2-R1-R1 worker using source-validated clean/gamma inputs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import REPO_ROOT
from . import stage_x_x2_r2_r1_worker as source_worker
from . import stage_x_x2_r2_r1_r1 as control

ATTEMPT_ID=control.ATTEMPT_ID
METHODS=source_worker.METHODS
COMPETITIVE=source_worker.COMPETITIVE
SECONDARY=source_worker.SECONDARY
CLEAN_SUMMARY_FIELDS=source_worker.CLEAN_SUMMARY_FIELDS
GAMMA_SUMMARY_FIELDS=source_worker.GAMMA_SUMMARY_FIELDS
FAITH_FIELDS=source_worker.FAITH_FIELDS
FAITH_ROAD_FIELDS=source_worker.FAITH_ROAD_FIELDS
FAITH_CURVE_FIELDS=source_worker.FAITH_CURVE_FIELDS
STABILITY_FIELDS=source_worker.STABILITY_FIELDS
SANITY_FIELDS=source_worker.SANITY_FIELDS


class X2R2R1R1WorkerError(ValueError): pass


def _records(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {row["name"]:row["value"] for row in value["payload"][name]}


def _model_spec(repo: Path, model_id: str) -> Mapping[str, Any]:
    return source_worker._model_spec(repo,model_id)


def _load_model(repo: Path, spec: Mapping[str, Any], device: str) -> Any:
    return source_worker._load_model(repo,spec,device)


def _load_image(repo: Path, sample: Mapping[str, Any], spec: Mapping[str, Any], device: str) -> tuple[Any,dict[str,Any]]:
    return source_worker._load_image(repo,sample,spec,device)


def _phase_dir(repo: Path, unit: Mapping[str, Any], phase: str) -> Path:
    return repo/control.ROOT/"units"/str(unit["unit_digest"])/"attempts"/ATTEMPT_ID/phase


def _journal(repo: Path, unit: Mapping[str, Any], phase: str, event: str, checkpoint_sha: str) -> Path:
    return repo/control.ROOT/"units"/str(unit["unit_digest"])/"attempts"/ATTEMPT_ID/"journal"/f"{phase}.{event}.json"


def _scientific_marker(repo: Path, unit: Mapping[str, Any], phase: str) -> Path:
    return repo/control.ROOT/"units"/str(unit["unit_digest"])/"attempts"/ATTEMPT_ID/"journal"/f"{phase}.scientific_kernel_started.marker"


def _journal_value(repo: Path, unit: Mapping[str, Any], phase: str, event: str, checkpoint_sha: str) -> dict[str, Any]:
    return control.signed("stage_x_x2_r2_r1_r1_attempt_journal","PENDING" if event=="started" else "PASS",control.payload(
        identity={"attempt_id":ATTEMPT_ID,"event":event,"phase":phase,"unit_digest":unit["unit_digest"]},
        bindings=[control.artifact_record(repo,control.RUN_SPEC),control.artifact_record(repo,control.IMPLEMENTATION_SEAL),control.artifact_record(repo,control.MODEL_BINDING),control.artifact_record(repo,control.ROOT/"development_unit_manifest.json"),control.artifact_record(repo,control.ROOT/"development_sample_manifest.json"),control.artifact_record(repo,control.PREFLIGHT_RECEIPT)],
        state={"sample_sha256":unit["sample_sha256"],"checkpoint_sha256":checkpoint_sha,"resume_source_validated":True},
        terminal=control.terminal_state(terminal=event=="completed",immutable=True,complete=event=="completed"),
    ))


def _start_journal(repo: Path, unit: Mapping[str, Any], phase: str, checkpoint_sha: str) -> None:
    path=_journal(repo,unit,phase,"started",checkpoint_sha);expected=_journal_value(repo,unit,phase,"started",checkpoint_sha)
    if path.exists():
        observed=control.strict_read(path);control.validate_artifact(observed,repo,verify_files=True)
        if observed!=expected: raise X2R2R1R1WorkerError("attempt start journal binding drifted")
    else: control.write_new(path,expected,repo)


def _complete_journal(repo: Path, unit: Mapping[str, Any], phase: str, checkpoint_sha: str) -> None:
    path=_journal(repo,unit,phase,"completed",checkpoint_sha)
    if not path.exists(): control.write_new(path,_journal_value(repo,unit,phase,"completed",checkpoint_sha),repo)


def _write_temp_bytes(path: Path, data: bytes) -> None:
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(descriptor,"wb") as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())


def _csv_bytes(fields: Any, rows: Any) -> bytes:
    output=io.StringIO(newline="");writer=csv.DictWriter(output,fieldnames=list(fields),lineterminator="\n",extrasaction="raise");writer.writeheader()
    for row in rows:
        if set(row)!=set(fields):raise X2R2R1R1WorkerError("CSV field mismatch")
        writer.writerow(row)
    return output.getvalue().encode()


def _save_array(path: Path, value: Any) -> None:
    array=np.ascontiguousarray(value,dtype=np.float32)
    if not np.isfinite(array).all():raise X2R2R1R1WorkerError("scientific array is non-finite")
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(descriptor,"wb") as stream:np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())


def _file_record_from_temp(repo: Path, temp: Path, final: Path, *, array: bool=False) -> dict[str, Any]:
    relative=final.relative_to(repo)
    if array:
        from .stage_f_f3c1_kernels import array_digest
        value=np.load(temp,allow_pickle=False)
        return {"kind":"array","path":relative.as_posix(),"bytes":temp.stat().st_size,"sha256":control.sha_file(temp),"canonical_digest":array_digest(value),"dtype":str(value.dtype),"shape":list(value.shape)}
    return {"kind":"file","path":relative.as_posix(),"bytes":temp.stat().st_size,"sha256":control.sha_file(temp),"canonical_digest":None}


def _publish_phase(repo: Path, unit: Mapping[str, Any], phase: str, rows: Any, arrays: Mapping[str,Any], resources: Mapping[str,Any], counts: Mapping[str,Any], state: Mapping[str,Any], *, status: str="PASS") -> dict[str,Any]:
    final=_phase_dir(repo,unit,phase)
    if final.exists():
        terminal=control.strict_read(final/"terminal.json");control.validate_artifact(terminal,repo,verify_files=True);return terminal
    final.parent.mkdir(parents=True,exist_ok=True);temporary=Path(tempfile.mkdtemp(prefix=f".{phase}.",dir=final.parent));files=[]
    try:
        for name,fields,data in rows:
            path=temporary/name;_write_temp_bytes(path,_csv_bytes(fields,data));files.append(_file_record_from_temp(repo,path,final/name))
        for name,value in sorted(arrays.items()):
            path=temporary/name;_save_array(path,value);files.append(_file_record_from_temp(repo,path,final/name,array=True))
        pending=control.signed("stage_x_x2_r2_r1_r1_phase_manifest","PENDING",control.payload(identity={"attempt_id":ATTEMPT_ID,"phase":phase,"unit_digest":unit["unit_digest"]},bindings=[control.artifact_record(repo,control.RUN_SPEC),control.artifact_record(repo,control.IMPLEMENTATION_SEAL)],files=files,counts=counts,state=state,resources=resources,terminal=control.terminal_state()))
        actual_resources=dict(resources);actual_resources["retained_bytes"]=sum(int(row["bytes"]) for row in files)
        terminal=control.signed("stage_x_x2_r2_r1_r1_phase_manifest",status,control.payload(identity={"attempt_id":ATTEMPT_ID,"phase":phase,"unit_digest":unit["unit_digest"]},bindings=[control.artifact_record(repo,control.RUN_SPEC),control.artifact_record(repo,control.IMPLEMENTATION_SEAL)],files=files,counts=counts,state=state,resources=actual_resources,terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))
        control.validate_artifact(pending,repo);control.validate_artifact(terminal,repo)
        _write_temp_bytes(temporary/"terminal.pending.json",control.canonical_json_bytes(pending)+b"\n");_write_temp_bytes(temporary/"terminal.json.atomic",control.canonical_json_bytes(terminal)+b"\n");os.replace(temporary/"terminal.json.atomic",temporary/"terminal.json");(temporary/"terminal.pending.json").unlink();os.replace(temporary,final);return terminal
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def _resource(started: float, cpu_started: float, device: str, forward_calls: int, backward_calls: int, retained_bytes: int=0) -> dict[str,Any]:
    return source_worker._resource(started,cpu_started,device,forward_calls,backward_calls,retained_bytes)


def _summary_map(value: Mapping[str, Any]) -> np.ndarray:
    return source_worker._summary_map(value)


def _read_one_csv(path: Path) -> dict[str,str]:
    return source_worker._read_one_csv(path)


def _phase_file(repo: Path, phase: Path, name: str) -> Path:
    return control.phase_file(repo,phase,control.strict_read(phase/"terminal.json"),name)


def _delete_support(image: Any, score: np.ndarray, *, positive: bool) -> Any:
    return source_worker._delete_support(image,score,positive=positive)


def _randomize_module(module: Any, seed: int, device: str) -> None:
    return source_worker._randomize_module(module,seed,device)


def _selected_gamma(repo: Path) -> tuple[float,float]:
    path=repo/control.ROOT/"gamma_selection.json";value=control.strict_read(path);control.validate_artifact(value,repo,verify_files=True);state=_records(value,"state")
    return float(state["selected_conv_gamma"]),float(state["selected_linear_gamma"])


def _load_map(path: Path) -> np.ndarray:
    value=np.load(path,allow_pickle=False)
    if value.dtype!=np.float32 or list(value.shape)!=[224,224] or not np.isfinite(value).all():raise X2R2R1R1WorkerError(f"map drift: {path}")
    return value


def _source_gamma_file(repo: Path, unit: Mapping[str,Any], gamma_name: str) -> Path:
    terminal_path=repo/control.SOURCE_ROOT/"units"/unit["unit_digest"]/f"attempts/{ATTEMPT_ID}/gamma/terminal.json"
    terminal=control.source.strict_read(terminal_path);control.source.validate_artifact(terminal,repo,verify_files=True)
    matches=[repo/row["path"] for row in terminal["payload"]["files"] if Path(row["path"]).name==gamma_name]
    if len(matches)!=1:raise X2R2R1R1WorkerError(f"source selected gamma map not uniquely bound: {gamma_name}")
    return matches[0]


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
    started=time.monotonic();cpu_started=time.process_time();torch.cuda.reset_peak_memory_stats(device);image,_=_load_image(repo,sample,spec,device);_write_temp_bytes(_scientific_marker(repo,unit,"formal"),b"scientific_kernel_started\n");conv,linear=_selected_gamma(repo);gamma_name=f"attnlrp_c{conv:g}_l{linear:g}.npy"
    maps={name:_load_map(_phase_file(repo,clean,name+".npy")) for name in METHODS if name not in {"attnlrp"}};sealed_gamma_map=_load_map(_source_gamma_file(repo,unit,gamma_name));selected_gamma=comparators.attnlrp(model,image,target,conv_gamma=conv,linear_gamma=linear);selected_gamma_map=_summary_map(selected_gamma)
    if not np.allclose(selected_gamma_map,sealed_gamma_map,atol=1e-6,rtol=1e-6):raise X2R2R1R1WorkerError("sealed gamma recomputation drifted")
    maps["attnlrp"]=selected_gamma_map
    faith=[];road_rows=[];curve_rows=[];forward_calls=0
    for method in METHODS:
        attribution=torch.from_numpy(maps[method]).to(device);identity=str(unit["unit_digest"])+"|formal|"+method;road=metrics.road_combined(model,image,attribution,target,identity);curve=metrics.insertion_deletion(model,image,attribution,target,identity)
        faith.append({"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"class_id":sample["class_id"],"method_id":method,"road_combined":road["road_combined"],"road_morf":road["road_morf"],"road_lerf":road["road_lerf"],**curve["means"]})
        road_rows.extend({"unit_digest":unit["unit_digest"],"method_id":method,**row} for row in road["rows"])
        for record in curve["repeats"]:
            curve_rows.extend({"unit_digest":unit["unit_digest"],"method_id":method,"repeat":record["repeat"],"seed":record["seed"],"fraction":metrics.FRACTIONS[index],"actual_fraction":record["actual_fractions"][index],"insertion":record["insertion"][index],"deletion":record["deletion"][index]} for index in range(len(metrics.FRACTIONS)))
        forward_calls+=road["forward_calls"]+curve["forward_calls"]
    clean_nns=maps["nns_v2"];stability=[];backward_calls=1;forward_calls+=1
    for transformation in control.strict_read(repo/control.OLD_CONFIG)["stability"]["transformations"]:
        changed,metadata=metrics.transform_image(image,transformation,str(unit["unit_digest"]),spec["preprocess"])
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
    if phase!="formal":raise X2R2R1R1WorkerError("X2-R2-R1-R1 worker is formal-only")
    preflight=control.preflight_worker(repo,"formal");control.validate_artifact(preflight,repo,verify_files=True)
    for required in (control.RUN_SPEC,control.IMPLEMENTATION_SEAL,control.MODEL_BINDING,control.ROOT/"development_sample_manifest.json",control.ROOT/"development_unit_manifest.json",control.ROOT/"clean_coverage.json",control.ROOT/"gamma_selection.json"):
        control.validate_artifact(control.strict_read(repo/required),repo,verify_files=True)
    units=[row for row in control.load_units(repo) if int(row["shard"])==shard]
    units=[row for row in units if control.strict_read(_phase_dir(repo,row,"clean")/"terminal.json")["status"]=="PASS"]
    if limit is not None:units=units[:limit]
    samples={row["source_object_id"]:row for row in control.load_samples(repo)};results=[];model=None;loaded=None
    for unit in units:
        stage_failure=repo/control.ROOT/"stage_failure.json"
        if stage_failure.exists():
            stopped=control.strict_read(stage_failure);control.validate_artifact(stopped,repo,verify_files=True);return {"status":stopped["status"],"phase":phase,"shard":shard,"global_stop_before_unit":unit["unit_digest"],"results":results}
        unit_started=time.monotonic();unit_cpu_started=time.process_time();model_loaded=False;spec=None
        try:
            if loaded!=unit["model_id"]:
                if model is not None:del model
                spec=_model_spec(repo,str(unit["model_id"]));model=_load_model(repo,spec,device);loaded=unit["model_id"]
            else:spec=_model_spec(repo,str(unit["model_id"]))
            model_loaded=True;result=run_formal_unit(repo,model,spec,samples[str(unit["sample_id"])],unit,device)
        except Exception as error:
            from .stage_x_nns_v2 import NNSV2ConvergenceError, NNSV2Error, NNSV2MemoryPlanError
            from .stage_x_comparators import ComparatorError
            scientific_started=_scientific_marker(repo,unit,phase).exists()
            scientific=isinstance(error,(NNSV2ConvergenceError,NNSV2Error,ComparatorError)) and not isinstance(error,NNSV2MemoryPlanError) and scientific_started
            status="NO_GO_METHOD_VALIDITY" if scientific else ("BLOCKED_DETERMINISTIC_MEMORY_PLAN" if isinstance(error,NNSV2MemoryPlanError) else "BLOCKED_INFRASTRUCTURE")
            bindings=[]
            if scientific:
                terminal=control.signed("stage_x_x2_r2_r1_r1_unit_terminal","FAILED",control.payload(identity={"attempt_id":ATTEMPT_ID,"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"sample_id":unit["sample_id"],"sample_sha256":unit["sample_sha256"]},counts={"completed_phase_count":len(results)},state={"failure_code":status,"failure_type":type(error).__name__,"failure_message":str(error),"actual_execution_attempted":True,"model_loaded":bool(model_loaded),"jpeg_decoded":True,"scientific_kernel_started":True},terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))
                terminal_path=repo/control.ROOT/"units"/str(unit["unit_digest"])/"terminal.json"
                if not terminal_path.exists():control.publish_atomic(terminal_path.with_name("terminal.pending.json"),terminal_path,terminal,repo)
                bindings=[control.artifact_record(repo,terminal_path.relative_to(repo))]
            stage_path=repo/control.ROOT/"stage_failure.json";stage=control.signed("stage_x_x2_r2_r1_r1_status",status,control.payload(bindings=bindings,counts={"completed_units_before_stop":len(results)},state={"failure_code":status,"failure_type":type(error).__name__,"failure_message":str(error),"global_stop":True,"actual_execution_attempted":scientific,"model_loaded":bool(model_loaded),"jpeg_decoded":scientific_started,"scientific_kernel_started":scientific_started,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=control.terminal_state(terminal=True,immutable=True,complete=True)))
            if not stage_path.exists():control.publish_atomic(stage_path.with_name("stage_failure.pending.json"),stage_path,stage,repo)
            return {"status":status,"phase":phase,"shard":shard,"failed_unit":unit["unit_digest"],"error":f"{type(error).__name__}: {error}","results":results}
        results.append(result);del model;model=None;loaded=None
    return {"status":"PASS","phase":phase,"shard":shard,"results":results}


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("formal",));parser.add_argument("--repo-root",type=Path,default=REPO_ROOT);parser.add_argument("--shard",type=int,required=True,choices=range(4));parser.add_argument("--device",required=True);parser.add_argument("--limit",type=int);args=parser.parse_args(argv)
    result=run_phase(args.repo_root.resolve(),args.phase,args.shard,args.device,args.limit);print(json.dumps(result,sort_keys=True,indent=2));return 0 if result["status"]=="PASS" else 2


if __name__=="__main__": raise SystemExit(main())
