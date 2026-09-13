#!/usr/bin/env python3
"""Independent CPU-only Y4-R5 source recomputation and validation."""
from __future__ import annotations

import csv
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vitnns import stage_y_y4_r5 as y
from scripts import validate_stage_y_y4_r1 as base

PRIMITIVE_COLUMNS = base.PRIMITIVE_COLUMNS + tuple(
    f"{member}_{metric}"
    for member in ("first_identity", "first_attention", "second_residual", "second_mlp")
    for metric in ("signed_conductance", "positive_mass", "negative_mass", "absolute_mass", "cancellation")
) + ("classification_stratum",)
HEAD_COLUMNS = base.HEAD_COLUMNS + ("raw_ablation_loss","classification_stratum")
AGG_COLUMNS = base.AGG_COLUMNS
PERT_COLUMNS = base.PERT_COLUMNS
INTERVENTION_COLUMNS = ("image_index","selection_order","source_object_id","classification_stratum","block","status","nns_auc","control_median_auc","auc_gain","fisher_z_spearman","top3_minus_bottom3_normalized_loss")
CONTROL_COLUMNS = ("image_index","selection_order","source_object_id","block","seed_index","status","auc")
CURVE_COLUMNS = ("image_index","selection_order","source_object_id","block","rank_group","k","status","heads","raw_logit_loss","normalized_loss")
RAW_COLUMNS = ("unit_index","selection_order","source_object_id","condition_id","block","absolute_mass_cosine_direct","absolute_mass_cosine_transposed","spearman_direct","spearman_transposed","raw_off_diagonal_fraction","nns_off_diagonal_fraction")
TRAJECTORY_COLUMNS = ("metric","block","n","estimate","ci_lower","ci_upper")
SUMMARY_COLUMNS = TRAJECTORY_COLUMNS
STRATUM_COLUMNS = ("metric","classification_stratum","block","n","estimate","ci_lower","ci_upper")
MIDDLE_COLUMNS = ("metric","contrast","n","estimate","ci_lower","ci_upper")
INTERACTION_COLUMNS = ("metric","condition_id","depth_group","n","estimate","ci_lower","ci_upper")
EXTREMA_COLUMNS = ("metric","minimum_block","minimum_estimate","maximum_block","maximum_estimate")
DENOMINATOR_COLUMNS = ("hypothesis_id","expected","VALID","FAILED","NA")
PATCH_METRICS = ("diagonal_absolute_mass_fraction","off_diagonal_absolute_mass_fraction","source_to_destination_rms_radius","destination_entropy","effective_destination_count","source_concentration","destination_concentration","cls_routing_fraction","matrix_signed_cancellation","matrix_effective_rank")
BRANCH_METRICS = tuple(c for c in PRIMITIVE_COLUMNS if c.startswith(("first_identity_","first_attention_","second_residual_","second_mlp_"))) + ("first_attention_share","second_mlp_share","first_reconstruction_max_abs","second_reconstruction_max_abs")
TRAJ_METRICS = tuple(c for c in PRIMITIVE_COLUMNS if c not in ("population","image_index","selection_order","source_object_id","condition_id","block","valid_domain_coverage","classification_stratum"))
PERTURB_METRICS = ("within_patch_ntv_absolute_change","one_minus_head_share_cosine","head_share_spearman","effective_head_count_change","one_minus_inverse_patch_matrix_cosine","first_branch_share_absolute_change","second_branch_share_absolute_change","mean_branch_share_absolute_change","valid_domain_coverage")
K_VALUES = (1,2,3,4,6,8,10,12)

def mass_summary_na(value: np.ndarray) -> dict[str, float | None]:
    x=np.asarray(value,dtype=np.float64);y.require(np.isfinite(x).all(),"nonfinite mass input",y.ScientificError);absolute=float(np.abs(x).sum(dtype=np.float64))
    return {"signed_conductance":float(x.sum(dtype=np.float64)),"positive_mass":float(np.maximum(x,0).sum(dtype=np.float64)),"negative_mass":float(np.minimum(x,0).sum(dtype=np.float64)),"absolute_mass":absolute,"cancellation":None if absolute==0 else 1.0-abs(float(x.sum(dtype=np.float64)))/absolute}

def spatial_support_na(cell: np.ndarray) -> dict[str, float | None]:
    x=np.asarray(cell,dtype=np.float64);y.require(x.shape==(56,56) and np.isfinite(x).all(),"spatial map invalid",y.ScientificError);a=np.abs(x);total=float(a.sum(dtype=np.float64))
    keys=("absolute_mass_entropy","effective_spatial_support","log_effective_spatial_support","gini","top_1_percent_absolute_mass_share","top_10_percent_absolute_mass_share","source_patch_rms_radius","positive_center_displacement","negative_center_displacement","cancellation")
    if total==0:return dict.fromkeys(keys)
    return y.spatial_support(x)

def within_patch_na(cell: np.ndarray) -> dict[str,float|None]:
    x=np.asarray(cell,dtype=np.float64);y.require(x.shape==(56,56) and np.isfinite(x).all(),"within-patch array invalid",y.ScientificError);patch=x.reshape(14,4,14,4).transpose(0,2,1,3).reshape(196,4,4);absolute=np.abs(patch);mass=absolute.sum((1,2),dtype=np.float64);tv=(np.abs(np.diff(patch,axis=1)).sum((1,2),dtype=np.float64)+np.abs(np.diff(patch,axis=2)).sum((1,2),dtype=np.float64))/(mass+1e-12);valid=mass>0;top=entropy=effective=None
    if bool(valid.all()):
        flat=absolute.reshape(196,16);prob=flat/mass[:,None];ent=-(np.where(prob>0,prob*np.log(prob),0.0)).sum(1,dtype=np.float64);top=float((flat.max(1)/mass).mean(dtype=np.float64));entropy=float(ent.mean(dtype=np.float64));effective=float(np.exp(ent).mean(dtype=np.float64))
    return {"within_patch_population_variance":float(np.var(patch,axis=(1,2),ddof=0).mean(dtype=np.float64)),"within_patch_normalized_total_variation":float(tv.mean(dtype=np.float64)),"within_patch_top_cell_absolute_mass_share":top,"within_patch_entropy":entropy,"within_patch_effective_cell_count":effective,**mass_summary_na(x)}

def head_organization_na(head_cells: np.ndarray) -> tuple[dict[str,float|None],list[dict[str,float|None]]]:
    x=np.asarray(head_cells,dtype=np.float64);y.require(x.shape==(12,56,56) and np.isfinite(x).all(),"head roster invalid",y.ScientificError);mass=np.abs(x).sum((1,2),dtype=np.float64);total=float(mass.sum(dtype=np.float64));share=None if total==0 else mass/total
    if total==0:entropy=effective=order=None
    else:entropy,effective=y.entropy_effective(mass);order=np.sort(share)[::-1]
    rows=[]
    for h in y.HEADS:rows.append({"head":h,"absolute_share":None if share is None else float(share[h]),**mass_summary_na(x[h])})
    return {"head_top1_share":None if order is None else float(order[0]),"head_top3_share":None if order is None else float(order[:3].sum()),"head_hhi":None if share is None else float(np.square(share).sum(dtype=np.float64)),"head_entropy":entropy,"effective_head_count":effective},rows

def patch_mixing_na(matrix: np.ndarray, cls: np.ndarray, with_rank: bool=True) -> dict[str,float|None]:
    t=np.asarray(matrix,dtype=np.float64);c=np.asarray(cls,dtype=np.float64);y.require(t.shape==(196,196) and c.shape==(196,) and np.isfinite(t).all() and np.isfinite(c).all(),"patch matrix invalid",y.ScientificError);a=np.abs(t);tm=float(a.sum(dtype=np.float64));cm=float(np.abs(c).sum(dtype=np.float64));total=tm+cm
    out={"diagonal_absolute_mass_fraction":None,"off_diagonal_absolute_mass_fraction":None,"source_to_destination_rms_radius":None,"destination_entropy":None,"effective_destination_count":None,"source_concentration":None,"destination_concentration":None,"cls_routing_fraction":None if total==0 else cm/total,"matrix_signed_cancellation":None if total==0 else 1-abs(float(t.sum(dtype=np.float64)+c.sum(dtype=np.float64)))/total,"matrix_effective_rank":None}
    if tm==0:return out
    diag=float(np.trace(a));src=a.sum(1,dtype=np.float64);dst=a.sum(0,dtype=np.float64);yy,xx=np.indices((14,14));coords=np.stack((yy.reshape(-1),xx.reshape(-1)),axis=1).astype(np.float64);dist2=np.square(coords[:,None,:]-coords[None,:,:]).sum(2,dtype=np.float64);dh,de=y.entropy_effective(dst)
    out.update(diagonal_absolute_mass_fraction=diag/tm,off_diagonal_absolute_mass_fraction=1-diag/tm,source_to_destination_rms_radius=float(math.sqrt(float((a*dist2).sum(dtype=np.float64))/tm)),destination_entropy=dh,effective_destination_count=de,source_concentration=float(np.square(src/tm).sum(dtype=np.float64)),destination_concentration=float(np.square(dst/tm).sum(dtype=np.float64)))
    if with_rank:out["matrix_effective_rank"]=y.entropy_effective(np.linalg.svd(t,compute_uv=False))[1]
    return out

def branch_allocation_na(identity,attention,residual,mlp,total_first,total_second):
    values=[np.asarray(v,dtype=np.float64) for v in (identity,attention,residual,mlp,total_first,total_second)];y.require(all(v.shape==(56,56) and np.isfinite(v).all() for v in values),"branch arrays invalid",y.ScientificError);im,am,rm,mm=[float(np.abs(v).sum(dtype=np.float64)) for v in values[:4]]
    out={"first_attention_share":None if im+am==0 else am/(im+am),"second_mlp_share":None if rm+mm==0 else mm/(rm+mm),"first_reconstruction_max_abs":float(np.max(np.abs(values[0]+values[1]-values[4]))),"second_reconstruction_max_abs":float(np.max(np.abs(values[2]+values[3]-values[5])))}
    for name,value in zip(("first_identity","first_attention","second_residual","second_mlp"),values[:4]):
        for key,item in mass_summary_na(value).items():out[f"{name}_{key}"]=item
    return out

def primitive(arrays,image,selection,source,condition,population,science,stratum):
    rows=[];heads=[];ablocks={} if science is None else {b["block"]:b for b in base.legacy(science)["ablations"]["blocks"]};coverage=float(arrays.get("valid_mask",np.ones((224,224),dtype=bool)).mean(dtype=np.float64))
    for b in y.BLOCKS:
        p=f"b{b:02d}";wp=within_patch_na(arrays[p+"_first_attention_cell"]);sp=spatial_support_na(arrays[p+"_first_attention_cell"]);hs,hr=head_organization_na(arrays[p+"_first_heads_cell"]);pm=patch_mixing_na(arrays[p+"_first_attention_matrix"],arrays[p+"_first_attention_cls"],population=="clean");ba=branch_allocation_na(arrays[p+"_first_identity_cell"],arrays[p+"_first_attention_cell"],arrays[p+"_second_residual_cell"],arrays[p+"_second_mlp_cell"],arrays[p+"_input_cell"],arrays[p+"_input_cell"]);row={"population":population,"image_index":image,"selection_order":selection,"source_object_id":source,"condition_id":condition,"block":b,**wp,**sp,**hs,**pm,**ba,"valid_domain_coverage":coverage};row["attention_positive_mass"]=row.pop("positive_mass");row["attention_negative_mass"]=row.pop("negative_mass");row["attention_absolute_mass"]=row.pop("absolute_mass");row["attention_cancellation"]=row["cancellation"];row["spatial_cancellation"]=row.pop("cancellation");row["spatial_gini"]=row.pop("gini");row["classification_stratum"]=stratum;rows.append(row)
        losses={v["head"]:v for v in ablocks.get(b,{}).get("single_head",[])}
        for h in hr:
            loss=losses.get(h["head"]);heads.append({"population":population,"image_index":image,"selection_order":selection,"source_object_id":source,"condition_id":condition,"block":b,**h,"normalized_ablation_loss":None if loss is None else loss["normalized_loss"],"raw_ablation_loss":None if loss is None else loss["raw_logit_loss"],"classification_stratum":stratum})
    return rows,heads

def recompute_interventions_full(science,head_cells,source,image,selection,stratum):
    block_rows=[];control_rows=[];curve_rows=[]
    if science is None:
        for b in y.BLOCKS:
            block_rows.append({"image_index":image,"selection_order":selection,"source_object_id":source,"classification_stratum":stratum,"block":b,"status":"NA","nns_auc":None,"control_median_auc":None,"auc_gain":None,"fisher_z_spearman":None,"top3_minus_bottom3_normalized_loss":None})
            control_rows.extend({"image_index":image,"selection_order":selection,"source_object_id":source,"block":b,"seed_index":seed,"status":"NA","auc":None} for seed in range(32))
            curve_rows.extend({"image_index":image,"selection_order":selection,"source_object_id":source,"block":b,"rank_group":kind,"k":k,"status":"NA","heads":"[]","raw_logit_loss":None,"normalized_loss":None} for kind in ("top","bottom") for k in K_VALUES)
        return {"A1":None,"B6":None,"B7":None},block_rows,control_rows,curve_rows
    value=base.legacy(science);deletion=value["deletion_operator"];ablation=value["ablations"];y.require(deletion["status"]==ablation["status"]=="PASS","intervention status invalid",y.ScientificError);gains=[];fishers=[];topbottom=[]
    deletes={v["block"]:v for v in deletion["blocks"]};ablates={v["block"]:v for v in ablation["blocks"]};y.require(sorted(deletes)==list(y.BLOCKS) and sorted(ablates)==list(y.BLOCKS),"intervention block roster invalid",y.ScientificError)
    for b in y.BLOCKS:
        d=deletes[b];a=ablates[b];x=np.asarray(d["percentiles"],dtype=np.float64)/100;losses=np.asarray(d["nns_losses"],dtype=np.float64);nns=float(np.trapz(losses,x));controls=[]
        y.require(len(d["controls"])==32 and [v["seed_index"] for v in d["controls"]]==list(range(32)),"deletion control roster invalid",y.ScientificError)
        for control in d["controls"]:
            auc=float(np.trapz(np.asarray(control["losses"],dtype=np.float64),x));y.require(math.isclose(auc,control["auc"],rel_tol=0,abs_tol=1e-12),"control AUC drift",y.ScientificError);controls.append(auc);control_rows.append({"image_index":image,"selection_order":selection,"source_object_id":source,"block":b,"seed_index":control["seed_index"],"status":"VALID","auc":auc})
        median=float(np.median(np.asarray(controls,dtype=np.float64)));gain=nns-median;y.require(math.isclose(nns,d["nns_auc"],rel_tol=0,abs_tol=1e-12) and math.isclose(median,d["control_median_auc"],rel_tol=0,abs_tol=1e-12) and math.isclose(gain,d["auc_gain"],rel_tol=0,abs_tol=1e-12),"deletion endpoint drift",y.ScientificError);gains.append(gain)
        alloc=np.abs(np.asarray(head_cells[b],dtype=np.float64)).sum((1,2),dtype=np.float64);y.require(np.allclose(alloc,np.asarray(a["allocation_absolute_mass"]),rtol=0,atol=1e-9),"stored head allocation drift",y.ScientificError);order=y.stable_descending(alloc,"STAGE_Y_HEAD_RANK_V1_20260910",source,b);y.require(order==a["rank_order"] and len(a["single_head"])==12,"head rank/roster drift",y.ScientificError);single=np.asarray([v["normalized_loss"] for v in sorted(a["single_head"],key=lambda z:z["head"])],dtype=np.float64)
        try:rho=y.spearman(alloc,single);fz=float(np.arctanh(np.clip(rho,-.999999,.999999)))
        except y.ScientificError:fz=None
        for kind,ranking in (("top",order),("bottom",order[::-1])):
            entries={v["k"]:v for v in a["curves"][kind]};y.require(sorted(entries)==list(K_VALUES),"head curve k roster drift",y.ScientificError)
            for k in K_VALUES:
                point=entries[k];expected=ranking[:k];y.require(point["heads"]==expected,"head curve rank drift",y.ScientificError);curve_rows.append({"image_index":image,"selection_order":selection,"source_object_id":source,"block":b,"rank_group":kind,"k":k,"status":"VALID","heads":json.dumps(expected,separators=(",",":")),"raw_logit_loss":point["raw_logit_loss"],"normalized_loss":point["normalized_loss"]})
        top=next(v for v in a["curves"]["top"] if v["k"]==3);bottom=next(v for v in a["curves"]["bottom"] if v["k"]==3);tb=float(top["normalized_loss"]-bottom["normalized_loss"]);fishers.append(fz);topbottom.append(tb);block_rows.append({"image_index":image,"selection_order":selection,"source_object_id":source,"classification_stratum":stratum,"block":b,"status":"VALID","nns_auc":nns,"control_median_auc":median,"auc_gain":gain,"fisher_z_spearman":fz,"top3_minus_bottom3_normalized_loss":tb})
    return {"A1":float(np.mean(gains,dtype=np.float64)),"B6":None if any(v is None for v in fishers) else float(np.mean(fishers,dtype=np.float64)),"B7":float(np.mean(topbottom,dtype=np.float64))},block_rows,control_rows,curve_rows

def load_raw_attention(repo,q,roots):
    result={}
    for row in q["array_records"]:
        name=Path(row["path"]).stem
        if name.startswith("raw_attention_b"):result[int(name[-2:])]=y.validate_array_record(repo,row,roots)
    y.require(len(result) in (0,12),"partial raw-attention roster",y.ScientificError);return result

def safe_cosine(a,b):
    return None if float(np.linalg.norm(np.asarray(a,dtype=np.float64))*np.linalg.norm(np.asarray(b,dtype=np.float64)))==0 else y.cosine_abs(a,b)
def safe_spearman(a,b):
    try:return y.spearman(np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64))
    except y.ScientificError:return None

def source_validation(repo: Path):
    clean_meta=[];non_meta=[];primitive_rows=[];head_rows=[];array_rows=[];clean_cache={};intervention_rows=[];control_rows=[];curve_rows=[];raw_rows=[];class_by_order={r["selection_order"]:r["class_id"] for r in base.clean_sources(repo)}
    for row in base.clean_sources(repo):
        arrays,science,q,valid=y.load_unit(repo,row["source_terminal"],y.Y2_SOURCE_ROOTS,row.get("scientific_result"));y.require(q["selection_order"]==row["selection_order"] and q["source_object_id"]==row["source_object_id"] and q["target_identity"]["rule"]=="clean predicted-class raw logit","clean target/source drift");stratum="correct" if int(q["target_identity"]["class_id"])==int(class_by_order[row["selection_order"]]) else "misclassified";pr,hr=primitive(arrays,row["selection_order"],row["selection_order"],row["source_object_id"],"clean","clean",science,stratum);inter,ir,cr,curves=recompute_interventions_full(science,{b:arrays[f"b{b:02d}_first_heads_cell"] for b in y.BLOCKS},row["source_object_id"],row["selection_order"],row["selection_order"],stratum);clean_meta.append({"index":row["selection_order"],"source_object_id":row["source_object_id"],"condition_id":"clean","status":"VALID","source_terminal":row["source_terminal"]});primitive_rows.extend(pr);head_rows.extend(hr);array_rows.extend(valid);intervention_rows.extend(ir);control_rows.extend(cr);curve_rows.extend(curves);clean_cache[row["selection_order"]]={"primitive":pr,"heads":hr,"intervention":inter,"matrices":{b:arrays[f"b{b:02d}_first_attention_matrix"] for b in y.BLOCKS},"stratum":stratum,"source_object_id":row["source_object_id"]}
    flips=0;condition_rows=[];subset={}
    for row in base.nonclean_sources(repo):
        arrays,science,q,valid=y.load_unit(repo,row["source_terminal"],y.Y3_SOURCE_ROOTS);y.require(q["unit_index"]==row["unit_index"] and q["fixed_target"]["rule"]=="clean predicted-class raw logit" and q["condition_id"]==y.CONDITIONS[q["condition_index"]],"nonclean coordinate/target drift");selection=q["selection_order"];subset[q["subset_rank"]]=selection;clean=clean_cache[selection];pr,hr=primitive(arrays,q["subset_rank"],selection,q["source_object_id"],q["condition_id"],"nonclean",science,clean["stratum"]);primitive_rows.extend(pr);head_rows.extend(hr);array_rows.extend(valid);pred=science["transformed_prediction"];flips+=int(pred["prediction_changed"]);non_meta.append({"index":q["unit_index"],"source_object_id":q["source_object_id"],"condition_id":q["condition_id"],"status":"VALID","source_terminal":row["source_terminal"]});w=arrays["patch_inverse_operator"].astype(np.float64);valid_mask=arrays["valid_mask"];valid_patch=valid_mask.reshape(14,16,14,16).all((1,3)).reshape(196);y.require(np.array_equal(valid_patch,arrays["fully_valid_patch_mask"]),"fully-valid patch mask drift");contract=science["inverse_materialization"];y.require(contract["matrix_formula"]=="W @ T_transformed @ W.T" and contract["cls_formula"]=="W @ C_transformed" and contract["fully_valid_pairs_only"] is True,"inverse formula contract drift")
        y.require(valid_patch.any(),"empty fully-valid patch domain",y.ScientificError)
        if q["condition_id"] in ("brightness_0.9","brightness_1.1","gaussian_noise_0.01_seed_0"):y.require(np.array_equal(w,np.eye(196,dtype=np.float64)) and valid_mask.all(),"identity inverse operator drift")
        if q["condition_id"]=="horizontal_flip":
            expected=np.zeros((196,196),dtype=np.float64)
            for i in range(196):yy,xx=divmod(i,14);expected[i,yy*14+13-xx]=1
            y.require(np.array_equal(w,expected) and valid_mask.all(),"horizontal-flip inverse operator drift")
        cp={r["block"]:r for r in clean["primitive"]};ch={(r["block"],r["head"]):r for r in clean["heads"]};tp={r["block"]:r for r in pr};th={(r["block"],r["head"]):r for r in hr};pairs=np.outer(valid_patch,valid_patch)
        for b in y.BLOCKS:
            ca_values=[ch[(b,h)]["absolute_share"] for h in y.HEADS];ta_values=[th[(b,h)]["absolute_share"] for h in y.HEADS];shares_valid=not any(v is None for v in ca_values+ta_values);ca=np.asarray(ca_values,dtype=np.float64) if shares_valid else None;ta=np.asarray(ta_values,dtype=np.float64) if shares_valid else None;name=f"b{b:02d}_first_attention_matrix";inverse=w@arrays[name].astype(np.float64)@w.T;clean_t=clean["matrices"][b].astype(np.float64);patch_similarity=safe_cosine(clean_t[pairs],inverse[pairs]);condition_rows.append({"image_index":q["subset_rank"],"selection_order":selection,"source_object_id":q["source_object_id"],"condition_id":q["condition_id"],"block":b,"within_patch_ntv_absolute_change":abs(tp[b]["within_patch_normalized_total_variation"]-cp[b]["within_patch_normalized_total_variation"]),"one_minus_head_share_cosine":None if not shares_valid else 1-y.cosine_abs(ca,ta),"head_share_spearman":None if not shares_valid else safe_spearman(ca,ta),"effective_head_count_change":None if tp[b]["effective_head_count"] is None or cp[b]["effective_head_count"] is None else tp[b]["effective_head_count"]-cp[b]["effective_head_count"],"one_minus_inverse_patch_matrix_cosine":None if patch_similarity is None else 1-patch_similarity,"first_branch_share_absolute_change":None if tp[b]["first_attention_share"] is None or cp[b]["first_attention_share"] is None else abs(tp[b]["first_attention_share"]-cp[b]["first_attention_share"]),"second_branch_share_absolute_change":None if tp[b]["second_mlp_share"] is None or cp[b]["second_mlp_share"] is None else abs(tp[b]["second_mlp_share"]-cp[b]["second_mlp_share"]),"mean_branch_share_absolute_change":None if any(v is None for v in (tp[b]["first_attention_share"],cp[b]["first_attention_share"],tp[b]["second_mlp_share"],cp[b]["second_mlp_share"])) else float(np.mean([abs(tp[b]["first_attention_share"]-cp[b]["first_attention_share"]),abs(tp[b]["second_mlp_share"]-cp[b]["second_mlp_share"])],dtype=np.float64)),"valid_domain_coverage":float(valid_mask.mean(dtype=np.float64)),"prediction_changed":pred["prediction_changed"],"full_domain_flip_control":q["condition_id"]=="horizontal_flip" and bool(valid_mask.all())})
        for b,raw in load_raw_attention(repo,q,y.Y3_SOURCE_ROOTS).items():
            raw_patch=raw[:,1:,1:].astype(np.float64).mean(0,dtype=np.float64);nns=arrays[f"b{b:02d}_first_attention_matrix"].astype(np.float64);rm=float(np.abs(raw_patch).sum(dtype=np.float64));nm=float(np.abs(nns).sum(dtype=np.float64));raw_rows.append({"unit_index":q["unit_index"],"selection_order":selection,"source_object_id":q["source_object_id"],"condition_id":q["condition_id"],"block":b,"absolute_mass_cosine_direct":safe_cosine(raw_patch,nns),"absolute_mass_cosine_transposed":safe_cosine(raw_patch.T,nns),"spearman_direct":safe_spearman(raw_patch.reshape(-1),np.abs(nns).reshape(-1)),"spearman_transposed":safe_spearman(raw_patch.T.reshape(-1),np.abs(nns).reshape(-1)),"raw_off_diagonal_fraction":None if rm==0 else 1-float(np.trace(np.abs(raw_patch)))/rm,"nns_off_diagonal_fraction":None if nm==0 else 1-float(np.trace(np.abs(nns)))/nm})
    y.require(flips==2 and sorted(subset)==list(range(20)) and len(set(subset.values()))==20,"perturbation provenance drift");return {"clean":clean_meta,"non":non_meta,"primitive":primitive_rows,"heads":head_rows,"arrays":array_rows,"clean_cache":clean_cache,"perturb":condition_rows,"interventions":intervention_rows,"controls":control_rows,"curves":curve_rows,"raw_attention":raw_rows,"subset":subset,"flips":flips}

def mean_required(values):return None if not values or any(v is None for v in values) else float(np.mean(np.asarray(values,dtype=np.float64),dtype=np.float64))
def aggregate_rows(source):
    rows=[]
    for order,clean in sorted(source["clean_cache"].items()):
        by={r["block"]:r for r in clean["primitive"]};depth=lambda key,group:mean_required([by[b][key] for b in y.DEPTH[group]]);contrast=lambda key:None if depth(key,"deep") is None or depth(key,"shallow") is None else depth(key,"deep")-depth(key,"shallow");inter=clean["intervention"];rows.append({"image_index":order,"selection_order":order,"source_object_id":clean["source_object_id"],"A1":inter["A1"],"B1":contrast("log_effective_spatial_support"),"B2":contrast("effective_head_count"),"B3":contrast("off_diagonal_absolute_mass_fraction"),"B4":contrast("first_attention_share"),"B5":contrast("second_mlp_share"),"B6":inter["B6"],"B7":inter["B7"],"C1":None,"C2":None,"C3":None,"C4":None})
    for selection in sorted(source["subset"].values()):
        condition=[r for r in source["perturb"] if r["selection_order"]==selection];y.require(len(condition)==96,"incomplete perturbation image");target=next(r for r in rows if r["selection_order"]==selection);target["C1"]=mean_required([r["within_patch_ntv_absolute_change"] for r in condition]);target["C2"]=mean_required([r["one_minus_head_share_cosine"] for r in condition]);target["C3"]=mean_required([r["one_minus_inverse_patch_matrix_cosine"] for r in condition]);target["C4"]=mean_required([r["mean_branch_share_absolute_change"] for r in condition])
    return rows

def inference(aggregate):
    rows=[];boot=[];sign=[];valid_ids=[]
    for hid,family,direction,threshold in y.HYPOTHESES:
        values=[r[hid] for r in aggregate if r[hid] is not None];minimum=18 if family=="C" else 90
        if len(values)>=minimum:estimate,lo,hi,br=y.bootstrap(values);p,sr=y.sign_flip(values,direction,threshold);boot.append(br);sign.append(sr);valid_ids.append(hid)
        else:estimate=lo=hi=p=None
        rows.append({"hypothesis_id":hid,"endpoint":y.HYPOTHESIS_ENDPOINTS[hid],"family":family,"direction":direction,"threshold":threshold,"n":len(values),"estimate":estimate,"ci_lower":lo,"ci_upper":hi,"raw_p":p,"holm_rank":None,"holm_threshold":None,"adjusted_p":None,"reject":False,"outcome":"INVALID" if len(values)<minimum else "INCONCLUSIVE"})
    final=[]
    for family in "ABC":
        group=[r for r in rows if r["family"]==family];effective=[{**r,"raw_p":1.0 if r["raw_p"] is None else r["raw_p"]} for r in group];adjusted={r["hypothesis_id"]:r for r in y.holm(effective)}
        for r in group:
            a=adjusted[r["hypothesis_id"]];r.update(holm_rank=a["holm_rank"],holm_threshold=a["holm_threshold"],adjusted_p=a["adjusted_p"],reject=a["reject"]);r["outcome"]=y.hypothesis_outcome(r) if r["raw_p"] is not None else "INVALID";final.append(r)
    final.sort(key=lambda r:r["hypothesis_id"]);return final,(np.stack(boot) if boot else np.empty((0,y.BOOTSTRAP_REPLICATES),dtype="<f8")),(np.stack(sign) if sign else np.empty((0,2,y.SIGN_REPLICATES),dtype="<f8")),valid_ids

def interval(values,with_ci=True):
    values=[v for v in values if v is not None]
    if not values:return 0,None,None,None
    if not with_ci:return len(values),float(np.median(np.asarray(values,dtype=np.float64))),None,None
    est,lo,hi,_=y.bootstrap(values);return len(values),est,lo,hi
def descriptive_tables(source):
    clean=[r for r in source["primitive"] if r["population"]=="clean"];trajectories=[];strata=[];middle=[]
    for metric in TRAJ_METRICS:
        for b in y.BLOCKS:
            n,est,lo,hi=interval([r[metric] for r in clean if r["block"]==b]);trajectories.append({"metric":metric,"block":b,"n":n,"estimate":est,"ci_lower":lo,"ci_upper":hi})
            for stratum in ("correct","misclassified"):
                n,est,lo,hi=interval([r[metric] for r in clean if r["block"]==b and r["classification_stratum"]==stratum],False);strata.append({"metric":metric,"classification_stratum":stratum,"block":b,"n":n,"estimate":est,"ci_lower":lo,"ci_upper":hi})
        by_image={}
        for r in clean:by_image.setdefault(r["selection_order"],{})[r["block"]]=r
        for name,left,right in (("middle_minus_shallow","middle","shallow"),("deep_minus_middle","deep","middle")):
            values=[]
            for by in by_image.values():
                a=mean_required([by[b][metric] for b in y.DEPTH[left]]);c=mean_required([by[b][metric] for b in y.DEPTH[right]])
                if a is not None and c is not None:values.append(a-c)
            n,est,lo,hi=interval(values,False);middle.append({"metric":metric,"contrast":name,"n":n,"estimate":est,"ci_lower":lo,"ci_upper":hi})
    extrema=[]
    for metric in TRAJ_METRICS:
        rows=[r for r in trajectories if r["metric"]==metric and r["estimate"] is not None]
        extrema.append({"metric":metric,"minimum_block":None if not rows else min(rows,key=lambda r:(r["estimate"],r["block"]))["block"],"minimum_estimate":None if not rows else min(r["estimate"] for r in rows),"maximum_block":None if not rows else max(rows,key=lambda r:(r["estimate"],-r["block"]))["block"],"maximum_estimate":None if not rows else max(r["estimate"] for r in rows)})
    interactions=[]
    for metric in PERTURB_METRICS:
        for condition in y.CONDITIONS:
            for group,blocks in y.DEPTH.items():
                values=[]
                for selection in sorted(source["subset"].values()):
                    block_rows=[r for r in source["perturb"] if r["selection_order"]==selection and r["condition_id"]==condition and r["block"] in blocks];value=mean_required([r[metric] for r in block_rows])
                    if value is not None:values.append(value)
                n,est,lo,hi=interval(values,False);interactions.append({"metric":metric,"condition_id":condition,"depth_group":group,"n":n,"estimate":est,"ci_lower":lo,"ci_upper":hi})
    return trajectories,strata,middle,interactions,extrema

def derive(repo):
    y.verify_implementation(repo);source=source_validation(repo);aggregate=aggregate_rows(source);hyp,boot,sign,valid_ids=inference(aggregate);registry=y.read(repo/"manifests/stage_y/claim_registry.json");allowed={r["claim_id"]:r["allowed_wording"] for r in registry["claims"]};claims=y.claim_decisions({r["hypothesis_id"]:r for r in hyp},allowed);trajectories,strata,middle,interactions,extrema=descriptive_tables(source);source.update(aggregate=aggregate,hypotheses=hyp,bootstrap=boot,signflip=sign,valid_hypothesis_ids=valid_ids,claims=claims,trajectories=trajectories,strata=strata,middle=middle,interactions=interactions,extrema=extrema);return source

def table_payload(repo,path,rows,columns,ordering):return y.table_sidecar(repo,path,columns,rows,ordering)
def write_table(repo,name,rows,columns,ordering):return base.write_table(repo,name,rows,columns,ordering)
def producer_access():return {"jpeg_bytes_read":0,"image_decode_count":0,"checkpoint_bytes_read":0,"model_load_count":0,"torch_imported":False,"timm_imported":False,"torchvision_imported":False,"cuda_initialized":False,"cuda_access_count":0,"gpu_execution":False,"download_bytes":0,"new_empirical_attempt_count":0,"clean_rerun_count":0,"perturbation_rerun_count":0,**y.ACCESS_COUNTS}
def resource_payload(start_cpu,start_wall,start_io,root,data):
    value=y.resource_snapshot(start_cpu,start_wall,start_io,root);value["bootstrap_operation_count"]=(len(data["valid_hypothesis_ids"])+sum(r["n"]>0 for r in data["trajectories"]))*y.BOOTSTRAP_REPLICATES;value["sign_flip_operation_count"]=len(data["valid_hypothesis_ids"])*y.SIGN_REPLICATES;return value

def produce(repo):
    y.runtime_guard();start_wall=time.monotonic();start_cpu=time.process_time();start_io=y.process_io();data=derive(repo);root=repo/y.ROOT
    whitelist=[y.file_record(repo/p,repo) for p in (y.Y2_COMPLETION,y.Y2_MANIFEST,y.Y3_COMPLETION,y.Y3_MANIFEST)];y.write(root/"source_read_whitelist.json","binding",{"name":"sealed roots: "+",".join(str(p) for p in y.Y2_SOURCE_ROOTS+y.Y3_SOURCE_ROOTS),"files":whitelist,"files_digest":y.canonical(whitelist),"commit":None});y.write(root/"clean_100_image_source_manifest.json","source_manifest",{"population":"clean","records":data["clean"],"record_count":100,"records_digest":y.canonical(data["clean"]),"arrays_copied":False})
    clean_by_order={r["index"]:r for r in data["clean"]};non_by_index={r["index"]:r for r in data["non"]};combined=[]
    for rank in range(20):
        clean=dict(clean_by_order[data["subset"][rank]]);clean["index"]=rank*9;combined.append(clean)
        for ci in range(8):row=dict(non_by_index[rank*8+ci]);row["index"]=rank*9+ci+1;combined.append(row)
    y.write(root/"perturbation_20x9_source_manifest.json","source_manifest",{"population":"perturbation clean plus eight non-clean conditions","records":combined,"record_count":180,"records_digest":y.canonical(combined),"arrays_copied":False})
    array_cols=("source_terminal","name","path","bytes","sha256","canonical_digest","dtype","shape","axes");array_export=[{**r,"shape":json.dumps(r["shape"],separators=(",",":")),"axes":json.dumps(r["axes"],separators=(",",":"))} for r in data["arrays"]];array_table=write_table(repo,"scientific_array_validation_manifest.csv",array_export,array_cols,["source_terminal","name"]);y.write(root/"scientific_array_validation_manifest.json","array_validation",{"source_unit_count":260,"array_count":len(data["arrays"]),"validated_bytes":sum(r["bytes"] for r in data["arrays"]),"records_digest":y.canonical(data["arrays"]),"dtype_shape_axes_finite_pass":True,"closure_pass":True})
    specs=[("per_image_primitive_metrics.csv",data["primitive"],PRIMITIVE_COLUMNS,["population","image_index","condition_id","block"]),("all_head_organization.csv",data["heads"],HEAD_COLUMNS,["population","image_index","condition_id","block","head"]),("within_patch_intervention.csv",data["interventions"],INTERVENTION_COLUMNS,["selection_order","block"]),("patch_uniform_control_auc.csv",data["controls"],CONTROL_COLUMNS,["selection_order","block","seed_index"]),("head_ablation_curves.csv",data["curves"],CURVE_COLUMNS,["selection_order","block","rank_group","k"]),("per_image_confirmatory_aggregates.csv",data["aggregate"],AGG_COLUMNS,["selection_order"]),("perturbation_modulation.csv",data["perturb"],PERT_COLUMNS,["image_index","condition_id","block"]),("all_block_depth_trajectories.csv",data["trajectories"],TRAJECTORY_COLUMNS,["metric","block"]),("patch_mixing_summary.csv",[r for r in data["trajectories"] if r["metric"] in PATCH_METRICS],SUMMARY_COLUMNS,["metric","block"]),("branch_allocation_summary.csv",[r for r in data["trajectories"] if r["metric"] in BRANCH_METRICS],SUMMARY_COLUMNS,["metric","block"]),("correctness_strata.csv",data["strata"],STRATUM_COLUMNS,["metric","classification_stratum","block"]),("middle_depth_contrasts.csv",data["middle"],MIDDLE_COLUMNS,["metric","contrast"]),("perturbation_condition_depth_summary.csv",data["interactions"],INTERACTION_COLUMNS,["metric","condition_id","depth_group"]),("local_extrema.csv",data["extrema"],EXTREMA_COLUMNS,["metric"]),("raw_attention_associations.csv",data["raw_attention"],RAW_COLUMNS,["unit_index","block"])];tables=[array_table]+[write_table(repo,*spec) for spec in specs]
    bp,bps=base.write_array(repo,"bootstrap_replicates.npy",data["bootstrap"],["valid_hypothesis:"+",".join(data["valid_hypothesis_ids"]),"replicate"]);sp,sps=base.write_array(repo,"sign_flip_replicates.npy",data["signflip"],["valid_hypothesis:"+",".join(data["valid_hypothesis_ids"]),"tail","replicate"]);stream=y.read(repo/y.SPEC)["rng_stream_rule"];y.write(root/"bootstrap_replicate_manifest.json","bootstrap_manifest",{"replicates":y.BOOTSTRAP_REPLICATES,"seed":y.BOOTSTRAP_SEED,"prng":"numpy.random.Generator(numpy.random.PCG64)","stream_rule":stream,"statistic":"cohort median of preregistered image aggregate; INVALID endpoints have no replicate row","quantiles":[.025,.975],"quantile_method":"linear","table":y.file_record(bp,repo),"sidecar":y.file_record(bps,repo)});y.write(root/"sign_flip_replicate_manifest.json","signflip_manifest",{"replicates":y.SIGN_REPLICATES,"seed":y.SIGN_SEED,"prng":"numpy.random.Generator(numpy.random.PCG64)","stream_rule":stream,"plus_one":True,"table":y.file_record(sp,repo),"sidecar":y.file_record(sps,repo)});y.write(root/"hypothesis_test_table.json","hypothesis_table",{"records":data["hypotheses"],"record_count":12,"records_digest":y.canonical(data["hypotheses"])})
    for family in "ABC":
        rows=[r for r in data["hypotheses"] if r["family"]==family];y.write(root/f"family_{family.lower()}_holm.json","holm_table",{"family":family,"alpha":.05,"records":rows,"records_digest":y.canonical(rows)})
    for claim in data["claims"]:y.write(root/"claims"/(claim["claim_id"]+".json"),"claim_decision",claim,claim["decision"])
    y.write(root/"claim_decision_summary.json","claim_summary",{"records":data["claims"],"record_count":6,"records_digest":y.canonical(data["claims"]),"omnibus_decision":False});den=[{"hypothesis_id":r["hypothesis_id"],"expected":20 if r["family"]=="C" else 100,"VALID":r["n"],"FAILED":0,"NA":(20 if r["family"]=="C" else 100)-r["n"]} for r in data["hypotheses"]];den_table=write_table(repo,"hypothesis_denominators.csv",den,DENOMINATOR_COLUMNS,["hypothesis_id"]);tables.append(den_table);y.write(root/"missingness_denominator_report.json","missingness",{"clean":{"expected":100,"VALID":100,"FAILED":0,"NA":0},"perturbation_images":{"expected":20,"VALID":20,"FAILED":0,"NA":0},"nonclean_conditions":{"expected":160,"VALID":160,"FAILED":0,"NA":0},"rules":["three absent historical intervention payloads make only A1/B6/B7 NA","metric-level zero-mass NA retained","no replacement or imputation","paired sides required","image-only inference"]});y.write(root/"descriptive_only_manifest.json","descriptive_manifest",{"tables":[y.file_record(p,repo) for pair in tables for p in pair],"roles":["144-head atlas","source-bound individual 196x196 entries","matrix effective rank","all-block trajectories","middle-depth contrasts and local extrema","correct/misclassified strata","per-condition and condition-by-depth results","all-k ablation curves","available retained raw-attention associations"],"confirmatory_promotion":False});audit=base.false_pass_audit();y.write(root/"false_pass_audit.json","false_pass_audit",audit);y.write(root/"producer_access_audit.json","access_audit",producer_access());y.write(root/"producer_resource_ledger.json","resource",resource_payload(start_cpu,start_wall,start_io,root,data));return {"status":"PASS","clean_valid_count":100,"complete_images":20,"prediction_flip_count":data["flips"],"hypotheses":data["hypotheses"],"claims":data["claims"]}

def compare_csv(path,expected,columns):base.compare_csv(path,expected,columns)
def validate_sidecar(repo,path,rows,columns,ordering):
    doc=y.read(path.with_suffix(path.suffix+".schema.json"));y.validate_document(doc);y.require(doc["payload"]==table_payload(repo,path,rows,columns,ordering),"table sidecar drift: "+path.name)
def compare_table(repo,name,rows,columns,ordering):
    path=repo/y.ROOT/name;compare_csv(path,rows,columns);validate_sidecar(repo,path,rows,columns,ordering)

def validate_derived(repo):
    data=derive(repo);root=repo/y.ROOT;specs=[("scientific_array_validation_manifest.csv",[{**r,"shape":json.dumps(r["shape"],separators=(",",":")),"axes":json.dumps(r["axes"],separators=(",",":"))} for r in data["arrays"]],("source_terminal","name","path","bytes","sha256","canonical_digest","dtype","shape","axes"),["source_terminal","name"]),("per_image_primitive_metrics.csv",data["primitive"],PRIMITIVE_COLUMNS,["population","image_index","condition_id","block"]),("all_head_organization.csv",data["heads"],HEAD_COLUMNS,["population","image_index","condition_id","block","head"]),("within_patch_intervention.csv",data["interventions"],INTERVENTION_COLUMNS,["selection_order","block"]),("patch_uniform_control_auc.csv",data["controls"],CONTROL_COLUMNS,["selection_order","block","seed_index"]),("head_ablation_curves.csv",data["curves"],CURVE_COLUMNS,["selection_order","block","rank_group","k"]),("per_image_confirmatory_aggregates.csv",data["aggregate"],AGG_COLUMNS,["selection_order"]),("perturbation_modulation.csv",data["perturb"],PERT_COLUMNS,["image_index","condition_id","block"]),("all_block_depth_trajectories.csv",data["trajectories"],TRAJECTORY_COLUMNS,["metric","block"]),("patch_mixing_summary.csv",[r for r in data["trajectories"] if r["metric"] in PATCH_METRICS],SUMMARY_COLUMNS,["metric","block"]),("branch_allocation_summary.csv",[r for r in data["trajectories"] if r["metric"] in BRANCH_METRICS],SUMMARY_COLUMNS,["metric","block"]),("correctness_strata.csv",data["strata"],STRATUM_COLUMNS,["metric","classification_stratum","block"]),("middle_depth_contrasts.csv",data["middle"],MIDDLE_COLUMNS,["metric","contrast"]),("perturbation_condition_depth_summary.csv",data["interactions"],INTERACTION_COLUMNS,["metric","condition_id","depth_group"]),("local_extrema.csv",data["extrema"],EXTREMA_COLUMNS,["metric"]),("raw_attention_associations.csv",data["raw_attention"],RAW_COLUMNS,["unit_index","block"])];den=[{"hypothesis_id":r["hypothesis_id"],"expected":20 if r["family"]=="C" else 100,"VALID":r["n"],"FAILED":0,"NA":(20 if r["family"]=="C" else 100)-r["n"]} for r in data["hypotheses"]];specs.append(("hypothesis_denominators.csv",den,DENOMINATOR_COLUMNS,["hypothesis_id"]))
    for spec in specs:compare_table(repo,*spec)
    for name,array in (("bootstrap_replicates.npy",data["bootstrap"]),("sign_flip_replicates.npy",data["signflip"])):
        actual=np.load(root/name,allow_pickle=False);y.require(np.array_equal(actual,array),"replicate regeneration drift: "+name);side=y.read(root/(name+".schema.json"));y.validate_document(side);y.require(side["payload"]["data_canonical_digest"]==y.array_digest(array) and side["payload"]["shape"]==list(array.shape),"replicate sidecar drift: "+name)
    clean_payload={"population":"clean","records":data["clean"],"record_count":100,"records_digest":y.canonical(data["clean"]),"arrays_copied":False};doc=y.read(root/"clean_100_image_source_manifest.json");y.validate_document(doc);y.require(doc["payload"]==clean_payload,"clean source manifest drift")
    clean_by_order={r["index"]:r for r in data["clean"]};non_by_index={r["index"]:r for r in data["non"]};combined=[]
    for rank in range(20):
        clean=dict(clean_by_order[data["subset"][rank]]);clean["index"]=rank*9;combined.append(clean)
        for ci in range(8):row=dict(non_by_index[rank*8+ci]);row["index"]=rank*9+ci+1;combined.append(row)
    expected_perturb={"population":"perturbation clean plus eight non-clean conditions","records":combined,"record_count":180,"records_digest":y.canonical(combined),"arrays_copied":False};doc=y.read(root/"perturbation_20x9_source_manifest.json");y.validate_document(doc);y.require(doc["payload"]==expected_perturb,"perturbation source manifest drift");arr=y.read(root/"scientific_array_validation_manifest.json");y.validate_document(arr);y.require(arr["payload"]["source_unit_count"]==260 and arr["payload"]["array_count"]==len(data["arrays"]) and arr["payload"]["validated_bytes"]==sum(r["bytes"] for r in data["arrays"]) and arr["payload"]["records_digest"]==y.canonical(data["arrays"]),"array validation summary drift")
    hypothesis=y.read(root/"hypothesis_test_table.json");y.validate_document(hypothesis);y.require(hypothesis["payload"]["records"]==data["hypotheses"],"hypothesis drift");summary=y.read(root/"claim_decision_summary.json");y.validate_document(summary);y.require(summary["payload"]["records"]==data["claims"],"claim drift")
    for family in "ABC":
        rows=[r for r in data["hypotheses"] if r["family"]==family];doc=y.read(root/f"family_{family.lower()}_holm.json");y.validate_document(doc);y.require(doc["payload"]["records"]==rows,"Holm drift: "+family)
    for claim in data["claims"]:
        doc=y.read(root/"claims"/(claim["claim_id"]+".json"));y.validate_document(doc);y.require(doc["payload"]==claim and doc["status"]==claim["decision"],"per-claim record drift")
    expected_missing={"clean":{"expected":100,"VALID":100,"FAILED":0,"NA":0},"perturbation_images":{"expected":20,"VALID":20,"FAILED":0,"NA":0},"nonclean_conditions":{"expected":160,"VALID":160,"FAILED":0,"NA":0},"rules":["three absent historical intervention payloads make only A1/B6/B7 NA","metric-level zero-mass NA retained","no replacement or imputation","paired sides required","image-only inference"]};doc=y.read(root/"missingness_denominator_report.json");y.validate_document(doc);y.require(doc["payload"]==expected_missing,"missingness report drift")
    for name,array,axes in (("bootstrap_replicates.npy",data["bootstrap"],["valid_hypothesis:"+",".join(data["valid_hypothesis_ids"]),"replicate"]),("sign_flip_replicates.npy",data["signflip"],["valid_hypothesis:"+",".join(data["valid_hypothesis_ids"]),"tail","replicate"])):
        side=y.read(root/(name+".schema.json"));expected=y.npy_sidecar(repo,root/name,array,axes);y.require(side["payload"]==expected,"replicate full sidecar drift: "+name)
    audit=y.read(root/"false_pass_audit.json");y.validate_document(audit);y.require(audit["payload"]==base.false_pass_audit(),"false-pass audit drift");return data

def combine_access(producer,validator):
    result=dict(producer);result["scientific_files_read"]+=validator["scientific_files_read"];result["scientific_bytes_read"]+=validator["scientific_bytes_read"];return result
def combine_resource(producer,validator,root,started_ns):
    result=dict(producer)
    for key in ("cpu_seconds","cpu_hours","process_wall_seconds","read_bytes","written_bytes","bootstrap_operation_count","sign_flip_operation_count"):result[key]+=validator[key]
    result["stage_wall_seconds"]=(time.time_ns()-started_ns)/1e9;result["peak_rss_bytes"]=max(result["peak_rss_bytes"],validator["peak_rss_bytes"]);files=[p for p in root.rglob("*") if p.is_file()];result["retained_bytes"]=sum(p.stat().st_size for p in files);result["file_count"]=len(files);return result

def seal(repo):
    y.runtime_guard();start_wall=time.monotonic();start_cpu=time.process_time();start_io=y.process_io();data=validate_derived(repo);root=repo/y.ROOT;producer_access_doc=y.read(root/"producer_access_audit.json");producer_resource_doc=y.read(root/"producer_resource_ledger.json");y.validate_document(producer_access_doc);y.validate_document(producer_resource_doc);validator_access=producer_access();validator_resource=resource_payload(start_cpu,start_wall,start_io,root,data);y.write(root/"validator_access_audit.json","access_audit",validator_access);y.write(root/"validator_resource_ledger.json","resource",validator_resource);y.write(root/"access_audit.json","access_audit",combine_access(producer_access_doc["payload"],validator_access));pending=y.read(root/"completion/y4_r5.pending.json");final_resource=combine_resource(producer_resource_doc["payload"],validator_resource,root,pending["payload"]["started_unix_ns"]);y.write(root/"resource_ledger.json","resource",final_resource)
    checks=["Y0/Y2/Y3 historical Git/LFS bindings","100 clean and 160 non-clean producing-source recomputation","180 condition-coordinate identity","all file and array SHA/canonical/dtype/shape/axes/finite identities","closure target inverse-map and valid-mask contracts","registered coordinate-level NA retained","image-only aggregation","20,000 bootstrap and 100,000 sign-flip byte regeneration","per-family Holm and twelve outcomes","six claim truth tables","all derived tables and sidecars","CPU-only access/resource and false-pass audits"];val={"source_recomputed":True,"checks":checks,"source_digest":y.canonical(data["clean"]+data["non"]),"metric_digest":y.canonical(data["primitive"]+data["heads"]+data["aggregate"]+data["perturb"]+data["interventions"]+data["curves"]),"bootstrap_digest":y.array_digest(data["bootstrap"]),"signflip_digest":y.array_digest(data["signflip"]),"hypothesis_digest":y.canonical(data["hypotheses"]),"claim_digest":y.canonical(data["claims"]),"status":"PASS"};y.write(root/"source_recomputing_validation_report.json","validation_report",val)
    report=repo/y.REPORT;report.parent.mkdir(parents=True,exist_ok=True);fmt=lambda value:"" if value is None else format(value,".9g");lines=["# Stage Y4-R5 scientific decision report","","Workflow: **PASS**. CPU-only source-recomputed analysis of the frozen ViT-B/16 Stage Y evidence.","","|ID|n|estimate|95% CI|family|Holm adjusted p|reject|outcome|","|---|---:|---:|---:|---|---:|---|---|"]+[f"|{r['hypothesis_id']}|{r['n']}|{fmt(r['estimate'])}|{'' if r['ci_lower'] is None else '['+fmt(r['ci_lower'])+', '+fmt(r['ci_upper'])+']'}|{r['family']}|{fmt(r['adjusted_p'])}|{r['reject']}|{r['outcome']}|" for r in data["hypotheses"]]+["","## Independent claim decisions",""]+[f"- {c['claim_id']}: {c['decision']}" for c in data["claims"]]+["",f"Prediction flips (provenance/descriptive only): {data['flips']}.","","All wording is limited to the frozen ViT-B/16, Stage Y cohort/subset, fixed clean predicted-class raw-logit target, cuts, metrics, thresholds and endpoints.","","No Y5, F3-Final, Stage G, paper writing, image decode, model load or GPU execution occurred.","",f"Next authorization: `{y.NEXT}`.",""];fd=os.open(report,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);os.write(fd,("\n".join(lines)).encode());os.fsync(fd);os.close(fd)
    prior=repo/"evidence/stage_y/y2_r2_r2_clean_formal_20260912/formal/cumulative_resource_ledger.json";y3=repo/"evidence/stage_y/y3_r6_reconciliation_formal_20260913/formal/resource_ledger.json";prior_payload=y.read(prior)["payload"]["cumulative"];y3_payload=y.read(y3)["payload"];y2_cpu=float(prior_payload["measured_cpu_seconds"]);y2_gpu=float(prior_payload["gpu_active_seconds"]);y3_cpu=float(y3_payload["combined_cpu_seconds"]);y3_gpu=float(y3_payload["combined_gpu_seconds"]);y4_cpu=float(final_resource["cpu_seconds"]);y.write(root/"cumulative_stage_y_resource_ledger.json","cumulative_resource",{"prior_stage_ledger":y.file_record(prior,repo),"y3_stage_ledger":y.file_record(y3,repo),"y4_ledger":y.file_record(root/"resource_ledger.json",repo),"y0_y2_cpu_seconds_lower_bound":y2_cpu,"y0_y2_gpu_seconds":y2_gpu,"y3_cpu_seconds":y3_cpu,"y3_gpu_seconds":y3_gpu,"y4_cpu_seconds":y4_cpu,"y4_gpu_seconds":0,"cumulative_cpu_seconds_lower_bound":y2_cpu+y3_cpu+y4_cpu,"cumulative_gpu_seconds":y2_gpu+y3_gpu,"cumulative_h20_gpu_hours":(y2_gpu+y3_gpu)/3600,"note":"Prior sealed ledgers retained verbatim; cumulative CPU is a lower bound where historical observations were absent. Y4 has no frozen numeric CPU/wall/storage cap and used zero GPU/download."});rec={"workflow_status":"PASS","clean_denominator":100,"clean_valid_count":100,"perturbation_denominator_images":20,"perturbation_complete_image_count":20,"condition_record_count":180,"nonclean_terminal_count":160,"hypothesis_count":12,"claim_count":6,"formal_outcome_access":True,"y5_executed":False,"next_authorization":y.NEXT};y.write(root/"reconciliation.json","reconciliation",rec);excluded={"completion/y4_r5.pending.json","completion/y4_r5.json","whole_root_inventory.json","payload_manifest.json"};y.write(root/"payload_manifest.json","inventory",y.payload_inventory(root,excluded,repo));y.write(root/"whole_root_inventory.json","inventory",y.payload_inventory(root,excluded-{"payload_manifest.json"},repo));completion={**rec,"new_empirical_attempt_count":0,"gpu_execution":False,"bootstrap_replicates":y.BOOTSTRAP_REPLICATES,"sign_flip_replicates":y.SIGN_REPLICATES,"holm_families":["A","B","C"],"per_claim_decisions":{c["claim_id"]:c["decision"] for c in data["claims"]},"reconciliation":y.file_record(root/"reconciliation.json",repo),"validation_report":y.file_record(root/"source_recomputing_validation_report.json",repo),"whole_root_inventory":y.file_record(root/"whole_root_inventory.json",repo),"payload_manifest":y.file_record(root/"payload_manifest.json",repo),"report":y.file_record(report,repo),"failure":None};y.publish_terminal(root/"completion/y4_r5.pending.json",root/"completion/y4_r5.json",y.signed("completion",completion));return {"status":"PASS","hypotheses":data["hypotheses"],"claims":data["claims"],"prediction_flip_count":data["flips"]}

def validate_static_artifacts(repo):
    root=repo/y.ROOT;c=y.read(root/"completion/y4_r5.json");y.validate_document(c);y.require(c["status"]=="PASS" and not list(root.rglob("*.pending.json")) and not list(root.rglob("*.tmp")),"completion/pending drift");y.require(not any(p.suffix.lower() in y.FORBIDDEN_SUFFIXES for p in root.rglob("*") if p.is_file()),"forbidden formal file",y.AccessError)
    for name in ("producer_access_audit.json","validator_access_audit.json","access_audit.json","producer_resource_ledger.json","validator_resource_ledger.json","resource_ledger.json","false_pass_audit.json","source_recomputing_validation_report.json","reconciliation.json","payload_manifest.json","whole_root_inventory.json"):
        doc=y.read(root/name);y.validate_document(doc)
    for name in ("payload_manifest.json","whole_root_inventory.json"):
        doc=y.read(root/name);y.require(doc["payload"]==y.payload_inventory(root,set(doc["payload"]["excluded"]),repo),"inventory drift: "+name)
    access=y.read(root/"access_audit.json")["payload"];y.require(all(access[k] in (0,False) for k in ("jpeg_bytes_read","image_decode_count","checkpoint_bytes_read","model_load_count","torch_imported","timm_imported","torchvision_imported","cuda_initialized","cuda_access_count","gpu_execution","download_bytes","new_empirical_attempt_count","clean_rerun_count","perturbation_rerun_count")),"access audit failure",y.AccessError);return c

def validate_completion(repo,publication=False):
    data=validate_derived(repo);c=validate_static_artifacts(repo);root=repo/y.ROOT;h=data["hypotheses"];claims=data["claims"]
    if publication:
        rem=y.remote_refs(repo);head=y.git(repo,"rev-parse","HEAD");y.require(head==y.git(repo,"rev-parse","@{upstream}")==rem.get("refs/heads/"+y.BRANCH)==y.git(repo,"rev-parse",y.PASS_REF)==rem.get(y.PASS_REF),"publication refs drift");y.require(rem.get(y.PRODUCER_REF)==y.verify_implementation(repo),"producer ref drift");y.require(not y.git(repo,"status","--porcelain"),"publication worktree dirty");current=[r for r in y.stage_ref_snapshot(repo) if r["ref"] not in (y.PRODUCER_REF,y.PASS_REF)];pre=y.read(root/"producer_preflight.json")["payload"];y.require(y.canonical(current)==pre["startup_refs_digest"],"historical refs changed since startup")
    return {"status":"PASS","hypotheses":h,"claims":claims,"prediction_flip_count":2,"completion_sha256":y.sha(root/"completion/y4_r5.json"),"completion_canonical_digest":c["canonical_digest"],"inventory_sha256":y.sha(root/"whole_root_inventory.json"),"inventory_canonical_digest":y.read(root/"whole_root_inventory.json")["canonical_digest"],"publication":publication,"next_authorization":y.NEXT}

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("false-pass-audit","recompute","validate-completion","validate-publication"));parser.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();repo=args.repo_root.resolve();os.environ.setdefault("CUDA_VISIBLE_DEVICES","");y.runtime_guard()
    try:out=base.false_pass_audit() if args.command=="false-pass-audit" else {"status":"PASS","prediction_flip_count":derive(repo)["flips"]} if args.command=="recompute" else validate_completion(repo,args.command=="validate-publication")
    except Exception as error:print(json.dumps({"status":getattr(error,"code","BLOCKED_INFRASTRUCTURE"),"error":str(error),"next_authorization":"NONE"},sort_keys=True));return 2
    print(json.dumps(out,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
