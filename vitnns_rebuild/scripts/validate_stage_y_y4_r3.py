#!/usr/bin/env python3
from __future__ import annotations
import os
import numpy as np
from vitnns import stage_y_y4_r3 as y
from scripts import validate_stage_y_y4_r1 as _v

def primitive(arrays,image,selection,source,condition,population,science):
 rows=[];heads=[];ablocks={} if science is None else {b["block"]:b for b in _v.legacy(science)["ablations"]["blocks"]};coverage=float(arrays.get("valid_mask",np.ones((224,224),dtype=bool)).mean(dtype=np.float64))
 for b in y.BLOCKS:
  p=f"b{b:02d}";wp=y.within_patch(arrays[p+"_first_attention_cell"]);sp=y.spatial_support(arrays[p+"_first_attention_cell"]);hs,hr=y.head_organization(arrays[p+"_first_heads_cell"]);pm=y.patch_mixing(arrays[p+"_first_attention_matrix"],arrays[p+"_first_attention_cls"]);ba=y.branch_allocation(arrays[p+"_first_identity_cell"],arrays[p+"_first_attention_cell"],arrays[p+"_second_residual_cell"],arrays[p+"_second_mlp_cell"],arrays[p+"_input_cell"],arrays[p+"_input_cell"]);row={"population":population,"image_index":image,"selection_order":selection,"source_object_id":source,"condition_id":condition,"block":b,**wp,**sp,**hs,**pm,**ba,"valid_domain_coverage":coverage};row["attention_positive_mass"]=row.pop("positive_mass");row["attention_negative_mass"]=row.pop("negative_mass");row["attention_absolute_mass"]=row.pop("absolute_mass");row["attention_cancellation"]=row.pop("cancellation");row["spatial_gini"]=row.pop("gini");rows.append(row);losses={v["head"]:v["normalized_loss"] for v in ablocks.get(b,{}).get("single_head",[])}
  for h in hr:heads.append({"population":population,"image_index":image,"selection_order":selection,"source_object_id":source,"condition_id":condition,"block":b,**h,"normalized_ablation_loss":losses.get(h["head"])})
 return rows,heads

def image_aggregates(clean_cache,non_cache):
 rows=[]
 for order,(arrays,pr,hr,science,inter) in sorted(clean_cache.items()):
  by={r["block"]:r for r in pr};deep=lambda k:float(np.mean([by[b][k] for b in y.DEPTH["deep"]],dtype=np.float64));shallow=lambda k:float(np.mean([by[b][k] for b in y.DEPTH["shallow"]],dtype=np.float64));rows.append({"image_index":order,"selection_order":order,"source_object_id":pr[0]["source_object_id"],"A1":inter["A1"],"B1":deep("log_effective_spatial_support")-shallow("log_effective_spatial_support"),"B2":deep("effective_head_count")-shallow("effective_head_count"),"B3":deep("off_diagonal_absolute_mass_fraction")-shallow("off_diagonal_absolute_mass_fraction"),"B4":deep("first_attention_share")-shallow("first_attention_share"),"B5":deep("second_mlp_share")-shallow("second_mlp_share"),"B6":inter["B6"],"B7":inter["B7"],"C1":None,"C2":None,"C3":None,"C4":None})
 perturb=[];selections=sorted({v[4]["selection_order"] for v in non_cache.values()})
 for image_idx,selection in enumerate(selections):
  clean_arrays,clean_pr,clean_hr,_,_=clean_cache[selection];cp={r["block"]:r for r in clean_pr};ch={(r["block"],r["head"]):r for r in clean_hr};condition_rows=[];units=[v for v in non_cache.values() if v[4]["selection_order"]==selection];y.require(len(units)==8,"incomplete perturbation image")
  for arrays,pr,hr,science,q in sorted(units,key=lambda v:v[4]["condition_index"]):
   tp={r["block"]:r for r in pr};th={(r["block"],r["head"]):r for r in hr};w=arrays["patch_inverse_operator"].astype(np.float64);valid=arrays["fully_valid_patch_mask"].astype(bool);y.require(w.shape==(196,196) and valid.shape==(196,) and valid.any(),"inverse operator/mask invalid");pred=science["transformed_prediction"]
   for b in y.BLOCKS:
    ca=np.asarray([ch[(b,h)]["absolute_share"] for h in y.HEADS]);ta=np.asarray([th[(b,h)]["absolute_share"] for h in y.HEADS]);name=f"b{b:02d}_first_attention_matrix";inverse=w@arrays[name].astype(np.float64)@w.T;clean_t=clean_arrays[name].astype(np.float64);pairs=np.outer(valid,valid);condition_rows.append({"image_index":image_idx,"selection_order":selection,"source_object_id":q["source_object_id"],"condition_id":q["condition_id"],"block":b,"within_patch_ntv_absolute_change":abs(tp[b]["within_patch_normalized_total_variation"]-cp[b]["within_patch_normalized_total_variation"]),"one_minus_head_share_cosine":1-y.cosine_abs(ca,ta),"head_share_spearman":y.spearman(ca,ta),"effective_head_count_change":tp[b]["effective_head_count"]-cp[b]["effective_head_count"],"one_minus_inverse_patch_matrix_cosine":1-y.cosine_abs(clean_t[pairs],inverse[pairs]),"first_branch_share_absolute_change":abs(tp[b]["first_attention_share"]-cp[b]["first_attention_share"]),"second_branch_share_absolute_change":abs(tp[b]["second_mlp_share"]-cp[b]["second_mlp_share"]),"mean_branch_share_absolute_change":float(np.mean([abs(tp[b]["first_attention_share"]-cp[b]["first_attention_share"]),abs(tp[b]["second_mlp_share"]-cp[b]["second_mlp_share"])])),"valid_domain_coverage":float(arrays["valid_mask"].mean()),"prediction_changed":pred["prediction_changed"],"full_domain_flip_control":q["condition_id"]=="horizontal_flip" and bool(arrays["valid_mask"].all())})
  perturb.extend(condition_rows);target=next(r for r in rows if r["selection_order"]==selection);target["C1"]=float(np.mean([r["within_patch_ntv_absolute_change"] for r in condition_rows]));target["C2"]=float(np.mean([r["one_minus_head_share_cosine"] for r in condition_rows]));target["C3"]=float(np.mean([r["one_minus_inverse_patch_matrix_cosine"] for r in condition_rows]));target["C4"]=float(np.mean([r["mean_branch_share_absolute_change"] for r in condition_rows]))
 return rows,perturb

_v.y=y;_v.primitive=primitive;_v.image_aggregates=image_aggregates
produce=_v.produce;seal=_v.seal;validate_completion=_v.validate_completion;derive=_v.derive;false_pass_audit=_v.false_pass_audit
if __name__=="__main__":os.environ.setdefault("CUDA_VISIBLE_DEVICES","");raise SystemExit(_v.main())
