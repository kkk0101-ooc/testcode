#!/usr/bin/env python3
"""Frozen-environment NumPy/Torch numeric validator for S3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa:E402


def require(condition:bool,message:str)->None:
    if not condition: raise ValueError(message)


def arrays(phase:str,root:Path|None=None)->dict[str,Any]:
    if root is None: root=REPO/s3.ROOT_REL
    schedule=json.loads((REPO/s3.SCHEDULE_REL).read_text())["payload"]["records"]
    allowed={"probe_train","probe_validation"} if phase=="train_validation" else {"d2_test"}
    checked=0
    for row in schedule:
        if row["split"] not in allowed: continue
        terminal=json.loads((root/"unit_terminals"/f"unit-{int(row['schedule_index']):05d}.json").read_text())
        if terminal["payload"]["terminal_state"]!="VALID_FEATURE": continue
        expected={"block_features.npy":((12,196,768),np.dtype("float32")),"occupancy.npy":((4,196),np.dtype("float32")),"low_level_features.npy":((196,5),np.dtype("float32"))}
        if phase=="test": expected.update({"feature_prediction.npy":((12,196),np.dtype("float32")),"control_prediction.npy":((196,),np.dtype("float32")),"feature_mse.npy":((12,),np.dtype("float64")),"control_mse.npy":((),np.dtype("float64"))})
        by_name={Path(record["path"]).name:record for record in terminal["payload"]["arrays"]}
        corrected=[record for record in terminal["payload"]["arrays"] if "corrected_control_mse" in record["path"] and record["axes"]==[] and record["dtype"]=="float64" and record["shape"]==[]]
        if corrected:
            require("control_mse.npy" not in by_name and len(corrected)==1,"ambiguous corrected control MSE")
            by_name.pop(Path(corrected[0]["path"]).name)
            by_name["control_mse.npy"]=corrected[0]
        require(set(by_name)==set(expected),"array inventory name drift")
        loaded={}
        for name,(shape,dtype) in expected.items():
            record=by_name[name];path=REPO/record["path"]
            require(s3.sha_file(path)==record["sha256"],"array SHA drift")
            value=np.load(path,allow_pickle=False);loaded[name]=value
            require(value.shape==shape and value.dtype==dtype and np.isfinite(value).all(),"array numeric contract failed")
            require(s3.array_digest(value)==record["canonical_array_digest"],"canonical array digest drift")
        if phase=="test":
            target=loaded["occupancy.npy"][3].astype(np.float64)
            feature=loaded["feature_prediction.npy"].astype(np.float64)
            control=loaded["control_prediction.npy"].astype(np.float64)
            recomputed=np.mean((feature-target[None,:])**2,axis=1,dtype=np.float64)
            require(np.allclose(recomputed,loaded["feature_mse.npy"],rtol=1e-6,atol=1e-9),"test feature MSE drift")
            require(np.allclose(np.mean((control-target)**2,dtype=np.float64),loaded["control_mse.npy"],rtol=1e-6,atol=1e-9),"test control MSE drift")
            wnid=str(row["annotation_wnid"]);weight_path=REPO/s3.PAYLOAD_REL/"weights"/f"{wnid}.npz"
            with np.load(weight_path,allow_pickle=False) as bundle:
                predicted=np.empty((12,196),dtype=np.float64)
                for block in range(12):
                    model=s3.RidgeModel(bundle["feature_mean"][block],bundle["feature_scale"][block],bundle["feature_weight"][block],float(bundle["feature_intercept"][block]),float(bundle["feature_alpha"][block]));predicted[block]=s3.predict_ridge(model,loaded["block_features.npy"][block])
                control_model=s3.RidgeModel(bundle["control_mean"],bundle["control_scale"],bundle["control_weight"],float(bundle["control_intercept"]),float(bundle["control_alpha"]));control_pred=s3.predict_ridge(control_model,loaded["low_level_features.npy"])
            require(np.array_equal(predicted.astype("<f4"),loaded["feature_prediction.npy"]),"test feature prediction recomputation drift")
            require(np.array_equal(control_pred.astype("<f4"),loaded["control_prediction.npy"]),"test control prediction recomputation drift")
        checked+=1
    return {"status":"PASS","phase":phase,"valid_array_units_checked":checked}


def weights()->dict[str,Any]:
    manifests=["alpha_grid_validation_ledger.json","normalization_manifest.json","class_block_probe_weight_manifest.json","low_level_control_weight_manifest.json","train_only_crossfit_report.json"]
    for name in manifests: require((REPO/s3.ROOT_REL/name).is_file(),f"missing {name}")
    weight_manifest=json.loads((REPO/s3.ROOT_REL/"class_block_probe_weight_manifest.json").read_text())["payload"]
    require(weight_manifest["valid_class_count"]==859 and weight_manifest["typed_na_class_count"]==13 and len(weight_manifest["records"])==859,"weight class count drift")
    alpha_payload=json.loads((REPO/s3.ROOT_REL/"alpha_grid_validation_ledger.json").read_text())["payload"];alpha_by_class={row["annotation_wnid"]:row for row in alpha_payload["records"]}
    schedule=json.loads((REPO/s3.SCHEDULE_REL).read_text())["payload"]["records"];rows_by_class={}
    for row in schedule:
        if row["split"] in ("probe_train","probe_validation"):rows_by_class.setdefault(row["annotation_wnid"],[]).append(row)
    recomputed=0
    for record in weight_manifest["records"]:
        path=REPO/record["path"];require(s3.sha_file(path)==record["sha256"],"weight bundle SHA drift")
        with np.load(path,allow_pickle=False) as bundle:
            expected={"feature_mean":(12,768),"feature_scale":(12,768),"feature_weight":(12,768),"feature_intercept":(12,),"feature_alpha":(12,),"control_mean":(5,),"control_scale":(5,),"control_weight":(5,),"control_intercept":(),"control_alpha":()}
            require(set(bundle.files)==set(expected),"weight bundle fields drift")
            for key,shape in expected.items(): require(bundle[key].shape==shape and bundle[key].dtype==np.float64 and np.isfinite(bundle[key]).all(),"weight numeric contract failed")
            require(set(bundle["feature_alpha"].tolist()).issubset(set(s3.ALPHAS)) and float(bundle["control_alpha"]) in s3.ALPHAS,"selected alpha outside grid")
            require(np.all(bundle["feature_scale"]>0) and np.all(bundle["control_scale"]>0),"normalization scale not positive")
            wnid=path.stem;rows=rows_by_class[wnid];train=[row for row in rows if row["split"]=="probe_train"];validation=[row for row in rows if row["split"]=="probe_validation"]
            def load(row):
                unit=REPO/s3.PAYLOAD_REL/"units"/f"unit-{int(row['schedule_index']):05d}";return np.load(unit/"block_features.npy",allow_pickle=False),np.load(unit/"occupancy.npy",allow_pickle=False)[3].astype(np.float64),np.load(unit/"low_level_features.npy",allow_pickle=False)
            tr=[load(row) for row in train];va=[load(row) for row in validation];ledger=alpha_by_class[wnid]
            for block in range(12):
                x=np.concatenate([item[0][block] for item in tr]);y=np.concatenate([item[1] for item in tr]);decomp=s3.ridge_decompose(x,y);require(np.allclose(decomp.mean,bundle["feature_mean"][block],rtol=1e-10,atol=1e-10) and np.allclose(decomp.scale,bundle["feature_scale"][block],rtol=1e-10,atol=1e-10),"feature normalization recompute drift")
                values={}
                for alpha in s3.ALPHAS:
                    model=s3.model_from_decomposition(decomp,alpha);values[alpha]=s3.equal_image_mse([s3.predict_ridge(model,item[0][block]) for item in va],[item[1] for item in va])
                selected=s3.select_alpha(values);require(selected==float(bundle["feature_alpha"][block])==float(ledger["feature"][block]["selected_alpha"]),"feature alpha selection drift");model=s3.model_from_decomposition(decomp,selected);require(np.allclose(model.weight,bundle["feature_weight"][block],rtol=1e-10,atol=1e-10) and abs(model.intercept-float(bundle["feature_intercept"][block]))<=1e-10,"feature ridge fit drift")
            x=np.concatenate([item[2] for item in tr]);y=np.concatenate([item[1] for item in tr]);decomp=s3.ridge_decompose(x,y);values={alpha:s3.equal_image_mse([s3.predict_ridge(s3.model_from_decomposition(decomp,alpha),item[2]) for item in va],[item[1] for item in va]) for alpha in s3.ALPHAS};selected=s3.select_alpha(values);model=s3.model_from_decomposition(decomp,selected)
            require(selected==float(bundle["control_alpha"])==float(ledger["selected_control_alpha"]),"control alpha selection drift");require(np.allclose(model.mean,bundle["control_mean"],rtol=1e-10,atol=1e-10) and np.allclose(model.scale,bundle["control_scale"],rtol=1e-10,atol=1e-10) and np.allclose(model.weight,bundle["control_weight"],rtol=1e-10,atol=1e-10),"control normalization or fit drift");recomputed+=1
    return {"status":"PASS","weight_class_count":859,"typed_na_class_count":13,"full_class_numeric_recompute_count":recomputed,"ridge_recompute_tolerance":{"atol":1e-10,"rtol":1e-10}}


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("train-validation","weights","test","final"));args=parser.parse_args()
    if args.mode=="weights":result=weights()
    elif args.mode=="final":result={"status":"PASS","train_validation":arrays("train_validation",REPO/"evidence/nns_vit_patch_depth/s3_r5_20260915"),"test":arrays("test",REPO/"evidence/nns_vit_patch_depth/s3_r5_20260915"),"weights":weights()}
    else:result=arrays("train_validation" if args.mode=="train-validation" else "test")
    print(json.dumps(result,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
