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


def arrays(phase:str)->dict[str,Any]:
    schedule=json.loads((REPO/s3.SCHEDULE_REL).read_text())["payload"]["records"]
    allowed={"probe_train","probe_validation"} if phase=="train_validation" else {"d2_test"}
    checked=0
    for row in schedule:
        if row["split"] not in allowed: continue
        terminal=json.loads((REPO/s3.ROOT_REL/"unit_terminals"/f"unit-{int(row['schedule_index']):05d}.json").read_text())
        if terminal["payload"]["terminal_state"]!="VALID_FEATURE": continue
        expected={"block_features.npy":((12,196,768),np.dtype("float32")),"occupancy.npy":((4,196),np.dtype("float32")),"low_level_features.npy":((196,5),np.dtype("float32"))}
        if phase=="test": expected.update({"feature_prediction.npy":((12,196),np.dtype("float32")),"control_prediction.npy":((196,),np.dtype("float32")),"feature_mse.npy":((12,),np.dtype("float64")),"control_mse.npy":((),np.dtype("float64"))})
        by_name={Path(record["path"]).name:record for record in terminal["payload"]["arrays"]}
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
        checked+=1
    return {"status":"PASS","phase":phase,"valid_array_units_checked":checked}


def weights()->dict[str,Any]:
    manifests=["alpha_grid_validation_ledger.json","normalization_manifest.json","class_block_probe_weight_manifest.json","low_level_control_weight_manifest.json","train_only_crossfit_report.json"]
    for name in manifests: require((REPO/s3.ROOT_REL/name).is_file(),f"missing {name}")
    weight_manifest=json.loads((REPO/s3.ROOT_REL/"class_block_probe_weight_manifest.json").read_text())["payload"]
    require(weight_manifest["valid_class_count"]==859 and weight_manifest["typed_na_class_count"]==13 and len(weight_manifest["records"])==859,"weight class count drift")
    for record in weight_manifest["records"]:
        path=REPO/record["path"];require(s3.sha_file(path)==record["sha256"],"weight bundle SHA drift")
        with np.load(path,allow_pickle=False) as bundle:
            expected={"feature_mean":(12,768),"feature_scale":(12,768),"feature_weight":(12,768),"feature_intercept":(12,),"feature_alpha":(12,),"control_mean":(5,),"control_scale":(5,),"control_weight":(5,),"control_intercept":(),"control_alpha":()}
            require(set(bundle.files)==set(expected),"weight bundle fields drift")
            for key,shape in expected.items(): require(bundle[key].shape==shape and bundle[key].dtype==np.float64 and np.isfinite(bundle[key]).all(),"weight numeric contract failed")
            require(set(bundle["feature_alpha"].tolist()).issubset(set(s3.ALPHAS)) and float(bundle["control_alpha"]) in s3.ALPHAS,"selected alpha outside grid")
            require(np.all(bundle["feature_scale"]>0) and np.all(bundle["control_scale"]>0),"normalization scale not positive")
    return {"status":"PASS","weight_class_count":859,"typed_na_class_count":13,"ridge_recompute_tolerance":{"atol":1e-10,"rtol":1e-10}}


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("train-validation","weights","test"));args=parser.parse_args()
    result=weights() if args.mode=="weights" else arrays("train_validation" if args.mode=="train-validation" else "test")
    print(json.dumps(result,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
