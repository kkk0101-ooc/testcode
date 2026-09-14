#!/usr/bin/env python3
"""S3 pilot-only feature repeatability probe; never reads a formal split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa:E402
from vitnns import nns_patch_depth_s3_worker as worker  # noqa:E402


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--physical-index",type=int,required=True);parser.add_argument("--gpu-uuid",required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    schedule=json.loads((REPO/"manifests/nns_vit_patch_depth/s1_pilot_schedule.json").read_text())["payload"]["records"]
    source=schedule[0]
    row={"split":"pilot","image_local_path":source["image_path"],"image_bytes":source["image_bytes"],"image_sha256":source["image_sha256"],"annotation_archive_path":source["annotation_archive_path"],"annotation_sha256":source["annotation_sha256"],"image_width":source["image_size"][1],"image_height":source["image_size"][0],"mask_width":source["mask_size"][1],"mask_height":source["mask_size"][0],"annotation_label_id":source["target_annotation_label_id"],"source_image_id":source["source_image_id"]}
    model,captured,identity,handles=worker.initialize(REPO,args.physical_index,args.gpu_uuid)
    with zipfile.ZipFile(REPO/worker.MASK_ZIP) as archive:
        raw_image=(REPO/source["image_path"]).read_bytes();raw_mask=archive.read(source["annotation_archive_path"])
    image,labels,rgb,geometry=worker.preprocess(raw_image,raw_mask,row)
    occupancy=s3.patch_occupancy(labels,row["annotation_label_id"]);control=s3.low_level_features(rgb);features,resources=worker.capture_features(model,captured,image)
    for handle in handles:handle.remove()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,features=features,occupancy=occupancy,control=control)
    report={"source_image_id":row["source_image_id"],"permanently_pilot_only":True,"formal_split_access_count":0,"identity":identity,"geometry":geometry,"resources":resources,"feature_array_digest":s3.array_digest(features),"occupancy_array_digest":s3.array_digest(occupancy),"control_array_digest":s3.array_digest(control),"output_sha256":s3.sha_file(args.output)}
    print(json.dumps(report,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
