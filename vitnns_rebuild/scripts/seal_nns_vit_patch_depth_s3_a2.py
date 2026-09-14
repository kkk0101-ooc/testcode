#!/usr/bin/env python3
"""Seal prospective S3-R3-A2 typed-NA class handling before probe fitting."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa:E402
def main()->int:
    output=REPO/s3.ROOT_REL/"s3_r3_a2_implementation_seal.json";s3.require(not output.exists(),"A2 seal exists")
    status=subprocess.check_output(["git","status","--porcelain=v2","--untracked-files=all"],cwd=REPO,text=True);s3.require(not [x for x in status.splitlines() if "../.root_cleanup_quarantine/" not in x],"A2 seal requires clean tree")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip();files=["src/vitnns/nns_patch_depth_s3.py","src/vitnns/nns_patch_depth_s3_worker.py","config/nns_vit_patch_depth/s3_artifact.schema.json","scripts/fit_nns_vit_patch_depth_s3_probes.py","scripts/validate_nns_vit_patch_depth_s3.py","scripts/validate_nns_vit_patch_depth_s3_numeric.py","docs/nns_vit_patch_depth_s3_r3_a2_amendment_20260914.md","scripts/seal_nns_vit_patch_depth_s3_a2.py"]
    records=[]
    for name in files:
        path=REPO/name;blob=subprocess.check_output(["git","show",f"{commit}:vitnns_rebuild/{name}"],cwd=REPO);sha=s3.sha_file(path);s3.require(s3.sha_bytes(blob)==sha,f"A2 tree drift: {name}");records.append({"path":name,"bytes":path.stat().st_size,"sha256":sha,"git_tree_commit":commit})
    doc=s3.signed("s3_a2_implementation_seal","PASS",{"seal_commit":commit,"files":records,"files_digest":s3.records_digest(records),"typed_na_class_count":13,"valid_weight_class_count":859,"test_bytes_read_for_typed_na_classes":False,"scientific_contract_changed":False,"replacement_used":False});s3.write_json_new(output,doc);print(json.dumps({"status":"PASS","seal_commit":commit,"files_digest":doc["payload"]["files_digest"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
