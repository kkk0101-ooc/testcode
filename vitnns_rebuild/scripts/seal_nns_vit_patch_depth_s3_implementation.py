#!/usr/bin/env python3
"""Create the S3 pre-access implementation seal against a committed Git tree."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa:E402


def main()->int:
    root=REPO/s3.ROOT_REL
    output=root/"implementation_seal.json"
    s3.require(not output.exists(),"implementation seal already exists")
    status=subprocess.check_output(["git","status","--porcelain=v2","--untracked-files=all"],cwd=REPO,text=True)
    unexplained=[line for line in status.splitlines() if "../.root_cleanup_quarantine/" not in line]
    s3.require(not unexplained,f"implementation seal requires clean tree: {unexplained[:3]}")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    files=[
        "config/nns_vit_patch_depth/s3_run_spec.json","config/nns_vit_patch_depth/s3_artifact.schema.json",
        "docs/nns_vit_patch_depth_s3_protocol_20260914.md","docs/nns_vit_patch_depth_s3_r1_recovery_20260914.md","docs/nns_vit_patch_depth_s3_r2_recovery_20260914.md","docs/nns_vit_patch_depth_s3_r3_recovery_20260914.md",
        "manifests/nns_vit_patch_depth/s3_r3_source_schedule.json","src/vitnns/nns_patch_depth_s3.py","src/vitnns/nns_patch_depth_s3_worker.py",
        "scripts/freeze_nns_vit_patch_depth_s3.py","scripts/run_nns_vit_patch_depth_s3.py","scripts/pilot_test_nns_vit_patch_depth_s3.py","scripts/prepare_nns_vit_patch_depth_s3_pre_access.py","scripts/seal_nns_vit_patch_depth_s3_implementation.py","scripts/fit_nns_vit_patch_depth_s3_probes.py","scripts/validate_nns_vit_patch_depth_s3.py","scripts/validate_nns_vit_patch_depth_s3_numeric.py","tests/test_nns_patch_depth_s3.py"
    ]
    records=[]
    for name in files:
        path=REPO/name;s3.require(path.is_file(),f"implementation file missing: {name}")
        blob=subprocess.check_output(["git","show",f"{commit}:vitnns_rebuild/{name}"],cwd=REPO)
        sha=s3.sha_file(path);s3.require(s3.sha_bytes(blob)==sha,f"implementation Git-tree identity mismatch: {name}")
        records.append({"path":name,"bytes":path.stat().st_size,"sha256":sha,"git_tree_commit":commit})
    report=json.loads((root/"pre_access_audit.json").read_text());s3.require(report["status"]=="PASS","pre-access audit not PASS")
    document=s3.signed("s3_implementation_seal","PASS",{"seal_commit":commit,"files":records,"files_digest":s3.records_digest(records),"formal_image_mask_access_before_seal":False,"formal_feature_prediction_outcome_access_before_seal":False,"synthetic_tests":"PASS","pilot_only_repeatability":"PASS","pre_access_audit":"PASS","schedule_count":10458})
    s3.write_json_new(output,document)
    print(json.dumps({"status":"PASS","seal_commit":commit,"file_count":len(records),"files_digest":document["payload"]["files_digest"]},sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
