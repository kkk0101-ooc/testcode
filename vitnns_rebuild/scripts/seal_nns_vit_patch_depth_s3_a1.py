#!/usr/bin/env python3
"""Seal the prospective S3-R3-A1 atomic terminal publication repair."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa:E402


def main()->int:
    output=REPO/s3.ROOT_REL/"s3_r3_a1_implementation_seal.json"
    s3.require(not output.exists(),"A1 implementation seal already exists")
    status=subprocess.check_output(["git","status","--porcelain=v2","--untracked-files=all"],cwd=REPO,text=True)
    s3.require(not [line for line in status.splitlines() if "../.root_cleanup_quarantine/" not in line],"A1 seal requires clean tree")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    files=["src/vitnns/nns_patch_depth_s3_worker.py","docs/nns_vit_patch_depth_s3_r3_a1_amendment_20260914.md","tests/test_nns_patch_depth_s3_terminal_atomic.py","scripts/seal_nns_vit_patch_depth_s3_a1.py"]
    records=[]
    for name in files:
        path=REPO/name;blob=subprocess.check_output(["git","show",f"{commit}:vitnns_rebuild/{name}"],cwd=REPO);sha=s3.sha_file(path)
        s3.require(s3.sha_bytes(blob)==sha,f"A1 Git-tree identity mismatch: {name}")
        records.append({"path":name,"bytes":path.stat().st_size,"sha256":sha,"git_tree_commit":commit})
    document=s3.signed("s3_a1_implementation_seal","PASS",{"seal_commit":commit,"files":records,"files_digest":s3.records_digest(records),"repair":"atomic hard-link publication of fully fsynced unit terminal temporary file","scientific_contract_changed":False,"schedule_changed":False,"access_order_changed":False,"applies_to":"fresh launches after train_validation-launch-001 exits","launch_001_completed_terminals_preserved":True})
    s3.write_json_new(output,document);print(json.dumps({"status":"PASS","seal_commit":commit,"files_digest":document["payload"]["files_digest"]},sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
