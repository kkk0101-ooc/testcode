#!/usr/bin/env python3
"""R2 recovery binding for the Endpoint revision-v3 Step-2 queue."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_endpoint_revision_v3_step2_r1 as recovery

recovery.PARENT = REPO / "reports/nns_vit_endpoint_revision_v3_step2_r1_20260921"
recovery.ROOT = REPO / "reports/nns_vit_endpoint_revision_v3_step2_r2_20260921"
recovery.PARENT_BINDINGS = {
    "acquisition_receipt.json": "1024a3f2c1c36b3a5a84aea631f1628e61fb105a888af2628e5c32e42169757e",
    "failure_primary__vit_b16__230.json": "c5771a2f9eba8daa2a91dd8016a996a5dbc59635594093d7d56fcedd67ce13ea",
    "task_manifest.json": "0fce654435af35320ffe8f8b4390db857fcfd7223c74282044565bbfeab6acf0",
    "freeze.json": "3f8aca42aae883442d15c1a9bbf44cf568cfdb83e87393aa0b606e7d7e564b19",
}
recovery.EXPECTED_PARENT_STATES = {"completed": 632, "failed": 1, "pending": 367}
recovery.EXPECTED_INHERITED = 632
recovery.SOURCE_FILES = (*recovery.SOURCE_FILES,
    "docs/Endpoint修改意见v3第二步R2恢复修正_20260921.md",
    "scripts/run_endpoint_revision_v3_step2_r2.py",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    recovery.prepare() if args.prepare else recovery.run()
