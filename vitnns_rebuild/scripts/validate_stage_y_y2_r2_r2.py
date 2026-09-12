#!/usr/bin/env python3
"""Read-only CPU validation of the complete source-recomputed recovery."""
import argparse
import json
from pathlib import Path

from vitnns import stage_y_y2_r2_r2 as r


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("validate-completion","validate-publication"));p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    r.runtime_guard()
    try:result=r.validate_completion(a.repo_root.resolve(),a.command=="validate-publication")
    except Exception as error:
        print(json.dumps({"status":getattr(error,"code","BLOCKED_PROVENANCE"),"error":str(error),"next_authorization":"NONE"}));return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
