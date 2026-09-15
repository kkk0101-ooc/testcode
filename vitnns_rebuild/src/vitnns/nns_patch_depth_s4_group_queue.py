"""Look-scoped dynamic queue with immutable S4-R4 rebind support."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from . import nns_patch_depth_s4 as s4
from . import nns_patch_depth_s4_group_sequential as gs


def new_terminal_path(root: Path, unit_type: str, schedule_index: int) -> Path:
    return root / "unit_terminals" / unit_type.lower() / f"unit-{schedule_index:04d}.json"


def rebind_path(root: Path, unit_type: str, schedule_index: int) -> Path:
    return root / "rebind_terminals" / unit_type.lower() / f"unit-{schedule_index:04d}.json"


def resolved_terminal(root: Path, unit_type: str, schedule_index: int) -> tuple[Path, dict[str, Any], str] | None:
    new = new_terminal_path(root, unit_type, schedule_index)
    if new.is_file():
        value=json.loads(new.read_text()); gs.require(gs.verify_signed(value), "new terminal canonical digest drift")
        return new,value,"NEW"
    binding=rebind_path(root,unit_type,schedule_index)
    if not binding.is_file(): return None
    record=json.loads(binding.read_text()); gs.require(gs.verify_signed(record), "rebind record canonical digest drift")
    source=Path(record["payload"]["source_terminal_path"]); source=source if source.is_absolute() else root.parents[2]/source
    value=json.loads(source.read_text()); gs.require(s4.verify_signed(value), "rebound source terminal canonical digest drift")
    gs.require(s4.sha_file(source)==record["payload"]["source_terminal_sha256"], "rebound source terminal SHA drift")
    return source,value,"REBIND"


def claim_next(launch_root: Path, root: Path, schedule: Sequence[Mapping[str, Any]], look: int, worker: Mapping[str, Any]) -> dict[str, Any] | None:
    authorized=[row for row in schedule if int(row["first_authorized_look"])<=look]
    for row in authorized:
        for unit_type in s4.UNIT_TYPES:
            index=int(row["schedule_index"])
            if resolved_terminal(root,unit_type,index) is not None: continue
            if unit_type!="NNS_IMAGE" and resolved_terminal(root,"NNS_IMAGE",index) is None: continue
            claim_path=launch_root/"claims"/unit_type.lower()/f"unit-{index:04d}.claim.json"
            payload={"launch_id":launch_root.name,"look":look,"unit_type":unit_type,"schedule_index":index,"source_id":row["source_image_id"],"formal_class_rank":int(row["formal_class_rank"]),"first_authorized_look":int(row["first_authorized_look"]),"operator":None if unit_type=="NNS_IMAGE" else (s4.OPERATORS[0] if unit_type=="OCCLUSION_PRIMARY" else s4.OPERATORS[1]),"component":"first_residual_addition_all_head_aggregate_attention_branch" if unit_type=="NNS_IMAGE" else None,"worker_id":worker["worker_id"],"worker_pid":int(worker["worker_pid"]),"physical_gpu_index":int(worker["physical_gpu_index"]),"gpu_uuid":worker["gpu_uuid"],"claimed_unix_ns":time.time_ns(),"claim_scope":"THIS_LAUNCH_AND_LOOK_ONLY","future_tranche_claim":False,"scientific_seed_depends_on_worker_gpu_claim_launch_or_time":False}
            claim=gs.signed("s4_group_sequential_unit_claim","CLAIMED",payload)
            claim_path.parent.mkdir(parents=True,exist_ok=True)
            try: fd=os.open(claim_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
            except FileExistsError: continue
            with os.fdopen(fd,"wb") as stream:
                stream.write(json.dumps(claim,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n"); stream.flush(); os.fsync(stream.fileno())
            gs.fsync_directory(claim_path.parent)
            return {**payload,"claim_path":claim_path.as_posix(),"claim_canonical_digest":claim["canonical_digest"]}
    return None


def missing_units(root: Path, schedule: Sequence[Mapping[str, Any]], look: int) -> list[tuple[str,int]]:
    return [(unit_type,int(row["schedule_index"])) for row in schedule if int(row["first_authorized_look"])<=look for unit_type in s4.UNIT_TYPES if resolved_terminal(root,unit_type,int(row["schedule_index"])) is None]


def terminal_counts(root: Path, schedule: Sequence[Mapping[str, Any]], look: int) -> dict[str, Any]:
    rows=[]
    for row in schedule:
        if int(row["first_authorized_look"])>look: continue
        for unit_type in s4.UNIT_TYPES:
            resolved=resolved_terminal(root,unit_type,int(row["schedule_index"]))
            if resolved is not None: rows.append({"unit_type":unit_type,"schedule_index":int(row["schedule_index"]),"status":resolved[1]["status"],"provenance":resolved[2],"path":resolved[0].as_posix()})
    return {"planned":3*sum(int(row["first_authorized_look"])<=look for row in schedule),"terminal":len(rows),"records":rows}
