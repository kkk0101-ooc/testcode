"""Prospective Y4-R2 control-plane repair over the frozen Y4-R1 engine."""
from __future__ import annotations
from pathlib import Path
from typing import Any,Mapping
from . import stage_y_y4_r1 as _b

STAGE_ID="Y4-R2";STARTUP_COMMIT="4cbcf763f81455b09f09d7e489161225f62c8237"
ROOT=Path("evidence/stage_y/y4_r2_mechanism_analysis_20260914/formal")
REPORT_ROOT=Path("reports/stage_y_y4_r2_20260914");REPORT=REPORT_ROOT/"report.md"
DOC=Path("docs/stage_y_y4_r2_control_plane_recovery_20260914.md")
SPEC=Path("config/stage_y/y4_r2_run_spec.json");SPEC_SCHEMA=Path("config/stage_y/y4_r2_run_spec.schema.json");ARTIFACT_SCHEMA=Path("config/stage_y/y4_r2_artifact.schema.json")
PRODUCER_REF="refs/archive/stage-y-y4-r2-mechanism-analysis-producer";PASS_REF="refs/archive/stage-y-y4-r2-mechanism-analysis-and-decision";BLOCKED_REF="refs/archive/stage-y-y4-r2-mechanism-analysis-and-decision-blocked"
R1_PRODUCER_REF="refs/archive/stage-y-y4-r1-mechanism-analysis-producer"
PRODUCER_FILES=(DOC,SPEC,SPEC_SCHEMA,ARTIFACT_SCHEMA,Path("src/vitnns/stage_y_y4_r2.py"),Path("scripts/run_stage_y_y4_r2.py"),Path("scripts/validate_stage_y_y4_r2.py"),Path("tests/test_stage_y_y4_r2.py"))
DEPENDENCIES=_b.PRODUCER_FILES+_b.CONTRACT_FILES+_b.DEPENDENCIES

def activate()->None:
    for name in ("STAGE_ID","STARTUP_COMMIT","ROOT","REPORT_ROOT","REPORT","DOC","SPEC","SPEC_SCHEMA","ARTIFACT_SCHEMA","PRODUCER_REF","PASS_REF","BLOCKED_REF","PRODUCER_FILES","DEPENDENCIES"):_b.__dict__[name]=globals()[name]
    _b.EXPECTED_REFS[R1_PRODUCER_REF]=STARTUP_COMMIT
activate()

def seal_producer(repo:Path,preflight:Mapping[str,Any])->dict[str,Any]:
    _b.runtime_guard();head=_b.git(repo,"rev-parse","HEAD");rem=_b.remote_refs(repo);_b.require(head==_b.git(repo,"rev-parse","@{upstream}")==rem.get("refs/heads/"+_b.BRANCH)==_b.git(repo,"rev-parse",PRODUCER_REF)==rem.get(PRODUCER_REF),"producer publication incomplete",_b.StartupError);_b.require(not _b.git(repo,"status","--porcelain"),"producer worktree dirty",_b.StartupError);_b.validate_schema_sources(repo);history=_b.verify_expected_refs(repo);historical=[r for r in history if r["ref"]!=PRODUCER_REF];_b.require(_b.canonical(historical)==preflight["startup_refs_digest"],"historical ref snapshot changed",_b.StartupError)
    root=repo/ROOT;_b.require(not root.exists(),"Y4-R2 root exists",_b.StartupError);_b.write(root/"completion/y4_r1.pending.json","pending",{"started_unix_ns":__import__("time").time_ns(),"next_authorization":"NONE_WHILE_PENDING"},"PENDING");files=_b.producer_files(repo);_b.write(root/"producer_implementation_seal.json","implementation_seal",{"producer_commit":head,"producer_ref":PRODUCER_REF,"source_files":files,"files_digest":_b.canonical({"files":files})});_b.write(root/"producer_preflight.json","producer_preflight",preflight);_b.write(root/"startup_ref_snapshot.json","binding",{"name":"startup_stage_y_refs","files":[],"files_digest":preflight["startup_refs_digest"],"commit":STARTUP_COMMIT});contracts=_b.contract_bindings(repo);_b.write(root/"y0_contract_bundle_binding.json","binding",{"name":"Y0 contract bundle","files":contracts,"files_digest":_b.canonical(contracts),"commit":_b.EXPECTED_REFS["refs/archive/stage-y-y0-protocol-freeze"]});_b.write(root/"y2_completion_publication_binding.json","binding",_b.publication_binding(repo,"Y2"));_b.write(root/"y3_completion_publication_binding.json","binding",_b.publication_binding(repo,"Y3"));protocol=[_b.file_record(repo/DOC,repo),_b.file_record(repo/SPEC,repo)];_b.write(root/"prospective_protocol_binding.json","binding",{"name":"Y4-R2 prospective control-plane recovery","files":protocol,"files_digest":_b.canonical(protocol),"commit":head});_b.verify_implementation(repo);return {"status":"PASS","producer_commit":head}

for _name in dir(_b):
    if not _name.startswith("_") and _name not in globals():globals()[_name]=getattr(_b,_name)
