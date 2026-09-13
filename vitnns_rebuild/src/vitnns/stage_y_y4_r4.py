"""Prospective Y4-R4 missing-value and independent-validation recovery."""
import os
from pathlib import Path
from . import stage_y_y4_r1 as _b
STAGE_ID="Y4-R4";STARTUP_COMMIT="0c2b7ff116e751d53021017f04d6a446d7272aed"
ROOT=Path("evidence/stage_y/y4_r4_mechanism_analysis_20260914/formal");REPORT_ROOT=Path("reports/stage_y_y4_r4_20260914");REPORT=REPORT_ROOT/"report.md"
DOC=Path("docs/stage_y_y4_r4_zero_mass_na_recovery_20260914.md");SPEC=Path("config/stage_y/y4_r4_run_spec.json");SPEC_SCHEMA=Path("config/stage_y/y4_r4_run_spec.schema.json");ARTIFACT_SCHEMA=Path("config/stage_y/y4_r4_artifact.schema.json")
PRODUCER_REF="refs/archive/stage-y-y4-r4-mechanism-analysis-producer";PASS_REF="refs/archive/stage-y-y4-mechanism-analysis-and-decision";BLOCKED_REF="refs/archive/stage-y-y4-r4-mechanism-analysis-and-decision-blocked"
PRODUCER_FILES=(DOC,SPEC,SPEC_SCHEMA,ARTIFACT_SCHEMA,Path("src/vitnns/stage_y_y4_r4.py"),Path("scripts/run_stage_y_y4_r4.py"),Path("scripts/validate_stage_y_y4_r4.py"),Path("tests/test_stage_y_y4_r4.py"));DEPENDENCIES=(Path("src/vitnns/stage_y_y4_r1.py"),Path("scripts/validate_stage_y_y4_r1.py"),Path("scripts/validate_stage_y_y4_r3.py"),)+_b.CONTRACT_FILES+_b.DEPENDENCIES
for _n in ("STAGE_ID","STARTUP_COMMIT","ROOT","REPORT_ROOT","REPORT","DOC","SPEC","SPEC_SCHEMA","ARTIFACT_SCHEMA","PRODUCER_REF","PASS_REF","BLOCKED_REF","PRODUCER_FILES","DEPENDENCIES"):_b.__dict__[_n]=globals()[_n]
_b.EXPECTED_REFS.update({"refs/archive/stage-y-y4-r1-mechanism-analysis-producer":"4cbcf763f81455b09f09d7e489161225f62c8237","refs/archive/stage-y-y4-r2-mechanism-analysis-producer":"6ac4add6d806a1ad53948433561a1f9f7ec65757","refs/archive/stage-y-y4-r2-mechanism-analysis-and-decision-blocked":"823e5d694184b9d26e8d96c735a071ef267a741c","refs/archive/stage-y-y4-r3-mechanism-analysis-producer":"154ac899f42596dc9169e574a8520707b2f9c338","refs/archive/stage-y-y4-r3-mechanism-analysis-and-decision-blocked":STARTUP_COMMIT})
for _name in dir(_b):
    if not _name.startswith("_") and _name not in globals():globals()[_name]=getattr(_b,_name)
_inherited_artifact_schema=_b.artifact_schema
def artifact_schema():
    schema=_inherited_artifact_schema();schema["$defs"]["cumulative_resource"]=obj({"prior_stage_ledger":FILE,"y3_stage_ledger":FILE,"y4_ledger":FILE,"y0_y2_cpu_seconds_lower_bound":NN,"y0_y2_gpu_seconds":NN,"y3_cpu_seconds":NN,"y3_gpu_seconds":NN,"y4_cpu_seconds":NN,"y4_gpu_seconds":{"const":0},"cumulative_cpu_seconds_lower_bound":NN,"cumulative_gpu_seconds":NN,"cumulative_h20_gpu_hours":NN,"note":S});return schema
_b.artifact_schema=artifact_schema;_b._SCHEMA=None
_inherited_startup=startup
def _ancestor_pids():
    out=set();pid=os.getpid()
    while pid>1 and pid not in out:
        out.add(pid)
        try:
            fields=Path(f"/proc/{pid}/stat").read_text().split();pid=int(fields[3])
        except (OSError,ValueError,IndexError):break
    return out
def _check_no_stage_workers():
    ancestors=_ancestor_pids();bad=[]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:continue
        try:cmd=(entry/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
        except (OSError,PermissionError):continue
        if any(token in cmd for token in ("stage_y_y2_worker","stage_y_y3_worker","run_stage_y_y4","validate_stage_y_y4","stage_y_y5")):bad.append({"pid":int(entry.name),"command":cmd[:300]})
    require(not bad,f"active Stage Y worker/analysis process: {bad}",StartupError)
def startup(repo,allow_sources=False):
    result=_inherited_startup(repo,allow_sources);_check_no_stage_workers();return result
def seal_producer(repo,preflight):
    runtime_guard();head=git(repo,"rev-parse","HEAD");rem=remote_refs(repo);require(head==git(repo,"rev-parse","@{upstream}")==rem.get("refs/heads/"+BRANCH)==git(repo,"rev-parse",PRODUCER_REF)==rem.get(PRODUCER_REF),"producer publication incomplete",StartupError);require(not git(repo,"status","--porcelain"),"producer worktree dirty",StartupError);validate_schema_sources(repo);history=verify_expected_refs(repo);require(canonical([r for r in history if r["ref"]!=PRODUCER_REF])==preflight["startup_refs_digest"],"historical ref snapshot changed",StartupError);root=repo/ROOT;require(not root.exists(),"Y4-R4 root exists",StartupError);write(root/"completion/y4_r4.pending.json","pending",{"started_unix_ns":__import__("time").time_ns(),"next_authorization":"NONE_WHILE_PENDING"},"PENDING");files=producer_files(repo);write(root/"producer_implementation_seal.json","implementation_seal",{"producer_commit":head,"producer_ref":PRODUCER_REF,"source_files":files,"files_digest":canonical({"files":files})});write(root/"producer_preflight.json","producer_preflight",preflight);write(root/"startup_ref_snapshot.json","binding",{"name":"startup_stage_y_refs","files":[],"files_digest":preflight["startup_refs_digest"],"commit":STARTUP_COMMIT});contracts=contract_bindings(repo);write(root/"y0_contract_bundle_binding.json","binding",{"name":"Y0 contract bundle","files":contracts,"files_digest":canonical(contracts),"commit":EXPECTED_REFS["refs/archive/stage-y-y0-protocol-freeze"]});write(root/"y2_completion_publication_binding.json","binding",publication_binding(repo,"Y2"));write(root/"y3_completion_publication_binding.json","binding",publication_binding(repo,"Y3"));protocol=[file_record(repo/DOC,repo),file_record(repo/SPEC,repo)];write(root/"prospective_protocol_binding.json","binding",{"name":"Y4-R4 zero-mass NA recovery","files":protocol,"files_digest":canonical(protocol),"commit":head});verify_implementation(repo);return {"status":"PASS","producer_commit":head}
