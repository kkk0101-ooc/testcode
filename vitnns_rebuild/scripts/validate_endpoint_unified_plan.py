#!/usr/bin/env python3
"""Validate the single current Endpoint Scanner planning entry."""
from __future__ import annotations
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/"docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md"
def require(c,m):
    if not c:raise ValueError(m)
def main():
    text=DOC.read_text();required=["ENDPOINT_PRIMARY_TRACK_AUTHORIZED_BY_AMENDMENT","PATH_DEPENDENT_TRACK_BLOCKED","EP2_R1_PATH_NUMERICAL_CALIBRATION_FAILED_IMMUTABLE","ORIGINAL_EP3_NOT_AUTHORIZED","ENDPOINT-E2_PASS_REGISTERED_ATLAS_AND_RESOURCES","ENDPOINT-E3_PASS_CORRECTNESS_ONLY","ENDPOINT-E4_FAIL_LOCAL_DIRECTION_CONTRACT","ENDPOINT-E4-R2_FAIL_ANALYTIC_JVP_CONTRACT","ENDPOINT-E4-R3_PASS_EXECUTION_CONDITION_SCALED_JVP_AND_FINITE_REPLACEMENT","ENDPOINT-E5_PASS_SPATIAL_INTERVENTION_COMPLETE_NO_CLAIM_DECISION","ENDPOINT-E6-R3_PASS_STATISTICS_AND_PAPER_RELEASE_COMPLETE","NONE_ENDPOINT_PRIMARY_TRACK_COMPLETE","L_s^{\\mathrm{Endpoint}}(x)=x\\odot\\nabla_xs(x)","ENDPOINT-E0","ENDPOINT-E1","ENDPOINT-E2","ENDPOINT-E3","ENDPOINT-E4","ENDPOINT-E5","ENDPOINT-E6","P-IG1","P-IG2","P-IG3","Gradient × Input","n_valid_min=360","28 个 Holm tests","64 GPU-hours"]
    for value in required:require(value in text,"MISSING:"+value)
    forbidden=["EP2_R1_PASS","ORIGINAL_EP3_AUTHORIZED","ENDPOINT-E2_NOT_STARTED_REQUIRES_EXPLICIT_USER_REQUEST"]
    for value in forbidden:require(value not in text,"FORBIDDEN:"+value)
    links=re.findall(r"\[[^]]+\]\(([^)]+)\)",text);missing=[]
    for target in links:
        if target.startswith(("http://","https://","#")):continue
        path=(DOC.parent/target).resolve()
        if not path.exists():missing.append(target)
    require(not missing,"BROKEN_LINKS:"+str(missing));require(text.count("```")%2==0 and text.count("$$")%2==0,"UNBALANCED_FENCES")
    readme=(ROOT/"README.md").read_text();roadmap=(ROOT/"docs/阶段路线图.md").read_text();name=DOC.name;require(name in readme and name in roadmap,"NAVIGATION_NOT_UNIFIED")
    print({"status":"PASS","document":DOC.relative_to(ROOT).as_posix(),"links":len(links),"required_markers":len(required),"next_authorization":"NONE_ENDPOINT_PRIMARY_TRACK_COMPLETE"})
if __name__=="__main__":main()
