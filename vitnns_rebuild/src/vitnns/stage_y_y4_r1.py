"""Stage Y4-R1 CPU-only mechanism analysis.

This module is the prospective, outcome-blind numerical contract for Y4-R1.
It intentionally has no model, image-decoder, accelerator, or network
dependency.  All reductions that define scientific results use float64.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.abc
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

PROTOCOL_ID = "STAGE_Y_VIT_BLOCK_LEVEL_NNS_MECHANISM_V1_20260910"
STAGE_ID = "Y4-R1"
TOKEN = "STAGE_Y4_CPU_ONLY_MECHANISM_ANALYSIS_AND_DECISION"
STARTUP_COMMIT = "8a19056c4e5e66b4ef4e7766cabb0fc6e7232f09"
BRANCH = "stage-d-d3-pass"
PYTHON = Path("/data01/home/yeziqi.831/nxtrec_wksp/testcode/.conda/envs/vitnns/bin/python")
ROOT = Path("evidence/stage_y/y4_r1_mechanism_analysis_20260914/formal")
REPORT_ROOT = Path("reports/stage_y_y4_r1_20260914")
REPORT = REPORT_ROOT / "report.md"
DOC = Path("docs/stage_y_y4_r1_mechanism_analysis_20260914.md")
SPEC = Path("config/stage_y/y4_r1_run_spec.json")
SPEC_SCHEMA = Path("config/stage_y/y4_r1_run_spec.schema.json")
ARTIFACT_SCHEMA = Path("config/stage_y/y4_r1_artifact.schema.json")
PRODUCER_REF = "refs/archive/stage-y-y4-r1-mechanism-analysis-producer"
PASS_REF = "refs/archive/stage-y-y4-r1-mechanism-analysis-and-decision"
BLOCKED_REF = "refs/archive/stage-y-y4-r1-mechanism-analysis-and-decision-blocked"
OLD_BLOCKED_REF = "refs/archive/stage-y-y4-mechanism-analysis-and-decision-blocked"
NEXT = "STAGE_Y5_RESULTS_TO_PAPER_HANDOFF_ONLY"
Y2_ROOT = Path("evidence/stage_y/y2_r2_r2_clean_formal_20260912/formal")
Y3_ROOT = Path("evidence/stage_y/y3_r6_reconciliation_formal_20260913/formal")
Y3_R4_ROOT = Path("evidence/stage_y/y3_r4_perturbation_formal_20260912/formal")
Y3_R5_ROOT = Path("evidence/stage_y/y3_r5_perturbation_formal_20260913/formal")
Y2_SOURCE_ROOTS = (Path("evidence/stage_y/y2_r1_clean_formal_20260911/formal"),Path("evidence/stage_y/y2_r2_r1_clean_formal_20260912/formal"))
Y3_SOURCE_ROOTS = (Path("evidence/stage_y/y3_r3_perturbation_formal_20260912/formal"),Y3_R4_ROOT,Y3_R5_ROOT)
Y0_COMPLETION = Path("evidence/stage_y/y0_protocol_20260910/formal/completion/y0.json")
Y2_COMPLETION = Y2_ROOT / "completion/y2_r2_r2.json"
Y3_COMPLETION = Y3_ROOT / "completion/y3_r6.json"
Y2_MANIFEST = Y2_ROOT / "unified_100_unit_manifest.json"
Y3_MANIFEST = Y3_ROOT / "source_manifest.json"

PRODUCER_FILES = (DOC, SPEC, SPEC_SCHEMA, ARTIFACT_SCHEMA,
    Path("src/vitnns/stage_y_y4_r1.py"), Path("scripts/run_stage_y_y4_r1.py"),
    Path("scripts/validate_stage_y_y4_r1.py"), Path("tests/test_stage_y_y4_r1.py"))
CONTRACT_FILES = (Path("config/stage_y/y0_protocol.json"),
    Path("manifests/stage_y/claim_registry.json"), Path("manifests/stage_y/metric_registry.json"),
    Path("manifests/stage_y/statistical_analysis_plan.json"), Path("manifests/stage_y/component_cut_registry.json"),
    Path("manifests/stage_y/retention_plan.json"), Path("manifests/stage_y/schema_registry.json"))
DEPENDENCIES = (Path("manifests/stage_y/transform_registry.json"),
    Path("src/vitnns/stage_y_y2_r2.py"), Path("src/vitnns/stage_y_y2_r2_schema.py"),
    Path("src/vitnns/stage_y_y2_r2_r2.py"), Path("src/vitnns/stage_y_y3.py"),
    Path("src/vitnns/stage_y_y3_r5.py"), Path("src/vitnns/stage_y_y3_r6.py"))
EXPECTED_REFS = {
    "refs/archive/stage-y-y0-protocol-freeze": "a6efa1426902708d83de1d83ef257d524cacf95b",
    "refs/archive/stage-y-y1-r1-resource-telemetry-waiver": "70f70ca7290316531406791a6533d65590424a24",
    "refs/archive/stage-y-y2-r2-r2-clean-cohort-evidence-completion": "f679987a517412dc54fcde4a0d5820241ca7c6c8",
    "refs/archive/stage-y-y3-r4-perturbation-mechanism-producer": "e46e7500e1c1b3ef00f96f0bdd907070b70ec11e",
    "refs/archive/stage-y-y3-r4-perturbation-mechanism-evidence-blocked": "fb431aa1792a671d790baf224773e327ec017f21",
    "refs/archive/stage-y-y3-r5-global-budget-gate-recovery-producer": "d4726af9f11a41fd7d890a19b4c3441cea914136",
    "refs/archive/stage-y-y3-r5-perturbation-mechanism-evidence-blocked": "b027612e1963bb8636e575d8f6dc44d71ee0c302",
    "refs/archive/stage-y-y3-r6-validator-recovery-producer": "1e1e5e85245563ec872194568f7406d3c12599df",
    "refs/archive/stage-y-y3-r6-perturbation-mechanism-evidence": "cbec8e1ec258c9ae018d4f75275f9c0feaa19cd7",
    OLD_BLOCKED_REF: STARTUP_COMMIT,
}
EXPECTED_RAW_SHA = {
    "config/stage_y/y0_protocol.json": "3b5acf5aebfdb8ce6a42ab39e9915ea89dd5bea942b1e4a2d651de55ee50a56e",
    "manifests/stage_y/claim_registry.json": "2d408eef4452f229b7204b555515ab7dbcad0525d98cf754f6e6e490390667a6",
    "manifests/stage_y/metric_registry.json": "7bedbb64cb1bbb392727a5d12583d1c819579fbea943ccfb5c3eb9c51f91dfdf",
    "manifests/stage_y/statistical_analysis_plan.json": "4a937143eef978c34d248234c3be86ba4939cf66c9d72700725000699a02d278",
    "manifests/stage_y/component_cut_registry.json": "cebc4b35f2097b3cc28dad954b88988f8cee4b8a86ea90243ffcc7f0f1c78856",
    "manifests/stage_y/retention_plan.json": "77268df5ec2c33b41b87360ac68b4a23b1ff77cbdecdd82eb460d8e3ad6af398",
    "manifests/stage_y/schema_registry.json": "00dd63edd3c321a8403221aac0ae150ccb0806e3a61608ae49f6a6530c96cdc0",
}
EXPECTED_CANONICAL = {
    "config/stage_y/y0_protocol.json": "3880530811d5b1a9ed12006244e6b316d01a4b8ab6e595acc25727317e154c40",
    "manifests/stage_y/claim_registry.json": "99ab056718f5f5a0d8ac1de009bd656d8a1c9cbf6e4bca6563db83f56cb408ef",
    "manifests/stage_y/metric_registry.json": "dd0b53bfffcd23d9e6746f5331e188e8678e43709441f7796bb9bff27a64a16e",
    "manifests/stage_y/statistical_analysis_plan.json": "26fd005e674e9006bfcdb2db642d5a4a5f3fec6cb78b3c24f572348415706e9a",
    "manifests/stage_y/component_cut_registry.json": "54f3ca77177a901b4ec2f763c0ea0d697a76f737cd035145f932e2c65c507411",
    "manifests/stage_y/retention_plan.json": "339912d3452100924cfa81a27e9e001717e76e1dec163b6c8fcb25edbaf671c0",
    "manifests/stage_y/schema_registry.json": "bfca9e05b9815083139eeb9894c49db40ddb9dd56015c761b089c5d44305ec82",
}
EXPECTED_COMPLETIONS = {
    str(Y0_COMPLETION): ("ccf1be03e05a4ff441afe0609113c50e3a848a82a006df735646d8a0523e6c2c", "a2c3e6b214297f6f9ab41a7faeab1a4b56ae8231a3fb15026c3c16ca149f68b5"),
    str(Y2_COMPLETION): ("f09205218a1c24d6b4e3be457f3bb22c60dd099c7cf73a0db7cb6a8e545e54d5", "6387ff5fbd895340cacd189e7f86759840af1448f67d655d873352f35a692a3c"),
    str(Y3_COMPLETION): ("29e18b412c7c78a075160b140fafd0ef5af8e4234bb49e1abc218e62ac03783e", "d72336f0111354d362eb57f6aa4c6d0b007f966cc6aeeb0958eaddbecff8ee5a"),
}
BLOCKS = tuple(range(12)); HEADS = tuple(range(12))
DEPTH = {"shallow": (0,1,2,3), "middle": (4,5,6,7), "deep": (8,9,10,11)}
CONDITIONS = ("horizontal_flip","translate_x_-8","translate_x_8","rotate_-10","rotate_10","brightness_0.9","brightness_1.1","gaussian_noise_0.01_seed_0")
HYPOTHESES = (
    ("A1","A","greater",0.02),("B1","B","two_sided",0.09531017980432493),
    ("B2","B","two_sided",0.5),("B3","B","two_sided",0.05),
    ("B4","B","two_sided",0.05),("B5","B","two_sided",0.05),
    ("B6","B","greater",0.2027325540540822),("B7","B","greater",0.02),
    ("C1","C","greater",0.05),("C2","C","greater",0.05),
    ("C3","C","greater",0.05),("C4","C","greater",0.03),
)
HYPOTHESIS_ENDPOINTS = {
    "A1":"hierarchical_positive_deletion_auc_gain",
    "B1":"log_effective_spatial_support_deep_minus_shallow",
    "B2":"effective_head_count_deep_minus_shallow",
    "B3":"off_diagonal_absolute_mass_fraction_deep_minus_shallow",
    "B4":"first_residual_attention_absolute_mass_share_deep_minus_shallow",
    "B5":"second_residual_mlp_absolute_mass_share_deep_minus_shallow",
    "B6":"head_allocation_ablation_spearman_fisher_z",
    "B7":"top3_minus_bottom3_head_ablation_loss",
    "C1":"mean_absolute_condition_change_within_patch_normalized_total_variation",
    "C2":"mean_one_minus_head_absolute_share_cosine",
    "C3":"mean_one_minus_inverse_mapped_patch_matrix_absolute_mass_cosine",
    "C4":"mean_absolute_branch_share_change",
}
BOOTSTRAP_REPLICATES=20000; BOOTSTRAP_SEED=792020260910
SIGN_REPLICATES=100000; SIGN_SEED=792120260910
FORBIDDEN_IMPORTS={"torch","torchvision","timm","PIL","cv2","cupy","jax","tensorflow"}
FORBIDDEN_SUFFIXES={".jpg",".jpeg",".png",".svg",".pdf",".pth",".pt",".ckpt"}
FALSE_PASS_ATTACKS=(
 "historical_evidence_mutation","historical_ref_drift","summary_only_Y2_validation","summary_only_Y3_validation","image_byte_read","image_decode","checkpoint_load","forbidden_framework_import","cuda_initialization","gpu_execution","new_empirical_attempt","clean_rerun","perturbation_rerun","display_array_input","array_hash_drift","array_shape_drift","array_dtype_drift","array_axes_drift","array_nonfinite","target_reselection","stage_d_target_mixing","raw_attention_as_truth","best_head_selection","best_block_selection","best_condition_selection","pseudo_replication","depth_group_drift","pool_before_image_aggregate","failed_na_deletion","zero_imputation","replacement_image","unpaired_contrast","bootstrap_replicate_drift","bootstrap_seed_drift","bootstrap_prng_drift","bootstrap_median_drift","bootstrap_quantile_drift","bootstrap_split_image","signflip_replicate_drift","signflip_seed_drift","signflip_margin_drift","signflip_plus_one_missing","holm_missing","holm_family_merge","holm_hypothesis_omission","holm_tie_instability","raw_p_claim_decision","threshold_direction_drift","posthoc_endpoint_promotion","selective_trajectory_output","incomplete_head_complete_case","three_way_branch_share","cross_layer_sum","inverse_map_direction","inverse_matrix_direction","geometric_direct_comparison","valid_mask_ignored","prediction_flip_deletion","broad_robustness_wording","x2_reopened","omnibus_decision","all_supported_required_for_pass","unfavorable_result_hidden","open_nested_schema","early_pass_ref","pending_or_temp_left","unverified_publication","Y5_started")

class Y4Error(RuntimeError): code="BLOCKED_PROVENANCE"
class StartupError(Y4Error): code="BLOCKED_STARTUP_STATE"
class ProtocolError(Y4Error): code="BLOCKED_PROTOCOL_UNDERSPECIFIED"
class ProvenanceError(Y4Error): code="BLOCKED_PROVENANCE"
class AccessError(Y4Error): code="BLOCKED_ACCESS"
class InfrastructureError(Y4Error): code="BLOCKED_INFRASTRUCTURE"
class ScientificError(Y4Error): code="BLOCKED_SCIENTIFIC_VALIDITY"

def require(flag: Any, message: str, error: type[Y4Error]=ProvenanceError) -> None:
    if not flag: raise error(message)

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def canonical(value: Any) -> str: return hashlib.sha256(canonical_bytes(value)).hexdigest()
def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def read(path: Path) -> Any:
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,f"duplicate JSON key: {path}:{k}");out[k]=v
        return out
    def reject(v): raise ProvenanceError(f"nonfinite JSON: {path}:{v}")
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=reject)
def file_record(path: Path, repo: Path) -> dict[str,Any]:
    require(path.is_file() and not path.is_symlink(),f"missing/symlink file: {path}")
    row={"path":path.resolve().relative_to(repo.resolve()).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)}
    if path.suffix==".json":
        value=read(path)
        if isinstance(value,dict) and "canonical_digest" in value:
            unsigned=dict(value);claimed=unsigned.pop("canonical_digest");require(canonical(unsigned)==claimed,f"canonical drift: {path}");row["canonical_digest"]=claimed
    return row
def git(repo:Path,*args:str)->str:
    return subprocess.check_output(["git",*args],cwd=repo,text=True).strip()

class ImportBoundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in FORBIDDEN_IMPORTS: raise AccessError("forbidden import: "+fullname)
        return None
_GUARDED=False
ACCESS_COUNTS={"scientific_files_read":0,"scientific_bytes_read":0}
def runtime_guard() -> None:
    global _GUARDED
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CUDA_VISIBLE_DEVICES must be empty",AccessError)
    require(not FORBIDDEN_IMPORTS.intersection(x.split(".")[0] for x in sys.modules),"forbidden module already imported",AccessError)
    if _GUARDED:return
    sys.meta_path.insert(0,ImportBoundary())
    def audit(event,args):
        if event in ("socket.connect","socket.bind","socket.getaddrinfo"):raise AccessError("network access forbidden")
        if event=="subprocess.Popen":
            cmd=args[1]; require(isinstance(cmd,(list,tuple)) and Path(str(cmd[0])).name=="git","only Git subprocess is permitted",AccessError)
        if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0]));require(p.suffix.lower() not in FORBIDDEN_SUFFIXES,f"forbidden bytes access: {p}",AccessError)
    sys.addaudithook(audit);_GUARDED=True

def obj(props:Mapping[str,Any],optional:Sequence[str]=()):return {"type":"object","additionalProperties":False,"properties":dict(props),"required":[k for k in props if k not in optional]}
def arr(item:Mapping[str,Any],count:int|None=None):
    v={"type":"array","items":dict(item)}
    if count is not None:v.update(minItems=count,maxItems=count)
    return v
def nullable(item):return {"anyOf":[dict(item),{"type":"null"}]}
S={"type":"string"};I={"type":"integer"};NN={"type":"number"};B={"type":"boolean"};H={"type":"string","pattern":"^[0-9a-f]{64}$"};G={"type":"string","pattern":"^[0-9a-f]{40}$"}
FILE=obj({"path":S,"bytes":I,"sha256":H,"canonical_digest":H},("canonical_digest",))
COUNT=obj({"expected":I,"VALID":I,"FAILED":I,"NA":I})
CHECK=obj({"attack":S,"expected":{"const":"REJECT"},"observed":{"const":"REJECT"},"pass":{"const":True}})
def artifact_schema()->dict[str,Any]:
    binding=obj({"name":S,"files":arr(FILE),"files_digest":H,"commit":nullable(G)})
    table=obj({"file":FILE,"format":{"enum":["CSV","NPY"]},"columns":arr(obj({"name":S,"dtype":S})),"dtype":nullable(S),"shape":arr(I),"axes":arr(S),"row_ordering":arr(S),"data_canonical_digest":H})
    hypothesis=obj({"hypothesis_id":S,"endpoint":S,"family":S,"direction":S,"threshold":NN,"n":I,"estimate":nullable(NN),"ci_lower":nullable(NN),"ci_upper":nullable(NN),"raw_p":nullable(NN),"holm_rank":nullable(I),"holm_threshold":nullable(NN),"adjusted_p":nullable(NN),"reject":B,"outcome":{"enum":["SUPPORTED","NOT_SUPPORTED","INCONCLUSIVE","INVALID"]}})
    decision=obj({"claim_id":S,"decision":{"enum":["SUPPORTED","NOT_SUPPORTED","INCONCLUSIVE","INVALID"]},"hypotheses":arr(S),"n":I,"allowed_wording":S,"reason":S})
    variants={
      "pending":obj({"started_unix_ns":I,"next_authorization":{"const":"NONE_WHILE_PENDING"}}),
      "producer_preflight":obj({"startup_commit":G,"checks":arr(S),"test_count":I,"false_pass_attack_count":I,"startup_refs_digest":H,"formal_outcome_access":{"const":False}}),
      "implementation_seal":obj({"producer_commit":G,"producer_ref":{"const":PRODUCER_REF},"source_files":arr(FILE),"files_digest":H}),
      "binding":binding,
      "source_manifest":obj({"population":S,"records":arr(obj({"index":I,"source_object_id":S,"condition_id":S,"status":S,"source_terminal":FILE})),"record_count":I,"records_digest":H,"arrays_copied":{"const":False}}),
      "array_validation":obj({"source_unit_count":I,"array_count":I,"validated_bytes":I,"records_digest":H,"dtype_shape_axes_finite_pass":B,"closure_pass":B}),
      "table_sidecar":table,
      "bootstrap_manifest":obj({"replicates":{"const":BOOTSTRAP_REPLICATES},"seed":{"const":BOOTSTRAP_SEED},"prng":S,"stream_rule":S,"statistic":S,"quantiles":arr(NN,2),"quantile_method":S,"table":FILE,"sidecar":FILE}),
      "signflip_manifest":obj({"replicates":{"const":SIGN_REPLICATES},"seed":{"const":SIGN_SEED},"prng":S,"stream_rule":S,"plus_one":{"const":True},"table":FILE,"sidecar":FILE}),
      "hypothesis_table":obj({"records":arr(hypothesis,12),"record_count":{"const":12},"records_digest":H}),
      "holm_table":obj({"family":{"enum":["A","B","C"]},"alpha":NN,"records":arr(hypothesis),"records_digest":H}),
      "claim_decision":decision,
      "claim_summary":obj({"records":arr(decision,6),"record_count":{"const":6},"records_digest":H,"omnibus_decision":{"const":False}}),
      "missingness":obj({"clean":COUNT,"perturbation_images":COUNT,"nonclean_conditions":COUNT,"rules":arr(S)}),
      "access_audit":obj({"jpeg_bytes_read":{"const":0},"image_decode_count":{"const":0},"checkpoint_bytes_read":{"const":0},"model_load_count":{"const":0},"torch_imported":{"const":False},"timm_imported":{"const":False},"torchvision_imported":{"const":False},"cuda_initialized":{"const":False},"cuda_access_count":{"const":0},"gpu_execution":{"const":False},"download_bytes":{"const":0},"new_empirical_attempt_count":{"const":0},"clean_rerun_count":{"const":0},"perturbation_rerun_count":{"const":0},"scientific_files_read":I,"scientific_bytes_read":I}),
      "resource":obj({"cpu_seconds":NN,"cpu_hours":NN,"process_wall_seconds":NN,"stage_wall_seconds":NN,"peak_rss_bytes":I,"read_bytes":I,"written_bytes":I,"retained_bytes":I,"file_count":I,"gpu_hours":{"const":0},"cuda_access_count":{"const":0},"model_calls":{"const":0},"jvp_calls":{"const":0},"vjp_calls":{"const":0},"backward_calls":{"const":0},"download_bytes":{"const":0},"bootstrap_operation_count":I,"sign_flip_operation_count":I,"cap_status":S}),
      "cumulative_resource":obj({"prior_stage_ledger":FILE,"y3_stage_ledger":FILE,"y4_ledger":FILE,"note":S}),
      "false_pass_audit":obj({"checks":arr(CHECK),"attack_count":I,"rejected_count":I}),
      "validation_report":obj({"source_recomputed":{"const":True},"checks":arr(S),"source_digest":H,"metric_digest":H,"bootstrap_digest":H,"signflip_digest":H,"hypothesis_digest":H,"claim_digest":H,"status":{"const":"PASS"}}),
      "descriptive_manifest":obj({"tables":arr(FILE),"roles":arr(S),"confirmatory_promotion":{"const":False}}),
      "inventory":obj({"root":S,"excluded":arr(S),"files":arr(FILE),"file_count":I,"total_bytes":I,"files_digest":H}),
      "reconciliation":obj({"workflow_status":S,"clean_denominator":{"const":100},"clean_valid_count":I,"perturbation_denominator_images":{"const":20},"perturbation_complete_image_count":I,"condition_record_count":{"const":180},"nonclean_terminal_count":{"const":160},"hypothesis_count":{"const":12},"claim_count":{"const":6},"formal_outcome_access":{"const":True},"y5_executed":{"const":False},"next_authorization":S}),
      "completion":obj({"workflow_status":S,"clean_denominator":{"const":100},"clean_valid_count":I,"perturbation_denominator_images":{"const":20},"perturbation_complete_image_count":I,"condition_record_count":{"const":180},"nonclean_terminal_count":{"const":160},"new_empirical_attempt_count":{"const":0},"gpu_execution":{"const":False},"bootstrap_replicates":{"const":BOOTSTRAP_REPLICATES},"sign_flip_replicates":{"const":SIGN_REPLICATES},"holm_families":{"const":["A","B","C"]},"hypothesis_count":{"const":12},"claim_count":{"const":6},"per_claim_decisions":obj({k:{"enum":["SUPPORTED","NOT_SUPPORTED","INCONCLUSIVE","INVALID"]} for k in ("Y-CLM-INTRAPATCH","Y-CLM-DEPTH-SPATIAL","Y-CLM-HEAD-ORGANIZATION","Y-CLM-PATCH-MIXING","Y-CLM-BRANCH-ALLOCATION","Y-CLM-PERTURBATION-MODULATION")}),"formal_outcome_access":{"const":True},"y5_executed":{"const":False},"next_authorization":S,"reconciliation":FILE,"validation_report":FILE,"whole_root_inventory":FILE,"payload_manifest":FILE,"report":FILE,"failure":nullable(FILE)}),
      "failure":obj({"phase":S,"failure_type":S,"message":S,"next_authorization":{"const":"NONE"}}),
    }
    env=obj({"$schema":{"const":ARTIFACT_SCHEMA.name},"schema_version":{"const":1},"protocol_id":{"const":PROTOCOL_ID},"stage_id":{"const":STAGE_ID},"document_type":{"enum":list(variants)},"status":S,"payload":{"oneOf":[{"$ref":"#/$defs/"+k} for k in variants]},"canonical_digest":H})
    env.update({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":ARTIFACT_SCHEMA.name,"$defs":variants,"allOf":[{"if":{"properties":{"document_type":{"const":k}}},"then":{"properties":{"payload":{"$ref":"#/$defs/"+k}}}} for k in variants]})
    return env

_SCHEMA=None
def signed(kind:str,payload:Mapping[str,Any],status:str="PASS")->dict[str,Any]:
    value={"$schema":ARTIFACT_SCHEMA.name,"schema_version":1,"protocol_id":PROTOCOL_ID,"stage_id":STAGE_ID,"document_type":kind,"status":status,"payload":dict(payload)};value["canonical_digest"]=canonical(value);validate_document(value);return value
def validate_document(value:Mapping[str,Any])->None:
    global _SCHEMA
    u=dict(value);claim=u.pop("canonical_digest",None);require(claim==canonical(u),"artifact canonical digest mismatch")
    if _SCHEMA is None:_SCHEMA=Draft202012Validator(artifact_schema())
    _SCHEMA.validate(value)
def write(path:Path,kind:str,payload:Mapping[str,Any],status:str="PASS")->dict[str,Any]:
    value=signed(kind,payload,status);path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(canonical_bytes(value)+b"\n");f.flush();os.fsync(f.fileno())
    d=os.open(path.parent,os.O_RDONLY);os.fsync(d);os.close(d);return value
def publish_terminal(pending:Path,path:Path,value:Mapping[str,Any])->None:
    validate_document(value);require(pending.is_file() and not path.exists(),"invalid terminal transition");tmp=path.with_name(path.name+".tmp");fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(canonical_bytes(value)+b"\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);pending.unlink();d=os.open(path.parent,os.O_RDONLY);os.fsync(d);os.close(d)

def mass_summary(value:np.ndarray)->dict[str,float]:
    x=np.asarray(value,dtype=np.float64);require(np.isfinite(x).all(),"nonfinite mass input",ScientificError);a=np.abs(x);absolute=float(a.sum(dtype=np.float64));require(absolute>0,"zero absolute mass",ScientificError)
    return {"signed_conductance":float(x.sum(dtype=np.float64)),"positive_mass":float(np.maximum(x,0).sum(dtype=np.float64)),"negative_mass":float(np.minimum(x,0).sum(dtype=np.float64)),"absolute_mass":absolute,"cancellation":1.0-abs(float(x.sum(dtype=np.float64)))/absolute}
def entropy_effective(weights:np.ndarray)->tuple[float,float]:
    a=np.asarray(weights,dtype=np.float64).reshape(-1);s=float(a.sum(dtype=np.float64));require(s>0 and np.all(a>=0),"invalid entropy mass",ScientificError);p=a[a>0]/s;h=float(-(p*np.log(p)).sum(dtype=np.float64));return h,float(math.exp(h))
def gini(weights:np.ndarray)->float:
    x=np.sort(np.asarray(weights,dtype=np.float64).reshape(-1));s=float(x.sum(dtype=np.float64));require(s>0 and np.all(x>=0),"invalid Gini mass",ScientificError);n=x.size;return float((2*np.dot(np.arange(1,n+1,dtype=np.float64),x)/(n*s))-(n+1)/n)
def within_patch(cell:np.ndarray)->dict[str,float]:
    x=np.asarray(cell,dtype=np.float64);require(x.shape==(56,56) and np.isfinite(x).all(),"within-patch array invalid",ScientificError);p=x.reshape(14,4,14,4).transpose(0,2,1,3).reshape(196,4,4);a=np.abs(p);mass=a.sum((1,2),dtype=np.float64);require(np.all(mass>0),"zero-mass native patch",ScientificError)
    tv=(np.abs(np.diff(p,axis=1)).sum((1,2),dtype=np.float64)+np.abs(np.diff(p,axis=2)).sum((1,2),dtype=np.float64))/(mass+1e-12)
    flat=a.reshape(196,16);probs=flat/mass[:,None];ent=-(np.where(probs>0,probs*np.log(probs),0.0)).sum(1,dtype=np.float64);m=mass_summary(x)
    return {"within_patch_population_variance":float(np.var(p,axis=(1,2),ddof=0).mean(dtype=np.float64)),"within_patch_normalized_total_variation":float(tv.mean(dtype=np.float64)),"within_patch_top_cell_absolute_mass_share":float((flat.max(1)/mass).mean(dtype=np.float64)),"within_patch_entropy":float(ent.mean(dtype=np.float64)),"within_patch_effective_cell_count":float(np.exp(ent).mean(dtype=np.float64)),**m}
def spatial_support(cell:np.ndarray)->dict[str,float]:
    x=np.asarray(cell,dtype=np.float64);a=np.abs(x);total=float(a.sum(dtype=np.float64));require(x.shape==(56,56) and total>0 and np.isfinite(x).all(),"spatial map invalid",ScientificError);h,e=entropy_effective(a);flat=np.sort(a.reshape(-1))[::-1];patch=a.reshape(14,4,14,4).sum((1,3),dtype=np.float64);yy,xx=np.indices((14,14),dtype=np.float64);cy=float((patch*yy).sum()/total);cx=float((patch*xx).sum()/total);r=float(np.sqrt((patch*((yy-cy)**2+(xx-cx)**2)).sum()/total))
    def displacement(z):
        s=float(z.sum());return None if s<=0 else float(math.hypot(float((z*yy).sum()/s)-6.5,float((z*xx).sum()/s)-6.5)/(13*math.sqrt(2)))
    pos=np.maximum(x,0).reshape(14,4,14,4).sum((1,3),dtype=np.float64);neg=np.maximum(-x,0).reshape(14,4,14,4).sum((1,3),dtype=np.float64);m=mass_summary(x)
    return {"absolute_mass_entropy":h,"effective_spatial_support":e,"log_effective_spatial_support":math.log(e),"gini":gini(a),"top_1_percent_absolute_mass_share":float(flat[:32].sum()/total),"top_10_percent_absolute_mass_share":float(flat[:314].sum()/total),"source_patch_rms_radius":r,"positive_center_displacement":displacement(pos),"negative_center_displacement":displacement(neg),"cancellation":m["cancellation"]}
def head_organization(head_cells:np.ndarray)->tuple[dict[str,float],list[dict[str,float]]]:
    x=np.asarray(head_cells,dtype=np.float64);require(x.shape==(12,56,56) and np.isfinite(x).all(),"head roster invalid",ScientificError);mass=np.abs(x).sum((1,2),dtype=np.float64);total=float(mass.sum());require(total>0 and np.all(mass>0),"head mass invalid",ScientificError);share=mass/total;h,e=entropy_effective(mass);order=np.sort(share)[::-1]
    rows=[]
    for i in HEADS:rows.append({"head":i,"absolute_share":float(share[i]),**mass_summary(x[i])})
    return {"head_top1_share":float(order[0]),"head_top3_share":float(order[:3].sum()),"head_hhi":float(np.square(share).sum()),"head_entropy":h,"effective_head_count":e},rows
def patch_mixing(matrix:np.ndarray,cls:np.ndarray,with_rank:bool=True)->dict[str,float]:
    t=np.asarray(matrix,dtype=np.float64);c=np.asarray(cls,dtype=np.float64);require(t.shape==(196,196) and c.shape==(196,) and np.isfinite(t).all() and np.isfinite(c).all(),"patch matrix invalid",ScientificError);a=np.abs(t);tm=float(a.sum());cm=float(np.abs(c).sum());require(tm>0 and tm+cm>0,"zero patch matrix mass",ScientificError);diag=float(np.trace(a));src=a.sum(1);dst=a.sum(0);yy,xx=np.indices((14,14));coords=np.stack((yy.reshape(-1),xx.reshape(-1)),axis=1).astype(np.float64);dist2=np.square(coords[:,None,:]-coords[None,:,:]).sum(2);dh,de=entropy_effective(dst);out={"diagonal_absolute_mass_fraction":diag/tm,"off_diagonal_absolute_mass_fraction":1-diag/tm,"source_to_destination_rms_radius":float(math.sqrt(float((a*dist2).sum())/tm)),"destination_entropy":dh,"effective_destination_count":de,"source_concentration":float(np.square(src/tm).sum()),"destination_concentration":float(np.square(dst/tm).sum()),"cls_routing_fraction":cm/(tm+cm),"matrix_signed_cancellation":1-abs(float(t.sum()+c.sum()))/(tm+cm)}
    if with_rank:
        singular=np.linalg.svd(t,compute_uv=False);out["matrix_effective_rank"]=entropy_effective(singular)[1]
    else:out["matrix_effective_rank"]=float("nan")
    return out
def branch_allocation(identity:np.ndarray,attention:np.ndarray,residual:np.ndarray,mlp:np.ndarray,total_in:np.ndarray,total_out:np.ndarray)->dict[str,float]:
    values=[np.asarray(v,dtype=np.float64) for v in (identity,attention,residual,mlp,total_in,total_out)];require(all(v.shape==(56,56) and np.isfinite(v).all() for v in values),"branch arrays invalid",ScientificError);im,am,rm,mm=[float(np.abs(v).sum()) for v in values[:4]];require(im+am>0 and rm+mm>0,"branch pair zero mass",ScientificError)
    out={"first_attention_share":am/(im+am),"second_mlp_share":mm/(rm+mm),"first_reconstruction_max_abs":float(np.max(np.abs(values[0]+values[1]-values[4]))),"second_reconstruction_max_abs":float(np.max(np.abs(values[2]+values[3]-values[5])))}
    for name,v in zip(("first_identity","first_attention","second_residual","second_mlp"),values[:4]):
        for k,val in mass_summary(v).items():out[name+"_"+k]=val
    return out
def rankdata_average(x:np.ndarray)->np.ndarray:
    x=np.asarray(x,dtype=np.float64);order=np.argsort(x,kind="mergesort");r=np.empty(x.size,dtype=np.float64);i=0
    while i<x.size:
        j=i+1
        while j<x.size and x[order[j]]==x[order[i]]:j+=1
        r[order[i:j]]=(i+j-1)/2+1;i=j
    return r
def spearman(x:np.ndarray,y:np.ndarray)->float:
    a=rankdata_average(x);b=rankdata_average(y);a-=a.mean();b-=b.mean();den=float(np.sqrt(np.dot(a,a)*np.dot(b,b)));require(den>0,"undefined Spearman",ScientificError);return float(np.dot(a,b)/den)
def seed64(namespace:str,*parts:object)->int:
    return int.from_bytes(hashlib.sha256((namespace+"\0"+"\0".join(str(x) for x in parts)).encode()).digest()[:8],"big")
def stable_descending(values:np.ndarray,namespace:str,*parts:object)->list[int]:
    x=np.asarray(values,dtype=np.float64).reshape(-1);ties=np.asarray([seed64(namespace,*parts,i) for i in range(x.size)],dtype=np.uint64);return [int(i) for i in np.lexsort((ties,-x))]
def cosine_abs(a:np.ndarray,b:np.ndarray)->float:
    x=np.abs(np.asarray(a,dtype=np.float64)).reshape(-1);y=np.abs(np.asarray(b,dtype=np.float64)).reshape(-1);den=float(np.linalg.norm(x)*np.linalg.norm(y));require(den>0,"undefined cosine",ScientificError);return float(np.dot(x,y)/den)
def recompute_interventions(science:Mapping[str,Any],head_cells:Mapping[int,np.ndarray],source_object_id:str)->dict[str,Any]:
    deletion=science["deletion_operator"];ablation=science["ablations"];require(deletion["status"]==ablation["status"]=="PASS","intervention status invalid",ScientificError);gains=[];fisher=[];topbottom=[]
    for block,d in zip(BLOCKS,deletion["blocks"]):
        require(d["block"]==block and len(d["controls"])==32,"deletion roster invalid",ScientificError);x=np.asarray(d["percentiles"],dtype=np.float64)/100;y=np.asarray(d["nns_losses"],dtype=np.float64);nauc=float(np.trapz(y,x));controls=[float(np.trapz(np.asarray(c["losses"],dtype=np.float64),x)) for c in d["controls"]];gain=nauc-float(np.median(controls));require(math.isclose(nauc,d["nns_auc"],rel_tol=0,abs_tol=1e-12) and math.isclose(gain,d["auc_gain"],rel_tol=0,abs_tol=1e-12),"deletion endpoint drift",ScientificError);gains.append(gain)
    for block,a in zip(BLOCKS,ablation["blocks"]):
        require(a["block"]==block and len(a["single_head"])==12,"ablation roster invalid",ScientificError);alloc=np.abs(np.asarray(head_cells[block],dtype=np.float64)).sum((1,2),dtype=np.float64);require(np.allclose(alloc,np.asarray(a["allocation_absolute_mass"]),rtol=0,atol=1e-9),"stored head allocation drift",ScientificError);order=stable_descending(alloc,"STAGE_Y_HEAD_RANK_V1_20260910",source_object_id,block);require(order==a["rank_order"],"head rank/tie-break drift",ScientificError);loss=np.asarray([v["normalized_loss"] for v in sorted(a["single_head"],key=lambda z:z["head"])],dtype=np.float64);rho=float(np.clip(spearman(alloc,loss),-0.999999,0.999999));fisher.append(float(np.arctanh(rho)));top=next(v for v in a["curves"]["top"] if v["k"]==3);bottom=next(v for v in a["curves"]["bottom"] if v["k"]==3);require(top["heads"]==order[:3] and bottom["heads"]==order[::-1][:3],"top/bottom-3 curve rank drift",ScientificError);topbottom.append(float(top["normalized_loss"]-bottom["normalized_loss"]))
    return {"A1":float(np.mean(gains,dtype=np.float64)),"B6":float(np.mean(fisher,dtype=np.float64)),"B7":float(np.mean(topbottom,dtype=np.float64)),"block_auc_gain":gains,"block_fisher_z":fisher,"block_top3_minus_bottom3":topbottom}

def bootstrap(values:Sequence[float],replicates:int=BOOTSTRAP_REPLICATES,seed:int=BOOTSTRAP_SEED)->tuple[float,float,float,np.ndarray]:
    x=np.asarray(values,dtype=np.float64);require(x.ndim==1 and x.size>0 and np.isfinite(x).all(),"bootstrap input invalid",ScientificError);rng=np.random.Generator(np.random.PCG64(seed));indices=rng.integers(0,x.size,size=(replicates,x.size));samples=np.median(x[indices],axis=1);q=np.quantile(samples,[.025,.975],method="linear");return float(np.median(x)),float(q[0]),float(q[1]),np.ascontiguousarray(samples,dtype="<f8")
def sign_flip(values:Sequence[float],direction:str,margin:float,replicates:int=SIGN_REPLICATES,seed:int=SIGN_SEED)->tuple[float,np.ndarray]:
    x=np.asarray(values,dtype=np.float64);require(x.ndim==1 and x.size>0 and np.isfinite(x).all(),"sign-flip input invalid",ScientificError);rng=np.random.Generator(np.random.PCG64(seed));signs=rng.integers(0,2,size=(replicates,x.size),dtype=np.int8)*2-1
    def one(d):
        observed=float(np.median(d));null=np.median(signs*d[None,:],axis=1);return (1+int(np.count_nonzero(null>=observed)))/(replicates+1),null
    pp,npv=one(x-margin)
    pm,nmv=one(-x-margin)
    if direction=="greater":return float(pp),np.ascontiguousarray(np.stack((npv,nmv)),dtype="<f8")
    require(direction=="two_sided","unknown hypothesis direction",ProtocolError);return float(min(1.0,2*min(pp,pm))),np.ascontiguousarray(np.stack((npv,nmv)),dtype="<f8")
def holm(rows:Sequence[Mapping[str,Any]],alpha:float=.05)->list[dict[str,Any]]:
    ordered=sorted((dict(r) for r in rows),key=lambda r:(float(r["raw_p"]),str(r["hypothesis_id"])));m=len(ordered);running=0.0;active=True
    for rank,r in enumerate(ordered,1):
        threshold=alpha/(m-rank+1);raw=float(r["raw_p"]);running=max(running,(m-rank+1)*raw);r.update(holm_rank=rank,holm_threshold=threshold,adjusted_p=min(1.0,running),reject=bool(active and raw<threshold));active=active and raw<threshold
    return sorted(ordered,key=lambda r:r["hypothesis_id"])
def hypothesis_outcome(row:Mapping[str,Any])->str:
    if row["n"] < (18 if row["family"]=="C" else 90):return "INVALID"
    lo=float(row["ci_lower"]);hi=float(row["ci_upper"]);t=float(row["threshold"]);reject=bool(row["reject"])
    if row["direction"]=="greater":
        if reject and lo>t:return "SUPPORTED"
        if hi<=t:return "NOT_SUPPORTED"
    else:
        if reject and (lo>t or hi<-t):return "SUPPORTED"
        if lo>=-t and hi<=t:return "NOT_SUPPORTED"
    return "INCONCLUSIVE"
def claim_decisions(hyp:Mapping[str,Mapping[str,Any]],allowed:Mapping[str,str])->list[dict[str,Any]]:
    rules=(("Y-CLM-INTRAPATCH",("A1",)),("Y-CLM-DEPTH-SPATIAL",("B1",)),("Y-CLM-HEAD-ORGANIZATION",("B2","B6","B7")),("Y-CLM-PATCH-MIXING",("B3",)),("Y-CLM-BRANCH-ALLOCATION",("B4","B5")),("Y-CLM-PERTURBATION-MODULATION",("C1","C2","C3","C4")))
    out=[]
    for cid,ids in rules:
        states=[hyp[i]["outcome"] for i in ids];n=min(int(hyp[i]["n"]) for i in ids)
        if "INVALID" in states:decision="INVALID"
        elif all(x=="SUPPORTED" for x in states):decision="SUPPORTED"
        elif cid=="Y-CLM-HEAD-ORGANIZATION":
            b2=hyp["B2"];decision="NOT_SUPPORTED" if b2["ci_lower"]>=-.5 and b2["ci_upper"]<=.5 and hyp["B6"]["ci_upper"]<=hyp["B6"]["threshold"] and hyp["B7"]["ci_upper"]<=hyp["B7"]["threshold"] else "INCONCLUSIVE"
        elif cid=="Y-CLM-BRANCH-ALLOCATION":decision="NOT_SUPPORTED" if all(hyp[i]["ci_lower"]>=-hyp[i]["threshold"] and hyp[i]["ci_upper"]<=hyp[i]["threshold"] for i in ids) else "INCONCLUSIVE"
        elif cid=="Y-CLM-PERTURBATION-MODULATION":decision="NOT_SUPPORTED" if all(hyp[i]["ci_upper"]<=hyp[i]["threshold"] for i in ids) else "INCONCLUSIVE"
        else:decision=states[0]
        out.append({"claim_id":cid,"decision":decision,"hypotheses":list(ids),"n":n,"allowed_wording":allowed[cid],"reason":"frozen claim truth table applied to Holm-adjusted hypothesis outcomes"})
    return out

def array_digest(array:np.ndarray)->str:
    a=np.ascontiguousarray(array);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(b"\0");h.update(str(tuple(a.shape)).encode());h.update(b"\0");h.update(a.tobytes(order="C"));return h.hexdigest()
def write_npy(path:Path,value:np.ndarray)->None:
    path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:np.save(f,np.ascontiguousarray(value),allow_pickle=False);f.flush();os.fsync(f.fileno())
def write_csv(path:Path,rows:Sequence[Mapping[str,Any]],columns:Sequence[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True);require(not path.exists(),f"output exists: {path}");fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(columns),extrasaction="raise");w.writeheader()
        for row in rows:w.writerow({k:("" if row.get(k) is None else row.get(k)) for k in columns})
        f.flush();os.fsync(f.fileno())
def table_sidecar(repo:Path,path:Path,columns:Sequence[str],rows:Sequence[Mapping[str,Any]],ordering:Sequence[str])->dict[str,Any]:
    payload={"file":file_record(path,repo),"format":"CSV","columns":[{"name":c,"dtype":"string_or_float64_or_int"} for c in columns],"dtype":None,"shape":[len(rows),len(columns)],"axes":["row","column"],"row_ordering":list(ordering),"data_canonical_digest":canonical([{k:r.get(k) for k in columns} for r in rows])};return payload
def npy_sidecar(repo:Path,path:Path,array:np.ndarray,axes:Sequence[str])->dict[str,Any]:
    return {"file":file_record(path,repo),"format":"NPY","columns":[],"dtype":str(array.dtype),"shape":list(array.shape),"axes":list(axes),"row_ordering":[],"data_canonical_digest":array_digest(array)}
def payload_inventory(root:Path,excluded:set[str],repo:Path)->dict[str,Any]:
    rows=[file_record(p,repo) for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in excluded];return {"root":root.relative_to(repo).as_posix(),"excluded":sorted(excluded),"files":rows,"file_count":len(rows),"total_bytes":sum(x["bytes"] for x in rows),"files_digest":canonical({"files":rows})}
def process_io()->dict[str,int]:
    rows={}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():k,v=line.split(":",1);rows[k]=int(v.strip())
    except OSError:pass
    return rows
def resource_snapshot(start_cpu:float,start_wall:float,start_io:Mapping[str,int],root:Path)->dict[str,Any]:
    now=process_io();files=[p for p in root.rglob("*") if p.is_file()];cpu=time.process_time()-start_cpu;wall=time.monotonic()-start_wall
    return {"cpu_seconds":cpu,"cpu_hours":cpu/3600,"process_wall_seconds":wall,"stage_wall_seconds":wall,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),"read_bytes":max(0,now.get("read_bytes",0)-start_io.get("read_bytes",0)),"written_bytes":max(0,now.get("write_bytes",0)-start_io.get("write_bytes",0)),"retained_bytes":sum(p.stat().st_size for p in files),"file_count":len(files),"gpu_hours":0,"cuda_access_count":0,"model_calls":0,"jvp_calls":0,"vjp_calls":0,"backward_calls":0,"download_bytes":0,"bootstrap_operation_count":BOOTSTRAP_REPLICATES*12,"sign_flip_operation_count":SIGN_REPLICATES*12,"cap_status":"NO_Y4_STAGE_SPECIFIC_NUMERICAL_CAP_WAS_FROZEN"}

def const_schema(value:Any)->dict[str,Any]:
    if isinstance(value,dict):return {"type":"object","additionalProperties":False,"properties":{k:const_schema(v) for k,v in value.items()},"required":list(value)}
    if isinstance(value,list):return {"type":"array","prefixItems":[const_schema(v) for v in value],"items":False,"minItems":len(value),"maxItems":len(value)}
    return {"const":value}
def write_schema_sources(repo:Path)->None:
    spec=read(repo/SPEC);values=((repo/SPEC_SCHEMA,{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":SPEC_SCHEMA.name,**const_schema(spec)}),(repo/ARTIFACT_SCHEMA,artifact_schema()))
    for path,value in values:
        if path.exists():require(read(path)==value,f"existing prospective schema drift: {path}",StartupError);continue
        path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(fd,"wb") as f:f.write(canonical_bytes(value)+b"\n");f.flush();os.fsync(f.fileno())
def schema_closed(value:Any)->bool:
    if isinstance(value,dict):
        if value.get("type")=="object" and value.get("additionalProperties") is not False:return False
        return all(schema_closed(x) for x in value.values())
    if isinstance(value,list):return all(schema_closed(x) for x in value)
    return True
def validate_schema_sources(repo:Path)->None:
    a=read(repo/ARTIFACT_SCHEMA);s=read(repo/SPEC_SCHEMA);require(a==artifact_schema(),"artifact schema serialization drift");require(schema_closed(a) and schema_closed(s),"open nested schema");Draft202012Validator.check_schema(a);Draft202012Validator.check_schema(s);Draft202012Validator(s).validate(read(repo/SPEC))

def remote_refs(repo:Path)->dict[str,str]:
    return {line.split()[1]:line.split()[0] for line in git(repo,"ls-remote","origin").splitlines()}
def stage_ref_snapshot(repo:Path)->list[dict[str,str]]:
    local={line.split()[0]:line.split()[1] for line in git(repo,"for-each-ref","--format=%(refname) %(objectname)","refs/archive/stage-y-*").splitlines()}
    remote={k:v for k,v in remote_refs(repo).items() if k.startswith("refs/archive/stage-y-")}
    return [{"scope":scope,"ref":ref,"sha":sha1} for scope,rows in (("local",local),("remote",remote)) for ref,sha1 in sorted(rows.items())]
def verify_expected_refs(repo:Path)->list[dict[str,str]]:
    rem=remote_refs(repo)
    for ref,commit in EXPECTED_REFS.items():require(git(repo,"rev-parse",ref)==commit and rem.get(ref)==commit,f"immutable ref drift: {ref}",StartupError)
    wrong="refs/archive/stage-y-y2-r2-r1-clean-cohort-evidence-completion";require(wrong not in rem and not git(repo,"for-each-ref","--format=%(refname)",wrong),"old erroneous PASS ref exists",StartupError)
    return stage_ref_snapshot(repo)
def verify_git_file(repo:Path,path:Path,commit:str)->dict[str,Any]:
    top=Path(git(repo,"rev-parse","--show-toplevel"));rel=(repo/path).resolve().relative_to(top.resolve());blob=subprocess.check_output(["git","show",f"{commit}:{rel.as_posix()}"],cwd=repo);row=file_record(repo/path,repo)
    if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        lines=blob.decode().splitlines();require(lines[1].endswith(row["sha256"]) and int(lines[2].split()[1])==row["bytes"],f"LFS identity drift: {path}")
    else:require(hashlib.sha256(blob).hexdigest()==row["sha256"],f"Git blob drift: {path}")
    return row
def verify_json_identity(repo:Path,path:Path,raw:str,canon:str,commit:str|None=None)->dict[str,Any]:
    require(sha(repo/path)==raw,f"raw SHA drift: {path}");d=read(repo/path);u=dict(d);claim=u.pop("canonical_digest",None);require(claim==canon==canonical(u),f"canonical drift: {path}");return verify_git_file(repo,path,commit) if commit else file_record(repo/path,repo)
def contract_bindings(repo:Path)->list[dict[str,Any]]:
    rows=[]
    for p in CONTRACT_FILES:rows.append(verify_json_identity(repo,p,EXPECTED_RAW_SHA[str(p)],EXPECTED_CANONICAL[str(p)]))
    rows.append(verify_json_identity(repo,Y0_COMPLETION,*EXPECTED_COMPLETIONS[str(Y0_COMPLETION)],EXPECTED_REFS["refs/archive/stage-y-y0-protocol-freeze"]))
    return rows
def publication_binding(repo:Path,stage:str)->dict[str,Any]:
    if stage=="Y2":path=Y2_COMPLETION;manifest=Y2_MANIFEST;commit=EXPECTED_REFS["refs/archive/stage-y-y2-r2-r2-clean-cohort-evidence-completion"]
    else:path=Y3_COMPLETION;manifest=Y3_MANIFEST;commit=EXPECTED_REFS["refs/archive/stage-y-y3-r6-perturbation-mechanism-evidence"]
    completion=verify_json_identity(repo,path,*EXPECTED_COMPLETIONS[str(path)],commit);m=verify_git_file(repo,manifest,commit);d=read(repo/path);require(d["status"]=="PASS",f"{stage} completion not PASS")
    if stage=="Y2":require(d["payload"]["denominator"]==d["payload"]["valid_clean_count"]==100,"Y2 denominator drift")
    else:
        rec=read(repo/Y3_ROOT/"reconciliation.json")["payload"];require(d["payload"]["denominator_images"]==20 and d["payload"]["terminal_count"]==160 and d["payload"]["complete_image_count"]==20 and rec["valid_count"]==160 and rec["prediction_flip_count"]==2,"Y3 completion counts drift")
    return {"name":stage,"files":[completion,m],"files_digest":canonical([completion,m]),"commit":commit}
def startup(repo:Path,allow_sources:bool=False)->dict[str,Any]:
    runtime_guard();require(os.path.realpath(sys.executable)==os.path.realpath(PYTHON),"Python executable drift",StartupError);snapshot=verify_expected_refs(repo);rem=remote_refs(repo);head=git(repo,"rev-parse","HEAD");require(git(repo,"symbolic-ref","--short","HEAD")==BRANCH and head==git(repo,"rev-parse","@{upstream}")==rem.get("refs/heads/"+BRANCH)==STARTUP_COMMIT,"startup branch identity drift",StartupError)
    status=git(repo,"status","--porcelain");changed=[line[3:] for line in status.splitlines()];require(not changed or allow_sources and all(Path(p.removeprefix(repo.name+"/")) in PRODUCER_FILES for p in changed),"unexpected worktree change",StartupError);lfs=git(repo,"lfs","status");require("(LFS:" not in lfs and "(Git:" not in lfs,"pending LFS state",StartupError)
    for ref in (PRODUCER_REF,PASS_REF,BLOCKED_REF):require(ref not in rem and not git(repo,"for-each-ref","--format=%(refname)",ref),f"fresh ref exists: {ref}",StartupError)
    require(not (repo/ROOT).exists() and not (repo/REPORT_ROOT).exists(),"fresh Y4-R1 root exists",StartupError);require(not any(p.name.lower().startswith(("y5","stage_y_y5")) for parent in (repo/"evidence/stage_y",repo/"reports") for p in parent.iterdir()),"Y5 already started",StartupError)
    old=repo/"evidence/stage_y/y4_mechanism_analysis_20260914/formal/completion/y4.json";require(old.is_file() and sha(old)=="39e0a085f87b9a11f32001e7f0e06633cfc4397e90f234f8d645a0d5c8306c37","old blocked completion drift",StartupError)
    bindings=contract_bindings(repo);y2=publication_binding(repo,"Y2");y3=publication_binding(repo,"Y3");return {"snapshot":snapshot,"startup_refs_digest":canonical(snapshot),"contracts":bindings,"y2":y2,"y3":y3}
def producer_files(repo:Path)->list[dict[str,Any]]:return [file_record(repo/p,repo) for p in PRODUCER_FILES+CONTRACT_FILES+DEPENDENCIES]
def verify_implementation(repo:Path)->str:
    seal=read(repo/ROOT/"producer_implementation_seal.json");validate_document(seal);q=seal["payload"];require(git(repo,"rev-parse",PRODUCER_REF)==q["producer_commit"],"producer ref drift");rows=producer_files(repo);require(rows==q["source_files"] and canonical({"files":rows})==q["files_digest"],"producer files drift")
    top=Path(git(repo,"rev-parse","--show-toplevel"))
    for row in rows:
        rel=(repo/row["path"]).relative_to(top);blob=subprocess.check_output(["git","show",f"{q['producer_commit']}:{rel.as_posix()}"],cwd=repo);require(hashlib.sha256(blob).hexdigest()==row["sha256"],"producer Git blob drift: "+row["path"])
    return q["producer_commit"]
def seal_producer(repo:Path,preflight:Mapping[str,Any])->dict[str,Any]:
    runtime_guard();head=git(repo,"rev-parse","HEAD");rem=remote_refs(repo);require(head==git(repo,"rev-parse","@{upstream}")==rem.get("refs/heads/"+BRANCH)==git(repo,"rev-parse",PRODUCER_REF)==rem.get(PRODUCER_REF),"producer publication incomplete",StartupError);require(not git(repo,"status","--porcelain"),"producer worktree dirty",StartupError);validate_schema_sources(repo);history=verify_expected_refs(repo);require(canonical(history)==preflight["startup_refs_digest"],"historical ref snapshot changed",StartupError)
    root=repo/ROOT;require(not root.exists(),"Y4-R1 root exists",StartupError);write(root/"completion/y4_r1.pending.json","pending",{"started_unix_ns":time.time_ns(),"next_authorization":"NONE_WHILE_PENDING"},"PENDING");files=producer_files(repo);write(root/"producer_implementation_seal.json","implementation_seal",{"producer_commit":head,"producer_ref":PRODUCER_REF,"source_files":files,"files_digest":canonical({"files":files})});write(root/"producer_preflight.json","producer_preflight",preflight);write(root/"startup_ref_snapshot.json","binding",{"name":"startup_stage_y_refs","files":[],"files_digest":preflight["startup_refs_digest"],"commit":STARTUP_COMMIT})
    contracts=contract_bindings(repo);write(root/"y0_contract_bundle_binding.json","binding",{"name":"Y0 contract bundle","files":contracts,"files_digest":canonical(contracts),"commit":EXPECTED_REFS["refs/archive/stage-y-y0-protocol-freeze"]});write(root/"y2_completion_publication_binding.json","binding",publication_binding(repo,"Y2"));write(root/"y3_completion_publication_binding.json","binding",publication_binding(repo,"Y3"));write(root/"prospective_protocol_binding.json","binding",{"name":"Y4-R1 prospective protocol","files":[file_record(repo/DOC,repo),file_record(repo/SPEC,repo)],"files_digest":canonical([file_record(repo/DOC,repo),file_record(repo/SPEC,repo)]),"commit":head});verify_implementation(repo);return {"status":"PASS","producer_commit":head}

CORE_NAMES=tuple(sorted(["input_attribution","input_cell_map","post_embedding_matrix","post_embedding_cls","post_embedding_cell","final_norm_matrix","final_norm_cls","final_norm_cell","classifier_input_cls","classifier_input_cell"]+[f"b{b:02d}_{s}" for b in BLOCKS for s in ("first_identity_matrix","first_identity_cls","first_attention_matrix","first_attention_cls","second_residual_matrix","second_residual_cls","second_mlp_matrix","second_mlp_cls","input_matrix","input_cls","output_matrix","output_cls","first_heads_matrix","first_heads_cls","input_cell","first_identity_cell","first_heads_cell","first_attention_cell","second_residual_cell","second_mlp_cell","output_cell")]))
def expected_array(name:str)->tuple[tuple[int,...],str,list[str]]:
    if name in ("valid_mask",):return (224,224),"bool",["pixel_y","pixel_x"]
    if name=="fully_valid_patch_mask":return (196,),"bool",["clean_patch"]
    if name=="patch_inverse_operator":return (196,196),"float32",["clean_patch","transformed_patch"]
    if name in ("positive_input_map","negative_input_map"):return (224,224),"float32",["pixel_y","pixel_x"]
    if name.startswith("raw_attention_b"):return (12,197,197),"float32",["head","query_token","key_token"]
    base=name.split("_",1)[1] if name.startswith(("positive_","negative_")) else name
    require(base in CORE_NAMES,f"unknown array name: {name}")
    if base=="input_attribution":shape,axes=(3,224,224),["rgb","pixel_y","pixel_x"]
    elif base=="input_cell_map" or base.endswith("_cell"):shape,axes=(56,56),["cell_y","cell_x"]
    elif base.endswith("_cls"):shape,axes=(196,),["source_patch"]
    else:shape,axes=(196,196),["source_patch","destination_patch"]
    if "_first_heads_" in base:shape=(12,)+shape;axes=["head"]+axes
    return shape,"float32",axes
def checked_record(repo:Path,row:Mapping[str,Any],roots:Sequence[Path])->Path:
    p=repo/str(row["path"]);require(p.is_file() and not p.is_symlink(),f"missing source: {p}");require(any(p.resolve().is_relative_to((repo/r).resolve()) for r in roots),f"source outside whitelist: {p}",AccessError);actual=file_record(p,repo);require(all(actual.get(k)==v for k,v in row.items() if k in actual),f"file record drift: {p}");ACCESS_COUNTS["scientific_files_read"]+=1;ACCESS_COUNTS["scientific_bytes_read"]+=p.stat().st_size;return p
def validate_array_record(repo:Path,row:Mapping[str,Any],roots:Sequence[Path])->np.ndarray:
    p=checked_record(repo,row,roots);shape,dtype,axes=expected_array(p.stem);a=np.load(p,allow_pickle=False);observed="bool" if a.dtype==np.bool_ else "float32" if a.dtype==np.dtype("<f4") else str(a.dtype);require(tuple(row["shape"])==a.shape==shape and row["dtype"]==observed==dtype and row["axes"]==axes,f"array contract drift: {p}",ScientificError);require(a.flags.c_contiguous and (a.dtype==np.bool_ or np.isfinite(a).all()),f"array layout/nonfinite: {p}",ScientificError);require(row["canonical_digest"]==array_digest(a),f"array canonical digest drift: {p}",ScientificError);return a
def closure_check(arrays:Mapping[str,np.ndarray],delta:float)->dict[str,float]:
    attr=arrays["input_attribution"].astype(np.float64);cells=arrays["input_cell_map"].astype(np.float64);patch=attr.sum(0).reshape(14,16,14,16).sum((1,3)).reshape(196);input_error=abs(float(attr.sum())-delta);total_threshold=2e-4+.01*abs(delta);require(input_error<=1e-4+.005*abs(delta),"input completeness failure",ScientificError);maximum=0.0
    def cut(matrix,cls,cell):
        nonlocal maximum
        errors=(float(np.max(np.abs(matrix.astype(np.float64).sum(1)+cls.astype(np.float64)-patch))),float(np.max(np.abs(cell.astype(np.float64)-cells))),abs(float(matrix.astype(np.float64).sum()+cls.astype(np.float64).sum())-delta),abs(float(cell.astype(np.float64).sum()-matrix.astype(np.float64).sum()-cls.astype(np.float64).sum())));maximum=max(maximum,*errors);require(errors[2]<=total_threshold and errors[3]<=total_threshold,"cut scalar closure failure",ScientificError)
    cut(arrays["post_embedding_matrix"],arrays["post_embedding_cls"],arrays["post_embedding_cell"])
    for b in BLOCKS:
        p=f"b{b:02d}";cut(arrays[p+"_input_matrix"],arrays[p+"_input_cls"],arrays[p+"_input_cell"]);cut(arrays[p+"_output_matrix"],arrays[p+"_output_cls"],arrays[p+"_output_cell"]);require(max(float(np.max(np.abs(arrays[p+"_first_heads_matrix"].astype(np.float64).sum(0)-arrays[p+"_first_attention_matrix"].astype(np.float64)))),float(np.max(np.abs(arrays[p+"_first_heads_cls"].astype(np.float64).sum(0)-arrays[p+"_first_attention_cls"].astype(np.float64)))),float(np.max(np.abs(arrays[p+"_first_heads_cell"].astype(np.float64).sum(0)-arrays[p+"_first_attention_cell"].astype(np.float64)))))<=total_threshold,f"head closure failure block {b}",ScientificError)
        first_matrix=arrays[p+"_first_identity_matrix"].astype(np.float64)+arrays[p+"_first_attention_matrix"].astype(np.float64);first_cls=arrays[p+"_first_identity_cls"].astype(np.float64)+arrays[p+"_first_attention_cls"].astype(np.float64);first_cell=arrays[p+"_first_identity_cell"].astype(np.float64)+arrays[p+"_first_attention_cell"].astype(np.float64);second_matrix=arrays[p+"_second_residual_matrix"].astype(np.float64)+arrays[p+"_second_mlp_matrix"].astype(np.float64);second_cls=arrays[p+"_second_residual_cls"].astype(np.float64)+arrays[p+"_second_mlp_cls"].astype(np.float64);second_cell=arrays[p+"_second_residual_cell"].astype(np.float64)+arrays[p+"_second_mlp_cell"].astype(np.float64);cut(first_matrix,first_cls,first_cell);cut(second_matrix,second_cls,second_cell);require(float(np.max(np.abs(first_cell-cells)))<=total_threshold,f"first residual cell closure failure block {b}",ScientificError);require(float(np.max(np.abs(second_cell-cells)))<=total_threshold,f"second residual cell closure failure block {b}",ScientificError)
    cut(arrays["final_norm_matrix"],arrays["final_norm_cls"],arrays["final_norm_cell"]);return {"input_abs_error":input_error,"maximum_scalar_error":maximum,"total_threshold":total_threshold}
def load_unit(repo:Path,terminal_record:Mapping[str,Any],roots:Sequence[Path],science_record:Mapping[str,Any]|None=None)->tuple[dict[str,np.ndarray],Mapping[str,Any]|None,Mapping[str,Any],list[dict[str,Any]]]:
    tp=checked_record(repo,terminal_record,roots);t=read(tp);q=t["payload"];require(t["status"] in ("VALID","BLOCKED_BUDGET"),"source terminal not usable");records=q["array_records"];require({Path(x["path"]).stem for x in records}.issuperset(CORE_NAMES),"core array roster incomplete",ScientificError);arrays={};valid=[]
    for row in records:
        a=validate_array_record(repo,row,roots);valid.append({"source_terminal":terminal_record["path"],"name":Path(row["path"]).stem,**row});
        if Path(row["path"]).stem in set(CORE_NAMES)|{"valid_mask","fully_valid_patch_mask","patch_inverse_operator"}:arrays[Path(row["path"]).stem]=a
    for attempt in q.get("attempt_records",[]):checked_record(repo,attempt,roots)
    for key in ("array_inventory","resource_record","transform_record","prediction_record"):
        if q.get(key) is not None:checked_record(repo,q[key],roots)
    sr=science_record or q.get("scientific_result") or q.get("scientific_record");science=None
    if sr is not None:science=read(checked_record(repo,sr,roots))["payload"]
    delta=float((science["y2_compatible"] if science and "y2_compatible" in science else science)["delta"]) if science else float(q["quadrature_attempts"][-1]["input_delta"]);closure_check(arrays,delta);return arrays,science,q,valid
