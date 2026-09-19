#!/usr/bin/env python3
"""Independent prompt-to-artifact audit before the immutable M3 seal."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"reports/nns_vit_endpoint_m3_r1_20260919"
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(x,c):
 if not x:raise ValueError(c)
def main():
 task=load(ROOT/"task_validation.json");analysis=load(ROOT/"analysis_receipt.json");independent=load(ROOT/"independent_analysis_audit.json");documents=load(ROOT/"document_validation.json");scientific=load(ROOT/"scientific_decision.json");failure=load(ROOT/"initial_numeric_validation_failure.json");recovery=load(ROOT/"m3b_float64_recovery.json");figures=load(ROOT/"figure_manifest.json")
 require(task["tasks"]==200 and task["targets"]==1228800 and task["full_maps"]==120,"PRODUCTION_COUNTS");require(task["numeric_status"]=="PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE","NUMERIC_STATUS");require(failure["failed_condition_count"]==36 and failure["failed_task_count"]==10 and recovery["selected_task_ids"]==failure["failed_task_ids"],"RECOVERY_SCOPE");require(all(x["status"]=="PASS" for x in (analysis,independent,documents)),"POSTPROCESS_AUDITS");require(scientific["nns_endpoint_equivalence"]==scientific["vit_operator_jacobians"]=="SUPPORTED" and scientific["joint_local_to_global"]=={"vit_b16":True,"deit_s16":True},"SCIENTIFIC_DECISION");require(len(figures["figures"])==2 and all(len(x["files"])==len(x["paper_files"])==2 for x in figures["figures"]),"FIGURE_PACKAGE")
 test_rows=load(ROOT/"atlas_tests.json")["tests"];require(len(test_rows)==10 and all(x["reject"] for x in test_rows),"ATLAS_TESTS");require({x["contrast"] for x in test_rows}=={"late_minus_early_entropy","late_minus_early_locality","late_minus_early_distance","late_minus_early_cancellation","cls_minus_patch_entropy"},"TEST_FAMILY")
 required=["cohort.json","target_registry.json","task_manifest.json","m3a_fidelity.json","m3a_operator_validation.json","task_validation.json","image_values.csv","atlas_summary.csv","atlas_tests.json","similarity_image_values.csv","similarity_phase_summary.csv","full_map_manifest.json","figure_manifest.json","scientific_decision.json"]
 for name in required:require((ROOT/name).is_file() and (ROOT/name).stat().st_size>0,"ARTIFACT:"+name)
 external=["docs/endpoint_m3_results_20260919.md","papers/nns_vit_endpoint/zh/endpoint_m3_supplement.md","papers/nns_vit_endpoint/en/endpoint_m3_supplement.md","papers/nns_vit_endpoint/notes/endpoint_m3_evidence_matrix.md","papers/nns_vit_endpoint/qc/endpoint_m3_writing_audit.md","papers/nns_vit_endpoint/figure_manifest_m3.json"]
 for name in external:require((REPO/name).is_file() and (REPO/name).stat().st_size>0,"EXTERNAL:"+name)
 report={"status":"PASS","objective":"Execute complete Stage M M3 and stop after M3","production_counts":{"tasks":200,"targets":1228800,"full_maps":120},"m3a":{"fidelity_comparisons":5000,"operator_direction_records":88},"numeric":{"initial_float32_failures":36,"affected_tasks":10,"recovery_status":recovery["status"]},"analysis":{"image_records":200,"holm_tests":10,"independent_source_recompute":independent["status"]},"scientific_status":scientific["status"],"joint_local_to_global":scientific["joint_local_to_global"],"required_artifact_count":len(required),"external_artifact_count":len(external),"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()}
 (ROOT/"additional_completion_validation.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
if __name__=="__main__":main()
