#!/usr/bin/env python3
"""Seal N2-V2 I3 at the contract-defined independent-reviewer stop."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'evidence/n2_v2_i3_20260922'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for c in iter(lambda:s.read(1048576),b''): h.update(c)
 return h.hexdigest()
def main():
 p=OUT/'terminal_receipt.json'
 if p.exists(): raise RuntimeError('I3_TERMINAL_EXISTS')
 v=json.loads((OUT/'validation.json').read_text())
 if v['status']!='PASS_REVIEW_PREPARATION_VALIDATION': raise RuntimeError('VALIDATION_NOT_PASS')
 bindings={name:sha(path) for name,path in {'protocol_sha256':ROOT/'N2-V2输入可识别性恢复协议.md','contract_sha256':ROOT/'config/n2_v2/input_identifiability_contract.json','legacy_lineage_ledger_sha256':ROOT/'manifests/n2_v2/legacy_lineage_exclusions.json','scene_candidate_registry_sha256':OUT/'private/scene_candidate_registry.json','recipient_candidate_registry_sha256':OUT/'private/recipient_candidate_registry.json','mask_archive_bindings_sha256':OUT/'private/openimages_mask_archive_bindings.json','review_card_manifest_sha256':OUT/'review/review_card_manifest.json','reviewer_a_template_sha256':OUT/'review/templates/reviewer_A.json','reviewer_b_template_sha256':OUT/'review/templates/reviewer_B.json','review_instructions_sha256':OUT/'review/README.md','preparation_receipt_sha256':OUT/'preparation_receipt.json','validation_sha256':OUT/'validation.json','validator_sha256':ROOT/'validate_n2_v2_i3.py'}.items()}
 receipt={'schema_version':1,'study_version':'N2_V2','stage_id':'N2_V2_INPUT_IDENTIFIABILITY','attempt_id':'N2_V2_I3_20260922','terminal':True,'status':'BLOCKED_INDEPENDENT_REVIEWERS','reason':'Candidate assets and blinded review material are prepared; two independent human review bundles remain required.','gate_results':{'legacy_n2_closure':'PASS','v2_protocol_freeze':'PASS','legacy_lineage_exclusion':'PASS','scene_candidate_preparation':'PASS','recipient_candidate_preparation':'PASS','independent_reviewers':'BLOCKED_ZERO_OF_TWO_SEALED','blind_review':'NOT_RUN_PRECEDING_BLOCK','accepted_scene_only_registry':'NOT_CREATED_BEFORE_REVIEW','low_level_matching':'NOT_RUN_PRECEDING_BLOCK','identifiability_freeze':'NOT_ISSUED','behavior_gate':'NOT_AUTHORIZED','n3':'NOT_AUTHORIZED'},'access_accounting':{'scene_image_payload_reads_for_preparation':102,'recipient_image_payload_reads_for_preparation':12,'recipient_mask_payload_reads_for_preparation':12,'reviewer_bundles_sealed':0,'checkpoint_opens':0,'checkpoint_deserializations':0,'model_loads':0,'inference_runs':0,'behavior_outcome_reads':0,'activation_extractions':0,'endpoint_runs':0,'path_nns_runs':0,'jvp_runs':0,'intervention_runs':0,'gpu_uses':0},'artifact_bindings':bindings,'next_gate':'STOP_I3_WAIT_FOR_TWO_INDEPENDENT_HUMAN_REVIEW_BUNDLES_NEW_FRESH_ROOT_REQUIRED_AFTER_REVIEWS'}
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'w') as s: json.dump(receipt,s,indent=2,sort_keys=True,ensure_ascii=False); s.write('\n'); s.flush(); os.fsync(s.fileno())
 print(json.dumps({'status':receipt['status'],'bindings':len(bindings),'model_accesses':0},sort_keys=True))
if __name__=='__main__':main()
