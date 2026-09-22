#!/usr/bin/env python3
"""Independent validator for the N2-V2 I3 human-review handoff."""
from __future__ import annotations
import hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'evidence/n2_v2_i3_20260922'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for c in iter(lambda:s.read(1048576),b''):h.update(c)
 return h.hexdigest()
def ck(value,code,checks):
 if not value: raise RuntimeError(code)
 checks.append(code)
def main():
 checks=[]; prep=json.loads((OUT/'preparation_receipt.json').read_text()); scenes=json.loads((OUT/'private/scene_candidate_registry.json').read_text()); rec=json.loads((OUT/'private/recipient_candidate_registry.json').read_text()); cards=json.loads((OUT/'review/review_card_manifest.json').read_text()); legacy=json.loads((ROOT/'manifests/n2_v2/legacy_lineage_exclusions.json').read_text()); archives=json.loads((OUT/'private/openimages_mask_archive_bindings.json').read_text())
 ck(prep['status']=='BLOCKED_INDEPENDENT_REVIEWERS','CONTRACT_STATUS_ENUM_VALID',checks); ck(prep['model_access_count']==prep['checkpoint_access_count']==prep['gpu_use_count']==0,'MODEL_AND_GPU_ACCESS_ZERO',checks)
 ck(len(scenes['records'])==len(cards['records'])==102,'SCENE_AND_CARD_COUNT_102',checks); ck(Counter(r['candidate_stratum'] for r in scenes['records'])==Counter({'AQUATIC_CANDIDATE_STRATUM':54,'NON_AQUATIC_CANDIDATE_STRATUM':48}),'SCENE_STRATUM_COUNTS',checks); ck(len(rec['records'])==12,'RECIPIENT_COUNT_12',checks)
 ck(len({r['source_group_id'] for r in scenes['records']})==102 and len({r['image_sha256'] for r in scenes['records']})==102,'SCENE_IDENTITIES_UNIQUE',checks); ck(len({r['source_group_id'] for r in rec['records']})==12 and len({r['image_sha256'] for r in rec['records']})==12,'RECIPIENT_IDENTITIES_UNIQUE',checks)
 old_ids={r['source_group_id'] for r in legacy['records']}; old_hashes={r['image_sha256'] for r in legacy['records']}; ck(not old_ids.intersection(r['source_group_id'] for r in scenes['records']+rec['records']),'NO_LEGACY_SOURCE_ID_OVERLAP',checks); ck(not old_hashes.intersection(r['image_sha256'] for r in scenes['records']+rec['records']),'NO_LEGACY_IMAGE_HASH_OVERLAP',checks)
 by_card={r['card_id']:r for r in cards['records']}; ck(len(by_card)==102,'BLIND_CARD_IDS_UNIQUE',checks)
 for r in scenes['records']:
  image=OUT/r['image_path']; card=OUT/r['review_image_path']; ck(image.is_file() and sha(image)==r['image_sha256'],'SCENE_BINDING_'+r['asset_id'],checks); ck(card.is_file() and sha(card)==r['review_image_sha256'] and by_card[r['card_id']]['image_sha256']==r['review_image_sha256'],'CARD_BINDING_'+r['card_id'],checks); ck(r['foreground_removal_applied'] is False and r['producer_operations']==[],'NO_SCENE_EDIT_'+r['asset_id'],checks); ck(bool(r['license_short_name'] and r['description_url'] and (r['artist'] or r['license_short_name'] in {'Public domain','PDM'})),'SCENE_LICENSE_'+r['asset_id'],checks)
 for r in rec['records']:
  image=OUT/r['image_path']; mask=OUT/r['mask_path']; ck(image.is_file() and sha(image)==r['image_sha256'],'RECIPIENT_IMAGE_'+r['recipient_id'],checks); ck(mask.is_file() and sha(mask)==r['mask_sha256'],'RECIPIENT_MASK_'+r['recipient_id'],checks); ck(r['target_model_class_wnid']=='n01443537' and r['target_model_class_index']==1 and r['class_alignment_qc']=='PASS_VISUAL_GOLDFISH_COMPATIBLE','RECIPIENT_CLASS_'+r['recipient_id'],checks); ck(r['license_url']=='https://creativecommons.org/licenses/by/2.0/','RECIPIENT_LICENSE_'+r['recipient_id'],checks)
 ck(archives['annotation_license']=='CC_BY_4_0' and len(archives['archives'])==7 and len({a['sha256'] for a in archives['archives']})==7,'MASK_ARCHIVE_LICENSE_AND_BINDINGS',checks)
 manifest_sha=sha(OUT/'review/review_card_manifest.json')
 for template in sorted((OUT/'review/templates').glob('*.json')):
  d=json.loads(template.read_text()); ck(d['review_card_manifest_sha256']==manifest_sha,'TEMPLATE_BINDING_'+template.stem,checks); ck(len(d['reviews'])==102 and {x['card_id'] for x in d['reviews']}==set(by_card),'TEMPLATE_CARD_SET_'+template.stem,checks)
 result={'schema_version':1,'study_version':'N2_V2','attempt_id':'N2_V2_I3_20260922','status':'PASS_REVIEW_PREPARATION_VALIDATION','check_count':len(checks),'checks':checks,'scene_candidate_count':102,'recipient_candidate_count':12,'review_card_count':102,'sealed_review_bundle_count':0,'model_access_count':0,'behavior_gate':'NOT_AUTHORIZED'}
 if '--write' in sys.argv:(OUT/'validation.json').write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
 print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)); return 0
if __name__=='__main__':raise SystemExit(main())
