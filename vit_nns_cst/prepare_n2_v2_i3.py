#!/usr/bin/env python3
"""Create the corrected N2-V2 I3 human-review handoff from sealed I2 inputs."""
from __future__ import annotations
import hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent; I2=ROOT/'evidence/n2_v2_i2_20260922'; I3=ROOT/'evidence/n2_v2_i3_20260922'
ARCHIVE_BINDINGS=[
 {'archive_name':'train-masks-0.zip','bytes':385547550,'sha256':'cba021d051afc1e40d0c9cc6f60d40d32a995743f7c3468887ac64d013072322'},
 {'archive_name':'train-masks-2.zip','bytes':285456237,'sha256':'491887a097166f607877dc25c5eeee88e2bc4e4fdf3d4e4dc8020c5469d7ce77'},
 {'archive_name':'train-masks-3.zip','bytes':282275385,'sha256':'9b42b8f3a43a7b231d114ad26ba87574a24a2ce79e06f157828b6dfdfb430396'},
 {'archive_name':'train-masks-4.zip','bytes':277869617,'sha256':'82c8b152733fb576ce88860c54569a3740ced4bd34862002cf77e08bac98ff4e'},
 {'archive_name':'train-masks-9.zip','bytes':272384343,'sha256':'7648b6842384d517286ce6e610a002543ab6851c31f7d3ff3af7ef39fe5ecba3'},
 {'archive_name':'train-masks-d.zip','bytes':273088985,'sha256':'f2e390be07cefdc65ff1d145da8ea9af0802d2676a169010b0bb841180a17a16'},
 {'archive_name':'train-masks-e.zip','bytes':239589233,'sha256':'67a47ac5874de762ff40c405316fde3ce787a4cd2afd65a3d0742743c23bb9fe'},
]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for c in iter(lambda:s.read(1048576),b''): h.update(c)
 return h.hexdigest()
def write(path,v): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 if I3.exists(): raise RuntimeError('I3_ALREADY_EXISTS')
 I3.mkdir(parents=True)
 for rel in ['private/scene_originals','private/recipient_images','private/recipient_masks','review']:
  shutil.copytree(I2/rel,I3/rel)
 scene=json.loads((I2/'private/scene_candidate_registry.json').read_text()); scene['attempt_id']='N2_V2_I3_20260922'; write(I3/'private/scene_candidate_registry.json',scene)
 recipients=json.loads((I2/'private/recipient_candidate_registry.json').read_text()); recipients['attempt_id']='N2_V2_I3_20260922'; write(I3/'private/recipient_candidate_registry.json',recipients)
 cards=json.loads((I2/'review/review_card_manifest.json').read_text()); cards['attempt_id']='N2_V2_I3_20260922'; write(I3/'review/review_card_manifest.json',cards)
 card_sha=sha(I3/'review/review_card_manifest.json')
 for p in (I3/'review/templates').glob('*.json'):
  d=json.loads(p.read_text()); d['review_card_manifest_sha256']=card_sha; write(p,d)
 write(I3/'private/openimages_mask_archive_bindings.json',{'schema_version':1,'source':'OPEN_IMAGES_V5_TRAIN_INSTANCE_SEGMENTATION','annotation_license':'CC_BY_4_0','annotation_license_source':'https://storage.googleapis.com/openimages/web/factsfigures.html','archives':[dict(x,source_url='https://storage.googleapis.com/openimages/v5/train-masks/'+x['archive_name']) for x in ARCHIVE_BINDINGS]})
 prep={'schema_version':1,'study_version':'N2_V2','stage_id':'N2_V2_INPUT_IDENTIFIABILITY','attempt_id':'N2_V2_I3_20260922','status':'BLOCKED_INDEPENDENT_REVIEWERS','reason':'All auditable candidate assets and two blinded templates are ready; the two assigned human reviewers have not yet completed or sealed their independent bundles.','predecessors':[{'attempt_id':'N2_V2_I1_20260922','status':'INVALID_RECIPIENT_CLASS_ALIGNMENT'},{'attempt_id':'N2_V2_I2_20260922','status':'INVALID_PROTOCOL_STATUS_ENUM'}],'scene_candidate_count':102,'candidate_stratum_counts':{'AQUATIC_CANDIDATE_STRATUM':54,'NON_AQUATIC_CANDIDATE_STRATUM':48},'recipient_candidate_count':12,'review_card_count':102,'accepted_scene_asset_count':0,'sealed_review_bundle_count':0,'blind_review_status':'NOT_RUN_AWAITING_HUMANS','matching_status':'NOT_RUN_PRECEDING_GATE','identifiability_freeze':'NOT_ISSUED','behavior_gate':'NOT_AUTHORIZED','model_access_count':0,'checkpoint_access_count':0,'gpu_use_count':0}
 write(I3/'preparation_receipt.json',prep); print(json.dumps(prep,sort_keys=True))
if __name__=='__main__': main()
