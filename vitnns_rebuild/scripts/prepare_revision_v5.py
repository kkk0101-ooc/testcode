#!/usr/bin/env python3
"""Freeze v5 metadata before materializing selected images; no model inference."""
import argparse,hashlib,json,os,re,tarfile
from collections import Counter,defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1]
M=R/'manifests/nns_vit_revision_v5'
NS='NNS_VIT_REVISION_V5_20260921_R1'
P=R/'papers/nns_vit_endpoint/thesis_blind_review_20260921/notes/revision_v5_plan.md'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rank(kind,x):return hashlib.sha256((NS+'|'+kind+'|'+str(x)).encode()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def freeze():
 assert not (M/'protocol.json').exists(),'FROZEN_EXISTS'
 old=R/'manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json'
 obj=json.loads(old.read_text());excluded=set(obj['excluded_union'])|{r['sample_id'] for r in obj['rows']}
 sources={str(old.relative_to(R)):sha(old)}
 for base in [R/'assets/data',R/'artifacts/nns_vit_endpoint',R/'evidence/nns_vit_endpoint']:
  if base.exists():
   for root,ds,fs in os.walk(base):excluded.update(f for f in fs if re.fullmatch(r'ILSVRC2012_val_\d{8}\.JPEG',f))
 labels=(R/'assets/imagenet/imagenet_2012_validation_synset_labels.txt').read_text().splitlines()
 assert len(labels)==50000 and set(Counter(labels).values())=={50}
 wnids=sorted(set(labels));byclass=defaultdict(list)
 for i,w in enumerate(labels,1):
  name=f'ILSVRC2012_val_{i:08d}.JPEG'
  if name not in excluded:byclass[w].append(name)
 classes=sorted([w for w in wnids if len(byclass[w])>=5],key=lambda w:rank('class_e1',w))[:100]
 rows=[]
 for w in classes:
  for i,n in enumerate(sorted(byclass[w],key=lambda n:rank('image_e1',n))[:5]):
   rows.append(dict(sample_id=n,wnid=w,class_id=wnids.index(w),cohort='e1',within_class=i,deep_study=i==0))
 taken=excluded|{r['sample_id'] for r in rows}
 eligible=[w for w in wnids if any(n not in taken for n in byclass[w])]
 e5classes=sorted(eligible,key=lambda w:rank('class_e5',w))[:400]
 for i,w in enumerate(e5classes):
  n=min((n for n in byclass[w] if n not in taken),key=lambda n:rank('image_e5',n))
  rows.append(dict(sample_id=n,wnid=w,class_id=wnids.index(w),cohort='e5',within_class=0,deep_study=False,prediction_split='train' if i<200 else 'validation' if i<300 else 'test'))
 assert len(rows)==900 and len({r['sample_id'] for r in rows})==900 and not taken.intersection(r['sample_id'] for r in rows if r['cohort']=='e5')
 oldmodels=json.loads((R/'manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json').read_text())['models']
 models=[]
 for m in oldmodels:
  models.append(dict(model_id=m['model_id'],timm_name=m['timm_name'],checkpoint=m['checkpoint']['path'],format='pth',embedding_dim=m['embedding_dim'],heads=m['heads'],mean=m['preprocessing']['mean'],std=m['preprocessing']['std'],role='original',source=m['timm_name']))
 for mid,name,path,dim in [('vit_b16_in1k','vit_base_patch16_224.augreg_in1k','assets/models/revision_v5/vit_base_patch16_224.augreg_in1k.npz',768),('vit_s16_augreg','vit_small_patch16_224.augreg_in21k_ft_in1k','assets/models/stage_x_x2/vit_small_patch16_224.augreg_in21k_ft_in1k.npz',384)]:
  models.append(dict(model_id=mid,timm_name=name,checkpoint=path,format='npz',embedding_dim=dim,heads=dim//64,mean=[.5]*3,std=[.5]*3,role='training_recipe_replication_not_seed',source='timm 0.9.16 pretrained cfg public Google AugReg archive'))
 targets={}
 for m in models:
  m['checkpoint_sha256']=sha(R/m['checkpoint']);m['checkpoint_bytes']=(R/m['checkpoint']).stat().st_size
  targets[m['model_id']]={}
  for b in range(12):
   groups={}
   for kind,dim in [('post',m['embedding_dim']),('mlp',4*m['embedding_dim'])]:
    # Stable dimension-matched sampling; 2 CLS plus 30 patch units, with no duplicate channels.
    channels=sorted(range(dim),key=lambda c:rank(f'{kind}|{b}|channel',c))[:32]
    tokens=[0,0]+[1+int(rank(f'{kind}|{b}|token',i)[:8],16)%196 for i in range(30)]
    groups[kind]=[dict(token=t,channel=c) for t,c in zip(tokens,channels)]
   groups['family_channels']=[r['channel'] for r in groups['mlp'][:8]]
   targets[m['model_id']][str(b)]=groups
 write(M/'cohorts.json',dict(namespace=NS,exclusions=sorted(excluded),sources=sources,selection_uses_outcomes=False,selected_image_bytes_opened=False,rows=rows))
 write(M/'targets.json',targets)
 write(M/'protocol.json',dict(namespace=NS,scope='prospective supplementary validation',plan=str(P.relative_to(R)),plan_sha256=sha(P),cohorts_sha256=sha(M/'cohorts.json'),targets_sha256=sha(M/'targets.json'),models=models,representative_blocks=[0,3,7,11],all_blocks=list(range(12)),alpha=[.1,1.],topk=[1,2,4,8],random_repeats=20,bootstrap_repeats=10000,preprocessing=dict(resize=248,crop=224,interpolation='bicubic',antialias=True,blur_kernel=31,blur_sigma=10.,decode='Pillow RGB no EXIF transpose'),closure=dict(atol=1e-5,rtol=1e-4),classification_policy='all correctness retained',decision_policy='negative utility does not imply implementation failure'))
 print(json.dumps(dict(status='FROZEN_METADATA',e1=500,e5=400,models=len(models),exclusions=len(excluded))))
def materialize():
 protocol=json.loads((M/'protocol.json').read_text())
 assert sha(M/'cohorts.json')==protocol['cohorts_sha256'] and sha(M/'targets.json')==protocol['targets_sha256']
 rows=json.loads((M/'cohorts.json').read_text())['rows'];wanted={r['sample_id']:r for r in rows};cache=R/'assets/data/revision_v5';cache.mkdir(parents=True,exist_ok=True)
 bound=[]
 with tarfile.open(R/'assets/imagenet/ILSVRC2012_img_val.tar') as archive:
  for member in archive:
   n=Path(member.name).name
   if n not in wanted:continue
   data=archive.extractfile(member).read();p=cache/n
   if not p.exists():p.write_bytes(data)
   assert p.read_bytes()==data
   bound.append(dict(**wanted[n],cache_path=str(p.relative_to(R)),image_sha256=hashlib.sha256(data).hexdigest(),image_bytes=len(data)))
 assert len(bound)==900
 write(M/'image_bindings.json',dict(protocol_sha256=sha(M/'protocol.json'),rows=sorted(bound,key=lambda r:r['sample_id'])))
 print('MATERIALIZED_900_HASH_BOUND_IMAGES')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','materialize']);args=a.parse_args();globals()[args.mode]()
