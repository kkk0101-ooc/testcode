#!/usr/bin/env python3
"""Freeze a fresh E1 cohort with ImageNet WordNet-metadata matching before inference."""
import hashlib, io, json, os, re, tarfile
from collections import Counter, defaultdict, deque
from pathlib import Path
import numpy as np
import scipy.io

R=Path(__file__).resolve().parents[1]
OLD=R/'manifests/nns_vit_revision_v5'
OUT=R/'manifests/nns_vit_revision_v5_e1_r2'
NS='NNS_VIT_REVISION_V5_E1_R2_20260921'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rank(kind,*values):return hashlib.sha256((NS+'|'+kind+'|'+'|'.join(map(str,values))).encode()).hexdigest()
def write(name,value):
 path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

assert not OUT.exists(),'FROZEN_ROOT_EXISTS'
base=json.loads((OLD/'cohorts.json').read_text());excluded=set(base['exclusions'])|{r['sample_id'] for r in base['rows']}
labels=(R/'assets/imagenet/imagenet_2012_validation_synset_labels.txt').read_text().splitlines();assert len(labels)==50000 and set(Counter(labels).values())=={50}
all_wnids=sorted(set(labels));candidates=defaultdict(list)
for i,w in enumerate(labels,1):
 name=f'ILSVRC2012_val_{i:08d}.JPEG'
 if name not in excluded:candidates[w].append(name)
classes=sorted((w for w in all_wnids if len(candidates[w])>=5),key=lambda w:(rank('class',w),w))[:100];assert len(classes)==100
selected=[]
for w in classes:
 for order,name in enumerate(sorted(candidates[w],key=lambda x:(rank('image',x),x))[:5]):selected.append(dict(sample_id=name,wnid=w,class_id=all_wnids.index(w),cohort='e1',within_class=order,deep_study=False))
assert len({r['sample_id'] for r in selected})==500 and not ({r['sample_id'] for r in selected}&excluded)

devkit=R/'assets/imagenet/ILSVRC2012_devkit_t12.tar.gz'
with tarfile.open(devkit) as archive:data=archive.extractfile('ILSVRC2012_devkit_t12/data/meta.mat').read()
nodes=scipy.io.loadmat(io.BytesIO(data),squeeze_me=True,struct_as_record=False)['synsets'];bywnid={str(x.WNID):int(x.ILSVRC2012_ID) for x in nodes};graph=defaultdict(set)
for node in nodes:
 for child in np.atleast_1d(node.children):graph[int(node.ILSVRC2012_ID)].add(int(child));graph[int(child)].add(int(node.ILSVRC2012_ID))
def distance(a,b):
 q=deque([(bywnid[a],0)]);seen={bywnid[a]};target=bywnid[b]
 while q:
  node,d=q.popleft()
  if node==target:return d
  for nxt in graph[node]:
   if nxt not in seen:seen.add(nxt);q.append((nxt,d+1))
 raise RuntimeError('DISCONNECTED_WORDNET')
dist={(a,b):distance(a,b) for a in classes for b in classes if a!=b}
matching=[dict(wnid=a,matched_wnid=min((b for b in classes if b!=a),key=lambda b:(dist[a,b],rank('match_tie',a,b))),wordnet_distance=min(dist[a,b] for b in classes if b!=a)) for a in classes]

old_protocol=json.loads((OLD/'protocol.json').read_text());models=[m for m in old_protocol['models'] if m['role']=='original']
write('cohorts.json',dict(namespace=NS,selection_uses_outcomes=False,selected_image_bytes_opened=False,exclusion_source=str((OLD/'cohorts.json').relative_to(R)),excluded_count=len(excluded),excluded_sha256=hashlib.sha256(('\n'.join(sorted(excluded))+'\n').encode()).hexdigest(),rows=selected))
write('matching.json',dict(namespace=NS,source='ILSVRC2012_devkit_t12/data/meta.mat WordNet hierarchy',source_tar_sha256=sha(devkit),selection_uses_model_outputs=False,rows=matching))
write('targets.json',json.loads((OLD/'targets.json').read_text()))
protocol=dict(namespace=NS,scope='prospective E1 metadata-matched replication',models=models,representative_blocks=[0,3,7,11],bootstrap_repeats=10000,matching='frozen shortest undirected WordNet hierarchy distance, SHA256 tie break',primary='same-class minus metadata-matched different-class JS similarity, class-level inference',cohorts_sha256=sha(OUT/'cohorts.json'),matching_sha256=sha(OUT/'matching.json'),targets_sha256=sha(OUT/'targets.json'))
write('protocol.json',protocol)

cache=R/'assets/data/revision_v5_e1_r2';cache.mkdir(parents=True,exist_ok=True);wanted={r['sample_id']:r for r in selected};bound=[]
with tarfile.open(R/'assets/imagenet/ILSVRC2012_img_val.tar') as archive:
 for member in archive:
  name=Path(member.name).name
  if name not in wanted:continue
  payload=archive.extractfile(member).read();path=cache/name
  if not path.exists():path.write_bytes(payload)
  assert path.read_bytes()==payload
  bound.append(dict(**wanted[name],cache_path=str(path.relative_to(R)),image_sha256=hashlib.sha256(payload).hexdigest(),image_bytes=len(payload)))
assert len(bound)==500
write('image_bindings.json',dict(protocol_sha256=sha(OUT/'protocol.json'),rows=sorted(bound,key=lambda x:x['sample_id'])))
print(json.dumps({'status':'FROZEN_AND_MATERIALIZED','images':500,'classes':100,'excluded':len(excluded),'overlap':0,'models':len(models)}))
