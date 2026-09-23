#!/usr/bin/env python3
"""Read-only source audit for revision v6."""
import collections, datetime, hashlib, json, subprocess
from pathlib import Path
R = Path(__file__).resolve().parents[1]
P = R / 'papers/nns_vit_endpoint/publication_r2_20260920'
Q = P / 'qc/revision_v6'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def read(p): return json.loads((R/p).read_text())
def main():
 old=read('manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json')
 unit=read('reports/nns_vit_endpoint_m3_r1_20260919/cohort.json')
 base=R/'papers/nns_vit_endpoint/thesis_blind_review_20260921/evidence/revision_v5'
 v5=read(base/'cohorts.json')
 cohorts={k:[r for r in old['rows'] if r['split']==k] for k in ['discovery','calibration','evaluation']}
 cohorts['unit_study']=unit['images']
 cohorts.update({k:[r for r in v5['rows'] if r['cohort']==k] for k in ['e1','e5']})
 ids={k:{r['sample_id'] for r in rows} for k,rows in cohorts.items()}
 report=dict(schema_version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branch=subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=R,text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),cohorts=[dict(name=k,images=len(rows),classes=len({r['class_id'] for r in rows}),class_size_histogram=dict(collections.Counter(collections.Counter(r['class_id'] for r in rows).values()))) for k,rows in cohorts.items()],overlaps=[dict(left=k,right=l,shared_images=len(ids[k]&ids[l])) for k in ids for l in ids],bindings=[],source_failures=[],image_checks=[])
 files=[P/lang/name for lang in ['zh','en'] for name in ['manuscript.md','supplement.md','supplement_experiments_v1.md']]
 files += [R/'papers/nns_vit_endpoint/AGENTS.md',P/'references.md']
 files += list((P/'evidence').glob('gxi_step4_*'))+list((P/'qc').glob('gxi_step5_*'))
 files += [R/'src/vitnns'/name for name in ['endpoint_vit.py','endpoint_e3_closure.py','endpoint_e4_intervention.py','endpoint_e5_spatial.py']]
 files += [R/'docs'/name for name in ['NNS_ViT完整切面归因与匹配干预研究规划_20260917.md','NNS_ViT_P0_A_B_STATIC_PACKAGE执行提示词_20260917.md']]
 files += [base/name for name in ['protocol.json','cohorts.json','targets.json','image_bindings.json','analysis_estimands.json','e5_image_values.csv','e5_summary.csv','e5_receipt.json','package_receipt.json','frozen_revision_v5_plan.md']]
 for model in read(base/'protocol.json')['models']:
  p=R/model['checkpoint']; ok=p.is_file() and p.stat().st_size==model['checkpoint_bytes'] and sha(p)==model['checkpoint_sha256']
  files.append(p)
  if not ok:report['source_failures'].append(model['checkpoint'])
 for root in ['reports/nns_vit_endpoint_m3_r1_20260919','reports/nns_vit_endpoint_m4_r1_20260919']:
  tasks=read(root+'/task_manifest.json')['tasks']; seen=set()
  for t in tasks:
   row=t['payload'];p=R/row['cache_path']
   if p in seen:continue
   seen.add(p); observed=sha(p) if p.is_file() else None
   ok=p.is_file() and p.stat().st_size==row['image_bytes'] and observed==row['image_sha256']
   report['image_checks'].append(dict(path=row['cache_path'],sha256=observed,passed=ok))
   if not ok:report['source_failures'].append(row['cache_path'])
  files.append(R/root/'task_manifest.json')
 for p in files:report['bindings'].append(dict(path=str(p.relative_to(R)),sha256=sha(p),bytes=p.stat().st_size))
 report['source_failures']=sorted(set(report['source_failures']))
 report['status']='PASS_SOURCE_IDENTITY' if not report['source_failures'] else 'FAIL_SOURCE_IDENTITY'
 Q.mkdir(parents=True,exist_ok=True)
 (Q/'stage0_source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+chr(10))
 print(json.dumps({k:report[k] for k in ['status','cohorts','source_failures']},ensure_ascii=False))
 return bool(report['source_failures'])
if __name__=='__main__':raise SystemExit(main())
