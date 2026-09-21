#!/usr/bin/env python3
"""Rebuild source-derived supplementary counts and figure bindings (stdlib only)."""
from pathlib import Path
import csv,json,hashlib,re,os
P=Path(__file__).resolve().parents[1];R=P.parents[2];D=P/'evidence/revision_v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def dump(name,rs):
 with (D/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 a=rows(R/'reports/nns_vit_endpoint_revision_v3_step2_r2_20260921/multi_amplitude_image_values.csv');out=[]
 for model,expected in [('vit_b16',144),('deit_s16',72)]:
  for ref in ['normalized_zero','blur_rgb']:
   for alpha in sorted({r['alpha'] for r in a},key=float):
    part=[r for r in a if r['model_id']==model and r['reference_id']==ref and r['alpha']==alpha]
    valid=sum(int(r['valid_relative_components']) for r in part);out.append(dict(model=model,reference=ref,alpha=float(alpha),total=len(part)*expected,valid=valid,missing=len(part)*expected-valid))
 dump('amplitude_missing_counts.csv',out)
 fam=json.loads((R/'reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json').read_text())
 fields=['test_id','bootstrap_seed','estimate','ci_lower','ci_upper','n_total','n_nonzero','holm_adjusted_p_value']
 dump('original_test_family.csv',[{k:r[k] for k in fields} for f in fam['families'] for r in f['test_results']])
 figures=[]
 for label,rel in re.findall(r'!\[([^]]*)\]\(([^)]+)\)',(P/'zh/manuscript.md').read_text()):
  p=(P/'zh'/rel).resolve();figures.append(dict(id=label,role='main',path=os.path.relpath(p,P),sha256=sha(p)))
 for p in sorted((P/'figures/revision_v4').glob('*')):figures.append(dict(id=p.stem,role='supplement',path=str(p.relative_to(P)),sha256=sha(p)))
 (P/'figure_manifest.json').write_text(json.dumps(dict(schema_version=1,analysis='post-hoc descriptive depth reanalysis and regional estimator sensitivity; inherited fixed evidence for other figures',display='image medians, pointwise 95% intervals, fixed 12 image trajectories; all 98 families with adjacent differences',figures=figures),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(count_tables=2,figure_entries=len(figures))))
if __name__=='__main__':main()
