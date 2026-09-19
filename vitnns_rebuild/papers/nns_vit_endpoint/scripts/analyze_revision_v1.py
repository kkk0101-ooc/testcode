#!/usr/bin/env python3
"""CPU, stdlib-only descriptive reanalysis; never alters sealed sources."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics as st
import sys

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'reports/nns_vit_endpoint_paper_revision_v1_20260920'
sys.path.insert(0, str(REPO / 'src'))
from vitnns.endpoint_m5_analysis import average_ranks, spearman, quantile
SOURCES = {}

def read(path, expected=None):
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    SOURCES[str(path.relative_to(REPO))] = dict(sha256=digest, bytes=len(raw))
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)

def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def csvwrite(name, rows):
    with (OUT/name).open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summary(values):
    valid = sorted(x for x in values if x is not None)
    return dict(n=len(valid),na=len(values)-len(valid),median=st.median(valid) if valid else None,
                q10=quantile(valid,.1),q90=quantile(valid,.9))

def auc(points):
    return math.fsum((y0+y1)*(x1-x0)/2 for (x0,y0),(x1,y1) in zip(points,points[1:]))/.25

def tau_b(a,b):
    c=d=ta=tb=0
    for i in range(len(a)):
        for j in range(i):
            da=(a[i]>a[j])-(a[i]<a[j]);db=(b[i]>b[j])-(b[i]<b[j])
            c+=da*db>0;d+=da*db<0;ta+=da==0 and db!=0;tb+=db==0 and da!=0
    den=math.sqrt((c+d+ta)*(c+d+tb))
    return (c-d)/den if den else None

def spatial():
    root=REPO/'reports/nns_vit_endpoint_e5_spatial_r1_20260918'
    manifest=read(root/'artifact_manifest.json')
    paths=[r for r in manifest['files'] if r['path'].startswith('formal_results/attempts/') and r['path'].endswith('.json')]
    assert len(paths)==800
    frozen=read(REPO/'reports/nns_vit_endpoint_e6_release_r3_20260918/per_image_statistics.json')
    frozen={(r['task_id'],r['reference_id']):r['c4b']['scores'] for r in frozen['rows']}
    rows=[];maximum_error=0.;source_difference=0.
    for entry in paths:
        r=read(root/entry['path'],entry['sha256'])['result']
        for ref,data in r['references'].items():
            per_head=defaultdict(list);envelopes=defaultdict(dict)
            for curve in data['curves']:
                points=[(p['actual_area_fraction'],abs(r['clean_score']-p['isolated_score'])) for p in curve['points']]
                value=auc(points)
                maximum_error=max(maximum_error,abs(value-curve['absolute_isolated_effect_auc_div_0_25']))
                per_head[curve['method'],curve['head']].append(value)
                for area,effect in points:
                    envelopes[curve['head']][area]=max(envelopes[curve['head']].get(area,0),effect)
            # Frozen E5 random baseline: mean signed effects BEFORE absolute
            # value/integration. Averaging absolute repeat AUC changes estimand.
            for head in r['heads']:
                curves=[c for c in data['curves'] if c['head']==head and c['method']=='random_regions']
                assert len(curves)==20
                points=[(p['actual_area_fraction'],abs(st.mean(r['clean_score']-c['points'][i]['isolated_score'] for c in curves)))
                        for i,p in enumerate(curves[0]['points'])]
                per_head['random_regions',head]=[auc(points)]
            methods=sorted({key[0] for key in per_head})
            scores={m:st.mean(st.mean(per_head[m,h]) for h in r['heads']) for m in methods}
            task=f"{r['model_id']}__{r['split_order']:03d}"
            source_difference=max(source_difference,max(abs(v-frozen[task,ref][m]) for m,v in scores.items()))
            oracle=st.mean(auc(sorted(envelopes[h].items())) for h in r['heads'])
            assert all(v<=oracle+1e-12 for v in scores.values())
            for method,value in {**scores,'empirical_oracle':oracle}.items():
                rows.append(dict(model=r['model_id'],reference=ref,sample_id=r['sample_id'],method=method,auc=value,
                    oracle_auc=oracle,clean_logit=r['clean_score'],normalized_regret=(oracle-value)/oracle if oracle else None,
                    relative_to_abs_clean_logit=value/abs(r['clean_score']) if r['clean_score'] else None))
    assert maximum_error<1e-12 and source_difference<1e-12, (maximum_error,source_difference)
    result=[]
    for model in ('vit_b16','deit_s16'):
        for ref in ('normalized_zero','blur_rgb'):
            for method in sorted({r['method'] for r in rows}):
                subset=[r for r in rows if (r['model'],r['reference'],r['method'])==(model,ref,method)]
                assert len(subset)==400 and len({r['sample_id'] for r in subset})==400
                result.append(dict(model=model,reference=ref,method=method,**{metric:summary([r[metric] for r in subset]) for metric in ('auc','normalized_regret','relative_to_abs_clean_logit','clean_logit')}))
    csvwrite('spatial_image_values.csv',rows)
    return dict(summary=result,curve_auc_max_error=maximum_error,e6_image_score_max_error=source_difference,source_tasks=len(paths))

def reference():
    root=REPO/'reports/nns_vit_endpoint_m1_r1_20260918'
    manifest=read(root/'metrics_manifest.json');expected={r['task_id']:r['sha256'] for r in manifest['records']}
    changes=defaultdict(list);ranks=[];exact=defaultdict(int);counts=defaultdict(int);endpoint_max=0.;pairs=0
    for task in sorted(expected):
        data=read(root/'per_image'/f'{task}.json.gz',expected[task])
        z={r['component_id']:r for r in data['component_rows']['normalized_zero']}
        b={r['component_id']:r for r in data['component_rows']['blur_rgb']}
        assert z.keys()==b.keys();model=data['model_id']
        for key,left in z.items():
            right=b[key];pairs+=1
            endpoint_max=max(endpoint_max,abs(left['m']-right['m']),abs(left['a']-right['a']))
            for metric,field in [('taylor','p_ref'),('finite_effect','absolute_effect')]:
                q0,q1=abs(left[field]),abs(right[field]);delta=abs(q0-q1);scale=max(q0,q1)
                changes[model,metric].append((delta,delta/scale if scale else None,delta>1e-12))
        for metric,field in [('taylor','p_ref'),('finite_effect','absolute_effect')]:
            block_metrics=defaultdict(list)
            for block in range(12):
                ids=sorted(k for k in z if z[k]['operation']=='head' and z[k]['block']==block)
                a=[abs(z[k][field]) for k in ids];v=[abs(b[k][field]) for k in ids]
                ar,br=average_ranks(a),average_ranks(v)
                ai=sorted(range(len(a)),key=lambda j:(-a[j],ids[j]));bi=sorted(range(len(v)),key=lambda j:(-v[j],ids[j]))
                exact[model,metric]+=ar==br;counts[model,metric]+=1
                metrics=dict(spearman=spearman(a,v),kendall_tau_b=tau_b(a,v),top1_agreement=float(ai[0]==bi[0]),
                    top3_overlap=len(set(ai[:3])&set(bi[:3]))/3,mean_absolute_rank_displacement=st.mean(abs(x-y) for x,y in zip(ar,br)))
                for k,value in metrics.items(): block_metrics[k].append(value)
            assert all(all(v is not None for v in values) for values in block_metrics.values())
            ranks.append(dict(model=model,sample_id=data['sample_id'],quantity=metric,**{k:st.mean(v) for k,v in block_metrics.items()}))
    assert len(expected)==800 and pairs==163200 and endpoint_max==0
    assert sum(v for (m,q),v in exact.items() if q=='taylor')==46
    assert sum(v for (m,q),v in exact.items() if q=='finite_effect')==64
    cs=[];rs=[]
    for model,metric in sorted(changes):
        rows=changes[model,metric]
        cs.append(dict(model=model,quantity=metric,component_count=len(rows),changed=sum(r[2] for r in rows),
            changed_fraction=st.mean(r[2] for r in rows),absolute_change=summary([r[0] for r in rows]),relative_change=summary([r[1] for r in rows])))
        selected=[r for r in ranks if r['model']==model and r['quantity']==metric]
        assert len(selected)==400
        rs.append(dict(model=model,quantity=metric,exact_order=exact[model,metric],head_blocks=counts[model,metric],
            **{k:summary([r[k] for r in selected]) for k in selected[0] if k not in ('model','sample_id','quantity')}))
    csvwrite('reference_rank_image_values.csv',ranks)
    return dict(atol=1e-12,rtol=0,component_pairs=pairs,endpoint_max_difference=endpoint_max,change_summaries=cs,rank_summaries=rs)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    raw=(REPO/'修改意见/修改意见v1.md').read_bytes()
    result=dict(status='COMPLETE_DESCRIPTIVE_REANALYSIS',opinion_sha256=hashlib.sha256(raw).hexdigest(),
                spatial=spatial(),reference=reference(),new_inferential_tests=0,cpu_only=True)
    assert not any(m in sys.modules for m in ('torch','timm','PIL'))
    write('analysis.json',result);write('analysis_sources.json',SOURCES)
    print(json.dumps(dict(status=result['status'],sources=len(SOURCES),curve_error=result['spatial']['curve_auc_max_error'])))

if __name__=='__main__': main()
