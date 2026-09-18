"""CPU-only supplementary image-level M1 ranking and inference."""
import hashlib
import math

import numpy as np

PREDICTORS = ('endpoint','activation_norm','gradient_norm','activation_gradient',
              'displacement_norm','taylor','random')
METRICS = ('spearman','capture_at_1','capture_at_3','regret_at_1','ndcg',
           'pairwise_accuracy','top1_accuracy')
FAMILIES = ('head','routing','value','Q','K','V','residual_identity','residual_branch')
OPS = {'head':{'head'}, 'routing':{'routing'}, 'value':{'value'}, 'Q':{'query'},
       'K':{'key'}, 'V':{'value'}, 'residual_identity':{'identity1','identity2'},
       'residual_branch':{'attention','mlp'}}

def seed(key):
    return int.from_bytes(hashlib.sha256(('ENDPOINT_M1_20260918_V1|20260918|'+key).encode()).digest()[:8],'big')

def ranks(x):
    x=np.asarray(x); order=np.argsort(x,kind='stable'); result=np.empty(len(x),float)
    a=0
    while a<len(x):
        b=a+1
        while b<len(x) and x[order[b]]==x[order[a]]:
            b+=1
        result[order[a:b]]=(a+b-1)/2
        a=b
    return result

def decision_metrics(order,y):
    order=np.atleast_2d(order); y=np.asarray(y,float); n=len(y)
    selected=y[order]; oracle=np.sort(y)[::-1]
    out={}
    for k in (1,3):
        denominator=oracle[:min(k,n)].sum()
        out[f'capture_at_{k}']=np.full(len(order),np.nan) if denominator==0 else selected[:,:min(k,n)].sum(1)/denominator
    out['regret_at_1']=y.max()-selected[:,0]
    discount=1/np.log2(np.arange(n)+2)
    ideal=oracle@discount
    out['ndcg']=np.full(len(order),np.nan) if ideal==0 else (selected@discount)/ideal
    out['top1_accuracy']=(selected[:,0]==y.max()).astype(float)
    return out

def metrics(p,y):
    p=np.asarray(p,float); y=np.asarray(y,float); n=len(y)
    out={k:float(v[0]) for k,v in decision_metrics(np.argsort(-p,kind='stable'),y).items()}
    rp=ranks(p); ry=ranks(y); rp-=rp.mean(); ry-=ry.mean()
    denominator=np.linalg.norm(rp)*np.linalg.norm(ry)
    out['spearman']=float(rp@ry/denominator) if n>=4 and denominator>0 else np.nan
    i,j=np.triu_indices(n,1); truth=np.sign(y[i]-y[j]); valid=truth!=0
    order=np.sign(p[i]-p[j])
    out['pairwise_accuracy']=float(np.mean((1+order[valid]*truth[valid])/2)) if valid.any() else np.nan
    return {k:None if not np.isfinite(out[k]) else out[k] for k in METRICS}

def random_metrics(y,key,repeats=1000):
    y=np.asarray(y,float); n=len(y); rng=np.random.Generator(np.random.PCG64(seed(key)))
    # Each row is an independent uniform random component permutation.
    order=np.argsort(rng.random((repeats,n)),axis=1,kind='stable')
    inverse=np.argsort(order,axis=1)
    values=decision_metrics(order,y)
    ry=ranks(y); ry-=ry.mean(); denom=np.linalg.norm(ry)*np.linalg.norm(np.arange(n)-(n-1)/2)
    values['spearman']=((n-1-inverse-(n-1)/2)@ry)/denom if n>=4 and denom>0 else np.full(repeats,np.nan)
    i,j=np.triu_indices(n,1); truth=np.sign(y[i]-y[j]); valid=truth!=0
    if valid.any():
        # Bound peak memory for larger across-block head sets.
        sums=np.zeros(repeats)
        for start in range(0,len(i),1024):
            sl=slice(start,start+1024); v=valid[sl]
            sums+=((1+np.sign(inverse[:,j[sl][v]]-inverse[:,i[sl][v]])*truth[sl][v])/2).sum(1)
        values['pairwise_accuracy']=sums/valid.sum()
    else:
        values['pairwise_accuracy']=np.full(repeats,np.nan)
    summary={}; means={}
    for name in METRICS:
        a=values[name]; finite=a[np.isfinite(a)]
        means[name]=None if not len(finite) else float(finite.mean())
        summary[name]=None if not len(finite) else dict(mean=float(finite.mean()),std=float(finite.std()),
                         q025=float(np.quantile(finite,.025)),q500=float(np.median(finite)),q975=float(np.quantile(finite,.975)))
    rho=values['spearman']; finite=np.clip(rho[np.isfinite(rho)],-1,1)
    summary['spearman_histogram_20bins']=np.histogram(finite,bins=np.linspace(-1,1,21))[0].tolist()
    return means,dict(seed=seed(key),permutations=repeats,distribution=summary)

def comparison_sets(rows):
    rows=sorted(rows,key=lambda r:r['component_id'])
    strata={'across_block':range(12),'early':range(4),'middle':range(4,8),'late':range(8,12)}
    for family in FAMILIES:
        for stratum,blocks in strata.items():
            yield family,stratum,[r for r in rows if r['operation'] in OPS[family] and r['block'] in blocks]
    for block in range(12):
        local=[r for r in rows if r['block']==block]
        yield 'head',f'block_{block:02d}',[r for r in local if r['operation']=='head']
        yield 'QKV_triplet',f'block_{block:02d}',[r for r in local if r['operation'] in {'query','key','value'}]
        yield 'residual_identity',f'block_{block:02d}',[r for r in local if r['operation'] in {'identity1','identity2'}]
        yield 'residual_branch',f'block_{block:02d}',[r for r in local if r['operation'] in {'attention','mlp'}]

def evaluate(rows,task_id,reference):
    out=[]
    for family,stratum,part in comparison_sets(rows):
        y=[r['absolute_effect'] for r in part]
        predictors={'endpoint':[r['m'] for r in part], 'activation_norm':[r['output_norm'] for r in part],
                    'gradient_norm':[r['gradient_norm'] for r in part],
                    'activation_gradient':[abs(r['gradient_activation']) for r in part],
                    'displacement_norm':[r['reference_displacement_norm'] for r in part],
                    'taylor':[abs(r['p_ref']) for r in part]}
        computed={name:metrics(p,y) for name,p in predictors.items()}
        random,dist=random_metrics(y,f'{task_id}|{reference}|{family}|{stratum}')
        computed['random']=random
        out.append(dict(family=family,stratum=stratum,n=len(part),
                        component_ids=[r['component_id'] for r in part],metrics=computed,random=dist))
    # Only complete, defined block metrics enter the within-image mean.
    for family in ('head','QKV_triplet','residual_identity','residual_branch'):
        blocks=[r for r in out if r['family']==family and r['stratum'].startswith('block_')]
        aggregate={}
        for predictor in PREDICTORS:
            aggregate[predictor]={}
            for metric in METRICS:
                values=[r['metrics'][predictor][metric] for r in blocks]
                aggregate[predictor][metric]=None if any(v is None for v in values) else float(np.mean(values))
        out.append(dict(family=family,stratum='within_block_mean',n=12,component_ids=[],metrics=aggregate,random=None))
    return out

def bootstrap_median(values,key,replicates=10000):
    """Exact marginal image-bootstrap median law via uniform order statistics.

    For even n, U_(n/2)~Beta(n/2,n/2+1) and the next uniform order
    statistic conditional on it is u+(1-u)*Beta(1,n/2). Mapping these
    uniforms through the empirical image CDF is distributionally identical
    to drawing n image IDs with replacement and taking their median.
    """
    a=np.sort(np.asarray(values,float)); n=len(a)
    if n==0:
        return None
    rng=np.random.Generator(np.random.PCG64(seed('bootstrap|'+key)))
    if n%2:
        u=rng.beta((n+1)//2,(n+1)//2,replicates)
        draws=a[np.minimum((n*u).astype(int),n-1)]
    else:
        u=rng.beta(n//2,n//2+1,replicates)
        v=u+(1-u)*rng.beta(1,n//2,replicates)
        draws=(a[np.minimum((n*u).astype(int),n-1)]+a[np.minimum((n*v).astype(int),n-1)])/2
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]

def sign_test(values):
    values=np.asarray(values,float); positive=int((values>1e-12).sum()); negative=int((values< -1e-12).sum())
    n=positive+negative
    p=1.0 if n==0 else float(sum(math.comb(n,k) for k in range(positive,n+1))/2**n)
    return dict(positive=positive,negative=negative,ties=len(values)-n,p=p)

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['p'],rows[i]['test_id']))
    prev=0.0
    for rank,index in enumerate(order):
        prev=max(prev,min(1.0,(len(rows)-rank)*rows[index]['p']))
        rows[index]['holm_p']=prev
        rows[index]['reject']=prev<.05
    return rows
