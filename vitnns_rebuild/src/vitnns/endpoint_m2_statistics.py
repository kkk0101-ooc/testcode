"""Image-unit supplementary M2 statistics; no image/model imports."""
from collections import defaultdict
import numpy as np
from .endpoint_m2_regions import METHODS,HEADS,BLOCKS,COUNTS,GRIDS
from .endpoint_m1_statistics import bootstrap_median, sign_test, holm

def image_values(result):
    groups=defaultdict(list)
    for r in result['curves']:
        key=(r['geometry'],r['reference'],r['head'],r['method'])
        for outcome in ('full','isolated'):
            groups[(*key,outcome,'auc')].append(r['auc'][outcome])
            for p in r['points']:
                groups[(*key,outcome,f"effect_{p['count']}")].append(p['absolute'][outcome])
                groups[(*key,outcome,f"regret_{p['count']}")].append(p['regret'][outcome])
    heads={k:float(np.mean(v)) for k,v in groups.items()}
    for k,v in groups.items():
        if len(v)!=(20 if k[3]=='random' else 1): raise ValueError('REPEAT_COVERAGE')
    block_groups=defaultdict(list)
    for (geometry,ref,head,method,outcome,metric),v in heads.items():
        block_groups[(geometry,ref,method,outcome,metric)].append(v)
    if any(len(v)!=HEADS[result['model_id']] for v in block_groups.values()):
        raise ValueError('HEAD_COVERAGE')
    block={k:float(np.mean(v)) for k,v in block_groups.items()}
    return heads,block

def summarize(groups,expected=400):
    summaries=[];pairs=[];tests=[]
    for key,values_by_id in sorted(groups.items(),key=lambda item:str(item[0])):
        model,block,geometry,ref,outcome,metric=key
        if len(values_by_id)!=expected: raise ValueError('IMAGE_COVERAGE')
        values=np.array([values_by_id[i] for i in sorted(values_by_id)])
        if values.shape!=(expected,len(METHODS)) or not np.isfinite(values).all(): raise ValueError('VALUES')
        meta=dict(model=model,block=block,geometry=geometry,reference=ref,outcome=outcome,metric=metric,n_images=expected)
        prefix='M2|'+'|'.join(map(str,key))
        for j,method in enumerate(METHODS):
            summaries.append(dict(meta,method=method,median=float(np.median(values[:,j])),
                ci95=bootstrap_median(values[:,j],prefix+'|'+method)))
        for j,method in enumerate(METHODS[1:],1):
            difference=values[:,0]-values[:,j]
            if metric.startswith('regret'): difference=-difference
            paired=dict(meta,baseline=method,median=float(np.median(difference)),
                ci95=bootstrap_median(difference,prefix+'|paired|'+method))
            pairs.append(paired)
            if metric=='auc' and block in BLOCKS:
                for direction in ('endpoint_greater','baseline_greater'):
                    vals=difference if direction=='endpoint_greater' else -difference
                    tests.append(dict(meta,baseline=method,direction=direction,
                        test_id=prefix+'|'+method+'|'+direction,**sign_test(vals)))
    holm(tests)
    for r in tests: r['decision']='SUPPORTED' if r['reject'] else 'NOT_SUPPORTED'
    if len(tests)!=672: raise ValueError('HOLM_TEST_COUNT')
    return summaries,pairs,tests
