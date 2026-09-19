"""Image-level summaries and frozen M3 atlas decisions."""
from __future__ import annotations
import math
import hashlib
import numpy as np
from scipy.stats import binomtest

PHASES={"early":(0,1,2,3),"middle":(4,5,6,7),"late":(8,9,10,11)}
METRICS=("entropy","locality","distance","cancellation")

def image_metric_values(data) -> dict:
    block=data["block"];token=data["token"];zero=data["zero_gradient"]
    entropy=data["normalized_spatial_entropy"].astype(np.float64);cancellation=data["cancellation"].astype(np.float64)
    patch_absolute=data["patch_absolute_mass_196"].astype(np.float64);values={}
    locality=np.full(len(block),np.nan);distance=np.full(len(block),np.nan)
    for i in np.flatnonzero((token>0)&~zero):
        p=patch_absolute[i]/patch_absolute[i].sum();tr,tc=divmod(int(token[i])-1,14);rr,cc=np.indices((14,14));dr=np.abs(rr-tr);dc=np.abs(cc-tc);mask=np.maximum(dr,dc)<=1
        locality[i]=p.reshape(14,14)[mask].sum()-mask.sum()/196;distance[i]=(p.reshape(14,14)*np.sqrt(dr**2+dc**2)/(13*math.sqrt(2))).sum()
    arrays={"entropy":entropy,"locality":locality,"distance":distance,"cancellation":cancellation}
    for phase,blocks in PHASES.items():
        for metric,array in arrays.items():
            mask=np.isin(block,blocks)&np.isfinite(array);values[f"{phase}_{metric}"]=None if not mask.any() else float(np.median(array[mask]))
    for kind,mask0 in (("cls",token==0),("patch",token>0)):
        for metric,array in arrays.items():
            mask=mask0&np.isfinite(array);values[f"{kind}_{metric}"]=None if not mask.any() else float(np.median(array[mask]))
    # Center/edge are disjoint frozen descriptors for patch units.
    row=(token-1)//14;column=(token-1)%14;center=(token>0)&(np.maximum(np.abs(row-6.5),np.abs(column-6.5))<=3);edge=(token>0)&(np.minimum.reduce([row,column,13-row,13-column])<=2)
    for kind,mask0 in (("center",center),("edge",edge),("all_patch",token>0)):
        for metric,array in arrays.items():
            mask=mask0&np.isfinite(array);values[f"{kind}_{metric}"]=None if not mask.any() else float(np.median(array[mask]))
    values.update(late_minus_early_entropy=_difference(values,"late_entropy","early_entropy"),late_minus_early_locality=_difference(values,"late_locality","early_locality"),late_minus_early_distance=_difference(values,"late_distance","early_distance"),late_minus_early_cancellation=_difference(values,"late_cancellation","early_cancellation"),cls_minus_patch_entropy=_difference(values,"cls_entropy","patch_entropy"))
    values["zero_gradient_count"]=int(zero.sum());return values

def _difference(values,a,b):return None if values[a] is None or values[b] is None else values[a]-values[b]

def bootstrap_median(values,key,replicates=10000):
    a=np.sort(np.asarray(values,float));n=len(a)
    if n==0:return None
    seed=int.from_bytes(hashlib.sha256(("ENDPOINT_M3_BOOTSTRAP_20260919|"+key).encode()).digest()[:8],"big");rng=np.random.Generator(np.random.PCG64(seed))
    if n%2:
        u=rng.beta((n+1)//2,(n+1)//2,replicates);draws=a[np.minimum((n*u).astype(int),n-1)]
    else:
        u=rng.beta(n//2,n//2+1,replicates);v=u+(1-u)*rng.beta(1,n//2,replicates);draws=(a[np.minimum((n*u).astype(int),n-1)]+a[np.minimum((n*v).astype(int),n-1)])/2
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]

def exact_two_sided_sign(values,tie_tolerance=1e-12):
    a=np.asarray(values,float);positive=int((a>tie_tolerance).sum());negative=int((a< -tie_tolerance).sum());n=positive+negative
    return {"positive":positive,"negative":negative,"ties":int(len(a)-n),"p":1.0 if n==0 else float(binomtest(positive,n,.5).pvalue)}

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:(rows[i]["p"],rows[i]["test_id"]));previous=0.0
    for rank,index in enumerate(order):
        previous=max(previous,min(1.0,(len(rows)-rank)*rows[index]["p"]));rows[index]["holm_p"]=previous;rows[index]["reject"]=previous<.05
    return rows

def summarize_image_values(records,bootstrap=10000):
    summaries=[];tests=[]
    primary=("late_minus_early_entropy","late_minus_early_locality","late_minus_early_distance","late_minus_early_cancellation","cls_minus_patch_entropy")
    for model in sorted({r["model_id"] for r in records}):
        part=[r for r in records if r["model_id"]==model]
        keys=sorted(set(part[0])-{"model_id","sample_id","image_sha256"})
        for key in keys:
            values=[r[key] for r in part if r[key] is not None]
            summaries.append({"model":model,"metric":key,"n":len(values),"median":None if not values else float(np.median(values)),"ci95":None if not values else bootstrap_median(values,f"M3|{model}|{key}",bootstrap)})
        family=[]
        for key in primary:
            values=[r[key] for r in part if r[key] is not None];row={"test_id":f"{model}|{key}","model":model,"contrast":key,"n":len(values),"median_difference":None if not values else float(np.median(values))}
            row.update(exact_two_sided_sign(values));family.append(row)
        tests.extend(holm(family))
    return summaries,tests

def cosine(a,b):
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return None if denominator==0 else float(np.dot(a,b)/denominator)
