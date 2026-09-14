"""Standard-library S2-R4 power, Holm, and dynamic-queue primitives."""
from __future__ import annotations
import hashlib, json, math, os, random
from pathlib import Path
from typing import Any, Mapping, Sequence

ENDPOINTS=("P1","P2","D1","D2")
CLASS_NAMESPACE="NNS_VIT_PATCH_DEPTH_S2_R4_FORMAL_CLASS_COHORT_V1"
POWER_NAMESPACE="NNS_VIT_PATCH_DEPTH_S2_R4_POWER_SIMULATION_V1"

class S2R4Error(ValueError): pass
def require(c:bool,m:str)->None:
    if not c: raise S2R4Error(m)
def canonical(v:Any)->bytes: return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(v:Any)->str: return hashlib.sha256(canonical(v)).hexdigest()
def seed64(namespace:str,*parts:object)->int:
    return int.from_bytes(hashlib.sha256((namespace+"\0"+"\0".join(str(p) for p in parts)).encode()).digest()[:8],"big")
def class_key(wnid:str,namespace:str=CLASS_NAMESPACE)->tuple[str,str]: return hashlib.sha256((namespace+"\0"+wnid).encode()).hexdigest(),wnid
def select_classes(wnids:Sequence[str],count:int)->list[str]:
    values=sorted(set(wnids),key=class_key); require(0<count<=len(values),"invalid class count"); return values[:count]
def normal_cdf(x:float)->float: return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def normal_ppf(p:float)->float:
    require(0.0<p<1.0,"p outside (0,1)"); lo,hi=-10.0,10.0
    for _ in range(100):
        mid=(lo+hi)/2.0
        if normal_cdf(mid)<p: lo=mid
        else: hi=mid
    return (lo+hi)/2.0
def sample_variance(values:Sequence[float])->float:
    require(len(values)>=2,"variance needs two observations"); center=math.fsum(values)/len(values); return math.fsum((x-center)**2 for x in values)/(len(values)-1)
def covariance(left:Sequence[float],right:Sequence[float])->float:
    require(len(left)==len(right) and len(left)>=2,"covariance shape"); a=math.fsum(left)/len(left); b=math.fsum(right)/len(right); return math.fsum((x-a)*(y-b) for x,y in zip(left,right))/(len(left)-1)
def correlation_matrix(rows:Sequence[Mapping[str,float]])->list[list[float]]:
    cols={name:[float(row[name]) for row in rows] for name in ENDPOINTS}; out=[]
    for a in ENDPOINTS:
        row=[]
        for b in ENDPOINTS:
            denom=math.sqrt(sample_variance(cols[a])*sample_variance(cols[b])); row.append(1.0 if a==b else covariance(cols[a],cols[b])/denom)
        out.append(row)
    return out
def cholesky(matrix:Sequence[Sequence[float]])->list[list[float]]:
    n=len(matrix); out=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            value=float(matrix[i][j])-math.fsum(out[i][k]*out[j][k] for k in range(j))
            if i==j: out[i][j]=math.sqrt(max(value,1e-12))
            else: out[i][j]=value/out[j][j]
    return out
def holm(p_values:Mapping[str,float],alpha:float=0.05)->dict[str,Any]:
    require(set(p_values)==set(ENDPOINTS),"four fixed Holm slots required"); ordered=sorted(ENDPOINTS,key=lambda x:(float(p_values[x]),ENDPOINTS.index(x))); stopped=False; rows=[]
    for rank,name in enumerate(ordered,1):
        threshold=alpha/(5-rank); reject=(not stopped) and float(p_values[name])<=threshold
        if not reject: stopped=True
        rows.append({"endpoint":name,"rank":rank,"critical_alpha":threshold,"reject":reject})
    return {"family_size":4,"rows":rows,"rejected":[r["endpoint"] for r in rows if r["reject"]]}
def simulate_power(count:int,scales:Mapping[str,float],correlation:Sequence[Sequence[float]],margins:Mapping[str,float],missing_rate:float,replicates:int,namespace:str=POWER_NAMESPACE)->dict[str,Any]:
    valid=max(1,int(math.floor(2*count*(1.0-missing_rate)))); critical={"P1":normal_ppf(1-.05/4),"P2":normal_ppf(1-.05/4),"D1":normal_ppf(1-.05/8),"D2":normal_ppf(1-.05/8)}; lower=cholesky(correlation); rng=random.Random(seed64(namespace,count,replicates)); hits={x:0 for x in ENDPOINTS}
    for _ in range(replicates):
        base=[rng.gauss(0.0,1.0) for _ in ENDPOINTS]; z=[math.fsum(lower[i][j]*base[j] for j in range(i+1)) for i in range(4)]
        for i,name in enumerate(ENDPOINTS):
            statistic=z[i]+margins[name]*math.sqrt(valid)/scales[name]
            hits[name]+=statistic>critical[name]
    precision={name:normal_ppf(.975)*scales[name]/math.sqrt(valid) for name in ENDPOINTS}
    return {"class_count":count,"scheduled_images":2*count,"conservative_valid_images":valid,"power":{name:hits[name]/replicates for name in ENDPOINTS},"precision_half_width":precision,"critical_z":critical,"replicates":replicates}
def atomic_claim(launch:Path,index:int,source_id:str,worker:Mapping[str,Any],now_ns:int)->dict[str,Any]|None:
    claims=launch/"claims"; claims.mkdir(parents=True,exist_ok=True); path=claims/("unit-%04d.claim.json"%index); value={"schedule_index":index,"source_image_id":source_id,"claim_time_unix_ns":now_ns,"worker_id":worker["worker_id"],"worker_pid":worker["worker_pid"],"physical_gpu_index":worker["physical_gpu_index"],"gpu_uuid":worker["gpu_uuid"]}
    try: fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    except FileExistsError: return None
    with os.fdopen(fd,"wb") as f: f.write(canonical(value)+b"\n"); f.flush(); os.fsync(f.fileno())
    d=os.open(claims,os.O_RDONLY)
    try: os.fsync(d)
    finally: os.close(d)
    return value
def terminal_indices(stage:Path)->set[int]:
    out=set()
    for path in stage.glob("launches/*/units/unit-*/completion/unit.json"):
        v=json.loads(path.read_text()); p=v.get("payload",{})
        if p.get("terminal")=={"complete":True,"immutable":True}: out.add(int(p["schedule_index"]))
    return out
def claim_next(stage:Path,launch:Path,schedule:Sequence[Mapping[str,Any]],worker:Mapping[str,Any],now_ns:int)->dict[str,Any]|None:
    complete=terminal_indices(stage)
    for row in sorted(schedule,key=lambda x:int(x["schedule_index"])):
        i=int(row["schedule_index"]);
        if i in complete: continue
        value=atomic_claim(launch,i,str(row["source_image_id"]),worker,now_ns)
        if value is not None: return value
    return None
def aggregate_resources(rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    values=sorted(rows,key=lambda x:(int(x["schedule_index"]),str(x["launch_id"])))
    return {"unit_count":len(values),"gpu_seconds":math.fsum(float(x["gpu_seconds"]) for x in values),"cpu_seconds":math.fsum(float(x["cpu_seconds"]) for x in values),"wall_seconds_sum":math.fsum(float(x["wall_seconds"]) for x in values),"retained_bytes":sum(int(x["retained_bytes"]) for x in values),"stable_sort_math_fsum":True,"divide_GPU_hours_by_K":False}
