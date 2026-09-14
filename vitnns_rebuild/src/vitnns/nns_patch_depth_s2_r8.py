"""Deterministic hierarchical gatekeeping and power primitives for S2-R8."""
from __future__ import annotations
import hashlib,math,random
from typing import Mapping,Sequence,Any
ENDPOINTS=("P1","P2","D1","D2");PRIMARY=("P1","P2","D1")
def require(c:bool,m:str)->None:
    if not c:raise ValueError(m)
def seed64(namespace:str,*parts:object)->int:return int.from_bytes(hashlib.sha256((namespace+"\0"+"\0".join(str(p) for p in parts)).encode()).digest()[:8],"big")
def normal_cdf(x:float)->float:return .5*(1+math.erf(x/math.sqrt(2)))
def normal_ppf(p:float)->float:
    require(0<p<1,"p range");lo,hi=-10.,10.
    for _ in range(100):
        mid=(lo+hi)/2
        if normal_cdf(mid)<p:lo=mid
        else:hi=mid
    return (lo+hi)/2
def cholesky(a:Sequence[Sequence[float]])->list[list[float]]:
    n=len(a);out=[[0.]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            x=float(a[i][j])-math.fsum(out[i][k]*out[j][k] for k in range(j))
            if i==j:require(x>-1e-10,"not PSD");out[i][j]=math.sqrt(max(x,1e-12))
            else:out[i][j]=x/out[j][j]
    return out
def holm_primary(p:Mapping[str,float],alpha:float=.05)->dict[str,Any]:
    require(set(p)==set(PRIMARY),"primary family drift");ordered=sorted(PRIMARY,key=lambda x:(float(p[x]),PRIMARY.index(x)));stopped=False;rows=[]
    for rank,name in enumerate(ordered,1):
        critical=alpha/(4-rank);reject=(not stopped) and float(p[name])<=critical
        if not reject:stopped=True
        rows.append({"endpoint":name,"rank":rank,"critical_alpha":critical,"reject":reject})
    return {"rows":rows,"all_reject":all(x["reject"] for x in rows)}
def wilson_lower(success:int,n:int,z:float=1.6448536269514722)->float:
    require(0<=success<=n and n>0,"Wilson count");p=success/n;den=1+z*z/n;return max(0.,(p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den)
def wilson_upper(success:int,n:int,z:float=1.6448536269514722)->float:
    require(0<=success<=n and n>0,"Wilson count");p=success/n;den=1+z*z/n;return min(1.,(p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den)
def cluster_se(scale:float,icc:float,counts:Sequence[int])->float:
    require(counts and all(x>=1 for x in counts),"cluster counts");c=len(counts);return scale*math.sqrt(math.fsum(icc+(1-icc)/m for m in counts)/(c*c))
def pvalues(z:Sequence[float])->dict[str,float]:return {"P1":1-normal_cdf(z[0]),"P2":1-normal_cdf(z[1]),"D1":min(1.,2*(1-normal_cdf(abs(z[2])))),"D2":min(1.,2*(1-normal_cdf(abs(z[3]))))}
def decisions(z:Sequence[float])->tuple[bool,bool]:
    p=pvalues(z);gate=holm_primary({x:p[x] for x in PRIMARY})["all_reject"];return gate,gate and p["D2"]<=.05
def correlation_scenarios(base:Sequence[Sequence[float]])->dict[str,list[list[float]]]:
    result={}
    for name,factor in (("R4_CLASS_LEVEL_CORRELATION",1.),("D2_CROSS_CORRELATIONS_ZERO",0.),("D2_CROSS_CORRELATIONS_SIGN_FLIPPED",-1.)):
        a=[list(row) for row in base]
        for i in range(3):a[i][3]=a[3][i]=factor*float(base[i][3])
        cholesky(a);result[name]=a
    return result
