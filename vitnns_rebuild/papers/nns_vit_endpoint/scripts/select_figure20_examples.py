#!/usr/bin/env python3
"""Outcome-blind readability selection for the ViT-B Figure 20 examples."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
E2=REPO/'reports/nns_vit_endpoint_e2_atlas_r1_20260918/formal_results/attempts'
EXCLUDED={4,14,15}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ranks(values):
    order=sorted(range(len(values)),key=lambda i:(values[i],i));result=[0.0]*len(values);start=0
    while start<len(order):
        stop=start+1
        while stop<len(order) and values[order[stop]]==values[order[start]]:stop+=1
        rank=(start+stop-1)/2
        for i in range(start,stop):result[order[i]]=rank
        start=stop
    return result
def spearman(a,b):
    ra=np.asarray(ranks(a.tolist()));rb=np.asarray(ranks(b.tolist()));ra-=ra.mean();rb-=rb.mean()
    den=float(np.linalg.norm(ra)*np.linalg.norm(rb));return float(ra.dot(rb)/den) if den else None
def patch_sum(value):return value.reshape(14,16,14,16).sum((1,3))
def crop(path):
    with Image.open(path) as source:
        rgb=source.convert('RGB');w,h=rgb.size
        size=(248,round(h*248/w)) if w<h else (round(w*248/h),248)
        resized=rgb.resize(size,Image.Resampling.BICUBIC);left=(resized.width-224)//2;top=(resized.height-224)//2
        return resized.crop((left,top,left+224,top+224))
def stable_top(values,count):return set(sorted(range(len(values)),key=lambda i:(-float(values[i]),i))[:count])
def percentile(values,index):
    ordered=sorted((v,i) for i,v in enumerate(values));position=next(j for j,(_,i) in enumerate(ordered) if i==index)
    return position/(len(values)-1) if len(values)>1 else 1.0

def main():
    records=[]
    for order in range(16):
        if order in EXCLUDED:continue
        attempt=E2/f'vit_b16__{order:03d}/attempt-01.json';envelope=json.loads(attempt.read_text())['result']
        atlas_record=next(r for r in envelope['artifacts'] if r['kind']=='full_raw_atlas');atlas_path=REPO/atlas_record['path']
        assert sha(atlas_path)==atlas_record['sha256']
        index=next(i for i,r in enumerate(envelope['summaries']) if r['target_id']=='class.gt')
        raw=np.asarray(np.load(atlas_path,mmap_mode='r')[index],dtype=np.float64)
        endpoint=np.maximum(raw.sum(0),0);mass=patch_sum(endpoint).reshape(-1);total=float(mass.sum())
        assert total>0;q=mass/total;positive=q[q>0];effective=float(math.exp(-float((positive*np.log(positive)).sum())))
        image_path=REPO/'assets/data/nns_vit_endpoint_e2_evaluation_20260918'/envelope['sample_id']
        assert sha(image_path)==envelope['image_sha256']
        image=crop(image_path);rgb=np.asarray(image,dtype=np.float64)/255
        blurred=np.asarray(image.filter(ImageFilter.GaussianBlur(8)),dtype=np.float64)/255
        contrast=np.linalg.norm(rgb-blurred,axis=2);gray=rgb.mean(2)
        gy,gx=np.gradient(gray);detail=contrast+np.sqrt(gx*gx+gy*gy);detail_patch=patch_sum(detail).reshape(-1)
        detail_norm=detail_patch/(float(detail_patch.max()) or 1)
        rho=spearman(q,detail_patch);a=stable_top(q,49);b=stable_top(detail_patch,49);iou=len(a&b)/len(a|b)
        weighted=float(q.dot(detail_norm))
        records.append(dict(order=order,task_id=f'vit_b16__{order:03d}',sample_id=envelope['sample_id'],class_id=envelope['ground_truth_class_id'],wnid=envelope['wnid'],image_sha256=envelope['image_sha256'],atlas_sha256=atlas_record['sha256'],endpoint_raw_sha256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),effective_patches=effective,eligible=effective>=6,spearman=rho,top_quartile_iou=iou,mass_weighted_detail=weighted))
    eligible=[r for r in records if r['eligible']];metrics=('spearman','top_quartile_iou','mass_weighted_detail')
    for metric in metrics:
        values=[r[metric] for r in eligible]
        for i,r in enumerate(eligible):r[metric+'_percentile']=percentile(values,i)
    for r in eligible:r['selection_score']=sum(r[m+'_percentile'] for m in metrics)/len(metrics)
    ranking=sorted(eligible,key=lambda r:(-r['selection_score'],r['order']))
    for i,r in enumerate(ranking,1):r['rank']=i
    selected=[r['order'] for r in ranking[:3]]
    output=dict(status='SEALED_BEFORE_COMPARISON_METHOD_COMPUTE',protocol='evidence/figure20_reselection_protocol.md',universe_orders=list(range(16)),excluded_by_user=sorted(EXCLUDED),eligible_count=len(eligible),selected_orders=selected,metrics=list(metrics),ranking=ranking,ineligible=[r for r in records if not r['eligible']],comparison_methods_read=False)
    path=ROOT/'evidence/figure20_selection.json';path.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+chr(10))
    print(json.dumps({'selected_orders':selected,'top':[{k:r[k] for k in ('rank','order','sample_id','selection_score','spearman','top_quartile_iou','mass_weighted_detail','effective_patches')} for r in ranking[:6]]},indent=2))

if __name__=='__main__':main()
