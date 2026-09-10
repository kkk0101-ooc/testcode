"""Frozen deterministic evaluators for Stage X X2."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from .stage_x_nns_v2 import NNSV2Error

ROAD_PERCENTILES = (10,20,30,40,50,60,70,80,90)
FRACTIONS = tuple(index/20 for index in range(21))
ROAD_SEEDS = tuple(range(5))
TIE_REPEATS = tuple(range(5))


def seed64(namespace: str, *parts: object) -> int:
    text="|".join((namespace,*(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8],"big")


def pixel_map(value: torch.Tensor, height: int=224, width: int=224) -> torch.Tensor:
    """Represent method-defined cell support at pixels without interpolation."""
    value=value.detach().cpu().float()
    if value.ndim==3:
        if value.shape[0] not in (1,3): raise NNSV2Error("unexpected channel attribution")
        value=value.sum(0)
    if value.shape==(height,width): return value.contiguous()
    if value.ndim!=2 or height%value.shape[0] or width%value.shape[1]:
        raise NNSV2Error("method support cannot tile the input exactly")
    return value.repeat_interleave(height//value.shape[0],0).repeat_interleave(width//value.shape[1],1).contiguous()


def _ranking(score: np.ndarray, identity: str, repeat: int, *, descending: bool) -> np.ndarray:
    flat=np.asarray(score,dtype=np.float64).reshape(-1)
    rng=np.random.Generator(np.random.PCG64(seed64("STAGE_X_FAITHFULNESS_TIES_V1_20260908",identity,repeat)))
    priority=rng.permutation(flat.size)
    # lexsort uses the last key as primary; priority deterministically resolves ties.
    return np.lexsort((priority,-flat if descending else flat))


def noisy_linear_impute(image: torch.Tensor, keep_mask: np.ndarray, seed: int) -> torch.Tensor:
    """Official ROAD linear system with PCG64 noise and no global RNG."""
    from pytorch_grad_cam.metrics.road import NoisyLinearImputer, neighbors_weights
    if image.ndim!=3 or tuple(keep_mask.shape)!=tuple(image.shape[-2:]):
        raise ValueError("ROAD image/mask geometry mismatch")
    if keep_mask.dtype!=np.float32: keep_mask=keep_mask.astype(np.float32)
    missing=np.flatnonzero(keep_mask.reshape(-1)==0)
    if not len(missing): return image.detach().cpu().clone()
    if len(missing)==keep_mask.size: return torch.zeros_like(image,device="cpu")
    source=image.detach().cpu().float(); A,b=NoisyLinearImputer.setup_sparse_system(keep_mask,source.numpy(),neighbors_weights)
    solved=np.asarray(spsolve(csc_matrix(A),b),dtype=np.float32)
    if solved.ndim==1: solved=solved[:,None]
    noise=np.random.Generator(np.random.PCG64(seed)).standard_normal(solved.shape,dtype=np.float32)*np.float32(0.01)
    flat=source.reshape(source.shape[0],-1).clone(); flat[:,missing]=torch.from_numpy((solved+noise).T)
    result=flat.reshape_as(source)
    if not bool(torch.isfinite(result).all()): raise ValueError("ROAD imputation is non-finite")
    return result


def road_combined(model: torch.nn.Module, image: torch.Tensor, attribution: torch.Tensor, target: int, identity: str, *, batch_size: int=45) -> dict[str,Any]:
    score=pixel_map(attribution).numpy(); flat_count=score.size
    with torch.no_grad():
        clean=float(model(image)[0,target]); baseline=float(model(torch.zeros_like(image))[0,target])
    denominator=clean-baseline
    if not math.isfinite(denominator) or abs(denominator)<1e-12: raise ValueError("ROAD normalization denominator is zero")
    rows=[]; pending=[]; coordinates=[]
    for percentile in ROAD_PERCENTILES:
        k=int(math.floor(flat_count*percentile/100))
        for road_seed in ROAD_SEEDS:
            for direction,descending in (("morf",True),("lerf",False)):
                order=_ranking(score,identity+f"|road|{percentile}|{road_seed}|{direction}",0,descending=descending)
                keep=np.ones(flat_count,dtype=np.float32); keep[order[:k]]=0
                seed=seed64("STAGE_X_X2_ROAD_NOISY_LINEAR_V1_20260909",identity,percentile,road_seed,direction)
                pending.append(noisy_linear_impute(image[0],keep.reshape(score.shape),seed)); coordinates.append((percentile,road_seed,direction,seed))
    values=[]
    with torch.no_grad():
        for begin in range(0,len(pending),batch_size):
            batch=torch.stack(pending[begin:begin+batch_size]).to(image.device,image.dtype)
            values.extend(float(x) for x in model(batch)[:,target].detach().cpu())
    for coordinate,value in zip(coordinates,values):
        percentile,road_seed,direction,seed=coordinate
        rows.append({"percentile":percentile,"seed_index":road_seed,"direction":direction,"pcg64_seed":seed,"normalized_response":(value-baseline)/denominator})
    morf=np.mean([r["normalized_response"] for r in rows if r["direction"]=="morf"],dtype=np.float64)
    lerf=np.mean([r["normalized_response"] for r in rows if r["direction"]=="lerf"],dtype=np.float64)
    return {"road_combined":float(0.5*(lerf-morf)),"road_morf":float(morf),"road_lerf":float(lerf),"rows":rows,"clean_logit":clean,"baseline_logit":baseline,"normalization":"(perturbed-baseline)/(clean-baseline)","forward_calls":2+math.ceil(len(pending)/batch_size),"imputation_count":len(pending)}


def insertion_deletion(model: torch.nn.Module,image: torch.Tensor,attribution: torch.Tensor,target: int,identity: str,*,batch_size:int=64) -> dict[str,Any]:
    score=pixel_map(attribution).numpy(); count=score.size; clean=float(model(image)[0,target].detach()); baseline_tensor=torch.zeros_like(image); baseline=float(model(baseline_tensor)[0,target].detach()); denominator=clean-baseline
    if abs(denominator)<1e-12: raise ValueError("insertion/deletion normalization denominator is zero")
    records=[]
    for repeat in TIE_REPEATS:
        order=_ranking(score,identity+"|curve",repeat,descending=True); batches=[]
        actual=[]
        for fraction in FRACTIONS:
            k=int(round(count*fraction)); mask=np.zeros(count,dtype=np.float32);mask[order[:k]]=1;mask=torch.from_numpy(mask.reshape(1,1,224,224)).to(image.device)
            batches.extend((baseline_tensor[0]*(1-mask[0])+image[0]*mask[0],image[0]*(1-mask[0])+baseline_tensor[0]*mask[0]));actual.append(k/count)
        responses=[]
        with torch.no_grad():
            for begin in range(0,len(batches),batch_size): responses.extend(float(v) for v in model(torch.stack(batches[begin:begin+batch_size]))[:,target].detach().cpu())
        insertion=[(v-baseline)/denominator for v in responses[0::2]]; deletion=[(v-baseline)/denominator for v in responses[1::2]]
        records.append({"repeat":repeat,"seed":seed64("STAGE_X_FAITHFULNESS_TIES_V1_20260908",identity+"|curve",repeat),"actual_fractions":actual,"insertion":insertion,"deletion":deletion,"insertion_auc":float(np.trapz(insertion,actual)),"deletion_auc":float(np.trapz(deletion,actual)),"sufficiency_20":float(insertion[4]),"comprehensiveness_20":float(1-deletion[4])})
    keys=("insertion_auc","deletion_auc","sufficiency_20","comprehensiveness_20")
    return {"repeats":records,"means":{key:float(np.mean([row[key] for row in records],dtype=np.float64)) for key in keys},"forward_calls":2+len(records)*math.ceil(2*len(FRACTIONS)/batch_size),"auc":"trapezoidal","fraction_grid":list(FRACTIONS)}


def signed_spearman(left: np.ndarray,right: np.ndarray,mask: np.ndarray|None=None) -> float:
    from scipy.stats import spearmanr
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64)
    if mask is not None: a,b=a[mask],b[mask]
    value=float(spearmanr(a.reshape(-1),b.reshape(-1)).statistic)
    return value


def map_metrics(clean: np.ndarray,transformed: np.ndarray,mask: np.ndarray|None=None) -> dict[str,float]:
    from skimage.metrics import structural_similarity
    a=np.asarray(clean,dtype=np.float64);b=np.asarray(transformed,dtype=np.float64); valid=np.ones_like(a,dtype=bool) if mask is None else mask.astype(bool)
    av=a[valid];bv=b[valid];k=max(1,int(math.ceil(0.1*av.size)))
    ia=np.argpartition(av,-k)[-k:];ib=np.argpartition(bv,-k)[-k:]
    jaccard=len(set(ia).intersection(set(ib)))/len(set(ia).union(set(ib)))
    data_range=max(float(max(av.max(),bv.max())-min(av.min(),bv.min())),1e-12)
    indices=np.argwhere(valid); r0,c0=indices.min(0);r1,c1=indices.max(0)+1
    aa=a[r0:r1,c0:c1];bb=b[r0:r1,c0:c1];ssim=float(structural_similarity(aa,bb,data_range=data_range))
    coords=np.indices(a.shape).reshape(2,-1).T
    if mask is not None: coords=coords[valid.reshape(-1)]
    diag=math.hypot(*a.shape)
    def center(values:np.ndarray,positive:bool)->np.ndarray:
        weight=np.maximum(values if positive else -values,0); total=weight.sum()
        return np.array([(a.shape[0]-1)/2,(a.shape[1]-1)/2]) if total<=1e-20 else (coords*weight[:,None]).sum(0)/total
    pa,na=center(av,True),center(av,False);pb,nb=center(bv,True),center(bv,False)
    displacement=(np.linalg.norm(pa-pb)+np.linalg.norm(na-nb))/(2*diag)
    total_a=np.abs(av);total_b=np.abs(bv);ca=(coords*total_a[:,None]).sum(0)/max(total_a.sum(),1e-20);cb=(coords*total_b[:,None]).sum(0)/max(total_b.sum(),1e-20)
    return {"signed_spearman":signed_spearman(av,bv),"top10_mass_jaccard":float(jaccard),"signed_ssim":ssim,"normalized_positive_negative_mass_displacement":float(displacement),"center_of_mass_pixel_error":float(np.linalg.norm(ca-cb))}


def transform_image(image: torch.Tensor,name: str,identity: str,preprocess: Mapping[str,Any]) -> tuple[torch.Tensor,dict[str,Any]]:
    """Apply one frozen transform and return inverse-map metadata."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    if image.shape!=(1,3,224,224): raise ValueError("transformation input shape drift")
    value=image[0]
    mean=torch.tensor(preprocess["mean"],device=value.device,dtype=value.dtype).reshape(3,1,1)
    std=torch.tensor(preprocess["std"],device=value.device,dtype=value.dtype).reshape(3,1,1)
    rgb=(value*std+mean).clamp(0,1)
    metadata: dict[str,Any]={"name":name,"inverse":"identity","valid_box":[0,224,0,224]}
    if name.startswith("translate_"):
        axis=name.split("_")[1]; amount=int(name.split("_")[2]); pad=abs(amount)
        if axis=="x": padded=torch.nn.functional.pad(value,(pad,pad,0,0),mode="reflect"); result=padded[:,:,pad-amount:pad-amount+224]; metadata.update({"inverse":"translate","axis":"x","amount":amount,"valid_box":[0,224,max(0,amount),min(224,224+amount)]})
        else: padded=torch.nn.functional.pad(value,(0,0,pad,pad),mode="reflect"); result=padded[:,pad-amount:pad-amount+224,:]; metadata.update({"inverse":"translate","axis":"y","amount":amount,"valid_box":[max(0,amount),min(224,224+amount),0,224]})
    elif name=="center_crop_216_resize_224": result=TF.resize(TF.center_crop(value,[216,216]),[224,224],interpolation=InterpolationMode.BICUBIC,antialias=True);metadata.update({"inverse":"resize_224_to_216_then_pad4","valid_box":[4,220,4,220]})
    elif name=="horizontal_flip": result=TF.hflip(value);metadata["inverse"]="horizontal_flip"
    elif name.startswith("rotate_"): angle=float(name.split("_")[1]);result=TF.rotate(value,angle,interpolation=InterpolationMode.BILINEAR,fill=0.0);metadata.update({"inverse":"rotate","angle":angle,"valid_box":[20,204,20,204]})
    elif name.startswith("gaussian_noise_0.01_seed_"):
        seed_index=int(name.rsplit("_",1)[1]);rng=np.random.Generator(np.random.PCG64(seed64("STAGE_X_X2_STABILITY_NOISE_V1_20260909",identity,seed_index)));noise=torch.from_numpy(rng.normal(0,0.01,size=tuple(rgb.shape)).astype(np.float32)).to(value.device);result=((rgb+noise).clamp(0,1)-mean)/std;metadata["seed"]=seed64("STAGE_X_X2_STABILITY_NOISE_V1_20260909",identity,seed_index)
    elif name.startswith("brightness_"): result=(TF.adjust_brightness(rgb,float(name.split("_")[1])).clamp(0,1)-mean)/std
    elif name.startswith("contrast_"): result=(TF.adjust_contrast(rgb,float(name.split("_")[1])).clamp(0,1)-mean)/std
    elif name.startswith("jpeg_"):
        from PIL import Image
        import io
        quality=int(name.split("_")[1]); pil=TF.to_pil_image(rgb.detach().cpu()); buffer=io.BytesIO();pil.save(buffer,format="JPEG",quality=quality,subsampling=0,optimize=False);buffer.seek(0);decoded=Image.open(buffer).convert("RGB");decoded_rgb=(TF.pil_to_tensor(decoded).float()/255).to(value.device);result=(decoded_rgb-mean)/std;metadata["quality"]=quality
    else: raise ValueError(f"unknown transformation: {name}")
    return result.unsqueeze(0),metadata


def _as_chw_map(image: torch.Tensor) -> torch.Tensor:
    if image.ndim!=2: raise NNSV2Error("inverse map must be 2D before transform")
    return image.unsqueeze(0)


def _restore_2d_map(image: torch.Tensor) -> torch.Tensor:
    if image.ndim==3 and image.shape[0]==1: image=image.squeeze(0)
    if image.ndim!=2: raise NNSV2Error("inverse map transform did not restore 2D map")
    return image.contiguous()


def inverse_map(value: torch.Tensor,metadata: Mapping[str,Any]) -> tuple[np.ndarray,np.ndarray]:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    image=pixel_map(value); mask=torch.ones_like(image,dtype=torch.bool); kind=metadata["inverse"]
    if kind=="translate":
        amount=int(metadata["amount"]); axis=metadata["axis"]; restored=torch.zeros_like(image);valid=torch.zeros_like(mask)
        if axis=="x":
            if amount>0: restored[:,:-amount]=image[:,amount:];valid[:,:-amount]=True
            else: restored[:,-amount:]=image[:,:amount];valid[:,-amount:]=True
        else:
            if amount>0: restored[:-amount,:]=image[amount:,:];valid[:-amount,:]=True
            else: restored[-amount:,:]=image[:amount,:];valid[-amount:,:]=True
        image,mask=restored,valid
    elif kind=="horizontal_flip": image=_restore_2d_map(TF.hflip(_as_chw_map(image)))
    elif kind=="rotate":
        image=_restore_2d_map(TF.rotate(_as_chw_map(image),-float(metadata["angle"]),interpolation=InterpolationMode.BILINEAR,fill=0.0));r0,r1,c0,c1=metadata["valid_box"];mask[:]=False;mask[r0:r1,c0:c1]=True
    elif kind=="resize_224_to_216_then_pad4":
        restored=torch.zeros_like(image);restored[4:220,4:220]=_restore_2d_map(TF.resize(_as_chw_map(image),[216,216],interpolation=InterpolationMode.BILINEAR,antialias=False));image=restored;mask[:]=False;mask[4:220,4:220]=True
    return image.numpy(),mask.numpy()
