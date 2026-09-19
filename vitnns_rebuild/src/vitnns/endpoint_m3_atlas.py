"""Streaming token-channel Endpoint summaries for M3."""
from __future__ import annotations
import math
from typing import Callable
import numpy as np
import torch
from torch import Tensor, nn


def summarize_maps(maps: Tensor, tokens: tuple[int,...], channels: tuple[int,...], block: int) -> list[dict]:
    """Summarize [units,3,224,224] signed maps without an epsilon fiction."""
    if maps.ndim!=4 or maps.shape[0]!=len(tokens)*len(channels) or maps.shape[1:]!=(3,224,224):raise ValueError("MAP_SHAPE")
    patch_signed=maps.reshape(len(tokens)*len(channels),3,14,16,14,16).sum((1,3,5))
    patch_absolute=maps.abs().reshape(len(tokens)*len(channels),3,14,16,14,16).sum((1,3,5))
    flat=maps.flatten(1);absolute=flat.abs().sum(1);signed=flat.sum(1);positive=flat.clamp_min(0).sum(1);negative=(-flat.clamp_max(0)).sum(1);l2=flat.square().sum(1).sqrt()
    rows=[]
    for index,(token,channel) in enumerate((t,c) for t in tokens for c in channels):
        zero=bool(absolute[index]==0); p=patch_absolute[index]/absolute[index] if not zero else None
        entropy=None if zero else float(-(p[p>0]*p[p>0].log()).sum()/math.log(196))
        cancellation=None if zero else float(1-signed[index].abs()/absolute[index])
        rows.append({
            "block":block,"token":token,"channel":channel,"token_type":"cls" if token==0 else "patch",
            "signed_sum":float(signed[index]),"absolute_mass":float(absolute[index]),"l2_norm":float(l2[index]),
            "positive_mass":float(positive[index]),"negative_mass":float(negative[index]),"zero_gradient":zero,
            "cancellation":cancellation,"normalized_spatial_entropy":entropy,
            "patch_absolute_mass_196":patch_absolute[index].flatten().cpu().tolist(),
            "patch_signed_sum_196":patch_signed[index].flatten().cpu().tolist(),"numeric_status":"PASS_FINITE",
        })
    return rows


def endpoint_chunk(captured: Tensor, clean: Tensor, units: tuple[tuple[int,int],...]) -> Tensor:
    """Batched VJPs for independent tensor coordinates on a single graph."""
    scores=torch.stack([captured[0,t,c] for t,c in units])
    identity=torch.eye(len(units),device=scores.device,dtype=scores.dtype)
    gradients=torch.autograd.grad(scores,clean,grad_outputs=identity,is_grads_batched=True,retain_graph=True)[0][:,0]
    maps=clean.detach()[0].unsqueeze(0)*gradients
    if not bool(torch.isfinite(maps).all()):raise ValueError("NONFINITE_ENDPOINT")
    return maps


def scan_model(
    model: nn.Module, image: Tensor, tokens: tuple[int,...], channels_by_block: dict[int,tuple[int,...]],
    chunk_size: int, progress: Callable[[],None]|None=None, save_full: bool=False,
    direction_arrays: tuple[np.ndarray,...]=(), finite_difference_epsilons: tuple[float,...]=(),
) -> tuple[list[dict],dict[str,np.ndarray],dict]:
    if image.shape!=(1,3,224,224):raise ValueError("IMAGE_SHAPE")
    captured={};handles=[]
    for block in channels_by_block:
        handles.append(model.blocks[block].register_forward_hook(lambda _m,_a,out,b=block:captured.__setitem__(b,out)))
    clean=image.detach().clone().requires_grad_(True)
    try:
        with torch.enable_grad():logits=model(clean)
        rows=[];full={};checks=[];analytic={direction:[] for direction in range(len(direction_arrays))}
        for block in sorted(channels_by_block):
            channels=channels_by_block[block];all_units=tuple((t,c) for t in tokens for c in channels)
            block_maps=[]
            for start in range(0,len(all_units),chunk_size):
                units=all_units[start:start+chunk_size];maps=endpoint_chunk(captured[block],clean,units);block_maps.append(maps.detach().cpu());
                if progress:progress()
            block_maps=torch.cat(block_maps);rows.extend(summarize_maps(block_maps,tokens,channels,block))
            if save_full and block in (0,5,11):
                for token in tokens[:2]:
                    index=tokens.index(token)*len(channels)
                    full[f"b{block:02d}_t{token:03d}_c{channels[0]:04d}"]=block_maps[index].numpy().astype(np.float32)
            # Exact same-graph scalar VJP control for the frozen 8-unit subset.
            for token in tokens[:2]:
                for channel in channels[:4]:
                    expected=block_maps[tokens.index(token)*len(channels)+channels.index(channel)].to(image.device)
                    gradient=torch.autograd.grad(captured[block][0,token,channel],clean,retain_graph=True)[0][0]
                    observed=clean.detach()[0]*gradient;error=float((observed-expected).abs().max())
                    checks.append({"block":block,"token":token,"channel":channel,"batched_vjp_scalar_vjp_max_abs":error,"scalar_endpoint_max_abs":float(observed.abs().max())})
                    for direction,array in enumerate(direction_arrays):
                        vector=torch.as_tensor(array,device=gradient.device,dtype=gradient.dtype)
                        analytic[direction].append(float((gradient*vector).sum()))
        finite_difference=[]
        if direction_arrays and finite_difference_epsilons:
            validation_units=[(block,token,channel) for block in sorted(channels_by_block) for token in tokens[:2] for channel in channels_by_block[block][:4]]
            def selected_values(input_tensor):
                with torch.no_grad():model(input_tensor)
                return torch.stack([captured[block][0,token,channel] for block,token,channel in validation_units])
            for direction,array in enumerate(direction_arrays):
                vector=torch.as_tensor(array,device=clean.device,dtype=clean.dtype).unsqueeze(0)
                for epsilon in finite_difference_epsilons:
                    plus=selected_values(clean.detach()+epsilon*vector);minus=selected_values(clean.detach()-epsilon*vector);fd=(plus-minus)/(2*epsilon)
                    expected=torch.tensor(analytic[direction],device=fd.device,dtype=fd.dtype);relative=(fd-expected).abs()/(expected.abs()+torch.finfo(fd.dtype).eps)
                    finite_difference.append({"direction":direction,"epsilon":epsilon,"max_abs":float((fd-expected).abs().max()),"median_relative":float(relative.median()),"max_relative":float(relative.max()),"per_target_analytic":expected.cpu().tolist(),"per_target_finite_difference":fd.cpu().tolist()})
        return rows,full,{"logits_sha256":__import__('hashlib').sha256(logits.detach().cpu().numpy().tobytes()).hexdigest(),"checks":checks,"finite_difference":finite_difference}
    finally:
        for handle in handles:handle.remove()


def derived_metrics(row: dict) -> dict:
    """Compute locality and distance from the stored 196-vector."""
    result=dict(row);mass=np.asarray(row["patch_absolute_mass_196"],dtype=np.float64)
    if row["zero_gradient"]:
        result.update(own_token_locality=None,own_token_locality_above_uniform=None,input_dependency_distance=None);return result
    p=mass/mass.sum();token=row["token"]
    if token==0:result.update(own_token_locality=None,own_token_locality_above_uniform=None,input_dependency_distance=None);return result
    tr,tc=divmod(token-1,14);grid=np.indices((14,14));dr=np.abs(grid[0]-tr);dc=np.abs(grid[1]-tc);neighbor=np.maximum(dr,dc)<=1
    locality=float(p.reshape(14,14)[neighbor].sum());uniform=float(neighbor.sum()/196)
    distance=float((p.reshape(14,14)*np.sqrt(dr**2+dc**2)/(13*math.sqrt(2))).sum())
    result.update(own_token_locality=locality,own_token_locality_above_uniform=locality-uniform,input_dependency_distance=distance);return result


def pack_rows(rows: list[dict]) -> dict[str,np.ndarray]:
    """Closed columnar representation; NA is encoded as IEEE NaN plus zero flag."""
    scalar_int=("block","token","channel");scalar_float=("signed_sum","absolute_mass","l2_norm","positive_mass","negative_mass","cancellation","normalized_spatial_entropy")
    packed={name:np.asarray([r[name] for r in rows],dtype=np.int16) for name in scalar_int}
    packed.update({name:np.asarray([np.nan if r[name] is None else r[name] for r in rows],dtype=np.float32) for name in scalar_float})
    packed["zero_gradient"]=np.asarray([r["zero_gradient"] for r in rows],dtype=np.bool_);packed["numeric_status"]=np.asarray([1 if r["numeric_status"]=="PASS_FINITE" else 0 for r in rows],dtype=np.uint8)
    packed["patch_absolute_mass_196"]=np.asarray([r["patch_absolute_mass_196"] for r in rows],dtype=np.float32);packed["patch_signed_sum_196"]=np.asarray([r["patch_signed_sum_196"] for r in rows],dtype=np.float32)
    return packed
