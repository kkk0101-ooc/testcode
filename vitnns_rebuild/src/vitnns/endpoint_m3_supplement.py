"""Supplementary M3 numerical summaries with explicit scale semantics."""
from __future__ import annotations
import numpy as np

def cancellation_metrics(value) -> dict[str,float|None]:
    """Decompose signed cancellation across nested RGB/pixel/patch scales."""
    x=np.asarray(value,dtype=np.float64)
    if x.shape!=(3,224,224) or not np.isfinite(x).all():raise ValueError("MAP_IDENTITY")
    coordinate_absolute=float(np.abs(x).sum())
    if coordinate_absolute==0:
        return {k:None for k in ("within_rgb_channels","within_patch_after_rgb_sum","across_patch_signed","whole_image_cancellation","net_ratio","pixel_signed_mass","patch_signed_mass","coordinate_absolute_mass","absolute_net","survival_identity_error")}
    pixel=x.sum(0);pixel_signed=float(np.abs(pixel).sum())
    patch=pixel.reshape(14,16,14,16).sum((1,3));patch_signed=float(np.abs(patch).sum());absolute_net=float(abs(patch.sum()))
    rgb=1-pixel_signed/coordinate_absolute
    within_patch=None if pixel_signed==0 else 1-patch_signed/pixel_signed
    across_patch=None if patch_signed==0 else 1-absolute_net/patch_signed
    net=absolute_net/coordinate_absolute;whole=1-net
    survival=(pixel_signed/coordinate_absolute)*(0 if pixel_signed==0 else patch_signed/pixel_signed)*(0 if patch_signed==0 else absolute_net/patch_signed)
    return {"within_rgb_channels":rgb,"within_patch_after_rgb_sum":within_patch,"across_patch_signed":across_patch,"whole_image_cancellation":whole,"net_ratio":net,"pixel_signed_mass":pixel_signed,"patch_signed_mass":patch_signed,"coordinate_absolute_mass":coordinate_absolute,"absolute_net":absolute_net,"survival_identity_error":abs(survival-net)}

def tolerance_ratio(observed:float,expected:float,atol:float,rtol:float)->float:
    return abs(float(observed)-float(expected))/(atol+rtol*abs(float(expected)))
