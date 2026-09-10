from __future__ import annotations

import numpy as np
import pytest
import torch

from vitnns import stage_x_x2_metrics as metrics


def test_seed_is_deterministic_and_namespaced():
    assert metrics.seed64("x",1)==metrics.seed64("x",1)
    assert metrics.seed64("x",1)!=metrics.seed64("x",2)


def test_pixel_map_uses_method_support_without_interpolation():
    value=torch.arange(4,dtype=torch.float32).reshape(2,2)
    result=metrics.pixel_map(value,4,4)
    assert tuple(result.shape)==(4,4)
    assert torch.equal(result,torch.tensor([[0,0,1,1],[0,0,1,1],[2,2,3,3],[2,2,3,3]],dtype=torch.float32))


def test_map_metrics_identical_pass_exactly():
    value=np.arange(64,dtype=np.float32).reshape(8,8)-20
    result=metrics.map_metrics(value,value)
    assert result["signed_spearman"]==pytest.approx(1.0)
    assert result["top10_mass_jaccard"]==1.0
    assert result["signed_ssim"]==1.0
    assert result["normalized_positive_negative_mass_displacement"]==0.0
    assert result["center_of_mass_pixel_error"]==0.0


def test_noisy_linear_imputer_is_seed_deterministic_without_global_rng():
    image=torch.linspace(-1,1,3*8*8).reshape(3,8,8)
    mask=np.ones((8,8),dtype=np.float32);mask[2:5,3:6]=0
    one=metrics.noisy_linear_impute(image,mask,19);two=metrics.noisy_linear_impute(image,mask,19)
    assert torch.equal(one,two) and not torch.equal(one,image)
