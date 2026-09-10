from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator

from vitnns import stage_y_y1 as y1

ROOT = Path(__file__).resolve().parents[1]


def test_run_spec_is_closed_and_frozen_before_outcome_access() -> None:
    value = json.loads((ROOT / "config/stage_y/y1_run_spec.json").read_text())
    schema = json.loads((ROOT / "config/stage_y/y1_run_spec.schema.json").read_text())
    Draft202012Validator(schema).validate(value)
    assert value["conditions"] == ["clean", "horizontal_flip"]
    assert value["repeats"] == [1, 2]
    assert value["quadrature"]["attempts"] == list(y1.STEPS)
    assert value["planner"]["full_pixel_by_hidden_dense_jacobian"] is False
    assert value["access"]["formal_outcome_access"] is False


def test_signed_artifact_and_array_contract_fail_closed() -> None:
    value = y1.signed("fixture", "PASS", {"a": 1})
    y1.validate_signed(value)
    changed = dict(value); changed["payload"] = {"a": 2}
    with pytest.raises(y1.StageYY1Error): y1.validate_signed(changed)
    valid = np.zeros((56,56), dtype=np.float32)
    y1.validate_scientific_array(valid, (56,56))
    for bad in (np.zeros((55,56),np.float32), np.zeros((56,56),np.float64), np.full((56,56),np.nan,np.float32)):
        with pytest.raises(y1.StageYY1Error): y1.validate_scientific_array(bad, (56,56))


def test_array_digest_binds_dtype_shape_axes_and_bytes() -> None:
    a = np.arange(12,dtype=np.float32).reshape(3,4)
    assert y1.array_digest(a) == y1.array_digest(a.copy())
    assert y1.array_digest(a) != y1.array_digest(a.T.copy())
    assert y1.array_digest(a) != y1.array_digest(a.astype(np.float64))


def test_transform_forward_inverse_masks_all_nine_and_flip_exact() -> None:
    image = torch.linspace(-1,1,3*224*224,dtype=torch.float32).reshape(1,3,224,224)
    fixture = torch.arange(224*224,dtype=torch.float32).reshape(224,224)
    for name in y1.TRANSFORMS:
        transformed, metadata = y1.transform_image(image, name, "order-44")
        assert transformed.shape == image.shape and torch.isfinite(transformed).all()
        restored, mask = y1.inverse_map(fixture if name != "horizontal_flip" else torch.fliplr(fixture), metadata)
        assert restored.shape == mask.shape == (224,224) and mask.dtype == np.bool_
        if name == "horizontal_flip":
            np.testing.assert_array_equal(restored, fixture.numpy()); assert mask.all()
        if name.startswith("translate") or name.startswith("rotate"):
            assert not mask.all() and mask.any()


def test_hash_seeded_tie_permutations_are_deterministic() -> None:
    left = y1.tie_permutation("order-44", 3, 17, 5)
    right = y1.tie_permutation("order-44", 3, 17, 5)
    assert np.array_equal(left,right) and sorted(left.tolist()) == list(range(16))
    assert not np.array_equal(left,y1.tie_permutation("order-44",3,17,6))


def test_mass_pool_and_hierarchical_operator_contracts() -> None:
    attribution = torch.arange(3*224*224,dtype=torch.float32).reshape(3,224,224) - 50000
    cells = y1.pool_cells(attribution)
    assert cells.shape == (56,56) and float(cells.sum()) == pytest.approx(float(attribution.double().sum()), rel=1e-6)
    summary = y1.mass_summary(cells.numpy())
    assert summary["positive_mass"] >= 0 and summary["negative_mass"] <= 0 and summary["absolute_mass"] >= 0
    order = y1.hierarchical_order(cells.numpy(),"order-44",0)
    control = y1.hierarchical_order(cells.numpy(),"order-44",0,0)
    assert sorted(order.tolist()) == sorted(control.tolist()) == list(range(3136))


class TinyAttention(torch.nn.Module):
    def __init__(self, width: int = 12, heads: int = 3):
        super().__init__(); self.num_heads=heads; self.head_dim=width//heads; self.scale=self.head_dim**-0.5
        self.qkv=torch.nn.Linear(width,width*3); self.q_norm=torch.nn.Identity(); self.k_norm=torch.nn.Identity(); self.attn_drop=torch.nn.Identity(); self.proj=torch.nn.Linear(width,width); self.proj_drop=torch.nn.Identity(); self.fused_attn=False

    def forward(self,x):
        b,n,c=x.shape; q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
        mixed=((q*self.scale)@k.transpose(-2,-1)).softmax(-1)@v
        return self.proj_drop(self.proj(mixed.transpose(1,2).reshape(b,n,c)))


class TinyMLP(torch.nn.Module):
    def __init__(self,width=12): super().__init__(); self.fc1=torch.nn.Linear(width,24); self.act=torch.nn.GELU(); self.fc2=torch.nn.Linear(24,width)
    def forward(self,x): return self.fc2(self.act(self.fc1(x)))


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.norm1=torch.nn.LayerNorm(12); self.attn=TinyAttention(); self.ls1=torch.nn.Identity(); self.drop_path1=torch.nn.Identity(); self.norm2=torch.nn.LayerNorm(12); self.mlp=TinyMLP(); self.ls2=torch.nn.Identity(); self.drop_path2=torch.nn.Identity()
    def forward(self,x):
        x=x+self.drop_path1(self.ls1(self.attn(self.norm1(x)))); return x+self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


class TinyViT(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.patch_embed=torch.nn.Identity(); self.patch_drop=torch.nn.Identity(); self.norm_pre=torch.nn.Identity(); self.blocks=torch.nn.ModuleList([TinyBlock()]); self.norm=torch.nn.LayerNorm(12); self.attn_pool=None; self.global_pool="token"; self.fc_norm=torch.nn.Identity(); self.head_drop=torch.nn.Identity(); self.head=torch.nn.Linear(12,4)
    def _pos_embed(self,x): return x
    def forward(self,x):
        for block in self.blocks: x=block(x)
        return self.head(self.norm(x)[:,0])


class TinyImageViT(TinyViT):
    def __init__(self):
        super().__init__(); self.patch_embed=torch.nn.Conv2d(3,12,16,16); self.cls=torch.nn.Parameter(torch.zeros(1,1,12))
    def _pos_embed(self,x):
        x=x.flatten(2).transpose(1,2); return torch.cat((self.cls.expand(x.shape[0],-1,-1),x),dim=1)
    def forward(self,x):
        x=self._pos_embed(self.patch_embed(x))
        for block in self.blocks: x=block(x)
        return self.head(self.norm(x)[:,0])


def test_native_head_projection_decomposition_and_ablation_point() -> None:
    model=TinyViT().eval(); image=torch.randn(2,5,12)
    original_blocks,original_heads=y1.BLOCKS,y1.HEADS
    try:
        y1.BLOCKS,y1.HEADS=1,3
        captured=y1.native_forward_capture(model,image)
        native=model(image)
        assert torch.allclose(native,captured.logits,atol=1e-6,rtol=1e-6)
        concat=captured.heads[0].permute(0,2,1,3).reshape(2,5,12)
        projected=model.blocks[0].attn.proj(concat)
        assert torch.allclose(projected,captured.tensors["b0.attention_branch"],atol=1e-6,rtol=1e-6)
        with y1.ablate_heads(model,0,[1]): changed=model(image)
        assert not torch.equal(native,changed)
    finally:
        y1.BLOCKS,y1.HEADS=original_blocks,original_heads


def test_analytic_grouped_jvp_vjp_reconstructs_total_without_dense_jacobian() -> None:
    torch.manual_seed(7); model=TinyImageViT().eval(); image=torch.randn(1,3,224,224)*0.02; baseline=torch.zeros_like(image)
    original_blocks,original_heads=y1.BLOCKS,y1.HEADS
    try:
        y1.BLOCKS,y1.HEADS=1,3; target=2
        result=y1.integrate_routes(model,image,baseline,target,2,alpha_chunk=4,source_patch_chunk=4,component_vjp_chunk=4)
        with torch.no_grad(): delta=float(model(image)[0,target]-model(baseline)[0,target])
        assert float(result.attribution.double().sum()) == pytest.approx(delta,abs=2e-4,rel=0.01)
        grouped={}
        for row in result.route_layout:
            routed=float(result.routes[:,row["start"]:row["stop"]].double().sum())
            grouped.setdefault(row["cut"],0.0); grouped[row["cut"]]+=routed
        # First residual contains identity plus 3 heads plus a redundant
        # attention aggregate view, so validate the complete pair separately.
        first=[row for row in result.route_layout if row["cut"]=="first_residual_addition"]
        identity=next(row for row in first if row["component"].endswith("identity"))
        aggregate=next(row for row in first if row["component"].endswith("attention_aggregate"))
        pair=sum(float(result.routes[:,row["start"]:row["stop"]].double().sum()) for row in (identity,aggregate))
        assert pair==pytest.approx(delta,abs=2e-4,rel=0.01)
        for cut in ("post_patch_embedding_tokens","block_input_tokens","second_residual_addition","block_output_tokens","final_norm"):
            assert grouped[cut]==pytest.approx(delta,abs=2e-4,rel=0.01),cut
        assert result.calls["jvp_calls"]==1 and result.calls["vjp_calls"]>0
    finally:
        y1.BLOCKS,y1.HEADS=original_blocks,original_heads


def test_budget_projection_uses_maximum_fixed_and_safety_rules() -> None:
    rows=[{"condition":"clean","gpu_active_seconds":10,"wall_seconds":20,"cpu_seconds":30},{"condition":"clean","gpu_active_seconds":12,"wall_seconds":18,"cpu_seconds":29},{"condition":"horizontal_flip","gpu_active_seconds":15,"wall_seconds":25,"cpu_seconds":35},{"condition":"horizontal_flip","gpu_active_seconds":14,"wall_seconds":26,"cpu_seconds":34}]
    result=y1.project_budget({"gpu_active_seconds":2,"wall_seconds":3,"cpu_seconds":4},rows,1000,2000)
    assert result["clean_unit_cost_maximum_rule"]["gpu_active_seconds"]==12
    assert result["perturbation_incremental_cost"]["gpu_active_seconds"]==3
    assert result["safety_multiplier"]==1.5 and result["status"]=="PASS"


def test_source_structure_forbids_dense_jacobian_and_uses_grouped_jvp_vjp() -> None:
    source=(ROOT/"src/vitnns/stage_y_y1.py").read_text(); ast.parse(source)
    assert "torch.func.jvp" in source and "torch.vmap" in source and "is_grads_batched=True" in source
    assert "torch.autograd.functional.jacobian" not in source and "torch.func.jacrev" not in source and "torch.func.jacfwd" not in source


@pytest.mark.parametrize("bad", [torch.full((1,3,224,224),float("nan")), torch.zeros(3,224,224), torch.zeros(1,3,223,224)])
def test_transform_malformed_inputs_fail_closed(bad: torch.Tensor) -> None:
    with pytest.raises(y1.StageYY1Error): y1.transform_image(bad.float(),"clean","order-44")
