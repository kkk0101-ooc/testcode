from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vitnns import stage_x_nns_v2 as nns


class LinearToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=True, dtype=torch.float64)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value.flatten(1))

    def capture(self, value: torch.Tensor, target: int):
        flat = value.flatten(1)
        return self.linear(flat)[:, target], {"input": flat}


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads, self.head_dim, self.scale = 2, 2, 2**-0.5
        self.fused_attn = False
        self.qkv = nn.Linear(4, 12, dtype=torch.float64)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Linear(4, 4, dtype=torch.float64)
        self.proj_drop = nn.Dropout(0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = value.shape
        query, key, val = self.qkv(value).reshape(
            batch, tokens, 3, 2, 2
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attention = self.attn_drop(((query * self.scale) @ key.transpose(-2, -1)).softmax(-1))
        return self.proj_drop(self.proj((attention @ val).transpose(1, 2).reshape(batch, tokens, channels)))


class TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8, dtype=torch.float64)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(8, 4, dtype=torch.float64)
        self.drop2 = nn.Dropout(0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(value))))))


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(4, dtype=torch.float64)
        self.attn = TinyAttention()
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(4, dtype=torch.float64)
        self.mlp = TinyMlp()
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.drop_path1(self.ls1(self.attn(self.norm1(value))))
        return value + self.drop_path2(self.ls2(self.mlp(self.norm2(value))))


class TinyViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(1, 4, 2, 2, dtype=torch.float64)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 4, dtype=torch.float64))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, 4, dtype=torch.float64))
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([TinyBlock()])
        self.norm = nn.LayerNorm(4, dtype=torch.float64)
        self.head = nn.Linear(4, 3, dtype=torch.float64)
        self.embed_dim, self.num_prefix_tokens = 4, 1
        self.patch_embed.patch_size = (2, 2)
        self.patch_embed.grid_size = (2, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.patch_embed(value).flatten(2).transpose(1, 2)
        value = torch.cat((self.cls_token.expand(value.shape[0], -1, -1), value), 1)
        value = self.norm_pre(value + self.pos_embed)
        value = self.blocks[0](value)
        return self.head(self.norm(value)[:, 0])


def factory(captures):
    return nns.timm_cut_specs(captures, 1, 2)


def test_linear_bias_and_signed_contributions_are_exact():
    model = LinearToy()
    with torch.no_grad():
        model.linear.weight[0] = torch.tensor([1.0, -2.0, 3.0, -4.0])
        model.linear.bias[0] = 9.0
    value = torch.tensor([[[[0.2, 0.4], [0.1, -0.3]]]], dtype=torch.float64)
    cut = nns.CutSpec("input", (
        nns.ComponentSpec("a", "input", -1, 0, 2),
        nns.ComponentSpec("b", "input", -1, 2, 4),
    ))
    target = nns.resolve_target(model, value)
    attribution, cuts, plan = nns.integrate(
        model.capture, value, torch.zeros_like(value), target.predicted_class,
        (cut,), 8, 2, 2, detailed_cut_ids=("input",), retained_components="all",
    )
    torch.testing.assert_close(attribution, cuts["input"].reconstructed, atol=1e-7, rtol=1e-7)
    assert float(attribution.min()) < 0 < float(attribution.max())
    assert plan.accumulation_device == "cpu" and not plan.full_component_maps_on_gpu


def test_target_keeps_prediction_truth_runner_up_and_margin_separate():
    model = LinearToy()
    value = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    record = nns.resolve_target(model, value, 1).to_dict()
    assert record["true_class"] == 1
    assert record["predicted_class"] != record["runner_up_class"]
    assert record["clean_margin"] == pytest.approx(
        record["predicted_logit"] - record["runner_up_logit"]
    )
    assert record["separate_target_denominators"]


def test_partition_rejects_gap_overlap_and_bool_axis():
    tensor = torch.ones(1, 4)
    with pytest.raises(nns.NNSV2Error):
        nns.validate_partition({"x": tensor}, nns.CutSpec("gap", (nns.ComponentSpec("a", "x", -1, 0, 2),)))
    with pytest.raises(nns.NNSV2Error):
        nns.validate_partition({"x": tensor}, nns.CutSpec("overlap", (
            nns.ComponentSpec("a", "x", -1, 0, 3), nns.ComponentSpec("b", "x", -1, 2, 4)
        )))
    with pytest.raises(nns.NNSV2Error):
        nns.ComponentSpec("a", "x", True, 0, 1).validate()


def test_complete_cut_multiplication_softmax_and_residual():
    class Graph(nn.Module):
        def forward(self, value):
            flat = value.flatten(1)
            left, right = flat + 0.2, flat.square() - 0.1
            product = left * right
            probabilities = product.softmax(-1)
            residual = probabilities + flat
            logits = torch.stack((residual.sum(-1), -residual.sum(-1)), 1)
            return logits

        def capture(self, value, target):
            flat = value.flatten(1)
            left, right = flat + 0.2, flat.square() - 0.1
            product = left * right
            probabilities = product.softmax(-1)
            residual = probabilities + flat
            logits = torch.stack((residual.sum(-1), -residual.sum(-1)), 1)
            return logits[:, target], {"left": left, "right": right, "prob": probabilities, "flat": flat, "residual": residual}

    value = torch.tensor([[[[0.2, -0.4], [0.3, 0.7]]]], dtype=torch.float64)
    model = Graph()
    target = nns.resolve_target(model, value).predicted_class
    cuts = (
        nns.CutSpec("multiply", (
            nns.ComponentSpec("identity", "flat", cotangent_key="residual"),
            nns.ComponentSpec("left", "left"), nns.ComponentSpec("right", "right")
        )),
        nns.CutSpec("softmax", (
            nns.ComponentSpec("identity", "flat", cotangent_key="residual"),
            nns.ComponentSpec("probability", "prob")
        )),
        nns.CutSpec("residual", (
            nns.ComponentSpec("identity", "flat", cotangent_key="residual"),
            nns.ComponentSpec("branch", "prob"),
        )),
    )
    attribution, results, _ = nns.integrate(
        model.capture, value, torch.zeros_like(value), target, cuts, 32, 2, 2,
        detailed_cut_ids=("multiply", "residual"), retained_components="all",
    )
    for result in results.values():
        torch.testing.assert_close(attribution, result.reconstructed, atol=1e-6, rtol=1e-6)
    assert set(results["residual"].component_maps) == {"identity", "branch"}


def test_tiny_vit_all_cuts_cover_layernorm_gelu_and_restore_state():
    torch.manual_seed(4)
    model = TinyViT().train()
    value = torch.linspace(-0.7, 0.9, 16, dtype=torch.float64).reshape(1, 1, 4, 4)
    states = [module.training for module in model.modules()]
    result = nns.explain(
        model, value, true_class=2, steps=(8, 16, 32, 64), alpha_chunk=2,
        component_chunk=4, cut_factory=factory,
        detailed_cut_ids=("b0.heads", "b0.mlp_hidden"),
        retained_components={"b0.heads": ("residual", "head_0", "head_1")},
    )
    assert result.converged and result.signed_rgb_map.shape == (4, 4)
    assert len(result.cuts) == 14 and all(cut.passed for cut in result.cuts.values())
    assert [module.training for module in model.modules()] == states
    assert all(parameter.grad is None for parameter in model.parameters())
    assert result.constant_direct_terms == {"bias": 0.0, "class_token": 0.0, "position_embedding": 0.0}


def test_total_matches_independent_ig_identity_control_exactly():
    torch.manual_seed(7)
    model, value = TinyViT(), torch.randn(1, 1, 4, 4, dtype=torch.float64)
    result = nns.explain(model, value, steps=(16, 32, 64), alpha_chunk=2, cut_factory=factory)
    control = nns.input_ig(model, value, torch.zeros_like(value), result.target.predicted_class, result.steps, 2)
    torch.testing.assert_close(result.attribution, control, atol=1e-6, rtol=1e-6)
    assert result.identity_max_abs_error <= 1e-6


def test_head_branch_neuron_sums_are_native_and_not_layer_summed():
    model, value = TinyViT(), torch.randn(1, 1, 4, 4, dtype=torch.float64)
    result = nns.explain(
        model, value, steps=(16, 32, 64, 128), alpha_chunk=2,
        component_chunk=4, cut_factory=factory,
        detailed_cut_ids=("b0.heads", "b0.mlp_hidden"), retained_components="all",
    )
    assert set(result.cuts["b0.heads"].component_maps) == {"residual", "head_0", "head_1"}
    for cut_id in ("b0.heads", "b0.mlp_hidden"):
        reconstructed = sum(result.cuts[cut_id].component_maps.values(), torch.zeros_like(result.attribution))
        torch.testing.assert_close(reconstructed, result.attribution, atol=1e-5, rtol=1e-5)


def test_target_swap_progressive_randomization_and_subpatch_location():
    torch.manual_seed(9)
    model = TinyViT().eval()
    value = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    zero = torch.zeros_like(value)
    first = nns.input_ig(model, value, zero, 0, 16, 2)
    swapped = nns.input_ig(model, value, zero, 1, 16, 2)
    assert not torch.allclose(first, swapped)
    original = copy.deepcopy(model.state_dict())
    nn.init.normal_(model.head.weight)
    randomized_head = nns.input_ig(model, value, zero, 0, 16, 2)
    nn.init.normal_(model.blocks[0].attn.qkv.weight)
    randomized_deeper = nns.input_ig(model, value, zero, 0, 16, 2)
    model.load_state_dict(original)
    assert not torch.allclose(first, randomized_head)
    assert not torch.allclose(randomized_head, randomized_deeper)
    left, right = value.clone(), value.clone()
    left.zero_(); right.zero_(); left[0, 0, 0, 0] = 1; right[0, 0, 0, 1] = 1
    assert nns.array_digest(nns.input_ig(model, left, zero, 0, 8, 2)) != nns.array_digest(nns.input_ig(model, right, zero, 0, 8, 2))


def test_scientific_and_display_types_are_isolated():
    scientific = nns.ScientificArray(torch.ones(3, 4, 4), "signed")
    display = nns.DisplayArray(np.ones((4, 4), dtype=np.uint8), "coolwarm")
    assert not isinstance(scientific, nns.DisplayArray) and not isinstance(display, nns.ScientificArray)
    with pytest.raises(nns.NNSV2Error):
        nns.ScientificArray(torch.ones(3, 4, 4, dtype=torch.float64), "signed")
    with pytest.raises(nns.NNSV2Error):
        nns.DisplayArray(np.ones((4, 4), dtype=np.float32), "coolwarm")


def test_nonconvergence_and_memory_failure_are_typed():
    model, value = TinyViT(), torch.randn(1, 1, 4, 4, dtype=torch.float64)
    old = (nns.INPUT_ATOL, nns.INPUT_RTOL, nns.CUT_PIXEL_ATOL, nns.CUT_PIXEL_RTOL, nns.CUT_TOTAL_ATOL, nns.CUT_TOTAL_RTOL)
    try:
        nns.INPUT_ATOL = nns.INPUT_RTOL = nns.CUT_PIXEL_ATOL = nns.CUT_PIXEL_RTOL = nns.CUT_TOTAL_ATOL = nns.CUT_TOTAL_RTOL = 0
        with pytest.raises(nns.NNSV2ConvergenceError) as error:
            nns.explain(model, value, steps=(1, 2), alpha_chunk=1, cut_factory=factory)
        assert error.value.status == "NO_GO_CLASS_LOGIT_COMPLETENESS" and len(error.value.attempts) == 2
    finally:
        nns.INPUT_ATOL, nns.INPUT_RTOL, nns.CUT_PIXEL_ATOL, nns.CUT_PIXEL_RTOL, nns.CUT_TOTAL_ATOL, nns.CUT_TOTAL_RTOL = old
    production = torch.zeros(1, 3, 224, 224)
    with pytest.raises(nns.NNSV2MemoryPlanError):
        nns.deterministic_memory_plan(production, cap_bytes=1)


def test_registered_architectures_are_hook_compatible_without_weights():
    import timm
    for model_id in (
        "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "deit_small_patch16_224.fb_in1k",
        "deit_base_patch16_224.fb_in1k",
        "vit_base_patch32_224.augreg_in21k_ft_in1k",
    ):
        model = timm.create_model(model_id, pretrained=False)
        report = nns.architecture_contract(model, model_id)
        assert report["status"] == "PASS" and report["depth"] == 12


def test_no_surrogate_subtraction_or_cross_layer_sum_in_source():
    source = Path(nns.__file__).read_text()
    assert nns.METHOD_ID == "nns_v2_complete_cut_path_jacobian_ig"
    assert "total - branch" not in source
    assert "fixed_attention_intervention" not in source
    assert "layer_views_must_not_be_summed" in source
