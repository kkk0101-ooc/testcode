from dataclasses import replace

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from vitnns.endpoint_nns import endpoint_learning_image, paper_relu_weight_propagation

from vitnns.endpoint_vit import (
    EndpointTarget, attention_endpoint_components, decompose_endpoint_vit,
    default_endpoint_targets, scan_endpoint_vit,
)


@pytest.fixture
def tiny():
    torch.manual_seed(19)
    model = VisionTransformer(
        img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3,
        mlp_ratio=2, num_classes=4,
    ).double().eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    image = torch.randn(1, 3, 8, 8, dtype=torch.float64)
    return model, image


def test_native_scan_identity_residual_closure_and_single_forward(tiny):
    model, image = tiny
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    expected_logits = model(image).detach()
    calls.clear()
    result = scan_endpoint_vit(model, image, default_endpoint_targets(model, blocks=(0, 1)))
    assert len(calls) == result.forward_count == 1
    handle.remove()
    torch.testing.assert_close(result.logits, expected_logits, atol=0, rtol=0)
    clean = image.clone().requires_grad_()
    target = result.units["class_logit"].target.class_id
    gradient = torch.autograd.grad(model(clean)[:, target].sum(), clean)[0]
    torch.testing.assert_close(result.units["class_logit"].attribution, image * gradient, atol=0, rtol=0)
    for block in (0, 1):
        a = {name: unit.attribution for name, unit in result.units.items()}
        torch.testing.assert_close(a[f"b{block}_residual1"], a[f"b{block}_input"] + a[f"b{block}_attention"], atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(a[f"b{block}_residual2"], a[f"b{block}_residual1"] + a[f"b{block}_mlp"], atol=1e-12, rtol=1e-10)
    assert result.units["patch_embedding"].token_values.shape == (1, 4)
    assert result.units["b0_head1"].token_values.shape == (1, 5)
    assert result.units["class_margin"].target.class_id != result.units["class_margin"].target.other_class_id


@pytest.mark.parametrize("kind", ["logit", "head_channel", "token_channel"])
def test_native_targets_against_directional_finite_difference(tiny, kind):
    model, image = tiny
    specs = {
        "logit": EndpointTarget("s", "", kind=kind, class_id=1),
        "head_channel": EndpointTarget("s", "blocks.1.attn.proj", kind=kind, location="input", head=2, channel=1, token=0),
        "token_channel": EndpointTarget("s", "blocks.1.mlp.act", kind=kind, channel=3, token=2),
    }
    spec = specs[kind]
    result = scan_endpoint_vit(model, image, [spec]).units["s"]
    direction = torch.randn_like(image)
    direction /= direction.norm()

    def scalar(value):
        observed = []
        handle = None
        if kind == "head_channel":
            handle = model.get_submodule(spec.module).register_forward_pre_hook(
                lambda _m, args: observed.append(args[0][:, 0, 9])
            )
        elif kind == "token_channel":
            handle = model.get_submodule(spec.module).register_forward_hook(
                lambda _m, _args, output: observed.append(output[:, 2, 3])
            )
        try:
            with torch.no_grad():
                logits = model(value)
                return logits[:, 1] if kind == "logit" else observed[0]
        finally:
            if handle is not None:
                handle.remove()

    epsilon = 1e-5
    finite = (scalar(image + epsilon * direction) - scalar(image - epsilon * direction)) / (2 * epsilon)
    analytic = (result.gradient * direction).sum().reshape(1)
    torch.testing.assert_close(analytic, finite, atol=1e-9, rtol=2e-5)


def test_scan_restores_mixed_training_states_and_hooks_on_error(tiny):
    model, image = tiny
    model.train()
    model.blocks[0].eval()
    model.head.weight.requires_grad_(False)
    states = [m.training for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    hook_counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    with pytest.raises(ValueError, match="channel out of range"):
        scan_endpoint_vit(model, image, [EndpointTarget("bad", "norm", channel=100)])
    assert [m.training for m in model.modules()] == states
    assert [p.requires_grad for p in model.parameters()] == flags
    assert [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()] == hook_counts


def test_scan_rejects_ambiguous_specs(tiny):
    model, image = tiny
    spec = EndpointTarget("a", "norm")
    for specs, message in [
        ([spec, spec], "unique"),
        ([replace(spec, kind="head_channel", head=0)], "attn.proj input"),
        ([EndpointTarget("a", "", kind="margin", class_id=0, other_class_id=0)], "different"),
    ]:
        with pytest.raises(ValueError, match=message):
            scan_endpoint_vit(model, image, specs)
    with pytest.raises(ValueError, match="one floating-point image"):
        scan_endpoint_vit(model, image.expand(2, -1, -1, -1), [spec])


@pytest.mark.parametrize("block", [0, 1])
def test_native_attention_qkv_edges_close_and_match_head_scan(tiny, block):
    model, image = tiny
    parts = attention_endpoint_components(model, image, block=block, head=2, channel=1)
    spec = EndpointTarget("head", f"blocks.{block}.attn.proj", kind="head_channel",
                          location="input", head=2, channel=1)
    result = scan_endpoint_vit(model, image, [spec]).units["head"]
    torch.testing.assert_close(parts["full"], result.attribution, atol=0, rtol=0)
    torch.testing.assert_close(parts["full"], parts["query"] + parts["key"] + parts["value"], atol=1e-12, rtol=1e-10)
    assert parts["key"].abs().sum() > 0
    if block == 1:
        assert parts["query"].abs().sum() > 0
    else:
        # Before the first attention the CLS query is an image-independent parameter.
        torch.testing.assert_close(parts["query"], torch.zeros_like(image), atol=0, rtol=0)


@pytest.mark.parametrize("block", [0, 1])
@pytest.mark.parametrize("kind", ["logit", "margin"])
def test_final_target_edge_cuts_include_bypass_without_duplicate_branch(tiny, block, kind):
    model, image = tiny
    target = EndpointTarget("terminal", "", kind=kind, class_id=1, other_class_id=2)
    result = decompose_endpoint_vit(model, image, block=block, target=target)
    scan = scan_endpoint_vit(model, image, [target])
    torch.testing.assert_close(result.full, scan.units["terminal"].attribution, atol=0, rtol=0)
    for cut in result.cuts.values():
        total = sum(result.components[key] for key in cut)
        torch.testing.assert_close(total, result.full, atol=1e-12, rtol=1e-9)
    heads = sum(result.components[f"head_{head}"] for head in range(3))
    torch.testing.assert_close(heads, result.components["attention"], atol=1e-12, rtol=1e-9)
    assert result.components["identity"].abs().sum() > 0
    assert result.components["attention"].abs().sum() > 0
    assert max(result.closure_relative_l2.values()) < 1e-9


def test_original_cnn_weight_gate_propagation_matches_same_endpoint_principle():
    # Conv -> ReLU -> linear readout, with a nonzero activated path.
    x = torch.tensor([[[[1., 2., 3.], [4., -2., 1.], [2., 0., 5.]]]], dtype=torch.float64)
    kernel = torch.tensor([[[[0.5, -0.3], [0.2, 0.4]]]], dtype=torch.float64)
    readout = torch.tensor([[1., -2., 0.5, 0.7]], dtype=torch.float64)
    basis = torch.eye(9, dtype=torch.float64).reshape(9, 1, 3, 3)
    matrix = torch.nn.functional.conv2d(basis, kernel).flatten(1).T
    pre = x.flatten(1) @ matrix.T
    paper = paper_relu_weight_propagation(x.flatten(1), (matrix, readout), (pre,))
    def callback(value):
        conv = torch.nn.functional.conv2d(value, kernel)
        return torch.relu(conv).flatten(1) @ readout.T
    result = endpoint_learning_image(callback, x)
    assert paper.abs().sum() > 1
    torch.testing.assert_close(result.attribution.flatten(1), paper, atol=1e-12, rtol=1e-12)
