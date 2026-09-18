from dataclasses import replace

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from vitnns.endpoint_nns import endpoint_learning_image, paper_relu_weight_propagation

from vitnns.endpoint_vit import (
    EndpointTarget, attention_endpoint_components, capture_unfused_attention_edges, decompose_endpoint_vit,
    default_endpoint_targets, numeric_parity, scan_endpoint_vit, validate_decomposition_cut,
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
    if kind == "logit":
        _, jvp = torch.func.jvp(lambda value: model(value)[:, 1], (image,), (direction,))
        torch.testing.assert_close(analytic, jvp, atol=1e-9, rtol=1e-7)


def test_float32_directional_finite_difference_uses_frozen_tolerance():
    torch.manual_seed(29)
    model = VisionTransformer(
        img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3,
        mlp_ratio=2, num_classes=4,
    ).float().eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    image = torch.randn(1, 3, 8, 8)
    result = scan_endpoint_vit(model, image, [EndpointTarget("s", "", kind="logit", class_id=1)]).units["s"]
    direction = torch.randn_like(image); direction /= direction.norm()
    analytic = (result.gradient * direction).sum()
    epsilon = 0.01
    finite = ((model(image + epsilon * direction)[:, 1] - model(image - epsilon * direction)[:, 1]) / (2 * epsilon)).sum()
    torch.testing.assert_close(analytic, finite, atol=0.002, rtol=0.02)


def test_local_direction_contract_uses_edge_displacement_jvp(tiny):
    model, image = tiny
    block = model.blocks[1]
    target_class = 1
    observed = {}
    handle = block.attn.proj.register_forward_pre_hook(lambda _m, args: observed.setdefault("z", args[0]))
    clean = image.clone().requires_grad_()
    try:
        score = model(clean)[:, target_class].sum()
    finally:
        handle.remove()
    z = observed["z"]
    g = torch.autograd.grad(score, z, retain_graph=True)[0]
    direction = torch.randn_like(image); direction /= direction.norm()
    _, jz = torch.func.jvp(
        lambda value: _capture_projection_input(model, value, 1), (image,), (direction,),
    )
    predicted = (g.detach() * jz).sum()
    epsilon = 1e-4
    def intervene(eta):
        handle = block.attn.proj.register_forward_pre_hook(
            lambda _m, _args: (z.detach() + eta * jz.detach(),),
        )
        try:
            return model(image)[:, target_class]
        finally:
            handle.remove()
    plus = intervene(epsilon)
    minus = intervene(-epsilon)
    finite = ((plus - minus) / (2 * epsilon)).sum()
    torch.testing.assert_close(predicted, finite, atol=1e-6, rtol=1e-4)


def _capture_projection_input(model, value, block):
    observed = []
    handle = model.blocks[block].attn.proj.register_forward_pre_hook(lambda _m, args: observed.append(args[0]))
    try:
        model(value)
        return observed[0]
    finally:
        handle.remove()


def test_energy_target_uses_the_same_captured_activation(tiny):
    model, image = tiny
    spec = EndpointTarget(
        "energy", "blocks.1", kind="patch_channel_energy", channel=3,
    )
    result = scan_endpoint_vit(model, image, [spec]).units["energy"]
    expected = 0.5 * result.token_values[:, model.num_prefix_tokens:].square().sum(1)
    torch.testing.assert_close(result.value, expected, atol=0, rtol=0)
    assert result.attribution.abs().sum() > 0


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
    for name, cut in result.cuts.items():
        total = sum(result.components[key] for key in cut)
        torch.testing.assert_close(
            total, result.reconstructions[result.cut_reconstruction[name]],
            atol=1e-12, rtol=1e-9,
        )
    heads = sum(result.components[f"head_{head}"] for head in range(3))
    torch.testing.assert_close(heads, result.components["attention"], atol=1e-12, rtol=1e-9)
    assert result.components["identity"].abs().sum() > 0
    assert result.components["attention"].abs().sum() > 0
    assert max(result.closure_relative_l2.values()) < 1e-9
    assert {"qkv_projection", "attention_softmax", "av_before_concat",
            "projection_input", "projection_output"}.issubset(result.captured_edges)


def test_omission_and_duplicate_accounting_fixtures_are_rejected(tiny):
    model, image = tiny
    with torch.no_grad():
        model.blocks[1].attn.proj.bias.copy_(torch.linspace(-0.2, 0.2, 12, dtype=torch.float64))
    result = decompose_endpoint_vit(
        model, image, block=1, target=EndpointTarget("terminal", "", kind="logit", class_id=1),
    )
    assert result.components["head_2"].abs().sum() > 1e-12
    assert result.components["routing"].abs().sum() > 1e-12
    assert model.blocks[1].attn.proj.bias.abs().sum() > 0  # bias changes the downstream state/cotangent
    wrong_total = dict(result.components)
    wrong_total["identity"] = result.components["identity"] + result.components["attention"]
    cases = [
        (wrong_total, result.cuts["residual1"], "L_F", result.full),
        (result.components, tuple(k for k in result.cuts["heads_branch"] if k != "head_2"), "A_F", result.reconstructions["A_F"]),
        (result.components, ("value",), "A_F", result.reconstructions["A_F"]),
    ]
    for components, keys, reconstruction_id, reference in cases:
        with pytest.raises(ValueError, match="CUT_CLOSURE_FAILED"):
            validate_decomposition_cut(
                components=components, keys=keys, reference=reference,
                reconstruction_id=reconstruction_id, atol=1e-12, rtol=1e-9,
            )


def test_ground_truth_and_frozen_top1_target_contract(tiny):
    model, image = tiny
    with pytest.raises(ValueError, match="requires class_id"):
        scan_endpoint_vit(model, image, [EndpointTarget("gt", "", kind="ground_truth_logit")])
    top1 = scan_endpoint_vit(model, image, [EndpointTarget("top1", "", kind="frozen_clean_top1_logit")])
    assert top1.units["top1"].target.class_id == int(top1.logits.argmax())


def test_unfused_edge_capture_matches_native_and_restores_forward(tiny):
    model, image = tiny
    attention = model.blocks[0].attn
    original = attention.forward
    expected = model(image)
    observed = {}
    with capture_unfused_attention_edges(attention, observed):
        actual = model(image)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert attention.forward == original
    assert {"query_projection", "key_projection", "value_projection",
            "attention_softmax", "av_before_concat", "projection_input",
            "projection_output"}.issubset(observed)


def test_fused_backend_is_not_accepted_for_structural_edge_capture(tiny):
    model, _image = tiny
    model.blocks[0].attn.fused_attn = True
    with pytest.raises(ValueError, match="STRUCTURAL_BACKEND_REQUIRES_UNFUSED_ATTENTION"):
        with capture_unfused_attention_edges(model.blocks[0].attn, {}):
            pass


@pytest.mark.parametrize("dtype,atol,rtol,near_zero", [
    (torch.float64, 1e-9, 1e-7, 1e-12),
    (torch.float32, 1e-5, 1e-4, 1e-6),
])
def test_absolute_and_relative_numeric_contract_branches(dtype, atol, rtol, near_zero):
    zero = torch.zeros(8, dtype=dtype)
    absolute = numeric_parity(zero.clone(), zero, atol=atol, rtol=rtol, near_zero_norm=near_zero)
    assert absolute["pass"] and absolute["branch"] == "absolute_near_zero"
    reference = torch.ones(8, dtype=dtype)
    relative = numeric_parity(reference.clone(), reference, atol=atol, rtol=rtol, near_zero_norm=near_zero)
    assert relative["pass"] and relative["branch"] == "relative_nonzero"


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
