from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vitnns import stage_f_f3c1_kernels as f3c1


class ToyAttention(nn.Module):
    def __init__(self, width: int = 12, heads: int = 12) -> None:
        super().__init__()
        self.num_heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(width, width, bias=False)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = value.shape
        return value.reshape(batch, tokens, self.num_heads, width // self.num_heads).permute(0, 2, 1, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.q(value))
        key = self._heads(self.k(value))
        vectors = self._heads(self.v(value))
        attention = ((query * self.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        attention = self.attn_drop(attention)
        output = (attention @ vectors).transpose(1, 2).reshape(value.shape)
        return self.proj(output)


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(12)
        self.attn = ToyAttention()
        self.norm2 = nn.LayerNorm(12)
        self.mlp = nn.Sequential(nn.Linear(12, 24), nn.GELU(), nn.Linear(24, 12))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attn(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class ToyAttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, 12, kernel_size=16, stride=16, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, 12))
        self.blocks = nn.ModuleList([ToyBlock() for _ in range(12)])
        self.norm = nn.LayerNorm(12)
        self.head = nn.Linear(12, 4)
        self.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch = self.patch(image).flatten(2).transpose(1, 2)
        value = torch.cat((self.cls.expand(image.shape[0], -1, -1), patch), dim=1)
        for block in self.blocks:
            value = block(value)
        return self.head(self.norm(value)[:, 0])


@pytest.fixture
def toy() -> tuple[ToyAttentionModel, torch.Tensor, f3c1.CleanCapture]:
    torch.manual_seed(9)
    model = ToyAttentionModel()
    clean = torch.linspace(-0.8, 0.9, 3 * 224 * 224, dtype=torch.float32).reshape(1, 3, 224, 224)
    capture = f3c1.capture_clean_target_and_all_attention(
        model, clean, expected_heads=12, expected_tokens=197
    )
    return model, clean, capture


def test_all_block_capture_is_same_forward_detached_hashed_and_complete(toy) -> None:
    _model, _clean, capture = toy
    manifest = capture.attentions.manifest()
    assert manifest["count"] == 12
    assert manifest["blocks"] == list(range(12))
    assert len(manifest["tensor_sha256"]) == 12
    assert manifest["detached_cloned_frozen"] is True
    assert manifest["manual_or_learned_head_weights"] is False
    assert manifest["source"] == "same_clean_forward_attn_drop_output_actually_consumed"


def test_endpoint_parity_is_strict_all_logits(toy) -> None:
    model, clean, capture = toy
    report = f3c1.validate_endpoint_parity(model, clean, capture)
    assert report["status"] == "PASS"
    assert report["max_absolute_error"] < 1e-6
    assert report["class_count"] == 4


def test_fixed_attention_cuts_qk_but_keeps_value_and_downstream_live(toy) -> None:
    model, clean, capture = toy
    model.zero_grad(set_to_none=True)
    logits = f3c1.conditional_logits(model, clean, capture.attentions)
    logits[0, capture.target.class_index].backward()
    for block in model.blocks:
        assert block.attn.q.weight.grad is None or not torch.count_nonzero(block.attn.q.weight.grad)
        assert block.attn.k.weight.grad is None or not torch.count_nonzero(block.attn.k.weight.grad)
        assert block.attn.v.weight.grad is not None and torch.count_nonzero(block.attn.v.weight.grad)
        assert block.attn.proj.weight.grad is not None and torch.count_nonzero(block.attn.proj.weight.grad)
        assert any(parameter.grad is not None for parameter in block.mlp.parameters())
    assert model.head.weight.grad is not None and torch.count_nonzero(model.head.weight.grad)


def test_synthetic_conditional_ig_conserves_and_projects_signed_rgb(toy) -> None:
    model, clean, capture = toy
    result = f3c1.synthetic_conditional_class_logit_ig(
        model, clean, capture, steps=(64, 128, 256), step_chunk_size=8
    )
    assert result["ig_result"].converged is True
    assert result["native_signed_rgb"].shape == (3, 224, 224)
    expected = result["native_signed_rgb"].sum(axis=0).reshape(14, 16, 14, 16).sum(axis=(1, 3))
    assert np.allclose(result["patch_mass"], expected.astype(np.float32), atol=1e-6, rtol=1e-6)
    assert result["baseline"]["canonical_rgb_value"] == 0.5
    assert result["baseline"]["is_black_image"] is False


@pytest.mark.parametrize("bad_count", [11, 13])
def test_wrong_attention_count_rejected(toy, bad_count: int) -> None:
    model, _clean, capture = toy
    tensors = capture.attentions.owned_tensors()
    values = tensors[:bad_count] if bad_count == 11 else tensors + (tensors[0],)
    with pytest.raises(f3c1.StageFF3C1KernelError, match="exactly 12"):
        f3c1.FrozenAllBlockAttention(
            issuer=f3c1._ISSUER, model=model, tensors=values,
            source_forward_id="a" * 64, expected_heads=12, expected_tokens=197,
        )


def test_partial_heads_and_non_post_softmax_tensors_rejected(toy) -> None:
    model, _clean, capture = toy
    tensors = list(capture.attentions.owned_tensors())
    partial = copy.copy(tensors); partial[0] = partial[0][:, :11]
    with pytest.raises(f3c1.StageFF3C1KernelError, match="shape drift"):
        f3c1.FrozenAllBlockAttention(
            issuer=f3c1._ISSUER, model=model, tensors=partial, source_forward_id="x",
            expected_heads=12, expected_tokens=197,
        )
    invalid = copy.copy(tensors); invalid[0] = invalid[0] * 0.5
    with pytest.raises(f3c1.StageFF3C1KernelError, match="sum to one"):
        f3c1.FrozenAllBlockAttention(
            issuer=f3c1._ISSUER, model=model, tensors=invalid, source_forward_id="x",
            expected_heads=12, expected_tokens=197,
        )


def test_attention_tamper_is_rejected(toy) -> None:
    model, _clean, capture = toy
    capture.attentions._FrozenAllBlockAttention__tensors[0].data[0, 0, 0, 0] += 0.1
    with pytest.raises(f3c1.StageFF3C1KernelError, match="mutated"):
        capture.attentions.verify(model)


def test_projection_keeps_sign_until_after_rgb_sum() -> None:
    native = np.zeros((3, 224, 224), dtype=np.float32)
    native[0] = 1.0
    native[1] = -1.0
    projected = pixel_to_patch_mass(signed_rgb_channel_sum(raw_array(native))).data
    assert not np.count_nonzero(projected)
    wrong = np.abs(native).sum(axis=0).reshape(14, 16, 14, 16).sum(axis=(1, 3))
    assert np.count_nonzero(wrong)


def test_tie_ranking_and_statistics_are_deterministic() -> None:
    values = np.zeros((14, 14), dtype=np.float32); values[0, :3] = 1.0
    first = f3c1.method_ranking(values, sigma=1, seed=f3c1.tie_seed("sample", 0))
    second = f3c1.method_ranking(values, sigma=1, seed=f3c1.tie_seed("sample", 0))
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    differences = np.linspace(-0.1, 0.2, 50, dtype=np.float64)
    p1 = f3c1.paired_sign_flip_p_value(differences, f3c1.PRIMARY_CONTRAST)
    p2 = f3c1.paired_sign_flip_p_value(differences, f3c1.PRIMARY_CONTRAST)
    assert p1 == p2
    b1 = f3c1.bootstrap_mean_ci(differences, f3c1.PRIMARY_CONTRAST)
    b2 = f3c1.bootstrap_mean_ci(differences, f3c1.PRIMARY_CONTRAST)
    assert b1 == b2


def test_statistical_fail_closed_and_holm_ties_are_frozen() -> None:
    with pytest.raises(f3c1.StageFF3C1KernelError, match="exactly 50"):
        f3c1.paired_sign_flip_p_value([0.0] * 49, f3c1.PRIMARY_CONTRAST)
    bad = np.zeros(50); bad[4] = np.nan
    with pytest.raises(f3c1.StageFF3C1KernelError, match="finite"):
        f3c1.bootstrap_mean_ci(bad, f3c1.PRIMARY_CONTRAST)
    raw = {name: 0.02 for name in f3c1.SECONDARY_CONTRASTS}
    assert [row["contrast"] for row in f3c1.holm_adjust_secondary(raw)] == list(f3c1.SECONDARY_CONTRASTS)
    assert f3c1.statistical_method_freeze()["pilot_or_formal_outcomes_accessed_before_freeze"] is False


def test_display_array_and_method_roster_false_passes_are_rejected() -> None:
    with pytest.raises(f3c1.StageFF3C1KernelError):
        f3c1.method_ranking(np.ones((224, 224), dtype=np.float32), sigma=1, seed=1)
    methods = {name: {} for name in reversed(f3c1.METHODS)}
    model = ToyAttentionModel(); clean = torch.zeros((1, 3, 224, 224))
    capture = f3c1.capture_clean_target_and_all_attention(model, clean)
    with pytest.raises(f3c1.StageFF3C1KernelError, match="exact five-method roster"):
        f3c1.evaluate_common_fidelity(model, clean, capture, methods, sample_id="s")


def test_five_method_runner_preserves_roster_and_class_agnostic_labels(toy, monkeypatch) -> None:
    model, clean, capture = toy
    signed = np.ones((3, 224, 224), dtype=np.float32)
    patch = np.ones((14, 14), dtype=np.float32)
    nns = {
        "method_id": f3c1.METHOD_ID, "class_targeted": True,
        "class_agnostic": False, "conditional_functional": True,
        "native_signed_rgb": signed, "patch_mass": patch,
        "sign_semantics": "signed",
    }

    def fake(method_id: str, class_agnostic: bool, native: np.ndarray):
        return SimpleNamespace(
            method_id=method_id, class_agnostic=class_agnostic,
            native_tensor=SimpleNamespace(data=native),
            metric_map=SimpleNamespace(data=patch),
            sign_semantics=SimpleNamespace(value="nonnegative" if class_agnostic else "signed"),
            provenance={"fixture": True},
        )

    monkeypatch.setattr(f3c1, "conditional_class_logit_ig", lambda *_args, **_kwargs: nns)
    monkeypatch.setattr(f3c1, "vit_grad_cam_block11_norm1", lambda *_args, **_kwargs: fake(f3c1.METHODS[1], False, patch))
    monkeypatch.setattr(f3c1, "full_model_class_logit_ig", lambda *_args, **_kwargs: fake(f3c1.METHODS[2], False, signed))
    monkeypatch.setattr(f3c1, "attention_rollout_mean_head_residual_all_blocks", lambda *_args, **_kwargs: fake(f3c1.METHODS[3], True, patch))
    monkeypatch.setattr(f3c1, "direct_raw_cls_attention_last_block_head_mean", lambda *_args, **_kwargs: fake(f3c1.METHODS[4], True, patch))
    methods = f3c1.collect_five_method_maps(model, clean, capture, capture.as_attention_extraction())
    assert tuple(methods) == f3c1.METHODS
    assert {name for name, value in methods.items() if value["class_agnostic"]} == f3c1.CLASS_AGNOSTIC_METHODS


def test_common_evaluator_uses_21_fractions_five_shared_ties_and_formula() -> None:
    class Scalar(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            score = value.mean(dim=(1, 2, 3))
            return torch.stack((score, -score), dim=1)

    target = SimpleNamespace(
        class_index=0, clean_input_sha256="a" * 64,
        to_dict=lambda: {"class_index": 0, "clean_input_sha256": "a" * 64},
    )
    capture = SimpleNamespace(
        target=target,
    )
    patch = np.arange(196, dtype=np.float32).reshape(14, 14) + 1
    ig = SimpleNamespace(conservation={"delta": torch.tensor([1.0])})
    methods = {
        name: {"patch_mass": patch.copy(), **({"ig_result": ig} if name == f3c1.METHOD_ID else {})}
        for name in f3c1.METHODS
    }
    result = f3c1.evaluate_common_fidelity(
        Scalar(), torch.ones((1, 3, 224, 224)), capture, methods,
        sample_id="synthetic/sample", eval_chunk=21,
    )
    assert result["fractions"] == list(f3c1.FRACTIONS)
    assert result["score"] == "0.5*(insertion_auc-deletion_auc)"
    for method in result["methods"].values():
        assert len(method["curves"]) == 5
        assert all(len(curve["insertion"]["responses"]) == 21 for curve in method["curves"])
    priorities = [row["tie_priority_sha256"] for row in result["methods"][f3c1.METHODS[0]]["curves"]]
    assert all([row["tie_priority_sha256"] for row in result["methods"][name]["curves"]] == priorities for name in f3c1.METHODS)


from vitnns.stage_c_maps import pixel_to_patch_mass, raw_array, signed_rgb_channel_sum
