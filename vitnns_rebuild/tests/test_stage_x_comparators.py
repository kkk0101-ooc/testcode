from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from vitnns import stage_x_comparators as comp
from test_stage_x_nns_v2 import TinyViT


class MeanModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        score = value.mean((1, 2, 3))
        return torch.stack((score, -score), dim=1)


def test_chefer_matches_official_equation_and_source_binding():
    model, value = TinyViT(), torch.randn(1, 1, 4, 4, dtype=torch.float64)
    result = comp.chefer(model, value, 0)
    assert result["attribution"].shape == (2, 2)
    assert result["target_index"] == 0 and result["class_targeted"]
    assert result["official_commit"] == "c3e578f76b954e8528afeaaee26de3f07e3fe559"
    assert result["algorithm"].startswith("ReLU(attention_times_class_gradient)")


def test_gradcam_target_last_norm_prefix_exclusion_and_no_minmax():
    model, value = TinyViT(), torch.randn(1, 1, 4, 4, dtype=torch.float64)
    result = comp.vit_grad_cam(model, value, 1)
    assert result["target_index"] == 1 and result["attribution"].shape == (2, 2)
    assert result["layer"] == "blocks[0].norm1"
    assert result["prefix_tokens_excluded"] == 1
    assert result["minmax_normalization"] is False
    assert len(result["source_file"]["sha256"]) == 64


def test_rise_deterministic_official_mask_semantics():
    model = MeanModel()
    value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16
    first = comp.rise(model, value, 0, masks=32, grid=2, batch_size=8)
    second = comp.rise(model, value, 0, masks=32, grid=2, batch_size=8)
    torch.testing.assert_close(first["attribution"], second["attribution"])
    assert first["seed"] == second["seed"] and first["random_shift"]
    assert first["resize"] == "bilinear" and first["official_commit"] == "d91ea006d4bb9b7990347fe97086bdc0f5c1fe10"


def test_occlusion_raw_logit_zero_replacement_and_geometry():
    value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16
    result = comp.occlusion(MeanModel(), value, 0, window=2, stride=2)
    assert result["attribution"].shape == (2, 2)
    assert result["window"] == result["stride"] == 2
    assert result["replacement"] == "normalized_zero"


def test_attnlrp_official_hybrid_rule_and_gamma_grid():
    model, value = TinyViT(), torch.randn(1, 1, 4, 4, dtype=torch.float64)
    result = comp.attnlrp(model, value, 0, conv_gamma=0.1, linear_gamma=0.05)
    assert result["forward_parity_max_abs"] < 1e-6
    assert result["custom_backward"] and not result["ordinary_input_times_gradient"]
    assert result["exact_algorithm_name"] == "AttnLRP outside attention + CP-LRP inside attention"
    assert result["official_commit"] == "f04344177551bfcc3665e88afe96501f43fe560b"
    with pytest.raises(comp.ComparatorGammaDriftError):
        comp.attnlrp(model, value, 0, conv_gamma=0.2, linear_gamma=0.05)


def test_attnlrp_all_candidates_compatible_without_selection():
    model, value = TinyViT(), torch.randn(1, 1, 4, 4, dtype=torch.float64)
    results = comp.validate_attnlrp_grid(model, value, 0)
    assert len(results) == 15 and all(row["compatible"] for row in results)
    assert all(row["selected"] is False for row in results)
    assert {(row["conv_gamma"], row["linear_gamma"]) for row in results} == {
        (conv, linear) for conv in comp.ATTNLRP_CONV_GAMMAS for linear in comp.ATTNLRP_LINEAR_GAMMAS
    }


def test_attnlrp_restores_native_methods():
    model, value = TinyViT(), torch.randn(1, 1, 4, 4, dtype=torch.float64)
    before = model.blocks[0].attn.forward.__func__
    result = comp.attnlrp_official_timm(model, value, 0, conv_gamma=0.25, linear_gamma=0.1)
    assert result["forward_parity_max_abs"] < 1e-6
    assert result["attribution"].shape == (4, 4)
    assert model.blocks[0].attn.forward.__func__ is before


def test_identity_control_requires_same_target_steps_and_map():
    model, value = TinyViT(), torch.randn(1, 1, 4, 4, dtype=torch.float64)
    nns_result = type("R", (), {})()
    nns_result.target = type("T", (), {"predicted_class": 1})()
    nns_result.steps = 32
    nns_result.attribution = value[0].float()
    control = {
        "method_id": comp.IDENTITY,
        "target_index": 1,
        "steps": 32,
        "attribution": value[0].float(),
        "superiority_test_allowed": False,
    }
    assert comp.validate_identity(nns_result, control)["status"] == "PASS"
    with pytest.raises(comp.ComparatorTargetDriftError):
        comp.validate_identity(nns_result, {**control, "steps": 64})
    with pytest.raises(comp.ComparatorForwardDriftError):
        comp.validate_identity(nns_result, {**control, "attribution": value[0].float() + 1})


def test_source_tamper_target_drift_and_display_rejected(tmp_path, monkeypatch):
    manifest = json.loads(comp.DEPENDENCY_MANIFEST.read_text())
    record = next(row for row in manifest["files"] if row["path"].endswith("ViT_LRP.py"))
    fake = tmp_path / "dependency_manifest.json"
    changed = {**manifest, "files": [{**row, "sha256": "0" * 64} if row is record else row for row in manifest["files"]]}
    fake.write_text(json.dumps(changed))
    monkeypatch.setattr(comp, "DEPENDENCY_MANIFEST", fake)
    with pytest.raises(comp.ComparatorSourceError):
        comp._manifest_record(record["path"])
    with pytest.raises(comp.ComparatorTargetDriftError):
        comp.vit_grad_cam(TinyViT(), torch.zeros(1, 1, 4, 4, dtype=torch.float64), -1)
    with pytest.raises(comp.ComparatorError):
        comp.rise(MeanModel(), torch.zeros(1, 1, 4, 4, dtype=torch.uint8), 0, masks=2, grid=2)


def test_architecture_compatibility_has_exact_five_member_holm_family():
    import timm
    for model_id in (
        "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "deit_small_patch16_224.fb_in1k",
        "deit_base_patch16_224.fb_in1k",
        "vit_base_patch32_224.augreg_in21k_ft_in1k",
    ):
        result = comp.architecture_compatibility(timm.create_model(model_id, pretrained=False), model_id)
        assert result["competitive_methods"] == list(comp.PRIMARY)
        assert result["holm_family"] == list(comp.PRIMARY)
        assert result["identity_control"] == comp.IDENTITY
