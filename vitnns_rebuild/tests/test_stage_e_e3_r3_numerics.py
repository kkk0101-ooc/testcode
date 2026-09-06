"""Outcome-free E3-R3 numerical stability tests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import roots_legendre

from vitnns import attribution
from vitnns import stage_c_value_path as value_path
from vitnns import stage_e_e3_r3 as r3

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_r3_config_and_upstream_are_closed() -> None:
    config = r3._validate_config(REPO_ROOT)
    assert config["canonical_digest"] == r3.EXPECTED_CONFIG_DIGEST
    assert config["pilot"]["selection_order"] == 117
    assert config["pilot"]["coordinate"] == [11, 2]
    assert config["authorization"]["e4_started"] is False
    upstream = r3.validate_upstream(REPO_ROOT)
    assert upstream["status"] == "FAIL"
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(json.loads((REPO_ROOT / r3.COMPLETION_SCHEMA_PATH).read_text()))


def test_scipy_float64_nodes_weights_match_legacy_small_order() -> None:
    nodes, weights = attribution._gauss_legendre_unit_interval(
        16, device=torch.device("cpu"), dtype=torch.float64,
    )
    roots, raw = roots_legendre(16)
    np.testing.assert_allclose(nodes.numpy(), (roots + 1.0) * 0.5, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(weights.numpy(), raw * 0.5, atol=2e-7, rtol=2e-7)


def test_float64_kahan_keeps_model_and_stored_attribution_float32() -> None:
    inputs = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(1, 3, 2, 2)
    seen = []
    def callback(value: torch.Tensor) -> torch.Tensor:
        seen.append(value.dtype)
        return (value.sin() + 0.25 * value.square()).flatten(1).sum(1)
    result = attribution.integrated_gradients_float64_kahan(
        callback, inputs, torch.zeros_like(inputs), steps=(16, 32, 64),
        step_chunk_size=8, atol=1e-5, rtol=1e-5,
    )
    assert seen and set(seen) == {torch.float32}
    assert result.attribution.dtype is torch.float32
    assert result.metadata["quadrature_dtype"] == "float64"
    assert result.metadata["accumulation_dtype"] == "float64"
    assert result.metadata["accumulation_algorithm"] == "kahan_per_chunk"
    assert result.metadata["conservation_reduction_dtype"] == "float64"
    assert result.converged
    assert result.conservation["sum"].dtype is torch.float64


def test_float64_kahan_is_deterministic_on_synthetic_fixture() -> None:
    inputs = torch.tensor([[[[0.2, -0.7]], [[1.1, 0.3]], [[-0.5, 0.9]]]], dtype=torch.float32)
    callback = lambda value: (value.tanh() * torch.tensor(0.7, device=value.device)).flatten(1).sum(1)
    left = attribution.integrated_gradients_float64_kahan(callback, inputs, steps=(32, 64), atol=1e-6, rtol=1e-6)
    right = attribution.integrated_gradients_float64_kahan(callback, inputs, steps=(32, 64), atol=1e-6, rtol=1e-6)
    assert left.to_dict() == right.to_dict()
    torch.testing.assert_close(left.attribution, right.attribution, atol=0, rtol=0)


def test_legacy_integrator_keeps_float32_conservation_reduction() -> None:
    inputs = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    result = attribution.integrated_gradients(
        lambda value: value.flatten(1).sum(1), inputs, steps=4,
    )
    assert result.conservation["sum"].dtype is torch.float32
    assert "conservation_reduction_dtype" not in result.metadata


def test_r3_schedule_and_identity_are_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert value_path.E3_R3_STEPS == (65536, 131072, 262144)
    captured = {}
    monkeypatch.setattr(
        value_path,
        "conditional_fixed_clean_attention_head_direction_value_path_ig_at_intervention",
        lambda *_args, **kwargs: captured.update(kwargs) or None,
    )
    target = type("Target", (), {"clean_input_sha256": "a" * 64})()
    with pytest.raises(AttributeError):
        value_path.e3_r3_conditional_fixed_clean_attention_head_direction_value_path_ig(
            object(), object(), target, provenance={"stage": "E3-R3"},
        )
    assert tuple(captured["_steps_schedule"]) == value_path.E3_R3_STEPS
    assert captured["_numerical_stability_id"] == "STAGE-E-E3-R3-NUMERICAL-STABILITY-20260907"


def test_r3_partition_preserves_1164_and_runs_only_1836() -> None:
    inherited, eligible = r3._partition(REPO_ROOT)
    from collections import Counter
    assert Counter(value[0] for value in inherited.values()) == Counter({
        "E3_V1_INHERITED": 516, "E3_R1_INHERITED": 12,
        "E3_R2_INHERITED": 636,
    })
    assert len(eligible) == 1836
    pilot = [unit for unit in eligible if unit["selection_order"] == 117 and (unit["block"], unit["head"]) == (11, 2)]
    assert len(pilot) == 2 and {unit["method"] for unit in pilot} == set(r3.e3.METHODS)


def test_r3_schedule_for_pilot_replacement_and_general_unit() -> None:
    nns = {"selection_order": 117, "block": 11, "head": 2, "method": r3.e3.NNS_METHOD}
    matched = {**nns, "method": r3.e3.MATCHED_METHOD}
    general = {**nns, "selection_order": 118}
    assert r3._r3_schedule_for(nns) == r3.STEPS
    assert r3._r3_schedule_for(matched) == r3.r1.COMPLETE_SCHEDULE + r3.STEPS
    assert r3._r3_schedule_for(general) == r3.r1.COMPLETE_SCHEDULE + r3.STEPS


def test_no_e4_execution_surface() -> None:
    public = {name for name in dir(r3) if name.startswith(("run_", "start_", "execute_"))}
    assert public == {"run_pilot", "run_shard"}
