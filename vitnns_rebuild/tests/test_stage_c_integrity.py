from __future__ import annotations

import json
from collections.abc import Callable

import pytest
import torch
from torch import nn

import vitnns.stage_c_integrity as integrity
from vitnns.stage_c_integrity import (
    CheapModelGuard,
    ModelIntegrityBoundary,
    ModelIntegrityError,
    StrongModelSnapshot,
)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("persistent", torch.tensor([3.0, 4.0]))
        self.register_buffer(
            "nonpersistent", torch.tensor([5.0, 6.0]), persistent=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight + self.persistent


def _replace_parameter(model: _ToyModel) -> None:
    model.weight = nn.Parameter(
        model.weight.detach().clone(), requires_grad=model.weight.requires_grad
    )


def _replace_buffer(model: _ToyModel) -> None:
    model.persistent = model.persistent.detach().clone()


def _rebind_parameter_storage(model: _ToyModel) -> None:
    replacement = torch.full_like(model.weight, 9.0)
    with torch.no_grad():
        model.weight.set_(replacement)


def _mutate_parameter_in_place(model: _ToyModel) -> None:
    with torch.no_grad():
        model.weight.add_(1.0)


def _mutate_nonpersistent_buffer_in_place(model: _ToyModel) -> None:
    model.nonpersistent.add_(1.0)


def _toggle_requires_grad(model: _ToyModel) -> None:
    model.weight.requires_grad_(False)


def test_strong_snapshot_capture_verify_compare_and_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyModel()
    original_signer = integrity.strong_model_signature
    calls: list[nn.Module] = []

    def counted_signer(candidate: nn.Module):
        calls.append(candidate)
        return original_signer(candidate)

    monkeypatch.setattr(integrity, "strong_model_signature", counted_signer)

    baseline = StrongModelSnapshot.capture(model)
    assert calls == [model]

    verified = baseline.verify()
    assert verified.matched
    assert verified.signature_calls == 1
    assert calls == [model, model]

    current = StrongModelSnapshot.capture(model)
    signer_calls_before_compare = len(calls)
    compared = baseline.compare_snapshot(current, signature_calls=7)
    assert compared.matched
    assert compared.signature_calls == 7
    assert len(calls) == signer_calls_before_compare

    model.weight.data.add_(10.0)
    changed = StrongModelSnapshot.capture(model)
    signer_calls_before_compare = len(calls)
    compared_changed = baseline.compare_snapshot(changed)
    assert not compared_changed.matched
    assert "parameter:weight" in compared_changed.differences
    assert len(calls) == signer_calls_before_compare

    verified_changed = baseline.verify(model)
    assert not verified_changed.matched
    assert "parameter:weight" in verified_changed.differences
    assert calls == [model] * 5


@pytest.mark.parametrize("body_raises", [False, True], ids=["success", "exception"])
def test_model_integrity_boundary_uses_two_unit_signatures(
    monkeypatch: pytest.MonkeyPatch, body_raises: bool
) -> None:
    model = _ToyModel()
    original_signer = integrity.strong_model_signature
    calls: list[nn.Module] = []

    def counted_signer(candidate: nn.Module):
        calls.append(candidate)
        return original_signer(candidate)

    monkeypatch.setattr(integrity, "strong_model_signature", counted_signer)

    # Simulate the one-time model-load signature separately from this unit.
    load_baseline = StrongModelSnapshot.capture(model)
    assert calls == [model]
    calls.clear()

    boundary = ModelIntegrityBoundary(model, reference_snapshot=load_baseline)
    if body_raises:
        primary = LookupError("body failed")
        with pytest.raises(LookupError, match="body failed") as caught:
            with boundary:
                raise primary
        assert caught.value is primary
    else:
        with boundary:
            pass

    assert calls == [model, model]
    assert boundary.entry_evidence is not None
    assert boundary.entry_evidence.matched
    assert boundary.entry_evidence.signature_calls == 1
    assert boundary.evidence is not None
    assert boundary.evidence.matched
    assert boundary.evidence.signature_calls == 2
    assert boundary.restoration_evidence is not None
    assert boundary.restoration_evidence.restored
    assert boundary.evidence.body_exception_type == (
        "builtins.LookupError" if body_raises else None
    )


@pytest.mark.parametrize(
    ("mutation", "expected_difference"),
    [
        (_replace_parameter, "parameter:weight"),
        (_replace_buffer, "buffer:persistent"),
        (_rebind_parameter_storage, "parameter:weight"),
        (_mutate_parameter_in_place, "parameter:weight"),
        (_mutate_nonpersistent_buffer_in_place, "buffer:nonpersistent"),
        (_toggle_requires_grad, "parameter:weight"),
    ],
    ids=[
        "parameter-replacement",
        "buffer-replacement",
        "storage-rebind",
        "parameter-in-place",
        "buffer-in-place",
        "requires-grad",
    ],
)
def test_cheap_model_guard_detects_live_state_changes(
    mutation: Callable[[_ToyModel], None], expected_difference: str
) -> None:
    model = _ToyModel()

    with pytest.raises(ModelIntegrityError, match="changed cheap model") as caught:
        with CheapModelGuard(model, context="toy operation"):
            mutation(model)

    error = caught.value
    assert not error.evidence.matched
    assert error.evidence.signature_calls == 0
    assert expected_difference in error.evidence.differences


def test_same_storage_data_mutation_evades_cheap_guard_but_not_strong_exit() -> None:
    model = _ToyModel()
    original_value = model.weight.detach().clone()
    live_state = integrity.tensor_live_signature(model.weight)

    with CheapModelGuard(model) as cheap_guard:
        model.weight.data.add_(2.0)

    assert torch.equal(model.weight, original_value + 2.0)
    assert integrity.tensor_live_signature(model.weight) == live_state
    assert cheap_guard.evidence is not None
    assert cheap_guard.evidence.matched

    strong_boundary = ModelIntegrityBoundary(model)
    with pytest.raises(ModelIntegrityError, match="exit model integrity") as caught:
        with strong_boundary:
            model.weight.data.add_(3.0)

    assert not caught.value.evidence.matched
    assert "parameter:weight" in caught.value.evidence.differences
    assert caught.value.evidence.signature_calls == 2
    assert integrity.tensor_live_signature(model.weight) == live_state


def test_boundary_restores_lifecycle_then_fails_closed_with_evidence() -> None:
    model = _ToyModel()
    model.fused_attn = False
    original_grad = torch.tensor([7.0, 8.0])
    model.weight.grad = original_grad
    original_hook = model.register_forward_pre_hook(lambda _module, _args: None)
    boundary = ModelIntegrityBoundary(model)

    with pytest.raises(ModelIntegrityError) as caught:
        with boundary:
            model.eval()
            model.fused_attn = True
            model.weight.requires_grad_(False)
            model.weight.grad = torch.zeros_like(original_grad)
            original_hook.remove()
            model.register_forward_hook(lambda _module, _args, output: output)

    assert model.training
    assert model.fused_attn is False
    assert model.weight.requires_grad
    assert model.weight.grad is original_grad
    torch.testing.assert_close(original_grad, torch.tensor([7.0, 8.0]))
    assert original_hook.id in model._forward_pre_hooks
    assert not model._forward_hooks
    restoration = caught.value.restoration_evidence
    assert restoration is not None and restoration.restored
    assert restoration.changes_detected
    assert caught.value.evidence.signature_calls == 2
    assert not caught.value.evidence.matched
    assert any(
        item.startswith("lifecycle:")
        for item in caught.value.evidence.differences
    )
def test_strong_signature_checks_persistent_but_not_nonpersistent_buffer_content() -> None:
    persistent_model = _ToyModel()
    persistent_baseline = StrongModelSnapshot.capture(persistent_model)
    persistent_model.persistent.data.add_(1.0)
    persistent_evidence = persistent_baseline.verify()
    assert not persistent_evidence.matched
    assert "persistent_buffer:persistent" in persistent_evidence.differences

    nonpersistent_model = _ToyModel()
    nonpersistent_baseline = StrongModelSnapshot.capture(nonpersistent_model)
    strong_entries = {
        (entry[0], entry[1])
        for entry in nonpersistent_baseline.signature
        if len(entry) >= 2
    }
    assert ("persistent_buffer", "persistent") in strong_entries
    assert ("persistent_buffer", "nonpersistent") not in strong_entries

    nonpersistent_model.nonpersistent.data.add_(1.0)
    nonpersistent_evidence = nonpersistent_baseline.verify()
    assert nonpersistent_evidence.matched
    assert nonpersistent_evidence.differences == ()


def test_evidence_serialization_contains_no_runtime_object_or_pointer_values() -> None:
    model = _ToyModel()
    baseline = StrongModelSnapshot.capture(model)
    runtime_values = {id(model)}
    for tensor in (model.weight, model.persistent, model.nonpersistent):
        storage = tensor.untyped_storage()
        runtime_values.update(
            (id(tensor), tensor.data_ptr(), storage.data_ptr(), storage._cdata)
        )

    model.weight.data.add_(1.0)
    evidence = baseline.verify()
    payload = evidence.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert json.loads(encoded) == payload
    assert set(payload) == {
        "schema_version",
        "algorithm",
        "checked",
        "entry_sha256",
        "exit_sha256",
        "matched",
        "restored",
        "differences",
        "signature_calls",
        "verification_error_type",
        "verification_error_message",
        "body_exception_type",
    }
    assert "model_object_id" not in encoded
    assert "data_ptr" not in encoded
    assert "_cdata" not in encoded
    assert all(str(value) not in encoded for value in runtime_values)


@pytest.mark.parametrize("boundary_kind", ["cheap", "strong"])
def test_integrity_failure_preserves_body_exception_and_cause(
    boundary_kind: str,
) -> None:
    model = _ToyModel()
    primary = LookupError("primary body failure")
    boundary = (
        CheapModelGuard(model, context="toy operation")
        if boundary_kind == "cheap"
        else ModelIntegrityBoundary(model)
    )

    with pytest.raises(ModelIntegrityError) as caught:
        with boundary:
            if boundary_kind == "cheap":
                model.weight.requires_grad_(False)
            else:
                model.weight.data.add_(1.0)
            raise primary

    error = caught.value
    assert error.body_exception is primary
    assert error.__cause__ is primary
    assert error.evidence.body_exception_type == "builtins.LookupError"
    assert not error.evidence.matched
