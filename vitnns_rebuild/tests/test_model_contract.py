import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vitnns.constants import MODEL_NAME, MODEL_SHA256
from vitnns.io_utils import sha256_file
from vitnns import modeling
from vitnns.modeling import (
    canonical_preprocess_spec, canonical_transform, load_canonical_model,
)


RECOVERY_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
RECOVERY_TENSOR_DIGEST = (
    "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
)
RECOVERY_IDENTITY = "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718"
RECOVERY_MODEL_NAME = "vit_base_patch16_224.recovery_test"
EXPLICIT_TEST_IDENTITY = "EXPLICIT_SEMANTIC_TEST_IDENTITY"


class _FakeModel:
    def __init__(self) -> None:
        self.training = True
        self.blocks = [SimpleNamespace(attn=SimpleNamespace(fused_attn=True))]
        self.load_calls = []
        self.device = None

    def load_state_dict(self, state, *, strict):
        self.load_calls.append((state, strict))
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def eval(self):
        self.training = False
        return self

    def to(self, device):
        self.device = device
        return self


def _mock_lightweight_load(monkeypatch, *, file_sha256: str, state: dict):
    observed = {}
    model = _FakeModel()

    def source_path(source):
        return Path(source.name if hasattr(source, "read") else source).resolve()

    def fake_sha256(source):
        observed["sha256_source"] = source
        observed["sha256_path"] = source_path(source)
        return file_sha256

    def fake_torch_load(source, *, map_location, weights_only):
        observed["torch_load_source"] = source
        observed["torch_load"] = (
            source_path(source), map_location, weights_only,
        )
        return state

    def fake_create_model(name, *, pretrained):
        observed["create_model"] = (name, pretrained)
        return model

    monkeypatch.setattr(modeling, "sha256_file", fake_sha256)
    monkeypatch.setattr(modeling.torch, "load", fake_torch_load)
    monkeypatch.setattr(modeling.timm, "create_model", fake_create_model)
    monkeypatch.setattr(
        modeling, "set_fused_attn",
        lambda enabled: observed.setdefault("fused_attn", enabled),
    )
    monkeypatch.setattr(
        modeling, "_assert_model_contract",
        lambda value: observed.setdefault("contract_model", value),
    )
    return model, observed


def _recovery_binding():
    from vitnns.recovery_v2 import load_recovery_v2_binding

    return load_recovery_v2_binding(verify_assets=False)


def _mock_recovery_revalidation(monkeypatch, binding, *, revalidated=None):
    from vitnns import recovery_v2

    calls = []

    def fake_revalidate(*, path, schema_path, verify_assets):
        calls.append((path, schema_path, verify_assets))
        return binding if revalidated is None else revalidated

    monkeypatch.setattr(
        recovery_v2, "load_recovery_v2_binding", fake_revalidate,
    )
    return calls


def test_state_dict_loads_strictly_and_geometry(state_dict_path):
    assert sha256_file(state_dict_path) == MODEL_SHA256
    model = load_canonical_model(state_dict_path, verify_hash=True)
    assert not model.training
    assert len(model.blocks) == 12
    assert model.embed_dim == 768
    assert model.blocks[0].attn.num_heads == 12
    assert model.blocks[0].attn.head_dim == 64
    assert model.pos_embed.shape == (1, 197, 768)
    assert sum(parameter.numel() for parameter in model.parameters()) == 86_567_656


def test_preprocess_contract_and_repeatability(state_dict_path, scanner_root, sample_relative_path):
    from PIL import Image

    model = load_canonical_model(state_dict_path)
    spec = canonical_preprocess_spec(model)
    assert spec.input_size == (3, 224, 224)
    assert spec.resize_shorter_side == 248
    assert spec.center_crop == (224, 224)
    assert spec.interpolation == "bicubic"
    assert spec.mean == (0.5, 0.5, 0.5)
    assert spec.std == (0.5, 0.5, 0.5)
    transform = canonical_transform(model)
    with Image.open(scanner_root / sample_relative_path) as source:
        image = source.convert("RGB")
    first, second = transform(image), transform(image)
    assert first.shape == (3, 224, 224)
    assert torch.equal(first, second)


def test_loader_historical_defaults_preserve_safe_strict_contract(
    monkeypatch, tmp_path,
):
    path = tmp_path / "historical.pth"
    path.write_bytes(b"mock state dict")
    state = {"weight": torch.tensor([1.0])}
    model, observed = _mock_lightweight_load(
        monkeypatch, file_sha256=MODEL_SHA256, state=state,
    )

    loaded = load_canonical_model(path)

    assert loaded is model
    assert observed["sha256_path"] == path.resolve()
    assert observed["torch_load"] == (path.resolve(), "cpu", True)
    assert observed["sha256_source"] is observed["torch_load_source"]
    assert observed["create_model"] == (MODEL_NAME, False)
    assert observed["fused_attn"] is True
    assert observed["contract_model"] is model
    assert model.load_calls == [(state, True)]
    assert model.training is False
    assert model.device == "cpu"
    assert not hasattr(model, "_vitnns_model_identity")


def test_loader_historical_hash_opt_out_remains_available(monkeypatch, tmp_path):
    path = tmp_path / "unverified-historical.pth"
    path.write_bytes(b"mock state dict")
    state = {"weight": torch.tensor([1.0])}
    model, observed = _mock_lightweight_load(
        monkeypatch, file_sha256="0" * 64, state=state,
    )

    assert load_canonical_model(path, verify_hash=False) is model
    assert "sha256_path" not in observed
    assert observed["torch_load"] == (path.resolve(), "cpu", True)
    assert isinstance(observed["torch_load_source"], Path)


def test_loader_explicit_expected_hash_cannot_be_disabled(monkeypatch, tmp_path):
    path = tmp_path / "explicit-hash.pth"
    path.write_bytes(b"mock state dict")
    monkeypatch.setattr(modeling, "sha256_file", lambda _path: "0" * 64)
    monkeypatch.setattr(
        modeling.torch, "load",
        lambda *_args, **_kwargs: pytest.fail("mismatched bytes were deserialized"),
    )

    with pytest.raises(ValueError, match="State-dict SHA-256 mismatch"):
        load_canonical_model(
            path, verify_hash=False, expected_state_dict_sha256=MODEL_SHA256,
        )


def test_recovery_identity_forces_file_and_canonical_tensor_verification(
    monkeypatch, tmp_path,
):
    from vitnns import recovery_v2

    path = tmp_path / "recovery-489b.pth"
    path.write_bytes(b"mock recovery state dict")
    state = {"weight": torch.tensor([2.0])}
    model, observed = _mock_lightweight_load(
        monkeypatch, file_sha256=RECOVERY_SHA256, state=state,
    )

    def fake_tensor_digest(value):
        observed["tensor_digest_state"] = value
        return RECOVERY_TENSOR_DIGEST

    monkeypatch.setattr(
        recovery_v2, "canonical_state_dict_tensor_digest", fake_tensor_digest,
    )

    loaded = load_canonical_model(
        path,
        verify_hash=False,
        expected_state_dict_sha256=RECOVERY_SHA256,
        expected_model_name=RECOVERY_MODEL_NAME,
        expected_model_identity=EXPLICIT_TEST_IDENTITY,
        expected_canonical_tensor_digest=RECOVERY_TENSOR_DIGEST,
    )

    assert loaded is model
    assert observed["sha256_path"] == path.resolve()
    assert observed["torch_load"] == (path.resolve(), "cpu", True)
    assert observed["sha256_source"] is observed["torch_load_source"]
    assert observed["tensor_digest_state"] is state
    assert observed["create_model"] == (RECOVERY_MODEL_NAME, False)
    assert model.load_calls == [(state, True)]
    assert not hasattr(model, "_vitnns_model_identity")


def test_verified_load_rejects_path_replacement_before_deserialization(
    monkeypatch, tmp_path,
):
    path = tmp_path / "verified.pth"
    displaced = tmp_path / "verified-original.pth"
    replacement = tmp_path / "verified-replacement.pth"
    trusted_bytes = b"trusted state-dict bytes"
    replacement_bytes = b"replacement bytes"
    path.write_bytes(trusted_bytes)
    replacement.write_bytes(replacement_bytes)
    expected_sha256 = hashlib.sha256(trusted_bytes).hexdigest()
    state = {"weight": torch.tensor([2.5])}
    model, observed = _mock_lightweight_load(
        monkeypatch, file_sha256=expected_sha256, state=state,
    )

    def swapping_hash(source):
        observed["hash_source"] = source
        payload = source.read()
        path.rename(displaced)
        replacement.rename(path)
        return hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(modeling, "sha256_file", swapping_hash)
    monkeypatch.setattr(
        modeling.torch, "load",
        lambda *_args, **_kwargs: pytest.fail("replacement was deserialized"),
    )

    with pytest.raises(ValueError, match="changed while it was being hashed"):
        load_canonical_model(path, expected_state_dict_sha256=expected_sha256)

    assert model.load_calls == []
    assert path.read_bytes() == replacement_bytes


@pytest.mark.parametrize("supply_matching_explicit_values", [False, True])
def test_recovery_binding_derives_path_and_complete_identity(
    monkeypatch, supply_matching_explicit_values,
):
    from vitnns import recovery_v2

    binding = _recovery_binding()
    path = binding.model_path
    revalidation_calls = _mock_recovery_revalidation(monkeypatch, binding)
    state = {"weight": torch.tensor([3.0])}
    model, observed = _mock_lightweight_load(
        monkeypatch, file_sha256=RECOVERY_SHA256, state=state,
    )
    monkeypatch.setattr(
        recovery_v2, "canonical_state_dict_tensor_digest",
        lambda value: observed.setdefault("tensor_digest_state", value)
        and RECOVERY_TENSOR_DIGEST,
    )

    kwargs = {
        "recovery_binding": binding,
        "verify_hash": False,
        "unfused_attention": True,
    }
    if supply_matching_explicit_values:
        kwargs.update({
            "state_dict_path": binding.model_path,
            "expected_state_dict_sha256": binding.model["state_dict_sha256"],
            "expected_model_name": binding.model["name"],
            "expected_model_identity": binding.model["identity"],
            "expected_canonical_tensor_digest": binding.model[
                "canonical_tensor_digest"
            ],
        })
    loaded = load_canonical_model(**kwargs)

    assert loaded is model
    assert observed["sha256_path"] == path.resolve()
    assert observed["torch_load"] == (path.resolve(), "cpu", True)
    assert observed["sha256_source"] is observed["torch_load_source"]
    assert revalidation_calls == [(binding.path, binding.schema_path, True)]
    assert observed["tensor_digest_state"] is state
    assert observed["create_model"] == (MODEL_NAME, False)
    assert observed["fused_attn"] is False
    assert model.blocks[0].attn.fused_attn is False
    assert not hasattr(model, "_vitnns_model_identity")


def test_recovery_binding_rejects_duck_typed_claim(tmp_path):
    fake = SimpleNamespace(
        model_path=tmp_path / "forged.pth",
        model={
            "state_dict_sha256": RECOVERY_SHA256,
            "name": MODEL_NAME,
            "identity": RECOVERY_IDENTITY,
            "canonical_tensor_digest": RECOVERY_TENSOR_DIGEST,
        },
    )

    with pytest.raises(TypeError, match="must be a RecoveryV2Binding"):
        load_canonical_model(recovery_binding=fake)


def test_recovery_binding_rejects_directly_constructed_dataclass(
    monkeypatch, tmp_path,
):
    from vitnns.recovery_v2 import RecoveryV2Binding

    binding = _recovery_binding()
    forged_document = dict(binding.document)
    forged_document["model"] = {
        **binding.model,
        "state_dict_sha256": "0" * 64,
    }
    forged = RecoveryV2Binding(
        path=binding.path,
        schema_path=binding.schema_path,
        document=forged_document,
        model_path=tmp_path / "forged.pth",
        data_root=binding.data_root,
        devkit_path=binding.devkit_path,
        source_manifest_path=binding.source_manifest_path,
    )
    calls = _mock_recovery_revalidation(monkeypatch, binding)
    monkeypatch.setattr(
        modeling, "sha256_file",
        lambda *_args, **_kwargs: pytest.fail("forged model was hashed"),
    )
    monkeypatch.setattr(
        modeling.torch, "load",
        lambda *_args, **_kwargs: pytest.fail("forged model was deserialized"),
    )

    with pytest.raises(
        ValueError, match="conflicts with its revalidated source: document, model_path",
    ):
        load_canonical_model(recovery_binding=forged)

    assert calls == [(forged.path, forged.schema_path, True)]


@pytest.mark.parametrize("field", ["path", "schema_path", "model_path"])
def test_recovery_binding_rejects_replaced_core_path(
    monkeypatch, tmp_path, field,
):
    binding = _recovery_binding()
    forged = replace(binding, **{field: tmp_path / f"forged-{field}"})
    _mock_recovery_revalidation(monkeypatch, binding)

    with pytest.raises(
        ValueError, match=rf"conflicts with its revalidated source: {field}",
    ):
        load_canonical_model(recovery_binding=forged)


def test_revalidated_binding_requires_fixed_recovery_identity(monkeypatch):
    binding = _recovery_binding()
    drifted_document = dict(binding.document)
    drifted_document["model"] = {
        **binding.model,
        "identity": "RECOVERY_V2_DRIFTED_IDENTITY",
    }
    drifted = replace(binding, document=drifted_document)
    _mock_recovery_revalidation(monkeypatch, drifted, revalidated=drifted)

    with pytest.raises(ValueError, match="fixed recovery-v2 identity"):
        load_canonical_model(recovery_binding=drifted)


def test_standalone_cannot_claim_recovery_v2_identity():
    with pytest.raises(
        ValueError, match="recovery-v2 model identity requires a RecoveryV2Binding"
    ):
        load_canonical_model(
            "unused.pth",
            expected_state_dict_sha256=RECOVERY_SHA256,
            expected_model_name=MODEL_NAME,
            expected_model_identity=RECOVERY_IDENTITY,
            expected_canonical_tensor_digest=RECOVERY_TENSOR_DIGEST,
        )


@pytest.mark.parametrize(
    "explicit",
    [
        {"expected_state_dict_sha256": "0" * 64},
        {"expected_model_name": "different_model"},
        {
            "expected_model_identity": "DIFFERENT_IDENTITY",
            "expected_canonical_tensor_digest": RECOVERY_TENSOR_DIGEST,
        },
        {
            "expected_model_identity": RECOVERY_IDENTITY,
            "expected_canonical_tensor_digest": "0" * 64,
        },
    ],
)
def test_recovery_binding_rejects_explicit_identity_conflicts(
    monkeypatch, explicit,
):
    binding = _recovery_binding()
    _mock_recovery_revalidation(monkeypatch, binding)

    with pytest.raises(
        ValueError, match="Explicit model identity conflicts with recovery_binding"
    ):
        load_canonical_model(recovery_binding=binding, **explicit)


@pytest.mark.parametrize(
    "explicit",
    [
        {"expected_model_identity": RECOVERY_IDENTITY},
        {"expected_canonical_tensor_digest": RECOVERY_TENSOR_DIGEST},
    ],
)
def test_recovery_binding_still_requires_explicit_semantic_pair(
    explicit,
):
    binding = _recovery_binding()

    with pytest.raises(ValueError, match="must be provided together"):
        load_canonical_model(recovery_binding=binding, **explicit)


def test_recovery_binding_rejects_explicit_path_conflict(monkeypatch, tmp_path):
    different_path = tmp_path / "different.pth"
    binding = _recovery_binding()
    _mock_recovery_revalidation(monkeypatch, binding)

    with pytest.raises(
        ValueError, match="Explicit model identity conflicts with recovery_binding"
    ):
        load_canonical_model(different_path, recovery_binding=binding)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"expected_model_identity": EXPLICIT_TEST_IDENTITY},
            "must be provided together",
        ),
        (
            {"expected_canonical_tensor_digest": RECOVERY_TENSOR_DIGEST},
            "must be provided together",
        ),
        (
            {
                "expected_model_identity": "",
                "expected_canonical_tensor_digest": RECOVERY_TENSOR_DIGEST,
            },
            "expected_model_identity must be a non-empty string",
        ),
        (
            {
                "expected_model_identity": EXPLICIT_TEST_IDENTITY,
                "expected_canonical_tensor_digest": "A" * 64,
            },
            "expected_canonical_tensor_digest must be a lowercase SHA-256 digest",
        ),
        (
            {"expected_state_dict_sha256": "not-a-sha"},
            "expected_state_dict_sha256 must be a lowercase SHA-256 digest",
        ),
        (
            {"expected_model_name": "  "},
            "expected_model_name must be a non-empty string",
        ),
    ],
)
def test_loader_rejects_incomplete_or_malformed_identity(kwargs, message):
    with pytest.raises(ValueError, match=message):
        load_canonical_model("unused.pth", **kwargs)


def test_recovery_file_hash_mismatch_fails_before_deserialization(
    monkeypatch, tmp_path,
):
    path = tmp_path / "recovery-489b.pth"
    path.write_bytes(b"wrong bytes")
    monkeypatch.setattr(modeling, "sha256_file", lambda _path: "0" * 64)
    monkeypatch.setattr(
        modeling.torch, "load",
        lambda *_args, **_kwargs: pytest.fail("mismatched bytes were deserialized"),
    )

    with pytest.raises(ValueError, match="State-dict SHA-256 mismatch"):
        load_canonical_model(
            path,
            expected_state_dict_sha256=RECOVERY_SHA256,
            expected_model_name=MODEL_NAME,
            expected_model_identity=EXPLICIT_TEST_IDENTITY,
            expected_canonical_tensor_digest=RECOVERY_TENSOR_DIGEST,
        )


def test_recovery_tensor_digest_mismatch_fails_before_model_construction(
    monkeypatch, tmp_path,
):
    from vitnns import recovery_v2

    path = tmp_path / "recovery-489b.pth"
    path.write_bytes(b"mock recovery state dict")
    state = {"weight": torch.tensor([2.0])}
    _model, _observed = _mock_lightweight_load(
        monkeypatch, file_sha256=RECOVERY_SHA256, state=state,
    )
    monkeypatch.setattr(
        recovery_v2, "canonical_state_dict_tensor_digest",
        lambda _state: "0" * 64,
    )
    monkeypatch.setattr(
        modeling.timm, "create_model",
        lambda *_args, **_kwargs: pytest.fail("model was constructed before digest check"),
    )

    with pytest.raises(ValueError, match="Canonical state-dict tensor digest mismatch"):
        load_canonical_model(
            path,
            expected_state_dict_sha256=RECOVERY_SHA256,
            expected_model_name=MODEL_NAME,
            expected_model_identity=EXPLICIT_TEST_IDENTITY,
            expected_canonical_tensor_digest=RECOVERY_TENSOR_DIGEST,
        )

