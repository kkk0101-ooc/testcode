import os

import pytest
import torch

from vitnns.extractor import capture_block_io
from vitnns.legacy import load_trusted_legacy_full_model
from vitnns.modeling import load_canonical_model


def test_pickle_loader_requires_explicit_opt_in():
    with pytest.raises(PermissionError):
        load_trusted_legacy_full_model()


@pytest.mark.unsafe_pickle
@pytest.mark.skipif(os.environ.get("VITNNS_ALLOW_UNSAFE_PICKLE") != "1", reason="explicit opt-in required")
def test_trusted_legacy_pickle_matches_state_dict(state_dict_path):
    canonical = load_canonical_model(state_dict_path, unfused_attention=True)
    legacy = load_trusted_legacy_full_model(allow_unsafe_pickle=True)
    for block in legacy.blocks:
        block.attn.fused_attn = False
    torch.manual_seed(19)
    image = torch.randn(1, 3, 224, 224)
    expected, expected_inputs, expected_outputs = capture_block_io(canonical, image)
    actual, actual_inputs, actual_outputs = capture_block_io(legacy, image)
    assert float((expected - actual).abs().max()) < 1e-5
    for left, right in zip(expected_inputs + expected_outputs, actual_inputs + actual_outputs):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-5)

