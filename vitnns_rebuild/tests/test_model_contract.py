import torch

from vitnns.constants import MODEL_SHA256
from vitnns.io_utils import sha256_file
from vitnns.modeling import canonical_preprocess_spec, canonical_transform, load_canonical_model


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

