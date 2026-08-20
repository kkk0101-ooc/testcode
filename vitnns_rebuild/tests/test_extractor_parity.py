import torch
from PIL import Image

from vitnns.extractor import NativeViTExtractor, capture_block_io
from vitnns.modeling import canonical_transform, load_canonical_model


@torch.inference_mode()
def test_hook_extractor_parity(state_dict_path, scanner_root, sample_relative_path):
    reference = load_canonical_model(state_dict_path, unfused_attention=False)
    extracted_model = load_canonical_model(state_dict_path, unfused_attention=True)
    with Image.open(scanner_root / sample_relative_path) as source:
        image = canonical_transform(reference)(source.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        reference_logits = reference(image)
    unfused_plain_logits, reference_inputs, reference_outputs = capture_block_io(extracted_model, image)
    result = NativeViTExtractor(extracted_model).extract(image)
    assert float((reference_logits - result.logits).abs().max()) < 1e-5
    assert reference_logits.topk(5).indices.tolist() == result.logits.topk(5).indices.tolist()
    assert torch.equal(unfused_plain_logits, result.logits)
    for index, layer in enumerate(result.layers):
        assert torch.equal(reference_inputs[index], layer.block_input)
        assert torch.equal(reference_outputs[index], layer.block_output)


@torch.inference_mode()
def test_repeatable_extraction(state_dict_path):
    torch.manual_seed(13)
    model = load_canonical_model(state_dict_path, unfused_attention=True)
    image = torch.randn(1, 3, 224, 224)
    first = NativeViTExtractor(model).extract(image)
    second = NativeViTExtractor(model).extract(image)
    assert torch.equal(first.logits, second.logits)
    assert all(torch.equal(a.attention, b.attention) for a, b in zip(first.layers, second.layers))

