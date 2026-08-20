import torch

from vitnns.provenance import topk_indices


def test_topk_semantics():
    logits = torch.full((1, 1000), -10.0)
    expected = [17, 42, 8, 999, 0]
    for score, index in zip((5.0, 4.0, 3.0, 2.0, 1.0), expected):
        logits[0, index] = score
    assert topk_indices(logits, 1) == [17]
    assert topk_indices(logits, 5) == expected
    assert len(set(topk_indices(logits, 5))) == 5

