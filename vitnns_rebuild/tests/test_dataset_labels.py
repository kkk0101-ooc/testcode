from collections import Counter

import pytest

from vitnns.constants import DEFAULT_VAL_ROOT
from vitnns.dataset import build_manifest
from vitnns.labels import ImageNet1KMapping, parse_official_devkit


def test_mapping_contract():
    mapping = ImageNet1KMapping()
    assert len(mapping.classes) == 1000
    assert mapping.classes[0].wnid == "n01440764"
    assert mapping.classes[-1].wnid == "n15075141"
    assert mapping.index_for_wnid("n03976467") == 732
    assert mapping.index_for_wnid("n15075141") == 999


def test_synset_manifest_is_deterministic(scanner_root):
    first = build_manifest(scanner_root)
    second = build_manifest(scanner_root)
    assert first == second
    assert len(first) == 100
    assert all(row.imagenet1k_index == ImageNet1KMapping().index_for_wnid(row.wnid) for row in first)
    assert set(Counter(row.wnid for row in first).values()) == {10}


def test_flat_validation_fails_closed_without_devkit():
    with pytest.raises(ValueError, match="official ILSVRC2012 devkit"):
        build_manifest(DEFAULT_VAL_ROOT, require_labels=True)
    rows = build_manifest(DEFAULT_VAL_ROOT, require_labels=False)
    assert len(rows) == 500
    assert all(row.imagenet1k_index is None and row.label_source == "unlabeled" for row in rows)


def test_official_devkit_mapping(official_devkit):
    id_to_wnid, ground_truth = parse_official_devkit(official_devkit)
    assert len(id_to_wnid) == 1000
    assert len(ground_truth) == 50_000
    rows = build_manifest(DEFAULT_VAL_ROOT, devkit_path=official_devkit)
    assert len(rows) == 500
    assert all(row.imagenet1k_index is not None for row in rows)
    # Image 441's official WNID comes from row 441, not its position in this subset.
    row = next(item for item in rows if item.relative_path == "ILSVRC2012_val_00000441.JPEG")
    assert row.wnid == "n02091635"
    assert row.imagenet1k_index == ImageNet1KMapping().index_for_wnid(row.wnid)


@pytest.fixture(scope="session")
def official_devkit():
    from pathlib import Path
    path = Path("/tmp/ILSVRC2012_devkit_t12.tar.gz")
    if not path.is_file():
        pytest.skip("official devkit fixture unavailable")
    return path

