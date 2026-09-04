"""Mandatory process guard assertions for D7 non-reserve tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_d7_guard_blocks_reserve_paths_but_allows_synthetic_and_model() -> None:
    import sitecustomize

    guard = Path(__file__).resolve().parent / "d7_guard/sitecustomize.py"
    assert os.environ.get("VITNNS_D7_STRICT") == "1"
    assert os.environ.get("VITNNS_D7_GUARD_ACTIVE") == "1"
    assert Path(sitecustomize.__file__).resolve() == guard
    for path in (
        "/tmp/ILSVRC2012_val/reserve.JPEG",
        "/tmp/stage_d_results/result.json",
        "/tmp/vitnns_d8_formal_deadbee_20990101/control.json",
    ):
        with pytest.raises(PermissionError):
            os.stat(path)
        with pytest.raises(PermissionError):
            Path(path).read_bytes()
    for path in (
        "/tmp/d7-synthetic.npy",
        "/tmp/vit_base_patch16_224_weights.semantic_candidate.pth",
    ):
        with pytest.raises(FileNotFoundError):
            os.stat(path)
