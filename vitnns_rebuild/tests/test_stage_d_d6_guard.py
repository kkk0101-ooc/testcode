"""Mandatory self-test for the D6 synthetic-only process guard."""

from __future__ import annotations

import os
from pathlib import Path


def test_d6_guard_is_active_and_is_the_d6_wrapper() -> None:
    import sitecustomize

    guard = Path(__file__).resolve().parent / "d6_guard" / "sitecustomize.py"
    assert os.environ.get("VITNNS_D6_STRICT") == "1"
    assert os.environ.get("VITNNS_D6_GUARD_ACTIVE") == "1"
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert Path(sitecustomize.__file__).resolve() == guard
