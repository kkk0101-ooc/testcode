"""Self-test for the process-wide D4 synthetic-only safety guard."""

from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess

import pytest
import torch


def test_d4_guard_is_mandatory_and_blocks_forbidden_operations() -> None:
    assert os.environ.get("VITNNS_D4_STRICT") == "1"
    assert os.environ.get("VITNNS_D4_GUARD_ACTIVE") == "1"
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    for path in (
        "/tmp/d4-probe.JPEG",
        "/tmp/d4-probe.pth",
        "/tmp/d4-probe.ckpt",
        "/tmp/stage_c_plus_v2/probe.json",
        "/tmp/stage_d_outcomes/probe.json",
        "/dev/nvidia0",
    ):
        with pytest.raises(PermissionError):
            Path(path).read_bytes()
        with pytest.raises(PermissionError):
            os.stat(path)

    for safe_path in ("/tmp/d4-synthetic.npy", "/tmp/d4-synthetic.npz"):
        with pytest.raises(FileNotFoundError):
            Path(safe_path).read_bytes()

    with pytest.raises(ImportError):
        __import__("vitnns.modeling")
    with pytest.raises(PermissionError):
        torch.nn.Linear(2, 2)
    with pytest.raises(PermissionError):
        torch.load(io.BytesIO(b"not-a-checkpoint"))
    with pytest.raises(PermissionError):
        torch.cuda.synchronize()
    with pytest.raises(PermissionError):
        subprocess.Popen(["true"])
    with pytest.raises(PermissionError):
        os.system("true")
