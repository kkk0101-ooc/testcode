"""Mandatory, non-invasive self-test for the D5 process guard."""

from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess

import pytest
import torch


def test_d5_guard_is_active_and_blocks_forbidden_operations() -> None:
    import sitecustomize

    guard = Path(__file__).resolve().parent / "d5_guard" / "sitecustomize.py"
    assert os.environ.get("VITNNS_D5_STRICT") == "1"
    assert os.environ.get("VITNNS_D5_GUARD_ACTIVE") == "1"
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert Path(sitecustomize.__file__).resolve() == guard

    for path in (
        "/tmp/d5-guard-probe.JPEG",
        "/tmp/d5-guard-probe.pth",
        "/tmp/ILSVRC2012_val/d5-guard-probe",
        "/tmp/stage_d_outcomes/d5-guard-probe.json",
        "/tmp/stage_d_smoke/d5-guard-probe.json",
        "/tmp/vitnns_d9_formal_deadbee_20990101/d5-guard-probe.json",
        "/dev/nvidia0",
    ):
        with pytest.raises(PermissionError):
            Path(path).read_bytes()
        with pytest.raises(PermissionError):
            os.stat(path)

    for safe_path in ("/tmp/d5-synthetic.npy", "/tmp/d5-synthetic.npz"):
        with pytest.raises(FileNotFoundError):
            Path(safe_path).read_bytes()

    with pytest.raises(ImportError):
        __import__("vitnns.attribution")
    with pytest.raises(ImportError):
        __import__("vitnns.modeling")
    with pytest.raises(PermissionError):
        torch.nn.Module.load_state_dict(object(), {})
    with pytest.raises(PermissionError):
        torch.nn.Linear(2, 2)
    with pytest.raises(PermissionError):
        torch.load(io.BytesIO(b"not-a-checkpoint"))
    with pytest.raises(PermissionError):
        torch.inference_mode()
    with pytest.raises(PermissionError):
        torch.nn.functional.linear(object(), object())
    with pytest.raises(PermissionError):
        torch.Tensor.cuda(object())
    with pytest.raises(PermissionError):
        torch.cuda.synchronize()
    with pytest.raises(PermissionError):
        subprocess.Popen(["true"])
    with pytest.raises(PermissionError):
        subprocess.Popen(["git", "show", "HEAD:probe.JPEG"])
    with pytest.raises(PermissionError):
        os.system("true")
