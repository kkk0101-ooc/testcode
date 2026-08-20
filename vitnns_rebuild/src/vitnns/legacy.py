from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from timm.models.vision_transformer import Attention

from .constants import DEFAULT_FULL_MODEL, FULL_MODEL_SHA256
from .io_utils import sha256_file


def load_trusted_legacy_full_model(
    path: Path = DEFAULT_FULL_MODEL,
    *,
    device: str | torch.device = "cpu",
    allow_unsafe_pickle: bool = False,
) -> torch.nn.Module:
    """Compatibility-only loader for the known local pickle.

    A matching hash identifies the artifact but does not make pickle safe. The
    caller must explicitly opt in. Production code must use the state dict.
    """
    if not allow_unsafe_pickle:
        raise PermissionError("Full-model pickle loading requires allow_unsafe_pickle=True")
    path = Path(path).resolve()
    actual = sha256_file(path)
    if actual != FULL_MODEL_SHA256:
        raise ValueError(f"Legacy full-model SHA-256 mismatch: {actual}")

    module_name = "timm.layers.attention"
    previous = sys.modules.get(module_name)
    shim = types.ModuleType(module_name)
    shim.Attention = Attention
    sys.modules[module_name] = shim
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    model.eval().to(device)
    return model

