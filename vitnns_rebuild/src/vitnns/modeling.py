from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import timm
import torch
from timm.data import create_transform, resolve_model_data_config
from timm.layers import set_fused_attn

from .constants import DEFAULT_STATE_DICT, MODEL_NAME, MODEL_SHA256
from .io_utils import sha256_file


@dataclass(frozen=True)
class PreprocessSpec:
    input_size: tuple[int, int, int]
    resize_shorter_side: int
    center_crop: tuple[int, int]
    interpolation: str
    antialias: bool
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    crop_pct: float
    crop_mode: str
    color_mode: str = "RGB"

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_data_config(model: torch.nn.Module) -> dict:
    return resolve_model_data_config(model)


def canonical_preprocess_spec(model: torch.nn.Module) -> PreprocessSpec:
    cfg = canonical_data_config(model)
    height, width = cfg["input_size"][-2:]
    resize = int(height / cfg["crop_pct"])
    return PreprocessSpec(
        input_size=tuple(cfg["input_size"]),
        resize_shorter_side=resize,
        center_crop=(height, width),
        interpolation=str(cfg["interpolation"]),
        antialias=True,
        mean=tuple(cfg["mean"]),
        std=tuple(cfg["std"]),
        crop_pct=float(cfg["crop_pct"]),
        crop_mode=str(cfg["crop_mode"]),
    )


def canonical_transform(model: torch.nn.Module):
    return create_transform(**canonical_data_config(model), is_training=False)


def load_canonical_model(
    state_dict_path: Path = DEFAULT_STATE_DICT,
    *,
    device: str | torch.device = "cpu",
    verify_hash: bool = True,
    unfused_attention: bool = False,
) -> torch.nn.Module:
    """Build the one authoritative Stage A model from a safe state dict."""
    path = Path(state_dict_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"State dict not found: {path}")
    if verify_hash:
        actual = sha256_file(path)
        if actual != MODEL_SHA256:
            raise ValueError(f"State-dict SHA-256 mismatch: expected {MODEL_SHA256}, got {actual}")

    # This must happen before model construction; each Attention instance snapshots
    # the fused-attention setting in its constructor.
    set_fused_attn(not unfused_attention)
    model = timm.create_model(MODEL_NAME, pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    if unfused_attention:
        for block in model.blocks:
            block.attn.fused_attn = False
    _assert_model_contract(model)
    return model


def _assert_model_contract(model: torch.nn.Module) -> None:
    if len(model.blocks) != 12 or model.embed_dim != 768:
        raise RuntimeError("Unexpected ViT-B architecture")
    if any(block.attn.num_heads != 12 or block.attn.head_dim != 64 for block in model.blocks):
        raise RuntimeError("Unexpected attention geometry")
    if sum(parameter.numel() for parameter in model.parameters()) != 86_567_656:
        raise RuntimeError("Unexpected parameter count")

