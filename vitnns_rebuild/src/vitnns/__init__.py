"""Auditable native ViT baseline and conditional Value-path attribution."""

from __future__ import annotations

from .constants import MODEL_NAME


def __getattr__(name: str):
    if name != "load_canonical_model":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .modeling import load_canonical_model

    globals()[name] = load_canonical_model
    return load_canonical_model


__all__ = ["MODEL_NAME", "load_canonical_model"]
__version__ = "0.2.0"

