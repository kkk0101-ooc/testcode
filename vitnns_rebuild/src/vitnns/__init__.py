"""Auditable native ViT baseline and conditional Value-path attribution."""

from .modeling import MODEL_NAME, load_canonical_model

__all__ = ["MODEL_NAME", "load_canonical_model"]
__version__ = "0.2.0"

