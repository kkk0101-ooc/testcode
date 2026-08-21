from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


def split_fused_qkv(
    projected: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split timm's fused linear projection in the canonical Q, K, V order."""
    if projected.ndim != 3:
        raise ValueError(f"Expected [batch,tokens,3*embed], got {tuple(projected.shape)}")
    batch, tokens, width = projected.shape
    expected = 3 * num_heads * head_dim
    if width != expected:
        raise ValueError(f"Expected projected width {expected}, got {width}")
    return tuple(
        projected.reshape(batch, tokens, 3, num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
        .unbind(0)
    )


def recompute_attention(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    # Preserve timm's operation order for bitwise parity.
    return ((q * scale) @ k.transpose(-2, -1)).softmax(dim=-1)


@dataclass(frozen=True)
class LayerCapture:
    block_input: torch.Tensor
    block_output: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attention: torch.Tensor
    attention_used: torch.Tensor
    attention_recomputed: torch.Tensor
    scale: float

    @property
    def attention_max_abs_error(self) -> float:
        return float((self.attention - self.attention_recomputed).abs().max().item())


@dataclass(frozen=True)
class ExtractionResult:
    logits: torch.Tensor
    layers: tuple[LayerCapture, ...]

    @property
    def max_attention_abs_error(self) -> float:
        return max(layer.attention_max_abs_error for layer in self.layers)

    def tensor_dict(self, *, cpu: bool = True) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {"logits": self.logits}
        for index, layer in enumerate(self.layers):
            for name in ("block_input", "block_output", "q", "k", "v", "attention"):
                tensors[f"blocks.{index}.{name}"] = getattr(layer, name)
        if cpu:
            tensors = {name: value.detach().cpu() for name, value in tensors.items()}
        return tensors


class NativeViTExtractor:
    """Observe the real timm forward; never reimplement a transformer block."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._validate_observable_state()

    def _validate_observable_state(self) -> None:
        """Fail closed without changing caller-owned model state.

        The check is repeated for every extraction because callers can switch a
        model back to training mode or re-enable fused attention after this
        extractor is constructed.  In particular, ``attn_drop`` must remain an
        eval-mode identity for ``attention`` and ``attention_used`` to have the
        frozen Stage A/C semantics.
        """

        if self.model.training or any(
            module.training for module in self.model.modules()
        ):
            raise ValueError("Extraction requires every model module to remain in eval mode")
        if any(block.attn.fused_attn for block in self.model.blocks):
            raise ValueError("Extraction requires unfused attention so actual A can be hooked")

    @staticmethod
    def _save_output(store: dict, name: str) -> Callable:
        def hook(_module, _args, output):
            if name in store:
                raise RuntimeError(f"Hook fired more than once: {name}")
            store[name] = output.detach().clone()
        return hook

    @staticmethod
    def _save_input(store: dict, name: str) -> Callable:
        def hook(_module, args):
            if name in store:
                raise RuntimeError(f"Hook fired more than once: {name}")
            store[name] = args[0].detach().clone()
        return hook

    def extract(self, images: torch.Tensor) -> ExtractionResult:
        self._validate_observable_state()
        captures: dict[str, torch.Tensor] = {}
        handles = []
        try:
            # Register and append one hook at a time inside the protected
            # region.  If a later registration itself raises, every earlier
            # successfully registered hook is still removed below.
            for index, block in enumerate(self.model.blocks):
                prefix = f"blocks.{index}"
                handles.append(
                    block.register_forward_pre_hook(
                        self._save_input(captures, f"{prefix}.input")
                    )
                )
                handles.append(
                    block.register_forward_hook(
                        self._save_output(captures, f"{prefix}.output")
                    )
                )
                handles.append(
                    block.attn.qkv.register_forward_hook(
                        self._save_output(captures, f"{prefix}.qkv")
                    )
                )
                handles.append(
                    block.attn.q_norm.register_forward_hook(
                        self._save_output(captures, f"{prefix}.q")
                    )
                )
                handles.append(
                    block.attn.k_norm.register_forward_hook(
                        self._save_output(captures, f"{prefix}.k")
                    )
                )
                handles.append(
                    block.attn.attn_drop.register_forward_pre_hook(
                        self._save_input(captures, f"{prefix}.attention")
                    )
                )
                handles.append(
                    block.attn.attn_drop.register_forward_hook(
                        self._save_output(captures, f"{prefix}.attention_used")
                    )
                )
            with torch.inference_mode():
                logits = self.model(images).detach().clone()
        finally:
            for handle in reversed(handles):
                handle.remove()

        layers = []
        for index, block in enumerate(self.model.blocks):
            prefix = f"blocks.{index}"
            _, _, v = split_fused_qkv(
                captures[f"{prefix}.qkv"],
                num_heads=block.attn.num_heads,
                head_dim=block.attn.head_dim,
            )
            q = captures[f"{prefix}.q"]
            k = captures[f"{prefix}.k"]
            attention_recomputed = recompute_attention(q, k, block.attn.scale)
            layers.append(LayerCapture(
                block_input=captures[f"{prefix}.input"],
                block_output=captures[f"{prefix}.output"],
                q=q,
                k=k,
                v=v,
                attention=captures[f"{prefix}.attention"],
                attention_used=captures[f"{prefix}.attention_used"],
                attention_recomputed=attention_recomputed,
                scale=float(block.attn.scale),
            ))
        return ExtractionResult(logits=logits, layers=tuple(layers))


def capture_block_io(model: torch.nn.Module, images: torch.Tensor):
    """Capture only block boundaries without changing the model's native path."""
    inputs: list[torch.Tensor | None] = [None] * len(model.blocks)
    outputs: list[torch.Tensor | None] = [None] * len(model.blocks)
    handles = []
    for index, block in enumerate(model.blocks):
        def pre_hook(_module, args, layer=index):
            inputs[layer] = args[0].detach().clone()

        def post_hook(_module, _args, output, layer=index):
            outputs[layer] = output.detach().clone()

        handles.extend([block.register_forward_pre_hook(pre_hook), block.register_forward_hook(post_hook)])
    try:
        with torch.inference_mode():
            logits = model(images).detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in inputs + outputs):
        raise RuntimeError("A block boundary hook did not fire")
    return logits, tuple(inputs), tuple(outputs)

