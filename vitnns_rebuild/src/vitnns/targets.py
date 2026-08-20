"""Serializable attribution targets.

Only raw class logits are supported.  A deliberately small, strict schema
prevents a saved request from silently changing meaning (for example, from a
logit to a probability).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch


CLASS_LOGIT = "class_logit"


@dataclass(frozen=True)
class TargetSpec:
    """Strict JSON-serializable raw class-logit target."""

    class_index: int
    kind: str = CLASS_LOGIT

    def __post_init__(self) -> None:
        if self.kind != CLASS_LOGIT:
            raise ValueError(
                f"unsupported target kind {self.kind!r}; only {CLASS_LOGIT!r} is supported"
            )
        if isinstance(self.class_index, bool) or not isinstance(self.class_index, int):
            raise TypeError("class_index must be an integer")
        if self.class_index < 0:
            raise ValueError("class_index must be non-negative")

    @classmethod
    def class_logit(cls, class_index: int) -> "TargetSpec":
        return cls(class_index=class_index)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "class_index": self.class_index}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetSpec":
        if not isinstance(value, Mapping):
            raise TypeError("serialized target must be a mapping")
        expected = {"kind", "class_index"}
        keys = set(value.keys())
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError("invalid target schema (" + ", ".join(details) + ")")
        return cls(class_index=value["class_index"], kind=value["kind"])

    def select(self, logits: torch.Tensor) -> torch.Tensor:
        """Select one raw logit per item from ``[B, classes]`` logits."""

        if not isinstance(logits, torch.Tensor):
            raise TypeError("class_logit target requires a Tensor model output")
        if logits.ndim != 2:
            raise ValueError(
                f"class_logit target requires [B, classes], got {tuple(logits.shape)}"
            )
        if self.class_index >= logits.shape[1]:
            raise IndexError(
                f"class_index {self.class_index} is out of range for {logits.shape[1]} classes"
            )
        return logits[:, self.class_index]


# A descriptive alias, not a second target type or schema.
ClassLogitTarget = TargetSpec


def parse_target(value: TargetSpec | Mapping[str, Any]) -> TargetSpec:
    if isinstance(value, TargetSpec):
        return value
    if isinstance(value, Mapping):
        return TargetSpec.from_dict(value)
    raise TypeError("target must be a TargetSpec or serialized target mapping")


__all__ = [
    "CLASS_LOGIT",
    "ClassLogitTarget",
    "TargetSpec",
    "parse_target",
]
