"""Framework-neutral interfaces for later NNS-CST production adapters.

N1 freezes interfaces only. Concrete timm/Torch adapters, image access and
model execution are intentionally absent until a separately authorized stage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class PortIdentity:
    model_id: str
    checkpoint_sha256: str
    graph_revision: str
    block_index: int
    branch: str
    producer_tensor: str
    consumer_operator: str
    input_slot: int
    invocation_index: int


@dataclass(frozen=True)
class ActivationBatch:
    layout: str
    token_ids: Sequence[str]
    channel_dimension: int
    values: Any


@dataclass(frozen=True)
class FrozenSubspace:
    layer_id: str
    rank: int
    channel_dimension: int
    basis: Any
    projector_digest: str
    contract_digest: str


class ModelGraphAdapter(ABC):
    """Required boundary for a later, separately reviewed ViT adapter."""

    @abstractmethod
    def registered_ports(self) -> Sequence[PortIdentity]:
        raise NotImplementedError

    @abstractmethod
    def capture(self, inputs: Any, port_ids: Sequence[str]) -> Mapping[str, ActivationBatch]:
        raise NotImplementedError

    @abstractmethod
    def jvp(self, source_port_id: str, target_port_id: str, direction: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def vjp(self, source_port_id: str, target_port_id: str, cotangent: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def intervene(self, port_id: str, replacement: Any, metadata: Dict[str, Any]) -> Any:
        raise NotImplementedError
