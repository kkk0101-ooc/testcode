"""Model-free deterministic seed derivation shared by fidelity stages."""

from __future__ import annotations

import hashlib


class FidelitySeedError(ValueError):
    """A seed namespace or coordinate is malformed."""


def derive_fidelity_seed(
    namespace: str, sample_id: str, block: int, head: int, repeat_index: int,
    *, repeat_count: int = 5,
) -> int:
    """Return SHA-256 first-eight-byte big-endian seed for one coordinate."""

    if not isinstance(namespace, str) or not namespace.endswith("|"):
        raise FidelitySeedError(
            "fidelity seed namespace must be a non-empty string ending in '|'"
        )
    if not isinstance(sample_id, str) or not sample_id or "|" in sample_id:
        raise FidelitySeedError("sample_id must be non-empty and contain no '|'")
    for value, name in (
        (block, "block"), (head, "head"), (repeat_index, "repeat_index"),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"{name} must be a non-negative integer (not bool)")
    if block >= 12 or head >= 12:
        raise FidelitySeedError("block and head must be in [0,11]")
    if type(repeat_count) is not int or repeat_count <= 0:
        raise TypeError("repeat_count must be a positive integer (not bool)")
    if repeat_index >= repeat_count:
        raise FidelitySeedError(
            f"repeat_index must be in [0,{repeat_count - 1}]"
        )
    material = (
        f"{namespace}{sample_id}|{block}|{head}|{repeat_index}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(material).digest()[:8], byteorder="big", signed=False
    )


__all__ = ["FidelitySeedError", "derive_fidelity_seed"]
