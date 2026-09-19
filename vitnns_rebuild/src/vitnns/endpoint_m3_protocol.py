"""Frozen deterministic selection rules for Endpoint M3."""
from __future__ import annotations

import hashlib
import itertools
from typing import Iterable


def _score(namespace: str, *parts: object) -> bytes:
    return hashlib.sha256("|".join((namespace, *(str(x) for x in parts))).encode()).digest()


def select_shared_cohort(rows: Iterable[dict], count: int, namespace: str) -> list[dict]:
    """Select a shared image cohort without looking at model outcomes."""
    rows = list(rows)
    if count <= 0 or count > len(rows):
        raise ValueError("INVALID_COHORT_COUNT")
    if len({(r["sample_id"], r["image_sha256"]) for r in rows}) != len(rows):
        raise ValueError("DUPLICATE_IMAGE_IDENTITY")
    return sorted(rows, key=lambda r: (_score(namespace, r["sample_id"], r["image_sha256"]), r["sample_id"]))[:count]


def stratified_patch_tokens(grid: int = 14, row_strata: int = 3, column_strata: int = 5) -> tuple[int, ...]:
    """One patch nearest each rectangular stratum center, returned as ViT token ids."""
    if grid <= 0 or row_strata <= 0 or column_strata <= 0:
        raise ValueError("INVALID_STRATA")
    selected = []
    for sr in range(row_strata):
        lo_r, hi_r = sr * grid / row_strata, (sr + 1) * grid / row_strata
        for sc in range(column_strata):
            lo_c, hi_c = sc * grid / column_strata, (sc + 1) * grid / column_strata
            center_r, center_c = (lo_r + hi_r - 1) / 2, (lo_c + hi_c - 1) / 2
            candidates = [
                (abs(r - center_r) ** 2 + abs(c - center_c) ** 2, r, c)
                for r in range(grid) for c in range(grid)
                if lo_r <= r < hi_r and lo_c <= c < hi_c
            ]
            _, row, column = min(candidates)
            selected.append(1 + row * grid + column)
    if len(selected) != row_strata * column_strata or len(set(selected)) != len(selected):
        raise ValueError("TOKEN_SELECTION_NOT_UNIQUE")
    return tuple(selected)


def selected_tokens() -> tuple[int, ...]:
    return (0, *stratified_patch_tokens())


def selected_channels(model_id: str, block: int, width: int, count: int, namespace: str) -> tuple[int, ...]:
    if not model_id or block < 0 or width <= 0 or not 0 < count <= width:
        raise ValueError("INVALID_CHANNEL_SCOPE")
    values = sorted(range(width), key=lambda channel: (_score(namespace, model_id, block, channel), channel))[:count]
    return tuple(sorted(values))


def validation_targets(tokens: tuple[int, ...], channels: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if len(tokens) < 2 or len(channels) < 4:
        raise ValueError("VALIDATION_SCOPE_TOO_SMALL")
    return tuple(itertools.product(tokens[:2], channels[:4]))


def similarity_pairs(
    model_id: str, block: int, tokens: tuple[int, ...], channels: tuple[int, ...],
    group: str, count: int, namespace: str,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    units = tuple(itertools.product(tokens, channels))
    pairs = []
    for left, right in itertools.combinations(units, 2):
        same_token = left[0] == right[0]
        if group == "same_token_different_channel" and same_token:
            pairs.append((left, right))
        elif group == "different_token" and not same_token:
            pairs.append((left, right))
    if group not in {"same_token_different_channel", "different_token"}:
        raise ValueError("UNKNOWN_SIMILARITY_GROUP")
    if count > len(pairs):
        raise ValueError("PAIR_SCOPE_TOO_SMALL")
    return tuple(sorted(pairs, key=lambda p: (_score(namespace, model_id, block, group, p), p))[:count])


def rademacher(shape: tuple[int, ...], namespace: str, *parts: object):
    """Generate a deterministic Rademacher tensor without global RNG state."""
    import numpy as np
    seed = int.from_bytes(_score(namespace, *parts)[:8], "big")
    rng = np.random.default_rng(seed)
    return np.where(rng.integers(0, 2, size=shape, dtype=np.int8) == 0, -1.0, 1.0)
