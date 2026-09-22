"""Numerical core for the finite N2 pilot search."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


class N2TypedError(ValueError):
    """Fail-closed numerical or contract error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_finite(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise N2TypedError("NONFINITE_VALUE", f"{name} must be a finite matrix")
    return array


def row_basis(value: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    matrix = require_finite(value, "row-basis input")
    if matrix.shape[0] == 0:
        return np.empty((0, matrix.shape[1]), dtype=np.float64)
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] == 0:
        return np.empty((0, matrix.shape[1]), dtype=np.float64)
    keep = singular > tolerance * singular[0]
    return np.ascontiguousarray(vh[keep], dtype=np.float64)


def nuisance_residualize(signal: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal_matrix = require_finite(signal, "signal")
    control_matrix = require_finite(control, "control")
    if signal_matrix.shape[1] != control_matrix.shape[1]:
        raise N2TypedError("DIMENSION_MISMATCH", "signal/control channel dimensions differ")
    nuisance = row_basis(control_matrix)
    residual = signal_matrix.copy()
    if nuisance.shape[0]:
        residual -= (residual @ nuisance.T) @ nuisance
    if not np.isfinite(residual).all():
        raise N2TypedError("NONFINITE_VALUE", "residualization produced nonfinite values")
    return residual, nuisance


def subspace_basis(signal: np.ndarray, rank: int, ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    matrix = require_finite(signal, "subspace signal")
    if rank < 1 or rank > matrix.shape[1]:
        raise N2TypedError("RANK_EXCEEDS_DIMENSION", "requested rank is outside channel dimension")
    if ridge < 0 or not np.isfinite(ridge):
        raise N2TypedError("INVALID_REGULARIZATION", "ridge must be finite and nonnegative")
    covariance = matrix.T @ matrix / max(1, matrix.shape[0])
    covariance.flat[:: covariance.shape[0] + 1] += ridge
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    selected = eigenvalues[order[:rank]]
    if selected.size < rank or selected[-1] <= max(1e-12, ridge * 1.000001):
        raise N2TypedError("RANK_DEFICIENT", "requested positive semantic directions do not exist")
    basis = eigenvectors[:, order[:rank]].T
    return np.ascontiguousarray(basis), np.ascontiguousarray(selected)


def projector(basis: np.ndarray) -> np.ndarray:
    value = require_finite(basis, "basis")
    gram = value @ value.T
    if not np.allclose(gram, np.eye(value.shape[0]), atol=1e-8, rtol=1e-8):
        raise N2TypedError("NON_ORTHONORMAL_BASIS", "basis rows are not orthonormal")
    return np.ascontiguousarray(value.T @ value)


def projector_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = projector(left); b = projector(right)
    rank = max(left.shape[0], right.shape[0])
    return float(np.trace(a @ b) / rank)


def random_basis(dimension: int, rank: int, seed: int) -> np.ndarray:
    if rank < 1 or rank > dimension:
        raise N2TypedError("RANK_EXCEEDS_DIMENSION", "random rank is invalid")
    matrix = np.random.Generator(np.random.PCG64(seed)).normal(size=(dimension, rank))
    q, _r = np.linalg.qr(matrix)
    return np.ascontiguousarray(q[:, :rank].T, dtype=np.float64)


def fit_readout(positive: np.ndarray, negative: np.ndarray, basis: np.ndarray) -> dict[str, Any]:
    pos = require_finite(positive, "positive readout samples")
    neg = require_finite(negative, "negative readout samples")
    if pos.shape != neg.shape or pos.shape[1] != basis.shape[1]:
        raise N2TypedError("DIMENSION_MISMATCH", "readout inputs do not match")
    projection = projector(basis)
    direction = projection @ (pos.mean(0) - neg.mean(0))
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise N2TypedError("ZERO_PAIR_DIFFERENCE", "readout direction is zero")
    weights = direction / norm
    threshold = float(0.5 * (pos.mean(0) @ weights + neg.mean(0) @ weights))
    return {
        "weights": weights.tolist(),
        "threshold": threshold,
        "rank": int(basis.shape[0]),
        "dimension": int(basis.shape[1]),
    }
