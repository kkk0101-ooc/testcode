"""Standard-library NNS-CST mathematical and graph-contract skeleton.

This module deliberately contains no model, image, GPU, Torch, or NumPy code.
It is a small auditable reference for N1 analytic fixtures. Production ViT
adapters are outside the authorized N1 scope.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple


FAIL_RANK_EXCEEDS_DIMENSION = "RANK_EXCEEDS_DIMENSION"
FAIL_RANK_DEFICIENT = "RANK_DEFICIENT"
FAIL_ZERO_PAIR_DIFFERENCE = "ZERO_PAIR_DIFFERENCE"
FAIL_NO_SEMANTIC_AFTER_CONTROL = "NO_SEMANTIC_AFTER_CONTROL"
FAIL_NEAR_SINGULAR_CONTROL_COVARIANCE = "NEAR_SINGULAR_CONTROL_COVARIANCE"
FAIL_NONFINITE_VALUE = "NONFINITE_VALUE"
FAIL_MISSING_TOKEN = "MISSING_TOKEN"
FAIL_TOKEN_ALIGNMENT = "TOKEN_ALIGNMENT_ERROR"
FAIL_PORT_IDENTITY = "PORT_IDENTITY_MISMATCH"
FAIL_PORT_DUPLICATE = "DUPLICATE_PORT"
FAIL_CUT_MISSING_EDGE = "CUT_MISSING_EDGE"
FAIL_OFF_MANIFOLD = "OFF_MANIFOLD_INTERVENTION"

PORT_IDENTITY_FIELDS = (
    "model_id", "checkpoint_sha256", "graph_revision", "block_index",
    "branch", "producer_tensor", "consumer_operator", "input_slot",
    "invocation_index",
)


class CSTError(RuntimeError):
    """Typed, fail-closed N1 contract violation."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else code + ": " + detail)


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _vector(value: Sequence[float], name: str = "vector") -> List[float]:
    result = [float(item) for item in value]
    if not result:
        raise CSTError(FAIL_RANK_DEFICIENT, name + " is empty")
    if not _finite(result):
        raise CSTError(FAIL_NONFINITE_VALUE, name)
    return result


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot dimension mismatch")
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def norm(value: Sequence[float]) -> float:
    return math.sqrt(max(0.0, dot(value, value)))


def add(left: Sequence[float], right: Sequence[float]) -> List[float]:
    if len(left) != len(right):
        raise ValueError("add dimension mismatch")
    return [float(a) + float(b) for a, b in zip(left, right)]


def subtract(left: Sequence[float], right: Sequence[float]) -> List[float]:
    if len(left) != len(right):
        raise ValueError("subtract dimension mismatch")
    return [float(a) - float(b) for a, b in zip(left, right)]


def scale(value: Sequence[float], factor: float) -> List[float]:
    return [float(factor) * float(item) for item in value]


def matvec(matrix: Sequence[Sequence[float]], value: Sequence[float]) -> List[float]:
    if not matrix or any(len(row) != len(value) for row in matrix):
        raise ValueError("matrix/vector dimension mismatch")
    return [dot(row, value) for row in matrix]


def transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("ragged matrix")
    return [[float(matrix[i][j]) for i in range(len(matrix))] for j in range(width)]


def orthonormalize(vectors: Sequence[Sequence[float]], tolerance: float = 1e-12) -> List[List[float]]:
    """Deterministic modified Gram-Schmidt; basis vectors are returned as rows."""
    if not vectors:
        raise CSTError(FAIL_RANK_DEFICIENT, "no basis vectors")
    dimension = len(vectors[0])
    if dimension == 0 or any(len(row) != dimension for row in vectors):
        raise CSTError(FAIL_RANK_DEFICIENT, "ragged or empty basis")
    basis: List[List[float]] = []
    for index, raw in enumerate(vectors):
        residual = _vector(raw, "basis[%d]" % index)
        for _ in range(2):
            for existing in basis:
                residual = subtract(residual, scale(existing, dot(residual, existing)))
        length = norm(residual)
        if length <= tolerance:
            raise CSTError(FAIL_RANK_DEFICIENT, "dependent basis vector %d" % index)
        normalized = scale(residual, 1.0 / length)
        pivot = max(range(dimension), key=lambda j: abs(normalized[j]))
        if normalized[pivot] < 0.0:
            normalized = scale(normalized, -1.0)
        basis.append(normalized)
    return basis


def project(value: Sequence[float], basis: Sequence[Sequence[float]]) -> List[float]:
    vector = _vector(value)
    if not basis:
        return [0.0] * len(vector)
    if any(len(row) != len(vector) for row in basis):
        raise ValueError("project dimension mismatch")
    result = [0.0] * len(vector)
    for direction in basis:
        result = add(result, scale(direction, dot(vector, direction)))
    return result


def complement(value: Sequence[float], basis: Sequence[Sequence[float]]) -> List[float]:
    return subtract(value, project(value, basis))


def projector_matrix(basis: Sequence[Sequence[float]]) -> List[List[float]]:
    if not basis:
        raise CSTError(FAIL_RANK_DEFICIENT, "empty projector basis")
    dimension = len(basis[0])
    return [[math.fsum(row[i] * row[j] for row in basis) for j in range(dimension)] for i in range(dimension)]


def projector_distance(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    """Normalized Frobenius distance between projectors."""
    p_left = projector_matrix(left)
    p_right = projector_matrix(right)
    if len(p_left) != len(p_right):
        raise ValueError("projector dimensions differ")
    rank_scale = max(1.0, float(max(len(left), len(right))))
    squared = math.fsum(
        (p_left[i][j] - p_right[i][j]) ** 2
        for i in range(len(p_left)) for j in range(len(p_left))
    )
    return math.sqrt(squared / rank_scale)


def projector_similarity(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    """Mean squared canonical correlation, in [0, 1] for equal ranks."""
    if len(left) != len(right) or not left:
        raise ValueError("similarity requires equal nonzero ranks")
    return math.fsum(dot(a, b) ** 2 for a in left for b in right) / float(len(left))


def _covariance(vectors: Sequence[Sequence[float]], weights: Sequence[float], ridge: float) -> List[List[float]]:
    dimension = len(vectors[0])
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    result = [[0.0] * dimension for _ in range(dimension)]
    for vector, weight in zip(vectors, weights):
        for i in range(dimension):
            for j in range(dimension):
                result[i][j] += float(weight) * vector[i] * vector[j] / total
    for i in range(dimension):
        result[i][i] += ridge
    return result


def _top_eigenvectors(matrix: Sequence[Sequence[float]], rank: int, tolerance: float) -> Tuple[List[List[float]], List[float]]:
    dimension = len(matrix)
    residual = [[float(value) for value in row] for row in matrix]
    basis: List[List[float]] = []
    eigenvalues: List[float] = []
    for component in range(rank):
        best_index = max(range(dimension), key=lambda i: abs(residual[i][i]))
        vector = [1.0 if i == best_index else 0.0 for i in range(dimension)]
        for _ in range(256):
            candidate = matvec(residual, vector)
            for existing in basis:
                candidate = subtract(candidate, scale(existing, dot(candidate, existing)))
            length = norm(candidate)
            if length <= tolerance:
                raise CSTError(FAIL_RANK_DEFICIENT, "component %d" % component)
            candidate = scale(candidate, 1.0 / length)
            if min(norm(subtract(candidate, vector)), norm(add(candidate, vector))) <= tolerance:
                vector = candidate
                break
            vector = candidate
        eigenvalue = dot(vector, matvec(matrix, vector))
        if eigenvalue <= tolerance:
            raise CSTError(FAIL_RANK_DEFICIENT, "non-positive component %d" % component)
        pivot = max(range(dimension), key=lambda j: abs(vector[j]))
        if vector[pivot] < 0.0:
            vector = scale(vector, -1.0)
        basis.append(vector)
        eigenvalues.append(eigenvalue)
        for i in range(dimension):
            for j in range(dimension):
                residual[i][j] -= eigenvalue * vector[i] * vector[j]
    return orthonormalize(basis, tolerance), eigenvalues


def estimate_subspace(
    paired_differences: Sequence[Sequence[float]],
    rank: int,
    token_weights: Sequence[float] = (),
    control_differences: Sequence[Sequence[float]] = (),
    ridge: float = 0.0,
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    """Estimate a residualized paired-difference subspace."""
    if not paired_differences:
        raise CSTError(FAIL_ZERO_PAIR_DIFFERENCE, "no paired rows")
    rows = [_vector(row, "paired_difference") for row in paired_differences]
    dimension = len(rows[0])
    if any(len(row) != dimension for row in rows):
        raise CSTError(FAIL_MISSING_TOKEN, "inconsistent activation dimension")
    if rank < 1 or rank > dimension:
        raise CSTError(FAIL_RANK_EXCEEDS_DIMENSION, "rank=%d dimension=%d" % (rank, dimension))
    weights = [1.0] * len(rows) if not token_weights else [float(value) for value in token_weights]
    if len(weights) != len(rows) or not _finite(weights) or any(value < 0.0 for value in weights):
        raise ValueError("invalid token weights")
    if max(norm(row) for row in rows) <= tolerance:
        raise CSTError(FAIL_ZERO_PAIR_DIFFERENCE)

    control_basis: List[List[float]] = []
    if control_differences:
        controls = [_vector(row, "control_difference") for row in control_differences]
        if any(len(row) != dimension for row in controls):
            raise CSTError(FAIL_MISSING_TOKEN, "control dimension mismatch")
        if any(norm(row) <= tolerance for row in controls):
            raise CSTError(FAIL_NEAR_SINGULAR_CONTROL_COVARIANCE, "zero control direction")
        try:
            control_basis = orthonormalize(controls, tolerance)
        except CSTError as exc:
            raise CSTError(FAIL_NEAR_SINGULAR_CONTROL_COVARIANCE, exc.detail)
        rows = [complement(row, control_basis) for row in rows]
        if max(norm(row) for row in rows) <= tolerance:
            raise CSTError(FAIL_NO_SEMANTIC_AFTER_CONTROL)

    covariance = _covariance(rows, weights, ridge)
    basis, eigenvalues = _top_eigenvectors(covariance, rank, tolerance)
    return {
        "basis": basis,
        "eigenvalues": eigenvalues,
        "dimension": dimension,
        "rank": rank,
        "control_rank": len(control_basis),
        "projector": projector_matrix(basis),
    }


def fit_frozen_readout(
    positive: Sequence[Sequence[float]],
    negative: Sequence[Sequence[float]],
    basis: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    if not positive or not negative:
        raise ValueError("readout needs both classes")
    dimension = len(positive[0])
    if any(len(row) != dimension for row in list(positive) + list(negative)):
        raise CSTError(FAIL_MISSING_TOKEN, "readout row dimension")
    mean_pos = [math.fsum(row[j] for row in positive) / len(positive) for j in range(dimension)]
    mean_neg = [math.fsum(row[j] for row in negative) / len(negative) for j in range(dimension)]
    direction = project(subtract(mean_pos, mean_neg), basis)
    length = norm(direction)
    if length <= 1e-12:
        raise CSTError(FAIL_ZERO_PAIR_DIFFERENCE, "readout means coincide in subspace")
    direction = scale(direction, 1.0 / length)
    threshold = 0.5 * (dot(direction, mean_pos) + dot(direction, mean_neg))
    return {
        "schema_version": 1,
        "method": "projected_class_mean_difference",
        "weights": direction,
        "threshold": threshold,
        "rank": len(basis),
        "dimension": dimension,
        "training_only_fit": True,
        "validation_updates_allowed": False,
    }


def readout_score(bundle: Dict[str, Any], value: Sequence[float]) -> float:
    return dot(bundle["weights"], value) - float(bundle["threshold"])


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def transport_linear(matrix: Sequence[Sequence[float]], value: Sequence[float]) -> List[float]:
    return matvec(matrix, value)


def transport_residual(
    identity_matrix: Sequence[Sequence[float]],
    branch_matrix: Sequence[Sequence[float]],
    value: Sequence[float],
) -> Dict[str, List[float]]:
    identity = matvec(identity_matrix, value)
    branch = matvec(branch_matrix, value)
    return {"identity": identity, "branch": branch, "merged": add(identity, branch)}


def transport_two_branch(
    first_matrix: Sequence[Sequence[float]],
    second_matrix: Sequence[Sequence[float]],
    value: Sequence[float],
) -> Dict[str, List[float]]:
    first = matvec(first_matrix, value)
    second = matvec(second_matrix, value)
    return {"first": first, "second": second, "merged": add(first, second)}


def semantic_complement_attribution(
    value: Sequence[float],
    gradient: Sequence[float],
    basis: Sequence[Sequence[float]],
) -> Dict[str, List[float]]:
    if len(value) != len(gradient):
        raise ValueError("attribution dimension mismatch")
    semantic = project(value, basis)
    other = complement(value, basis)
    semantic_attr = [a * float(g) for a, g in zip(semantic, gradient)]
    complement_attr = [a * float(g) for a, g in zip(other, gradient)]
    return {
        "semantic": semantic_attr,
        "complement": complement_attr,
        "reconstructed": add(semantic_attr, complement_attr),
        "complete_cut": [float(x) * float(g) for x, g in zip(value, gradient)],
    }


def validate_port_registry(records: Sequence[Dict[str, Any]], expected_ids: Sequence[str]) -> bool:
    seen: Dict[str, int] = {}
    expected = list(expected_ids)
    for record in records:
        if set(record) != set(PORT_IDENTITY_FIELDS) | {"port_id", "tensor_role", "token_set"}:
            raise CSTError(FAIL_PORT_IDENTITY, "field set")
        identity = tuple(record[field] for field in PORT_IDENTITY_FIELDS)
        if any(value is None or value == "" for value in identity):
            raise CSTError(FAIL_PORT_IDENTITY, record.get("port_id", "unknown"))
        port_id = str(record["port_id"])
        identity_text = "|".join(str(record[field]) for field in PORT_IDENTITY_FIELDS)
        derived_id = str(record["model_id"]) + ":" + hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:20]
        if port_id != derived_id:
            raise CSTError(FAIL_PORT_IDENTITY, port_id + " does not bind its identity")
        seen[port_id] = seen.get(port_id, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count != 1)
    if duplicates:
        raise CSTError(FAIL_PORT_DUPLICATE, ",".join(duplicates))
    missing = sorted(set(expected) - set(seen))
    extra = sorted(set(seen) - set(expected))
    if missing:
        raise CSTError(FAIL_CUT_MISSING_EDGE, ",".join(missing[:3]))
    if extra:
        raise CSTError(FAIL_PORT_IDENTITY, "unexpected ports: " + ",".join(extra[:3]))
    return True


def validate_token_alignment(source: Sequence[str], target: Sequence[str], mapping: Sequence[Tuple[str, str]]) -> bool:
    if not source or not target:
        raise CSTError(FAIL_MISSING_TOKEN, "empty topology")
    source_set, target_set = set(source), set(target)
    if len(source_set) != len(source) or len(target_set) != len(target):
        raise CSTError(FAIL_TOKEN_ALIGNMENT, "duplicate token id")
    mapped_source = [pair[0] for pair in mapping]
    mapped_target = [pair[1] for pair in mapping]
    if (set(mapped_source) != source_set or set(mapped_target) != target_set
            or len(mapped_source) != len(set(mapped_source))
            or len(mapped_target) != len(set(mapped_target))):
        raise CSTError(FAIL_TOKEN_ALIGNMENT, "mapping is not bijective")
    return True


def intervene(
    recipient: Sequence[float],
    basis: Sequence[Sequence[float]],
    operation: str,
    donor: Sequence[float] = (),
    dose: float = 1.0,
    manifold_limit: float = 10.0,
) -> List[float]:
    original = _vector(recipient, "recipient")
    semantic = project(original, basis)
    if operation == "erase":
        result = subtract(original, scale(semantic, dose))
    elif operation == "inject":
        if not donor or len(donor) != len(original):
            raise ValueError("inject needs a dimension-matched donor")
        donor_semantic = project(_vector(donor, "donor"), basis)
        result = add(original, scale(subtract(donor_semantic, semantic), dose))
    elif operation == "orthogonal_complement":
        result = complement(original, basis)
    elif operation == "random_subspace":
        dimension = len(original)
        rank = len(basis)
        random_basis = [[1.0 if j == (i + rank) % dimension else 0.0 for j in range(dimension)] for i in range(rank)]
        result = subtract(original, scale(project(original, random_basis), dose))
    else:
        raise ValueError("unknown intervention: " + operation)
    shift = norm(subtract(result, original))
    scale_reference = max(1.0, norm(original))
    if not _finite(result) or shift > manifold_limit * scale_reference:
        raise CSTError(FAIL_OFF_MANIFOLD, "relative_shift=%.12g" % (shift / scale_reference))
    return result
