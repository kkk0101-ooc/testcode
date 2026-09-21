"""CPU float64 Endpoint-PCIA-PEFA reference implementation.

The implementation keeps the use-port cotangent in the autograd graph. It
computes T directly from the Jacobian of Q, while R is independently computed
from the Jacobian of D. No CUDA path is present in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from . import analytic_fixtures

torch.set_default_dtype(torch.float64)


FAIL_PORT_ID_AMBIGUOUS = "FAIL_PORT_ID_AMBIGUOUS"
FAIL_CUT_NOT_LEGAL = "FAIL_CUT_NOT_LEGAL"
FAIL_SECOND_ORDER_GRAPH_TRUNCATED = "FAIL_SECOND_ORDER_GRAPH_TRUNCATED"
FAIL_NONSMOOTH_NODE_HIT = "FAIL_NONSMOOTH_NODE_HIT"
FAIL_QUADRATURE_CLOSURE = "FAIL_QUADRATURE_CLOSURE"


@dataclass(frozen=True)
class PointQuantities:
    value: np.ndarray
    gradient: np.ndarray
    port_d: np.ndarray
    port_jac_d: np.ndarray
    port_q: np.ndarray
    port_jac_q: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class PathQuantities:
    output_delta: np.ndarray
    gradient_x: np.ndarray
    endpoint: np.ndarray
    port_d_x: np.ndarray
    port_q_x: np.ndarray
    ig: np.ndarray
    a: np.ndarray
    t: np.ndarray
    r: np.ndarray
    omega_x: np.ndarray


def _torch_graph(fixture_id: str, u: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Build a fresh scalar DAG and return its ordered consumer-use inputs."""
    if u.dtype != torch.float64 or u.device.type != "cpu":
        raise TypeError("Step 2 accepts CPU torch.float64 tensors only")

    if fixture_id == "LINEAR3":
        z = u * 1.0
        coefficient = torch.tensor([1.5, -2.0, 0.25])
        return coefficient @ z, (z,)

    if fixture_id == "BILINEAR2":
        z0 = u[0] * 1.0
        z1 = u[1] * 1.0
        z_linear = u[0] * 1.0
        return z0 * z1 + 0.2 * z_linear, (z0, z1, z_linear)

    if fixture_id == "SATURATING3":
        coefficient = torch.tensor([0.7, -1.1, 0.5])
        z = coefficient @ u + 0.2
        return torch.tanh(z), (z,)

    if fixture_id == "FANOUT2":
        h = 0.6 * u[0] - 0.4 * u[1]
        z_linear = h * 1.0
        z_nonlinear = h * 1.0
        return 2.0 * z_linear + 0.7 * torch.tanh(z_nonlinear), (z_linear, z_nonlinear)

    if fixture_id == "RESIDUAL2":
        w = torch.tensor([[0.8, -0.3], [0.2, 0.9]])
        q = torch.tensor([0.1, -0.2])
        b_matrix = torch.tensor([[0.7, 0.1], [-0.4, 0.6]])
        v = torch.tensor([1.2, -0.5])
        identity_use = u * 1.0
        branch_use = b_matrix @ torch.tanh(w @ u + q)
        return v @ (identity_use + branch_use), (identity_use, branch_use)

    if fixture_id == "MULTIINPUT3":
        a = u[0] + 2.0 * u[1]
        c = 0.5 * u[1] - u[2]
        a_mul = a * 1.0
        c_mul = c * 1.0
        a_linear = a * 1.0
        c_linear = c * 1.0
        value = a_mul * c_mul + 0.3 * a_linear - 0.2 * c_linear
        return value, (a_mul, c_mul, a_linear, c_linear)

    raise KeyError(f"unknown Torch fixture: {fixture_id}")


def _derivative_rows(vector: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Differentiate a vector row-by-row without a Hessian convenience API."""
    rows = []
    for component in vector.reshape(-1):
        if not component.requires_grad:
            rows.append(torch.zeros_like(u))
            continue
        row = torch.autograd.grad(
            component, u, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        rows.append(torch.zeros_like(u) if row is None else row)
    return torch.stack(rows).reshape(vector.shape + u.shape)


def autodiff_point(fixture_id: str, u_array: np.ndarray, b_array: np.ndarray) -> PointQuantities:
    u = torch.tensor(np.asarray(u_array, dtype=np.float64), requires_grad=True)
    b = torch.tensor(np.asarray(b_array, dtype=np.float64))
    value, uses = _torch_graph(fixture_id, u)
    gradient = torch.autograd.grad(value, u, create_graph=True, retain_graph=True)[0]

    port_d_rows = []
    for use in uses:
        cotangent = torch.autograd.grad(value, use, create_graph=True, retain_graph=True)[0]
        allocation = torch.autograd.grad(
            use, u, grad_outputs=cotangent, create_graph=True, retain_graph=True
        )[0]
        port_d_rows.append(allocation)
    port_d = torch.stack(port_d_rows)

    jac_d = torch.stack([_derivative_rows(row, u) for row in port_d])
    port_q = (u - b).unsqueeze(0) * port_d
    jac_q = torch.stack([_derivative_rows(row, u) for row in port_q])
    omega = jac_d.transpose(1, 2) - jac_d

    def array(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float64, copy=True)

    return PointQuantities(
        value=array(value),
        gradient=array(gradient),
        port_d=array(port_d),
        port_jac_d=array(jac_d),
        port_q=array(port_q),
        port_jac_q=array(jac_q),
        omega=array(omega),
    )


def analytic_point(fixture_id: str, u_array: np.ndarray, b_array: np.ndarray) -> PointQuantities:
    u = np.asarray(u_array, dtype=np.float64)
    b = np.asarray(b_array, dtype=np.float64)
    raw = analytic_fixtures.evaluate(fixture_id, u)
    port_q = (u - b)[None, :] * raw.port_d
    identity = np.eye(u.size, dtype=np.float64)[None, :, :]
    jac_q = identity * raw.port_d[:, :, None] + (u - b)[None, :, None] * raw.port_jac_d
    omega = raw.port_jac_d.transpose(0, 2, 1) - raw.port_jac_d
    return PointQuantities(
        value=np.asarray(raw.value, dtype=np.float64),
        gradient=raw.gradient.copy(),
        port_d=raw.port_d.copy(),
        port_jac_d=raw.port_jac_d.copy(),
        port_q=port_q,
        port_jac_q=jac_q,
        omega=omega,
    )


def integrate_path(
    point_function: Callable[[str, np.ndarray, np.ndarray], PointQuantities],
    fixture_id: str,
    b_array: np.ndarray,
    x_array: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> PathQuantities:
    b = np.asarray(b_array, dtype=np.float64)
    x = np.asarray(x_array, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    delta = x - b
    endpoint_point = point_function(fixture_id, x, b)
    baseline_point = point_function(fixture_id, b, b)
    endpoint = delta * endpoint_point.gradient
    port_count, dimension = endpoint_point.port_d.shape
    ig = np.zeros(dimension, dtype=np.float64)
    a = np.zeros((port_count, dimension), dtype=np.float64)
    t_result = np.zeros((port_count, dimension, dimension), dtype=np.float64)
    r = np.zeros_like(t_result)

    for node, weight in zip(nodes, weights):
        u = b + node * delta
        point = point_function(fixture_id, u, b)
        ig += weight * delta * point.gradient
        a += weight * delta[None, :] * point.port_d
        t_result += weight * point.port_jac_q * delta[None, None, :]
        r += (
            weight
            * (u - b)[None, :, None]
            * delta[None, None, :]
            * point.port_jac_d
        )

    return PathQuantities(
        output_delta=np.asarray(endpoint_point.value - baseline_point.value, dtype=np.float64),
        gradient_x=endpoint_point.gradient,
        endpoint=endpoint,
        port_d_x=endpoint_point.port_d,
        port_q_x=endpoint_point.port_q,
        ig=ig,
        a=a,
        t=t_result,
        r=r,
        omega_x=endpoint_point.omega,
    )


def finite_difference_point(
    fixture_id: str, u_array: np.ndarray, b_array: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central FD of the executable scalar graph and its first-order D field."""
    u = np.asarray(u_array, dtype=np.float64)
    b = np.asarray(b_array, dtype=np.float64)
    gradient = np.zeros_like(u)

    def scalar_and_d(point: np.ndarray) -> tuple[float, np.ndarray]:
        tensor = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        value, uses = _torch_graph(fixture_id, tensor)
        rows = []
        for use in uses:
            cotangent = torch.autograd.grad(value, use, retain_graph=True, create_graph=False)[0]
            row = torch.autograd.grad(
                use, tensor, grad_outputs=cotangent, retain_graph=True, create_graph=False
            )[0]
            rows.append(row.detach().numpy())
        return float(value.detach()), np.stack(rows).astype(np.float64, copy=False)

    _, base_d = scalar_and_d(u)
    jac_d = np.zeros(base_d.shape + (u.size,), dtype=np.float64)
    jac_q = np.zeros_like(jac_d)
    for j in range(u.size):
        step = np.zeros_like(u)
        step[j] = epsilon
        plus_value, plus_d = scalar_and_d(u + step)
        minus_value, minus_d = scalar_and_d(u - step)
        gradient[j] = (plus_value - minus_value) / (2.0 * epsilon)
        jac_d[:, :, j] = (plus_d - minus_d) / (2.0 * epsilon)
        plus_q = (u + step - b)[None, :] * plus_d
        minus_q = (u - step - b)[None, :] * minus_d
        jac_q[:, :, j] = (plus_q - minus_q) / (2.0 * epsilon)
    return gradient, jac_d, jac_q


def check_cut_legality(paths: Sequence[str], ports: Sequence[dict]) -> tuple[bool, dict[str, int]]:
    counts = {path: 0 for path in paths}
    for port in ports:
        for path in port.get("intersects_paths", []):
            if path not in counts:
                return False, counts
            counts[path] += 1
    return all(count == 1 for count in counts.values()), counts


def check_port_identity(port: dict) -> bool:
    identity = port.get("identity", {})
    required = (
        "model_id",
        "graph_revision",
        "block_index",
        "producer_tensor_id",
        "consumer_operator_id",
        "input_slot",
        "invocation_index",
        "slice_spec",
    )
    string_fields = (
        "model_id",
        "graph_revision",
        "producer_tensor_id",
        "consumer_operator_id",
        "slice_spec",
    )
    return (
        all(key in identity for key in required)
        and all(isinstance(identity.get(key), str) and identity[key] for key in string_fields)
        and isinstance(identity.get("block_index"), int)
        and isinstance(identity.get("input_slot"), int)
        and identity["input_slot"] >= 0
        and isinstance(identity.get("invocation_index"), int)
        and identity["invocation_index"] >= 0
    )


def port_registry_failure(ports: Sequence[dict], frozen_ports: Sequence[dict]) -> str | None:
    """Fail closed on ambiguous or changed serialized use-port identities."""
    if not all(check_port_identity(port) for port in ports):
        return FAIL_PORT_ID_AMBIGUOUS
    identities = [port["identity"] for port in ports]
    serialized = [repr(sorted(identity.items())) for identity in identities]
    if len(set(serialized)) != len(serialized):
        return FAIL_PORT_ID_AMBIGUOUS
    expected = [port["identity"] for port in frozen_ports]
    if identities != expected:
        return FAIL_PORT_ID_AMBIGUOUS
    return None


def second_order_graph_failure(port_d: torch.Tensor) -> str | None:
    """Detect an explicitly truncated D graph before any numerical closure test."""
    if not port_d.requires_grad or port_d.grad_fn is None:
        return FAIL_SECOND_ORDER_GRAPH_TRUNCATED
    return None


def rel_l2(observed: np.ndarray, reference: np.ndarray, near_zero: float) -> float:
    numerator = float(np.linalg.norm(np.asarray(observed) - np.asarray(reference)))
    denominator = max(float(np.linalg.norm(reference)), near_zero)
    return numerator / denominator


def comparison(
    observed: np.ndarray,
    reference: np.ndarray,
    atol: float,
    rtol: float,
    near_zero: float,
) -> dict:
    observed = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    finite = bool(np.isfinite(observed).all() and np.isfinite(reference).all())
    absolute = np.abs(observed - reference)
    coordinate_ok = absolute <= atol + rtol * np.abs(reference)
    reference_norm = float(np.linalg.norm(reference))
    relative = rel_l2(observed, reference, near_zero)
    if reference_norm <= near_zero:
        norm_ok = float(absolute.max(initial=0.0)) <= atol
    else:
        norm_ok = relative <= rtol
    return {
        "passed": bool(finite and coordinate_ok.all() and norm_ok),
        "finite": finite,
        "max_abs": float(absolute.max(initial=0.0)),
        "relative_l2": relative,
        "reference_l2": reference_norm,
        "coordinates_checked": int(reference.size),
    }


def identity_checks(path: PathQuantities, tolerance: dict[str, float]) -> dict[str, dict]:
    kwargs = {key: float(tolerance[key]) for key in ("atol", "rtol", "near_zero")}
    diagonal_a = np.zeros_like(path.t)
    indices = np.arange(path.a.shape[1])
    diagonal_a[:, indices, indices] = path.a
    return {
        "port_gradient_closure": comparison(path.port_d_x.sum(axis=0), path.gradient_x, **kwargs),
        "port_endpoint_closure": comparison(path.port_q_x.sum(axis=0), path.endpoint, **kwargs),
        "pcia_to_ig": comparison(path.a.sum(axis=0), path.ig, **kwargs),
        "pcia_to_output_delta": comparison(
            np.asarray(path.a.sum()), path.output_delta, **kwargs
        ),
        "pefa_per_port_to_q": comparison(path.t.sum(axis=2), path.port_q_x, **kwargs),
        "pefa_to_endpoint": comparison(path.t.sum(axis=(0, 2)), path.endpoint, **kwargs),
        "t_equals_delta_a_plus_r": comparison(path.t, diagonal_a + path.r, **kwargs),
        "curvature_cut_zero_sum": comparison(
            path.omega_x.sum(axis=0), np.zeros(path.omega_x.shape[1:], dtype=np.float64), **kwargs
        ),
        "r_to_endpoint_minus_ig": comparison(
            path.r.sum(axis=(0, 2)), path.endpoint - path.ig, **kwargs
        ),
    }


def checks_pass(checks: dict[str, dict]) -> bool:
    return all(row["passed"] for row in checks.values())
