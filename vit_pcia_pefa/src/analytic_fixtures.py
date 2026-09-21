"""Independent closed-form derivatives for the frozen CPU fixtures.

This module deliberately does not import Torch or the autodiff implementation.
It is the analytic side of the Step 2 three-way comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PORT_ORDER = {
    "LINEAR3": ("linear.input",),
    "BILINEAR2": ("mul.slot0", "mul.slot1", "linear.slot0"),
    "SATURATING3": ("preactivation.tanh.slot0",),
    "FANOUT2": ("h.linear.slot0", "h.tanh.slot0"),
    "RESIDUAL2": ("u.add.slot0", "branch.add.slot1"),
    "MULTIINPUT3": (
        "a.mul.slot0",
        "c.mul.slot1",
        "a.linear.slot0",
        "c.linear.slot0",
    ),
}


@dataclass(frozen=True)
class AnalyticQuantities:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray
    port_d: np.ndarray
    port_jac_d: np.ndarray


@dataclass(frozen=True)
class AnalyticPathQuantities:
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


def _outer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.outer(left, right).astype(np.float64, copy=False)


def evaluate(fixture_id: str, u: np.ndarray) -> AnalyticQuantities:
    """Evaluate F, its first two derivatives, and every use-port D field."""
    u = np.asarray(u, dtype=np.float64)

    if fixture_id == "LINEAR3":
        coefficient = np.array([1.5, -2.0, 0.25], dtype=np.float64)
        zero = np.zeros((3, 3), dtype=np.float64)
        return AnalyticQuantities(
            value=float(coefficient @ u),
            gradient=coefficient.copy(),
            hessian=zero.copy(),
            port_d=coefficient[None, :].copy(),
            port_jac_d=zero[None, :, :].copy(),
        )

    if fixture_id == "BILINEAR2":
        u0, u1 = u
        gradient = np.array([u1 + 0.2, u0], dtype=np.float64)
        hessian = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        port_d = np.array([[u1, 0.0], [0.0, u0], [0.2, 0.0]], dtype=np.float64)
        port_jac_d = np.array(
            [
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        return AnalyticQuantities(
            value=float(u0 * u1 + 0.2 * u0),
            gradient=gradient,
            hessian=hessian,
            port_d=port_d,
            port_jac_d=port_jac_d,
        )

    if fixture_id == "SATURATING3":
        coefficient = np.array([0.7, -1.1, 0.5], dtype=np.float64)
        z = float(coefficient @ u + 0.2)
        y = np.tanh(z)
        first = 1.0 - y * y
        second = -2.0 * y * first
        gradient = first * coefficient
        hessian = second * _outer(coefficient, coefficient)
        return AnalyticQuantities(
            value=float(y),
            gradient=gradient,
            hessian=hessian,
            port_d=gradient[None, :].copy(),
            port_jac_d=hessian[None, :, :].copy(),
        )

    if fixture_id == "FANOUT2":
        coefficient = np.array([0.6, -0.4], dtype=np.float64)
        h = float(coefficient @ u)
        y = np.tanh(h)
        first = 1.0 - y * y
        second = -2.0 * y * first
        linear_d = 2.0 * coefficient
        nonlinear_d = 0.7 * first * coefficient
        nonlinear_jac = 0.7 * second * _outer(coefficient, coefficient)
        return AnalyticQuantities(
            value=float(2.0 * h + 0.7 * y),
            gradient=linear_d + nonlinear_d,
            hessian=nonlinear_jac.copy(),
            port_d=np.stack((linear_d, nonlinear_d)),
            port_jac_d=np.stack((np.zeros((2, 2), dtype=np.float64), nonlinear_jac)),
        )

    if fixture_id == "RESIDUAL2":
        w = np.array([[0.8, -0.3], [0.2, 0.9]], dtype=np.float64)
        q = np.array([0.1, -0.2], dtype=np.float64)
        b_matrix = np.array([[0.7, 0.1], [-0.4, 0.6]], dtype=np.float64)
        v = np.array([1.2, -0.5], dtype=np.float64)
        z = w @ u + q
        y = np.tanh(z)
        first = 1.0 - y * y
        second = -2.0 * y * first
        downstream = b_matrix.T @ v
        branch_d = w.T @ (downstream * first)
        branch_jac = w.T @ np.diag(downstream * second) @ w
        return AnalyticQuantities(
            value=float(v @ (u + b_matrix @ y)),
            gradient=v + branch_d,
            hessian=branch_jac.copy(),
            port_d=np.stack((v, branch_d)),
            port_jac_d=np.stack((np.zeros((2, 2), dtype=np.float64), branch_jac)),
        )

    if fixture_id == "MULTIINPUT3":
        grad_a = np.array([1.0, 2.0, 0.0], dtype=np.float64)
        grad_c = np.array([0.0, 0.5, -1.0], dtype=np.float64)
        a = float(grad_a @ u)
        c = float(grad_c @ u)
        a_mul_d = c * grad_a
        c_mul_d = a * grad_c
        a_linear_d = 0.3 * grad_a
        c_linear_d = -0.2 * grad_c
        a_mul_jac = _outer(grad_a, grad_c)
        c_mul_jac = _outer(grad_c, grad_a)
        zero = np.zeros((3, 3), dtype=np.float64)
        hessian = a_mul_jac + c_mul_jac
        return AnalyticQuantities(
            value=float(a * c + 0.3 * a - 0.2 * c),
            gradient=(c + 0.3) * grad_a + (a - 0.2) * grad_c,
            hessian=hessian,
            port_d=np.stack((a_mul_d, c_mul_d, a_linear_d, c_linear_d)),
            port_jac_d=np.stack((a_mul_jac, c_mul_jac, zero, zero)),
        )

    raise KeyError(f"unknown analytic fixture: {fixture_id}")


def _ridge_integrals(z0: float, z1: float) -> tuple[float, float]:
    """Exact integrals of tanh' and t*tanh'' along a scalar affine path."""
    delta_z = z1 - z0
    y0, y1 = np.tanh(z0), np.tanh(z1)
    if abs(delta_z) < 1e-14:
        first = 1.0 - y0 * y0
        second = -2.0 * y0 * first
        return float(first), float(0.5 * second)
    integral_first = (y1 - y0) / delta_z
    first_at_one = 1.0 - y1 * y1
    integral_t_second = first_at_one / delta_z - integral_first / delta_z
    return float(integral_first), float(integral_t_second)


def _ridge_port(
    delta: np.ndarray, coefficient: np.ndarray, z0: float, z1: float, downstream: float
) -> tuple[np.ndarray, np.ndarray]:
    integral_first, integral_t_second = _ridge_integrals(z0, z1)
    a = delta * coefficient * downstream * integral_first
    r = (
        delta[:, None]
        * delta[None, :]
        * coefficient[:, None]
        * coefficient[None, :]
        * downstream
        * integral_t_second
    )
    return a, r


def closed_form_path(fixture_id: str, b: np.ndarray, x: np.ndarray) -> AnalyticPathQuantities:
    """Closed-form straight-path quantities, independent of quadrature/autodiff."""
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    delta = x - b
    start = evaluate(fixture_id, b)
    end = evaluate(fixture_id, x)
    port_count, dimension = end.port_d.shape
    a = np.zeros((port_count, dimension), dtype=np.float64)
    r = np.zeros((port_count, dimension, dimension), dtype=np.float64)

    if fixture_id == "LINEAR3":
        a[0] = delta * end.port_d[0]

    elif fixture_id == "BILINEAR2":
        a[0, 0] = delta[0] * delta[1] / 2.0
        a[1, 1] = delta[1] * delta[0] / 2.0
        a[2, 0] = 0.2 * delta[0]
        r[0, 0, 1] = delta[0] * delta[1] / 2.0
        r[1, 1, 0] = delta[1] * delta[0] / 2.0

    elif fixture_id == "SATURATING3":
        coefficient = np.array([0.7, -1.1, 0.5], dtype=np.float64)
        z0 = float(coefficient @ b + 0.2)
        z1 = float(coefficient @ x + 0.2)
        a[0], r[0] = _ridge_port(delta, coefficient, z0, z1, 1.0)

    elif fixture_id == "FANOUT2":
        coefficient = np.array([0.6, -0.4], dtype=np.float64)
        a[0] = delta * (2.0 * coefficient)
        z0, z1 = float(coefficient @ b), float(coefficient @ x)
        a[1], r[1] = _ridge_port(delta, coefficient, z0, z1, 0.7)

    elif fixture_id == "RESIDUAL2":
        w = np.array([[0.8, -0.3], [0.2, 0.9]], dtype=np.float64)
        q = np.array([0.1, -0.2], dtype=np.float64)
        b_matrix = np.array([[0.7, 0.1], [-0.4, 0.6]], dtype=np.float64)
        v = np.array([1.2, -0.5], dtype=np.float64)
        downstream = b_matrix.T @ v
        a[0] = delta * v
        for unit in range(2):
            z0 = float(w[unit] @ b + q[unit])
            z1 = float(w[unit] @ x + q[unit])
            unit_a, unit_r = _ridge_port(delta, w[unit], z0, z1, float(downstream[unit]))
            a[1] += unit_a
            r[1] += unit_r

    elif fixture_id == "MULTIINPUT3":
        grad_a = np.array([1.0, 2.0, 0.0], dtype=np.float64)
        grad_c = np.array([0.0, 0.5, -1.0], dtype=np.float64)
        a0, c0 = float(grad_a @ b), float(grad_c @ b)
        da, dc = float(grad_a @ delta), float(grad_c @ delta)
        mean_a, mean_c = a0 + da / 2.0, c0 + dc / 2.0
        a[0] = delta * grad_a * mean_c
        a[1] = delta * grad_c * mean_a
        a[2] = delta * 0.3 * grad_a
        a[3] = delta * -0.2 * grad_c
        r[0] = delta[:, None] * delta[None, :] * _outer(grad_a, grad_c) / 2.0
        r[1] = delta[:, None] * delta[None, :] * _outer(grad_c, grad_a) / 2.0

    else:
        raise KeyError(f"unknown closed-form fixture: {fixture_id}")

    t_result = r.copy()
    indices = np.arange(dimension)
    t_result[:, indices, indices] += a
    endpoint = delta * end.gradient
    port_q = delta[None, :] * end.port_d
    omega = end.port_jac_d.transpose(0, 2, 1) - end.port_jac_d
    return AnalyticPathQuantities(
        output_delta=np.asarray(end.value - start.value, dtype=np.float64),
        gradient_x=end.gradient.copy(),
        endpoint=endpoint,
        port_d_x=end.port_d.copy(),
        port_q_x=port_q,
        ig=a.sum(axis=0),
        a=a,
        t=t_result,
        r=r,
        omega_x=omega,
    )
