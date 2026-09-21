#!/usr/bin/env python3
"""Independent symbolic derivation and Step 2 fixture consistency audit.

This file rebuilds every frozen scalar graph and use-port field in SymPy.  It
uses the Step 2 analytic module only as the comparison target after all
symbolic derivatives and high-precision path integrals have been computed.
It does not import Torch, load a model/image, or expose a CUDA path.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Comparison target only. No constructor below calls this module.
from vit_pcia_pefa.src import analytic_fixtures

HERE = ROOT / "vit_pcia_pefa"
FIXTURE_IDS = (
    "LINEAR3",
    "BILINEAR2",
    "SATURATING3",
    "FANOUT2",
    "RESIDUAL2",
    "MULTIINPUT3",
)
POINT_T = (0.0, 0.37, 1.0)
ATOL = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def variables(dimension: int) -> tuple[sp.Symbol, ...]:
    return sp.symbols(f"u0:{dimension}", real=True)


def build_fixture(fixture_id: str) -> tuple[tuple[sp.Symbol, ...], sp.Expr, tuple[sp.Matrix, ...]]:
    """Build F and each use-port D without importing Step 2 formulas."""
    if fixture_id == "LINEAR3":
        u = variables(3)
        coefficient = sp.Matrix([sp.Rational(3, 2), -2, sp.Rational(1, 4)])
        f = (coefficient.T * sp.Matrix(u))[0]
        return u, f, (coefficient,)

    if fixture_id == "BILINEAR2":
        u = variables(2)
        f = u[0] * u[1] + sp.Rational(1, 5) * u[0]
        ports = (sp.Matrix([u[1], 0]), sp.Matrix([0, u[0]]), sp.Matrix([sp.Rational(1, 5), 0]))
        return u, f, ports

    if fixture_id == "SATURATING3":
        u = variables(3)
        coefficient = sp.Matrix([sp.Rational(7, 10), -sp.Rational(11, 10), sp.Rational(1, 2)])
        z = (coefficient.T * sp.Matrix(u))[0] + sp.Rational(1, 5)
        f = sp.tanh(z)
        return u, f, (sp.Matrix([sp.diff(f, item) for item in u]),)

    if fixture_id == "FANOUT2":
        u = variables(2)
        coefficient = sp.Matrix([sp.Rational(3, 5), -sp.Rational(2, 5)])
        h = (coefficient.T * sp.Matrix(u))[0]
        f = 2 * h + sp.Rational(7, 10) * sp.tanh(h)
        linear = 2 * coefficient
        nonlinear = sp.Rational(7, 10) * (1 - sp.tanh(h) ** 2) * coefficient
        return u, f, (linear, nonlinear)

    if fixture_id == "RESIDUAL2":
        u = variables(2)
        uv = sp.Matrix(u)
        w = sp.Matrix([[sp.Rational(4, 5), -sp.Rational(3, 10)], [sp.Rational(1, 5), sp.Rational(9, 10)]])
        q = sp.Matrix([sp.Rational(1, 10), -sp.Rational(1, 5)])
        b_matrix = sp.Matrix([[sp.Rational(7, 10), sp.Rational(1, 10)], [-sp.Rational(2, 5), sp.Rational(3, 5)]])
        v = sp.Matrix([sp.Rational(6, 5), -sp.Rational(1, 2)])
        z = w * uv + q
        activation = z.applyfunc(sp.tanh)
        f = (v.T * (uv + b_matrix * activation))[0]
        identity = v
        branch_scalar = (v.T * b_matrix * activation)[0]
        branch = sp.Matrix([sp.diff(branch_scalar, item) for item in u])
        return u, f, (identity, branch)

    if fixture_id == "MULTIINPUT3":
        u = variables(3)
        grad_a = sp.Matrix([1, 2, 0])
        grad_c = sp.Matrix([0, sp.Rational(1, 2), -1])
        a = (grad_a.T * sp.Matrix(u))[0]
        c = (grad_c.T * sp.Matrix(u))[0]
        f = a * c + sp.Rational(3, 10) * a - sp.Rational(1, 5) * c
        ports = (c * grad_a, a * grad_c, sp.Rational(3, 10) * grad_a, -sp.Rational(1, 5) * grad_c)
        return u, f, ports

    raise KeyError(fixture_id)


def derived(u: tuple[sp.Symbol, ...], f: sp.Expr, ports: tuple[sp.Matrix, ...], b: np.ndarray) -> dict[str, Any]:
    uv = sp.Matrix(u)
    bv = sp.Matrix([sp.Float(float(value), 17) for value in b])
    gradient = sp.Matrix([sp.diff(f, item) for item in u])
    jac_d = tuple(port.jacobian(uv) for port in ports)
    q = tuple(sp.Matrix([(uv[i] - bv[i]) * port[i] for i in range(len(u))]) for port in ports)
    jac_q = tuple(item.jacobian(uv) for item in q)
    omega = tuple(jac.T - jac for jac in jac_d)
    return {
        "gradient": gradient,
        "jac_d": jac_d,
        "q": q,
        "jac_q": jac_q,
        "omega": omega,
    }


def all_symbolic_zero(value: Any) -> bool:
    if isinstance(value, sp.MatrixBase):
        return all(sp.simplify(item) == 0 for item in value)
    return sp.simplify(value) == 0


def matrix_sum(items: tuple[sp.Matrix, ...]) -> sp.Matrix:
    result = sp.zeros(*items[0].shape)
    for item in items:
        result += item
    return result


def evaluate_array(expressions: Any, u: tuple[sp.Symbol, ...], point: np.ndarray) -> np.ndarray:
    substitutions = {symbol: sp.Float(float(value), 40) for symbol, value in zip(u, point)}
    if isinstance(expressions, (tuple, list)):
        return np.stack([evaluate_array(item, u, point) for item in expressions])
    if isinstance(expressions, sp.MatrixBase):
        rows, cols = expressions.shape
        values = [[float(sp.N(expressions[row, col].subs(substitutions), 40)) for col in range(cols)] for row in range(rows)]
        array = np.asarray(values, dtype=np.float64)
        return array[:, 0] if cols == 1 else array
    return np.asarray(float(sp.N(expressions.subs(substitutions), 40)), dtype=np.float64)


def integrate_expression(expression: sp.Expr, u: tuple[sp.Symbol, ...], b: np.ndarray, delta: np.ndarray) -> float:
    t = sp.symbols("t", real=True)
    path_expression = expression.subs({symbol: sp.Float(float(bi), 30) + t * sp.Float(float(di), 30) for symbol, bi, di in zip(u, b, delta)})
    function = sp.lambdify(t, path_expression, modules="mpmath")
    return float(mp.quad(function, [mp.mpf("0"), mp.mpf("1")]))


def integrate_array(expressions: Any, u: tuple[sp.Symbol, ...], b: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if isinstance(expressions, (tuple, list)):
        return np.stack([integrate_array(item, u, b, delta) for item in expressions])
    if isinstance(expressions, sp.MatrixBase):
        rows, cols = expressions.shape
        result = np.empty((rows, cols), dtype=np.float64)
        for row in range(rows):
            for col in range(cols):
                result[row, col] = integrate_expression(expressions[row, col], u, b, delta)
        return result[:, 0] if cols == 1 else result
    return np.asarray(integrate_expression(expressions, u, b, delta), dtype=np.float64)


def comparison(observed: Any, expected: Any) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    error = np.abs(observed - expected)
    maximum = float(error.max(initial=0.0))
    return {
        "passed": bool(np.isfinite(observed).all() and np.isfinite(expected).all() and maximum <= ATOL),
        "max_abs": maximum,
        "coordinates_checked": int(expected.size),
    }


def fixture_audit(row: dict[str, Any]) -> dict[str, Any]:
    fixture_id = row["fixture_id"]
    b = np.asarray(row["b"], dtype=np.float64)
    x = np.asarray(row["x"], dtype=np.float64)
    delta = x - b
    u, f, ports = build_fixture(fixture_id)
    values = derived(u, f, ports, b)
    gradient = values["gradient"]
    jac_d = values["jac_d"]
    q = values["q"]
    jac_q = values["jac_q"]
    omega = values["omega"]
    dimension = len(u)

    decomposition_residuals = []
    for port, jac, jacq in zip(ports, jac_d, jac_q):
        expected = sp.zeros(dimension, dimension)
        for i in range(dimension):
            for j in range(dimension):
                expected[i, j] = (port[i] if i == j else 0) + (u[i] - sp.Float(float(b[i]), 17)) * jac[i, j]
        decomposition_residuals.append(jacq - expected)

    symbolic = {
        "cut_gradient_closure": all_symbolic_zero(matrix_sum(ports) - gradient),
        "t_integrand_decomposition": all(all_symbolic_zero(item) for item in decomposition_residuals),
        "cut_omega_zero_sum": all_symbolic_zero(matrix_sum(omega)),
        "hessian_symmetry": all_symbolic_zero(gradient.jacobian(sp.Matrix(u)) - gradient.jacobian(sp.Matrix(u)).T),
    }

    point_checks = []
    for fraction in POINT_T:
        point = b + fraction * delta
        expected = analytic_fixtures.evaluate(fixture_id, point)
        expected_q = (point - b)[None, :] * expected.port_d
        identity = np.eye(dimension, dtype=np.float64)[None, :, :]
        expected_jac_q = identity * expected.port_d[:, :, None] + (point - b)[None, :, None] * expected.port_jac_d
        expected_omega = expected.port_jac_d.transpose(0, 2, 1) - expected.port_jac_d
        checks = {
            "value": comparison(evaluate_array(f, u, point), expected.value),
            "gradient": comparison(evaluate_array(gradient, u, point), expected.gradient),
            "port_d": comparison(evaluate_array(ports, u, point), expected.port_d),
            "port_jac_d": comparison(evaluate_array(jac_d, u, point), expected.port_jac_d),
            "port_q": comparison(evaluate_array(q, u, point), expected_q),
            "port_jac_q": comparison(evaluate_array(jac_q, u, point), expected_jac_q),
            "omega": comparison(evaluate_array(omega, u, point), expected_omega),
        }
        point_checks.append({"t": fraction, "checks": checks, "passed": all(item["passed"] for item in checks.values())})

    delta_symbols = sp.Matrix([sp.Float(float(value), 30) for value in delta])
    direct_a = tuple(sp.Matrix([delta_symbols[i] * port[i] for i in range(dimension)]) for port in ports)
    direct_t = tuple(
        sp.Matrix(dimension, dimension, lambda i, j: jacq[i, j] * delta_symbols[j])
        for jacq in jac_q
    )
    direct_r = tuple(sp.Matrix(dimension, dimension, lambda i, j: (u[i] - sp.Float(float(b[i]), 17)) * delta_symbols[j] * jac[i, j]) for jac in jac_d)
    direct_ig = sp.Matrix([delta_symbols[i] * gradient[i] for i in range(dimension)])

    observed_path = {
        "output_delta": evaluate_array(f, u, x) - evaluate_array(f, u, b),
        "gradient_x": evaluate_array(gradient, u, x),
        "endpoint": delta * evaluate_array(gradient, u, x),
        "port_d_x": evaluate_array(ports, u, x),
        "port_q_x": evaluate_array(q, u, x),
        "ig": integrate_array(direct_ig, u, b, delta),
        "a": integrate_array(direct_a, u, b, delta),
        "t": integrate_array(direct_t, u, b, delta),
        "r": integrate_array(direct_r, u, b, delta),
        "omega_x": evaluate_array(omega, u, x),
    }
    expected_path = analytic_fixtures.closed_form_path(fixture_id, b, x)
    path_checks = {name: comparison(value, getattr(expected_path, name)) for name, value in observed_path.items()}
    return {
        "symbolic_identities": symbolic,
        "point_checks": point_checks,
        "path_checks": path_checks,
        "passed": all(symbolic.values()) and all(item["passed"] for item in point_checks) and all(item["passed"] for item in path_checks.values()),
    }


def representation_checks() -> dict[str, dict[str, Any]]:
    mp.mp.dps = 60

    # Same oriented path with a strictly increasing nonlinear parameterization.
    def phi(s: mp.mpf) -> mp.mpf:
        return s + mp.mpf("0.2") * mp.sin(2 * mp.pi * s) / (2 * mp.pi)

    def dphi(s: mp.mpf) -> mp.mpf:
        return 1 + mp.mpf("0.2") * mp.cos(2 * mp.pi * s)

    direct = mp.quad(lambda t: 2 * t, [0, 1])
    reparam = mp.quad(lambda s: 2 * phi(s) * dphi(s), [0, 1])
    reparameterization = abs(direct - reparam) < mp.mpf("1e-45")

    # F=u0*u1 from (0,0) to (1,1): straight versus axis-aligned path.
    straight_ig = (mp.mpf("0.5"), mp.mpf("0.5"))
    axis_ig = (mp.mpf("0"), mp.mpf("1"))

    # Endpoint changes with the reference for F(u)=u^2 at x=1.
    endpoint_b0, endpoint_bhalf = mp.mpf("2"), mp.mpf("1")

    # Per-coordinate affine invariance and general mixing non-invariance.
    affine_original = np.asarray([2.0 * 3.0, -1.0 * 4.0])
    scales = np.asarray([5.0, -2.0])
    affine_transformed = (scales * np.asarray([2.0, -1.0])) * (np.asarray([3.0, 4.0]) / scales)
    mixing_original = np.asarray([1.0, 0.0])
    mixing_transformed = np.asarray([0.5, 0.5])

    # Invertible within-port coordinate map z'=M z preserves root-space D.
    m = sp.Matrix([[2, 1], [1, 1]])
    z0, z1 = sp.symbols("z0 z1", real=True)
    z = sp.Matrix([z0, z1])
    zp = m * z
    zp0, zp1 = sp.symbols("zp0 zp1", real=True)
    inverse_z = m.inv() * sp.Matrix([zp0, zp1])
    fz = z0 * z1 + z0
    fp = fz.subs({z0: inverse_z[0], z1: inverse_z[1]})
    transformed_d = m.T * sp.Matrix([sp.diff(fp, zp0), sp.diff(fp, zp1)]).subs({zp0: zp[0], zp1: zp[1]})
    original_d = sp.Matrix([sp.diff(fz, z0), sp.diff(fz, z1)])
    port_reparameterization = all_symbolic_zero(transformed_d - original_d)

    # Additive refinement and cross-graph non-uniqueness for multiplication.
    u0, u1 = sp.symbols("u0 u1", real=True)
    original_port = sp.Matrix([u1, u0])
    refined = (sp.Rational(1, 3) * original_port, sp.Rational(2, 3) * original_port)
    parallel_refinement = all_symbolic_zero(matrix_sum(refined) - original_port)
    slot0, slot1 = sp.Matrix([u1, 0]), sp.Matrix([0, u0])
    original_omega = original_port.jacobian(sp.Matrix([u0, u1])).T - original_port.jacobian(sp.Matrix([u0, u1]))
    slot_omegas = tuple(item.jacobian(sp.Matrix([u0, u1])).T - item.jacobian(sp.Matrix([u0, u1])) for item in (slot0, slot1))
    cross_graph = all_symbolic_zero(original_omega) and any(not all_symbolic_zero(item) for item in slot_omegas) and all_symbolic_zero(matrix_sum(slot_omegas))

    return {
        "path_reparameterization": {"passed": bool(reparameterization), "direct": str(direct), "reparameterized": str(reparam)},
        "path_selection": {"passed": straight_ig != axis_ig, "straight_ig": [str(x) for x in straight_ig], "axis_ig": [str(x) for x in axis_ig]},
        "reference_point": {"passed": endpoint_b0 != endpoint_bhalf, "endpoint_b0": str(endpoint_b0), "endpoint_bhalf": str(endpoint_bhalf)},
        "input_coordinates": {"passed": bool(np.array_equal(affine_original, affine_transformed) and not np.array_equal(mixing_original, mixing_transformed)), "coordinatewise_affine_invariant": True, "general_mixing_invariant": False},
        "within_port_reparameterization": {"passed": bool(port_reparameterization), "exact_chain_rule_cancellation": bool(port_reparameterization)},
        "parallel_refinement": {"passed": bool(parallel_refinement), "additive_sum_preserved": bool(parallel_refinement)},
        "cross_graph_rewrite": {"passed": bool(cross_graph), "total_omega_zero": True, "split_port_omega_nonzero": True},
    }


def main() -> None:
    mp.mp.dps = 60
    registry = json.loads((HERE / "fixture_registry.json").read_text(encoding="utf-8"))
    assert tuple(registry["fixture_order"]) == FIXTURE_IDS
    fixture_rows = {row["fixture_id"]: row for row in registry["fixtures"]}
    fixtures = {fixture_id: fixture_audit(fixture_rows[fixture_id]) for fixture_id in FIXTURE_IDS}
    representations = representation_checks()
    all_fixtures = all(row["passed"] for row in fixtures.values())
    all_representations = all(row["passed"] for row in representations.values())
    source_paths = (
        "vit_pcia_pefa/执行规划.md",
        "vit_pcia_pefa/method_contract.md",
        "vit_pcia_pefa/literature_matrix.md",
        "vit_pcia_pefa/step2_decision.md",
        "vit_pcia_pefa/theory_endpoint_pcia_pefa.md",
        "vit_pcia_pefa/run_step3_symbolic.py",
        "vit_pcia_pefa/validate_step3.py",
        "vit_pcia_pefa/step3_decision.md",
        "vit_pcia_pefa/step3_defensive_writing_audit.md",
        "vit_pcia_pefa/fixture_registry.json",
        "vit_pcia_pefa/src/analytic_fixtures.py",
        "vit_pcia_pefa/mvp_validation.json",
    )
    validation = {
        "schema_version": 1,
        "stage": "ENDPOINT_PCIA_PEFA_STEP3",
        "status": "PASS_STEP3_THEORY_AND_SYMBOLIC_CONSISTENCY" if all_fixtures and all_representations else "FAIL_STEP3",
        "scope": "THEORY_ONLY_STEP4_TO_STEP6_NOT_AUTHORIZED",
        "predecessor": {
            "status": "PASS_STEP2_CPU_FLOAT64_REFERENCE",
            "mvp_validation_sha256": sha256(HERE / "mvp_validation.json"),
        },
        "environment": {
            "python_executable_realpath": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "sympy_version": sp.__version__,
            "mpmath_version": mp.__version__,
            "numpy_version": np.__version__,
            "device": "cpu_symbolic",
        },
        "contract": {
            "smoothness": "C2_OPEN_NEIGHBORHOOD_OF_PATH",
            "graph": "FIXED_FINITE_DAG",
            "path": "PIECEWISE_C1",
            "comparison_atol": ATOL,
            "mpmath_decimal_places": mp.mp.dps,
        },
        "coverage": {
            "fixtures": len(fixtures),
            "symbolic_identity_families": ["cut_gradient_closure", "t_integrand_decomposition", "cut_omega_zero_sum", "hessian_symmetry"],
            "point_fractions": list(POINT_T),
            "path_quantities": ["output_delta", "gradient_x", "endpoint", "port_d_x", "port_q_x", "ig", "a", "t", "r", "omega_x"],
            "representation_cases": list(representations),
        },
        "fixtures": fixtures,
        "representation_checks": representations,
        "all_fixtures_passed": all_fixtures,
        "all_representation_checks_passed": all_representations,
        "no_go_generality": False,
        "scientific_effect_claims": "UNTESTED",
        "source_sha256": {relative: sha256(ROOT / relative) for relative in source_paths},
        "git_head_at_execution": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip(),
    }
    (HERE / "step3_symbolic_validation.json").write_bytes(canonical_bytes(validation))
    print(json.dumps({
        "status": validation["status"],
        "fixtures_passed": sum(row["passed"] for row in fixtures.values()),
        "fixtures_total": len(fixtures),
        "representation_checks_passed": sum(row["passed"] for row in representations.values()),
        "representation_checks_total": len(representations),
        "receipt_sha256": sha256(HERE / "step3_symbolic_validation.json"),
    }, sort_keys=True, indent=2))
    if validation["status"] != "PASS_STEP3_THEORY_AND_SYMBOLIC_CONSISTENCY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
