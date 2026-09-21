from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vit_pcia_pefa.src import analytic_fixtures
from vit_pcia_pefa.src.reference import (
    FAIL_CUT_NOT_LEGAL,
    FAIL_PORT_ID_AMBIGUOUS,
    FAIL_QUADRATURE_CLOSURE,
    FAIL_SECOND_ORDER_GRAPH_TRUNCATED,
    _torch_graph,
    analytic_point,
    autodiff_point,
    check_cut_legality,
    checks_pass,
    comparison,
    finite_difference_point,
    identity_checks,
    integrate_path,
    port_registry_failure,
    second_order_graph_failure,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((ROOT / "fixture_registry.json").read_text())
QUADRATURE = json.loads((ROOT / "quadrature_registry.json").read_text())
FIXTURE_ROWS = {row["fixture_id"]: row for row in FIXTURES["fixtures"]}
SCHEDULES = {row["count"]: row for row in QUADRATURE["schedules"]}
TOLERANCE = FIXTURES["tolerance"]
FIXTURE_IDS = FIXTURES["fixture_order"]


def schedule(count: int) -> tuple[np.ndarray, np.ndarray]:
    row = SCHEDULES[count]
    return np.asarray(row["nodes"]), np.asarray(row["weights"])


def fixture_arrays(fixture_id: str) -> tuple[np.ndarray, np.ndarray]:
    row = FIXTURE_ROWS[fixture_id]
    return np.asarray(row["b"]), np.asarray(row["x"])


def assert_comparison(observed: np.ndarray, reference: np.ndarray) -> None:
    result = comparison(observed, reference, **TOLERANCE)
    assert result["passed"], result


def test_registry_exact_fixture_order_and_cpu_float64_contract() -> None:
    assert FIXTURE_IDS == [
        "LINEAR3", "BILINEAR2", "SATURATING3",
        "FANOUT2", "RESIDUAL2", "MULTIINPUT3",
    ]
    assert FIXTURES["dtype"] == "float64"
    assert FIXTURES["path"] == "gamma(t)=b+t*(x-b)"
    assert TOLERANCE == {"atol": 1e-9, "near_zero": 1e-12, "rtol": 1e-7}
    assert torch.get_default_dtype() == torch.float64
    assert not torch.cuda.is_available()


@pytest.mark.parametrize("count", [64, 128])
def test_quadrature_registry_reconstructs_frozen_float64_bytes(count: int) -> None:
    row = SCHEDULES[count]
    source_nodes, source_weights = np.polynomial.legendre.leggauss(count)
    source_nodes = ((source_nodes + 1.0) / 2.0).astype("<f8")
    source_weights = (source_weights / 2.0).astype("<f8")
    nodes, weights = schedule(count)
    assert np.array_equal(nodes, source_nodes)
    assert np.array_equal(weights, source_weights)
    digest = hashlib.sha256(source_nodes.tobytes() + source_weights.tobytes()).hexdigest()
    assert digest == row["nodes_then_weights_float64_sha256"]
    assert np.all(np.diff(nodes) > 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
@pytest.mark.parametrize("t", [0.0, 0.37, 1.0])
def test_analytic_autodiff_and_finite_difference_agree(fixture_id: str, t: float) -> None:
    b, x = fixture_arrays(fixture_id)
    u = b + t * (x - b)
    analytic = analytic_point(fixture_id, u, b)
    autodiff = autodiff_point(fixture_id, u, b)
    for field in ("value", "gradient", "port_d", "port_jac_d", "port_q", "port_jac_q", "omega"):
        assert_comparison(getattr(autodiff, field), getattr(analytic, field))
    finite_gradient, finite_jac_d, finite_jac_q = finite_difference_point(
        fixture_id, u, b, FIXTURES["finite_difference"]["epsilon"]
    )
    assert_comparison(finite_gradient, analytic.gradient)
    assert_comparison(finite_jac_d, analytic.port_jac_d)
    assert_comparison(finite_gradient, autodiff.gradient)
    assert_comparison(finite_jac_d, autodiff.port_jac_d)
    assert_comparison(finite_jac_q, analytic.port_jac_q)
    assert_comparison(finite_jac_q, autodiff.port_jac_q)


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
@pytest.mark.parametrize("count", [64, 128])
def test_all_raw_coordinate_identities_and_closed_forms(fixture_id: str, count: int) -> None:
    b, x = fixture_arrays(fixture_id)
    nodes, weights = schedule(count)
    observed = integrate_path(autodiff_point, fixture_id, b, x, nodes, weights)
    closed = analytic_fixtures.closed_form_path(fixture_id, b, x)
    assert checks_pass(identity_checks(observed, TOLERANCE))
    for field in ("output_delta", "gradient_x", "endpoint", "port_d_x", "port_q_x",
                  "ig", "a", "t", "r", "omega_x"):
        assert_comparison(getattr(observed, field), getattr(closed, field))


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_64_node_gate_converges_to_fixed_128_node_diagnostic(fixture_id: str) -> None:
    b, x = fixture_arrays(fixture_id)
    p64 = integrate_path(autodiff_point, fixture_id, b, x, *schedule(64))
    p128 = integrate_path(autodiff_point, fixture_id, b, x, *schedule(128))
    for field in ("ig", "a", "t", "r"):
        assert_comparison(getattr(p64, field), getattr(p128, field))


def test_bilinear_fixture_has_frozen_off_diagonal_formation() -> None:
    b, x = fixture_arrays("BILINEAR2")
    path = integrate_path(autodiff_point, "BILINEAR2", b, x, *schedule(64))
    assert_comparison(path.endpoint - path.ig, np.asarray([-3.0, -3.0]))
    assert path.r[0, 0, 1] == pytest.approx(-3.0, abs=1e-14)
    assert path.r[1, 1, 0] == pytest.approx(-3.0, abs=1e-14)


def test_linear_fixture_has_exact_zero_curvature_transport() -> None:
    b, x = fixture_arrays("LINEAR3")
    path = integrate_path(autodiff_point, "LINEAR3", b, x, *schedule(64))
    assert np.count_nonzero(path.r) == 0
    assert np.count_nonzero(path.omega_x) == 0


def test_cut_enumerator_detects_omitted_duplicated_and_serial_edges() -> None:
    residual = FIXTURE_ROWS["RESIDUAL2"]
    legal, omitted = check_cut_legality(residual["paths"], [residual["ports"][1]])
    assert not legal and sorted(omitted.values()) == [0, 0, 1, 1]
    duplicate_ports = residual["ports"] + [residual["ports"][0]]
    legal, duplicated = check_cut_legality(residual["paths"], duplicate_ports)
    assert not legal and sorted(duplicated.values()) == [1, 1, 2, 2]
    linear = FIXTURE_ROWS["LINEAR3"]
    legal, serial = check_cut_legality(linear["paths"], linear["ports"] * 2)
    assert not legal and set(serial.values()) == {2}
    assert FAIL_CUT_NOT_LEGAL == "FAIL_CUT_NOT_LEGAL"


def test_fanout_merge_and_input_slot_swap_fail_port_identity() -> None:
    fanout = FIXTURE_ROWS["FANOUT2"]
    merged = copy.deepcopy(fanout["ports"][0])
    merged["identity"]["consumer_operator_id"] = "MERGED"
    merged["intersects_paths"] = fanout["paths"]
    assert port_registry_failure([merged], fanout["ports"]) == FAIL_PORT_ID_AMBIGUOUS
    multi = FIXTURE_ROWS["MULTIINPUT3"]
    swapped = copy.deepcopy(multi["ports"])
    swapped[0]["identity"]["input_slot"] = 1
    swapped[1]["identity"]["input_slot"] = 0
    assert port_registry_failure(swapped, multi["ports"]) == FAIL_PORT_ID_AMBIGUOUS


def test_detach_is_detected_before_second_derivative() -> None:
    u = torch.tensor([0.31, -0.47, 0.22], requires_grad=True)
    value, uses = _torch_graph("SATURATING3", u)
    cotangent = torch.autograd.grad(value, uses[0], create_graph=True, retain_graph=True)[0]
    d = torch.autograd.grad(uses[0], u, grad_outputs=cotangent, create_graph=True)[0]
    assert second_order_graph_failure(d) is None
    assert second_order_graph_failure(d.detach()) == FAIL_SECOND_ORDER_GRAPH_TRUNCATED


def test_hessian_diagonal_only_fails_t_decomposition() -> None:
    b, x = fixture_arrays("BILINEAR2")
    path = integrate_path(autodiff_point, "BILINEAR2", b, x, *schedule(64))
    diagonal = np.arange(path.a.shape[1])
    defective_r = np.zeros_like(path.r)
    defective_r[:, diagonal, diagonal] = path.r[:, diagonal, diagonal]
    defective = defective_r.copy()
    defective[:, diagonal, diagonal] += path.a
    assert not comparison(defective, path.t, **TOLERANCE)["passed"]
    assert FAIL_QUADRATURE_CLOSURE == "FAIL_QUADRATURE_CLOSURE"


def test_nonsmooth_node_policy_distinguishes_safe_grid_and_exact_hit() -> None:
    nodes, weights = schedule(64)
    path_u = -1.0 + 2.0 * nodes
    assert not np.any(path_u == 0.0)
    assert np.min(np.abs(path_u)) == pytest.approx(0.024350292663424478, abs=0.0)
    almost_everywhere_t = np.sum(weights * 2.0 * (path_u > 0.0))
    assert almost_everywhere_t == pytest.approx(1.0, abs=1e-14)
    assert not comparison(np.asarray([almost_everywhere_t]), np.asarray([2.0]), **TOLERANCE)["passed"]
    kink = -1.0 + 2.0 * nodes[31]
    assert (-1.0 + 2.0 * nodes[31]) == kink
    assert nodes[31] == pytest.approx(0.48782485366828776, abs=0.0)
