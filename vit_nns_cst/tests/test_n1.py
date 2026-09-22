#!/usr/bin/env python3
"""Unit tests for the NNS-CST N1 standard-library reference implementation."""
from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from nns_cst.core import (  # noqa: E402
    CSTError, FAIL_CUT_MISSING_EDGE, FAIL_NEAR_SINGULAR_CONTROL_COVARIANCE,
    FAIL_NONFINITE_VALUE, FAIL_OFF_MANIFOLD, FAIL_PORT_IDENTITY,
    FAIL_PORT_DUPLICATE, FAIL_RANK_DEFICIENT, FAIL_RANK_EXCEEDS_DIMENSION,
    FAIL_TOKEN_ALIGNMENT, FAIL_ZERO_PAIR_DIFFERENCE, estimate_subspace,
    intervene, orthonormalize, project, projector_distance,
    projector_similarity, semantic_complement_attribution, transport_residual,
    validate_port_registry, validate_token_alignment,
)


class SubspaceTests(unittest.TestCase):
    def assertVectorClose(self, left, right, places=10):
        self.assertEqual(len(left), len(right))
        for actual, expected in zip(left, right):
            self.assertAlmostEqual(actual, expected, places=places)

    def test_basis_rotation_invariance(self):
        root = math.sqrt(0.5)
        first = orthonormalize([[1, 0, 0], [0, 1, 0]])
        second = orthonormalize([[root, root, 0], [-root, root, 0]])
        self.assertVectorClose(project([3, -4, 5], first), project([3, -4, 5], second))
        self.assertAlmostEqual(projector_distance(first, second), 0.0, places=10)
        self.assertAlmostEqual(projector_similarity(first, second), 1.0, places=10)

    def test_brightness_nuisance_is_removed(self):
        result = estimate_subspace([[10, 2, 0], [9, 2, 0], [11, 2, 0]], 1,
                                   control_differences=[[1, 0, 0]])
        self.assertVectorClose(project([1, 0, 0], result["basis"]), [0, 0, 0])
        self.assertAlmostEqual(abs(project([0, 1, 0], result["basis"])[1]), 1.0, places=10)

    def test_complete_cut_coordinate_closure(self):
        result = semantic_complement_attribution(
            [2, -3, 4], [0.5, 2, -1], [[1, 0, 0], [0, 1, 0]])
        self.assertVectorClose(result["reconstructed"], result["complete_cut"])

    def test_residual_transport(self):
        result = transport_residual([[1, 0], [0, 1]], [[0.5, 0], [0, 0.5]], [2, -1])
        self.assertVectorClose(result["identity"], [2, -1])
        self.assertVectorClose(result["branch"], [1, -0.5])
        self.assertVectorClose(result["merged"], [3, -1.5])

    def test_intervention_analytics(self):
        basis = [[1, 0, 0]]
        self.assertVectorClose(intervene([2, 3, 4], basis, "erase"), [0, 3, 4])
        self.assertVectorClose(intervene([2, 3, 4], basis, "inject", [-1, 7, 9]), [-1, 3, 4])

    def test_subspace_typed_failures(self):
        cases = [
            (FAIL_RANK_EXCEEDS_DIMENSION, lambda: estimate_subspace([[1, 0]], 3)),
            (FAIL_RANK_DEFICIENT, lambda: estimate_subspace([[1, 0], [2, 0]], 2)),
            (FAIL_ZERO_PAIR_DIFFERENCE, lambda: estimate_subspace([[0, 0]], 1)),
            (FAIL_NONFINITE_VALUE, lambda: estimate_subspace([[float("inf"), 0]], 1)),
            (FAIL_NEAR_SINGULAR_CONTROL_COVARIANCE,
             lambda: estimate_subspace([[1, 2]], 1, control_differences=[[1, 0], [2, 0]])),
            (FAIL_OFF_MANIFOLD,
             lambda: intervene([1, 0], [[1, 0]], "inject", [100, 0], manifold_limit=0.1)),
        ]
        for expected, function in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(CSTError) as caught:
                    function()
                self.assertEqual(caught.exception.code, expected)


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((HERE / "manifests/n1_r2/model_port_registry.json").read_text(encoding="utf-8"))
        cls.records = cls.registry["records"]
        cls.expected = [row["port_id"] for row in cls.records]

    def test_frozen_registry(self):
        self.assertTrue(validate_port_registry(self.records, self.expected))

    def test_omit_edge(self):
        with self.assertRaises(CSTError) as caught:
            validate_port_registry(self.records[:-1], self.expected)
        self.assertEqual(caught.exception.code, FAIL_CUT_MISSING_EDGE)

    def test_duplicate_edge(self):
        with self.assertRaises(CSTError) as caught:
            validate_port_registry(self.records + [copy.deepcopy(self.records[0])], self.expected)
        self.assertEqual(caught.exception.code, FAIL_PORT_DUPLICATE)

    def test_swap_slot(self):
        rows = copy.deepcopy(self.records)
        index = next(i for i, row in enumerate(rows) if row["tensor_role"] == "attention_head_query_input")
        rows[index]["input_slot"] = 1
        with self.assertRaises(CSTError) as caught:
            validate_port_registry(rows, self.expected)
        self.assertEqual(caught.exception.code, FAIL_PORT_IDENTITY)

    def test_wrong_token_alignment(self):
        with self.assertRaises(CSTError) as caught:
            validate_token_alignment(["CLS", "P0", "P1"], ["CLS", "P0", "P1"],
                                     [("CLS", "CLS"), ("P0", "P1"), ("P1", "P1")])
        self.assertEqual(caught.exception.code, FAIL_TOKEN_ALIGNMENT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
