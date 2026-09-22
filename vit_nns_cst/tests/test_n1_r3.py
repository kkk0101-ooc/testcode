#!/usr/bin/env python3
"""N1-R3 closed frozen-readout tests."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from nns_cst.core import readout_score  # noqa: E402
from nns_cst_r3.readout import fit_frozen_readout  # noqa: E402


class FrozenReadoutTests(unittest.TestCase):
    def setUp(self):
        self.contract_sha = hashlib.sha256((HERE / "N1数学合同.md").read_bytes()).hexdigest()
        self.bundle = fit_frozen_readout(
            [[0, 2, 0], [1, 3, 0]], [[0, -2, 0], [-1, -3, 0]],
            [[0, 1, 0]], "engineering_fixture", self.contract_sha)

    def test_closed_field_set_and_bindings(self):
        self.assertEqual(set(self.bundle), {
            "schema_version", "document_type", "method", "weights", "threshold",
            "rank", "dimension", "fit_split", "contract_sha256",
            "validation_refit_allowed", "confirmation_refit_allowed",
        })
        self.assertEqual(self.bundle["fit_split"], "engineering_fixture")
        self.assertEqual(self.bundle["contract_sha256"], self.contract_sha)
        self.assertFalse(self.bundle["validation_refit_allowed"])
        self.assertFalse(self.bundle["confirmation_refit_allowed"])

    def test_score_direction(self):
        self.assertGreater(readout_score(self.bundle, [0, 4, 0]), 0)
        self.assertLess(readout_score(self.bundle, [0, -4, 0]), 0)

    def test_invalid_contract_digest_rejected(self):
        with self.assertRaises(ValueError):
            fit_frozen_readout([[0, 1]], [[0, -1]], [[0, 1]], "engineering_fixture", "bad")

    def test_readout_schema_is_closed(self):
        schema = json.loads((HERE / "schemas/n1_r3_frozen_readout.schema.json").read_text())
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(set(schema["required"]), set(schema["properties"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
