import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nns_cst_n2.core import (
    N2TypedError, fit_readout, nuisance_residualize, projector,
    projector_similarity, random_basis, subspace_basis,
)
from nns_cst_n2.producer import low_level_metrics, matching_score, remove_donor_foreground


class N2CoreTests(unittest.TestCase):
    def test_nuisance_removal(self):
        control = np.array([[1., 0., 0.], [2., 0., 0.]])
        signal = np.array([[3., 2., 0.], [4., -2., 0.]])
        residual, nuisance = nuisance_residualize(signal, control)
        self.assertEqual(nuisance.shape, (1, 3))
        self.assertTrue(np.allclose(residual[:, 0], 0.0))
        self.assertTrue(np.allclose(residual[:, 1], signal[:, 1]))

    def test_projector_rotation(self):
        basis = np.array([[1., 0., 0.], [0., 1., 0.]])
        angle = 0.63
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.assertTrue(np.allclose(projector(basis), projector(rotation @ basis), atol=1e-12))
        self.assertAlmostEqual(projector_similarity(basis, rotation @ basis), 1.0, places=12)

    def test_subspace_and_readout(self):
        signal = np.array([[2., 0., .1], [1.8, .1, 0.], [-2., 0., -.1], [-1.7, -.1, 0.]])
        basis, values = subspace_basis(signal, 1)
        self.assertGreater(values[0], 0)
        bundle = fit_readout(signal[:2], signal[2:], basis)
        score_pos = np.asarray(signal[:2]) @ np.asarray(bundle["weights"]) - bundle["threshold"]
        score_neg = np.asarray(signal[2:]) @ np.asarray(bundle["weights"]) - bundle["threshold"]
        self.assertGreater(score_pos.mean(), score_neg.mean())

    def test_random_basis(self):
        first = random_basis(9, 3, 4); second = random_basis(9, 3, 4)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.allclose(first @ first.T, np.eye(3)))

    def test_rank_failure(self):
        with self.assertRaises(N2TypedError) as error:
            subspace_basis(np.eye(2), 3)
        self.assertEqual(error.exception.code, "RANK_EXCEEDS_DIMENSION")

    def test_donor_removal(self):
        image = np.zeros((5, 5, 3), dtype=np.float32); image[..., 1] = 0.4; image[2, 2] = 1
        mask = np.zeros((5, 5), dtype=bool); mask[2, 2] = True
        filled = remove_donor_foreground(image, mask)
        self.assertFalse(np.allclose(filled[2, 2], [1., 1., 1.]))
        self.assertTrue(np.isfinite(filled).all())
        self.assertEqual(filled.dtype, np.float32)

    def test_low_level_metrics(self):
        base = np.zeros((8, 8, 3), dtype=np.float32); other = base.copy(); other[:, 4:] = .5
        mask = np.zeros((8, 8), dtype=bool); mask[2:6, 2:6] = True
        metrics = low_level_metrics(base, other, mask)
        self.assertEqual(set(metrics), {"L1", "L2", "RGB_MEAN_SHIFT", "HIGHPASS_L1", "BOUNDARY_L1"})
        score, maximum, ratios = matching_score(metrics, metrics)
        self.assertEqual(score, 0.0); self.assertEqual(maximum, 1.0); self.assertEqual(set(ratios), set(metrics))


if __name__ == "__main__":
    unittest.main()
