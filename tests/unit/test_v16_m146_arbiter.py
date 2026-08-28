"""Unit tests for the M146 arbiter cell (pure model arithmetic only)."""
import unittest

import numpy as np
import torch

from experiments.tier4.eval_v15_m103_atoms import Whitener
from experiments.tier4.eval_v16_m142_c2 import _spm_pool
from experiments.tier4.eval_v16_m142_factorial import power_norm
from experiments.tier4.eval_v16_m146_arbiter import HeadOnly, SpmSqrtModel


class TestSpmPoolInvariants(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        # contract: activation is (count * grid * grid, atoms)
        act = torch.from_numpy(
            rng.standard_normal((2 * 729, 3)).astype(np.float32))
        pooled = _spm_pool(act, 2, 27)
        self.assertEqual(pooled.shape, (2, 21 * 3))

    def test_bins_partition_the_grid(self):
        rng = np.random.default_rng(1)
        act = torch.from_numpy(
            rng.standard_normal((2 * 729, 3)).astype(np.float32))
        pooled = _spm_pool(act, 2, 27)
        full = act.reshape(2, 27, 27, 3).sum(dim=(1, 2))
        # the 1x1 level is the whole-grid sum per atom; all 21 bins sum to 3x
        np.testing.assert_allclose(pooled[:, :3].numpy(), full.numpy(),
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(pooled.sum(dim=1).numpy(),
                                   (3 * full.sum(dim=1)).numpy(),
                                   rtol=1e-4, atol=1e-3)


class TestSpmSqrtModelParity(unittest.TestCase):
    def test_features_match_the_numpy_path(self):
        torch.manual_seed(1)
        rng = np.random.default_rng(2)
        mean = rng.standard_normal(108).astype(np.float32)
        whiten = np.linalg.qr(rng.standard_normal((108, 108)))[0].astype(
            np.float32)
        dictionary = rng.standard_normal((8, 108)).astype(np.float32)
        whitener = Whitener(6, 1, 10.0, mean, whiten, 27)
        images = rng.integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8)

        # the sealed numpy path
        white = whitener(images)  # (3, 729, 108)
        distances = torch.cdist(torch.from_numpy(white),
                                torch.from_numpy(dictionary))
        activation = torch.clamp(distances.mean(dim=1, keepdim=True)
                                 - distances, min=0.0)
        pooled = _spm_pool(activation, 3, 27)
        expected = power_norm(pooled.numpy(), 0.5)

        model = SpmSqrtModel(dictionary, mean, whiten, 27, 345, 10.0, 0.5,
                             False, torch.device("cpu"))
        with torch.no_grad():
            got = model.features(
                torch.from_numpy(images.transpose(0, 3, 1, 2)
                                 ).float() / 255.0).numpy()
        # pipeline-level parity: same whitening, cdist, pooling and
        # power-norm up to (a) float32-vs-float64 arithmetic and (b) the
        # registered 1e-12 clamp before the square root (tiny non-zeros
        # where the numpy path has exact zeros). Structural errors would
        # be O(1).
        np.testing.assert_allclose(got, expected, rtol=1e-2, atol=1e-3)
        # the clamp is visible and tiny, never structural
        self.assertLessEqual(float(np.abs(got - expected).max()), 0.02)

    def test_forward_shape_and_determinism(self):
        torch.manual_seed(3)
        rng = np.random.default_rng(4)
        dictionary = rng.standard_normal((6, 108)).astype(np.float32)
        model = SpmSqrtModel(dictionary,
                             np.zeros(108, np.float32),
                             np.eye(108, dtype=np.float32), 27, 345, 10.0,
                             0.5, True, torch.device("cpu"))
        x = torch.rand(4, 3, 32, 32)
        logits = model(x)
        self.assertEqual(logits.shape, (4, 345))
        np.testing.assert_allclose(logits.detach().numpy(),
                                   model(x).detach().numpy(), rtol=0, atol=0)


class TestHeadOnly(unittest.TestCase):
    def test_shape(self):
        model = HeadOnly(63, 345, torch.device("cpu"))
        logits = model(torch.rand(5, 63))
        self.assertEqual(logits.shape, (5, 345))


if __name__ == "__main__":
    unittest.main()
