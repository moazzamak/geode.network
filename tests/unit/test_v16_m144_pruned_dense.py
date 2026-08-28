"""Unit tests for the M144 pruning logic (the exact-removal invariant).

Run from the repo root with the CPU environment::

    .\\.venv\\Scripts\\python.exe -m unittest experiments.common.test_v16_m144_pruned_dense
"""

import types
import unittest

import numpy as np
import torch

from experiments.tier4.eval_v16_m144_pruned_dense import _prune


def _fake_model(width=4, n_heads=2, mlp_hidden=4):
    """Minimal object with the transformers attribute shape _prune walks."""
    layer = types.SimpleNamespace()
    layer.attention = types.SimpleNamespace()
    layer.attention.attention = types.SimpleNamespace(
        query=torch.nn.Linear(width, width, bias=True),
        key=torch.nn.Linear(width, width, bias=True),
        value=torch.nn.Linear(width, width, bias=True),
    )
    layer.attention.output = types.SimpleNamespace(
        dense=torch.nn.Linear(width, width, bias=True),
    )
    layer.mlp = types.SimpleNamespace(
        fc1=torch.nn.Linear(width, mlp_hidden, bias=True),
        fc2=torch.nn.Linear(mlp_hidden, width, bias=True),
    )
    model = types.SimpleNamespace(
        encoder=types.SimpleNamespace(layer=[layer]),
        config=types.SimpleNamespace(num_attention_heads=n_heads,
                                     hidden_size=width,
                                     mlp_ratio=mlp_hidden / width),
    )

    def parameters():
        for p in (layer.attention.attention.query.parameters(),
                  layer.attention.attention.key.parameters(),
                  layer.attention.attention.value.parameters(),
                  layer.attention.output.dense.parameters(),
                  layer.mlp.fc1.parameters(), layer.mlp.fc2.parameters()):
            yield from p

    model.parameters = parameters
    return model


class PruneInvariantTests(unittest.TestCase):
    def test_zeroing_covers_both_sides_and_biases(self):
        torch.manual_seed(0)
        model = _fake_model()
        stats = _prune(model, 0.5)
        head_dim = 2
        attn = model.encoder.layer[0].attention.attention
        out = model.encoder.layer[0].attention.output.dense
        mlp = model.encoder.layer[0].mlp
        for proj in (attn.query, attn.key, attn.value):
            w = proj.weight.detach().numpy()
            b = proj.bias.detach().numpy()
            for h in range(2):
                lo, hi = h * head_dim, (h + 1) * head_dim
                if np.abs(w[lo:hi]).max() == 0.0:  # dropped head
                    self.assertEqual(np.abs(out.weight.detach().numpy()
                                             [:, lo:hi]).max(), 0.0)
                    self.assertEqual(np.abs(b[lo:hi]).max(), 0.0)
        for u in range(4):
            if np.abs(mlp.fc1.weight.detach().numpy()[u]).max() == 0.0:
                self.assertEqual(np.abs(mlp.fc2.weight.detach().numpy()
                                        [:, u]).max(), 0.0)
                self.assertEqual(float(mlp.fc1.bias.detach().numpy()[u]), 0.0)

    def test_fraction_arithmetic(self):
        torch.manual_seed(1)
        model = _fake_model()
        stats = _prune(model, 0.5)
        self.assertEqual(stats["n_heads"], 2)
        self.assertEqual(stats["mlp_hidden"], 4)
        self.assertAlmostEqual(stats["kept_head_fraction"], 0.5)
        self.assertAlmostEqual(stats["kept_mlp_fraction"], 0.5)
        self.assertEqual(stats["dropped_heads"], 1)
        self.assertEqual(stats["dropped_mlp_units"], 2)

    def test_pruned_forward_equals_subnetwork(self):
        # the pruned model must equal the kept subnetwork built by hand
        # (zeroing both sides makes the removal exact)
        torch.manual_seed(2)
        model = _fake_model(width=4, n_heads=2, mlp_hidden=4)
        x = torch.randn(3, 4)
        stats = _prune(model, 0.5)
        attn = model.encoder.layer[0].attention.attention
        out = model.encoder.layer[0].attention.output.dense
        mlp = model.encoder.layer[0].mlp
        width, head_dim = 4, 2

        with torch.no_grad():
            pruned = mlp.fc2(mlp.fc1(out(attn.query(x)
                                        + attn.key(x) + attn.value(x))))

            head_keep = np.flatnonzero(
                np.abs(attn.query.weight.detach().numpy()).max(axis=1) > 0)
            a_sub = torch.zeros(3, width)
            for proj in (attn.query, attn.key, attn.value):
                a_sub[:, head_keep] += torch.nn.functional.linear(
                    x, proj.weight[head_keep], proj.bias[head_keep])
            sub = out(a_sub)

            unit_keep = np.flatnonzero(
                np.abs(mlp.fc1.weight.detach().numpy()).max(axis=1) > 0)
            h_sub = torch.zeros(3, mlp.fc1.out_features)
            h_sub[:, unit_keep] = torch.nn.functional.linear(
                sub, mlp.fc1.weight[unit_keep], mlp.fc1.bias[unit_keep])
            sub = mlp.fc2(h_sub)
        np.testing.assert_allclose(pruned.numpy(), sub.numpy(),
                                   rtol=0, atol=1e-5)
        self.assertGreaterEqual(stats["nonzero_fraction"], 0.4)


if __name__ == "__main__":
    unittest.main()
