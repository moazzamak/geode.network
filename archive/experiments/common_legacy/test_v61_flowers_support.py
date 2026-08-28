from __future__ import annotations

import json
import unittest

import numpy as np

from experiments.common.v61_flowers_support import (
    deserialize_primitive_head,
    fit_global_temperature,
    fit_rank3_primitives,
    predict_primitives,
    serialize_primitive_head,
    support_tier_status,
)
from experiments.tier4.eval_v61_flowers_support import (
    DEFAULT_CONFIG,
    _validate_config,
)


class FlowersSupportTests(unittest.TestCase):
    def _data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(11)
        features = np.vstack(
            [rng.normal(loc=offset, scale=0.1, size=(5, 8)) for offset in (-1, 1)]
        )
        return features, np.repeat([0, 1], 5)

    def test_support_tiers_are_separate_and_train_only(self):
        _, labels = self._data()
        self.assertEqual(
            support_tier_status(labels, rank=3, allowed_fit_splits=["train"])[
                "status"
            ],
            "feasible",
        )
        self.assertEqual(
            support_tier_status(labels, rank=32, allowed_fit_splits=["train"])[
                "status"
            ],
            "blocked",
        )
        with self.assertRaises(ValueError):
            support_tier_status(labels, rank=3, allowed_fit_splits=["train", "dev"])

    def test_rank3_affine_and_tangent_heads_replay(self):
        features, labels = self._data()
        for tangent in (False, True):
            with self.subTest(tangent=tangent):
                candidates = fit_rank3_primitives(
                    features, labels, tangent=tangent
                )
                logits = np.column_stack(
                    [
                        -candidate.radial_field(
                            features / np.linalg.norm(features, axis=1, keepdims=True)
                            if tangent
                            else features
                        )
                        for candidate in candidates
                    ]
                )
                temperature = fit_global_temperature(
                    logits,
                    labels,
                    np.array([0, 1]),
                    minimum=0.05,
                    maximum=20.0,
                )
                first = predict_primitives(
                    candidates,
                    features,
                    tangent=tangent,
                    temperature=temperature,
                )
                payload = serialize_primitive_head(
                    candidates,
                    tangent=tangent,
                    temperature=temperature,
                    representation_hash="a" * 64,
                )
                replay, replay_tangent, replay_temperature = (
                    deserialize_primitive_head(payload)
                )
                second = predict_primitives(
                    replay,
                    features,
                    tangent=replay_tangent,
                    temperature=replay_temperature,
                )
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(first[1], second[1])

    def test_registered_config_is_strict_and_test_sealed(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        _validate_config(config)
        config["test_labels_opened"] = True
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_rank_or_partition_drift_fails_closed(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["a4_f5"]["rank"] = 4
        with self.assertRaises(ValueError):
            _validate_config(config)
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["a4_f34"]["allowed_fit_splits"].append("dev")
        with self.assertRaises(ValueError):
            _validate_config(config)


if __name__ == "__main__":
    unittest.main()
