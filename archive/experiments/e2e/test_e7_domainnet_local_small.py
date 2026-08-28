from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from experiments.e2e.run_e7_domainnet_local_small import (
    _bounded_records,
    _image_feature,
    _nearest_centroid_metrics,
    _validate_config,
)


class E7DomainNetLocalSmallTests(unittest.TestCase):
    def test_features_budgets_and_centroid_replay_are_deterministic(self):
        image = Image.new("RGB", (20, 20), (64, 128, 192))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.assertEqual(
            _image_feature(buffer.getvalue(), 16),
            _image_feature(buffer.getvalue(), 16),
        )

        records = []
        for domain in (0, 1):
            for label in (0, 1):
                for index in range(3):
                    feature = [float(label), float(domain), float(index)]
                    records.append({
                        "domain": domain,
                        "label": label,
                        "image_path": f"{domain}-{label}-{index}.png",
                        "feature": feature,
                    })
        bounded = _bounded_records(
            [{"records": records}],
            {(domain, label): 2 for domain in (0, 1) for label in (0, 1)},
        )
        self.assertEqual(len(bounded), 8)
        source = [record for record in bounded if record["domain"] == 1]
        evaluation = [record for record in bounded if record["domain"] == 0]
        first = _nearest_centroid_metrics(source, evaluation, [0, 1])
        second = _nearest_centroid_metrics(source, evaluation, [0, 1])
        self.assertEqual(first, second)
        self.assertEqual(first["accuracy"], 1.0)

    def test_config_rejects_overlapping_domains(self):
        config = {
            "class_ids": [0, 1],
            "source_domain_ids": [1, 2],
            "validation_domain_id": 0,
            "final_domain_id": 2,
            "samples_per_class_domain": 1,
            "image_size": 16,
        }
        with self.assertRaisesRegex(ValueError, "valid and disjoint"):
            _validate_config(config)


if __name__ == "__main__":
    unittest.main()