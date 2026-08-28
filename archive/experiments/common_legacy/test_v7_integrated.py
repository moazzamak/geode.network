from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v7_integrated import select_review_support
from src.rejection_buffer import RejectionRecord


class IntegratedLoopTests(unittest.TestCase):
    def test_no_clustering_reviews_every_reject_without_semantic_creation(self) -> None:
        records = tuple(
            RejectionRecord(
                record_id=index,
                embedding=(float(index), 0.0),
                timestamp=0.0,
                window_id=0,
                source_model_signature="model",
                support_profile_version="profile",
                novelty_score=float(index),
                decision_margin=1.0,
                nearest_candidates=(),
                source_sample_id=f"seed-1-sample-{index:05d}",
            )
            for index in range(4)
        )
        labels = {
            str(record.source_sample_id): index % 2
            for index, record in enumerate(records)
        }
        support, reviewed, objects = select_review_support(
            records,
            labels,
            discovery_arm="no_clustering",
            target_label=1,
            review_budget=2,
            minimum_cluster_size=2,
            seed=1,
        )
        self.assertEqual(reviewed, 4)
        self.assertEqual(objects, 1)
        self.assertEqual(support.tolist(), [3, 1])
