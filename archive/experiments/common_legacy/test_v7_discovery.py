from __future__ import annotations

import unittest

from experiments.common.v7_discovery import PersistentReviewTracker
from src.rejection_buffer import RejectionRecord


def _record(identifier: int, x: float, window: int) -> RejectionRecord:
    return RejectionRecord(
        record_id=identifier,
        embedding=(x, 0.0),
        timestamp=float(window),
        window_id=window,
        source_model_signature="model",
        support_profile_version="profile",
        novelty_score=1.0,
        decision_margin=0.1,
        nearest_candidates=(),
        source_sample_id=f"sample-{identifier}",
    )


class PersistentReviewTrackerTests(unittest.TestCase):
    def test_review_id_survives_cluster_growth(self) -> None:
        tracker = PersistentReviewTracker(minimum_cluster_size=2)
        first = tracker.update(((_record(0, 0.0, 0), _record(1, 0.1, 0)),), 0)
        second = tracker.update(
            (
                (
                    _record(0, 0.0, 0),
                    _record(1, 0.1, 0),
                    _record(2, 0.2, 1),
                ),
            ),
            1,
        )
        self.assertEqual(first[0].review_id, second[0].review_id)
        self.assertEqual(second[0].state, "established")
        self.assertEqual(
            tracker.request_reviews(1)[0].state,
            "review_requested",
        )
