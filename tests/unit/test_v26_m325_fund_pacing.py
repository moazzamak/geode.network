"""Unit tests for the M325 liveness amendment — default-release
fund pacing."""
from __future__ import annotations

import unittest

from geode.core.fund_pacing import (
    MAJORITY,
    ReleaseSchedule,
    VoteSet,
    ZakatDisbursement,
)
from geode.privacy.vote_machinery import ratifies


def _schedule() -> ReleaseSchedule:
    return ReleaseSchedule(ratify=ratifies, hold_epochs=1)


class TestDefaultRelease(unittest.TestCase):

    def test_silence_releases_at_window_close(self):
        s = _schedule()
        s.schedule("r1", 100.0, scheduled_epoch=1, window_epochs=1)
        s.advance(epoch=2, votes={})   # no votes at all
        self.assertEqual(s.executed[-1]["release_id"], "r1")
        self.assertEqual(s.executed[-1]["path"], "window_closed_default")
        self.assertEqual(len(s.releases), 0)

    def test_absence_of_signatures_cannot_block_indefinitely(self):
        s = _schedule()
        s.schedule("r1", 100.0, scheduled_epoch=1, window_epochs=1)
        for epoch in range(1, 6):
            # every epoch: nobody votes
            s.advance(epoch=epoch, votes={})
        self.assertEqual(s.executed[-1]["release_id"], "r1")
        self.assertEqual(s.executed[-1]["path"], "window_closed_default")

    def test_positive_majority_releases(self):
        s = _schedule()
        s.schedule("r1", 100.0, scheduled_epoch=1, window_epochs=2)
        # three distinct supporting identities meet the diversity
        # floor for three responders
        vs = VoteSet(weights={"a": 3.0, "b": 2.0, "c": 1.0,
                              "d": 2.0},
                     votes={"a": True, "b": True, "c": True,
                            "d": False},
                     responders=4, pool_size=9)
        s.advance(epoch=1, votes={"r1": vs})
        self.assertEqual(s.executed[-1]["release_id"], "r1")
        self.assertEqual(s.executed[-1]["path"], "quorum_release")


class TestAffirmativeNegative(unittest.TestCase):

    def test_negative_majority_holds(self):
        s = _schedule()
        s.schedule("r1", 100.0, scheduled_epoch=1, window_epochs=1)
        self.assertTrue(s.hold(epoch=1, release_id="r1"))
        s.advance(epoch=2, votes={})
        # held: not executed, re-enters after the hold window
        self.assertEqual(s.executed, [])
        self.assertFalse(s.releases[0].held or
                         s.releases[0].hold_until_epoch > 2)
        s.advance(epoch=3, votes={})
        self.assertEqual(s.executed[-1]["release_id"], "r1")

    def test_hold_is_never_a_cancel(self):
        s = _schedule()
        s.schedule("r1", 100.0, scheduled_epoch=1, window_epochs=1)
        s.hold(epoch=1, release_id="r1")
        s.advance(epoch=2, votes={})
        s.advance(epoch=3, votes={})   # hold window (1 epoch) elapsed
        self.assertEqual(s.executed[-1]["release_id"], "r1")

    def test_below_majority_is_not_a_block(self):
        s = _schedule()
        s.schedule("r1", 100.0, scheduled_epoch=1, window_epochs=1)
        vs = VoteSet(weights={"a": 3.0, "b": 2.0},
                     votes={"a": True, "b": False},
                     responders=2, pool_size=9)
        s.advance(epoch=1, votes={"r1": vs})   # 3/5 > 1/2 but below
                                               # responders minimum
        s.advance(epoch=2, votes={})           # default release
        self.assertEqual(s.executed[-1]["path"], "window_closed_default")


class TestZakat(unittest.TestCase):
    """M325-G4/G5: the end state has no vote path and no pause."""

    def test_mechanical_disbursement_has_no_hold(self):
        z = ZakatDisbursement(recipients=[("class-a", 50.0),
                                          ("class-b", 50.0)])
        out = z.disburse()
        self.assertEqual(sum(o["amount"] for o in out), 100.0)
        self.assertFalse(hasattr(z, "hold"))
        self.assertFalse(hasattr(z, "advance"))


if __name__ == "__main__":
    unittest.main()
