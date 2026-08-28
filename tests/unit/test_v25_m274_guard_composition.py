"""M274 — guard composition tests.

The M263 lesson pinned as policy: the geometric guard alone scored
its own OOD probes in-distribution; the composed guard (geometric +
structural vocab coverage) rejects them; and a guard that leaks its
own probes is never admitted (registration raises).
"""
import unittest

from geode.core.guard_composition import (
    ComposedGuard,
    GuardRegistry,
    VocabCoveragePrimitive,
)
from geode.core.ood import OodGate


def _geometric_gate():
    """A broad-profile gate that ADMITS junk (the M263 reproduction):
    reference = natural sentences, so token soup scores inside
    distribution."""
    gate = OodGate(threshold=3.0)
    gate.fit_profile([
        [0.1, 0.2, 0.1],
        [0.2, 0.1, 0.2],
        [0.1, 0.1, 0.1],
    ])
    return gate


class TestComposedGuard(unittest.TestCase):
    def test_geometric_alone_admits_soup(self):
        gate = _geometric_gate()
        # a probe close to the centroid: geometry alone admits it
        self.assertTrue(gate.admits([0.1, 0.2, 0.1])["admitted"])

    def test_composed_rejects_soup_via_structural(self):
        guard = ComposedGuard(
            _geometric_gate(),
            [("vocab", VocabCoveragePrimitive(
                {"the", "movie", "was", "good"}).check)])
        soup = "XKCD 0x1F4A9 base64 ZGF0YQ=="
        decision = guard.admit(soup, [0.1, 0.2, 0.1])
        self.assertFalse(decision["admitted"])
        self.assertEqual(decision["reason"], "low_vocab_coverage")

    def test_composed_admits_natural_text(self):
        guard = ComposedGuard(
            _geometric_gate(),
            [("vocab", VocabCoveragePrimitive(
                {"the", "movie", "was", "good"}).check)])
        decision = guard.admit("the movie was good",
                               [0.1, 0.2, 0.1])
        self.assertTrue(decision["admitted"])

    def test_unfitted_geometric_fails_closed(self):
        guard = ComposedGuard(
            OodGate(threshold=3.0),
            [("vocab", VocabCoveragePrimitive({"a"}).check)])
        decision = guard.admit("a", [0.1])
        self.assertFalse(decision["admitted"])
        self.assertEqual(decision["reason"], "guard_unfitted")


class TestGuardRegistry(unittest.TestCase):
    def _guard(self):
        return ComposedGuard(
            _geometric_gate(),
            [("vocab", VocabCoveragePrimitive(
                {"the", "movie", "was", "good"}).check)])

    def test_guard_admitted_after_own_probes_rejected(self):
        registry = GuardRegistry()
        probes = [("XKCD base64 ZGF0YQ==", [0.1, 0.2, 0.1]),
                  ("<script> alert(1) </script>", [0.15, 0.15, 0.15])]
        registry.register_guard("sentiment", self._guard(), probes)
        self.assertIsNotNone(registry.guard("sentiment"))

    def test_guard_leaking_probe_never_admitted(self):
        registry = GuardRegistry()
        # a probe the guard ADMITS -> registration raises
        probes = [("the movie was good", [0.1, 0.2, 0.1])]
        with self.assertRaises(ValueError):
            registry.register_guard("sentiment", self._guard(), probes)

    def test_per_arm_isolation(self):
        registry = GuardRegistry()
        registry.register_guard(
            "a", self._guard(),
            [("junk", [0.1, 0.2, 0.1])])
        self.assertIsNone(registry.guard("b"))
        self.assertEqual(
            registry.admit("b", "the movie was good",
                           [0.1, 0.2, 0.1])["reason"],
            "guard_missing")


if __name__ == "__main__":
    unittest.main()
