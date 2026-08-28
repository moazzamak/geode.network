"""M283 — authored adversarial-router probe suite: artifact
integrity (commit-reveal per probe) and deterministic evaluation."""
import json
import unittest
from pathlib import Path

from geode.core.router_probes import RouterProbeSuite

ARTIFACT = (Path(__file__).resolve().parents[2] / "analysis"
            / "router_probes_v0.json")


class TestRouterProbes(unittest.TestCase):
    def setUp(self):
        self.suite = RouterProbeSuite(ARTIFACT)

    def test_artifact_loads(self):
        self.assertEqual(self.suite.axis, "routing")
        self.assertEqual(len(self.suite.probes), 16)
        self.assertEqual(set(self.suite.categories()),
                         {"surface_spoof", "contract_spoof",
                          "marker_salting", "injection_spoof"})

    def test_integrity_ok(self):
        report = self.suite.verify_integrity()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["broken_probes"], [])

    def test_tampered_probe_detected(self):
        raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        raw["probes"][0]["text"] = "tampered"
        tampered = json.dumps(raw)
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            fh.write(tampered)
            path = Path(fh.name)
        try:
            suite = RouterProbeSuite(path)
            report = suite.verify_integrity()
            self.assertFalse(report["ok"])
            self.assertIn(raw["probes"][0]["id"],
                          report["broken_probes"])
        finally:
            path.unlink(missing_ok=True)

    def test_perfect_router_zero_misroutes(self):
        result = self.suite.evaluate(
            lambda text: next(
                p["expected_family"] for p in
                self.suite.probes.values() if p["text"] == text))
        self.assertEqual(result["overall_misroute_rate"], 0.0)
        self.assertTrue(result["within_bar"])
        self.assertEqual(result["n_probes"], 16)

    def test_blind_router_fails_bar(self):
        result = self.suite.evaluate(lambda _text: "sentiment")
        self.assertGreater(result["overall_misroute_rate"],
                           self.suite.bar)
        self.assertFalse(result["within_bar"])
        # 7 of the 16 authored probes expect sentiment
        self.assertEqual(
            sum(p["misroute"] for p in result["per_probe"].values()),
            9)

    def test_unknown_family_counts_as_misroute(self):
        result = self.suite.evaluate(lambda _text: "unknown")
        self.assertEqual(result["overall_misroute_rate"], 1.0)
        self.assertFalse(result["within_bar"])


if __name__ == "__main__":
    unittest.main()
