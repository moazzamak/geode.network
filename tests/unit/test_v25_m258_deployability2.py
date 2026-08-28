"""Unit tests for M258: artifact fetch/verify, the serve subcommand,
live metrics, snapshot round-trip, and the Dockerfile presence.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from geode.api.metrics import MetricsCollector
from geode.api.persistence import load_snapshot, save_snapshot
from geode.audit import sha256_file
from geode.cli import main as cli_main
from geode.core.artifacts import ArtifactStore, verify_artifact
from geode.core.orchestrator import Orchestrator

REPO = Path(__file__).resolve().parents[2]


def _cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(argv)
    return code, buf.getvalue()


class TestM258Artifacts(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ArtifactStore(Path(self.tmp.name) / "store")

    def test_publish_fetch_round_trip(self):
        payload = Path(self.tmp.name) / "weights.bin"
        payload.write_bytes(b"geode" * 100)
        ref = self.store.publish(payload, "weights.bin", location="arm-a")
        dest = Path(self.tmp.name) / "out.bin"
        self.store.fetch_and_verify(ref, dest)
        self.assertEqual(dest.read_bytes(), payload.read_bytes())

    def test_mismatched_digest_never_admitted(self):
        payload = Path(self.tmp.name) / "weights.bin"
        payload.write_bytes(b"geode" * 100)
        ref = self.store.publish(payload, "weights.bin", location="arm-a")
        ref2 = type(ref)(name=ref.name, digest="0" * 64, size=ref.size,
                         location=ref.location)
        with self.assertRaises(ValueError):
            self.store.fetch_and_verify(ref2,
                                        Path(self.tmp.name) / "out.bin")

    def test_missing_artifact_raises(self):
        from geode.core.artifacts import ArtifactRef
        ref = ArtifactRef(name="x", digest="0" * 64, size=1,
                          location="missing")
        with self.assertRaises(FileNotFoundError):
            self.store.fetch_and_verify(ref, Path(self.tmp.name) / "o")

    def test_cli_artifacts_verify(self):
        payload = Path(self.tmp.name) / "f.bin"
        payload.write_bytes(b"hello")
        digest = sha256_file(payload)
        code, out = _cli(["artifacts", "verify", "--path",
                          str(payload), "--digest", digest])
        self.assertEqual(code, 0)
        self.assertIn("digest matches", out)
        code, _ = _cli(["artifacts", "verify", "--path",
                        str(payload), "--digest", "0" * 64])
        self.assertEqual(code, 1)


class TestM258Metrics(unittest.TestCase):

    def test_summary_percentiles(self):
        m = MetricsCollector(window=100)
        for d in range(1, 101):
            m.record(float(d))
        s = m.summary()
        self.assertEqual(s["count"], 100)
        self.assertAlmostEqual(s["p50_ms"], 50.5, delta=0.6)
        self.assertAlmostEqual(s["p99_ms"], 99.0, delta=0.6)

    def test_empty_summary(self):
        s = MetricsCollector().summary()
        self.assertEqual(s["count"], 0)
        self.assertIsNone(s["p50_ms"])

    def test_negative_duration_raises(self):
        with self.assertRaises(ValueError):
            MetricsCollector().record(-1.0)


class TestM258ServeAndSnapshot(unittest.TestCase):

    def test_serve_dry_run(self):
        code, out = _cli(["serve", "--dry-run", "--port", "9001"])
        self.assertEqual(code, 0)
        self.assertIn("uvicorn geode.api.service:app", out)
        self.assertIn("9001", out)

    def test_snapshot_round_trip_deterministic(self):
        from geode.core.arm import arm_from_sealed_head
        orch = Orchestrator()
        orch.register(arm_from_sealed_head(
            "a", "ms", 32, 0.5, "demo", price=1.0))
        reqs = [{"query_id": "q1", "fingerprint": [0.9, 0.1], "k": 1,
                 "task_id": None, "contract_kind": None}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            h1 = save_snapshot(orch, reqs, path)
            orch2 = Orchestrator()
            load_snapshot(orch2, path)
            self.assertEqual(sorted(orch2.router.list_arms()),
                             ["a"])
            h2 = save_snapshot(orch2, reqs, path)
            self.assertEqual(h1, h2)

    def test_unknown_snapshot_schema_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            path.write_text(json.dumps({"schema_version": 99}),
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                load_snapshot(Orchestrator(), path)

    def test_dockerfile_present(self):
        self.assertTrue((REPO / "Dockerfile").exists())


if __name__ == "__main__":
    unittest.main()
