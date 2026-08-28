"""M223 — API persistence and demo-seed integration tests."""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from geode.api import create_app


class TestPersistence(unittest.TestCase):
    def test_snapshot_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            client = TestClient(create_app(snapshot_path=path))
            client.post("/arms", json={"arm_id": "a", "accuracy": 0.5})
            client.post("/route", json={"query_id": "q1",
                                        "task_id": "d0"})
            res = client.post("/snapshot")
            self.assertEqual(res.status_code, 200, res.text)
            tip_before = client.get("/ledger").json()["tip"]

            # a fresh app loads the snapshot and reproduces the chain
            client2 = TestClient(create_app(snapshot_path=path))
            body = client2.get("/ledger").json()
            self.assertEqual(body["tip"], tip_before)
            self.assertTrue(body["verify"]["ok"])
            self.assertEqual(body["record_count"], 2)

    def test_bad_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"schema_version": 99}), "utf-8")
            from geode.api.persistence import load_snapshot
            from geode.core.orchestrator import Orchestrator
            with self.assertRaises(ValueError):
                load_snapshot(Orchestrator(), path)


class TestDemoSeed(unittest.TestCase):
    def test_seed_registers_and_routes(self):
        client = TestClient(create_app())
        res = client.post("/demo/seed")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(len(res.json()["registered"]), 3)
        # d3 routes to the sealed ms arm (0.33638 beats synthetic 0.30)
        route = client.post("/route", json={"query_id": "q",
                                            "task_id": "d3"})
        self.assertEqual(route.json()["routed"][0]["arm_id"], "ms_ridge")
        # idempotent: a second seed registers nothing new
        again = client.post("/demo/seed")
        self.assertEqual(again.json()["registered"], [])


if __name__ == "__main__":
    unittest.main()
