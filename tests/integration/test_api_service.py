"""M217 — API service integration tests: the full product loop over
HTTP (register -> route -> ledger -> settlement), local-only."""
import unittest

from fastapi.testclient import TestClient

from geode.api import create_app


class TestApiService(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_health(self) -> None:
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")
        self.assertEqual(res.json()["arms"], 0)

    def test_register_and_route(self) -> None:
        for arm_id, acc in (("arm_a", 0.20), ("arm_b", 0.30)):
            res = self.client.post("/arms", json={
                "arm_id": arm_id, "family": "fam", "width": 100,
                "accuracy": acc, "per_task": {"d0": acc, "d1": acc + 0.01}})
            self.assertEqual(res.status_code, 200, res.text)
        res = self.client.post("/route", json={
            "query_id": "q1", "task_id": "d1", "k": 1})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["routed"][0]["arm_id"], "arm_b")

    def test_invalid_arm_rejected(self) -> None:
        res = self.client.post("/arms", json={
            "arm_id": "bad", "accuracy": 2.0})
        self.assertEqual(res.status_code, 422)

    def test_missing_field_rejected(self) -> None:
        res = self.client.post("/arms", json={"accuracy": 0.5})
        self.assertEqual(res.status_code, 422)

    def test_ledger_records_and_verifies(self) -> None:
        self.client.post("/arms", json={"arm_id": "a", "accuracy": 0.5})
        self.client.post("/route", json={"query_id": "q1"})
        res = self.client.get("/ledger")
        body = res.json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(body["verify"]["ok"])
        self.assertEqual(body["record_count"], 2)

    def test_settlement_conforms(self) -> None:
        self.client.post("/arms", json={"arm_id": "a", "accuracy": 0.5})
        self.client.post("/route", json={"query_id": "q1"})
        res = self.client.post("/settlement/batches", json={
            "price_per_query": 100})
        body = res.json()
        self.assertEqual(res.status_code, 200, res.text)
        self.assertTrue(body["conforms"], body["violations"])
        self.assertEqual(len(body["batches"]), 1)
        self.assertEqual(len(body["batches"][0]["entries"]), 1)

    def test_frontend_served(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("GEODE console", res.text)

    def test_duplicate_arm_id_is_append_only(self) -> None:
        spec = {"arm_id": "dup", "accuracy": 0.5}
        self.assertEqual(self.client.post("/arms", json=spec).status_code,
                         200)
        res = self.client.post("/arms", json=spec)
        self.assertEqual(res.status_code, 409)  # duplicate key rejected


if __name__ == "__main__":
    unittest.main()
