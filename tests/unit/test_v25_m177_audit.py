"""Unit tests for M177: the GEODE audit API v0 (L0 replay + L1 provenance).

Pure stdlib + a trivial in-memory runner; no data, no GPU. Covers the
registered contract: bit-exact replay, timing-field exclusion,
tamper detection, provenance chain resolution.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from geode.audit import AuditAPI, evidence_content_hash
from experiments.common.v5_artifacts import write_canonical_json


def _toy_runner(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "milestone": "TOY",
        "configuration_hash": evidence_content_hash(config),
        "value": config["value"],
        "runtime_seconds": 0.123,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    return evidence


def _fixture() -> tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp())
    config_path = tmp / "config.json"
    config_path.write_text(json.dumps({"value": 42}), encoding="utf-8")
    evidence = _toy_runner(config_path, tmp / "out")
    return config_path, tmp / "out" / "evidence.json"


class TestAuditReplay(unittest.TestCase):

    def test_replay_bit_exact(self):
        api = AuditAPI()
        config_path, evidence_path = _fixture()
        report = api.replay(_toy_runner, config_path, evidence_path,
                            Path(tempfile.mkdtemp()))
        self.assertTrue(report.bit_exact)
        self.assertTrue(report.equal_excluding_timing)
        self.assertEqual(report.diffs, [])

    def test_replay_detects_tampered_config(self):
        api = AuditAPI()
        config_path, evidence_path = _fixture()
        config_path.write_text(json.dumps({"value": 7}), encoding="utf-8")
        report = api.replay(_toy_runner, config_path, evidence_path,
                            Path(tempfile.mkdtemp()))
        self.assertFalse(report.bit_exact)
        self.assertIn("configuration_hash", report.diffs)

    def test_timing_fields_never_hash(self):
        base = {"milestone": "TOY", "value": 1, "runtime_seconds": 3.5}
        self.assertEqual(evidence_content_hash(base),
                         evidence_content_hash({**base,
                                                "runtime_seconds": 99.0}))

    def test_int_keyed_dicts_compare_in_json_shape(self):
        api = AuditAPI()

        def runner(config_path, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            evidence = {"milestone": "TOY",
                        "histogram": {1: 2, 3: 4},
                        "runtime_seconds": 0.5}
            write_canonical_json(output_dir / "evidence.json", evidence)
            return evidence

        tmp = Path(tempfile.mkdtemp())
        config_path = tmp / "config.json"
        config_path.write_text(json.dumps({"value": 1}), encoding="utf-8")
        runner(config_path, tmp / "out")
        report = api.replay(runner, config_path,
                            tmp / "out" / "evidence.json",
                            Path(tempfile.mkdtemp()))
        self.assertTrue(report.bit_exact, report.diffs)
        self.assertTrue(report.equal_excluding_timing, report.diffs)

    def test_scratch_dir_must_be_empty(self):
        api = AuditAPI()
        config_path, evidence_path = _fixture()
        dirty = Path(tempfile.mkdtemp())
        (dirty / "x.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            api.replay(_toy_runner, config_path, evidence_path, dirty)


class TestAuditProvenance(unittest.TestCase):

    def test_provenance_chain(self):
        api = AuditAPI()
        config_path, _evidence = _fixture()
        report = api.provenance(config_path.parent / "out")
        self.assertIn("configuration_hash", report.chain["configuration"])
        self.assertIn("evidence_content_hash", report.chain["behavior"])
        self.assertTrue(any("artifact_index" in g for g in report.gaps))


if __name__ == "__main__":
    unittest.main()
