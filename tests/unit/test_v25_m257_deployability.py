"""Unit tests for M257: packaging, the CLI, and the hello-world
example (the deployability tranche).
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

import geode
from geode.cli import main as cli_main

REPO = Path(__file__).resolve().parents[2]


def _cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(argv)
    return code, buf.getvalue()


class TestM257Packaging(unittest.TestCase):

    def test_pyproject_parses_and_version_matches(self):
        data = tomllib.loads((REPO / "pyproject.toml").read_text(
            encoding="utf-8"))
        project = data["project"]
        self.assertEqual(project["name"], "geode-ml")
        # the entry point resolves to the shipped module
        self.assertEqual(project["scripts"]["geode"], "geode.cli:main")
        # version is single-sourced from the package attribute
        self.assertEqual(
            data["tool"]["setuptools"]["dynamic"]["version"],
            {"attr": "geode.__version__"})
        self.assertRegex(geode.__version__, r"^\d+\.\d+\.\d+$")

    def test_fingerprint_asset_is_package_data(self):
        asset = (REPO / "geode" / "core" / "assets"
                 / "fingerprint_v1.pt")
        self.assertTrue(asset.exists())
        data = tomllib.loads((REPO / "pyproject.toml").read_text(
            encoding="utf-8"))
        pkg_data = data["tool"]["setuptools"]["package-data"]
        self.assertEqual(pkg_data["geode.core"], ["assets/*"])


class TestM257Cli(unittest.TestCase):

    def test_version(self):
        code, out = _cli(["version"])
        self.assertEqual(code, 0)
        self.assertIn(geode.__version__, out)

    def test_route_demo(self):
        code, out = _cli(["route", "--fp", "0.9,0.3,0.2,0.1"])
        self.assertEqual(code, 0)
        self.assertIn("demo_general", out)
        self.assertIn("ranked_by=task", out)

    def test_route_with_safety_tags_abstains(self):
        # the demo arm is unvetted: a refusal-flagged task must
        # abstain (hard constraint, the M241 semantics)
        code, out = _cli(["route", "--fp", "0.9,0.3,0.2,0.1",
                          "--tags", "refusal"])
        self.assertEqual(code, 1)
        self.assertIn("empty route", out)

    def test_freeze_command(self):
        code, out = _cli(["freeze", "--attest", "v1,v2", "--ttl", "100",
                          "--reason", "drill"])
        self.assertEqual(code, 0)
        self.assertIn("effective until ledger index 100", out)

    def test_override_command_and_rejection(self):
        code, out = _cli(["override", "--actor", "op", "--action",
                          "kill_switch", "--justification", "drill",
                          "--counterfactual", '{"would_have": "x"}'])
        self.assertEqual(code, 0)
        self.assertIn("override recorded", out)
        code, _ = _cli(["override", "--actor", "op", "--action",
                        "kill_switch", "--justification", "   ",
                        "--counterfactual", '{"would_have": "x"}'])
        self.assertEqual(code, 2)  # blank justification rejected

    def test_verify_command_on_sealed_evidence(self):
        evidence = (REPO / "logs" / "results" / "v25"
                    / "m240_query_latency")
        if not (evidence / "evidence.json").exists():
            self.skipTest("sealed evidence not present locally")
        code, out = _cli(["verify", "--evidence", str(evidence)])
        self.assertEqual(code, 0)
        self.assertIn("evidence_content_hash", out)


class TestM257HelloWorld(unittest.TestCase):

    def test_hello_geode_runs_end_to_end(self):
        proc = subprocess.run(
            [sys.executable, str(REPO / "examples" / "hello_geode.py")],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout
        for needle in ["registered:", "routed:", "out-of-distribution",
                       "while frozen", "override recorded",
                       "chain verifies: True"]:
            self.assertIn(needle, out)


if __name__ == "__main__":
    unittest.main()
