"""Documentation-integrity tests: the sealed plan must exist, be
non-empty, and carry the execution log. (20 Aug 2026: the plan file
was once silently truncated to zero bytes by an external
editor/formatter race and committed empty — this test makes that
impossible to miss.)
"""
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_V25 = REPO_ROOT / "analysis" / "RESEARCH_IMPLEMENTATION_PLAN_v25.md"
REQUIRED_HEADERS = ("EXECUTION LOG", "QUEUE STATUS")


class TestPlanIntegrity(unittest.TestCase):
    def test_plan_exists_and_is_substantial(self):
        self.assertTrue(PLAN_V25.exists(), "the v25 plan file is gone")
        text = PLAN_V25.read_text(encoding="utf-8")
        self.assertGreater(len(text), 50_000,
                           "the v25 plan looks truncated "
                           f"({len(text)} bytes)")
        self.assertIn("SEALED", text,
                      "the v25 plan has no sealed entries")

    def test_execution_log_headers_present(self):
        text = PLAN_V25.read_text(encoding="utf-8")
        for header in REQUIRED_HEADERS:
            self.assertIn(header, text,
                          f"missing {header!r} in the v25 plan")

    def test_recovery_note_present(self):
        text = PLAN_V25.read_text(encoding="utf-8")
        self.assertIn("RECOVERY NOTE", text,
                      "the recovery note is missing — the file may "
                      "have been restored without it")


if __name__ == "__main__":
    unittest.main()
