"""Unit tests for the programmatic memory library (v20 B4a).

Library behaviour tests for the ENGINEERING track
(``analysis/ENGINEERING_PLAN_v20.md``): no corpus, no measurement, no claims.
"""

import unittest

from src.programmatic_memory import Continuation, ProgrammaticMemory


class ProgrammaticMemoryTests(unittest.TestCase):
    def test_register_and_exact_counts(self):
        memory = ProgrammaticMemory(window=3)
        memory.register(["a", "b", "a", "b", "a"])
        # context "a" is followed by "b" at positions 1 and 3, never by "a"
        result = memory.continuations(["a"])
        self.assertEqual(result.matched_length, 1)
        self.assertEqual(result.as_dict(), {"b": 2})
        # context "b" is followed by "a" twice
        result = memory.continuations(["b"])
        self.assertEqual(result.as_dict(), {"a": 2})

    def test_longest_suffix_backoff(self):
        memory = ProgrammaticMemory(window=3)
        memory.register(["a", "b", "c", "d", "b", "c", "e"])
        # full context (x,b,c) never observed as a 3-gram; (b,c) was: b,c->d,e
        result = memory.continuations(["x", "b", "c"])
        self.assertEqual(result.matched_length, 2)
        self.assertEqual(result.as_dict(), {"d": 1, "e": 1})

    def test_novel_context_returns_empty(self):
        memory = ProgrammaticMemory(window=2)
        memory.register(["x", "y", "x"])
        result = memory.continuations(["z", "q"])
        self.assertEqual(result.matched_length, 0)
        self.assertEqual(result.as_dict(), {})

    def test_window_dial_limits_lookback(self):
        memory = ProgrammaticMemory(window=3)
        memory.register(["a", "b", "c", "d", "b", "c", "d"])
        # full 3-gram (a,b,c) exists; dialling window down to 2 uses (b,c)
        full = memory.continuations(["a", "b", "c"])
        self.assertEqual(full.matched_length, 3)
        dialled = memory.continuations(["a", "b", "c"], window=2)
        self.assertEqual(dialled.matched_length, 2)
        self.assertEqual(dialled.as_dict(), {"d": 2})

    def test_predict_next_smoothing(self):
        memory = ProgrammaticMemory(window=2)
        memory.register(["a", "b", "a"])
        vocab = ["a", "b", "c"]
        raw = memory.predict_next(["a"])
        self.assertEqual(raw.as_dict(), {"b": 1})
        smoothed = memory.predict_next(["a"], alpha=1.0, vocabulary=vocab)
        self.assertEqual(smoothed.as_dict(), {"a": 1, "b": 2, "c": 1})
        # smoothing a novel context assigns alpha to every vocab token
        novel = memory.predict_next(["z", "q"], alpha=1.0, vocabulary=vocab)
        self.assertEqual(novel.as_dict(), {"a": 1, "b": 1, "c": 1})
        with self.assertRaises(ValueError):
            memory.predict_next(["a"], alpha=1.0)  # alpha without vocabulary

    def test_matched_length_histogram(self):
        memory = ProgrammaticMemory(window=2)
        memory.register(["a", "b", "a", "b", "a", "b"])
        histogram = memory.matched_length_histogram(["a", "b", "a", "b"])
        self.assertEqual(histogram, {0: 1, 1: 1, 2: 2})

    def test_footprint_and_entry_count(self):
        memory = ProgrammaticMemory(window=3)
        self.assertEqual(memory.entry_count, 0)
        memory.register(["a", "b", "a"])
        # entries: (a)->b, (b)->a, (a,b)->a = 3
        self.assertEqual(memory.entry_count, 3)
        self.assertGreater(memory.footprint_bytes(), 0)
        payload = memory.to_dict()
        self.assertEqual(payload["window"], 3)
        self.assertEqual(payload["entry_count"], 3)

    def test_exact_continuations_no_backoff(self):
        memory = ProgrammaticMemory(window=3)
        memory.register(["a", "b", "c", "d", "b", "c", "e"])
        # exact 3-gram (a,b,c) was observed -> d
        result = memory.exact_continuations(["a", "b", "c"], order=3)
        self.assertEqual(result.matched_length, 3)
        self.assertEqual(result.as_dict(), {"d": 1})
        # exact 3-gram (x,b,c) was NOT observed -> empty, no backoff to (b,c)
        result = memory.exact_continuations(["x", "b", "c"], order=3)
        self.assertEqual(result.as_dict(), {})
        self.assertEqual(result.matched_length, 0)
        # order 0 / beyond window are empty
        self.assertEqual(memory.exact_continuations(["a"], order=0).as_dict(), {})
        with self.assertRaises(ValueError):
            memory.exact_continuations(["a"], order=-1)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            ProgrammaticMemory(window=0)

    def test_continuation_total(self):
        memory = ProgrammaticMemory(window=2)
        memory.register(["a", "b", "a", "b"])
        result = memory.continuations(["a"])
        self.assertEqual(result.total(), 2)


if __name__ == "__main__":
    unittest.main()
