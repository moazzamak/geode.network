"""Unit tests for geode.hashing: the canonical JSON shape and the
payload hash (the product hashing primitives)."""
import json
import unittest
from pathlib import Path

import numpy as np

from geode.hashing import canonical_json, payload_hash


class TestCanonicalShape(unittest.TestCase):
    def test_sorted_keys_and_separators(self):
        self.assertEqual(canonical_json({"b": 1, "a": 2}),
                         '{"a":2,"b":1}')

    def test_int_dict_keys_stringified(self):
        self.assertEqual(canonical_json({1: "x"}), '{"1":"x"}')

    def test_nested_shapes(self):
        value = {"list": [1, (2, 3)], "tuple": (4, 5),
                 "none": None, "float": 0.5, "bool": True}
        self.assertEqual(canonical_json(value),
                         '{"bool":true,"float":0.5,"list":[1,[2,3]],'
                         '"none":null,"tuple":[4,5]}')

    def test_numpy_values_normalise(self):
        self.assertEqual(
            canonical_json({"arr": np.array([1, 2]),
                            "scalar": np.float64(0.5)}),
            '{"arr":[1,2],"scalar":0.5}')

    def test_path_normalises(self):
        path = Path("a") / "b"
        self.assertEqual(canonical_json({"p": path}),
                         json.dumps({"p": str(path)}, sort_keys=True,
                                    separators=(",", ":")))

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            canonical_json({"bad": object()})

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            canonical_json({"nan": float("nan")})


class TestPayloadHash(unittest.TestCase):
    def test_deterministic_and_order_free(self):
        a = payload_hash({"x": 1, "y": [2, 3]})
        b = payload_hash({"y": [2, 3], "x": 1})
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_content_sensitive(self):
        self.assertNotEqual(payload_hash({"x": 1}), payload_hash({"x": 2}))


if __name__ == "__main__":
    unittest.main()
