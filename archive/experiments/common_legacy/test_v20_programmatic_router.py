"""Unit tests for the programmatic-primitive + contract-router library (v20 B1/B2).

These are library behaviour tests for the ENGINEERING track
(``analysis/ENGINEERING_PLAN_v20.md``): no corpus, no measurement, no claims.
"""

import unittest

import numpy as np

from src.contract_router import ContractGatedRouter, RouteResult
from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.programmatic_primitive import (
    COST_ORDER,
    OutOfContractError,
    PrimitiveContract,
    ProgrammaticPrimitive,
)


def _fp(task: str, out_type: str = "labels", classes: tuple = (0, 1)) -> ModelFingerprint:
    return ModelFingerprint(
        task_name=task,
        input_spec=InputSpec(source="passthrough", dim=-1),
        output_spec=OutputSpec(type=out_type, classes=classes),
    )


class ContractGuardTests(unittest.TestCase):
    def test_ndim_guard(self):
        contract = PrimitiveContract(ndim=2)
        ok, reason = contract.accepts(np.zeros((4, 3)))
        self.assertTrue(ok)
        ok, reason = contract.accepts(np.zeros((4,)))
        self.assertFalse(ok)
        self.assertIn("ndim", reason)

    def test_shape_guard_with_wildcards(self):
        contract = PrimitiveContract(shape=(None, 1280))
        self.assertTrue(contract.accepts(np.zeros((7, 1280)))[0])
        self.assertFalse(contract.accepts(np.zeros((7, 64)))[0])

    def test_dtype_guard(self):
        contract = PrimitiveContract(dtype="float64")
        self.assertTrue(contract.accepts(np.zeros((3, 3), dtype=np.float64))[0])
        self.assertFalse(contract.accepts(np.zeros((3, 3), dtype=np.int32))[0])

    def test_finite_guard(self):
        contract = PrimitiveContract(require_finite=True)
        self.assertFalse(contract.accepts(np.array([[1.0, np.nan]]))[0])
        self.assertFalse(contract.accepts(np.array([[1.0, np.inf]]))[0])

    def test_value_range_guard(self):
        contract = PrimitiveContract(value_range=(0.0, 1.0))
        self.assertTrue(contract.accepts(np.array([[0.2, 0.9]]))[0])
        self.assertFalse(contract.accepts(np.array([[0.2, 1.5]]))[0])

    def test_domain_guards_are_row_wise(self):
        # A (N, d) batch must be judged per row: a 2x3 simplex batch has row
        # sums of 1.0 but a whole-array sum of 2.0.
        simplex = PrimitiveContract(domain="unit_simplex")
        ok, _ = simplex.accepts(np.array([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]]))
        self.assertTrue(ok)
        ok, reason = simplex.accepts(np.array([[0.2, 0.3, 0.5], [5.0, -3.0, 2.0]]))
        self.assertFalse(ok)

        ball = PrimitiveContract(domain="unit_ball")
        ok, _ = ball.accepts(np.array([[0.5, 0.5], [0.1, 0.1]]))
        self.assertTrue(ok)
        ok, _ = ball.accepts(np.array([[0.5, 0.5], [2.0, 0.0]]))
        self.assertFalse(ok)

    def test_negated_contract_is_reject_gate(self):
        # A negated contract accepts exactly the OUT-of-contract inputs, so a
        # primitive can act as a cheap reject gate (B3 semantics).
        in_contract = PrimitiveContract(
            ndim=2, shape=(None, 3), dtype="float64", require_finite=True
        )
        gate = PrimitiveContract(
            ndim=2, shape=(None, 3), dtype="float64", require_finite=True,
            negate=True,
        )
        ok, _ = gate.accepts(np.array([[1.0, np.nan, 0.0]]))   # non-finite -> reject
        self.assertTrue(ok)
        ok, _ = gate.accepts(np.array([[0.0, 1.0, 0.0]]))      # in contract -> pass
        self.assertFalse(ok)
        ok, _ = gate.accepts(np.zeros((2, 5)))                 # wrong shape -> reject
        self.assertTrue(ok)
        # serialization preserves negate
        rebuilt = PrimitiveContract.from_dict(gate.to_dict())
        self.assertTrue(rebuilt.negate)
        ok, _ = rebuilt.accepts(np.array([[1.0, np.nan, 0.0]]))
        self.assertTrue(ok)


class PrimitiveBehaviourTests(unittest.TestCase):
    def test_predict_gates_on_contract(self):
        primitive = ProgrammaticPrimitive(
            fingerprint=_fp("contract_check"),
            fn=lambda a: ProgrammaticPrimitive.check(a, 0.0, 1.0),
            contract=PrimitiveContract(ndim=2, require_finite=True, domain="unit_simplex"),
            cost_class="constant",
        )
        result = primitive.predict(np.array([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]]))
        self.assertEqual(result.shape, (2,))
        with self.assertRaises(OutOfContractError):
            primitive.predict(np.array([[5.0, -3.0, 2.0]]))

    def test_predict_validates_output_shape(self):
        # fn emits (N,) but the fingerprint declares a 2-class score output:
        # the boundary check must catch the contract violation.
        primitive = ProgrammaticPrimitive(
            fingerprint=_fp("bad", out_type="sdf_scores"),
            fn=lambda a: np.zeros(a.shape[0], dtype=np.int64),
            contract=PrimitiveContract(ndim=2),
            cost_class="constant",
        )
        with self.assertRaises(OutOfContractError):
            primitive.predict(np.zeros((4, 3)))

    def test_zero_learned_parameters(self):
        # The primitive's entire state is its fingerprint + contract + fn ref:
        # to_dict carries no weights, only a fn_repr.
        primitive = ProgrammaticPrimitive(
            fingerprint=_fp("contract_check"),
            fn=lambda a: ProgrammaticPrimitive.check(a, 0.0, 1.0),
            contract=PrimitiveContract(ndim=2),
            cost_class="constant",
        )
        payload = primitive.to_dict()
        self.assertNotIn("weights", payload)
        self.assertNotIn("parameters", payload)
        self.assertIn("fn_repr", payload)

    def test_serialization_round_trip(self):
        fn = lambda a: ProgrammaticPrimitive.check(a, 0.0, 1.0)  # noqa: E731
        primitive = ProgrammaticPrimitive(
            fingerprint=_fp("contract_check"),
            fn=fn,
            contract=PrimitiveContract(ndim=2, domain="unit_simplex"),
            cost_class="constant",
        )
        rebuilt = ProgrammaticPrimitive.from_dict(primitive.to_dict(), fn=fn)
        self.assertEqual(
            rebuilt.fingerprint.signature, primitive.fingerprint.signature
        )
        ok, _ = rebuilt.accepts(np.array([[0.2, 0.3, 0.5]]))
        self.assertTrue(ok)

    def test_cost_order(self):
        self.assertLess(COST_ORDER["constant"], COST_ORDER["linear"])
        self.assertLess(COST_ORDER["linear"], COST_ORDER["learned"])


class RouterBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.check = ProgrammaticPrimitive(
            fingerprint=_fp("contract_check"),
            fn=lambda a: ProgrammaticPrimitive.check(a, 0.0, 1.0),
            contract=PrimitiveContract(ndim=2, require_finite=True, domain="unit_simplex"),
            cost_class="constant",
        )
        self.norm = ProgrammaticPrimitive(
            fingerprint=_fp("l2_norm"),
            fn=lambda a: (np.linalg.norm(a, axis=1) <= 1.0).astype(np.int64),
            contract=PrimitiveContract(ndim=2, require_finite=True),
            cost_class="linear",
        )

    def test_dispatch_to_cheapest_accepting(self):
        router = ContractGatedRouter(programmatic=[self.check, self.norm])
        in_simplex = np.array([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]])
        result = router.route(in_simplex, task_name="contract_check")
        self.assertIsInstance(result, RouteResult)
        self.assertTrue(result.accepted)
        self.assertFalse(result.rejected)
        self.assertEqual(result.cost_class, "constant")
        self.assertIsNotNone(result.predictions)
        self.assertTrue(len(result.decision_log) >= 2)

    def test_task_and_output_filtering(self):
        router = ContractGatedRouter(programmatic=[self.check, self.norm])
        result = router.route(np.zeros((2, 3)), task_name="does_not_exist")
        self.assertTrue(result.rejected)
        result = router.route(np.zeros((2, 3)), output_type="sdf_scores")
        self.assertTrue(result.rejected)

    def test_out_of_contract_reject(self):
        router = ContractGatedRouter(programmatic=[self.check])
        out = np.array([[5.0, -3.0, 2.0]])
        result = router.route(out, task_name="contract_check")
        self.assertTrue(result.rejected)
        self.assertIsNone(result.predictions)
        self.assertFalse(result.fallback)
        self.assertIn("out of contract", result.reason)

    def test_fallback_path(self):
        class FakeLearned:
            fingerprint = _fp("contract_check")

            def predict(self, array):
                return np.zeros(array.shape[0], dtype=np.int64)

        router = ContractGatedRouter(
            programmatic=[self.check],
            fallback={"contract_check": FakeLearned()},
            enable_fallback=True,
        )
        out = np.array([[5.0, -3.0, 2.0]])
        result = router.route(out, task_name="contract_check")
        self.assertFalse(result.rejected)
        self.assertTrue(result.fallback)
        self.assertEqual(result.cost_class, "learned")
        self.assertEqual(result.predictions.shape, (1,))

    def test_fallback_disabled(self):
        class FakeLearned:
            fingerprint = _fp("contract_check")

            def predict(self, array):
                return np.zeros(array.shape[0], dtype=np.int64)

        router = ContractGatedRouter(
            programmatic=[self.check],
            fallback={"contract_check": FakeLearned()},
            enable_fallback=False,
        )
        result = router.route(np.array([[5.0, -3.0, 2.0]]), task_name="contract_check")
        self.assertTrue(result.rejected)
        self.assertFalse(result.fallback)

    def test_duplicate_primitive_rejected(self):
        with self.assertRaises(ValueError):
            ContractGatedRouter(programmatic=[self.check, self.check])

    def test_decision_log_shape(self):
        router = ContractGatedRouter(programmatic=[self.check])
        result = router.route(np.array([[0.2, 0.3, 0.5]]), task_name="contract_check")
        self.assertTrue(result.decision_log)
        entry = result.decision_log[0]
        for key in ("primitive_id", "cost_class", "contract_ok", "reason"):
            self.assertIn(key, entry)


if __name__ == "__main__":
    unittest.main()
