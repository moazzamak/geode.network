"""M268 cell 5 — the remaining primitive tier (registered cell d):
z3 SAT solving, CNF rewriting with an exact truth-preservation
certificate, scipy analysis cross-checked against sympy's exact
values.

Registered and dispatched 22 Aug 2026 (plan v25, amendment 38).
CPU-only, deterministic — pure exactness certificates, no LLM, no
GPU. The registered float32 lesson applied: guarantees live in exact
arithmetic and exhaustive checks, never in tolerances (the single
tolerance used is for scipy's floating-point quadrature/ODE against
a closed-form value, registered before the run).

Honest scope notes: the SAT batch is planted-satisfiable only —
guaranteed-UNSAT instances would need an independent refutation and
are deferred; the CNF transform is variable-preserving (standard
distributivity, no Tseitin variables); Lean-class kernel stays
deferred (toolchain).

Evidence: logs/results/v25/m268_routing_study/evidence_primitives.json.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

from experiments.common.data_cache import data_cache_root
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m268_routing_study")

REGISTERED = {
    "sat": {"n": 100, "seed": 20260822, "n_vars": [6, 10],
            "clause_ratio": 3.5},
    "cnf": {"n": 100, "seed": 20260823, "depth": [2, 3],
            "n_literals": [3, 6]},
    "scipy": {
        "integral": {"expr": "x**2", "lo": 0.0, "hi": 1.0,
                     "exact": "1/3", "abs_tolerance": 1e-9},
        "ode": {"f": "y", "y0": 1.0, "t_span": [0.0, 1.0],
                "exact_at_1": 2.718281828459045, "abs_tolerance": 1e-6},
    },
}


# ------------------------------ CNF -----------------------------------

class _BNode:
    __slots__ = ("op", "leaf", "kids")

    def __init__(self, op: str | None, leaf: str | None,
                 kids: list["_BNode"] | None = None):
        self.op, self.leaf, self.kids = op, leaf, kids or []


def _gen_formula(rng: random.Random, depth: int, n_lit: int) -> _BNode:
    letters = [chr(ord("A") + i) for i in range(n_lit)]
    if depth <= 0:
        leaf = rng.choice(letters)
        if rng.random() < 0.3:
            return _BNode("not", None, [_BNode(None, leaf, None)])
        return _BNode(None, leaf, None)
    op = rng.choice(["and", "or", "not"])
    if op == "not":
        return _BNode(op, None, [_gen_formula(rng, depth - 1, n_lit)])
    return _BNode(op, None, [_gen_formula(rng, depth - 1, n_lit),
                             _gen_formula(rng, depth - 1, n_lit)])


def _render(node: _BNode) -> str:
    if node.leaf:
        return node.leaf
    if node.op == "not":
        return f"(not {_render(node.kids[0])})"
    return ("(" + f" {node.op} ".join(_render(k) for k in node.kids)
            + ")")


def _eval(node: _BNode, assign: dict[str, bool]) -> bool:
    if node.leaf:
        return assign[node.leaf]
    if node.op == "not":
        return not _eval(node.kids[0], assign)
    a, b = _eval(node.kids[0], assign), _eval(node.kids[1], assign)
    return a and b if node.op == "and" else a or b


def _clause_eval(clause: list[tuple[str, bool]], assign: dict[str, bool]
                 ) -> bool:
    return any((assign[v] if sign else not assign[v])
               for v, sign in clause)


def _cnf_transform(node: _BNode) -> list[list[tuple[str, bool]]]:
    """Variable-preserving CNF via standard distributivity + De Morgan.
    Returns a list of clauses; each clause is a list of (var, sign)."""
    def to_nnf(n: _BNode) -> Any:  # returns ('lit', var, sign) | ('and', l, r) | ('or', l, r)
        if n.leaf:
            return ("lit", n.leaf, True)
        if n.op == "not":
            child = n.kids[0]
            if child.leaf:
                return ("lit", child.leaf, False)
            if child.op == "not":
                return to_nnf(child.kids[0])
            if child.op == "and":
                return ("or", to_nnf(_BNode("not", None, [child.kids[0]])),
                        to_nnf(_BNode("not", None, [child.kids[1]])))
            return ("and", to_nnf(_BNode("not", None, [child.kids[0]])),
                    to_nnf(_BNode("not", None, [child.kids[1]])))
        return (n.op, to_nnf(n.kids[0]), to_nnf(n.kids[1]))

    def clauses(f: Any) -> list[list[tuple[str, bool]]]:
        if f[0] == "lit":
            return [[(f[1], f[2])]]
        if f[0] == "and":
            return clauses(f[1]) + clauses(f[2])
        left, right = clauses(f[1]), clauses(f[2])
        out = []
        for a in left:
            for b in right:
                out.append(a + b)
        return out

    return clauses(to_nnf(node))


def _render_cnf(clauses: list[list[tuple[str, bool]]]) -> str:
    def one(c: list[tuple[str, bool]]) -> str:
        return "(" + " or ".join(("" if s else "not ") + v
                                 for v, s in c) + ")"
    return " and ".join(one(c) for c in clauses)


def run_m268_cell5(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    import sympy
    from scipy import integrate
    from scipy.integrate import solve_ivp
    import z3

    # ---- 1. z3 planted-SAT -------------------------------------------
    sat_cfg = REGISTERED["sat"]
    rng = random.Random(sat_cfg["seed"])
    sat_records: list[dict[str, Any]] = []
    sat_failures: list[dict[str, Any]] = []
    for i in range(sat_cfg["n"]):
        n_vars = rng.randint(*sat_cfg["n_vars"])
        names = [f"v{j}" for j in range(n_vars)]
        assign = {v: bool(rng.randint(0, 1)) for v in names}
        n_clauses = int(sat_cfg["clause_ratio"] * n_vars)
        clauses = []
        for _ in range(n_clauses):
            chosen = rng.sample(names, 3)
            signs = [rng.random() < 0.5 for _ in range(3)]
            lit = (chosen, signs)
            if not any((assign[v] if s else not assign[v])
                       for v, s in zip(*lit)):
                # force consistency with the planted assignment
                signs[0] = assign[chosen[0]]
            clauses.append((chosen, signs))
        # z3 solve
        vs = {v: z3.Bool(v) for v in names}
        solver = z3.Solver()
        for chosen, signs in clauses:
            solver.add(z3.Or(*[(vs[v] if s else z3.Not(vs[v]))
                               for v, s in zip(chosen, signs)]))
        result = solver.check()
        sat = str(result) == "sat"
        model = {v: bool(z3.is_true(solver.model()[vs[v]]))
                 for v in names} if sat else {}
        verified = sat and all(
            any((model[v] if s else not model[v])
                for v, s in zip(chosen, signs))
            for chosen, signs in clauses)
        sat_records.append({
            "index": i, "n_vars": n_vars, "n_clauses": n_clauses,
            "satisfiable": sat, "model_verified": verified,
        })
        if not (sat and verified):
            sat_failures.append(sat_records[-1])

    # ---- 2. CNF rewriting + exact certificate ------------------------
    cnf_cfg = REGISTERED["cnf"]
    rng = random.Random(cnf_cfg["seed"])
    cnf_records: list[dict[str, Any]] = []
    cnf_failures: list[dict[str, Any]] = []
    for i in range(cnf_cfg["n"]):
        depth = rng.randint(*cnf_cfg["depth"])
        n_lit = rng.randint(*cnf_cfg["n_literals"])
        formula = _gen_formula(rng, depth, n_lit)
        rendered = _render(formula)
        cnfs = _cnf_transform(formula)
        cnfs_str = _render_cnf(cnfs)
        letters = sorted({v for c in cnfs for v, _ in c})
        exact = True
        n_assignments = 0
        for bits in itertools.product([False, True], repeat=len(letters)):
            assign = dict(zip(letters, bits))
            n_assignments += 1
            if _eval(formula, assign) != all(
                    _clause_eval(c, assign) for c in cnfs):
                exact = False
                break
        cnf_records.append({
            "index": i, "formula": rendered, "cnf": cnfs_str,
            "n_literals": len(letters),
            "n_clauses": len(cnfs),
            "truth_preserved_exact": exact,
            "assignments_checked": n_assignments,
        })
        if not exact:
            cnf_failures.append(cnf_records[-1])

    # ---- 3. scipy vs sympy exact cross-checks ------------------------
    sc_cfg = REGISTERED["scipy"]
    x = sympy.Symbol("x")
    exact_third = sympy.Rational(1, 3)
    quad_val, _err = integrate.quad(lambda t: t ** 2, 0.0, 1.0)
    integral_ok = abs(float(exact_third) - quad_val) <= sc_cfg["integral"][
        "abs_tolerance"]
    e_exact = sympy.E
    sol = solve_ivp(lambda t, y: y, [0.0, 1.0], [1.0], rtol=1e-9,
                    atol=1e-12)
    ode_val = float(sol.y[0][-1])
    ode_ok = abs(float(e_exact) - ode_val) <= sc_cfg["ode"][
        "abs_tolerance"]
    scipy_record = {
        "integral": {"quad": quad_val, "exact": float(exact_third),
                     "tolerance": sc_cfg["integral"]["abs_tolerance"],
                     "within_tolerance": integral_ok},
        "ode": {"solve_ivp": ode_val, "exact": float(e_exact),
                "tolerance": sc_cfg["ode"]["abs_tolerance"],
                "within_tolerance": ode_ok},
        "parameter_hash": hashlib.sha256(
            json.dumps(REGISTERED["scipy"], sort_keys=True).encode()
        ).hexdigest()[:16],
    }

    verdicts = {
        "sat": ("EXACT" if not sat_failures else "FAILED"),
        "cnf": ("EXACT" if not cnf_failures else "FAILED"),
        "scipy": ("WITHIN_TOLERANCE" if (integral_ok and ode_ok)
                  else "OUT_OF_TOLERANCE"),
    }
    evidence: dict[str, Any] = {
        "milestone": "M268",
        "cell": "cell 5 — remaining primitive tier (z3 / CNF / scipy)",
        "admissible_as_evidence": True,
        "smoke": False,
        "registered_parameters": REGISTERED,
        "verdicts": verdicts,
        "sat": {"records": sat_records, "n_failures": len(sat_failures)},
        "cnf": {"records": cnf_records, "n_failures": len(cnf_failures)},
        "scipy": scipy_record,
        "scope_note": ("planted-SAT only (guaranteed-UNSAT deferred — "
                       "independent refutation needed); CNF transform "
                       "variable-preserving (no Tseitin variables); "
                       "Lean-class kernel deferred (toolchain)"),
        "license_recorded": "z3 MIT; scipy BSD; sympy BSD",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence_primitives.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"verdicts": verdicts,
                      "sat_failures": len(sat_failures),
                      "cnf_failures": len(cnf_failures),
                      "scipy": {"integral_ok": integral_ok,
                                "ode_ok": ode_ok}}, indent=1), flush=True)
    print(f"M268 cell 5 complete -> "
          f"{output_dir / 'evidence_primitives.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m268_cell5(args.output)


if __name__ == "__main__":
    main()
