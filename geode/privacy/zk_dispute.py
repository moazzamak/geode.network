"""GEODE zk measurement disputes (v25 M259, the M256 cell 2
STRUCTURE half).

A challenger proves, in zero knowledge, that a claimed measurement
satisfies the linear relation y = W x + b over a committed input x —
the shipped M193 argument (Pedersen commitment + single-move
Fiat-Shamir). The dispute payload pairs the commitment with the
proof; ``verify_dispute_payload`` is the injected verifier for the
SlashLedger: a lying attester's claim FAILS verification and is
slashed, and a false accusation (both sides verify the same claim)
slashes the challenger — the registered adjudication rules.

LIVE deployment stays gated on the M254/M194 public-chain anchor
(the original registration); this cell ships the proof plumbing and
its integration, deterministically testable offline.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from geode.privacy import zk_linear as zl


def build_dispute_payload(x: Sequence[int], r: int,
                          w: Sequence[Sequence[int]],
                          b: Sequence[int],
                          y: Sequence[int]) -> dict[str, Any]:
    """The proof package for the claim y = W x + b over committed x."""
    commitment = zl.commit(list(x), int(r))
    proof = zl.prove(list(x), int(r), [list(row) for row in w],
                     list(b), list(y))
    return {
        "commitment": commitment,
        "proof": proof,
        "w": [list(row) for row in w],
        "b": list(b),
        "y": list(y),
    }


def verify_dispute_payload(payload: dict[str, Any],
                           _reference: Any = None) -> bool:
    """The injected verifier: the payload is valid iff its proof
    verifies against its own commitment, weights, bias, and claim."""
    try:
        return bool(zl.verify(
            payload["proof"], payload["commitment"],
            [list(row) for row in payload["w"]],
            list(payload["b"]), list(payload["y"])))
    except (KeyError, TypeError, ValueError):
        return False
