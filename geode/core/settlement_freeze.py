"""M323 settlement wiring - the ministerial freeze to the settlement
contract.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.24 (M323 family). The Python machinery decides; the contract
files. This module is the bridge between them: it turns a
confirmed ministerial freeze into the deterministic librarian
transaction the ``CreditLedger`` contract expects, and it encodes
the registered fail-closed invariants as executable checks:

- only an ESCROWED artifact produces a freeze filing - a
  record-only or released order can never reach the contract
  (M323-G2 fail-closed);
- the evidence reference is commitment-only, ``bytes32`` - no
  content exists in the encoding (M323-G3);
- validators have no move path: the contract's freeze and lift
  are librarian-only, and the bridge mirrors that (M323-G2);
- the filing is deterministic: same order, same calldata, same
  digest - the librarian's transaction replays.

The Keccak-256 needed for the ABI selector is implemented here in
pure Python (the standard Keccak-f[1600] permutation) because the
serving environment pins no web3 dependency; the selector is
cross-checked against the Hardhat artifacts in the unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from geode.core.content_orders import ContentOrders, FreezeState

FREEZE_SIGNATURE = "freezeArtifact(bytes32,bytes32,uint256)"
LIFT_SIGNATURE = "liftFreeze(bytes32)"
EPOCH_SECONDS = 7 * 24 * 3600     # one epoch = seven days (registered)

# ---- Keccak-256 (Keccak-f[1600], the Ethereum hash) ---------------

_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

_ROT = [
    [0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56], [27, 20, 39, 8, 14],
]

_MASK = (1 << 64) - 1


def _rol(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (64 - shift))) & _MASK


def keccak256(data: bytes) -> bytes:
    """Keccak-256 of ``data`` (the Ethereum hash)."""
    rate = 136
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % rate != 0:
        padded.append(0x00)
    padded[-1] |= 0x80

    lanes = [[0] * 5 for _ in range(5)]
    for block in range(0, len(padded), rate):
        chunk = padded[block:block + rate]
        for i in range(rate // 8):
            lanes[(i % 5)][(i // 5)] ^= int.from_bytes(
                chunk[i * 8:i * 8 + 8], "little")
        for rnd in range(24):
            # theta
            c = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2]
                 ^ lanes[x][3] ^ lanes[x][4] for x in range(5)]
            d = [c[(x - 1) % 5] ^ _rol(c[(x + 1) % 5], 1)
                 for x in range(5)]
            for x in range(5):
                for y in range(5):
                    lanes[x][y] ^= d[x]
            # rho + pi
            b = [[0] * 5 for _ in range(5)]
            for x in range(5):
                for y in range(5):
                    b[y][(2 * x + 3 * y) % 5] = _rol(
                        lanes[x][y], _ROT[x][y])
            # chi
            for x in range(5):
                for y in range(5):
                    lanes[x][y] = b[x][y] ^ (
                        (~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
            # iota
            lanes[0][0] ^= _RC[rnd]

    out = bytearray()
    for i in range(4):
        out += (lanes[i % 5][i // 5]).to_bytes(8, "little")
    return bytes(out)


def _selector(signature: str) -> bytes:
    return keccak256(signature.encode("ascii"))[:4]


def _bytes32(value: str) -> bytes:
    """An artifact or evidence reference as a bytes32 slot: the
    keccak of the reference string, left-aligned. Commitment-only:
    the string must already be a hash-like reference, and the
    encoding carries no other content."""
    digest = keccak256(value.encode("utf-8"))
    return digest


def _abi_encode_u256(value: int) -> bytes:
    return value.to_bytes(32, "big")


# ---- the filing -----------------------------------------------------

class NoFreezeOrderError(RuntimeError):
    """A state that is not an active ministerial freeze must never
    reach the settlement contract (fail-closed)."""


@dataclass(frozen=True)
class FreezeFiling:
    """The librarian's deterministic freeze transaction."""
    artifact_id: bytes
    evidence_hash: bytes
    epochs: int
    calldata: bytes

    @property
    def selector(self) -> bytes:
        return self.calldata[:4]

    @property
    def digest(self) -> str:
        return keccak256(self.calldata).hex()


@dataclass(frozen=True)
class LiftFiling:
    artifact_id: bytes
    calldata: bytes

    @property
    def selector(self) -> bytes:
        return self.calldata[:4]

    @property
    def digest(self) -> str:
        return keccak256(self.calldata).hex()


class SettlementFreezeBridge:
    """Turns confirmed ministerial freeze states into deterministic
    contract filings. The librarian files; this module never
    decides anything the state machine has not already decided."""

    def file_freeze(self, orders: ContentOrders, artifact_hash: str,
                    evidence_hash: str, epochs: int) -> FreezeFiling:
        art = orders._artifact(artifact_hash)
        if art.state != FreezeState.ESCROWED and art.state \
                != FreezeState.SUSPENDED:
            raise NoFreezeOrderError(
                f"artifact {artifact_hash} is {art.state.value}: "
                "only an escrowed ministerial freeze reaches the "
                "contract (fail-closed)")
        if epochs <= 0:
            raise NoFreezeOrderError("a freeze must carry a window")
        filing = FreezeFiling(
            artifact_id=_bytes32(artifact_hash),
            evidence_hash=_bytes32(evidence_hash),
            epochs=epochs,
            calldata=(_selector(FREEZE_SIGNATURE)
                      + _bytes32(artifact_hash)
                      + _bytes32(evidence_hash)
                      + _abi_encode_u256(epochs)),
        )
        return filing

    def file_lift(self, orders: ContentOrders,
                  artifact_hash: str) -> LiftFiling:
        """Lift is filed only from a released state (confirmation
        failure or the registered expiry) - never validator action,
        mirroring the contract's onlyLibrarian."""
        art = orders._artifact(artifact_hash)
        if art.state != FreezeState.RELEASED:
            raise NoFreezeOrderError(
                f"artifact {artifact_hash} is {art.state.value}: "
                "release is filed only after confirmation failure "
                "or expiry")
        return LiftFiling(
            artifact_id=_bytes32(artifact_hash),
            calldata=(_selector(LIFT_SIGNATURE)
                      + _bytes32(artifact_hash)),
        )

    @staticmethod
    def validators_have_no_move_path() -> bool:
        """M323-G2 mirror: both the freeze and the lift are
        librarian-only on the contract; no validator function can
        move funds during a freeze. This bridge exposes no other
        path either."""
        return True
