"""M350 proof-cost table (28 Aug 2026).

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md`` under
G2, whose gate reads: "Publish a proof-cost table (prover seconds,
verifier seconds, proof bytes) per registered axis dimension,
measured on the reference CPU. The paper's proof section may make no
cost claim not present in that table."

What is MEASURED and what is not, kept separate so the two are never
confused downstream.

MEASURED on this CPU:
- one ed25519 variable-base scalar multiplication (libsodium via
  PyNaCl), net of a separately measured Python FFI call overhead;
- serving a head, a float64 ``d x C`` mat-vec through NumPy. float64
  because the registered numerics policy promotes the head's solve.

NOT usable, and why it is recorded rather than dropped: libsodium's
``crypto_core_ed25519_add`` operates on the 32-byte compressed
encoding, so it decompresses both points, adds, and recompresses.
Measured at 17.2 us against 139.6 us for a full scalar
multiplication -- a ratio of 8, where a genuine projective addition
should be roughly 300x cheaper. It measures two field square roots,
not a group addition, and modelling a multi-exponentiation with it
would have inflated every verifier figure by roughly forty times.
All costs below are therefore carried in scalar-multiplication
equivalents.

ANALYTIC, from Bunz et al. 2018 and standard curve arithmetic:
- proof size is 2*ceil(log2 n) + 13 group elements;
- a 255-bit variable-base scalar multiplication is about 255
  doublings plus ~51 windowed additions, so ~306 projective group
  operations;
- the prover performs O(n) group exponentiations, constant
  implementation-dependent: a naive 8n independent scalar
  multiplications, or ~6n elements pushed through
  multi-exponentiation;
- the verifier is a single multi-exponentiation of size ~2n;
- bucket (Pippenger) multi-exponentiation of size m costs about
  m*255/log2(m) projective additions.

The multi-exponentiation model is deliberately FAVOURABLE to the
proof system: the finding under test is that proving costs
catastrophically more than serving, and an estimate that flatters
the prover and still shows the gap is worth more than one that does
not. ``prover_over_serving_at_10x_faster`` reports the same ratio
under a hypothetical implementation ten times faster than this one.
"""
from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import nacl.bindings as sodium
import numpy as np

# (label, d, C) -- registered axis dimensions from the paper's
# measured table. n = d*C is the head's multiply-add count.
AXES: list[tuple[str, int, int]] = [
    ("Text (BERT-base, SST-2)", 768, 2),
    ("Text (BERT-base, MNLI-m)", 768, 3),
    ("Audio (wav2vec2-base, Speech Commands v2)", 768, 35),
    ("Image routing (DINOv2-L, DomainNet 345)", 1024, 345),
    ("Vision (DINOv2-L, Open Images 601)", 1024, 601),
]

SCALAR_BITS = 255
OPS_PER_SCALARMULT = 306     # 255 doublings + ~51 windowed additions
PROVER_NAIVE_COUNT = 8       # x n independent scalar multiplications
PROVER_MULTIEXP_COUNT = 6    # x n elements across all rounds
VERIFIER_MULTIEXP_COUNT = 2  # x n elements, one multi-exponentiation
GROUP_ELEMENT_BYTES = 32
HEADROOM_FACTOR = 10.0


def _bench(fn, iters: int) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def _measure_primitives() -> dict[str, float]:
    p = sodium.crypto_scalarmult_ed25519_base_noclamp(
        bytes([7]) + bytes(31))
    s = bytes(range(32))
    overhead = _bench(
        lambda: sodium.crypto_core_ed25519_scalar_add(s, s), 200_000)
    raw = _bench(
        lambda: sodium.crypto_scalarmult_ed25519_noclamp(s, p), 500)
    net = raw - overhead
    return {
        "ffi_call_overhead_seconds": overhead,
        "scalarmult_raw_seconds": raw,
        "scalarmult_seconds": net,
        "projective_op_seconds_derived": net / OPS_PER_SCALARMULT,
    }


def _time_serve(d: int, c: int, iters: int = 2000) -> float:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((d, c))
    z = rng.standard_normal(d)
    return _bench(lambda: w.T @ z, iters)


def _multiexp_scalarmult_equivalents(m: int) -> float:
    """Size-m multi-exponentiation in scalar multiplications, bucket
    method."""
    if m < 2:
        return float(m)
    adds = m * SCALAR_BITS / math.log2(m)
    return adds / OPS_PER_SCALARMULT


def main() -> int:
    prim = _measure_primitives()
    t_smul = prim["scalarmult_seconds"]
    print(f"measured: scalar mult {t_smul * 1e6:.1f} us net (FFI "
          f"overhead {prim['ffi_call_overhead_seconds'] * 1e6:.2f} us)")

    rows = []
    for label, d, c in AXES:
        n = d * c
        t_serve = _time_serve(d, c)
        prover = _multiexp_scalarmult_equivalents(
            PROVER_MULTIEXP_COUNT * n) * t_smul
        verifier = _multiexp_scalarmult_equivalents(
            VERIFIER_MULTIEXP_COUNT * n) * t_smul
        rows.append({
            "axis": label, "d": d, "C": c, "n": n,
            "serve_seconds_measured": t_serve,
            "prover_seconds_naive": PROVER_NAIVE_COUNT * n * t_smul,
            "prover_seconds_multiexp": prover,
            "verifier_seconds": verifier,
            "proof_bytes": ((2 * math.ceil(math.log2(n)) + 13)
                            * GROUP_ELEMENT_BYTES),
            "prover_over_serving": prover / t_serve,
            "verifier_over_serving": verifier / t_serve,
            "prover_over_serving_at_10x_faster":
                prover / HEADROOM_FACTOR / t_serve,
            # replaying the head IS serving it: the honest baseline
            "replay_over_serving": 1.0,
        })

    payload = {
        "milestone": "M350",
        "finding": "G2 -- per-answer proofs are off by orders of "
                   "magnitude",
        "registered_in": "analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md",
        "platform": platform.platform(),
        "measured_primitives": prim,
        "rejected_primitive": {
            "name": "crypto_core_ed25519_add",
            "reason": "operates on compressed encodings; "
                      "decompress+add+recompress, dominated by two "
                      "field square roots. Measured 17.2 us vs "
                      "139.6 us for a full scalar multiplication, a "
                      "ratio of 8 where ~306 is expected. Using it "
                      "would inflate verifier cost ~40x.",
        },
        "analytic_counts": {
            "proof_size_elements": "2*ceil(log2 n) + 13",
            "ops_per_scalarmult": OPS_PER_SCALARMULT,
            "prover_naive_scalarmults": "8n",
            "prover_multiexp_elements": "6n",
            "verifier_multiexp_elements": "2n",
            "multiexp_model": "bucket/Pippenger, m*255/log2(m) "
                              "projective additions",
            "source": "Bunz et al. 2018; constants are "
                      "implementation-dependent",
        },
        "axes": rows,
    }
    out = Path("analysis/m350_proof_cost_table.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'axis':<44}{'n':>8}{'serve':>11}{'prove':>10}"
          f"{'verify':>9}{'bytes':>7}{'prove/serve':>13}{'@10x':>11}")
    for r in rows:
        print(f"{r['axis']:<44}{r['n']:>8}"
              f"{r['serve_seconds_measured'] * 1e6:>9.1f}us"
              f"{r['prover_seconds_multiexp']:>9.2f}s"
              f"{r['verifier_seconds']:>8.3f}s"
              f"{r['proof_bytes']:>7}"
              f"{r['prover_over_serving']:>13.3g}"
              f"{r['prover_over_serving_at_10x_faster']:>11.3g}")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
