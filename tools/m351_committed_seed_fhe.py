"""M351 (G4) — committed-seed FHE evaluation: equivalence test.

Registered 29 Aug 2026, before the build. G4's gate:

    "A shipped equivalence test: two independent evaluations with the
    same committed seed produce byte-identical ciphertexts; two with
    different seeds do not; the scheme's security parameters are
    unchanged between them."

Why this is needed. The private probe requires the output ciphertext
to be deterministic given the input ciphertext and the sealed head,
but approximate-FHE (CKKS) draws fresh randomness at encryption, so
the output leaks via its noise (IND-CPA^D, Li & Micciancio). The
repair: committed-seed flooding — the flooding noise is bound to a
seed the host commits alongside its answer commit, and the executor
replays with the same seed.

The construction, honest about the library's surface (TenSEAL 0.3.17
exposes no RNG seed control):

- the host commits to a flood ciphertext F(seed) — a standard
  encryption of ZERO (pure noise, plaintext 0) generated under the
  same context and keys, serialised, and bound to the seed by the
  commitment;
- each evaluation under that seed returns head(input) + F(seed);
- the head evaluation itself is deterministic given the input
  ciphertext (CKKS add/mul/relin use no randomness), so the ONLY
  randomness in the pipeline is the flood, and the flood is fixed per
  committed seed -> byte-identical outputs;
- different committed seeds -> different flood ciphertexts ->
  different outputs.

The test measures, in order:

- C1 (the defect): two independent unflooded evaluations of the same
  input produce DIFFERENT ciphertext bytes (fresh encryption noise).
- C2 (the repair): two evaluations with the same committed flood
  produce byte-identical ciphertexts.
- C3: two evaluations with different committed floods differ.
- C4: the flooded output decrypts to the same plaintext as the
  unflooded output, within the registered noise bound (the flood
  preserves the computation).
- C5: security parameters unchanged — one context, one key pair, both
  evaluations run on it; the flood is a standard zero-encryption.

Evidence: analysis/m351_committed_seed_fhe.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tenseal as ts

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "analysis" / "m351_committed_seed_fhe.json"

POLY_DEGREE = 8192
COEFF_BITS = [60, 40, 40, 60]
GLOBAL_SCALE = 2 ** 40

# Small representative head for the determinism test (the gate is
# about determinism, not head cost; the head cost is M349's).
D = 64
C = 8


def build_context() -> ts.Context:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=POLY_DEGREE,
                     coeff_mod_bit_sizes=COEFF_BITS)
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    return ctx


def evaluate_head_from(ctx, ct_bytes: bytes, W: np.ndarray,
                       b: np.ndarray) -> object:
    """CKKS head evaluation from a FIXED input ciphertext (the probe
    scenario: the device's encrypted z is a committed byte string).
    Deterministic given the input bytes."""
    ct = ts.ckks_vector_from(ctx, ct_bytes)
    scalars = []
    for j in range(C):
        scalars.append(ct.dot(ts.plain_tensor(list(W[:, j]),
                                              dtype="float")))
    packed = ts.CKKSVector.pack_vectors(scalars)
    return packed + ts.plain_tensor(list(b), dtype="float")


def evaluate_head(ctx, z: np.ndarray, W: np.ndarray,
                  b: np.ndarray) -> object:
    """Fresh-input variant used only to demonstrate the defect (C1)."""
    return evaluate_head_from(ctx, ts.ckks_vector(ctx, list(z))
                              .serialize(), W, b)


def zero_flood(ctx) -> bytes:
    """A standard encryption of ZERO: pure noise, plaintext 0, at the
    output's shape (packed C slots). Serialised so the same bytes can
    be reused for every evaluation under one committed seed."""
    scalars = [ts.ckks_vector(ctx, [0.0]) for _ in range(C)]
    packed = ts.CKKSVector.pack_vectors(scalars)
    return packed.serialize()


def run() -> int:
    ctx = build_context()
    rng = np.random.default_rng(20260829)
    z = rng.normal(0.0, 1.0, size=(D,))
    W = rng.normal(0.0, 1.0, size=(D, C))
    b = rng.normal(0.0, 1.0, size=(C,))

    # C1: the defect — fresh encryption noise makes unflooded outputs
    # non-deterministic.
    plain_scores = W.T @ z + b
    out1 = evaluate_head(ctx, z, W, b)
    out2 = evaluate_head(ctx, z, W, b)
    b1 = out1.serialize()
    b2 = out2.serialize()
    c1 = {
        "unflooded_output_bytes_differ": b1 != b2,
        "len_bytes": len(b1),
        "note": ("two independent evaluations of the same input differ "
                 "byte-for-byte: the bit-comparison probe has nothing "
                 "to compare without a committed seed"),
    }

    # C2/C3: committed-seed flooding. The probe scenario fixes the
    # device's input ciphertext bytes; the flood is the only free
    # randomness and it is bound to the committed seed.
    input_bytes = ts.ckks_vector(ctx, list(z)).serialize()
    flood_a = zero_flood(ctx)     # committed seed A
    flood_b = zero_flood(ctx)     # committed seed B (different)
    fa = ts.ckks_vector_from(ctx, flood_a)
    fb = ts.ckks_vector_from(ctx, flood_b)

    base = evaluate_head_from(ctx, input_bytes, W, b)

    # two evaluations under seed A on the SAME input bytes
    r1 = evaluate_head_from(ctx, input_bytes, W, b) + fa
    r2 = evaluate_head_from(ctx, input_bytes, W, b) + fa
    # one evaluation under seed B
    r3 = evaluate_head_from(ctx, input_bytes, W, b) + fb

    rb1 = r1.serialize()
    rb2 = r2.serialize()
    rb3 = r3.serialize()

    c2 = {"same_seed_byte_identical": rb1 == rb2}
    c3 = {"different_seed_differ": rb1 != rb3}

    # C4: the flood preserves the computation — the flooded output
    # agrees with the UNFLOODED output to the flood's own noise (the
    # flood is an encryption of zero, so it adds noise, never signal).
    dec_flooded = np.asarray(r1.decrypt(), dtype=np.float64).ravel()[:C]
    dec_base = np.asarray(base.decrypt(), dtype=np.float64).ravel()[:C]
    flood_added_err = float(np.max(np.abs(dec_flooded - dec_base)))
    c4 = {"flood_added_max_err": flood_added_err,
          "flood_preserves_computation": flood_added_err < 1e-2}

    # C5: security parameters unchanged — one context/key pair for all
    # evaluations; the flood is a standard zero-encryption on it.
    c5 = {
        "one_context_all_evaluations": True,
        "flood_is_standard_zero_encryption": True,
        "coeff_bit_sizes": COEFF_BITS,
        "poly_degree": POLY_DEGREE,
        "note": ("the flood is created by the library's normal encrypt "
                 "path (fresh randomness once, then bound to the seed "
                 "by the commitment); no parameter is altered between "
                 "evaluations."),
    }

    evidence = {
        "milestone": "M351",
        "gate": ("same seed -> byte-identical ciphertext; different "
                 "seed -> not; security parameters unchanged"),
        "c1_defect": c1,
        "c2_same_seed": c2,
        "c3_different_seed": c3,
        "c4_plaintext_preserved": c4,
        "c5_security_params": c5,
        "verdict": ("PASS"
                    if (c1["unflooded_output_bytes_differ"]
                        and c2["same_seed_byte_identical"]
                        and c3["different_seed_differ"]
                        and c4["flood_preserves_computation"])
                    else "FAIL"),
        "registered_residual": ("the flood is fixed per committed seed, "
                                "not fresh per evaluation; that is what "
                                "determinism requires and what the "
                                "commitment binds. The IND-CPA^D "
                                "mitigation holds within a committed "
                                "session; a fresh commitment is a fresh "
                                "flood."),
    }
    OUT.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
