"""M363 (G17) — operations line: every working role gets an income line.

Registered 29 Aug 2026, before the build. G17's gate:

    "A cost-model run that closes: total operations revenue at the
    reference workload >= total measured operations cost, with each
    role's line itemized. If it does not close, the split is wrong
    and the paper says so rather than omitting the roles."

Reference workload (registered): the vision axis (DINOv2-L, 601
classes) at 10,000 sessions/epoch — the mid-scale row of M367's
traffic table. Inbox entries at the incorporation cap (8/epoch, the
librarian's maximum obligation). One attribution root per epoch. One
dispute replay per epoch. Proofs verified at the M350 sampled-batch
rate (1% of sessions).

Measured inputs (registered elsewhere, reused here):
- Librarian EVM gas, measured on the current contracts
  (scripts/measure_operations_gas.js): incorporate 163,187,
  postAttributionRoot 60,057. 8 incorporates + 1 root = 1,365,553.
- Anchor gas: a plain L1 anchor settlement, registered at 100,000.
- Batch verification: M350 measured vision-axis verify 7.721 s per
  proof (Pippenger model, scalar-mul equivalents).
- Dispute replay: one replay of the M375 chain cell (~10 s of
  compute, registered).
- Tally: threshold-opening one vote (~1 s, registered).
- Gateway: serving 10,000 sessions at the M350 measured 102 us/query
  on the vision axis = ~1 s of compute plus per-session bookkeeping,
  registered at 5 s-equivalents/epoch.

Registered prices (all stated, never implied):
- Gas price 0.1 gwei (conservative upper bound for Arbitrum One).
- ETH/USD 3,000.
- Compute 0.001 USD per CPU-core-second (3.60 USD/core-hour).

The fee is the variable. The model reports (a) whether the test-suite
BASE_FEE (10 wei) closes, and (b) the base fee that closes with the
registered margin.

Evidence: analysis/m363_operations_cost_model.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "analysis" / "m363_operations_cost_model.json"

# --- reference workload ---------------------------------------------
SESSIONS_PER_EPOCH = 10_000
ENTRIES_PER_EPOCH = 8            # inbox incorporation cap
ROOTS_PER_EPOCH = 1
DISPUTES_PER_EPOCH = 1
PROOFS_VERIFIED_PER_EPOCH = 100  # 1% sample of 10k sessions (M350)
ANCHORS_PER_EPOCH = 1

# --- measured per-unit costs ----------------------------------------
GAS_INCORPORATE = 163_187       # measured, current InclusionInbox
GAS_ROOT = 60_057               # measured, postAttributionRoot
GAS_ANCHOR = 100_000            # registered plain L1 anchor
VERIFY_S_PER_PROOF = 7.721      # measured M350, vision axis
REPLAY_S_PER_DISPUTE = 10.0     # registered (M375 chain cell scale)
TALLY_S_PER_VOTE = 1.0          # registered
GATEWAY_S_PER_EPOCH = 5.0       # registered bookkeeping equivalent

# --- registered prices ----------------------------------------------
GAS_PRICE_GWEI = 0.1
ETH_USD = 3_000.0
COMPUTE_USD_PER_CORE_S = 0.001

GWEI = 1e9


def gas_to_usd(gas: int) -> float:
    wei = gas * GAS_PRICE_GWEI * GWEI
    return wei / 1e18 * ETH_USD


def compute_to_usd(core_seconds: float) -> float:
    return core_seconds * COMPUTE_USD_PER_CORE_S


def run() -> int:
    # --- itemized cost, per epoch ------------------------------------
    librarian_gas = (ENTRIES_PER_EPOCH * GAS_INCORPORATE
                     + ROOTS_PER_EPOCH * GAS_ROOT
                     + ANCHORS_PER_EPOCH * GAS_ANCHOR)
    librarian_usd = gas_to_usd(librarian_gas)

    batch_verifier_usd = compute_to_usd(PROOFS_VERIFIED_PER_EPOCH
                                        * VERIFY_S_PER_PROOF)
    replay_executor_usd = compute_to_usd(DISPUTES_PER_EPOCH
                                         * REPLAY_S_PER_DISPUTE)
    tally_usd = compute_to_usd(TALLY_S_PER_VOTE)
    gateway_usd = compute_to_usd(GATEWAY_S_PER_EPOCH)

    total_cost_usd = (librarian_usd + batch_verifier_usd
                      + replay_executor_usd + tally_usd + gateway_usd)

    roles = {
        "librarian (on-chain gas)": {
            "units": f"{ENTRIES_PER_EPOCH} incorporate + "
                     f"{ROOTS_PER_EPOCH} root + {ANCHORS_PER_EPOCH} anchor",
            "gas_per_epoch": librarian_gas,
            "usd_per_epoch": round(librarian_usd, 4),
        },
        "batch verifier": {
            "units": f"{PROOFS_VERIFIED_PER_EPOCH} proofs sampled",
            "core_s_per_epoch": PROOFS_VERIFIED_PER_EPOCH * VERIFY_S_PER_PROOF,
            "usd_per_epoch": round(batch_verifier_usd, 4),
        },
        "reference executor (dispute replay)": {
            "units": f"{DISPUTES_PER_EPOCH} replay",
            "core_s_per_epoch": REPLAY_S_PER_DISPUTE,
            "usd_per_epoch": round(replay_executor_usd, 4),
        },
        "tally committee": {
            "units": f"{TALLY_S_PER_VOTE}s",
            "usd_per_epoch": round(tally_usd, 4),
        },
        "gateway / frontend": {
            "units": f"{SESSIONS_PER_EPOCH} sessions served",
            "core_s_per_epoch": GATEWAY_S_PER_EPOCH,
            "usd_per_epoch": round(gateway_usd, 4),
        },
    }

    # --- revenue and the closing fee ----------------------------------
    def revenue_usd(base_fee_wei: int) -> float:
        return base_fee_wei * ENTRIES_PER_EPOCH / 1e18 * ETH_USD

    test_fee_usd = revenue_usd(10)
    MARGIN = 1.25   # registered closing margin
    required_fee_wei = int(total_cost_usd * MARGIN / ETH_USD * 1e18
                           / ENTRIES_PER_EPOCH) + 1

    closing = {
        "test_suite_base_fee_wei": 10,
        "test_suite_revenue_usd": round(test_fee_usd, 6),
        "total_cost_usd": round(total_cost_usd, 4),
        "closes_at_test_fee": test_fee_usd >= total_cost_usd,
        "required_base_fee_wei_to_close": required_fee_wei,
        "required_base_fee_usd": round(required_fee_wei / 1e18 * ETH_USD, 6),
        "margin_multiple_at_required_fee": MARGIN,
        "note": ("the test-suite fee is a harness value, not the "
                 "registered deployment fee; the model sizes the "
                 "deployment fee so the line closes at 1.25x."),
    }

    evidence = {
        "milestone": "M363",
        "gate": ("cost model closes at the reference workload with "
                 "each role itemized; librarian gas included"),
        "reference_workload": {
            "axis": "vision (DINOv2-L, 601 classes)",
            "sessions_per_epoch": SESSIONS_PER_EPOCH,
            "entries_per_epoch": ENTRIES_PER_EPOCH,
            "roots_per_epoch": ROOTS_PER_EPOCH,
            "disputes_per_epoch": DISPUTES_PER_EPOCH,
            "proofs_verified_per_epoch": PROOFS_VERIFIED_PER_EPOCH,
        },
        "registered_prices": {
            "gas_price_gwei": GAS_PRICE_GWEI,
            "eth_usd": ETH_USD,
            "compute_usd_per_core_s": COMPUTE_USD_PER_CORE_S,
        },
        "roles_itemized": roles,
        "total_cost_usd_per_epoch": round(total_cost_usd, 4),
        "revenue_and_closing": closing,
        "verdict": ("CLOSES" if closing["closes_at_test_fee"]
                    else "DOES NOT CLOSE at the harness fee — "
                         "required base fee registered instead"),
    }
    OUT.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
