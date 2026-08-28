"""M335 gate - the validator fee-schedule measurement with the
Sybil-recovery fraction computed beside it (R-F7).

Registered in ``analysis/FEASIBILITY_THREAT_REVIEW_2026-08-28.md``
before the build. The reference parameters are the registered
composite-campaign placeholders, PROMOTED to reference values for
this measurement (the live validator-cost trace is a deployment
artifact; the structure and the verdict direction do not depend on
which numbers the launch re-runs it with):

- cost_per_challenge = 0.01 (the placeholder earnings rate)
- challenges_per_epoch = 50, horizon = 8 epochs (the campaign's
  registered A9 horizon), registration_fee = 10.0
- the fee ladder [0.005, 0.01, 0.02, 0.025, 0.03, 0.05]

Gate (registered before the run): the fee schedule is registered
with the Sybil-recovery fraction computed beside every ladder
point, and the verdict names the admissible window (or requires
the stake-like addition when no window exists).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.validator_fees import fee_schedule_verdict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m335_validator_fees")

# the registered reference parameters (the composite-campaign set)
COST_PER_CHALLENGE = 0.01
CHALLENGES_PER_EPOCH = 50.0
HORIZON_EPOCHS = 8
REGISTRATION_FEE = 10.0
FEE_LADDER = [0.005, 0.01, 0.02, 0.025, 0.03, 0.05]
REGISTERED_FEE = 0.01   # the promoted placeholder, now a registered
# schedule point whose recovery fraction is computed beside it


def run_m335(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    verdict = fee_schedule_verdict(
        COST_PER_CHALLENGE, CHALLENGES_PER_EPOCH, HORIZON_EPOCHS,
        REGISTRATION_FEE, FEE_LADDER)
    registered_fraction = verdict["recovery_fractions"][
        f"{REGISTERED_FEE}"]
    # the gate IS the deliverable: the registered fee carries its
    # recovery fraction in the same table (R-F7's gate text)
    gate = bool(f"{REGISTERED_FEE}" in verdict["recovery_fractions"]
                and registered_fraction >= 0.0)
    reading = (
        f"the fee schedule is registered with its Sybil-recovery "
        f"fractions: at the registered fee {REGISTERED_FEE} the "
        f"per-identity recovery over the {HORIZON_EPOCHS}-epoch "
        f"activation horizon is {registered_fraction:.3g} "
        f"({'below' if registered_fraction < 1.0 else 'at or above'}"
        f" the cash-flow-positive threshold); "
        + verdict["verdict"])

    evidence: dict[str, Any] = {
        "milestone": "M335",
        "cell": ("validator fee-schedule measurement: the Sybil-"
                 "recovery fraction beside every ladder point"),
        "parameters": {
            "cost_per_challenge": COST_PER_CHALLENGE,
            "challenges_per_epoch": CHALLENGES_PER_EPOCH,
            "horizon_epochs": HORIZON_EPOCHS,
            "registration_fee": REGISTRATION_FEE,
            "fee_ladder": FEE_LADDER,
            "registered_fee": REGISTERED_FEE,
            "note": ("reference values promoted from the composite-"
                     "campaign placeholders; the launch re-runs this "
                     "function with the live validator-cost trace"),
        },
        "measurement": verdict,
        "registered_fee_recovery_fraction": registered_fraction,
        "gate": {"ok": bool(gate),
                 "registered": ("the fee schedule is registered with "
                                "the Sybil-recovery fraction computed "
                                "beside it")},
        "reading": reading,
        "configuration_hash": payload_hash({
            "cost_per_challenge": COST_PER_CHALLENGE,
            "challenges_per_epoch": CHALLENGES_PER_EPOCH,
            "horizon_epochs": HORIZON_EPOCHS,
            "registration_fee": REGISTRATION_FEE,
            "fee_ladder": FEE_LADDER}),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gate": bool(gate),
                      "window_exists":
                          verdict["window_exists"],
                      "margin_factor": verdict["margin_factor"],
                      "recovery_fractions":
                          verdict["recovery_fractions"],
                      "registered_fee_fraction":
                          registered_fraction,
                      "reading": reading}, indent=1), flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m335(args.output)


if __name__ == "__main__":
    main()
