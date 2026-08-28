"""M305 — sequential-test probe adjudication (SPRT), margin-gated
mismatches, adaptive probe rate.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M305
(26 Aug 2026, before any build). The A5 repair: single-mismatch
triggering is replaced by a Wald sequential probability ratio test
over the per-artifact mismatch stream, which detects a sustained
small-delta substitution in bounded expected observations while
holding a registered false-conviction rate. The corrected detection
horizon is 1/(rho * delta), not 1/rho.

Registered semantics (written before any measurement):

- **Margin gate (R-A6d).** A probe mismatch counts only when the
  serving answer's margin |s_top - s_2nd| exceeds the registered
  numeric-noise floor. Below it, a disagreement is a tie the hardware
  broke differently - recorded, not counted.
- **SPRT.** Hypotheses: H0 honest (per-probe mismatch probability
  p0, the registered honest rate) vs H1 substitute (p1 = the
  registered minimum detectable delta). Decision bounds from the
  registered false-positive rate alpha and missed-detection rate
  beta: accept H1 when the log likelihood ratio exceeds ln((1-beta)/
  alpha); accept H0 when it falls below ln(beta/(1-alpha)).
- **Adaptive rho.** New or recently disputed contributors are probed
  at rho = 1.0, decaying with clean epochs: rho(t) = max(base,
  0.5^t), base the registered floor (M314: rho >= 0.05).
"""
from __future__ import annotations

import math
from typing import Any

PROBE_RATE_FLOOR = 0.05        # M314: outside ordinary governance
DEFAULT_ALPHA = 0.01           # registered false-conviction rate
DEFAULT_BETA = 0.01            # registered missed-detection rate


def margin_gated_mismatch(top_score: float, second_score: float,
                          noise_floor: float) -> tuple[bool, float]:
    """A mismatch counts only above the registered margin floor.
    Returns (counted, margin)."""
    margin = abs(float(top_score) - float(second_score))
    return margin > float(noise_floor), margin


def adaptive_probe_rate(clean_epochs: int, base: float
                        = PROBE_RATE_FLOOR) -> float:
    """New/recently-disputed contributors are probed at 1.0; the rate
    decays with clean history but never below the registered floor."""
    if clean_epochs < 0:
        raise ValueError("clean_epochs must be non-negative")
    return max(float(base), 0.5 ** clean_epochs)


def sprt(mismatches: int, trials: int, p0: float, p1: float,
         alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA
         ) -> dict[str, Any]:
    """One Wald SPRT step over (mismatches, trials). Returns the
    decision: 'convict', 'acquit', or 'continue'."""
    if not 0.0 < p0 < p1 <= 1.0:
        raise ValueError("require 0 < p0 < p1 <= 1")
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("alpha and beta must be in (0, 1)")
    if trials <= 0 or not 0 <= mismatches <= trials:
        raise ValueError("need 0 <= mismatches <= trials")
    m = int(mismatches)
    n = int(trials)
    lr = (m * math.log(p1 / p0)
          + (n - m) * math.log((1.0 - p1) / (1.0 - p0)))
    a_bound = math.log((1.0 - beta) / alpha)
    b_bound = math.log(beta / (1.0 - alpha))
    if lr >= a_bound:
        decision = "convict"
    elif lr <= b_bound:
        decision = "acquit"
    else:
        decision = "continue"
    return {"decision": decision, "log_likelihood_ratio": lr,
            "convict_bound": a_bound, "acquit_bound": b_bound,
            "mismatches": m, "trials": n, "p0": p0, "p1": p1}


def corrected_horizon(rho: float, delta: float) -> float:
    """The corrected detection horizon E[T] = 1/(rho * delta); the
    paper's 1/rho is the delta=1 special case."""
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    if not 0.0 < delta <= 1.0:
        raise ValueError("delta must be in (0, 1]")
    return 1.0 / (float(rho) * float(delta))
