# M181 — value-function spec (accuracy-delta x efficiency penalty +

coverage bonus), with the H4 / H5 sensitivity gates

Registered draft, 18 Aug 2026, in
`RESEARCH_IMPLEMENTATION_PLAN_v25.md` section 6. This is a SPEC; nothing
here is measured until the M180 bake-off seals and the gates below are
registered with numbers.

## Purpose

The value function answers: what is a component's contribution worth?
It is the input to Track I's reward mechanism. Per the user's standing
priorities — high accuracy, lightweight/low computation, good security —
the function must make accuracy and efficiency commensurable and must
not be gameable by bloat.

## Registered design (v1)

For a component c on a task t, with the sealed read (penalty 1.0,
frozen codes):

    V(c) = delta_accuracy(c) x efficiency(c) + coverage_bonus(c)

where

- `delta_accuracy(c)` = measured held-out accuracy of the coalition
  including c minus the measured accuracy without c (LOO marginal from
  the M180 game; Shapley is the fallback if H2 shows LOO unstable).
- `efficiency(c)` = a decreasing, scale-free function of the
  component's measured per-image MACs / params. Candidate form
  (registered): `efficiency = (cost_ref / cost(c))^gamma` with
  `cost_ref` = the registry's current best cost at-or-above c's
  accuracy, and `gamma` the efficiency weight. `gamma` and `cost_ref`
  are registered before any reward is computed; `gamma = 1` is the
  default stance.
- `coverage_bonus(c)` = a registered constant, paid ONLY when c unlocks
  a previously unserved task axis (the capability map's R-new-axis
  fires). Redundant high-accuracy components earn no bonus.

## The sensitivity gates (from the v25 plan, now with form)

- **H4 (no bloat incentive):** adding a uselessly large component must
  earn less than a smaller accurate one. Gate: for a registered
  bloat-vs-lean pair, V(bloat) < V(lean). This pins `gamma` and
  `cost_ref` — if the gate fails, gamma rises, not the formula.
- **H5 (coverage bonus != accuracy bonus):** a task-axis-unlocking
  component that scores LOW accuracy must still be ranked above a
  redundant high-accuracy one. Gate: V(axis-opener) > V(redundant) on
  the registered pair, which pins the coverage_bonus size.
- **Security constraint (registered):** V is computed ONLY from
  validator-replayed measurements (M177 L0 replay); no self-reported
  numbers enter the function. The anti-wash stack (Track I Phase C)
  sits on top of V and is simulated before any token exists.

## What waits on what

V's numbers are not measured yet: the M180 collection + bake-off seal
first (delta_accuracy operands), then the registered H4/H5 pairs are
chosen and the gates run. Until then this spec is a registered
conjecture, not a claim.
