# M176d — Label-matrix sampling strategy (registered spec, v0)

Frozen 2026-08-17. The behavioral-transfer label matrix is arms × tasks
— quadratic in the registry. This is the registered sampling strategy
plus the tolerance contract.

## Registered facts

- Current label inventory: ONE measured behavioral-transfer label
  (M167a d0→d1 = +0.0092, recorded as a ranking constraint only — too
  thin to train on). Therefore the measured tolerance is DEFERRED until
  the matrix exists; the strategy below is frozen now so labels are
  collected INTO it from the first day.

## Strategy (frozen)

1. **Anchor rows always measured**: every arm's own-task held-out
   accuracy is measured in full (it feeds R1, the selection score, and
   the failover chain — no sampling there).
2. **Off-diagonal budget**: off-diagonal cells are drawn from a
   registered budget `B = min(K(K-1), b·K)` where `b` = labels per arm
   (default 4), so label cost grows LINEARLY in the registry, not
   quadratically.
3. **Stratified sampling**: within the budget, stratify by descriptor
   distance band (near / mid / far in fingerprint space) so the
   fingerprint's decision boundaries are sampled where they exist.
4. **Directed top-ups**: cells where the current fingerprint misorders
   measured pairs (G2 failures) get priority for the next label budget.
5. **Trigger for the measured tolerance**: when ≥ 20 measured transfer
   labels exist, train the fingerprint twice — once on the full set,
   once on the sampled set — and require: G2 margin loss ≤ 0.05 and G3
   min-cos loss ≤ 0.05 (the registered tolerance). A violation doubles
   `b` and re-measures. Until the trigger fires, no accuracy claim on
   the sampled set is permitted.

## Record

- Every label carries: arm, task, value, payload hash of the fit, and
  the registered budget epoch that paid for it.
