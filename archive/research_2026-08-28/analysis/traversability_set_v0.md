# GEODE traversability set v0 (M170 artifact)

Frozen 2026-08-17. This file is the registered validation artifact for
Phase B traversability: the analogy quadruples the fingerprint encoder is
measured against, plus the measured axis-shift scores.

## Construction (registered in `experiments/tier4/eval_v24_m169_fingerprint_train.py`)

- One quadruple per task × axis for two axes: `input.modality` and `output.kind`.
- Swap rule: replace the axis value with the FIRST vocab token; if the value
  already IS the first token, replace with the LAST token.
- Direction vector: `e_new - e_old` of the token embeddings; score =
  cosine between the fingerprint movement `f_swap - f_orig` and that direction.
- Threshold: min-cos >= 0.5 (G3 gate).

Vocab (from the frozen `AXES` schema):

- `input.modality`: image, token-text, numeric-series, tabular, graph, audio, control-signal
- `output.kind`: class, regression, next-token, action, distribution, ranking

## Measured scores (M169 full run, 300 steps, seed 11)

| task         | axis           | swap                   | cos(dir, move) |
| ------------ | -------------- | ---------------------- | -------------- |
| domainnet    | input.modality | image → control-signal | 0.9932         |
| domainnet    | output.kind    | class → ranking        | 0.9123         |
| cifar10      | input.modality | image → control-signal | 0.9909         |
| cifar10      | output.kind    | class → ranking        | 0.9002         |
| mackey_glass | input.modality | numeric-series → image | 0.8936         |
| mackey_glass | output.kind    | regression → class     | 0.8485         |
| lorenz       | input.modality | numeric-series → image | 0.8936         |
| lorenz       | output.kind    | regression → class     | 0.8485         |
| dyck         | input.modality | token-text → image     | 0.8733         |
| dyck         | output.kind    | next-token → class     | 0.9150         |
| tabular      | input.modality | tabular → image        | 0.7549         |
| tabular      | output.kind    | regression → class     | 0.8875         |

- min = **0.7549** (tabular, input.modality); all 12 pass the 0.5 threshold.
- Weakest direction: tabular → image. Registered as the known-weak link for
  any future improvement arm.
- The traversability threshold recorded for MVP use: **0.5 (floor), 0.755
  (measured minimum)**.

## Status

- This set doubles as the frozen validation artifact: any future encoder
  change must re-run against these 12 quadruples with the same threshold.
- Extension path (registered, not yet run): quadruples over the remaining
  categorical axes (`input.value_kind`, `latent.recurrence`, `coupling`, …)
  once the v0 signal mix is upgraded.
