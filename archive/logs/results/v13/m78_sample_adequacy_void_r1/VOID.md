# VOID — M78 execution R1 (29 July 2026)

This directory holds the **first M78 execution, which is void**. It is retained
unaltered for lineage. It must not be cited, and no claim may rest on it.

## Defect

The `subspace_stability` operand was computed by calling
`initialize_projected_metric_fields` **independently on each disjoint half** of
the geometry split.

That function fits a global PCA projection before fitting per-class subspaces.
Fitting it twice gives each half **its own projection frame**. The per-class
bases returned for the two halves are therefore expressed in two different
64-dimensional coordinate systems, and the principal angles computed between
them measure **projection variance**, not basis identifiability — the quantity
the milestone registered.

## How it was caught

A positive control in `experiments/common/test_v13_sample_adequacy.py`
generated a synthetic corpus whose per-class subspaces are exactly rank-2 and
strongly identified by 400 samples per class. The metric should have returned a
near-zero angle. It returned **40.45 degrees**. The failure is a property of the
measurement, not of the data.

## Scope of the void

- **Void:** every `subspace_stability` field, and the `gate` entries derived
  from it.
- **Not void, but superseded:** the accuracy and unknown-recall numbers, which
  come from `_fit_field` / `_field_outputs` / `_control_outputs` and were not
  touched by the defect. They are reproduced unchanged in the R2 execution.

## Replacement

`logs/results/v13/m78_sample_adequacy/` (schema_version 2, amendments R1+R2).
Stability is now measured by `experiments.common.v13_sample_adequacy`, which
fits the projection **once** on the full geometry split, projects both halves
through that single shared frame, and reports the measured angle against a
Monte-Carlo random-subspace reference so the number is interpretable.

## Registered lesson

Carried into `analysis/RESEARCH_IMPLEMENTATION_PLAN_v13.md` Section 2 as design
principle 9: **every measurement operand must ship with a positive control that
fails if the operand is not measuring what it names.** M77 and M78 were both
defect-hunting milestones, and the second one shipped a defect of exactly the
class it existed to find.
