# v13 deliverable frontier

Assembled by `experiments/tier4/assemble_v13_frontier.py` from sealed
evidence only. No figure here was computed by the assembler; each is read
from the milestone named beside it. Cells a milestone did not produce are
marked **absent** rather than approximated (N85.5).

Assembled 2026-07-30T08:38:31.709895+00:00.

## I5-128 — 128-way, chance 0.781%, 8192 evaluation rows

Accuracy and I5 are means over M81's seeds 11, 23 and 37, with the spread across seeds in brackets. Every I5 figure is printed beside the shuffled-explanation null sharing its structure, budget and split (R5).

| Head | Basis | Accuracy | I5 | I5 null | I5 − null | Cited atoms | Active parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sparse_linear_l1_0.0` | sparse indexed | 60.836% (1.062%) | 6.384% (0.175%) | 0.891% | 5.493% | 32.00 | 1,048,704 |
| `sparse_linear_l1_0.03` | sparse indexed | 51.518% (0.854%) | 4.317% (0.957%) | 0.795% | 3.522% | 22.74 | 860,834 |
| `sparse_linear_l1_0.1` | sparse indexed | 50.269% (1.160%) | 4.433% (1.208%) | 0.864% | 3.570% | 21.38 | 793,282 |
| `sparse_linear_l1_0.3` | sparse indexed | 56.669% (1.733%) | 7.960% (0.478%) | 0.723% | 7.237% | 15.69 | 628,469 |
| `sparse_linear_budget_1024` | sparse indexed | 43.901% (1.685%) | 4.364% (1.029%) | 0.875% | 3.489% | 5.12 | 131,200 |
| `sparse_linear_budget_512` | sparse indexed | 35.274% (1.135%) | 4.455% (0.456%) | 0.746% | 3.709% | 3.39 | 65,664 |
| `sparse_linear_budget_256` | sparse indexed | 21.944% (2.014%) | 4.278% (0.266%) | 0.773% | 3.505% | 2.22 | 32,896 |
| `metric_field_shrinkage_0.1` | sparse indexed | 63.574% (0.232%) | 2.134% (0.465%) | 0.719% | 1.415% | 32.00 | 2,097,152 |
| `metric_field_shrinkage_0.5` | sparse indexed | 65.963% (0.195%) | 2.620% (0.050%) | 0.662% | 1.958% | 32.00 | 2,097,152 |
| `metric_field_shrinkage_1.0` | sparse indexed | 66.398% (0.208%) | 2.508% (0.423%) | 0.780% | 1.728% | 32.00 | 2,097,152 |
| `decision_list` | sparse indexed | 15.739% (3.003%) | 6.372% (2.560%) | 2.769% | 3.603% | 0.25 | 73 |
| `knn` | frozen dense | 66.125% (0.000%) | 3.312% (0.295%) | 0.788% | 2.524% | 6.72 | 25,165,824 |
| `rbf_nystroem` | frozen dense | 56.925% (0.586%) | 9.421% (1.360%) | 0.831% | 8.590% | 2047.96 | 262,144 |
| `mlp_integrated_gradients` | frozen dense | 65.918% (0.696%) | 4.508% (0.340%) | 0.840% | 3.668% | 384.00 | 262,784 |
| `mlp_expected_gradients` | frozen dense | 65.918% (0.696%) | 4.162% (0.211%) | 0.785% | 3.377% | 384.00 | 262,784 |

## I5-8 — 8-way, chance 12.500%, 512 evaluation rows

Accuracy and I5 are means over M81's seeds 11, 23 and 37, with the spread across seeds in brackets. Every I5 figure is printed beside the shuffled-explanation null sharing its structure, budget and split (R5).

| Head | Basis | Accuracy | I5 | I5 null | I5 − null | Cited atoms | Active parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sparse_linear_l1_0.0` | sparse indexed | 87.565% (0.977%) | 31.515% (3.577%) | 11.627% | 19.888% | 31.79 | 64,187 |
| `sparse_linear_l1_0.03` | sparse indexed | 87.565% (0.586%) | 31.028% (4.352%) | 12.119% | 18.909% | 29.64 | 60,223 |
| `sparse_linear_l1_0.1` | sparse indexed | 84.505% (0.391%) | 24.182% (9.278%) | 11.563% | 12.619% | 23.01 | 47,600 |
| `sparse_linear_l1_0.3` | sparse indexed | 85.352% (0.781%) | 25.008% (5.640%) | 10.113% | 14.895% | 14.88 | 30,078 |
| `sparse_linear_budget_1024` | sparse indexed | 86.133% (1.367%) | 33.073% (3.347%) | 13.092% | 19.981% | 11.93 | 8,200 |
| `sparse_linear_budget_512` | sparse indexed | 85.482% (0.977%) | 38.207% (9.970%) | 10.821% | 27.386% | 8.39 | 4,104 |
| `sparse_linear_budget_256` | sparse indexed | 84.831% (0.586%) | 38.660% (1.895%) | 11.940% | 26.719% | 5.72 | 2,056 |
| `metric_field_shrinkage_0.1` | sparse indexed | 87.240% (2.539%) | 18.267% (10.585%) | 12.855% | 5.412% | 32.00 | 131,072 |
| `metric_field_shrinkage_0.5` | sparse indexed | 87.956% (1.172%) | 21.183% (13.113%) | 13.325% | 7.858% | 32.00 | 131,072 |
| `metric_field_shrinkage_1.0` | sparse indexed | 88.346% (1.758%) | 21.610% (4.190%) | 14.190% | 7.420% | 32.00 | 131,072 |
| `decision_list` | sparse indexed | 78.906% (0.977%) | 16.806% (4.649%) | 12.701% | 4.105% | 0.97 | 308 |
| `knn` | frozen dense | 87.500% (0.000%) | 26.366% (4.361%) | 12.633% | 13.733% | 8.81 | 1,572,864 |
| `rbf_nystroem` | frozen dense | 77.995% (1.367%) | 26.435% (5.918%) | 13.121% | 13.314% | 2047.90 | 16,384 |
| `mlp_integrated_gradients` | frozen dense | 89.518% (0.195%) | 34.956% (6.178%) | 12.077% | 22.878% | 384.00 | 201,224 |
| `mlp_expected_gradients` | frozen dense | 89.518% (0.195%) | 31.732% (2.506%) | 14.563% | 17.168% | 384.00 | 201,224 |

## Open-set competence

Reported once, for the boundary, because every head above reads the same
frozen features and the same fitted geometry. A per-head column would
imply a distinction nobody measured (N85.12).

- **Rejection recall at matched known coverage:** 0.11875 (M84, verdict `ladder_flat`). the untrained zero rung. Every exposure-trained rung in M84 scores at or below 0.00012, so training on real out-group images destroys rejection rather than improving it.
- **AUROC:** 0.5851 pooled, 0.6580 within domain, against free baselines of 0.5749 (10-NN) and 0.5617 (nearest centre). Verdict `geometry_ties_free_baselines`; meets L2's threshold-free bar: **False**.

## Transfer

Retention is sparse-probe accuracy over dense-probe accuracy on the
same rows, split and budget (N85.8). Absolute accuracies across
corpora are not comparable and are not compared. The control column
is the same measurement over a random dictionary of identical size
and identical active-atom budget, so it isolates what fitting bought
from what the sparse code's shape alone buys (R5).

| Arm | Retention | Random-dictionary control | Fitting bought |
| --- | --- | --- | --- |
| native DomainNet, 20-way | 0.9936 | 0.9240 | +0.0696 |
| degraded to 32×32, 20-way | 0.9060 | 0.8680 | +0.0380 |
| CIFAR-100, 20-way | 0.9157 | 0.8373 | +0.0784 |

Resolution cost +0.0876; corpus cost beyond resolution -0.0097. Verdict `loss_is_resolution_not_corpus`.

## v12 historical reference (Amendment R7)

Amendment R7. v12 CIFAR-10 figures at 8-way I5, retained as historical reference only. Not v13 bars. The v13 corpus is 128-class DomainNet and its raw-probe accuracy bar is 61.304 percent, so no cell below is comparable to any cell above it.

| Head | Basis | Accuracy | I5 | Size |
| --- | --- | --- | --- | --- |
| RBF | frozen dense | 96.917% | 22.772% | absent |
| kNN | frozen dense | ~96.7% | 25.246% | 6.02 MB |
| GEODE v12 field | learned dense | 96.083% | 17.737% | absent |

---

Source evidence hashes:

- `m81_sparse_head` — 61181c1c878cab273e66d11d5d45ad758b365766a7e83141e159c5af2cf1a6e3
- `m84_exposure_ladder` — 68015109b3ab8b66ff7d4eb496c31a837869cfcdaf3cb125ed765fa7ca27cdf5
- `m85_open_set_auroc` — bf72f81de6f6bd7ed14f0f02101cfd13bd82a75bf1bf0791eada910f2910decb
- `m85_transfer_eval` — 46026f10a113c5dca015855b51bca4d2606d6af4c58b91c7a5bc63a475d94e51
