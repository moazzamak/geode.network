# E11 Principal Results

All values are reproduced from locked artifacts without training.

## Five-seed CIFAR-100 classification

| Method | Mean balanced accuracy | Standard deviation |
| --- | ---: | ---: |
| geode_multinomial | 0.6526 | 0.0048 |
| logistic_regression | 0.6733 | 0.0050 |
| rbf_svm | 0.6813 | 0.0060 |

## OOD and transfer

| Endpoint | Value |
| --- | ---: |
| Near-OOD AUROC | 0.6415 |
| Near-OOD FPR95 | 0.8081 |
| Far-OOD AUROC | 0.8053 |
| Far-OOD FPR95 | 0.6082 |
| Transfer GEODE balanced accuracy | 0.8196 |
| Transfer source forgetting | 0.0000 |

## Negative results and blocked work

- E4: GEODE passed non-inferiority but trailed logistic and RBF controls.
- E5: No router promoted (0 eligible of 16).
- E7: Blocked: ray_cluster_has_insufficient_nodes.
- E8: Text GEODE trailed matched linear and n-gram controls.
