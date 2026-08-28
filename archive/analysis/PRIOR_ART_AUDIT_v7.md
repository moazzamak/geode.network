# v7 M38 Prior-Art Audit

**Search date:** 26 July 2026  
**Scope:** systems relevant to the seven-stage GEODE open-world loop  
**Conclusion:** no system located and verified in this audit demonstrated all
seven stages.

## Search protocol

The audit queried arXiv, Crossref, OpenReview, DBLP where available, primary
project repositories, and direct publisher pages. Query families combined exact
titles and system names with `open world`, `novel class stream`, `continual
category discovery`, `expert routing`, `human confirmation`, `rollback`, and
`model provenance`. Primary-source text or an official repository was required
to mark a stage present. Unmentioned capabilities are recorded as unclear (`?`),
not as proven absent.

The seven stages are:

1. explicit known-class rejection;
2. rejected-sample buffering;
3. persistent unlabeled grouping;
4. human semantic confirmation;
5. incremental update or class creation;
6. empirical routing across separately owned models;
7. immutable audit and exact rollback.

## Coverage matrix

`Y` means explicitly demonstrated, `P` means partial or analogous, and `?`
means not established from the reviewed primary source.

| System | 1 | 2 | 3 | 4 | 5 | 6 | 7 | Qualification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NNO / Open World Recognition | Y | ? | ? | ? | Y | ? | ? | Defines rejection and incremental class addition. |
| OpenMax | Y | ? | ? | ? | ? | ? | ? | EVT-calibrated unknown rejection. |
| Extreme Value Machine | Y | ? | ? | P | Y | ? | ? | Incremental margin-tail model; human labeling is described, not evaluated as a gate. |
| ECSMiner | Y | Y | Y | ? | Y | ? | ? | Operational stream novelty loop without human semantic gating. |
| SAND | Y | Y | Y | ? | Y | ? | ? | Semi-supervised adaptive stream update. |
| ECHO | Y | Y | Y | ? | Y | ? | ? | Concept drift/evolution with buffered evidence. |
| MINAS | Y | Y | Y | ? | Y | ? | ? | Persistent micro-clusters and recurring classes. |
| SENCForest | Y | Y | Y | ? | Y | ? | ? | Streaming emerging-new-class detection and update. |
| Expert Gate | ? | ? | ? | ? | Y | Y | ? | Autoencoder reconstruction routes among task experts. |
| ORCA | ? | ? | P | ? | ? | ? | ? | Joint seen/novel discovery, not a persistent operational loop. |
| OpenLDN | ? | ? | P | ? | ? | ? | ? | One-shot open-world semi-supervised discovery. |
| NACH | ? | ? | P | ? | ? | ? | ? | Open-world SSL; acronym/full-text verification remains incomplete. |
| GCD | ? | ? | P | ? | ? | ? | ? | Static seen/unseen category discovery. |
| Grow and Merge | ? | ? | Y | ? | Y | ? | ? | Continuous category discovery and model growth/merge. |
| IGCD | ? | ? | Y | ? | Y | ? | ? | Category-incremental discovery across time steps. |
| MSc-iNCD | ? | ? | Y | ? | Y | ? | ? | Incremental novel-category discovery. |
| PromptCCD / PromptCCD++ | ? | ? | Y | ? | Y | ? | ? | Continual prompt-pool category discovery. |
| Virtual Category-Guided C-GCD | P | P | Y | ? | Y | ? | ? | July 2026 temporary virtual categories; no review/routing/rollback loop. |
| ART / ARTMAP / Fuzzy ART | Y | ? | ? | ? | Y | ? | ? | Vigilance mismatch creates category nodes directly. |
| SOINN | ? | ? | Y | ? | Y | ? | ? | Online topology and category-node growth. |
| DEMix | ? | ? | ? | ? | Y | P | ? | Adds/removes and mixes domain experts in one model family. |
| Branch-Train-Merge | ? | ? | ? | ? | Y | P | ? | Adds/removes expert language models and merges/ensembles them. |
| LoraHub | ? | ? | ? | ? | ? | Y | ? | Composes independently trained LoRA modules. |
| PHATGOOSE | ? | ? | ? | ? | ? | Y | ? | Post-hoc token/layer routing among PEFT experts. |
| Arrow | ? | ? | ? | ? | Y | Y | ? | Builds an adapter library and routes without retraining. |
| kNN OOD | Y | ? | ? | ? | ? | ? | ? | Non-parametric frozen-feature rejection control. |
| Mahalanobis / GMM / DDU | Y | ? | ? | ? | P | ? | ? | Proper density evidence; update semantics vary by implementation. |
| DUQ / SNGP | Y | ? | ? | ? | ? | ? | ? | Distance-aware discriminative uncertainty. |
| Deep SVDD | Y | ? | ? | ? | ? | ? | ? | Compact one-class support model. |

## Finding

No reviewed source demonstrated the conjunction of human semantic confirmation,
empirical routing across independently versioned models, and immutable exact
rollback. The stream-mining family covers stages 1–3 and 5 most completely.
Expert Gate and recent adapter systems cover stage 6. Continual-GCD systems
cover stages 3 and 5. This supports only the qualified positioning:

> GEODE evaluates an engineered composition of established open-world stages
> with explicit review, immutable provenance, transactional publication, and
> cross-model routing.

It does not prove universal absence. Patent literature, non-English work,
closed-source industrial systems, and sources unavailable to the audit remain
unresolved. DBLP and Semantic Scholar were intermittently rate-limited, and
full text was not available for every paper. Outcome E remains available if a
later source displaces the composition claim.

## Primary references

1. Bendale and Boult, “Towards Open World Recognition,” CVPR 2015,
   https://arxiv.org/abs/1412.5687.
2. Bendale and Boult, “Towards Open Set Deep Networks,” CVPR 2016,
   https://arxiv.org/abs/1511.06233.
3. Rudd et al., “The Extreme Value Machine,” TPAMI 2018,
   https://arxiv.org/abs/1506.06112.
4. Masud et al., “Classification and Novel Class Detection in Concept-Drifting
   Data Streams under Time Constraints,” TKDE 2011,
   https://doi.org/10.1109/TKDE.2010.61.
5. Haque et al., “SAND,” AAAI 2016,
   https://doi.org/10.1609/AAAI.V30I1.10283.
6. Haque et al., “ECHO,” ICDE 2016,
   https://doi.org/10.1109/ICDE.2016.7498264.
7. de Faria et al., “MINAS,” DMKD 2016,
   https://doi.org/10.1007/s10618-015-0433-y.
8. Aljundi et al., “Expert Gate,” CVPR 2017,
   https://arxiv.org/abs/1611.06194.
9. Cao et al., “Open-World Semi-Supervised Learning,” ICLR 2022,
   https://arxiv.org/abs/2102.03526.
10. Rizve et al., “OpenLDN,” ECCV 2022,
    https://arxiv.org/abs/2207.02261.
11. Vaze et al., “Generalized Category Discovery,” CVPR 2022,
    https://arxiv.org/abs/2201.02609.
12. Zhang et al., “Grow and Merge,” NeurIPS 2022,
    https://arxiv.org/abs/2210.04174.
13. Zhao and Mac Aodha, “Incremental Generalized Category Discovery,” ICCV 2023,
    https://arxiv.org/abs/2304.14310.
14. Cendra et al., “Effective Prompt Pool Learning for Continual Category
    Discovery,” ECCV 2024, https://arxiv.org/abs/2407.19001.
15. Xiong et al., “Virtual Category-Guided Continual Generalized Category
    Discovery,” ECCV 2026, https://arxiv.org/abs/2607.04984.
16. Carpenter, Grossberg, and Reynolds, “ARTMAP,” Neural Networks 1991,
    https://doi.org/10.1016/0893-6080(91)90012-T.
17. Shen and Hasegawa, “An Incremental Network for On-line Unsupervised
    Classification and Topology Learning,” Neural Networks 2006,
    https://doi.org/10.1016/j.neunet.2005.04.006.
18. Mu et al., “Classification Under Streaming Emerging New Classes,” TKDE 2017,
    https://doi.org/10.1109/TKDE.2017.2691702.
19. Gururangan et al., “DEMix Layers,” 2021,
    https://arxiv.org/abs/2108.05036.
20. Li et al., “Branch-Train-Merge,” 2022,
    https://arxiv.org/abs/2208.03306.
21. Huang et al., “LoraHub,” COLM 2024,
    https://arxiv.org/abs/2307.13269.
22. Muqeeth et al., “Learning to Route Among Specialized Experts for Zero-Shot
    Generalization,” 2024, https://arxiv.org/abs/2402.05859.
23. Ostapenko et al., “Towards Modular LLMs by Building and Reusing a Library of
    LoRAs,” 2024, https://arxiv.org/abs/2405.11157.
24. Sun et al., “Out-of-Distribution Detection with Deep Nearest Neighbors,”
    ICML 2022, https://arxiv.org/abs/2204.06507.
25. Lee et al., “A Simple Unified Framework for Detecting Out-of-Distribution
    Samples and Adversarial Attacks,” NeurIPS 2018,
    https://arxiv.org/abs/1807.03888.
26. Mukhoti et al., “Deep Deterministic Uncertainty,” TMLR 2023,
    https://arxiv.org/abs/2102.11582.
27. van Amersfoort et al., “Uncertainty Estimation Using a Single Deep
    Deterministic Neural Network,” ICML 2020,
    https://arxiv.org/abs/2003.02037.
28. Liu et al., “Simple and Principled Uncertainty Estimation with Deterministic
    Deep Learning via Distance Awareness,” NeurIPS 2020,
    https://arxiv.org/abs/2006.10108.
29. Ruff et al., “Deep One-Class Classification,” ICML 2018,
    https://proceedings.mlr.press/v80/ruff18a.html.
