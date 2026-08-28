# Prior art for the seven breakthrough directions — registered survey M135 (13 Aug 2026)

Registered search: `experiments/configs/v16/m135_breakthrough_prior_art_litsearch.json`
Evidence (live, dated snapshot): `logs/results/v16/m135_breakthrough_prior_art_litsearch/evidence.json`
Evidence (cache-only validation re-run, zero network): `logs/results/v16/m135_breakthrough_prior_art_litsearch_cached/evidence.json`
Instrument: `experiments/tier4/audit_v13_prior_art.py` (arXiv two-stage AND-then-OR + Semantic Scholar).

**Question registered before search.** Which of the seven directions of the advisory
breakthrough plan (13 Aug) is ALREADY done in the literature, and which direction
survives as an unmeasured object for this programme?

**Instrument honesty (all registered in the config, N94.x).**

- Anchors: 127 hits; instrument live.
- Recall probes: **1 of 8 retrieved** (only "Convolutional Kernel Networks" found in
  arXiv top-20). The classic probe papers (Fisher Vector 2013, Bilinear CNN 2015,
  VLAD 2012, scattering 2012–2013, Rahimi-Recht 2007, Rebuffi 2017, Yang SPM 2009)
  are **not surfaced by title-less topic queries** on these indexes — the same
  measured insensitivity as M88 (4/7). Their existence is not in doubt; they are
  registered in the config precisely as prior art that must exist.
- Semantic Scholar: **32 queries failed HTTP 429** and are recorded as failures,
  not as empty. Absence from any family is therefore UNRESOLVED — never "first".
- Local cache (registered N94.11/N94.12): successful results cached in
  `data/litsearch_cache.json`; the validation re-run served 52 queries from disk
  with zero API calls; the 32 429-failed queries are cache misses by design
  (failures are never cached).

---

## Per-direction adjudication (title/abstract level, not displacement — N94.7)

### D1 — data-axis scaling of frozen/untrained codes: **theory established, solver line live**

- The learning-curve theory for fixed/random features vs data is already verified
  prior art in this programme (Bordelon–Canatar–Pehlevan; Defilippis–Loureiro–
  Misiakiewicz; M132 survey).
- The search surfaces a large kernel-ridge-at-scale solver line rather than the
  measured object: ParK (2021), "Faster KRR Using Sketching" (2016), "Scaling up
  KRR via LSH" (2020), "Have ASkotch: large-scale KRR" (2024), "Efficient
  hyperparameter tuning for large-scale KRR" (2022), "Pack only the essentials:
  adaptive dictionary learning for KRR" (2026), "Emergent sparsity in frozen random
  CNN feature extractors" (2026), "Frozen Feature Augmentation for Few-Shot" (2024).
- **Not surfaced:** the measured Q(n) curve of a frozen patch-dictionary code vs a
  frozen DINOv2 trunk at matched per-image MACs on DomainNet — the programme's M116/
  M117 object. The object is the programme's to measure, not to claim.

### D2 — per-domain / per-group specialists: **fully established, modern MoE line dense**

- Classic: Rebuffi et al. 2017 (residual adapters, registered probe), "Learning
  Multi-Domain Convolutional Neural Networks for Visual Tracking" (2015).
- Modern found: Med-MoE (2024), DA-MoE (2025), AnchorMoE (2026), ViMoE (2024),
  "Mixture of Experts in Image Classification: What's the Sweet Spot?" (2024),
  Union of Experts (2025), OPERA (2026), MED-DSLC (2026), YOTOnet (2026),
  MDViT (2023). Domain-routed MoE with per-domain expert branches is a populated,
  moving field.
- **Consequence:** the programme's A5/M119/M124 specialist route is NOT novel — it
  never claimed to be. Its value is the sealed per-domain super-additivity
  measurement and the ~5.6× MAC accounting on a fixed corpus, measured against the
  field's own quantities (matched cost, per-domain accuracy).

### D3 — fixed nonlinear/spatial code geometry (Fisher/VLAD/power-norm/pyramids): **textbook prior art**

- Found live in the results: "Dense Image Representation with Spatial Pyramid VLAD
  Coding" (2016), "Encoding High Dimensional Local Features by Sparse Coding Based
  Fisher Vectors" (2014), "Geometric VLAD" (2014), "When VLAD met Hilbert" (2015),
  "RST-SHELO … square root normalization" (2015), "Linear Spatial Pyramid Matching
  Using Non-convex and Non-negative Sparse Coding" (2015). Plus the registered
  probes themselves (Fisher Vector 2013, VLAD 2012, Yang SPM 2009).
- This is the pre-deep standard image code. **Nothing in direction 3 is new; any
  use must be framed as re-measuring a known construction under the sealed
  protocol, never as a contribution of the construction.**

### D4 — second-order / bilinear / covariance pooling: **established**

- Bilinear CNN (2015, registered probe), Compact Bilinear Pooling (2015), Low-rank
  Bilinear Pooling (2016), spatially recurrent bilinear (2017), "Image Data
  Compression for Covariance and Histogram Descriptors" (2014). Fully prior art.

### D5 — scattering / fixed filter banks: **established, active applied line**

- Classic: "Invariant Scattering Convolution Networks" (2012), "Classification with
  Invariant Scattering Representations" (2011), Deep Scattering Network with
  Max-pooling (2021), Riesz scale-equivariant scattering (2023).
- Active applied use everywhere: medical imaging, radar, audio, EEG, plus learned
  scattering hybrids ("Wavelets Beat Monkeys at Adversarial Robustness" 2023).
- Fully prior art. A scattering arm would be a re-measurement.

### D6 — shallow learned patch encoder (the cheapest learned component): **direct antecedent exists**

- **The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
  (2021)** — the CKN patch-dictionary line, already registered as the programme's
  sparse-pipeline antecedent (Thiry et al. ICLR 2021, v16 §7.1). The probe P8
  ("Convolutional Kernel Networks", Mairal 2014) was the only retrieved probe.
- Also found: "Subgraph Clustering and Atom Learning for Improved Image
  Classification" (2024), exact sparse orthogonal dictionary learning (2021).
- The programme's M108/M113 verdicts (learned dictionary doesn't transfer / VQ
  doesn't lift) stand; the unmeasured cell is a gradient-trained single-layer
  filter bank vs the random dictionary at matched MACs — an incremental cell of a
  known family, claimable only as measurement.

### D7 — frontier-map framing ("price of learning"): **real and growing literature**

- "Untrained CNNs Match Backpropagation at V1" (2026), "Asymptotics of Learning with
  Deep Structured (Random) Features" (2024), "Bayes-optimal learning of deep random
  networks" (2023), "Deterministic equivalent and error universality of deep random
  features" (2023), "Contrasting random and learned features in deep Bayesian linear
  regression" (Phys. Rev. E 2022), "On the Power and Limitations of Random Features
  for Understanding Neural Networks" (2019), "handcrafted features versus deep
  learned features" (2024).
- The comparison of random/fixed vs learned is an active measured line. The
  programme's M134 "price of learning" (≈10× on the DSL) is an instance of a
  populated comparison class — claimable as the sealed measurement, never as the
  question.

---

## What survives as an unmeasured object (the honest niche)

Every mechanism behind the seven directions is prior art. What this search does
**not** surface, and what remains consistent with the programme's niche
(measurement, never novelty):

1. The **sealed, matched-cost, joint atoms × data surface** of a fixed patch-
   dictionary classifier vs a frozen trunk on a 345-class/6-domain corpus
   (already sealed: M116/M117/M125/M126).
2. The **specialist buy-back** end-to-end: routed per-domain specialists measured
   against the dense ladder at matched per-image MACs on the sealed corpus
   (M119/M124 measured the parts; the assembled routed system is the next
   measurement).
3. The **data-axis extension** past 138k rows on the sealed corpus — the steep
   lever, unspent.
4. The **gap-closing fraction map**: gradient-trained single-layer filter bank vs
   the random null vs the dense poles, at matched cost, same data.

**Recommendation to the plan.** Directions 3/4/5 are textbook prior art and should
be framed only as protocol-identical re-measurements if ever run. Direction 1's
theory is already owned by the literature; only its measurement on this system is
the programme's. Directions 2 and 7 are where the field is active and where the
programme's sealed matched-cost measurements remain distinguishable. No novelty
claim is licensed anywhere; all absence statements above are UNRESOLVED under
N94.6.

---

## Claim status of the seven suggestions (adjudicated 13 Aug 2026)

Every suggestion of the advisory plan is **already claimed** at the mechanism or
research-program level. None is claimable as a new idea.

| #   | Suggestion                                                      | Already claimed by (found or registered)                                                                                                                                                    | Level of claim                                    |
| --- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 3   | Power-norm / signed-square-root / spatial pyramid / multi-scale | Fisher Vector (Perronnin-Sanchez 2013), VLAD (Jegou 2012), spatial-pyramid VLAD coding (2016), square-root-normalized descriptors (2015), non-convex sparse SPM (2015)                      | Fully claimed (pre-deep standard code)            |
| 3b  | Covariance / bilinear pooling                                   | Bilinear CNN (Lin et al. 2015), Compact Bilinear (2015), Low-rank Bilinear (2016), covariance descriptors                                                                                   | Fully claimed                                     |
| 5   | Fixed filter-bank / scattering geometry                         | Invariant Scattering Networks (Mallat 2012), Bruna-Mallat PAMI 2013, learned-scattering hybrids                                                                                             | Fully claimed                                     |
| 2   | Per-domain specialists + routing                                | Rebuffi et al. 2017 residual adapters; Med-MoE (2024); DA-MoE (2025); AnchorMoE (2026); ViMoE (2024); Union of Experts (2025); "MoE in Image Classification: What's the Sweet Spot?" (2024) | Claimed (active line)                             |
| 6   | One learned filter bank + closed-form head                      | CKN (Mairal 2014); Thiry et al. ICLR 2021 (patch dictionaries - the programme's registered antecedent); patch-CKN effectiveness (2021)                                                      | Claimed (direct antecedent)                       |
| 1   | Data-axis scaling of frozen codes                               | Bordelon-Canatar-Pehlevan and the RF learning-curve theory (M132); live KRR-at-scale solver line                                                                                            | Theory claimed; specific measurement not surfaced |
| 7   | Frontier-map / "price of learning"                              | "Untrained CNNs Match Backpropagation at V1" (2026); "Contrasting random and learned features" (2022); "Power and Limitations of Random Features" (2019)                                    | Claimed as a research program                     |

**What survives, and only as measurement:** the sealed matched-cost objects - the
specialist buy-back assembled end-to-end, the data-axis extension past 138k rows,
the gap-closing-fraction map. These are protocol-identical re-measurements of
claimed mechanisms, never contributions of the mechanism itself. The M135 probe
recall (1/8) means the "not surfaced" entries are UNRESOLVED, not open; the
"found" entries above are decisive.
