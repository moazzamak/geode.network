# BUILDOUT-BLOCKER LITERATURE SEARCH (M164, 17 Aug 2026)

Registered before any query ran (v23 §6 dispatch entry; config
`experiments/configs/v23/m164_buildout_blockers.json`; runner
`tools/m164_buildout_blocker_search.py`; evidence
`logs/results/v23/m164_buildout_blockers/evidence.json`).

**Instrument result:** all six anchor queries hit in stage 1 (AND of
quoted `abs:` phrases) — the positive control passed, no stage-2
fallback needed. The search is admissible for its registered role.

**Role (unchanged):** displacement only. An unauthenticated public
search cannot support novelty claims. Below, "no displacer found"
means this pass found nothing that displaces a registered claim — it
does not establish novelty.

---

## Per-blocker findings (what already exists in print)

### L1 / M176c — better codes

- **Deep-patch spatial pyramids already exist:** "Dense Image
  Representation with Spatial Pyramid VLAD Coding of CNN for Locally
  Robust Captioning" (arXiv:1603.09046, 2016). The M176c candidate 1
  (SPM bins over deep patch tokens) is a published idea — the buildout
  must treat it as a comparison baseline, not a claim.
- **Deep dictionary learning exists:** "Deep Semantic Dictionary
  Learning for Multi-label Image Classification" (arXiv:2012.12509)
  and "Row-Sparse Discriminative Deep Dictionary Learning for
  Hyperspectral Image Classification" (arXiv:1912.10804). M113's
  negative was about VQ atom replacement inside one construction; the
  wider family is published and must be cited.
- Fisher-vector deep features: arXiv:1603.09046 and the Fisher-vector
  classification line (anchor a3) cover the M176c candidate 2
  territory.

### L4 / M176b — the quadratic fit wall

- **Divide-and-conquer kernel ridge:** "Divide and Conquer Kernel
  Ridge Regression: A Distributed Algorithm with Minimax Optimal
  Rates" (arXiv:1305.5029) and its CV extension (arXiv:1612.05907).
- **Preconditioned ridge:** "Two-level preconditioning for Ridge
  Regression" (arXiv:1806.05826).
- **Closed-form solvers as modules:** "Meta-learning with
  differentiable closed-form solvers" (arXiv:1805.08136).

The escape ladder for M176b should evaluate these constructions
first, not invent new ones.

### M167 — behavioral-transfer label protocol

- **Transfer metrics are unstable in print too:** "How stable are
  Transferability Metrics evaluations?" (arXiv:2204.01403) — a
  must-cite: the label protocol's stability gate (rankings preserved
  under suite perturbation) is the right gate because the literature
  reports instability.

### v24 fingerprinting / routing

- Task embeddings line: Vu et al. "Analysis and Prediction of NLP
  Models Via Task Embeddings" (arXiv:2112.05647) and the task-
  selection follow-ups (arXiv:2407.16245) — the additive-attribute
  fingerprint must be positioned against the task-embedding
  literature, which exists.
- Task-MoE routing in the LLM era (e.g., "AT-MoE", arXiv LoRA MoE
  lines, b4 hits) — routing literature is mature; v24's contribution
  is the measured-behavioral labels, not routing mechanics.

### v25 attribution (M180)

- Shapley for ML is mature: "The Shapley Value in Machine Learning"
  (arXiv:2202.05594), "Beta Shapley" (arXiv:2110.14049); data
  valuation: "EcoVal" (arXiv:2402.09288), "Data Overvaluation Attack
  and Truthful Data Valuation in Federated Learning"
  (arXiv:2502.00494). M180 should import these estimators and gate
  them on H2, not re-derive.

### Track P (M192/M193)

- zk proofs for ML: "zkDL: Efficient Zero-Knowledge Proofs of Deep
  Learning Training" (arXiv:2307.16273), "Zero-Knowledge Proof Based
  Verifiable Inference of Models" (arXiv:2511.19902).
- Secret-shared linear models: "Online Efficient Secure Logistic
  Regression based on Function Secret Sharing" (arXiv:2309.09486),
  "Secure PAC Bayesian Regression via Real Shamir Secret Sharing"
  (arXiv:2109.11200). The secret-shared Gram fit (M192) has close
  prior art to cite and build on.

### M162 context — pruning

- Retraining-free pruning is an active direction: "Accurate
  Retraining-free Pruning for Pretrained Encoder-based Language
  Models" (arXiv:2308.03449), "Pruning On-the-Fly: A Recoverable
  Pruning Method without Fine-tuning" (arXiv:2212.12651). M162's
  negative (retraining lost 4.8 points to no-retrain) sits
  consistently inside this literature.

### M176a — ceiling probes

- "Closed-Form Linear-Probe Dataset Distillation for Pre-trained
  Vision Models" (arXiv:2605.07194) — closed-form probe practice on
  frozen vision features exists; M176a's kNN/diagonal bounds should
  cite the probe line.

---

## Overall displacement verdict

No hit in this pass displaces the core registered claims (freezing
wins everywhere measured; additive closed-form composition; routing
across task types on measured behavioral labels; the token mechanism
design). The closest lines are task embeddings (v24 fingerprints) and
Shapley valuation (v25) — both are _components_ the plan already
treated as research objects, and both have mature literature to cite.

**Boundary:** this verdict is a no-displacer-found statement, not a
novelty statement. The buildout's novel-work claims (if any) need the
registered instrument with named-paper controls before publication,
per M164's gate.
