# RESEARCH IMPLEMENTATION PLAN v18

**Frontier measurement of a closed-form low-degree stack: does quality scale with hardware budget?**

Date: 6 August 2026 (written 7 August 2026).
Status: v17's sealed evidence stands as mechanism facts. v18 re-frames the **success criterion** (frontier, not accuracy-dominance), registers the **breakthrough requirement** (quality must scale with compute), and plans the two gating experiments (M115 closed-form stack; M116 data scaling) plus the pending M114. **Sealed outcomes (7 Aug):** M115 KS1 fired (closed-form depth does not lift the ceiling — B1 false on the quality axis; KS2's >=10x training-cost claim fails symmetric accounting at 1.11x); M114 KS2 fired (8.4x op reduction real, but 108-bit Hamming loses 3.3 pts — B2 partial); M116 KS NOT fired (ratio 1.351 — the frozen family's data scaling is steeper than the cost-matched dense trunk's; B3 positive as registered). **8 Aug:** exploitation of the two positives (M116 data scaling × A5 per-domain) registered as §5.5 (M117-M121); fresh data-scaling literature sweep added to §4 (the random-feature-ridge scaling-law theory predicts M116's mechanism; the sealed measurement under this protocol remains the open niche). **Sealed follow-ups (8 Aug):** M121's MSE certificate FIRED (the theory does not predict the accuracy crossing; but the sparse span holds 3.4x more one-hot label power than the trunk's — 11.8% vs 3.5%); **M117 PASSED — the 2D atoms×data surface is super-additive (excess +0.0097/+0.0101), the 0.2153 ceiling was a slice artifact, and head-width steepness rises with width (M120 confirmed) — the first positive on the quality axis, rescinding the "quality axis is closed" statement to "flat only on single slices".**

---

## 0. Executive summary

The v16/v17 program measured, with sealed evidence, that a single-stage frozen sparse family saturates at ~0.21 on the shared DomainNet subsample: random dictionaries (0.2153), learned/k-means dictionaries (0.2101), and routing (0.2092) all plateau; only **per-domain routed specialists beat a trained transformer at matched-or-lower cost (A5 KS2: 5/6 domains, 1.4-7.2x fewer MACs)** and the **parameter axis (M110)** is won.

v18 states the honest reading of those results:

1. **The quality (accuracy-dominance) axis is closed for the sparse family as currently built.** Ghorbani 2020 and this program's own sealed ladder (M108: 24x more atoms buys ~+10 points, flattening) both say so. This is a _fact about the single-stage frozen family_, not a license to consolidate.
2. **The efficiency axes were never the thing being gated.** A5 KS2 and M110 are cost-axis wins. The accuracy-gates' _consequences_ ("consolidate") were the mistake, not the gates.
3. **The long-term question reduces to one measurable condition:** does the family's quality scale with hardware budget, i.e. does $Q_S(C \cdot R) \ge Q_T(C)$ for the measured efficiency multiplier $R$? This requires **depth that raises effective degree** (single-stage is degree-2 and saturates) — the untested axis.
4. **The theory is on the program's side.** Neural LoFi (Dandi et al. 2026) describes deep learning as closed-form low-degree filtering; a spectral estimator provably achieves optimal scaling for hierarchical targets (Defilippis et al. 2026); higher cumulants become learnable at linear sample complexity _when a correlated latent structure is shared across degrees_ (Bardone et al. 2026) — the stack's exact property.
5. **The plan:** M115 (closed-form LoFi stack, gated on degree-lift and training-cost ratio) and M116 (data-scaling $Q(n)$), with the diagnostics that predict the ceiling before running. Prior art says the _techniques_ are done; the _sealed measurement under this protocol_ is not.

---

## 1. The success criterion (re-framed)

**Quality is a coordinate, not the objective.** Every cost claim carries its accuracy; the accuracy _floor_ (competitive at some cost point) stays. What changes is what a gate _licenses_:

- **Mechanism gates keep accuracy as the instrument** (a model that can't learn tells you nothing). These killed false claims correctly (M109, M113, A5-KS1) and continue to.
- **Success gates are frontier gates:**
  - **Frontier non-domination** — beats dense on cost at matched accuracy, OR on accuracy at matched cost, in some regime (A5 KS2 is the sealed example).
  - **Training-cost ratio** — closed-form cost to reach a fixed accuracy vs dense's cost (the axis nobody measures; the primary operand of M115).

**Revisits of v17 consequences (fact vs. license):**

| v17 statement                                                     | v18 reading                                                                                           |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| A5 KS1 "routing does not lift the ceiling → program consolidates" | Mechanism fact stands; **the "consolidate" license was wrong** — A5 KS2 was a frontier win beside it. |
| M109 "MAC crossing does not survive trunk training"               | Scoped to the **global** arm; the routed per-domain regime (A5) is separate and wins on MACs.         |
| M113 "the ~0.21 ceiling is the family's"                          | Scoped to **single-stage**; the depth axis (M115) is untested.                                        |
| "Option 1 / Arm S deferred, not motivated by ceiling-lift"        | Deferral was accuracy-gate-driven; **Arm S is re-opened on the cost axes** (deferred, not dead).      |
| "Prior art (Ghorbani) ⇒ the family can't win"                     | Ghorbani bounds the **accuracy** axis; the cost axes were always outside it.                          |

---

## 2. The breakthrough requirement (B1-B5)

The condition $Q_S(C \cdot R) \ge Q_T(C)$ is a **conjunction**; each factor is independently measurable, and any one failing kills the long-term thesis:

- **B1 — depth that raises degree.** A closed-form layer rule whose effective polynomial degree grows ~$2^L$ with depth, at O(atoms·data) per layer, no gradients. Without it, $Q_S(C)$ stays flat (measured).
- **B2 — a per-layer compute constant that keeps R large.** The encode is the wall (cdist = O(patches·atoms·108)); hardware-native/binary (M114) and sublinear/ANN are the candidates.
- **B3 — data scaling that is competitive.** $Q_S(n)$ must improve with data at a rate comparable to $Q_T(n)$. Independent of B1; cheaper to test (M116).
- **B5 — a certificate.** Free-probability spectral prediction and the BBP-cascade/RMT bulk diagnostic make "added degree" a measured quantity, and predict the ceiling _before_ running.
- **B4 — sequence transfer (long-term, contingent on B1-B3).** A closed-form analog for tokens/next-token. Only after 1-3.

**Measured status of the conjunction today:** B1 FALSE for single-stage (M108 ladder flat); B2 PARTIAL (M114's algorithmic 8x is pending); B3 UNMEASURED; B5 UNUSED. M115 and M116 are the gating tests.

---

## 3. The theoretical foundation

- **Dandi, Vilucchio, Arnaboldi, Tabanelli, Krzakala 2026 — _Deep Learning as Neural Low-Degree Filtering_ (Neural LoFi), arXiv 2605.13612.** Hierarchical feature learning = an iterative spectral procedure: each layer selects directions with maximal _accessible low-degree correlation to the label_; depth composes degrees. Improves over lazy random-feature baselines — the program's exact baseline (frozen 0.2153).
- **Defilippis, Krzakala, Loureiro, Maillard 2026 — _Optimal scaling laws in learning hierarchical multi-index models_, arXiv 2602.05846.** A target-agnostic **spectral estimator achieves the optimal rates** for hierarchical targets through a **cascade of phase transitions**, and is the small-learning-rate limit of gradient descent. The closed-form spectral stack is what backprop converges to in that regime.
- **Bardone, Merger, Goldt 2026 — _A theory of learning data statistics in diffusion models_, arXiv 2603.12901.** Pairwise statistics: linear sample complexity; the 4th cumulant: cubic — **unless higher-order statistics share a correlated latent structure, which restores linear complexity**. This is the precise justification for a _stack_: shared latent structure across layers makes higher cumulants cheap to learn (B3's theory).
- **Park, Bocchi, D'Amico, Lucini, Aarts 2026 — _Spectral phase transitions and trainability_, arXiv 2606.28486.** SGD training itself induces a BBP transition (signal detaches from the bulk). Validates the RMT/BBP diagnostic.
- **Mergny, Ko, Krzakala 2024 — arXiv 2403.03695.** BBP extended to _block-structured_ (domain-structured) spiked models — the per-domain regime's theory.
- **Chhaibi, Daouda, Kahn 2021 — arXiv 2111.00841.** Free-probability spectral metrics computed _before_ training correlate ~85% with final test accuracy — the ceiling-prediction certificate.
- **Howard, Jefferson, Maiti, Ringel 2025 — arXiv 2405.06008.** Wilsonian RG of NNGPs: "learnable vs unlearnable modes," universality classes — the collapse as a mode-integrability statement.

**Physics/math tools mapped to the requirements** (from the 6 Aug theory session): the degree hierarchy = the **cumulant hierarchy** (Wick's theorem; Anandkumar et al. 1210.7559 is the canonical cumulant-based learner); depth = **renormalization-group coarse-graining** (Mehta-Schwab 1410.3831; a Hebbian RG tokenizer exists, 2503.02057); budgeted expressivity = **tensor networks** (MPS canonical/Schmidt truncation; Saiapin-Batselier 2512.02547); a **degree-k Chebyshev polynomial filter on the cross-covariance** is the concrete closed-form layer rule (no direct prior found for this framing); the **BBP threshold** is the n-adaptive capacity schedule.

---

## 4. Prior art (registered, with recall caveats)

Two fresh arXiv sweeps (6 Aug 2026), claims registered before searching. **Anchors found:** Dandi 2605.13612 (by title AND topic), Mehta-Schwab 1410.3831 (by title AND topic), Anandkumar 1210.7559 (by title). **Recall flags:** `abs:"stacked sparse coding"` returned 0 (the concept exists via "deep dictionary learning" — Tariyal 1602.00203); the exact-title query for Stoudenmire-Schwab "Supervised Learning with Tensor Networks" returned 0 (paper exists, 1605.05775) — the tensor-network-ML area is present and undercounted by these queries.

**What is already done (displaces any novelty claim):**

- **Stacked/deep dictionaries:** Tariyal et al. 2016 (1602.00203, greedy multi-level DL); Mahdizadehaghdam et al. 2018 (1803.04022, "performance increases with layers"); Tang et al. 2020 DDLCN (2005.10940); Singhal-Majumdar joint DDL (1912.10801/10804).
- **Layerwise closed-form / no-backprop deep learning:** PCANet (Chan et al. 1404.3606 — cascaded PCA + binary hashing); Saak features (1902.09107); Forward Thinking (1706.02480); HSIC Bottleneck (1908.01580); gradient-isolated learning (1905.11786); Dual Propagation (2302.01228); FFzero (2603.24790).
- **Task-driven/discriminative dictionary learning:** Mairal-Bach-Ponce 2012 (1009.5358) and a large literature.
- **Low-degree / spectral feature learning:** Dandi 2026 (2605.13612); Ren-Wang-Lee 2026 depth separation via layerwise spectral reconstruction (2607.25200); Yang-Li low-degree feature learning (2603.21062, 2512.20562); Kunisky-Wein-Bandeira low-degree method (1907.11636); Holmgren-Wein counterexamples to the low-degree conjecture (2004.08454) — **the framework is a heuristic, never a bound**.
- **Cumulant/higher-order construction:** Anandkumar et al. (1210.7559); Viswanathan-Park cumulant probe of LLMs (2510.04285).
- **BBP/phase-transition learning:** Park et al. 2026 (2606.28486); Mergny-Krzakala 2024 (2403.03695); Defilippis et al. 2026 (2602.05846).
- **Free probability / OGP:** Chhaibi et al. (2111.00841); Pastur et al. (2001.06188, 2011.11439); Collins-Hayase (2103.13466); Gamarnik-Zadik OGP threshold (1711.04952); the OGP-not-always-predictive caveat (2411.01836).
- **Kronecker/structured dictionaries:** STARK (1711.04887), separability (2007.03800), minimax lower bounds (1605.05284) — heed the sample-complexity limits.
- **Leverage/sketching for dictionaries:** SQUEAK (1803.10172) — ridge-leverage kernel dictionaries; the program's M108 ridge-leverage is this.

**What is NOT done (the open space):** no published work measures a closed-form low-degree stack under this program's sealed protocol — matched atoms/MACs vs a trained transformer, per-domain, 345-class/6-domain DomainNet subsample, with a **training-cost ledger** as the primary operand. The degree-k Chebyshev-filter layer rule as a _closed-form label-coupled selection_ is not found as a framed method (the closest is the spectral estimator of 2602.05846). That is the program's niche: **measurement, never novelty.**

**Second sweep (8 Aug 2026) — data-scaling/exploitation literature.** The export API was 429-rate-limited for the entire session (recorded as residual error, VOID as evidence of absence); this sweep used the arXiv web-search UI instead. **Verified anchors:** Bordelon-Canatar-Pehlevan 2002.02561 (spectrum-dependent learning curves, ICML 2020); Canatar-Bordelon-Pehlevan 2006.13198 (spectral bias / task-model alignment, Nat. Comms.); Spigler-Geiger-Wyart 1905.10843 (measured kernel learning-curve exponents: MNIST β≈0.4, CIFAR10 β≈0.1); Liu-Liao-Suykens 2010.02681 (high-dim KRR beyond double descent); Kaplan 2001.08361; Hoffmann 2203.15556; Mei-Montanari 1908.05355; Fedus 2101.03961; Clark 2202.01169.

**Topic hits (all verified present):** the **random-feature ridge regression scaling-law literature is deep and active**: Defilippis-Loureiro-Misiakiewicz 2405.15699 (dimension-free deterministic equivalents → scaling laws for RFRR — the same Defilippis as 2602.05846); Ruben-Tong-Chaudhry-Pehlevan 2412.05418 (RF ensembles, no-free-lunch scaling); Xiao-Hu-Misiakiewicz-Lu-Pennington 2205.14846 (precise learning curves + higher-order scaling limits for dot-product kernels); Hu-Lu-Misiakiewicz 2403.08160 (RFRR beyond the linear scaling regime); Atanasov-Zavatone-Veth-Pehlevan 2405.00592 (scaling and renormalization in high-dim regression); Bordelon-Atanasov-Pehlevan 2402.01092 (dynamical model of neural scaling laws); Bahri-Dyer-Kaplan-Lee-Sharma 2102.06701 (explaining neural scaling laws, PNAS); Maloney-Roberts-Sully 2210.16859 (solvable model); Zavatone-Veth-Pehlevan 2303.00564 (learning curves of deep structured Gaussian feature models); Kramp-Lindner-Helias 2602.23039 (RF scaling dynamics with power-law kernel eigenvalues); Paquette-Xiao-Zhu 2603.14578 (power-law spectrum of the RF model); Bordelon-Mori 2602.04774 (optimal LR schedules + scaling for RF); Wu-Chen-Misiakiewicz-Mondelli 2603.05691 (weak-to-strong in RFRR); Bosch et al. 2204.02678 (precise asymptotic RF learning curves).

**What this means for the program:** M116's finding (the frozen family's Q(n) steeper than the trunk's, with a crossing) is a **predicted class of phenomenon** in this literature — the learning-curve exponent is governed by the feature Gram eigenspectrum (a longer-tailed spectrum ⇒ slower saturation ⇒ steeper Q(n)) — so the theory is prior art and the program can use it as a **predictive certificate** (§5.5 M121). What is NOT found in any of these: the **sealed measurement** of those exponents and crossing for a frozen _patch-dictionary code_ vs a frozen _transformer trunk_, per-domain, at matched cost, on a 345-class/6-domain benchmark. That remains the program's niche.

**Recall flags:** ITQ (Gong et al.) is not on arXiv (CVPR venue; 0 hits); `"feature eigenspectrum" transfer` 0 hits (the concept lives under "learning curves"/“spectral bias"); `"linear probing" scaling` returns 225 hits but none measure the data-scaling law of a linear probe on a frozen random-feature code (the hits are interpretability/foundation-model probing).

---

## 5. The plan

### 5.1 M115 — closed-form LoFi stack (B1 + the training-cost axis, sealed)

**Question.** Does closed-form depth raise the effective degree (lift the single-stage ceiling) at a measured training-cost ratio vs dense?

**Arms** (shared subsample, M108-exact whitener/dictionary construction, closed-form ridge head penalty 1.0, per-domain eval):

- **L=0** — the sealed frozen 3072-dictionary arm (reference 0.2153 @ 254.6M MACs; re-measured in-run and gated, or quoted with the deterministic-construction identity note).
- **L=1** — one label-coupled spectral layer: compute the cross-covariance between the frozen pooled codes and the (centered) one-hot labels; select the top-k label-correlated directions via a **degree-k Chebyshev polynomial filter** on the cross-covariance spectrum (closed-form, degree-controlled); project; re-whiten; fit a second dictionary on the projected codes; triangle-encode; pool; ridge. Effective degree ~4.
- **L=2** — repeat on L=1's codes (effective degree ~8), ridge on top.

**Training-cost ledger (primary operand, per A1's closed-form accounting).** Each layer costs O(data·atoms) encode + O(d·k) spectral filter + closed-form solves — no epochs, no backprop. Report total training ops vs dense's ledgered training cost to reach the same accuracy (M107/M109 evidence where available). The gate is the **ratio**.

**Gates.**

- **t1:** L=0 reproduces the sealed 0.2153 within 0.002 (instrument), or the run voids.
- **KS1 (degree):** best L does not beat L=0 by >= +0.01 overall -> closed-form depth does not lift the single-stage ceiling -> **B1 fails** and the long-term thesis is dead on the quality axis (scope: per-domain + fixed-accuracy training-cost wins remain).
- **KS2 (training-cost):** closed-form training cost to reach accuracy A vs dense's cost to reach A; fired (fail) if the ratio < 10x at the registered accuracy point.
- **Reported (frontier):** per-domain accuracy at matched MACs; the accuracy-cost point of each L.

**Registered prediction.** The theory says a spectral estimator is optimal for hierarchical targets (2602.05846) and higher cumulants become cheap with shared latent structure (2603.12901), so a modest degree-lift is expected; whether it clears +0.01 is the open question. **Cite:** 2605.13612, 2602.05846, 2603.12901, 2512.02547, 1404.3606, 1602.00203, 1009.5358.

**RESULT (sealed 7 Aug, `logs/results/v16/m115_lofi/evidence.json`, admissible).**

| Arm                           | Effective degree | Accuracy                               | Per-domain (clipart..sketch)        |
| ----------------------------- | ---------------- | -------------------------------------- | ----------------------------------- |
| L=0 (frozen 3072)             | 2                | **0.2153** (t1 delta +0.00000 vs M113) | .2300 .0649 .1121 .3110 .2344 .1199 |
| L=1 (spectral k=256, VQ 2048) | 4                | 0.1361                                 | .1271 .0323 .0730 .2157 .1432 .0550 |
| L=2 (repeat)                  | 8                | 0.1121                                 | .0971 .0241 .0524 .1939 .1088 .0414 |

- **KS1 FIRED** (best L = L=0; delta 0.000 < +0.01). Closed-form depth via the registered label-coupled spectral layer rule does NOT lift the single-stage ceiling; every deeper layer is strictly WORSE, uniformly across all six domains (~40-50% relative drop each layer). **B1 fails on the quality axis.**
- **KS2 NOT fired** under the primary (head-training-only) formula: ratio 479.2x (dense closed-form head fit, w=768) / 128.8x (SGD head, epochs·n·3·params). **However, the disclosed symmetric sensitivity — counting the dense trunk forward over the corpus as a corpus pass, exactly as the sparse encode pass is counted — is 1.11x.** The ≥10x training-cost headline does NOT survive symmetric accounting: the sparse stack's corpus cost is dominated by its O(n·12288²) ridge Gram, dense's by its trunk forward at the cheapest point reaching A (M107 r56, 0.2450 @ 367.5M/img). The 479x figure is an artifact of excluding the dense trunk forward, not a property of the stack.
- **Scope note (registered diagnostic, not an excuse):** the registered layer rule REPLACES the code at each layer (projection to top-256 label-correlated directions + whitening + VQ-triangle re-encode; no skip/residual). The degree-2 base is destroyed by the lossy re-encode, so "depth adds degree" was never isolated from "the replacement code is a worse representation of the label". A residual/skip variant (ridge on [L=0 code; L=1 code]) is the natural follow-up if the depth thesis is ever re-opened; under the registered rule the gate fired and stands.
- **Consequence (v18 §6, KS1-fails rule):** the buy-back is impossible on the quality axis; the honest program role is scoped to A5-KS2-style per-domain inference wins and fixed-accuracy training-cost wins. No LLM path. KS2's symmetric-accounting finding also removes the "training-cost ratio ≥10x" claim: the stack is NOT 10x cheaper to train than dense under symmetric accounting.

### 5.2 M116 — data scaling Q(n) (B3, sealed)

Fit the best stack (from M115) and dense at subsample sizes n of the shared corpus; compare $Q(n)$ curves. Test the Bardone-Goldt prediction: with shared latent structure across the stack's layers, higher-order statistics become learnable at linear sample complexity. **Gate:** does the stack's $Q_S(n)$ scale comparably to dense's $Q_T(n)$? Independent of B1.

**RESULT (sealed 7 Aug, `logs/results/v16/m116_scale/evidence.json`, admissible).** The best M115 arm was L=0 (the frozen single-stage 3072-dictionary arm; depth lost KS1), so M116 measures its Q(n) against the cost-matched dense trunk (M109 t1 protocol, DINOv2-small r42 = 215.6M/img vs sparse 254.6M). Features are image functions, computed once into D: memmaps; a closed-form ridge is fitted at each ladder point on the first n rows of the M107-shuffled train order (nested); all points score the same full 34500-row test set. t1 deltas: sparse +0.00000, dense +0.00006.

| n       | Q_S(n) sparse | Q_T(n) dense r42 | gap (T−S) |
| ------- | ------------- | ---------------- | --------- |
| 6,900   | 0.0484        | 0.0737           | +0.0253   |
| 13,800  | 0.0706        | 0.0863           | +0.0157   |
| 27,600  | 0.0952        | 0.0952           | 0.0000    |
| 55,200  | 0.1442        | 0.1326           | −0.0116   |
| 110,400 | 0.1935        | 0.1751           | −0.0184   |
| 138,000 | **0.2153**    | **0.1972**       | −0.0181   |

- **KS NOT FIRED (ratio 1.351 >= 0.5):** $\Delta_S = +0.1668$, $\Delta_T = +0.1235$. The frozen family's data scaling is comparable-to-steeper than the cost-matched dense trunk's, and the gap closes then inverts — sparse overtakes dense between n=27,600 and n=55,200 and holds a 1.8-pt lead at n_max. Per-domain at n_max: sparse wins 4/6 (clipart .2300, infograph .0649, quickdraw .3110, sketch .1199), dense wins 2 (painting .1502, real .2917). **B3 PASSES as registered: a second, independent route to the buy-back, testable without the degree-lift.**
- **Honest scope:** (1) the shape advantage is partly because the 12,288-wide sparse ridge is underdetermined at small n (depressed start), so its gain looks large; the end-state is what matters — it ends AHEAD at matched cost. (2) Corpus training ops at n_max are sparse 5.66e13 vs dense 2.99e13 (~1.9x more; wide-ridge Gram again) — the data-scaling win is about the Q(n) SHAPE, not training cost (consistent with M115 KS2's symmetric-accounting finding). (3) This is the single-stage family; M115 KS1 already ruled out the depth route to B1.

### 5.3 M114 — binary soft-code encode (B2, sealed)

**Status (7 Aug):** sealed CPU run in flight (~17.6 CPU-hr, no evidence yet; the CPU popcount path at 3072 atoms is the 2-3-day scenario in miniature). GPU sign-GEMM backend is implemented and verified bit-identical (smoke matches exactly); the **sealed M114 should be re-run on the GPU backend** (config `device.encode: "gpu"`, disclosed GEMM realization at n·B·A MACs; the registered algorithmic cost 2·P·A·words is realized by the CPU hardware-POPCNT path). Gates unchanged (accuracy floor 0.2053 + cost ratio <= 254.6M/3).

**RESULT (sealed 7 Aug, `logs/results/v16/m114_binary/evidence.json`, admissible, GPU sign-GEMM backend).**

| Arm                              | Accuracy | Total ops | Ops vs float (254.6M) |
| -------------------------------- | -------- | --------- | --------------------- |
| float (M113 random-3072, quoted) | 0.2153   | 254.6M    | 1.0x                  |
| b_random (RANDOM-108)            | 0.1842   | 30.2M     | ~8.4x fewer           |
| c_itq (ITQ-108)                  | 0.1820   | 30.2M     | ~8.4x fewer           |
| d_itq_128 (ITQ-64)               | 0.1765   | 22.3M     | ~11.4x fewer          |

- **KS1 FIRED** (learned hash delta -0.0022 < +0.01): ITQ bits do NOT beat random bits on Hamming distance; the binary-axis result is about bit quantisation, not hash learning. Random projection is as good a bit source.
- **KS2 FIRED** (MAC-axis breakthrough): the cost side PASSED (30.2M <= 254.6M/3) but the accuracy floor FAILED (ITQ-108 0.1820 < 0.2053 = float - 0.01). 108 bits of Hamming distance lose ~3.3 points vs the float triangle code, and neither hash learning (KS1) nor a 64-bit code recovers it. **The MAC axis is NOT won at preserved accuracy.** B2 is PARTIAL: the algorithmic op reduction is real and measured (~8.4-11.4x), but it buys a 3-point accuracy loss.
- **Honest reading:** bit-quantisation at 108 bits is too lossy for the triangle code's usefulness; the loss is in the code, not the hash (random ~ learned). A higher bit budget (e.g. 384-768 bits) is the only registered path to recover accuracy at still-reduced cost; not registered as a milestone, left as a follow-up option.

### 5.4 Diagnostics (B5) — ride along with M115/M116

- **Free-probability ceiling prediction (2111.00841):** compute the frozen pipeline's spectral metrics and predict its ceiling before running the stack.
- **BBP cascade per layer (2606.28486, 2602.05846):** signal eigenvalue vs bulk edge at each layer — "degree added" as a measured quantity.
- **Cumulant observables (2510.04285):** the degree probe on the stack's codes.
- **RMT bulk** on the ridge Gram (the 12288-wide head's spectrum).

### 5.5 Exploitation of the two positives (registered 8 Aug, NOT yet run)

**The mechanism reading of M116.** The frozen family's per-image compute is **data-elastic**: it sits in the 12,288-wide closed-form ridge head, which consumes data productively (Q(n) rises and overtakes the trunk). Dense's per-image compute sits in the frozen trunk, which is data-insensitive (its 768-dim head saturates sooner). The §4 learning-curve theory predicts the Q(n) exponent from the feature Gram eigenspectrum. Five registered follow-up directions exploit this (each gated; none run):

- **M117 — the 2D scaling surface Q_S(C, n).** M108 held n fixed (flat in atoms at the slice), M116 held atoms fixed (rising in n). The joint atoms × data surface is the real test of $Q_S(C \cdot R) \ge Q_T(C)$. **Gate:** is $Q_S(2C, 2n) - Q_S(C, n)$ super-additive over the single-axis gains? If yes, the 0.2153 "ceiling" was a slice artifact and the B-condition becomes a 2D measurement.
- **M118 — binary × data scaling (B2 rescue).** M114 lost 3.3 pts at 108 bits on full n; if the binary code inherits the steep Q(n) (same wide head on binary features), the gap may close at large n and the 8.4x op win is recovered at large data. **Gate:** binary Δ_S >= 0.5·Δ_T on the M116 ladder AND the binary-vs-float gap at n_max <= the full-n gap + 0.01.
- **M119 — per-domain specialists × data scaling (A5 × M116).** Per-domain routed specialists have lower domain entropy; their per-domain Q_d(n) may be steeper. **Gate:** specialist gain ratio >= 0.5 per domain on a per-domain ladder.
- **M120 — head-width mechanism test.** Steepness β as a function of head width W (subsampled widths at fixed atoms). Confirms "data-elastic compute" as a design rule; the theory predicts β from the spectrum — fit and compare. **[folded into M117 as the per-atoms Q(n) steepness, reported in M117's evidence]**
- **M121 — spectral certificate + shaping (B5, now theory-grounded).** Compute the sparse and dense feature-Gram eigenspectra; predict M116's crossing from the learning-curve theory (2002.02561 / 2405.15699); then engineer the code's spectrum (whitening / pooling / dictionary construction) to steepen β. **Gate:** predicted crossing within tolerance of the measured one; a spectral perturbation moves β in the predicted direction.

**M121 RESULT (sealed 8 Aug, `logs/results/v16/m121_spectrum/evidence.json`, admissible).** First measured Gram spectra of the two families on the shared subsample (M=138000): sparse (12288-dim frozen code) eff. rank 8, tail index −2.26, top-1 share 0.31, **captures 11.8% of the one-hot label power in its linear span**; dense (768-dim trunk) eff. rank 30, tail index −1.39, **captures 3.5%**. The Canatar-Bordelon-Pehlevan learning-curve formula (2006.13198 eq. 4, λ=1.0, discrete-measure kernel PCA on the standardised features, null-mode power reported not gated) predicts the sparse learnable error to stay ABOVE dense's for the whole ladder (dense's few captured modes are learned near-instantly) → **predicted crossing: none → KS1 FIRED; KS2 FIRED** (predicted sparse error not below dense at n_max). **The MSE-proxy certificate does NOT predict M116's measured accuracy crossing.** Why (recorded mechanism): (1) the measured accuracy crossing is an ARGMAX phenomenon over 345 one-hot channels that scalar MSE does not track — the shape-level proxy failed; (2) the spectral asymmetry that IS real and survives: the sparse span holds 3.4× more learnable label power (11.8% vs 3.5%), which is why the sparse family has more room to keep improving with data — consistent with M116's overtaking. The "spectral shaping" half of M121 (perturbing whitening/pooling to steepen β) is NOT licensed by this result; a classification-aware certificate (per-class score-margin prediction) is the registered follow-up if the direction is re-opened.

**M117 RESULT (sealed 8 Aug, `logs/results/v16/m117_scale/evidence.json`, admissible).** The 2D atoms × data surface (t1 anchor delta +0.00000 at (3072, 138000)):

| atoms \ n | 34,500 | 69,000 | 138,000    |
| --------- | ------ | ------ | ---------- |
| 1,536     | 0.1028 | 0.1457 | 0.1970     |
| 3,072     | 0.1004 | 0.1530 | **0.2153** |
| 6,144     | 0.0906 | 0.1525 | **0.2249** |

- **KS super-additivity NOT fired** (excess +0.0097 at cell (1536,34500), +0.0101 at cell (3072,69000); margin +0.005). Doubling atoms AND data jointly beats the sum of the single-axis gains. **The 0.2153 ceiling was a slice artifact: the family's quality scales with the JOINT atoms × data budget.** The atoms axis alone is flat-to-negative at small n (a wider head is more data-hungry: 0.1028 → 0.1004 at n=34500) but lifts the ceiling at full data (0.2153 → 0.2249 at n=138000). This is the first positive on the quality axis: B's condition is a 2D surface that rises, not a flat line.
- **M120 (folded in) CONFIRMED:** per-atom Q(n) steepness rises with head width — 0.094 (1536 atoms) → 0.115 (3072) → 0.134 (6144). The "data-elastic compute" mechanism is measured: wider closed-form heads consume data more productively.
- Per-domain at n=138000, atoms 6144 beats 3072 on 5/6 domains (clipart .2425, infograph .0693, quickdraw .3275, real .2393, sketch .1368; painting .1104 ~ tie). Corpus training ops at (6144, 138000) = 1.57e14 (~5.3x the dense r42 corpus total) — the joint-budget win is on the quality axis, not the training-cost axis.
- **Honest scope:** the gains are modest in absolute terms (+0.0096 doubling atoms at full data; +0.0724 on the n axis at 6144 atoms) and cost roughly doubles per atom step. This rescues the B-condition as a _measured 2D surface_, it does not yet make the family competitive with dense on accuracy (dense r224 is 0.54). The registered consequence: the "quality axis is closed" statement (v18 §1/§6) is **rescinded to "flat only on single slices"** — a joint-budget route to the buy-back is measured-open.

**M118 RESULT (sealed 8 Aug, `logs/results/v16/m118_binary_scale/evidence.json`, admissible).** Binary (RANDOM-108 and ITQ-108, M114's exact hashes) Q(n) on M116's ladder (t1: b_random reproduces M114's 0.1842 at delta +0.00000):

| n       | float (M116) | RANDOM-108 | ITQ-108    |
| ------- | ------------ | ---------- | ---------- |
| 6,900   | 0.0484       | 0.0276     | 0.0285     |
| 27,600  | 0.0952       | 0.0675     | 0.0706     |
| 55,200  | 0.1442       | 0.1202     | 0.1219     |
| 110,400 | 0.1935       | 0.1657     | 0.1645     |
| 138,000 | 0.2153       | **0.1842** | **0.1820** |

- **KS1 NOT fired:** the 108-bit code INHERITS the frozen family's steep Q(n) — gain ratio 0.94 (random) / 0.92 (ITQ) vs the float 0.1668 gain. The data-elastic-head mechanism applies to binary codes too.
- **KS2 FIRED:** the bit loss does NOT shrink with data — the binary-vs-float gap grows 0.021 → 0.031 (random, fired by +0.0003) and 0.020 → 0.033 (ITQ) against the +0.01 tolerance. M114's ~3.3-point loss is intrinsic to the 108-bit code (a constant ~0.02-0.03 gap at every data scale), NOT a small-n artifact. **Data does not buy back the bit loss; M114's KS2 failure is not rescued at scale.** The 8.4x op win remains real at every n, never at preserved accuracy.

**M119 RESULT (sealed 8 Aug, `logs/results/v16/m119_specialist_scale/evidence.json`, admissible).** Per-domain specialist Q_d(n) (A5 construction, 512 atoms/domain, 5-point per-domain ladder, full domain test) vs the M116 global arm's per-domain gains:

| domain    | specialist full-data | specialist gain | ratio vs global gain |
| --------- | -------------------- | --------------- | -------------------- |
| clipart   | 0.1936               | +0.1675         | 0.74                 |
| infograph | 0.0638               | +0.0472         | 0.77                 |
| painting  | 0.1111               | +0.0798         | 0.73                 |
| quickdraw | **0.3245**           | +0.1560         | 0.99                 |
| real      | 0.1948               | +0.1536         | 0.66                 |
| sketch    | 0.1240               | +0.0964         | 0.82                 |

- **KS NOT fired:** every domain's specialist gain clears the 0.5x threshold (ratios 0.66-0.99; quickdraw essentially 1.0). **Per-domain specialists inherit the data-scaling steepness on their own data.** The A5 x M116 composition holds.
- At full data the 512-atom specialist matches or beats the global 3072-atom arm's per-domain accuracy on 4/6 domains (quickdraw +0.0135, sketch +0.0041, infograph/painting ~tie; clipart -0.036, real -0.040) at ~5.6x fewer per-image MACs (specialist ~45M vs global 254.6M) — A5's per-domain cost win re-confirmed at scale AND with steep per-domain data scaling.
- **Honest scope:** the specialist underperforms the global arm on clipart and real at full data (the global arm's 6x more head capacity wins there), so the specialist story is a cost win on most domains, an accuracy win on two, and a wash on two — never an across-the-board accuracy dominance.

**Honest scope:** these are registered-for-execution directions, not claims. The theory is prior art (§4 sweep); the program's contribution remains the sealed measurement under this protocol. M117 is the highest-value (it decides whether the ceiling was an artifact); M121's certificate fired (the MSE proxy does not predict the accuracy crossing), leaving M117 as the decisive next measurement.

---

## 6. Decision rules (sealed outcomes in brackets)

- **M115 KS1 passes** (depth lifts the ceiling >= +0.01): the long-term thesis is alive; pursue the stack at scale, register the **sequence-transfer probe (B4)** as the next long-term question, and keep the training-cost ratio as the headline. **[sealed: KS1 FIRED — depth does not lift the ceiling]**
- **M115 KS1 fails** (depth saturates): B1 false -> the buy-back is impossible on the quality axis; the honest program role is scoped to A5-KS2-style per-domain inference wins and M115-KS2-style fixed-accuracy training-cost wins. No LLM path. **[sealed: applies — B1 false; KS2's >=10x training-cost claim also fails symmetric accounting (1.11x)]**
- **M116 passes** (data scales): a second, independent route to the buy-back, testable without the degree-lift. **[sealed: PASSES — ratio 1.351, sparse overtakes dense r42; B3 positive as registered]**
- **M114 lands** (accuracy within 0.01 at <= 1/3 cost): the MAC axis is won at preserved accuracy — the first time. **[sealed: KS2 FIRED — cost side passed (8.4x fewer ops) but accuracy floor failed (0.1820 < 0.2053); B2 partial]**

---

## 7. What is revisited vs v17

The re-scoping table in §1; Option 1/Arm S re-opened on cost axes; the "consolidate" licenses rescinded; M109's MAC claim scoped to the global arm; M113's ceiling claim scoped to single-stage; the Ghorbani reading scoped to the accuracy axis.

## 8. What is not changed

All sealed v16/v17 evidence stands as mechanism facts. The shared subsample (digest 63f590097008f749f3f1828b29d6f154de7b21a6828a7b017ac141c0615fa09d), the M108 exact whitener/dictionary arithmetic, the device precondition (HIP_VISIBLE_DEVICES=1, gfx1201), and the protocol rules (registered before measurement; no novelty claims; per-domain reporting; matched-cost comparison) are unchanged.

## 9. References (arXiv IDs)

2605.13612 (Neural LoFi), 2602.05846 (hierarchical multi-index scaling), 2603.12901 (learning data statistics / diffusion information exponent), 2512.02547 (TN feature learning), 2607.25200 (depth separation), 2603.21062 & 2512.20562 (low-degree feature learning), 2606.28486 (BBP in training), 2403.03695 (BBP block-structured), 2111.00841 (free probability predicts accuracy), 2405.06008 (Wilsonian RG of NNGPs), 1907.11636 (low-degree method), 2004.08454 (low-degree counterexamples), 1210.7559 (tensor decompositions), 2510.04285 (cumulant probe of LLMs), 1410.3831 (variational RG = deep learning), 1605.05775 (supervised tensor networks), 1404.3606 (PCANet), 1902.09107 (Saak), 1706.02480 (Forward Thinking), 1908.01580 (HSIC bottleneck), 2302.01228 (dual propagation), 1009.5358 (task-driven DL), 1602.00203 (greedy DDL), 1803.04022 (parametric DDL), 2005.10940 (DDLCN), 1711.04887 (STARK), 1605.05284 (Kronecker lower bounds), 1803.10172 (SQUEAK), 1711.04952 (OGP threshold), 2411.01836 (OGP caveat), 2103.13466 (free probability Jacobians), 2503.02057 (Hebbian RG tokenizer).

**Added 8 Aug (data-scaling/exploitation sweep):** 2002.02561 (spectrum-dependent learning curves), 2006.13198 (spectral bias / task-model alignment), 1905.10843 (kernel learning-curve exponents, measured), 2010.02681 (high-dim KRR beyond double descent), 2405.15699 (dimension-free deterministic equivalents / RFRR scaling laws), 2412.05418 (RF ensemble no-free-lunch scaling), 2205.14846 (precise learning curves + higher-order scaling limits), 2403.08160 (RFRR beyond linear scaling regime), 2405.00592 (scaling and renormalization in high-dim regression), 2402.01092 (dynamical model of neural scaling laws), 2102.06701 (explaining neural scaling laws), 2210.16859 (solvable model of neural scaling laws), 2303.00564 (learning curves of deep structured Gaussian features), 2602.23039 (RF scaling dynamics, power-law eigenvalues), 2603.14578 (power-law spectrum of RF model), 2602.04774 (optimal LR + RF scaling), 2603.05691 (weak-to-strong in RFRR), 2204.02678 (precise asymptotic RF learning curves).
