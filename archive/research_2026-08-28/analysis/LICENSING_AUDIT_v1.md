# Licensing audit — product transition (draft v1)

**21 August 2026.** The goal: commercial use without legal surprises.
Rule of thumb inherited from the registered C5 stance: **permissive
only (Apache-2.0 / MIT / BSD-class), cite methods but copy nothing
unlicensed, and record the license of every artifact at
registration.** Nothing below is legal advice; the starred items
need a real review before anything is sold.

## 1. Code

| Component                                            | License status                                                                                           | Verdict                                                                                                                                                    |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `geode/*` (our own code)                             | Unlicensed so far                                                                                        | Choose Apache-2.0 or MIT at first public release; pick one before any third party runs it                                                                  |
| `geode/erasure.py` (LEACE, copied verbatim per M179) | LEACE repo (EleutherAI) is MIT-class \*                                                                  | VERIFY at import time and record the exact file + license + URL in the module provenance (the M179 entry already demands provenance; add the license line) |
| Crypto/dependency stack                              | Registered C5 list: OpenZeppelin MIT, Zama TFHE BSD-3, MP-SPDZ BSD-3, SecretFlow Apache-2, OpenFHE BSD-2 | OK, but pin versions and regenerate the list per release                                                                                                   |
| Classical recipe (ZCA, dictionary, SPM, VLAD, ridge) | Methods are literature; our implementations are original                                                 | Cite, don't copy — the standing rule; no code was taken                                                                                                    |

## 2. Model weights / checkpoints

| Asset                                            | License (registered or to verify)                                                | Commercial use                                                                                                              |
| ------------------------------------------------ | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| DINOv2 (trunk for M261, dense ladder)            | Apache-2.0 (registered in the v24 cell)                                          | ✓ Permitted                                                                                                                 |
| DINOv3 (deferred SOTA cell)                      | **Bespoke license** (registered in the v24 cell) \*                              | ✗ NOT until reviewed — do not touch commercially before reading the terms; the deferred cell stays research-only by default |
| Language-arm checkpoints (M262/M263)             | Policy: Apache-2.0/MIT only (e.g., GPT-2-class, Mistral-7B-class, Qwen2.5-class) | ✓ subject to per-checkpoint verification at download                                                                        |
| Whisper (audio trunk, M266)                      | MIT                                                                              | ✓ Permitted                                                                                                                 |
| Wav2Vec2-class (audio trunk, M266)               | Apache-2.0                                                                       | ✓ Permitted                                                                                                                 |
| Anything Llama-community-licensed or Gemma-style | Use-restriction licenses                                                         | ✗ Excluded by policy                                                                                                        |
| Trained heads/probes we fit                      | Our own weights, derived data may matter (see §3)                                | ✓ as artifacts; the data they were fit on carries its own terms                                                             |

## 3. Datasets — the real risk area \*

| Dataset                                           | Known terms                                        | Product status                                                                                                                                                                   |
| ------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ImageNet-1k                                       | Research/non-commercial terms (academic agreement) | Fine for the M261 anchor measurement and validation; **commercial training/serving against it needs review** — and DINOv2 already saw it (contamination is declared, not hidden) |
| DomainNet                                         | Research-only release                              | Fine for sealed measurements and whitepaper numbers; not redistributed                                                                                                           |
| MNLI, SST-2, IMDb                                 | Research-use releases                              | Evaluation OK; review before any commercial product training                                                                                                                     |
| LibriSpeech                                       | CC-BY-4.0                                          | ✓ Permissive — Tier-3-safe                                                                                                                                                       |
| Speech Commands v2                                | CC-BY-4.0                                          | ✓ Permissive — Tier-3-safe                                                                                                                                                       |
| ESC-50                                            | CC-BY-NC                                           | ✗ Excluded by policy (non-commercial)                                                                                                                                            |
| wikitext-103                                      | Derived from Wikipedia text (CC-BY-SA lineage) \*  | Verify attribution obligations before any product use                                                                                                                            |
| Synthetic/first-party data (probes, fingerprints) | Ours                                               | ✓                                                                                                                                                                                |

Rule: the risk splits into three tiers, and most commercial
blockage fears belong to only the third:

- **Tier 1 — evaluation and measurement.** No dataset license
  blocks scoring a model on held-out data, and every sealed number
  in this paper is Tier 1. In this sense the common intuition is
  right: benchmark use is effectively universal, licensed or not.
- **Tier 2 — using pretrained weights.** The weights carry the
  model's own license, not the training data's. This is why
  Apache-2.0 DINOv2 is commercially usable even though LVD-142M
  contains ImageNet: training-data provenance sits with the
  trainer, not with the user of the weights.
- **Tier 3 — redistribution, or first-party commercial training
  products.** Here the dataset's own terms bind: republishing the
  dataset, training OUR OWN foundation model on it for a product,
  or shipping dataset content (IMDb reviews, Wikipedia-derived
  text). Permissive datasets (MNIST-class, CC-licensed corpora)
  are fine; ImageNet/DomainNet-class research releases are not.

So: nothing above blocks measuring, validating, or shipping arms
built on permissively-licensed checkpoints. The dataset review
gates only Tier 3 activities — which the current plan does not
include (we fit heads on frozen, permissively-licensed trunks; we
do not train foundation models).

## 4. Actions (ordered)

1. **License field on arms (registry change).** An arm's measured
   record gains a `license` entry (per-artifact: code, weights,
   data) at registration — cheap, fits the append-only design, and
   makes every future audit a lookup instead of an excavation.
   → added to the M264 production-gap spec.
2. **Provenance lines in vendored code.** `geode/erasure.py` gets
   its exact source license recorded (M179 already demands
   provenance; add the license).
3. **Checkpoint whitelist.** M262/M263 select from a written
   whitelist of Apache-2.0/MIT checkpoints; the whitelist is
   checked at download and recorded in the arm's evidence.
4. **DINOv3 review gate.** The deferred SOTA cell stays research
   unless/until the bespoke license is read; anything built on
   DINOv3 commercially is gated on that reading.
5. **External counsel before first sale** for the starred items
   (this slots next to the registered M188 legal gate).
