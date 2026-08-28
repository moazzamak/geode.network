# RunPod compute budget — scale-up wave (draft v1)

**21 August 2026. Planning estimate — prices must be re-verified at
booking (RunPod rates float).** This covers the registered,
not-yet-dispatched cells (M260, M261–M264, M265) plus the deferred
SOTA comparison cell.

**COMPUTE POLICY (user decision): local machine first.** The local
RX 9070 XT (16 GB VRAM, the machine that already ran the M222
DINOv2-small extraction) is the default for every cell it can
finish in reasonable wall time. RunPod is invoked ONLY when the
local machine (a) cannot hold the workload (VRAM-bound) or (b)
would take on the order of months. Everything below that runs on
RunPod is costed on **spot instances with checkpoint discipline** —
the evidence rule's reproducibility hashes and the M222
selection-digest feature cache make interruption cheap to recover
from. No number is a commitment; all are orders of magnitude,
anchored where the sealed record gives a measured rate.

## 1. Anchors (sealed rates we can plan from)

- M222 (DINOv2-small feature extraction, 32×32, one consumer GPU):
  20,010 rows cold-start in ~3.0 h (~1.8 rows/s under memory
  pressure); 34,500 test rows warm in ~49 s (~700 rows/s).
- ImageNet-scale extraction (224×224) is the dominant cost: a
  DINOv2-small forward pass is ~5.5 GFLOPs/image; one A100 80GB
  sustains ~40–60 images/s in practice → **~6–9 h per million
  images**, before overheads.
- Published linear-probe protocol (M261 anchor recipe): with
  features cached, the grid (13 lr × 2 layers × 2 concat) is a
  matrix-fit per cell — hours on one card, not days.
- The 409,832-row wide ridge fits are CPU-RAM bound (~36 GB), not
  GPU bound — any box with ≥64 GB RAM handles them.

## 2. Phase budget

| Phase                        | Cells                                                                                            | Workload                                                  | Instances                | Est. wall time         | Est. cost (spot) |
| ---------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------- | ------------------------ | ---------------------- | ---------------- |
| 0 — instrument & anchors     | M260 ablation; M265 decomposition; penalty certs                                                 | Cached-feature fits, CPU-heavy                            | 1× RTX 4090 (+64 GB RAM) | 2–4 days               | **$40–90**       |
| 1 — vision arm               | M261 ImageNet-1k: DINOv2-small extraction (1.28M imgs), linear-probe grid, deep-patch SPM encode | ~8 h extraction + ~4 h probes/encode                      | 1× A100 80GB             | 2–3 days (with reruns) | **$120–250**     |
| 2 — language arms            | M262: small-LM features on MNLI (392k pairs), SST-2, IMDb; trained task heads                    | Few hours extraction + probe fits                         | 1× RTX 4090              | 1–2 days               | **$30–70**       |
| 3 — generative arm           | M263: no training — serving, refusal tags, behaviour-diff baseline, probes                       | Inference + eval only                                     | 1× RTX 4090              | ~1 day                 | **$20–50**       |
| 4 — M264 production-gap spec | spec-only, no compute                                                                            | —                                                         | —                        | —                      | **$0**           |
| 5 — optional SOTA cell       | Deferred v24 cell: DINOv3 linear probes on ImageNet-1k (six sizes, extraction-dominant)          | Feature extraction ×6 sizes (largest ~5–10× DINOv2-small) | 1× A100 80GB             | 3–5 days               | **$250–550**     |

**Base wave (phases 0–4): ≈ $210–460.** With the DINOv3 SOTA
comparison (phase 5): **≈ $460–1,010.** Round with a 1.5× rerun
margin: **budget $700 (base) / $1,500 (with SOTA cell).**

### 2.1 Local-first reallocation

Under the compute policy, the RunPod portion shrinks sharply:

| Phase                                   | Local (RX 9070 XT)                       | RunPod only if                               |
| --------------------------------------- | ---------------------------------------- | -------------------------------------------- |
| 0 — anchors & instruments               | Fully local (CPU-bound fits + 16 GB GPU) | —                                            |
| 1 — ImageNet extraction (1.28M × 224px) | ~18–36 h at ~10–20 img/s                 | time-boxed delivery, or batches > 16 GB VRAM |
| 2 — language features                   | Fully local                              | —                                            |
| 3 — generative arm                      | Fully local                              | —                                            |
| 5 — DINOv3 linear probes                | All sizes fit 16 GB at small batches     | time-boxed delivery; license review gate     |

**Revised worst-case cloud spend: $120–250 (phase 1 time-box) +
$250–550 (phase 5 time-box) ≈ $370–800; likely much less, since
most phases never leave the local machine.**

### 2.2 Hardware sizing (the concrete answer)

Everything in the plan is **single-GPU and inference/extraction +
matrix fits** — there is no multi-GPU training anywhere. Per-phase
requirements, from the actual workload shapes:

| Phase                     | GPUs          | Peak VRAM                                                        | RAM                          | Disk (datasets + caches)                                                                                                     | Local OK?                          |
| ------------------------- | ------------- | ---------------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 0 — anchors & instruments | 1 (any 4 GB+) | < 2 GB                                                           | 64 GB (the 36 GB ridge fits) | < 5 GB                                                                                                                       | ✓ fully                            |
| 1 — ImageNet vision arm   | 1             | 2–4 GB at batch 128 (ViT-S: 44 MB weights + ~350 MB activations) | 64 GB                        | ImageNet raw ~150 GB; pooled codes ~11 GB; probe features ~4 GB; **token cache ~194 GB if kept** (stream instead to skip it) | ✓ extraction; RunPod only for time |
| 2 — language arms         | 1             | 3–6 GB at batch 32 (BERT-base 512 tok)                           | 32 GB                        | datasets < 1 GB; features ~1–2 GB                                                                                            | ✓ fully                            |
| 3 — generative arm        | 1             | 4–6 GB (1B-class fp16 + KV cache)                                | 32 GB                        | checkpoint ~2–4 GB                                                                                                           | ✓ fully                            |
| 5 — DINOv3 linear probes  | 1             | ≤ 16 GB at batch 4–8 even for giant (weights ~2–4 GB fp16)       | 64 GB                        | features ~4 GB/size; **giant token cache ~774 GB — never keep, stream or skip**                                              | ✓ small sizes; RunPod for time     |

Key numbers behind the table: ViT-S forward is ~5.5 GFLOPs/image;
activation footprint at batch 128 ≈ 128 × 197 tokens × 384 × 2 B ×
12 layers ≈ 350 MB (attention matrices transient). Cached feature
sizes: probe features 1.28M × 768 × 4 B ≈ 3.9 GB; pooled codes
1.28M × 2048 × 4 B ≈ 10.5 GB; deep-patch token cache at 224px
1.28M × 197 × 384 × 2 B ≈ 194 GB — the only number that matters
for disk planning, and it is optional (stream-then-discard).

**Bottom line: no cell needs more than one GPU, none needs more
than 16 GB VRAM, and 64 GB RAM + ~200 GB free disk cover the whole
wave — if token caches are streamed. The only reasons to rent are
wall time and the DINOv3 license gate.**

## 3. What is deliberately NOT in this budget

- **Fine-tuning large backbones** (DINOv2-g fine-tune alone ≈ 1k
  A100-GPU-hours ≈ $1,200+ spot): the registered vision protocol
  is frozen trunks + trained heads. Small-model LLM fine-tuning is
  the one exception — see §3.1.
- **Training an LLM from scratch**: out of scope; costs are orders
  of magnitude larger than the envelope (§3.1 covers fine-tuning
  only).
- **Persistent storage**: ImageNet raw ≈ 150–200 GB + cached
  features ≈ 50–100 GB → network volumes at ~$0.07/GB/month ≈
  **$15–25/month**; delete after sealing, keep digests (the
  standing feature-evidence policy).
- **Serving/fleet costs**: the deployed multi-arm service is a
  separate operational budget (M208 Bittensor vs hosted fleet
  decision), not exploration.

### 3.1 Small-model fine-tuning envelope (revised policy)

Policy change: LLM fine-tuning is no longer excluded outright; it
is admitted within a hard ceiling of **$999**, local-first, only
behind a pre-registered breakthrough criterion. What that buys:

- **Feasible:** LoRA/QLoRA fine-tuning of ≤1.5B permissive base
  models (Qwen2.5-0.5B/1.5B, SmolLM-135M/360M) on domain corpora —
  single GPU, hours per run. This is domain adaptation of publisher
  checkpoints, not pretraining.
- **Local-first:** the RX 9070 XT (16 GB) holds a 0.5B-1.5B QLoRA
  run comfortably → marginal cost near zero. Cloud is the
  time-boxed fallback: L40S/A100 spot at $0.5-1/hr means a 10-20
  GPU-hour run is $5-20; even ~30 runs stay under $600.
- **Pre-registered criterion (house rule):** a fine-tune cell
  dispatches only when a registered measured gap exists — e.g.,
  M268's generalist-vs-specialist reading shows the generalist
  misses a target held-out threshold that the primitive tier
  cannot close — with a pre-registered success threshold on the
  held-out split. The criterion is written before spend, never
  after.
- **Licensing:** base model permissive (Apache-2.0/MIT); the
  fine-tune corpus tier 3 permissive (OpenMathInstruct-2-class,
  verify at selection). Fine-tuning does not transfer corpus terms
  to the base model, but the redistribution stance still requires
  a permissive corpus.
- **Still excluded:** from-scratch pretraining, any fine-tune over
  ~1.5B params, and multi-GPU runs — those stay behind the
  raise-money decision.

## 4. Booking rules (house style)

1. **Local first, always.** A cell goes to RunPod only when the
   local machine cannot hold it (VRAM) or would take months; the
   trigger is recorded in the cell's dispatch note.
2. Spot first; every job restarts from selection-digest-verified
   caches, so preemption costs minutes, not hours.
3. Reproduce the sealed anchor (G1) before any new measurement on
   the instance — a bad node bills nothing meaningful.
4. Prices verified at booking; if spot A100 > $1.50/hr, fall back
   to L40S for extraction (48 GB holds the batches at our sizes).
5. Each sealed cell writes its own evidence hash; the cost ledger
   can sit alongside the evidence trail for later economics (M209).
