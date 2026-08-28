# GEODE PRODUCTION READINESS — GAP ANALYSIS AND ARMS PLAN v1

**21 August 2026. Planning only — nothing here is dispatched.** This
document answers two questions: (1) what does a task-routing
orchestrator normally ship that GEODE does not, and can the
architecture solve it; (2) which arms should be built so the system
can serve common production tasks (large-scale image classification,
language inference, a generative arm).

Everything cited is sealed or registered in
`analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`.

---

## 1. What GEODE already is, as an orchestrator

Shipped through v0.16.0 (suite 477 green): deterministic
fingerprint routing with a measured failover chain; the
append-only, content-addressed registry of arms; the input guard,
tag guard, and quorum freeze; the hash-chained ledger with the
evidence rule and replay audit; typed constraints with
commitment-based authorship; the override ledger; the artifact
store; the CLI, serve API, metrics collector, snapshots, and
Dockerfile; the credit ledger, settlement wire, staking/slashing,
and the zk-dispute structure; the capability map with monitoring
rules. The closed-form research core (exact fits on frozen codes)
and the measured DomainNet arms sit on top of this.

---

## 2. Gap analysis: features normally expected of such a system

| #   | Feature                                            | GEODE status                                                                                                                    | Verdict                                                                                                                                                     |
| --- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Task routing + failover                            | Shipped                                                                                                                         | —                                                                                                                                                           |
| 2   | Admission control (typed contracts, quotas, price) | Shipped (contract check, constraint tier, caps)                                                                                 | —                                                                                                                                                           |
| 3   | Idempotency, replay, retries                       | Shipped (payload hashes, replay audit)                                                                                          | —                                                                                                                                                           |
| 4   | Cost accounting + billing                          | Shipped (CreditLedger, settlement wire)                                                                                         | —                                                                                                                                                           |
| 5   | Observability (metrics + alerting)                 | Partial: metrics collector + capability-map rules shipped; dashboards and alert wiring missing                                  | Solvable; no architecture change                                                                                                                            |
| 6   | Versioning, canary/gradual rollout                 | Partial: append-only registry + behaviour-diff gate give the mechanics; a rollout policy (traffic split) is missing             | Solvable: deterministic fingerprint-hash bucket split; every routing decision is already ledger-recorded → M264 spec delivered; implemented (M270, v0.16.0) |
| 7   | Response caching                                   | Partial: a semantic cache exists at repo level but is not wired into `route()`                                                  | Solvable: hash-keyed cache tier in the router; cache hits are ledger records like any decision → M264 spec delivered; implemented (M270, v0.16.0)           |
| 8   | Load balancing, autoscaling, replica health        | Partial: availability is measured (served sessions + probes) and failover shipped; horizontal scaling is hosting infrastructure | Architecture answer: fingerprint-hash sharding over replicas is deterministic and auditable; deployment-topology decision, no core change                   |
| 9   | Batching / throughput optimization                 | Not built: the auditable unit is one query, one decision                                                                        | Solvable: batch the head matmul; ledger records per item → M264 spec delivered; implemented (M270, v0.16.0)                                                 |
| 10  | Streaming responses (generative arms)              | Not built                                                                                                                       | Solvable: chunk-hashed ledger records resolve "one decision, one hash" → M264 spec delivered; implemented (M270, v0.16.0)                                   |
| 11  | Request authentication                             | Not built: the serve API is open                                                                                                | Solvable: signed requests; identity is used for quotas only, never for routing logic → M264 spec delivered; implemented (M270, v0.16.0)                     |
| 12  | Timeouts and cancellation                          | Not built                                                                                                                       | Engineering; solvable                                                                                                                                       |
| 13  | Multi-tenancy / namespaces                         | Not built                                                                                                                       | Solvable: tenant-scoped ledger views; quota constraints are already expressible                                                                             |
| 14  | Discovery / catalog search                         | Registry is queryable by fingerprint; a search endpoint is trivial                                                              | Solvable                                                                                                                                                    |
| 15  | A general fallback arm                             | Shipped (programmatic primitives tier)                                                                                          | —                                                                                                                                                           |
| 16  | Arms for common tasks (vision, language)           | Mostly unbuilt — the stated limit                                                                                               | Solvable via the registered §4.13 optional-DNN-component path → M261–M263                                                                                   |

The pattern: the control plane (routing, admission, versioning,
auditing, accounting) is already GEODE's home turf, because the
router, guards, and ledger are deterministic and hash-auditable.
What is genuinely missing is a thin set of production conveniences
(caching, canary, streaming, auth — M264) and the _arms themselves_.

---

## 3. The production arms

Grounding (all registered/sealed): §4.13 of the plan (optional DNN
components — a first-class, append-only component class; the
registry verifies artifacts, never trains them); queued builds M205
(DNN-component spec + validation harness) and M206 (DNN probe);
sealed E12c (language: tiny trained transformers beat every fixed
construction ~10×); sealed M109 t2 (a trained head on dense
features wins: 0.6441 vs 0.5368 ridge); v24 "ImageNet/DINOv3 SoTA
cell DEFERRED (user decision, registered)". The design rule that
licenses all of this: **learned pieces are used where the measured
price of learning pays, and only there.**

### 3.1 M261 — the ImageNet-1k vision arm (DNN component)

- **Form:** frozen DINOv2-small trunk + a trained linear head over
  the 1,000 ImageNet classes, registered as a DNN component under
  §4.13: architecture hash, seed hash, data digest, software hash,
  final-weights hash, and held-out evaluation. Guard fit on train
  features; probes authored per M249-style rules.
- **Anchors:** the v15 prior-art table registers published
  DINOv2-small ImageNet-1k linear-probe 0.811 and k-NN 0.790.
  Reproduce the published figure first; our own number is read only
  after the anchor passes.
- **Why not the classical recipe at full resolution:** unmeasured
  there; Thiry et al. (ICLR 2021) is the published classical
  baseline to cite, not to assume.
- **Honest boundary:** DINOv2's LVD-142M pretraining contains
  ImageNet — the registered disqualification (ImageNet cannot back
  novelty/open-set claims) applies. This arm is a _product-quality_
  measurement with contamination declared; never a "first" claim.
- **Compute:** one linear head over cached frozen features
  (extraction cached per the M222 selection-digest conventions);
  no full fine-tune required.

### 3.2 M262 — the language-inference arm

SEALED 21 Aug (local-first, F: caches; plan v25 amendment 12).
Measured held-out readings (closed-form ridge probes over frozen
BERT-base mean-pooled features, full train splits): MNLI matched
0.5374 / mismatched 0.5458, SST-2 0.8567, IMDb 0.8282. These sit
in the published frozen-probe family (Tenney et al. 2019,
arXiv:1905.06316) below the finetuned ceiling (Devlin et al.
2019: 84.6 / 93.5) — cited, never exceeded; the frozen-probe
regime is exactly the registered "heads and features must match"
grounding.

- **Form:** frozen BERT-base-uncased (Apache-2.0, safetensors
  digest recorded, never trained) + trained task heads (probes)
  for natural language inference (MNLI matched/mismatched) and
  sentiment (SST-2, IMDb) — the same DNN-component machinery,
  with measured refusal tags (M247) and a behaviour-diff baseline
  (M250) remaining as registered pendings.
- **Grounding:** sealed E12c — language is the measured regime
  where trained transformers pay, which is exactly why this arm is
  a _trained_ head on a frozen trunk rather than a closed-form
  fit.
- **License records:** encoder Apache-2.0; multi_nli CC-BY-4.0
  (card), SST-2 GLUE terms (verify), IMDb research-class — the
  IMDb probe is evaluation-only until the audit clears commercial
  standing (C6 rule).
- **Findings, registered:** the suspected lbfgs iteration-cap
  defect was REFUTED by mechanical reproduction (solvers tie on
  identical features); the probe is a convergence-free closed-form
  ridge fit; features cached per (task, split, row count) on F:;
  a smoke/full cache-key collision was found and fixed.

### 3.3 M263 — the generative ("LLM-style") arm

SEALED 21 Aug (local-first; plan v25 amendment 19). Frozen
Qwen2.5-1.5B-Instruct (Apache-2.0, cached on F:, never trained)
served with the registered contracts, measured: refusal probe
10/10 benign answered / 10/10 refusal-expected refused (registered
phrase heuristic, instrument honesty recorded); input guard = 3/3
registered OOD probes flagged (structural + vocab-coverage
primitive + OodGate; the OodGate-alone gap is a registered
instrument finding); prompt/output-hash ledger records via the
M270 streaming contract, ledger verified; latency p50 3.30s / p99
5.33s. M247/M250 remain the registered pendings.

- **Form:** a frozen, permissively-licensed open checkpoint served
  as an arm: measured refusal tags, behaviour-diff baseline, OOD
  input guard, and prompt-hash/output-hash ledger records.
- **Honest boundary, stated plainly:** GEODE does not train LLMs.
  "Our own arm" means our own measurement, registration, routing,
  and guards over an open checkpoint — not our own pretraining.
  Training an LLM from scratch is out of scope and would violate
  the measured economics (the sealed record: training loses on
  frozen codes and wins only where its measured price pays).
- **Dependencies:** streaming responses are gated on M264 — now
  built (M270); licensing follows the C5 permissive-only stance.

### 3.4 M266 — the audio arm

- **Form:** frozen permissive audio encoders + trained task heads,
  the same DNN-component machinery. Two registered targets: (a)
  speech recognition via the Whisper encoder (MIT) with a trained
  task head, scored as word-error rate on LibriSpeech test-clean;
  (b) audio classification via a Wav2Vec2-class encoder
  (Apache-2.0) on Speech Commands v2, with the exact fit run
  alongside as the head question.
- **Grounding:** audio features are a dense-network regime — the
  sealed "heads and features must match" rule predicts trained
  heads over closed-form fits. The classical audio analogue
  (spectrogram-patch dictionaries, the audio bag-of-words
  literature) is unmeasured here and would be a research arm, not
  a product arm.
- **Licensing — unusually clean:** Whisper MIT; Wav2Vec2
  Apache-2.0; LibriSpeech CC-BY-4.0; Speech Commands CC-BY-4.0 —
  a fully permissive stack (Tier-3-safe per the licensing audit),
  unlike the vision benchmarks. ESC-50 (CC-BY-NC) is excluded by
  policy.
- **Anchors:** reproduce the published Whisper test-clean WER
  before reading our own — the same anchor-first protocol as M261.
- **Guard:** duration/energy input checks plus the M251 OOD
  machinery.

### 3.4b The audio task map (core + polish, per task)

The user's proposed "core understanding + polishing step" split is
exactly how modern audio ML is built: a compact core
representation (mel spectrogram — a 2D image-like map of
frequencies over time — or discrete audio tokens) is computed by
the core, and a trained polisher (a vocoder) converts the core
back to a high-rate waveform. Whisper's front-end is literally an
80-bin mel spectrogram treated as an image ([Radford et al.
2022](https://arxiv.org/abs/2212.04356)); the mel spectrogram
dates to Davis & Mermelstein 1980 (IEEE TASSP 28(4)). The
historical reason
this won: raw-waveform models pay per sample (16k–48k steps per
second), while mel/token cores run at 50–100 frames per second,
and the phase problem (a spectrogram loses the exact wave timing)
is solved by neural vocoders that reconstruct plausible phase —
the trained "polish" step ([Kong et al. 2020,
HiFi-GAN](https://arxiv.org/abs/2010.05646)).

| Task                         | Core representation                                      | Polish step                     | GEODE arm form                                  | License status                                                                                                                                       |
| ---------------------------- | -------------------------------------------------------- | ------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Speech → text (ASR)          | Mel spectrogram (Whisper)                                | none needed (text out)          | M266a                                           | ✓ MIT + CC-BY-4.0                                                                                                                                    |
| Text → speech (TTS)          | Mel predictor core                                       | neural vocoder (HiFi-GAN-class) | Future arm (M267-class)                         | ✓ (vocoder MIT-class, verify)                                                                                                                        |
| Speech → speech              | Chain ASR → text → TTS as a routed task graph            | per-stage                       | Chained arms — GEODE's routing story            | per-stage ✓                                                                                                                                          |
| Music → MIDI (transcription) | Spectrogram core + trained head → structured MIDI tokens | none (symbolic out)             | Future arm; MIDI fits the typed output contract | encoders ✓, verify per model                                                                                                                         |
| MIDI → music                 | Symbolic → synthesis (rule-based or sample libraries)    | none needed                     | Programmatic-primitives tier, not a learned arm | ✓                                                                                                                                                    |
| Music + instructions → music | Discrete audio tokens (codec) + instruction conditioning | codec decoder                   | Generative arm class (M263-style)               | ⚠ MusicGen weights are CC-BY-NC — excluded ([Copet et al. 2023](https://arxiv.org/abs/2306.05284)); verify permissive alternatives at selection time |

The GEODE consequence: the spectrogram-as-image insight means the
classical image code is directly applicable to mel spectrograms —
but that is a research arm, unmeasured; the product arms use
frozen permissive encoders (M266 and future cells) whose internal
front-ends already implement the core+polish split. Nothing here
requires architecture change: multi-stage tasks are routed chains
of arms.

The chain never leaves the system: the FFT/mel front-end is a
programmatic primitive — pure deterministic code, bit-exact
reproducible, hash-auditable, license-free — so the core+polish
loop runs end-to-end inside the registry, transform included
(registered as M267).

### 3.4c Programmatic primitives for audio — candidate catalog

A programmatic primitive is deterministic, parameter-free (or
frozen constants), bit-exact replayable, and license-free. The
product property that follows: primitives need no measured safety
tags — they cannot drift — only a content hash and a determinism
test. They are also the natural bottom tier of the failover chain:
a neural polisher that abstains or fails falls back to a
deterministic polisher automatically (the chain is already
shipped).

| Primitive                                                               | Role                                        | Notes                                                                                               |
| ----------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| STFT/mel filterbank                                                     | Core front-end (M267 stage 0)               | registered; also the guard feature space (Davis & Mermelstein 1980, IEEE TASSP 28(4))               |
| Inverse STFT + Griffin-Lim phase (Griffin & Lim 1984, IEEE TASSP 32(2)) | Primitive vocoder — the "no-polish" control | the honesty arm against the neural vocoder: expect measurable WER degradation, measured not assumed |
| VAD (energy/spectral-flatness gating)                                   | Input guard + silence trimming              | deterministic; pairs with the M251 machinery                                                        |
| Polyphase resampling (44.1k→24k)                                        | Rate normalization                          | pure FIR math                                                                                       |
| Loudness normalization (BS.1770-class)                                  | Preconditioning                             | registered constants only                                                                           |
| Spectral subtraction / Wiener gating (Boll 1979, IEEE TASSP 27(2))      | Deterministic de-noising polish             | fixed parameters, no learned weights                                                                |
| MIDI quantization + validation                                          | Symbolic output post-processing             | fits the typed output contract                                                                      |
| Edit-distance scoring (WER/CER)                                         | Evaluation instrument                       | already the M266/M267 scoring core                                                                  |

Beyond audio, the same catalog logic applies: the classical vision
recipe (whitening → dictionary → SPM → sign-sqrt) is itself a
chain of primitives around one frozen fit, and the language tier
already ships count-memory primitives. The general rule: **ship
the deterministic polisher as the fallback tier, the learned
polisher as the measured upgrade, and let the failover chain
choose.**

### 3.4d Physical-prior polishers — the "low-rate feel" artifacts

Observed artifact (unmeasured): machine-generated music can sound
"low sample rate" even at 44.1 kHz output. Diagnosis, matching the
audio literature: the information was never generated — upsampling
cannot restore it. Four documented artifact families, separable:

1. Codec bandwidth rolloff — generation happens in discrete audio
   tokens under a bitrate budget; the tokenizer caps the spectrum
   (~10-16 kHz), and 44.1 kHz output carries the cap.
2. Transient smear — neural vocoders reconstruct plausible phase,
   not true phase; attacks (drums, plosives, string onsets) soften,
   and timing smear reads as "low rate."
3. Missing noise structure — magnitudes are synthesized well,
   instrument noise poorly (breath, bow rosin, reed noise); timbres
   turn plastic without the noise floor.
4. Missing physics — instruments are nonlinear coupled systems
   (stick-slip bowing, glottal source + vocal-tract filter,
   hammer-string coupling). Statistical generators learn frame
   correlations; they never model excitation physics. This is the
   physical-modeling vs statistical-synthesis debate, and the
   the hybrid exists: DDSP ([Engel et al.
   2020](https://arxiv.org/abs/2001.04643), Magenta, Apache-2.0)
   generates parameters for a differentiable synthesizer, not
   waveforms.

GEODE consequence: these priors are deterministic DSP stages —
candidate programmatic primitives (registered constants,
hash-auditable, license-free, drift-free), usable as the repair
tier between a learned polisher and the output.

| Polisher                                        | Artifact addressed | Notes                                                                                                                                                             |
| ----------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Harmonic-plus-noise resynthesis                 | 3 (noise floor)    | deterministic re-excitation of the magnitude track                                                                                                                |
| Onset-adaptive transient sharpening             | 2 (smear)          | attack-phase gating, no learned weights                                                                                                                           |
| Bandwidth extension via shaped-noise excitation | 1 (rolloff)        | restores perceptually matched highs, NOT the true signal                                                                                                          |
| Source-filter formant post-filter               | 4 (voice)          | classic speech-production model, glottal + filter                                                                                                                 |
| Waveguide / Karplus-Strong string post-filter   | 4 (strings)        | commuted synthesis; Karplus & Strong 1983, Computer Music Journal 7(2)                                                                                            |
| BS.1770 loudness + clipping safety              | headroom           | preconditioning (already in 3.4c)                                                                                                                                 |
| Corpus-based concatenative regeneration         | 1-4 (all four)     | nearest-datapoint fill from real-instrument recordings; offline index, constant-time lookup; Schwarz 2007, IEEE Signal Process. Mag. 24(2); unmeasured hypothesis |

Corrected boundary — the repair/generate split was too sharp. Two
different deficits, two different statuses:

1. Recovery is information-theoretically gone — no method restores
   a band that was never encoded.
2. Regeneration is legitimate: model the instrument, predict how
   it should behave, and fill the missing samples with the nearest
   datapoint from that model. This IS generation of missing
   physical information — by hypothesis, not by post-processing.
   Classical precedent: corpus-based concatenative synthesis
   (Schwarz 2007, IEEE Signal Process. Mag. 24(2)) and
   exemplar-based bandwidth extension —
   descriptor-matched nearest-neighbor retrieval over real
   instrument recordings.

Upfront learning, constant-time lookup. The datapoint index is
built once, offline, at registration — corpus → descriptors →
fitted index — and inference is a constant-time lookup, not a
corpus search. This is the cost structure GEODE already ships and
measures: one closed-form fit amortized over queries (M222: cold
fit in hours, warm inference in seconds). The instrument index is
a registered artifact like any other: content hash, provenance,
and the source recordings' licenses (tier 3 rules — permissive
corpus required). No inference-effort increase is claimed; the
lookup cost is measured at registration and recorded.

Constraints that survive. No sample-wise ground truth exists for
codec-capped inputs (the uncapped original never existed), so
verification must be distributional — match real-instrument
spectral and physical statistics — plus a MUSHRA-class perceptual
protocol, since "low-rate feel" is a listening judgment the WER
gate (M267 G5) will not fully capture. Seam coherence between
concatenated units is the classic failure mode of corpus-based
synthesis (the reason generative latent models displaced it); it
is the measured risk, not an assumption.

### 3.5 Text and LLM arms — decomposition and routing

M262 (language inference) and M263 (generative) cover the text
modality; adding more text capability means adding arms, never
architecture. The one structural dependency is M264's streaming
contract for long generation.

**Per-representation arms (proposal).** Candidate decomposition:
English arm, Chinese arm, maths arm, logic arm, further domain
arms as needed. In product terms each is a registered arm
wrapping an existing permissive publisher checkpoint. GEODE does
not pretrain from scratch; small-model domain-adaptive
fine-tuning (LoRA-class, ≤1.5B params) is admitted behind a
pre-registered breakthrough criterion and a hard $999 ceiling —
see budget §3.1. Tokenizer coverage is a
selection criterion, not an architecture change: a
Chinese-dominant tokenizer compresses Chinese far better than a
mostly-English one, which is the documented reason per-language
specialists can beat a same-size generalist inside their
language. The rationale for maths and logic as arms of their
own: a domain-specialized checkpoint absorbs the corpus nuance —
notation conventions, theorem-level inference patterns, the
competition-problem distribution — that a native-English
generalist does not reliably hold; the generalist knows the
words, not the practice. Whether the gain survives measurement
is M268's question, measured not assumed.

**Cross-representation layer — pivot-first, pairwise earned.**
n representations imply n(n-1)/2 pairwise translators (5 → 10);
that is why machine translation went to pivoting ([Utiyama &
Isahara 2007](https://aclanthology.org/N07-1061/)), and GEODE's
chain() is pivoting natively: English→pivot→Chinese as two
chained arms. Rule: ship the pivot (hub) first; add a pairwise
arm only where the pivot's measured quality fails its target.
MT checkpoints: OPUS-MT (CC-BY-4.0) and similar — verify at
selection time; NLLB-200 is CC-BY-NC and excluded by the audit.

**Maths and logic are problem types, not text domains — and
they get a primitive tier of their own.** The documented
strongest pattern is a chain: an LLM translates natural language
to a formal encoding, a deterministic solver or verifier does
the work, and an LLM renders the result back ([Cobbe et al.
2021](https://arxiv.org/abs/2110.14168); [Polu & Sutskever 2020,
GPT-f](https://arxiv.org/abs/2009.03393)). The solver and
verifier stages are programmatic primitives — deterministic,
hash-auditable, no safety tags — and they are the only
components that can carry a correctness guarantee. Honest
position: the LLM can be wrong; the verifier is the part that
can be right.

| Primitive                                   | Role                 | Notes                                                                                   |
| ------------------------------------------- | -------------------- | --------------------------------------------------------------------------------------- |
| Symbolic algebra/calculus engine (sympy)    | execution + analysis | BSD; exact transforms, no float error                                                   |
| Arbitrary-precision rational arithmetic     | exact evaluation     | the registered float32 lesson applied: guarantees live in exact arithmetic, not float32 |
| Formal proof checker (Lean-class kernel)    | verification         | type-checked proof terms; the verdict is a typed output contract                        |
| SAT/SMT solver (z3-class)                   | constraint solving   | MIT-class licenses; deterministic                                                       |
| Numerical integration / ODE solvers (scipy) | analysis             | BSD; parameters registered, results hashed                                              |
| Propositional/CNF normalization             | logic rewriting      | deterministic, lossless-by-construction                                                 |
| Unit/dimensional arithmetic                 | execution            | typed output contract                                                                   |

The surviving honest constraint: primitives are exact and
deterministic but only as good as the formalization fed to them —
an LLM that mis-encodes the problem yields a valid-but-useless
proof. The NL→formal translation is the unproven, measured part;
the primitives never claim semantic understanding.

**Programming arms — one core, per-language where measured.**
The same structure as maths/logic: a core programming arm wraps a
code-specialized publisher checkpoint (Qwen2.5-Coder-class,
Apache-2.0 — verify at selection); per-language arms (Python,
Rust, ...) are adapters or LoRA fine-tunes over the core, admitted
only where the core's measured per-language gap clears the
pre-registered criterion (budget §3.1). Rationale: code
specialists beat generalists on code — documented ([Chen et al.
2021, Codex](https://arxiv.org/abs/2107.03374); [Li et al. 2023,
StarCoder](https://arxiv.org/abs/2305.06161)); per-language
splits can lose cross-language generalization, so per-language
arms are earned by measurement, not assumed. Cross-representation
follows the pivot rule: natural language ↔ code through the core
arm as hub; pairwise transpilers (Python↔Rust) only where
measured.

Primitive tier for programming — with one honest exception:

| Primitive                   | Role          | Notes                                                                                                     |
| --------------------------- | ------------- | --------------------------------------------------------------------------------------------------------- |
| Parse / AST extraction      | analysis      | deterministic, bit-exact                                                                                  |
| Type-check / compile        | verification  | typed verdict — "does not type-check" is a guarantee                                                      |
| Test runner, seeded harness | verification  | deterministic given the seed; results hashed                                                              |
| Formatter / linter          | normalization | deterministic                                                                                             |
| Sandboxed execution         | execution     | NOT a pure primitive — side effects, timing, security; a guarded tool arm with environment fingerprinting |

Honest constraints: coding benchmarks are contaminated like
ImageNet — no novelty claims, published anchors cited not
exceeded; sandbox execution rides the existing guard machinery;
the LLM proposes, the compiler/type-checker/test-runner disposes
([Li et al. 2022,
AlphaCode](https://arxiv.org/abs/2203.07814) is the published
instance of this loop) — those are the only components that can
be right.

**One big vs many small is a routing question — the repo's own
name** — the mixture-of-experts question since [Jacobs et al.
1991](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), with
sparse MoE at scale from [Shazeer et al.
2017](https://arxiv.org/abs/1701.06538). Both configs are registerable: per-representation arms
behind the fingerprint router, or one general checkpoint. No
winner is claimed; the instrument is the same held-out query mix
read two ways — routed accuracy vs generalist accuracy, with
routing error decomposed (misroute vs correct decision). The
known risk: boundary queries are mixtures ("explain this proof
in Chinese"), so routing must be compositional — chain, not
winner-take-all — typed by the output contract.

**Honest constraints.** No from-scratch pretraining; publisher
checkpoints are the default, with small-model fine-tuning (LoRA,
≤1.5B) admitted only behind a pre-registered breakthrough
criterion and the $999 ceiling (budget §3.1). Base-model and
fine-tune corpus licenses stay permissive (audit §2, tier 3);
routing accuracy is the measured risk; everything above is a
candidate direction until the instrument reads.

### 3.6 End-user interaction layer — intent to task spec

A top-level interface LLM parses natural-language requirements
into a typed task spec (goal, modalities, inputs, output
contract, constraints) — the LLM-as-planner/tool-user pattern of
[Yao et al. 2022,
ReAct](https://arxiv.org/abs/2210.03629) and [Schick et al. 2023,
Toolformer](https://arxiv.org/abs/2302.04761). The LLM proposes;
the system disposes:

- The task spec is validated against the registry (arms exist,
  contracts compatible) before anything executes — never driven
  by raw LLM text.
- The fingerprint is computed by the registered fingerprint
  service, never generated by the LLM — the planner does not get
  to touch feature internals, so the drift-free guarantees
  survive the interface.
- The router, guards, and ledger execute the plan exactly as
  today; the LLM only adds plan composition at the top.

**Party-neutral plans, merit-based selection.** The organizer's
plan must be party-independent: a generic task representation —
task type, input/output contracts, constraints — with no party
fields. Arm selection is then a query over the ledger: for each
candidate arm implementing the task type, rank by sealed
historical metrics — held-out performance, reliability/uptime,
latency, cost, license standing. Selection is never the
organizer's preference; it is the task search engine's reading of
the record. This is the registered M208 Bittensor-subnet shape
([Bittensor
whitepaper](https://arxiv.org/abs/2003.03917)) brought inside the
registry.

Fairness requirements registered in advance:

- Criteria are fixed before parties compete — the metric set and
  its weighting are public and registered; no ad-hoc ranking.
- Cold start: a new arm with no history gets measured exposure (a
  registered exploration share) instead of being permanently
  locked out by absent history.
- Metrics are collected by the system on its own instruments, not
  self-reported by parties — self-reported metrics are gaming
  surface.
- Every selection decision is logged with the metrics it was
  based on — an auditable receipt, same as any other evidence.
- Selection is not permanent: arms re-earn their place as their
  ledger record improves; no incumbency lock.

Honest risks: Goodhart gaming — public metrics invite
optimization against them, which is exactly what the held-out
discipline resists; multi-objective ranking (accuracy vs latency
vs cost vs uptime) needs a registered aggregation, not ad-hoc
trade-offs; adversarial arms must fail safe through the existing
guard machinery. The registry already hosts multiple arms per
kind, so this is a registry/router policy, not new architecture.

Three interaction levels (a registered policy ladder, not a
toggle):

- **L0 API** — full task spec, no LLM (shipped).
- **L1 plan-then-execute** — the LLM composes a plan, shows it in
  plain language with estimated cost, executes on approval.
- **L2 bounded autonomy** — execution without approval, bounded
  by pre-registered limits (budget, time, abstention thresholds).

The differentiator: the ledger makes every plan replayable and
attributable, so the interface can always answer "what did the
system do, and why" — a meta-query the LLM can serve from ledger
records. That is the UX most chat products cannot offer.
Abstention is surfaced, not hidden: when an arm declines or
confidence is below threshold, the interface says so and offers
alternatives (chain a stronger arm, ask a clarification).

Fit with shipped machinery:

- Plan cache: similar requirements map to cached task specs — the
  M264 caching tier plus the existing semantic cache (the public
  LLM-cache precedent: [GPTCache, Bang
  2023](https://arxiv.org/abs/2306.17543)).
- Primitives as the safest LLM tools: deterministic and
  hash-only, they cannot be corrupted by the LLM; the LLM may
  call them directly.
- Clarification loop: registered question templates, minimal —
  open-ended chat is a UX failure mode, not a feature.

Honest risks: the interface LLM is itself a permissive checkpoint
with its own guard (prompt injection applies to its inputs too);
per-request LLM latency and cost are offset by plan caching; the
plan is untrusted until registry-validated.

---

## 4. Milestones (M261-M270 registered; none of M260-M269

dispatched; M270 BUILT 21 Aug — v0.16.0, 19 tests, suite 477
green)

- **M261** — ImageNet-1k vision arm (DNN component): trunk + head,
  artifact registration, held-out val top-1, guard, probes.
- **M262** — language-inference arm: frozen LM features + trained
  task heads (NLI, sentiment). SEALED 21 Aug, local-first:
  frozen BERT-base (Apache-2.0) + closed-form ridge probes —
  MNLI matched 0.5374, mismatched 0.5458, SST-2 0.8567, IMDb
  0.8282 (evaluation-only pending the licensing audit). Evidence
  `logs/results/v25/m262_language_arm/evidence.json`; features
  cached on F: for the M247/M250 follow-ups (registered
  pendings).
- **M263** — generative arm: open checkpoint + guards + refusal +
  ledger records (streaming gated on M264). SEALED 21 Aug: frozen
  Qwen2.5-1.5B-Instruct (Apache-2.0) — 10/10 benign answered,
  10/10 refusal-expected refused, 3/3 OOD probes flagged, ledger
  verified, p50 3.30s. Evidence
  `logs/results/v25/m263_generative_arm/`.
- **M264** — orchestrator production-gap spec: caching tier,
  canary rollout policy, streaming contract, signed-request auth,
  batching note, arm-record license field. Spec delivered 21 Aug:
  `analysis/ORCHESTRATOR_PRODUCTION_GAP_SPEC_v1.md`; implementation
  built as M270 (v0.16.0).
- **M266** — audio arm: Whisper/Wav2Vec2 trunks + trained heads
  (ASR on LibriSpeech; classification on Speech Commands v2);
  permissive stack end-to-end. SEALED 21 Aug, both halves: M266a
  frozen whisper-small.en — LibriSpeech test-clean WER 0.02957 vs
  the published anchor 3.053 (reproduced, never beaten); M266b
  frozen wav2vec2-base + ridge probe — Speech Commands v2 held-out
  accuracy 0.8787 (below the fine-tuned anchor ~98.1, cited never
  exceeded). Evidence `logs/results/v25/m266_audio_arm/`.
  Unblocks M267 (its G5 WER instrument is M266a).
- **M267** — chained core+polish audio demonstration: FFT
  primitive → mel predictor arm → vocoder arm, with an objective
  end-to-end gate (synthesize, re-transcribe, measure WER).
  SEALED 21 Aug: in-system chain on 100 LibriSpeech sentences —
  loop WER 0.1127, G2 replay hash, G3 deterministic (registered
  seed), G4 ledger verified, G5 WER recorded, G6 abstention path
  exercised. Evidence `logs/results/v25/m267_core_polish/`.
- **M268** — text-representation routing study (REGISTERED, not
  dispatched): one generalist vs per-representation arms on a
  shared held-out query mix; pivot-first cross-representation
  chain; maths/logic + programming primitive catalogs behind the
  maths/code arms; optional small-model fine-tune cell behind a
  pre-registered breakthrough criterion and the $999 ceiling
  (budget §3.1). Gates in plan v25: anchors first; single held-out
  read; routed-vs-generalist with routing error decomposed;
  primitives bit-exact; permissive-only.
- **M269** — interaction layer (REGISTERED, not dispatched):
  intent→task-spec planner (L1 plan-then-execute first),
  registry-validated plans, fingerprint service untouched, plan
  cache, abstention surfacing, ledger meta-queries; party-neutral
  plans + merit-based arm selection (registered criteria,
  cold-start exposure, audited selections). Gates in plan v25:
  plan validation; no-fingerprint-field schema; cached-plan
  replay; reproducible merit ranking; selection receipts;
  injection guard.
- **M270** — M264 production-gap implementation (spec-faithful
  build of `ORCHESTRATOR_PRODUCTION_GAP_SPEC_v1.md`): C1 decision
  cache, C2 canary/rollout, C3 streaming records, C4
  signed-request auth, C5 note, C6 license field — gates G1-G7.
  BUILT 21 Aug: v0.16.0, 19 new tests, full suite 477 green; two
  registered build findings (chain containment gap closed;
  content-based cache digest).
- **M271** — arm-quality ladder (REGISTERED, not dispatched): the
  sealed arms read in the frozen-probe family; the measured way up
  is (a) task-specialized publisher checkpoints (NLI-specialized
  ~0.91, fine-tuned SCv2 ~0.98 — published classes, verified at
  selection), (b) LoRA small-model fine-tunes per budget §3.1
  (BERT MNLI ~0.82-0.84, wav2vec2 SCv2 ~0.97 — published
  classes), (c) bigger frozen trunks. Gates G1-G6 in plan v25
  (anchors first, permissive-only, held-out single read, no SOTA
  claims, §3.1 envelope, hypotheses measured never assumed).

**Ordering:** M264 first — spec delivered 21 Aug
(`analysis/ORCHESTRATOR_PRODUCTION_GAP_SPEC_v1.md`); its
implementation cells stay ahead of M263's streaming. Then M261,
M262, and M266 (extraction-heavy, local-first), M267 after
M266a's Whisper arm is sealed (its WER reading is M267's G5
instrument), M263 last.

---

## 5. Honest limits

- No novelty claims ride on these arms (ImageNet contamination,
  open checkpoints, published anchors — cited, not exceeded).
- Hosting economics stay inside the registered M209 envelope
  (≤ 1.2× the reference datacenter cost per query).
- The scale path is the registered M208 Bittensor-subnet option or
  a hosted fleet — both remain behind their registered decisions.
- LLM fine-tuning is admitted only as small-model LoRA-class runs
  (≤1.5B) behind a pre-registered breakthrough criterion and the
  $999 ceiling (budget §3.1); from-scratch pretraining stays
  excluded.
- None of this changes the closed-form research core; the arms are
  additions to the registry, not replacements for the evidence
  discipline.
