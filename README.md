# GEODE — Generalized Encoders for Open-Domain Expertise

**GEODE** is a network where anyone can contribute a capability and
anyone can buy answers, priced in Ethereum at market rate. A
capability is a *frozen* neural network — weights that no longer
change — or a small deterministic program called a *primitive*.
Capabilities are **composable** and **routable**: a contribution can
be reused and built upon by anyone, every use pays the work beneath
it, and payment follows the measured work that actually served the
query. Every decision is deterministic, replayable, and recorded on a
hash-chained ledger.

GEODE claims no new algorithm. Its parts are old and named; its value
is in the assembly and the discipline. The wager: collaboration is
the fastest path to advanced AI. Tasks break into smaller composable
pieces that are cheaper to train and cheaper to run; competitors
build on each other's work and share the rewards by measured use; a
network anyone can improve compounds faster than central labs that
fossilize under secrecy and intellectual-property protection.

**For further information**, read the whitepaper — the design, the
measured results, and the honesty boundaries:
[`analysis/WHITEPAPER_GEODE.pdf`](analysis/WHITEPAPER_GEODE.pdf)
(LaTeX source:
[`analysis/WHITEPAPER_GEODE.tex`](analysis/WHITEPAPER_GEODE.tex)).

## The architecture in one idea

The name says it: **generalized encoders**. A registry of frozen
encoders — each mapping one kind of input (image, audio, text, number
series) into a code space — is the spine. A head declares a *code
manifest*: an ordered list of the blocks it reads, concatenated into
one design. Nothing beneath it is rebuilt: a closed-form fit on the
composed codes is measured, priced, and routed on its own.

- **Frozen.** Every model is a frozen publisher checkpoint. Heads are
  closed-form solves — no epochs, no optimizer, no random seed.
- **Composable.** Codes concatenate additively; a new task enters as
  a measured addition, never a new foundation. Optional closed-form
  alignment bridges (Procrustes/CCA) are registered, measured, and
  priced per pair — never load-bearing.
- **Open contribution surface.** Learned representations — adapters,
  projections, feature maps, bridges — are registrable artifacts with
  an input contract, a measured utility, and a price, exactly like
  any arm. The free standard library holds code-defined transforms
  only; it never holds learned models.
- **Replayable.** The same declared task and input reproduce the same
  routing decisions and the same answer, re-runnable from the hash
  chain by anyone who holds the artifact.
- **Paid by measured use.** Payments flow to the contributors whose
  work actually served the query, vested over epochs, with a small
  development-fund share. Contribution is measured on held-out data,
  never self-reported.

## What has been measured

Held-out measurements of an assembly of published parts, sealed with
content hashes and reproducible from the recorded artifacts.
Single-corpus and single-party — a working demonstration, not a
competitive claim. The whitepaper reports the failures as loudly as
the wins.

| Axis | Measurement | Result |
| ---- | ----------- | ------ |
| Routing | DomainNet: the router picks the right specialist / the answer the user receives | 0.91 / 0.76 |
| Code | Qwen2.5-Coder 7B, HumanEval pass@1 / pass@3 | 0.860 / 0.884 |
| Speech | Whisper ladder, LibriSpeech test-clean WER | 0.0296 |
| Vision (scoped serving) | Open Images: 129 served classes, 472 refused | 0.901 on the served subset |
| Fusion | clean cell: two frozen encoders concatenated vs one | 0.548 vs 0.242 |
| Nonlinearity | quickdraw wall (four frozen backbones) vs + hash-seeded random features | 0.63 → 0.675 |
| Text recipe | frozen BERT-base + closed-form ridge: SST-2 / IMDb | 0.857 / 0.828 |
| Audio recipe | frozen wav2vec2-base + ridge, Speech Commands v2 | 0.879 |
| Number series | in-house temporal arms, Mackey-Glass one-step NRMSFE | 0.0032 |

One honesty statement organizes the whole project: the protocol
machinery is built and measured, and the generalized-encoder thesis
is an open experimental program with measured boundaries — where a
measurement fails, the failure is published.

## Install

```bash
pip install .            # the product package (numpy + torch)
pip install '.[api,dev]' # plus the local HTTP API and the test tooling
```

Then `python examples/hello_geode.py` (the five-minute tour) and the
`geode` CLI (`geode route --fp 0.9,0.3,0.2,0.1`). See
`archive/research_2026-08-28/docs/QUICKSTART.md`. The API is
local-only by design.

- **Whitepaper:** [`analysis/WHITEPAPER_GEODE.pdf`](analysis/WHITEPAPER_GEODE.pdf) — the
  canonical whitepaper: the shipped system, the design decisions and
  why, the held-out measurements and metrics, the safety and
  economics, and the prior art — with every technical term explained
  at its first use. The LaTeX source ships beside it
  ([`analysis/WHITEPAPER_GEODE.tex`](analysis/WHITEPAPER_GEODE.tex)).
  A plain-language markdown copy existed through 24 Aug 2026; it was
  deleted from the public release rather than maintained stale, and
  the .tex/.pdf pair is the single source of truth.
- **Deploying and using:** `archive/research_2026-08-28/docs/DEPLOYMENT.md`
  (the deployment guide in plain language) and
  `archive/research_2026-08-28/docs/QUICKSTART.md` (the five-minute tour).
- **Active plans:** `analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`
  (the sealed execution log; M215/M216 record the product
  architecture); the earlier phases live in
  `archive/research_2026-08-28/analysis/`.
- **Architecture & testing:** `archive/research_2026-08-28/docs/ARCHITECTURE.md`
  (layers and dependency rules) and `archive/research_2026-08-28/docs/TESTING.md`
  (the test pyramid).
- **Lessons & limits:** `archive/research_2026-08-28/analysis/LESSONS_ARCHIVE_v22.md`
  (epistemic discipline).

---

## Repository map

```
analysis/                 # the whitepaper (.tex + built PDF) and the
                          #   sealed v25 execution log
archive/                  # research-phase material superseded by the
  research_2026-08-28/    #   public release (plans, threat reviews,
                          #   sealed logs, tools, docs)
  ...                     #   the earlier legacy archive (archive/README.md)
experiments/              # measurement harnesses and configs
  tier4/                  #   the sealed milestones' runners
  common/                 #   shared experiment utilities
  configs/                #   registered configurations
geode/                    # THE PRODUCT PACKAGE (public API in __init__.py)
  audit/                  #   replay, provenance, erasure
  core/                   #   descriptors, ontology, registry, router,
                          #   ledger, federation bus, economics
  attribution/            #   Shapley/beta-Shapley, incentives, pricing
  settlement/             #   the CreditLedger attribution wire
  privacy/                #   FHE serving, secret sharing, ZK arguments
  hashing.py              #   canonical JSON + payload hash
src/                      # runtime and geometric core modules
infrastructure/evm/       # EVM contracts + the local Hardhat harness
tests/                    # the test pyramid
  unit/                   #   single-concern, in-process
  integration/            #   multi-module flows
  system/                 #   the EVM harness gate
examples/hello_geode.py   # the five-minute tour
requirements.txt
pytest.ini
```

**Rules of the road (from the evidence discipline):**

- Register before measuring. Reproduce the anchor before any new
  number. A smoke run declares itself inadmissible. A void is not a
  negative. Everything frozen carries a payload hash.
- Layering: `geode` never imports `experiments.*`; the dependency
  direction is enforced by `tests/unit/test_architecture_layering.py`.
- `$env:GEODE_CACHE_DIR` points at the data/cache root (external to the
  repo); GPU runs use `.venv-rocm` with `$env:HIP_VISIBLE_DEVICES="1"`.

**Running a cell** (example):

```powershell
$env:GEODE_CACHE_DIR="F:\geode-ml\data\cache"; $env:HIP_VISIBLE_DEVICES="1"
& ".\.venv-rocm\Scripts\python.exe" -m experiments.tier4.eval_v23_m162_prune_retrain `
    --config experiments/configs/v23/m162_prune_retrain_smoke.json `
    --output logs/results/v23/m162_prune_retrain_smoke
```

**Tests:** `python -m pytest` runs the whole pyramid (unit /
integration / system, marked per layer); `-m unit` selects one layer.
The EVM harness runs as `npx hardhat test` in `infrastructure/evm` and
is also covered by the system layer. See
`archive/research_2026-08-28/docs/TESTING.md`.
`.venv-rocm` is the only environment (the CPU-only `.venv` was deleted
on 17 Aug 2026).

---

## Where the research phase went

The repository transitioned from research to development on 17 Aug
2026, and was prepared for public release on 28 Aug 2026. Everything
superseded — historical plans, claim ledgers, research reports,
threat reviews, sealed experiment logs, tools, docs, legacy tiers,
and the legacy README — is preserved under `archive/` (the
post-17-Aug research material in `archive/research_2026-08-28/`, the
earlier material in the rest of `archive/`; see `archive/README.md`).
The git history is squashed to the single public-release commit; the
pre-release history is kept on the remote until the release and
locally on `backup/pre-public-master`.
