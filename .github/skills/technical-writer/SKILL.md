---
name: technical-writer
description: Use this skill when the user asks to write, review, or format technical whitepapers, architectural decision records (ADRs), or engineering documentation.
---

# Technical Writing & Whitepaper Skill

## Objective

Produce whitepapers and technical documents that convey scientific
ideas to **experts and a technically sophisticated general public
simultaneously** — the same text must satisfy a skeptical
researcher and a curious, technically literate outsider. The writing
is authoritative, rigorous, and honest; never marketing.

## Environment & File Context

- **Primary Outputs**: Highly polished, standalone documentation files (`.md` or `.tex`).
- **Permitted Inputs**: Read-only access to existing repository `README.md` files, architectural designs, and design specs to extract technical truth.
- **Strictly Ignored**: Disregard active code changes, developer logs, raw test output files, and commit histories.

## Audience strategy (dual-band writing)

One document serves two readers at once:

- **Experts** get the precise mechanism: exact rules, parameters,
  equations, honest boundaries, prior-art position.
- **The sophisticated general public** gets the intuition without
  dumbing anything down: every load-bearing term is glossed
  inline, in sentence form, the first time it matters (e.g.,
  "a benchmark is a standard, publicly shared test suite",
  "parameters are the adjustable numbers inside a model",
  "a rollup executes transactions cheaply and posts a compressed
  proof of them to Ethereum").

Techniques that make dual-band work:

- **Analogy for the novice, mechanism for the expert, in the same
  breath**: "one exact computation — like fitting a straight line
  through points, in high dimensions."
- **Layered explanations**: when a guarantee has multiple layers
  (provable / probabilistic / economic), say which layer gives
  which guarantee, and where the boundary between them sits.
- **"Stated plainly" passages**: after a table or dense argument,
  restate the two or three readings a reader should take away,
  in simple sentences.

## Structure

Standard whitepaper layout, in this order when applicable:

1. Abstract — enforce a five-beat progression, 200–250 words
   (beats may merge into one sentence, but the logic must appear
   in order):
   a. **Context**: the current state of the field, technology
   stack, or landscape being addressed.
   b. **Core problem**: the friction, bottleneck, or structural
   vulnerability, isolated explicitly.
   c. **Proposed solution**, introduced by name.
   d. **Technical execution**: how it works, naming the critical
   technologies, algorithms, or architectural patterns.
   e. **Close**: for product/engineering papers, the quantifiable
   impact (latency, availability, cost); for research papers,
   the thesis or position the paper stakes, stated plainly
   ("claims no new algorithm; the value is in the assembly and
   the discipline") — see the honesty principle below.
   No jargon the paper has not defined; end on what the paper
   does and does not claim.
2. Introduction (the problem, the mechanism, the wager argued).
3. Actors and roles (every rule in the paper must attach to a role).
4. Design principles (a small set; each testable).
5. Architecture (black-box view first, then the pipeline; a figure).
6. Protocol at implementation depth (an independent implementer can
   rebuild it; every adjustable parameter states its adjustment
   path).
7. The economic/incentive design (how money moves and why each rule
   exists — state the attack each rule answers).
8. What has been measured (held-out results, plain readings).
9. Prior art and position (claim the assembly, never the parts).
10. Known limits (honesty is part of the design — a dedicated
    section, enumerated).
11. Methodology (how it gets built and verified — the discipline is
    the product).
12. Conclusion (the wager restated; invite verification).

For ADRs, adapt the 12-section layout into this exact compact progression:

1. **Context & Status**: The precise engineering constraints driving the change, including current architecture limitations.
2. **Decision & Mechanism**: The exact technical choice made, including rules, parameters, and structural shifts.
3. **Consequences & Hedges**: What is unlocked vs. what is sacrificed (e.g., increased memory footprint for faster lookup).
4. **Alternatives Considered**: Prior art within the org or industry, stating exactly why they were rejected.

## Drafting rules

- **Objective, authoritative, concise.** No marketing fluff; focus
  on architectural realities, metrics, and data.
- **Ban AI Fluff Words**: Never use structural transitions or padding words such as _"In conclusion,"_, _"Furthermore,"_, _"It is important to note,"_, _"Crucial role,"_, or _"Tapestry."_ Move straight from evidence to conclusion.
- **Code vs. Pseudocode**: Do not inject raw repository source code into the paper. Translate active code into clean, human-readable pseudocode or mathematical notation to explain algorithmic mechanics.
- **Precise terminology, glossed.** Technical terms are used
  exactly; each is introduced for a general reader exactly once.
- **Short sentences; one idea each.** Break run-ons. No em-dashes:
  use a period, "and", or a restructure (see Voice rules).
- **Parallelism and repetition.** Kill echo words ("built upon …
  builds on"), restore parallel prepositions, remove filler
  ("core principles" → "principles").
- **Every claim attaches to a number or a mechanism.** "Validated
  by simulation, never by assumption." State which class of
  evidence supports which claim.

## Voice rules (natural-language fluency)

Rules that keep prose reading as natural human English rather than
compressed note-taking. These are the generalizable rules behind the
style passes (dropped function words, gloss-first, dash cleanup).

- **Write like a person speaks.** The most understandable text reads
  the way a fluent person talks: a natural flow, plain word order, and
  spoken rhythm, minus the fillers ("um", "uh", "ahem") and
  hesitations. Read each sentence aloud; if it sounds stilted or
  staccato, rewrite it. This is the governing principle behind every
  rule below, and it is what keeps the paper legible to the
  sophisticated general reader without dumbing anything down.
- **No dropped function words.** A clause must not omit the copula
  (is/are), an article (a/the), or a locative subject where a natural
  reading needs one. The signature violation is a verbless fragment
  that stands as a sentence, especially the double appositive with a
  relative clause but no main verb: "X, the Y that Z."
  - Broken: "The role that records credits, the one act that still
    needs a named party." Fixed: "This is the role that records
    credits, the one act that still needs a named party."
  - Broken: "is not privilege". Fixed: "is not a privilege" (missing
    article before a singular count noun).
  - Broken: "Inside, one input walks one path." Fixed: "Inside the
    network, one input walks one path." (dropped locative subject).
  - Broken: "The council's weight is not stake." Fixed: "is not
    staked." (bare noun where the natural verb form reads better).
  - **Deliberate exceptions, do not "fix" these**: definition-list
    appositions ("User. Anyone who pays for answers.") are standard
    dictionary style; colon-list fragments ("Examples: memory stores,
    math engines, transforms.") are natural; verb-phrase ellipsis
    ("A sampled fraction does carry a proof.") is standard English.
    Only fix constructions that read as broken sentences.
- **Gloss-first.** Lead with what a thing does in plain words, then
  attach the label, never the reverse ("the account that receives
  the 2.5% share and the registration fees" before naming the
  development fund).
- **No em-dashes.** Replace em-dashes (---) with a period, "and",
  or a restructure. En-dashes are reserved for numeric ranges and
  compound modifiers (80--90%). Scan for the em-dash character
  before committing.
- **Frame every list.** A list needs a framing sentence that says
  what the list is and why it matters ("The pricing rules are:",
  "The defenses are:", "The rules that keep the number honest are:").
  No bare item dump.
- **What before how.** State what a mechanism does before how it
  does it. A reader can reject a mechanism at the "what" without
  paying for the "how".

## Honesty discipline (non-negotiable)

- **The abstract states; the body hedges.** A whitepaper describes
  an as-yet-untested project by convention — the Bitcoin
  precedent: Satoshi stated the system in the abstract and put
  the honest-majority assumption and the attacker's calculated
  odds in the body. Hedging ("this is untested", "this is a
  conjecture") belongs in Known limits, never in the opening
  pitch.
- **The whitepaper is not a ledger.** Internal bookkeeping
  vocabulary — milestone IDs, test counts, "evidence hash",
  "registered simulation result", changelog dates — never enters
  the paper. Findings are stated as plain prose; the project's
  plan and design docs are the ledger. When editing results into
  the paper, translate: _what was found_, not _what was run_.
- **State limits as precisely as capabilities.** "Not yet provable
  — not for lack of mathematics but for lack of speed" is correct;
  "proving is future work" wrongly implies it is the plan. Say
  which option is the DESIGN (probing, here) and which is a
  conditional trigger, never a placeholder.
- **No unearned novelty.** Prior art is cited by name; the claimed
  contribution is the assembly and the discipline, nothing else.
- **Distinguish judgment from computation.** A quorum decision is
  a judgment with a public record; a replay-gated conviction is a
  computation. Never blur the two — the reader must be able to
  tell which kind of decision each mechanism makes.
- **Publish the failures with the successes.** A negative result
  gets the same ink as a win.

## Review checklist (Execute as an explicit multi-pass audit)

1. **Pass 1: The Two-Reader Audit**. Scan every paragraph. Is every load-bearing term glossed inline for the generalist? Is the exact mechanism intact for the expert?
2. **Pass 2: The Honesty Audit**. Strip out ledger-speak (commit IDs, test counts). Verify that boundaries and limitations are stated as precisely as capabilities.
3. **Pass 3: The Sentence Audit**. Break down run-ons. Delete echo words and filler words. Verify parallel structure.
4. **Pass 4: The Structure Audit**. Every rule attaches to a role;
   the flow goes black-box → detail → measurement → limits →
   methodology; the abstract follows the five beats in order.
5. **Pass 5: The Formatting & Compiling Audit**. Ensure clean LaTeX compilation. Check that figures use exact `text width` to prevent overlap. Verify that hyphenation (`\-`) is applied to long technical compound words overrunning line breaks.
6. **Pass 6: The Fluency Audit**. Read the text aloud; anything that
   sounds stilted or staccato should be rewritten (write like a
   person speaks). Scan for dropped function words: verbless
   fragments standing as sentences, missing articles before singular
   count nouns, and dropped locative subjects (see Voice rules).
   Verify gloss-first ordering at first use of every load-bearing
   term, no em-dashes, and that every list is framed by a sentence.
   Do not "fix" deliberate dictionary-style appositions or colon-list
   fragments.
