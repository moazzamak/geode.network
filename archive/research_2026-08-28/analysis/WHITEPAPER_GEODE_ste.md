# GEODE: a whitepaper in plain language

> **Superseded 21 Aug 2026 by `WHITEPAPER_GEODE.md`** — the merged
> canonical whitepaper: final system only, design rationale, tests and
> metrics, full jargon with each term explained at first use. This
> edition is kept as the record of the plain-language draft.

This is the GEODE whitepaper in Simplified Technical English.
It explains the system from the start.
It uses short sentences.
Every new word is explained where it first appears.
From then on, the word is part of the paper's vocabulary.
You can read this paper without prior machine-learning knowledge.

The technical edition of this paper is
`WHITEPAPER_GEODE_v23.md`.
It holds the full numbers and the sources.
This edition is the explanation.

## 1. What GEODE is

### 1.1 The problem

A service must answer many kinds of tasks: recognize objects in
photos, forecast noisy signals, answer questions.

The usual way is one very large model. The model is trained once, at
great cost. Then it is changed again for each new task. This
re-training is slow and costly. After it, the model is a new object.
No one can say exactly what changed. The knowledge cannot be copied
to another model.

GEODE takes the opposite path.

### 1.2 The idea in one page

GEODE is a catalogue of frozen learning models plus one exact
calculation per task.

Four sentences carry the design:

1. Every model is frozen. A model is a program with many small
   dials. Learning means turning the dials by trying examples.
   Frozen means the dials are fixed. The program can no longer
   change itself. It can only give answers. Writing the book is
   learning. Reading the book is using the model. A printed book
   does not change when you read it.
2. Every task gets one exact fit. The fit is one direct calculation.
   There are no trials, no loops, and no random numbers.
3. Models combine by addition. Two useful pieces give their sum.
   Nothing learns how to combine them.
4. The router selects by measurement, not by opinion. The system
   measures which model fits a task and routes the task there.

### 1.3 The code

An encoder is a frozen learning model that turns an input into a
code. The code is a list of numbers. The numbers stand for the
input: its shapes, its parts, and where the parts are.

The promoted encoder is old and simple. A reader can hold the whole
thing in the head:

1. **Whitening.** Remove the average and the dominant patterns from
   every small piece of the image. This leaves what makes images
   different.
2. **The piece book.** Draw a set of typical pieces from a fixed
   pool. Score every piece of the image against them. The image
   becomes a count of "how much does this piece appear here".
3. **The pyramid.** Divide the image into coarse and fine regions.
   Count the pieces in each region. The code says where things
   appear, not just whether they appear.
4. **Quiet the loud entries.** Take the square root of each entry
   and give each row length one. This evens out very common
   pieces.

The result is one list of numbers per image. The same code serves
every later step.

### 1.4 The exact fit

For each task, the system solves one equation in one step. The
equation finds the weights that best turn codes into answers, with a
small penalty for large weights. There is no training loop, no
learning rate, and no random seed. The same data always gives the
same weights.

### 1.5 The registry

The registry holds the models. Each entry has:

- a unique name,
- its fingerprint,
- its measured accuracy,
- its measured safety coverage,
- its artifact references (files by hash).

A model never changes after registration. An update is a new entry.

### 1.6 The fingerprint

A fingerprint is a short list of numbers that describes the kind of
task a model fits. It is the model's address in task space. Models
for similar tasks get similar fingerprints.

To make a fingerprint, describe the task with attributes from a
fixed list: the input kind, the output kind, the label count, and so
on. Each attribute has a small learned vector. The fingerprint is
the sum of those vectors, with length one. Because it is a sum:

- a combination never seen before still lands at a defined point,
- changing one attribute is vector arithmetic, and
- the same description always gives the same vector.

The fingerprint is never trained on task data. It is trained only
on the attribute description.

### 1.7 The router and the guards

The router scores each model's fingerprint against the task's. The
nearest wins. Ties break by measured accuracy, then by price.

Three guards protect the system:

1. **The input guard** blocks inputs outside the known range.
2. **The tag guard** removes models without measured safety
   coverage for a safety-flagged task. Such a model is
   provisional: declared but not measured.
3. **The freeze** stops all routing and admission in an emergency,
   for a limited time. It expires by itself.

A guard that stops a request gives an empty route. An empty route
is abstention: the system prefers no answer to a wrong or unsafe
answer.

### 1.8 The ledger and the evidence rule

Every decision is recorded in the ledger — the record book of all
decisions. The records are chained by hashes. A hash is a short
code computed from a file. The same file always gives the same
hash. A changed file gives a different hash. No record can change
without breaking the chain.

Every measured number must be reproducible. The number must match
its stored hash. If it does not match, the number is void — it is
not a number in this system.

## 2. What we measured

Every number in this section comes from a recorded measurement. Each
measurement reproduces from its hash. The verdict was written down
before the measurement ran. Accuracy is the share of answers that
are correct: 0.60 means 60 correct answers out of 100. Compute is
the amount of work one answer costs. Matched cost means comparing
two systems at equal work.

All accuracies in this paper are held-out scores. Held-out rows are
data kept aside during the fit, and used only for the test. The
system fits on the training rows and is scored on the held-out
rows. A score on training rows would measure memorization, not
ability. Where a reading was not held-out, the paper says so.

### 2.1 The building ladder

One image code, built step by step, measured on a fixed set of 345
picture classes, at matched cost:

| Build step                      | Accuracy |
| ------------------------------- | -------- |
| One patch size only             | 0.23     |
| Three patch sizes               | 0.24     |
| Quiet the loud entries          | 0.25     |
| Add the pyramid (coarse + fine) | 0.26     |
| Full recipe                     | 0.28     |

Each step pays for itself.

### 2.2 Frozen beats trained

One question was tested five ways. All five answer the same: on
these codes, the exact fit beats the trained head.

| Comparison                                    | Loss in points |
| --------------------------------------------- | -------------- |
| Trained head on the same code                 | −18            |
| Trained dictionary plus trained head          | −12            |
| Trained heads across the whole code family    | up to −20      |
| Every schedule of the trained head            | −9 to −19      |
| A trained extra layer on top of the exact fit | −15            |
| Prune a model, then retrain it                | −5 further     |

Retraining is not merely useless here. It is measurably harmful.

### 2.3 The better code

The limit was the code, not the fit. A probe proved it: no readout
can extract more from the old codes than the exact fit already does.

So we changed the code. The new code pools pieces of a large frozen
model, the same way the old code pools image pieces. No training.
The new code reads **0.59** where the old code read 0.28 — above
every other measured system at that cost.

### 2.4 Data is the lever

The same frozen codes, fit on three times the data, gain 3.7
points. Data, not training, is the measured lever that moves the
system.

### 2.5 The honest negative

On one picture kind — quick, hand-drawn sketches — four different
frozen backbones stop near 0.63. A special model for strokes
improved it to 0.65, still far from the 0.8 bar. We record the
limit. We do not hide it.

### 2.6 The deployed system: one arm per picture kind

The 0.59 above is one frozen model answering all 345 classes at
once, with no training anywhere. That is the construction story.

The deployed system asks an easier question. It splits the task into
one arm per picture kind. Each arm answers about 57 classes, not 345. Each arm works on full-resolution images. Each arm carries a
probe — a small trained reader placed on top of a frozen model.

The measured accuracy, per picture kind:

| Picture kind        | Accuracy |
| ------------------- | -------- |
| Real photos         | 0.91     |
| Drawings            | 0.86     |
| Paintings           | 0.83     |
| Sketches            | 0.81     |
| Charts and diagrams | 0.65     |
| Quick hand sketches | 0.63     |

Two kinds are easy and two kinds stay hard. We record both.

A router picks the right arm. The router is right 0.91 of the time.
When the router picks the arm, the overall accuracy is 0.76.

Why the two stories differ:

- **Class count.** 345 classes at once is harder than about 57.
- **Resolution.** The 0.59 uses 32×32 images. The arms use
  full-resolution images.
- **Training.** The 0.59 has no training anywhere. The arms use a
  small trained probe on top of a frozen model.

Both numbers are true. They answer different questions.

## 3. Why freezing wins

The code carries the information. The trained head destroys what the
exact fit preserves. When a trained head falls to 0.03 on codes the
exact fit reads at 0.28, the limit was never the head. It is what
the frozen model puts into the code. Adding trials does not add
information. On these codes it removes information.

Both probes fired. The first bound the head. The second changed the
codes, and the accuracy followed. Growth changes codes, not heads.

## 4. What we borrowed

GEODE is built from old, named parts. We claim none of them as new.
We measured which combinations win at matched cost.

- The classical image code: whitening, dictionaries, the pyramid,
  and the quieting steps come from the image-retrieval literature
  of the 2000s (Lazebnik and others, 2006; Perronnin and Sánchez,
  2013).
- The deep-patch code follows the deep pyramid line of work from
  2016, on top of the DINOv2 model (Oquab and others, 2023), used
  unchanged.
- The fingerprint idea follows word embeddings (word2vec, GloVe)
  and task embeddings (Task2Vec).
- Transfer scores (LEEP, LogME) are reused as training labels, not
  as decisions.
- The router follows mixture-of-experts routing (Shazeer and
  others, 2017, and the lines after it). We measured that on one
  set of tasks, learned routing loses to one global fit, and built
  the measured failover chain instead.
- The large-solve methods we may need later are named in the
  technical edition.
- Contribution accounting follows data-valuation work (Ghorbani and
  Zou, 2019, and the Shapley line).
- The privacy and proof work follows published zero-knowledge
  methods.
- Concept erasure follows LEACE (Belrose and others, 2023).

The honest summary: the pieces are borrowed. The measured difference
is which combinations win at matched cost, plus the discipline that
makes every decision replayable.

## 5. The honest limits

1. The results are on one benchmark with 32×32 images. Full
   resolution is an open question, registered and not claimed.
2. One corpus, one modality. Text and sound are mostly unbuilt.
3. The recipe has saturated. The last construction variants all
   lost.
4. The exact fit is quadratic in width. Very wide fits need the
   named solver escapes, not yet validated at our scale.
5. Numbers are fragile. A wrong precision once moved weights by
   39% without warning. Discipline is the substitute, not a fix.
6. The toolbox is shown at a small scale. Large registries and
   production traffic are unmeasured.
7. Each new task kind needs a new encoder. It is a pipeline, not a
   lookup.
8. Training of the fingerprint is not reproducible across
   processes on this hardware. Shipped fingerprints store their
   weights.

## 6. Safety and alignment

### 6.1 One rule

Alignment is a measured property, never a declared one. Declared
facts never count. Only measured facts move the system.

### 6.2 The human side

The measured human failure is not malice. It is changing the story
after seeing the result. The system counters it with mechanism:

- Register before measuring. Write the question, the gates, and
  the reading down first.
- A human override is a ledger record with a reason and the answer
  the system would have given. A blank reason is rejected.
- Every decision replays from its hash.

### 6.3 The model side

- A safety-flagged task removes models without measured safety
  coverage. Declared safety never counts.
- A model that drifts — that changes its behavior over time — past
  its bound is gated.
- A recorded harm (a demerit) reduces a model's credit, but only
  with the agreement of several independent checkers (the quorum).
- Credit decays with distance from the last verified measurement.
- Refusal is a measured capability. A model without measured
  refusal cannot serve open tasks.

### 6.4 Lie tolerance

Some participants may lie. The system is lie-tolerant: it works even
when some participants lie, up to a registered limit.

- A fact is used only after several independent checkers agree.
- Measurements combine by the middle value, so a minority of liars
  cannot move the admitted value.
- Checker groups rotate on a fixed schedule. Capture cannot last.
- False statements can cost the checker's stake — money the checker
  put at risk.
- The ledger's hash can be anchored publicly — written into a
  public record — so hidden changes become visible.

The honest limit: if a majority lies, no rule can stop it. Stake
changes that into "the majority has money at risk". That is an
economic assumption, not an institutional one.

### 6.5 What this does not solve

- **Choosing the good.** Humans write the task descriptions and the
  safety probes. The system verifies behavior. It does not define
  what is good.
- **A model smarter than every probe.** A model that can fool all
  probes can pass. Quorum stops collusion between liars, not one
  very smart liar.
- **Explaining why.** The system records what happened. It does not
  explain why a model behaved as it did.
- **Meaning beyond distance.** The input guard measures distance in
  number space, not meaning.

## 7. The economics

The system pays contributors for measured use. A small share of
each paid session goes to a development fund. The rest splits among
contributors by measured contribution, over time.

The rules, measured on synthetic scenarios:

- The shared registry beats the median solo worker.
- Self-dealing loses money: it pays the fees and the delay.
- A copied contribution earns nothing.
- A checker is honest when lying loses money. The required stake is
  a closed-form answer: the stake must exceed the gain from a lie
  divided by the chance of getting caught.
- A captured system is worth only the capped value of one freeze
  window. A registry too small to be worth capturing is itself a
  safety property.

The honest limits: what is measured gets gamed eventually; a rich
enough adversary can buy a majority; and the pricing rules were
measured on synthetic traces, not real demand.

## 8. What is next

- The public anchor decision: publish the ledger hash on a public
  chain. This turns the shipped staking, anchoring, and dispute
  structures live.
- More corpora: the claims are confined to the measured benchmark
  until new measurements land.
- The task-description language: richer rules, with several
  independent authors, under the same evidence discipline.
- Full-resolution measurements, when the compute budget allows.

## 9. Summary

GEODE takes the oldest architecture — frozen features and one exact
fit — and measures it against the modern defaults at matched cost.
On the measured benchmark, the frozen system wins: one frozen model
reads 0.59 on 345 classes with no training, and the deployed arms
read 0.63 to 0.91 per picture kind with a router that is right 0.91
of the time. The registry, the router, the guards, the ledger, and
the lie-tolerant machinery are shipping code. The claims are
confined to what was measured, and the limits are written down.
Every new word in this paper was explained where it first appeared.
