# GEODE Standard Library of Primitives (SLP) — developer guide

This folder is the developer-facing surface of the standard
library. Browse `CATALOG.md` to find primitives. Read this file to
add one.

## What a primitive is

A primitive is a deterministic, code-defined transform. It has no
learned parameters. It implements the `Primitive` interface and
carries a `PrimitiveSpec`: name, category, input and output
dimensions, dtypes, named parameters, and a one-line description.

## The determinism contract

A primitive admits only if all of these hold:

1. **Pure.** Same input, same output. No side effects.
2. **Deterministic.** Bit-exact across runs and across machines
   with the same pinned dependency versions.
3. **Bounded.** CPU, memory, and input size are capped. No
   unbounded search.
4. **Pinned.** Every dependency is pinned to an exact version.
5. **No randomness, no network, no wall clock.** A seeded stream
   is allowed only when the seed is an explicit argument.

Tier A items admit after the standard determinism test. Tier B
items (codecs, parsers, computer algebra) admit only with a
reference-vs-reference determinism certificate measured on
registered hardware.

## How to find a primitive

`CATALOG.md` is the index. Every entry has an ID
(`domain.family`), the access path, a status, a tier, and a
one-line summary. Import built primitives from `src.slp`, never
from the underlying modules.

## How to add a primitive

Follow the checklist, in order.

1. **Register the entry first.** Add the row to `_ENTRIES` in
   `src/slp/_catalog.py` with status `PENDING`.
2. **Write the factory.** Name it `make_<name>`. Give it a
   `PrimitiveSpec` and a docstring in STE style: purpose, input,
   output, determinism notes, example.
3. **Add the determinism test.** Same input twice gives
   bit-identical output. The payload hash replays.
4. **Flip the entry to `BUILT`** and set the access path.
5. **Regenerate the catalog.**
   `python -c "from src.slp._catalog import render_markdown; print(render_markdown())" > src/slp/CATALOG.md`

## Conventions

- One file per domain. `math.py`, `text.py`, `image.py`, and so
  on, all under `src/slp/`.
- Factory names are `make_<verb>`. Parameters are registration
  constants, not serve-time knobs.
- Descriptions are STE: short sentences, one idea each, no nested
  asides.

## Status vocabulary

- **BUILT** — importable today, tested, determinism verified.
- **PENDING** — launch backlog. The row exists; the code does not
  yet.
- **Tier A** — launch-ready under the entry rule.
- **Tier B** — needs a pinned dependency and a determinism
  certificate before admission.

The full possibility space — 36 domains, 18 operation families,
about 720 named primitives, with the tool-call equivalence map —
lives in `analysis/SLP_POSSIBILITY_SPACE_v1.md`. The launch ranking
lives in `analysis/SLP_CANDIDATE_CATALOG_v1.md`.
