"""Atom naming for M82, under the restrictions Amendment R9 registered.

R9 withdrew the plan's claim of "two independent naming channels". The v13
dictionary is fit over DINOv2 features, so an atom is a direction in DINOv2
space and cannot be compared to a text embedding at all — CLIP text and DINOv2
share neither a basis nor a dimensionality. The only route from an atom to a
phrase runs through that atom's top-activating images, and that is also the
exemplar channel's route. There is no second route, so agreement between the
two cannot evidence correctness.

What this module therefore measures is what the setup can actually test:

* **exemplar-resampling stability** — does a name survive being computed from a
  disjoint half of the atom's exemplars? Promoted by R9 from a side-check to
  the primary naming operand.
* **the class-purity positive control** — can the channel name an atom whose
  exemplars are almost all one class? A channel that fails this is not naming
  anything, and its output on mixed atoms is uninterpretable. Gating.
* **the far-field rate** — how often does it choose one of the 217 DomainNet
  names the corpus never depicts? N82.2 measured a 34.57% floor for this on
  single images, so an atom's name is read against that floor, never zero.
* **the shuffled-exemplar null** — R6's floor, retained.

Naming is deliberately kept to one deterministic function so that "the name of
an atom" is well defined wherever it appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPSILON = 1e-12


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, _EPSILON)


@dataclass(frozen=True)
class AtomExemplars:
    """Top-activating corpus rows per atom.

    ``rows`` is ragged in principle — a dead atom has none — so it is stored as
    a list of arrays rather than a matrix, and every consumer must handle the
    empty case rather than assuming a full set.
    """

    rows: list[np.ndarray]
    activations: list[np.ndarray]

    @property
    def atom_count(self) -> int:
        return len(self.rows)

    def live(self) -> np.ndarray:
        return np.array([len(row) > 0 for row in self.rows], dtype=bool)


def atom_exemplars(codes: np.ndarray, *, top_count: int) -> AtomExemplars:
    """Take each atom's ``top_count`` most strongly activating rows.

    Ties and dead atoms are both real: an atom that never activates yields an
    empty exemplar set rather than an arbitrary one.
    """
    if codes.ndim != 2:
        raise ValueError("codes must be (rows, atoms)")
    rows: list[np.ndarray] = []
    activations: list[np.ndarray] = []
    for atom in range(codes.shape[1]):
        column = codes[:, atom]
        firing = np.flatnonzero(column > 0.0)
        if firing.size == 0:
            rows.append(np.empty(0, dtype=np.int64))
            activations.append(np.empty(0, dtype=np.float32))
            continue
        order = firing[np.argsort(-column[firing], kind="stable")][:top_count]
        rows.append(order.astype(np.int64))
        activations.append(column[order].astype(np.float32))
    return AtomExemplars(rows=rows, activations=activations)


def split_exemplars(
    exemplars: AtomExemplars, *, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split each atom's exemplars into two disjoint halves.

    R9's primary operand. The halves are disjoint by construction, so a name
    that agrees across them is not agreeing because it saw the same picture
    twice.
    """
    generator = np.random.default_rng(seed)
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for rows in exemplars.rows:
        if len(rows) < 2:
            first.append(np.empty(0, dtype=np.int64))
            second.append(np.empty(0, dtype=np.int64))
            continue
        shuffled = generator.permutation(rows)
        midpoint = len(shuffled) // 2
        first.append(np.sort(shuffled[:midpoint]))
        second.append(np.sort(shuffled[midpoint : 2 * midpoint]))
    return first, second


def shuffled_exemplars(exemplars: AtomExemplars, *, seed: int) -> list[np.ndarray]:
    """R6's null: reassign exemplar sets across atoms at matched sizes.

    Set sizes are preserved, so any agreement this produces comes from the
    vocabulary's own geometry and the corpus's marginal image distribution
    rather than from the atoms.
    """
    generator = np.random.default_rng(seed)
    pool = np.concatenate([rows for rows in exemplars.rows if len(rows)])
    shuffled: list[np.ndarray] = []
    for rows in exemplars.rows:
        if len(rows) == 0:
            shuffled.append(np.empty(0, dtype=np.int64))
            continue
        shuffled.append(generator.choice(pool, size=len(rows), replace=False))
    return shuffled


def name_atoms(
    exemplar_rows: list[np.ndarray],
    image_embeddings: np.ndarray,
    term_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Name each atom by the term nearest its mean exemplar embedding.

    Returns ``(term_index, score)``. An atom with no exemplars is named ``-1``
    with score ``nan`` rather than being given a default term, so that
    "unnameable" stays distinguishable from "named badly" downstream.
    """
    names = np.full(len(exemplar_rows), -1, dtype=np.int64)
    scores = np.full(len(exemplar_rows), np.nan, dtype=np.float32)
    for atom, rows in enumerate(exemplar_rows):
        if len(rows) == 0:
            continue
        centroid = _normalise(image_embeddings[rows].mean(axis=0))
        similarity = term_embeddings @ centroid
        best = int(np.argmax(similarity))
        names[atom] = best
        scores[atom] = float(similarity[best])
    return names, scores


def naming_agreement(
    first: np.ndarray, second: np.ndarray
) -> dict[str, float | int | None]:
    """Agreement between two namings, over atoms both could name.

    An empty comparison reports ``None`` rather than ``nan``: the evidence hash
    refuses non-finite floats, and "there was nothing to compare" is a
    different statement from "they agreed on nothing."
    """
    comparable = (first >= 0) & (second >= 0)
    count = int(comparable.sum())
    if count == 0:
        return {"comparable_atoms": 0, "agreement": None}
    return {
        "comparable_atoms": count,
        "agreement": float(np.mean(first[comparable] == second[comparable])),
    }


def far_field_rate(
    names: np.ndarray, *, in_corpus_terms: int, style_terms_start: int | None = None
) -> dict[str, float | int | None]:
    """How often a name is one the corpus never depicts.

    Read against N82.2's measured single-image floor of 34.57%, not against
    zero.

    ``style_terms_start`` separates two outcomes that N82.3 anticipated but
    that a single rate conflates. A style term sits above the in-corpus object
    terms by index, so it would otherwise be counted as a far-field object
    name; but an atom named "a rough black and white doodle" has not been
    misnamed after an absent object, it has been named after a rendering style
    the corpus does contain. Both rates are returned, and the raw one is kept
    so the earlier figure remains reconstructible.
    """
    named = names[names >= 0]
    if named.size == 0:
        return {
            "named_atoms": 0,
            "false_naming_rate": None,
            "object_named_atoms": 0,
            "style_named_atoms": 0,
            "object_false_naming_rate": None,
        }
    style = (
        np.zeros(named.shape, dtype=bool)
        if style_terms_start is None
        else named >= style_terms_start
    )
    objects = named[~style]
    return {
        "named_atoms": int(named.size),
        "false_naming_rate": float(np.mean(named >= in_corpus_terms)),
        "object_named_atoms": int(objects.size),
        "style_named_atoms": int(style.sum()),
        "object_false_naming_rate": (
            float(np.mean(objects >= in_corpus_terms)) if objects.size else None
        ),
    }


def atom_class_purity(
    exemplar_rows: list[np.ndarray], labels: np.ndarray, *, class_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Dominant class of each atom's exemplars, and the share it holds."""
    dominant = np.full(len(exemplar_rows), -1, dtype=np.int64)
    purity = np.zeros(len(exemplar_rows), dtype=np.float32)
    for atom, rows in enumerate(exemplar_rows):
        if len(rows) == 0:
            continue
        counts = np.bincount(labels[rows], minlength=class_count)
        best = int(np.argmax(counts))
        dominant[atom] = best
        purity[atom] = float(counts[best] / len(rows))
    return dominant, purity


def purity_positive_control(
    names: np.ndarray,
    exemplar_rows: list[np.ndarray],
    labels: np.ndarray,
    *,
    class_count: int,
    purity_threshold: float,
) -> dict[str, float | int | None]:
    """R9's gating control: can the channel name an unambiguous atom?

    Restricted to atoms whose exemplars are at least ``purity_threshold`` one
    class, where the correct name is known independently of the channel. This
    is the naming instrument's positive end, in the role N1's far-field noise
    and held-out knowns played for the detection instrument.
    """
    dominant, purity = atom_class_purity(
        exemplar_rows, labels, class_count=class_count
    )
    eligible = (purity >= purity_threshold) & (names >= 0) & (dominant >= 0)
    count = int(eligible.sum())
    if count == 0:
        return {
            "eligible_atoms": 0,
            "purity_threshold": float(purity_threshold),
            "accuracy": None,
            "mean_purity": None,
        }
    return {
        "eligible_atoms": count,
        "purity_threshold": float(purity_threshold),
        "accuracy": float(np.mean(names[eligible] == dominant[eligible])),
        "mean_purity": float(np.mean(purity[eligible])),
    }


def domain_breakdown(
    names: np.ndarray,
    exemplar_rows: list[np.ndarray],
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    class_count: int,
    domain_count: int,
    purity_threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    """Per-domain naming accuracy, required by N82.1.

    An atom is attributed to the domain its exemplars mostly come from. N82.1
    measured CLIP reading quickdraw at 28.99% against clipart's 91.23% while
    quickdraw is 61% of the corpus, so an aggregate naming number averages a
    working instrument with a broken one and may not be quoted alone.
    """
    dominant, purity = atom_class_purity(
        exemplar_rows, labels, class_count=class_count
    )
    atom_domain = np.full(len(exemplar_rows), -1, dtype=np.int64)
    for atom, rows in enumerate(exemplar_rows):
        if len(rows) == 0:
            continue
        atom_domain[atom] = int(np.argmax(np.bincount(domains[rows], minlength=domain_count)))

    breakdown: dict[str, dict[str, float | int | None]] = {}
    for domain in range(domain_count):
        eligible = (
            (atom_domain == domain)
            & (purity >= purity_threshold)
            & (names >= 0)
            & (dominant >= 0)
        )
        count = int(eligible.sum())
        breakdown[str(domain)] = {
            "eligible_atoms": count,
            "accuracy": (
                float(np.mean(names[eligible] == dominant[eligible]))
                if count
                else None
            ),
            "atoms_in_domain": int((atom_domain == domain).sum()),
        }
    return breakdown


def sparse_atom_exemplars(
    indices: np.ndarray, values: np.ndarray, *, dictionary_size: int, top_count: int
) -> AtomExemplars:
    """``atom_exemplars`` over sparse codes, without densifying them.

    The fit split is 36,864 rows against an 8,192-atom dictionary; the dense
    matrix would be 1.2 GB of which 99.6 percent is zero. The result is
    identical to ``atom_exemplars`` on the dense form — the tests pin that.
    """
    buckets_rows: list[list[int]] = [[] for _ in range(dictionary_size)]
    buckets_values: list[list[float]] = [[] for _ in range(dictionary_size)]
    for row in range(indices.shape[0]):
        for slot in range(indices.shape[1]):
            value = float(values[row, slot])
            if value > 0.0:
                atom = int(indices[row, slot])
                buckets_rows[atom].append(row)
                buckets_values[atom].append(value)

    rows: list[np.ndarray] = []
    activations: list[np.ndarray] = []
    for atom in range(dictionary_size):
        if not buckets_rows[atom]:
            rows.append(np.empty(0, dtype=np.int64))
            activations.append(np.empty(0, dtype=np.float32))
            continue
        atom_rows = np.asarray(buckets_rows[atom], dtype=np.int64)
        atom_values = np.asarray(buckets_values[atom], dtype=np.float32)
        order = np.argsort(-atom_values, kind="stable")[:top_count]
        rows.append(atom_rows[order])
        activations.append(atom_values[order])
    return AtomExemplars(rows=rows, activations=activations)


def grouped_explanation(
    indices: np.ndarray,
    contributions: np.ndarray,
    group_of_atom: np.ndarray,
    *,
    group_count: int,
) -> np.ndarray:
    """Pool per-atom contributions into group columns, then append the same
    three summary statistics ``withheld_explanation`` uses.

    This is the identity-revealed explanation form. An atom whose group is -1
    contributes to no column, which is the honest encoding of an atom the
    channel could not name: the reader is told nothing about it, so the probe
    is told nothing about it either.
    """
    rows = indices.shape[0]
    pooled = np.zeros((rows, group_count), dtype=np.float32)
    groups = group_of_atom[indices]
    for slot in range(indices.shape[1]):
        column = groups[:, slot]
        assigned = column >= 0
        if not assigned.any():
            continue
        np.add.at(
            pooled,
            (np.flatnonzero(assigned), column[assigned]),
            contributions[assigned, slot],
        )
    summary = np.column_stack(
        [
            np.sum(contributions, axis=1),
            np.max(contributions, axis=1),
            np.mean(contributions, axis=1),
        ]
    )
    return np.column_stack([pooled, summary]).astype(np.float32)


def matched_random_grouping(
    group_of_atom: np.ndarray, *, seed: int
) -> np.ndarray:
    """N82.4's null: permute which atom sits in which group.

    Group count, group sizes and the number of ungrouped atoms are all
    preserved exactly, because the assignment vector itself is permuted rather
    than resampled. The named and matched-random arms therefore differ in
    nothing except whether the grouping means anything.
    """
    generator = np.random.default_rng(seed)
    return group_of_atom[generator.permutation(len(group_of_atom))]


def names_to_groups(names: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Compact the term indices atoms were named with into dense group columns.

    Many of the 351 vocabulary terms name no atom, and a column that is always
    zero would inflate the explanation width without carrying anything. The
    returned term list records which name each surviving column stands for, so
    the explanation remains readable as words rather than as column numbers.
    """
    used = sorted({int(name) for name in names if name >= 0})
    lookup = {term: position for position, term in enumerate(used)}
    groups = np.full(len(names), -1, dtype=np.int64)
    for atom, name in enumerate(names):
        if name >= 0:
            groups[atom] = lookup[int(name)]
    return groups, used
