"""Programmatic memory for the GEODE hybrid system (engineering plan B4a).

A :class:`ProgrammaticMemory` is the "fetch memory from how far back" primitive:
a fully programmatic, count-based, variable-order memory over a discrete token
stream. It has ZERO trained weights and no backprop — the entire model is the
observed (context -> continuation) count table, built by ``register`` and read
by ``continuations`` / ``predict_next``.

Design (registered in ``analysis/ENGINEERING_PLAN_v20.md``, B4a; literature
technique: cache / n-gram / variable-order-Markov (PPM) memory, M129-D7):

- The **window** is the max order: how many past tokens a context may consult.
  At query time it can be dialled *down* (shorter lookback) but never up.
- Lookup is longest-suffix match with backoff: if the full context is unseen,
  the memory returns the counts for the longest observed suffix of it. The
  matched length is reported, so a caller can see "how far back we had to go".
- ``predict_next`` optionally applies add-*alpha* smoothing over a caller
  vocabulary (zero counts for unseen continuations of a known context are a
  real property of a count model; smoothing is the caller's choice).
- Everything is integer counts + tuple keys; ``footprint_bytes`` reports the
  stored-table size so the footprint/energy story stays measurable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Continuation:
    """Observed continuations of a context, with the context length actually used.

    ``matched_length`` is the backoff answer to "how far back did we have to
    go": k if the counts come from the last k tokens of the queried context.
    """

    counts: tuple[tuple[Any, int], ...]
    matched_length: int

    def as_dict(self) -> dict[Any, int]:
        return dict(self.counts)

    def total(self) -> int:
        return sum(count for _, count in self.counts)


class ProgrammaticMemory:
    """Count-based variable-order memory over a discrete token stream.

    Parameters
    ----------
    window:
        Maximum context length stored and consulted (the "how far back" dial).
        Must be >= 1.
    """

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        # context-tuple (length 1..window) -> Counter of next-token counts
        self._counts: dict[tuple[Any, ...], Counter] = {}
        self._tokens_seen = 0

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def register(self, stream: Sequence[Hashable]) -> None:
        """Ingest a token stream, counting every observed context up to window.

        For each position i, every suffix ``stream[i-k:i]`` of length
        1 <= k <= min(window, i) is recorded as a context whose continuation
        is ``stream[i]``. O(n * window) integer increments, no learning.
        """
        for token in stream:
            if not isinstance(token, Hashable):
                raise TypeError("tokens must be hashable")
        tokens = list(stream)
        for i in range(len(tokens)):
            next_token = tokens[i]
            for k in range(1, min(self.window, i) + 1):
                context = tuple(tokens[i - k:i])
                self._counts.setdefault(context, Counter())[next_token] += 1
            self._tokens_seen += 1

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def continuations(
        self, context: Sequence[Hashable], window: int | None = None
    ) -> Continuation:
        """Return the observed continuation counts for *context*.

        Longest-suffix match with backoff: the largest k <= min(window,
        len(context)) for which the last k tokens have stored counts wins. If
        no suffix is seen (a genuinely novel context), an empty continuation
        with matched_length 0 is returned.
        """
        limit = self.window if window is None else min(int(window), self.window)
        if limit < 0:
            raise ValueError("window must be >= 0")
        context = tuple(context)
        for k in range(min(limit, len(context)), 0, -1):
            key = context[-k:]
            if key in self._counts:
                counts = tuple(sorted(self._counts[key].items()))
                return Continuation(counts=counts, matched_length=k)
        return Continuation(counts=(), matched_length=0)

    def exact_continuations(
        self, context: Sequence[Hashable], order: int
    ) -> Continuation:
        """Counts for the EXACT last *order* tokens of *context* (no backoff).

        Unlike :meth:`continuations` (longest-suffix backoff), this returns the
        counts for precisely one context length. It is the primitive the
        interpolated/additive next-token model needs: each order k contributes
        its own term to the mixture. An unseen exact context returns an empty
        continuation with matched_length 0.
        """
        if order < 0:
            raise ValueError("order must be >= 0")
        if order == 0 or order > min(self.window, len(context)):
            return Continuation(counts=(), matched_length=0)
        key = tuple(context)[-order:]
        counter = self._counts.get(key)
        if counter is None:
            return Continuation(counts=(), matched_length=0)
        return Continuation(
            counts=tuple(sorted(counter.items())), matched_length=order
        )

    def predict_next(
        self,
        context: Sequence[Hashable],
        window: int | None = None,
        alpha: float = 0.0,
        vocabulary: Sequence[Hashable] | None = None,
    ) -> Continuation:
        """Continuation counts for *context*, optionally add-*alpha* smoothed.

        With ``alpha > 0`` and a ``vocabulary``, every vocabulary token receives
        a floor of ``alpha``, so a known context assigns positive mass to unseen
        continuations (add-one style smoothing; zero counts for unseen tokens of
        an *unseen* context are left as-is unless the caller supplies the
        vocabulary AND alpha > 0, in which case the empty continuation is also
        smoothed). ``matched_length`` still reports the backoff length used.
        """
        result = self.continuations(context, window=window)
        counts = dict(result.counts)
        if alpha > 0.0:
            if vocabulary is None:
                raise ValueError("alpha > 0 requires a vocabulary")
            for token in vocabulary:
                counts.setdefault(token, 0)
                counts[token] += alpha
            return Continuation(
                counts=tuple(sorted(counts.items())), matched_length=result.matched_length
            )
        return result

    # ------------------------------------------------------------------
    # Footprint
    # ------------------------------------------------------------------

    @property
    def tokens_seen(self) -> int:
        return self._tokens_seen

    @property
    def context_count(self) -> int:
        """Number of distinct (context) keys stored."""
        return len(self._counts)

    @property
    def entry_count(self) -> int:
        """Number of stored (context, next-token) count entries."""
        return sum(len(counter) for counter in self._counts.values())

    def footprint_bytes(self) -> int:
        """Approximate stored-table size (context keys + counts + values).

        Counted as 8 bytes per (context, next) entry plus an estimated key
        overhead of 8 bytes per stored token instance (Python object refs in
        the tuple key are shared with the stream; the estimate is the dict
        slot + counter slot overhead, documented as approximate).
        """
        key_tokens = sum(len(key) for key in self._counts)
        return self.entry_count * 8 + key_tokens * 8

    def matched_length_histogram(
        self, stream: Sequence[Hashable], window: int | None = None
    ) -> dict[int, int]:
        """For each token in *stream*, how far back did lookup have to go?

        A measurement of the memory's effective order: how often the full
        window matched vs how often the model fell back. Pure read-only.
        """
        histogram: Counter[int] = Counter()
        stream = list(stream)
        for i in range(len(stream)):
            if i == 0:
                histogram[0] += 1
                continue
            result = self.continuations(stream[:i], window=window)
            histogram[result.matched_length] += 1
        return dict(sorted(histogram.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "window": self.window,
            "tokens_seen": self._tokens_seen,
            "context_count": self.context_count,
            "entry_count": self.entry_count,
            "footprint_bytes": self.footprint_bytes(),
        }
