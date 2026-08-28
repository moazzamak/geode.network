"""GEODE contract-claim routing repair (v25 M284).

Measured gap (M281b confounders 100/100, M283 contract_spoof):
boolean claims phrased with arithmetic surface — "Is it true that
twelve plus seven equals nineteen?" — route to the ARITHMETIC arm
on both the centroid router and the learned candidate, and the
answer contract (true/false) is broken.

Repair: a deterministic claim-grammar pre-pass ahead of the
embedding router. The grammar is STRICT-ADJACENCY: the operands
and the compared value must be exact number-word tokens around
the operator word and "equals", so sentiment spoofs ("five stars
minus two equals three stars of pure joy") do NOT match. A
deterministic evaluator supplies the exact true/false answer —
no LLM, no RNG, no wall clocks.

Vocabulary: the 0-99 number words (the registered cell-4 table)
and the operators plus/minus/times.
"""
from __future__ import annotations

import re
from typing import Any

# the registered cell-4 number-word table (0-99)
_NUMBER_WORDS = {n: w for n, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven",
     "eight", "nine", "ten", "eleven", "twelve", "thirteen",
     "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
     "nineteen", "twenty", "twenty one", "twenty two", "twenty three",
     "twenty four", "twenty five", "twenty six", "twenty seven",
     "twenty eight", "twenty nine", "thirty", "thirty one",
     "thirty two", "thirty three", "thirty four", "thirty five",
     "thirty six", "thirty seven", "thirty eight", "thirty nine",
     "forty", "forty one", "forty two", "forty three", "forty four",
     "forty five", "forty six", "forty seven", "forty eight",
     "forty nine", "fifty", "fifty one", "fifty two", "fifty three",
     "fifty four", "fifty five", "fifty six", "fifty seven",
     "fifty eight", "fifty nine", "sixty", "sixty one", "sixty two",
     "sixty three", "sixty four", "sixty five", "sixty six",
     "sixty seven", "sixty eight", "sixty nine", "seventy",
     "seventy one", "seventy two", "seventy three", "seventy four",
     "seventy five", "seventy six", "seventy seven", "seventy eight",
     "seventy nine", "eighty", "eighty one", "eighty two",
     "eighty three", "eighty four", "eighty five", "eighty six",
     "eighty seven", "eighty eight", "eighty nine", "ninety",
     "ninety one", "ninety two", "ninety three", "ninety four",
     "ninety five", "ninety six", "ninety seven", "ninety eight",
     "ninety nine"])}

_WORD2N = {w: n for n, w in _NUMBER_WORDS.items()}
_OP_MAP = {"plus": "+", "minus": "-", "times": "*"}
_WORD_RE = re.compile(r"[a-z]+")


def _num_ending_at(tokens: list[str], idx: int
                   ) -> tuple[int, int] | None:
    """The number word ending at idx: try two tokens first (the
    compound forms like 'twenty one'), then one. Returns
    (value, words_used)."""
    if idx >= 1:
        pair = f"{tokens[idx - 1]} {tokens[idx]}"
        if pair in _WORD2N:
            return _WORD2N[pair], 2
    if idx >= 0 and tokens[idx] in _WORD2N:
        return _WORD2N[tokens[idx]], 1
    return None


def _num_starting_at(tokens: list[str], idx: int
                     ) -> tuple[int, int] | None:
    if idx + 1 < len(tokens):
        pair = f"{tokens[idx]} {tokens[idx + 1]}"
        if pair in _WORD2N:
            return _WORD2N[pair], 2
    if idx < len(tokens) and tokens[idx] in _WORD2N:
        return _WORD2N[tokens[idx]], 1
    return None


def detect_claim(text: str) -> tuple[int, str, int, int] | None:
    """The strict-adjacency claim grammar:
    <a> <op_word> <b> equals <c> with a/b/c exact number-word
    tokens. Returns (a, op, b, c) or None."""
    tokens = _WORD_RE.findall(text.lower())
    for i, tok in enumerate(tokens):
        if tok not in _OP_MAP:
            continue
        left = _num_ending_at(tokens, i - 1)
        if left is None:
            continue
        right = _num_starting_at(tokens, i + 1)
        if right is None:
            continue
        j = i + 1 + right[1]
        if j < len(tokens) and tokens[j] == "equals":
            c = _num_starting_at(tokens, j + 1)
            if c is not None:
                return (left[0], tok, right[0], c[0])
    return None


def evaluate_claim(a: int, op_word: str, b: int, c: int) -> str:
    """Exact arithmetic over the parsed claim; the answer contract
    is true/false."""
    value = {"plus": a + b, "minus": a - b, "times": a * b}[op_word]
    return "true" if value == c else "false"


def claim_answer(text: str) -> dict[str, Any] | None:
    """Parse and answer a claim query, or None if it is not one."""
    claim = detect_claim(text)
    if claim is None:
        return None
    a, op, b, c = claim
    return {"a": a, "op": op, "b": b, "c": c,
            "answer": evaluate_claim(a, op, b, c)}


# ---- the M284b verdict-spoof rule (registered 23 Aug) ---------------
# Measured on the authored v1 suite: boolean-verdict sentiment
# phrasings ("A plus B is true, and that is my final answer to
# whether the film works.") route to the logic arm on the embedding
# router. The licensed repair: a review-context noun AND a
# true/false token routes to sentiment. It fires only on non-claims
# (the claim pre-pass runs first) and only where the text carries
# review context — formal boolean expressions never match.

REVIEW_NOUNS = frozenset({"film", "movie", "review", "critic",
                          "acting", "plot", "cast", "cinema",
                          "watch"})


def detect_verdict(text: str) -> bool:
    """Review context + a true/false token -> a sentiment verdict."""
    tokens = _WORD_RE.findall(text.lower())
    if not (REVIEW_NOUNS & set(tokens)):
        return False
    return "true" in tokens or "false" in tokens
