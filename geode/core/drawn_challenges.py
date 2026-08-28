"""M308 — drawn-challenge admission for finding A8 (26 Aug 2026).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M308
before any build. A8: validators author their own exams, so scores are
not commensurable and routing has no basis. Repair (R-A8): validators
must DRAW, not author. Challenges come from a registered sealed
per-axis corpus under a published stratified sampling rule; the
validator's job is to sample, pose, verify, and attest — never to
choose the exam. Validator-authored challenges may remain as a
supplementary stream that is reported separately and never enters the
routable score.
"""
from __future__ import annotations

import hashlib
from typing import Any


def register_corpus(rows: list[bytes], labels: list[int]
                    ) -> dict[str, Any]:
    """Commit to the sealed per-axis corpus: a Merkle root over the
    challenge rows plus a root over the labels. Neither can change
    without invalidating the commitment."""
    if not rows or len(rows) != len(labels):
        raise ValueError("rows and labels must be non-empty and equal "
                         "length")
    row_root = _merkle(rows)
    label_root = _merkle([int(l).to_bytes(8, "big", signed=True)
                          for l in labels])
    return {"row_count": len(rows),
            "class_count": len(set(labels)),
            "row_root": row_root,
            "label_root": label_root}


def stratified_draw(corpus: dict[str, Any], labels: list[int],
                    beacon_seed: str, epoch: int, count: int,
                    class_shares: dict[int, float] | None = None
                    ) -> list[int]:
    """The published stratified sampling rule: draw ``count`` rows,
    stratum by stratum (equal share per class unless ``class_shares``
    is registered), seeded by (beacon, epoch) so the draw is
    deterministic and ungrindable after commitment."""
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    n = int(corpus["row_count"])
    if len(labels) != n:
        raise ValueError("labels must match the committed corpus")
    classes = sorted(set(labels))
    if class_shares is not None:
        total_share = sum(float(class_shares.get(c, 0.0))
                          for c in classes)
        if total_share <= 0.0:
            raise ValueError("class_shares must be positive")
        per_class = {c: max(0, int(round(
            count * float(class_shares.get(c, 0.0)) / total_share)))
                     for c in classes}
    else:
        base, rem = divmod(int(count), len(classes))
        per_class = {c: base + (1 if i < rem else 0)
                     for i, c in enumerate(classes)}
    out: list[int] = []
    counter = 0
    for c in classes:
        indices = [i for i, lab in enumerate(labels) if lab == c]
        if not indices:
            continue
        wanted = int(per_class[c])
        got = 0
        local_counter = counter
        while got < wanted:
            h = hashlib.sha256(
                f"geode:draw:{beacon_seed}:{epoch}:{c}:{local_counter}"
                .encode("utf-8")).digest()
            index = indices[int.from_bytes(h[:8], "big") % len(indices)]
            if index not in out:
                out.append(index)
                got += 1
            local_counter += 1
        counter = local_counter
    return sorted(out)


def pose_challenge(corpus: dict[str, Any], row_index: int
                   ) -> dict[str, Any]:
    """Reveal a drawn row index as a challenge."""
    if not 0 <= int(row_index) < int(corpus["row_count"]):
        raise ValueError("row_index outside the committed corpus")
    return {"row_index": int(row_index),
            "corpus_row_root": corpus["row_root"]}


def verify_answer(corpus: dict[str, Any], rows: list[bytes],
                  labels: list[int], row_index: int, answer: int
                  ) -> bool:
    """Check an answer against the sealed label, after re-deriving
    both Merkle roots so a tampered corpus cannot score."""
    if _merkle(rows) != corpus["row_root"]:
        return False
    label_root = _merkle([int(l).to_bytes(8, "big", signed=True)
                          for l in labels])
    if label_root != corpus["label_root"]:
        return False
    if not 0 <= int(row_index) < len(labels):
        return False
    return int(answer) == int(labels[int(row_index)])


def score_draw(corpus: dict[str, Any], rows: list[bytes],
               labels: list[int], draw: list[int], answers: list[int]
               ) -> dict[str, Any]:
    """Bulk scoring of a drawn challenge set: the corpus roots are
    re-derived ONCE (a tampered corpus scores nothing), then every
    answer is checked against its sealed label. Returns the routable
    score plus per-challenge correctness."""
    if _merkle(rows) != corpus["row_root"]:
        return {"valid": False, "score": None,
                "reason": "row root mismatch"}
    label_root = _merkle([int(l).to_bytes(8, "big", signed=True)
                          for l in labels])
    if label_root != corpus["label_root"]:
        return {"valid": False, "score": None,
                "reason": "label root mismatch"}
    if len(draw) != len(answers):
        raise ValueError("draw and answers must match in length")
    correct = [bool(0 <= int(i) < len(labels)
                    and int(a) == int(labels[int(i)]))
               for i, a in zip(draw, answers)]
    out = routable_score(correct)
    out["valid"] = True
    out["correct_by_challenge"] = correct
    return out


def routable_score(correct: list[bool]) -> dict[str, Any]:
    """The routable score: the fraction correct over drawn challenges
    — an estimate of one fixed population quantity for every artifact
    on the axis."""
    if not correct:
        raise ValueError("correct cannot be empty")
    hits = sum(1 for c in correct if bool(c))
    return {"score": hits / len(correct),
            "answered": len(correct),
            "correct": hits}


def supplementary_stream(challenges: list[dict[str, Any]]
                         ) -> dict[str, Any]:
    """Validator-authored challenges: reported separately, never
    entering the routable score."""
    return {"authored_count": len(challenges),
            "reported_separately": True,
            "enters_routable_score": False}


def _merkle(values: list[bytes]) -> str:
    level = [hashlib.sha256(b"\x00" + v).digest() for v in values]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [hashlib.sha256(b"\x01" + a + b).digest()
                 for a, b in zip(level[0::2], level[1::2])]
    return level[0].hex()
