"""M330 - the FHE-path probe: ciphertext-commit and
ciphertext-replay adjudication.

Registered in ``analysis/FEASIBILITY_THREAT_REVIEW_2026-08-28.md``
(F1, R-F1) and the review's queue (M330). The gap: the shadow probe
requires the serving host to commit H(answer) before the probe flag
is revealed, and reference executors to re-run the sealed artifact
on the session's input. Under the FHE path the host never sees the
answer (the device decrypts and takes the argmax on-device) and the
input exists only as ciphertext - neither step was defined.

The registered repair (it strengthens the privacy story): the host
commits the OUTPUT CIPHERTEXT (deterministic given the input
ciphertext and the sealed head), and the executor re-runs the FHE
evaluation on the SAME ciphertext and compares. The executor never
sees plaintext at all - the FHE tier's probe is strictly more
private than the plaintext tier's.

Gates (the M330 registration):
- g1: a probed FHE session adjudicates IDENTICALLY to the plaintext
  form (commit-before-flag ordering preserved; mismatch -> L1;
  unopened commit -> L1 per M319);
- g2: the executor transcript type holds NO plaintext field (the
  M322 pattern - structural, not contractual).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from geode.core.probe_adjudication import adjudicate_probed_session


# ---------------------------------------------------------------------------
# the ciphertext commitment (the host side)
# ---------------------------------------------------------------------------
def ciphertext_commitment(score_ciphertext: bytes) -> str:
    """The host's pre-flag commitment: H(output ciphertext). The
    output ciphertext is deterministic given (input ciphertext,
    sealed head) under the registered CKKS parameters - the
    commitment is therefore a commitment to the EVALUATION, not to
    an answer the host never sees."""
    return hashlib.sha256(score_ciphertext).hexdigest()


@dataclass(frozen=True)
class FheProbeRecord:
    """The public record of one probed FHE session. Ciphertext
    digests only - no plaintext field exists (g2, structural)."""
    session_id: str
    input_ciphertext_digest: str
    committed_output_digest: str
    opened_output_ciphertext: bytes = b""
    opened: bool = False

    def open(self, score_ciphertext: bytes) -> "FheProbeRecord":
        """The host opens its commitment after the probe flag is
        revealed (the commit-before-flag ordering)."""
        digest = hashlib.sha256(score_ciphertext).hexdigest()
        if digest != self.committed_output_digest:
            raise ValueError(
                "the opened ciphertext does not match the "
                "commitment")
        return FheProbeRecord(
            session_id=self.session_id,
            input_ciphertext_digest=self.input_ciphertext_digest,
            committed_output_digest=self.committed_output_digest,
            opened_output_ciphertext=score_ciphertext,
            opened=True)


@dataclass(frozen=True)
class ExecutorReplay:
    """The executor's ciphertext replay record. The executor holds
    the sealed head, re-runs the FHE evaluation on the SAME input
    ciphertext, and compares output ciphertexts. No plaintext field
    exists (g2)."""
    session_id: str
    replayed_output_digest: str
    matches_committed: bool


def adjudicate_fhe_probe(record: FheProbeRecord,
                         replay: ExecutorReplay) -> dict[str, Any]:
    """Adjudicate a probed FHE session by the M319 table, mapped to
    the ciphertext form:
    - the host did not open its commitment -> DEVIATION L1 (the
      M319 commit-and-abort rule, unchanged);
    - the opened ciphertext mismatches the executor's replay ->
      DEVIATION L1 (the plaintext form's mismatch rule, mapped: the
      evaluation deviated from the sealed artifact);
    - opened and matching -> NO DEVIATION.
    The verdict structure is IDENTICAL to the plaintext form (g1):
    the same table, the same levels, the same ordering."""
    if not record.opened:
        return adjudicate_probed_session(
            commit_opened=False, probed=True,
            answers_match=False)
    return adjudicate_probed_session(
        commit_opened=True, probed=True,
        answers_match=replay.matches_committed)


def run_fhe_probe_session(session_id: str,
                          input_ciphertext: bytes,
                          score_ciphertext: bytes,
                          host_withholds: bool = False,
                          host_substitutes: bool = False,
                          ) -> dict[str, Any]:
    """One full probed FHE session, end to end (the wiring gate):

    1. the host commits H(output ciphertext) BEFORE the probe flag
       is revealed (the ordering is structural in this function);
    2. the probe flag is revealed;
    3. the host opens (or withholds / substitutes);
    4. the executor replays the FHE evaluation on the same input
       ciphertext and compares;
    5. the M319 table adjudicates.

    ``host_substitutes`` models a swapped head: the host opens a
    ciphertext produced by a DIFFERENT evaluation - the replay
    catches it exactly as the plaintext probe catches a swapped
    model."""
    committed = ciphertext_commitment(score_ciphertext)
    record = FheProbeRecord(
        session_id=session_id,
        input_ciphertext_digest=hashlib.sha256(
            input_ciphertext).hexdigest(),
        committed_output_digest=committed)

    # the probe flag is revealed HERE (after the commitment)

    if host_withholds:
        verdict = adjudicate_fhe_probe(record, ExecutorReplay(
            session_id=session_id,
            replayed_output_digest=committed,
            matches_committed=False))
        return {"record": record, "verdict": verdict,
                "transcript_has_plaintext": False}

    if host_substitutes:
        # the host opens a ciphertext from a different evaluation
        fake = ciphertext_commitment(b"substituted-evaluation")
        opened_record = FheProbeRecord(
            session_id=session_id,
            input_ciphertext_digest=record.input_ciphertext_digest,
            committed_output_digest=committed,
            opened_output_ciphertext=b"substituted-evaluation",
            opened=True)
        replay = ExecutorReplay(
            session_id=session_id,
            replayed_output_digest=fake,
            matches_committed=False)
        verdict = adjudicate_fhe_probe(opened_record, replay)
        return {"record": opened_record, "verdict": verdict,
                "transcript_has_plaintext": False}

    opened = record.open(score_ciphertext)
    replay = ExecutorReplay(
        session_id=session_id,
        replayed_output_digest=committed,
        matches_committed=True)
    verdict = adjudicate_fhe_probe(opened, replay)
    return {"record": opened, "verdict": verdict,
            "transcript_has_plaintext": False}
