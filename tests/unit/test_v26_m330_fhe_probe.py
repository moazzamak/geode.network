"""Unit tests for the M330 FHE-path probe (registered 28 Aug 2026,
before the build). Pins: the ciphertext commitment form; the
commit-before-flag ordering; the M319-identical adjudication (g1:
clean / mismatch -> L1 / withhold -> L1); the structural no-plaintext
transcript (g2); the substitution catch."""
from __future__ import annotations

import hashlib

from geode.privacy.fhe_probe import (
    ExecutorReplay,
    FheProbeRecord,
    adjudicate_fhe_probe,
    ciphertext_commitment,
    run_fhe_probe_session,
)


def test_ciphertext_commitment_form():
    ct = b"some-ciphertext-bytes"
    assert ciphertext_commitment(ct) == hashlib.sha256(ct).hexdigest()
    # deterministic
    assert ciphertext_commitment(ct) == ciphertext_commitment(ct)
    # distinct ciphertexts commit distinctly
    assert ciphertext_commitment(ct) != ciphertext_commitment(
        ct + b"x")


def test_commit_before_flag_ordering():
    # the wiring commits BEFORE the probe flag: the record exists
    # with the commitment digest before any open() call
    ct_in, ct_out = b"input-ct", b"output-ct"
    committed = ciphertext_commitment(ct_out)
    record = FheProbeRecord(
        session_id="s1",
        input_ciphertext_digest=hashlib.sha256(ct_in).hexdigest(),
        committed_output_digest=committed)
    assert not record.opened
    # the open happens only after the flag (the caller's step 3)
    opened = record.open(ct_out)
    assert opened.opened
    assert opened.opened_output_ciphertext == ct_out


def test_open_validates_against_commitment():
    record = FheProbeRecord(
        session_id="s1",
        input_ciphertext_digest="d" * 64,
        committed_output_digest=ciphertext_commitment(b"real"))
    import pytest
    with pytest.raises(ValueError):
        record.open(b"not-the-committed-ciphertext")


def test_g1_clean_session_adjudicates_like_plaintext():
    result = run_fhe_probe_session(
        session_id="s1", input_ciphertext=b"in",
        score_ciphertext=b"out")
    assert result["verdict"]["verdict"] == "clean"
    assert result["verdict"]["ladder_level"] is None


def test_g1_mismatch_adjudicates_l1_like_plaintext():
    result = run_fhe_probe_session(
        session_id="s1", input_ciphertext=b"in",
        score_ciphertext=b"out", host_substitutes=True)
    assert result["verdict"]["verdict"] == "deviation"
    assert result["verdict"]["ladder_level"] == 1
    assert result["verdict"]["basis"] == "opened mismatch"


def test_g1_withhold_adjudicates_l1_per_m319():
    result = run_fhe_probe_session(
        session_id="s1", input_ciphertext=b"in",
        score_ciphertext=b"out", host_withholds=True)
    assert result["verdict"]["verdict"] == "deviation"
    assert result["verdict"]["ladder_level"] == 1
    assert "A18" in result["verdict"]["basis"]


def test_g2_no_plaintext_field_in_transcript():
    # structural: the record and the replay carry digests and
    # ciphertexts only; the dataclass has no plaintext attribute
    record_fields = {f for f in FheProbeRecord.__dataclass_fields__}
    assert "plaintext" not in record_fields
    assert "input_plaintext" not in record_fields
    assert "answer" not in record_fields
    replay_fields = {f for f in ExecutorReplay.__dataclass_fields__}
    assert "plaintext" not in replay_fields
    # the wiring reports the structural property
    result = run_fhe_probe_session("s1", b"in", b"out")
    assert result["transcript_has_plaintext"] is False


def test_substitution_caught_by_replay():
    # a swapped head produces a different output ciphertext; the
    # executor's replay of the sealed head catches it
    result = run_fhe_probe_session(
        session_id="s1", input_ciphertext=b"in",
        score_ciphertext=b"out", host_substitutes=True)
    assert result["record"].opened
    assert result["verdict"]["verdict"] == "deviation"


def test_adjudicate_fhe_probe_direct():
    ct = b"the-real-output-ciphertext"
    record = FheProbeRecord(
        session_id="s", input_ciphertext_digest="d",
        committed_output_digest=ciphertext_commitment(ct))
    # unopened + probed -> A18 deviation
    verdict = adjudicate_fhe_probe(record, ExecutorReplay(
        session_id="s", replayed_output_digest="c",
        matches_committed=True))
    assert verdict["ladder_level"] == 1
    # opened + match -> clean
    opened = record.open(ct)
    verdict = adjudicate_fhe_probe(opened, ExecutorReplay(
        session_id="s",
        replayed_output_digest=ciphertext_commitment(ct),
        matches_committed=True))
    assert verdict["verdict"] == "clean"
