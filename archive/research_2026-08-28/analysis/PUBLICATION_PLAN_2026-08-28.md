# GEODE publication plan (M340)

Registered 2026-08-28, before any release step. The review's §5
finding: the paper says "the reference implementation, the
measurement records, and the replay tooling are published
alongside this paper", and the repository is not yet public. This
plan makes the sentence true or conditional — the paper now
carries the date-stamped conditional; this document registers the
release that completes it.

## 1. What ships

| Artifact                 | Contents                                                                                                                   | Notes                                                   |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| The whitepaper           | `analysis/WHITEPAPER_GEODE.pdf` (the built PDF) plus `WHITEPAPER_GEODE.tex`                                                | the measured-numbers edition; compiles clean            |
| Reference implementation | `geode/` (protocol modules) and `experiments/` (harnesses + configs)                                                       | the shipped module stack incl. this session's M330–M338 |
| Measurement records      | the sealed `logs/results/**/evidence.json` family + VOID records                                                           | the hash-chained evidence; artifact indexes included    |
| Replay tooling           | the pinned environment manifest (`.venv-rocm` spec, `requirements*.txt`), the artifact-index chain, and the replay scripts | the repo's registered replay discipline                 |

## 2. License

Per the licensing audit (LICENSING_AUDIT_v1.md): the code ships
under the recorded per-file terms (MIT/Apache-2.0 for the in-house
modules; the publisher-checkpoint weights are never redistributed
— they are pulled from their publishers under their own terms).
Nothing moves on the IMDb measurement row or the zakat-charter
legal character (the M188 counsel items) until counsel clears
them; those rows stay evaluation-only and charter-fixed
respectively.

## 3. Secrets scrub (before the release commit)

The M324 capability-audit checklist runs against the release
snapshot:

- EVM private keys: deployment-environment only, never in the
  repo — verified by scan.
- The authority-key registry and the testnet addresses: excluded
  by construction; the scan confirms no address files leak.
- History scrub: the git history contains no key material (the
  repo's discipline since M202); the release commit is clean by
  the same audit.

## 4. Release steps

1. Counsel sign-off on the M188 Q9/Q10 items (registered open).
2. The M324 audit passes on the release snapshot.
3. The repository flips to public with the audit evidence
   committed; the paper's conditional sentence is then replaced
   by the unqualified form in the release-tagged build.
4. Date-stamped release notes: what shipped, the commit hash, the
   audit results.

Until step 3 completes, the paper's "published alongside"
sentence remains the date-stamped conditional registered in
M340.
