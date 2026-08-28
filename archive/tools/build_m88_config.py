"""Derive M88's config from M87's rather than retyping it.

N88.6 fixes the C1/C3/C5 query families at the registration commit. Retyping
them by hand is exactly how a family quietly acquires a friendlier wording
between the run that gave an unwelcome answer and the run that gives a better
one. Copying them verbatim from the sealed M87 config makes that impossible to
do by accident, and the equality assertion at the end makes it impossible to do
on purpose without the script complaining.
"""

from __future__ import annotations

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "experiments/configs/v13/m87_prior_art_audit.json"
TARGET = REPO_ROOT / "experiments/configs/v13/m88_prior_art_recheck.json"

REOPENED = ("C1", "C3", "C5")

RECALL_PROBES = [
    {
        "id": "P1",
        "must_retrieve": "Deep Anomaly Detection with Outlier Exposure",
        "query": "anomaly detection auxiliary dataset of outliers improves detection",
        "covers": ["C1"],
        "positive_control": False,
    },
    {
        "id": "P2",
        "must_retrieve": "Exposing Outlier Exposure",
        "query": "how many outlier images are needed for anomaly detection one-class",
        "covers": ["C1"],
        "positive_control": False,
    },
    {
        "id": "P3",
        "must_retrieve": "Fixing the train-test resolution discrepancy",
        "query": "train test resolution discrepancy classification accuracy",
        "covers": ["C3"],
        "positive_control": False,
    },
    {
        "id": "P4",
        "must_retrieve": "Sanity Checks for Saliency Maps",
        "query": "randomizing model weights leaves saliency maps unchanged",
        "covers": ["C5"],
        "positive_control": False,
    },
    {
        "id": "P5",
        "must_retrieve": "Which Algorithmic Explanations Help Users Predict Model Behavior",
        "query": "do explanations help users predict model behaviour forward simulation",
        "covers": ["C5"],
        "positive_control": False,
    },
    {
        "id": "P6",
        "must_retrieve": "Full-Spectrum Out-of-Distribution Detection",
        "query": "benchmark separating covariate shift from semantic shift detection",
        "covers": ["C2"],
        "positive_control": True,
    },
    {
        "id": "P7",
        "must_retrieve": "Network Dissection",
        "query": "quantifying interpretability of individual units visual concepts",
        "covers": ["C4"],
        "positive_control": False,
    },
]

NOTES = {
    "N88.1": (
        "M88's scope is closed to the three obligations M87 wrote against itself in "
        "section 7.2. No sixth claim. C2's demotion is final; a second search is not "
        "an appeal."
    ),
    "N88.2": (
        "Recall probes: named papers that certainly exist, each paired with a topic "
        "query that omits the paper's title. If a topic query aimed at a paper cannot "
        "retrieve it, 'found nothing' from the family that probe covers is not "
        "evidence of absence, and that family drops to 'not searched'."
    ),
    "N88.3": (
        "Adding an index is asymmetric. A third index can only find displacing work; "
        "it can never establish absence. A claim surviving three indexes is not better "
        "supported than one that survived two - it has merely failed to be refuted "
        "more times. No verdict may be upgraded for having searched more places."
    ),
    "N88.4": (
        "Re-running a family does not reset it. M87's verdicts stand unless this run "
        "displaces them. A re-run may move a claim down to narrowed, displaced or "
        "not-searched, and may never move one up."
    ),
    "N88.5": (
        "Reading Liznerski et al. in full may displace or narrow C1 and cannot clear "
        "it. Confirming that one paper does not anticipate a claim says nothing about "
        "the papers not read."
    ),
    "N88.6": "The probe list and the reopened query families are fixed at the registration commit.",
    "N88.7": (
        "Like M87, this evidence is a dated snapshot of third-party services and is "
        "NOT replayable. Re-running it will not reproduce these records."
    ),
}


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))

    config = {
        "milestone": "m88_prior_art_recheck",
        "program": "v13 discharge of M87's registered consequences",
        "derived_from": {
            "config": SOURCE.relative_to(REPO_ROOT).as_posix(),
            "reopened_claims": list(REOPENED),
        },
        "registration_notes": {**source["registration_notes"], **NOTES},
        "sources": {
            **source["sources"],
            "openalex": {
                "endpoint": "https://api.openalex.org/works",
                "max_results": 20,
                "delay_seconds": 1.5,
                "attempts": 4,
            },
        },
        "anchors": list(source["anchors"]),
        "recall_probes": RECALL_PROBES,
        # Copied, never retyped.
        "claims": {claim_id: source["claims"][claim_id] for claim_id in REOPENED},
        "output_dir": "logs/results/v13/m88_prior_art_recheck",
    }

    for claim_id in REOPENED:
        assert config["claims"][claim_id]["queries"] == source["claims"][claim_id]["queries"], claim_id

    covered = {family for probe in RECALL_PROBES for family in probe["covers"]}
    assert covered == set(source["claims"]), f"probes must cover every M87 claim, got {sorted(covered)}"

    TARGET.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {TARGET.relative_to(REPO_ROOT)}")
    print(f"  reopened families: {', '.join(REOPENED)}")
    print(f"  probes: {len(RECALL_PROBES)} covering {', '.join(sorted(covered))}")
    print(f"  indexes: {', '.join(sorted(config['sources']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
