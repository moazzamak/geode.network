"""M82 — do M80's atoms admit stable names, and does naming raise I5?

H82 as registered: the atoms M80 gated admit stable natural-language names, and
supplying those names raises forward simulatability by turning coordinate
indices into concepts.

Two amendments bound what this runner may conclude, and both were registered
before it ran.

**R9** withdrew the "two independent naming channels" operand. SpLiCE
decomposes CLIP image embeddings, where text and image share a space; v13
decomposes 384-dimensional DINOv2 features, where they do not. A v13 atom
cannot be dotted with a text embedding at all, so the only route from an atom
to a phrase runs through its top-activating images — which is also the exemplar
channel's route. The two channels are one channel. In its place: exemplar
resampling stability is the primary operand, and a class-purity positive
control is **gating**, because a channel that cannot name an atom whose
exemplars are three-quarters one class is not naming anything.

**R8** separated naming from identity revelation. M81 measured I5 with
component identity withheld, so the gap between it and a named explanation
would confound two changes. Three arms are therefore run on identical splits,
atoms, budgets and widths: identity withheld, identity revealed but unnamed,
and identity revealed and named. The naming claim is the third minus the
second, never the third minus the first.

**N82.4** sharpens R8's arm (b). Naming does not only reveal an atom's
identity, it groups many atoms under one word, so a per-atom arm differs from a
named arm in width as well as in meaning. Arm (b) is therefore also run as a
matched-size random grouping, and the claim rests on that comparison.

**R8** also fixed the task width. M81 found no accuracy-comparable atom arm
inside the ten-atom budget at 128 classes at any tolerance, so there is no
128-way baseline for naming to raise. Every gated number here is an eight-way
number.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v13_heads import (
    cited_atoms_per_decision,
    fit_sparse_linear,
    withheld_explanation,
)
from experiments.common.v13_i5 import (
    forward_simulation,
    shuffled_explanation_control,
)
from experiments.common.v13_linear_probe import balanced_accuracy
from experiments.common.v13_naming import (
    AtomExemplars,
    atom_class_purity,
    domain_breakdown,
    far_field_rate,
    grouped_explanation,
    matched_random_grouping,
    name_atoms,
    names_to_groups,
    naming_agreement,
    purity_positive_control,
    shuffled_exemplars,
    sparse_atom_exemplars,
    split_exemplars,
)
from experiments.common.v13_sparse_dictionary import (
    SparseCodes,
    fit_sparse_dictionary,
)
from experiments.tier4.eval_v13_m80_sparse_dictionary import (
    _load_corpus,
    _partition,
    _verify_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v13" / "m82_atom_naming.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v13" / "m82_atom_naming"

_WORKER_STATE: dict[str, Any] = {}


def _resolve(path: str) -> Path:
    return (REPO_ROOT / path).resolve()


def _subset(codes: SparseCodes, rows: np.ndarray) -> SparseCodes:
    return SparseCodes(
        codes.indices[rows].copy(), codes.values[rows].copy(), codes.dictionary_size
    )


def _verify_clip(specification: dict[str, Any]) -> Path:
    """The CLIP embeddings are a sealed input artifact, verified like the
    corpus. They were produced outside the frozen replay venv, on the GPU, so
    the hash check is the only thing standing between this milestone and a
    silently regenerated input."""
    index_path = _resolve(specification["path"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for artifact in index["artifacts"]:
        artifact_path = index_path.parent / artifact["path"]
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"M82 CLIP artifact hash mismatch: {artifact_path}")
    evidence = json.loads(
        (index_path.parent / "evidence.json").read_text(encoding="utf-8")
    )
    if evidence["vocabulary_hash"] != specification["vocabulary_hash"]:
        raise ValueError("M82 CLIP embeddings were built from another vocabulary")
    if not evidence["cpu_agreement_control"]["passes"]:
        raise ValueError("M82 CLIP embeddings failed their own GPU/CPU control")
    return index_path


def _load_clip(index_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Image embeddings and the 351-term vocabulary, objects then styles.

    Concatenating the two term tables is what lets an atom be named by a
    rendering style rather than an object. N82.3 records that the style half is
    weak; it is kept because the vocabulary was sealed before any of this was
    measured."""
    arrays = index_path.parent / "arrays"
    images = np.load(arrays / "image_embeddings.npy").astype(np.float32)
    objects = np.load(arrays / "text_object_embeddings.npy").astype(np.float32)
    styles = np.load(arrays / "text_style_embeddings.npy").astype(np.float32)
    return images, np.concatenate([objects, styles], axis=0)


def _load_domains(manifest_path: Path) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return np.array(
        [row["domain"] for row in manifest["selection"]], dtype=np.int64
    )


def _name(
    exemplar_rows: list[np.ndarray],
    global_rows: np.ndarray,
    images: np.ndarray,
    terms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Name atoms, translating fit-split row numbers into corpus row numbers.

    The dictionary is fitted on a 65,536-row subset, but the CLIP embeddings
    are indexed by the full 73,728-row corpus. Getting this mapping wrong would
    name every atom after some unrelated image and still produce plausible
    output, so it is done in one place."""
    mapped = [global_rows[rows] for rows in exemplar_rows]
    return name_atoms(mapped, images, terms)


def _naming_channel(
    exemplars: AtomExemplars,
    global_rows: np.ndarray,
    images: np.ndarray,
    terms: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    naming = config["naming"]
    class_count = int(config["clip"]["in_corpus_terms"])
    purity_threshold = float(naming["purity_threshold"])

    names, scores = _name(exemplars.rows, global_rows, images, terms)

    # R9's primary operand. Disjoint halves of each atom's exemplar set are
    # named independently; a name that survives the split is a property of the
    # atom rather than of the particular pictures that happened to be drawn.
    first_rows, second_rows = split_exemplars(
        exemplars, seed=seed + int(naming["split_seed_offset"])
    )
    first_names, _ = _name(first_rows, global_rows, images, terms)
    second_names, _ = _name(second_rows, global_rows, images, terms)
    stability = naming_agreement(first_names, second_names)

    # R6's null at matched set sizes: exemplar sets reassigned across atoms,
    # then split and named the same way. Whatever agreement the vocabulary's
    # own geometry produces for free shows up here.
    scrambled = AtomExemplars(
        rows=shuffled_exemplars(
            exemplars, seed=seed + int(naming["shuffle_seed_offset"])
        ),
        activations=exemplars.activations,
    )
    null_first, null_second = split_exemplars(
        scrambled, seed=seed + int(naming["split_seed_offset"])
    )
    null_first_names, _ = _name(null_first, global_rows, images, terms)
    null_second_names, _ = _name(null_second, global_rows, images, terms)
    stability_null = naming_agreement(null_first_names, null_second_names)

    # The gating control. Atoms whose exemplars are at least three-quarters one
    # class have a ground-truth name, and the channel must recover it.
    dominant, purity = atom_class_purity(
        [global_rows[rows] for rows in exemplars.rows],
        labels,
        class_count=class_count,
    )
    control = purity_positive_control(
        names,
        [global_rows[rows] for rows in exemplars.rows],
        labels,
        class_count=class_count,
        purity_threshold=purity_threshold,
    )
    generator = np.random.default_rng(seed + 82_300)
    control_null = purity_positive_control(
        names[generator.permutation(len(names))],
        [global_rows[rows] for rows in exemplars.rows],
        labels,
        class_count=class_count,
        purity_threshold=purity_threshold,
    )

    named = names >= 0
    margin = (
        None
        if stability["agreement"] is None or stability_null["agreement"] is None
        else float(stability["agreement"] - stability_null["agreement"])
    )
    return {
        "atoms": int(exemplars.atom_count),
        "live_atoms": int(exemplars.live().sum()),
        "named_atoms": int(named.sum()),
        "distinct_names": int(len(np.unique(names[named]))) if named.any() else 0,
        "mean_name_score": float(np.nanmean(scores)) if named.any() else None,
        "style_named_atoms": int((names >= int(config["clip"]["style_terms_start"])).sum()),
        "stability": stability,
        "stability_null": stability_null,
        "stability_margin": margin,
        "purity_control": control,
        "purity_control_null": control_null,
        "pure_atoms": int((purity >= purity_threshold).sum()),
        "far_field": far_field_rate(
            names,
            in_corpus_terms=class_count,
            style_terms_start=int(config["clip"]["style_terms_start"]),
        ),
        "per_domain": domain_breakdown(
            names,
            [global_rows[rows] for rows in exemplars.rows],
            labels,
            domains,
            class_count=class_count,
            domain_count=int(config["corpus"]["domain_count"]),
            purity_threshold=purity_threshold,
        ),
        "names": names.tolist(),
        "dominant_class": dominant.tolist(),
    }


def _i5(
    explanations: np.ndarray,
    predicted: np.ndarray,
    *,
    class_count: int,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    shared = dict(
        class_count=class_count,
        train_fraction=float(config["i5"]["train_fraction"]),
        max_iter=int(config["i5"]["max_iter"]),
        seed=seed,
    )
    simulation = forward_simulation(explanations, predicted, **shared)
    null = shuffled_explanation_control(explanations, predicted, **shared)
    measured = simulation["probe_balanced_accuracy"]
    null_measured = null["probe_balanced_accuracy"]
    return {
        "i5": simulation,
        "i5_shuffled_null": null,
        "explanation_width": int(explanations.shape[1]),
        "i5_margin_over_null": (
            None
            if measured is None or null_measured is None
            else float(measured - null_measured)
        ),
    }


def _three_arms(
    fit_codes: SparseCodes,
    fit_labels: np.ndarray,
    evaluation_codes: SparseCodes,
    evaluation_labels: np.ndarray,
    names: np.ndarray,
    *,
    class_count: int,
    atom_budget: int,
    tag: str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """R8's three arms on one head, at identical split, atoms and budget."""
    heads = config["heads"]
    head = fit_sparse_linear(
        fit_codes,
        fit_labels,
        class_count=class_count,
        l1_penalty=0.0,
        epochs=int(heads["epochs"]),
        batch_size=int(heads["batch_size"]),
        learning_rate=float(heads["learning_rate"]),
        seed=seed,
        atom_budget=atom_budget,
    )
    predicted = head.scores(evaluation_codes).argmax(1)
    contributions = head.contributions(evaluation_codes, predicted)
    indices = evaluation_codes.indices

    active = np.flatnonzero(
        (head.weight.abs() > 1e-8).any(dim=0).numpy()
    ).astype(np.int64)
    dictionary_size = evaluation_codes.dictionary_size

    # (c) named. Only the head's own atoms can appear in its explanation, so
    # the grouping is built over those and the unnamed among them are dropped
    # — the reader is told nothing about an atom the channel could not name.
    named_groups, used_terms = names_to_groups(names[active])
    named_assignment = np.full(dictionary_size, -1, dtype=np.int64)
    named_assignment[active] = named_groups
    group_count = len(used_terms)

    # (b2) N82.4's matched-size random grouping. Permuting the assignment
    # vector over the same atoms preserves group count, group sizes and the
    # number of ungrouped atoms exactly.
    random_assignment = np.full(dictionary_size, -1, dtype=np.int64)
    random_assignment[active] = matched_random_grouping(
        named_groups, seed=seed + 82_400
    )

    # (b1) R8's literal wording: every atom its own stable arbitrary column.
    per_atom_assignment = np.full(dictionary_size, -1, dtype=np.int64)
    per_atom_assignment[active] = np.arange(len(active), dtype=np.int64)

    withheld = withheld_explanation(
        contributions, top_count=int(config["explanations"]["top_count"])
    )
    arms = {
        "a_identity_withheld": _i5(
            withheld, predicted, class_count=class_count, config=config, seed=seed
        ),
        "b1_revealed_per_atom": _i5(
            grouped_explanation(
                indices, contributions, per_atom_assignment, group_count=len(active)
            ),
            predicted,
            class_count=class_count,
            config=config,
            seed=seed,
        ),
        "b2_revealed_matched_random": _i5(
            grouped_explanation(
                indices, contributions, random_assignment, group_count=group_count
            )
            if group_count
            else withheld,
            predicted,
            class_count=class_count,
            config=config,
            seed=seed,
        ),
        "c_revealed_named": _i5(
            grouped_explanation(
                indices, contributions, named_assignment, group_count=group_count
            )
            if group_count
            else withheld,
            predicted,
            class_count=class_count,
            config=config,
            seed=seed,
        ),
    }

    def score(arm: str) -> float | None:
        return arms[arm]["i5"]["probe_balanced_accuracy"]

    def delta(left: str, right: str) -> float | None:
        first, second = score(left), score(right)
        return None if first is None or second is None else float(first - second)

    return {
        "arm": tag,
        "atom_budget": atom_budget,
        "balanced_accuracy": balanced_accuracy(predicted, evaluation_labels),
        "active_atoms_in_head": int(len(active)),
        "named_atoms_in_head": int((names[active] >= 0).sum()),
        "distinct_names_in_head": group_count,
        "explanation_length": cited_atoms_per_decision(
            contributions, budget=int(config["explanations"]["budget"])
        ),
        "arms": arms,
        "naming_delta": delta("c_revealed_named", "b2_revealed_matched_random"),
        "revelation_delta": delta("b2_revealed_matched_random", "a_identity_withheld"),
        "per_atom_delta": delta("b1_revealed_per_atom", "a_identity_withheld"),
    }


def _worker_init(config: dict[str, Any]) -> None:
    torch.set_num_threads(int(config["threading"]["torch_threads_per_worker"]))
    index_path = _verify_corpus(config["corpus"])
    features, labels = _load_corpus(index_path)
    domains = _load_domains(_resolve(config["corpus"]["manifest_path"]))
    if len(domains) != len(labels):
        raise ValueError("M82 manifest and corpus disagree on row count")
    images, terms = _load_clip(_verify_clip(config["clip"]))
    if len(images) != len(labels):
        raise ValueError("M82 CLIP embeddings and corpus disagree on row count")

    partition = config["partition"]
    fit_rows, evaluation_rows = _partition(
        labels,
        fit_per_class=int(partition["fit_per_class"]),
        evaluation_per_class=int(partition["evaluation_per_class"]),
    )
    _WORKER_STATE.update(
        config=config,
        fit_rows=fit_rows,
        fit_features=features[fit_rows],
        fit_labels=labels[fit_rows],
        evaluation_features=features[evaluation_rows],
        evaluation_labels=labels[evaluation_rows],
        labels=labels,
        domains=domains,
        images=images,
        terms=terms,
    )


def _worker_run(seed: int) -> dict[str, Any]:
    state = _WORKER_STATE
    config = state["config"]
    started = time.time()

    specification = config["dictionary"]
    dictionary, diagnostics = fit_sparse_dictionary(
        state["fit_features"],
        dictionary_size=int(specification["dictionary_size"]),
        active_atoms=int(specification["active_atoms"]),
        epochs=int(specification["epochs"]),
        batch_size=int(specification["batch_size"]),
        learning_rate=float(specification["learning_rate"]),
        seed=seed,
    )
    fit_codes = dictionary.codes(state["fit_features"])
    evaluation_codes = dictionary.codes(state["evaluation_features"])

    exemplars = sparse_atom_exemplars(
        fit_codes.indices,
        fit_codes.values,
        dictionary_size=fit_codes.dictionary_size,
        top_count=int(config["naming"]["exemplars_per_atom"]),
    )
    channel = _naming_channel(
        exemplars,
        state["fit_rows"],
        state["images"],
        state["terms"],
        state["labels"],
        state["domains"],
        config=config,
        seed=seed,
    )
    names = np.asarray(channel["names"], dtype=np.int64)

    # Eight-way only, per R8. The 128-class basis is reused unchanged; only the
    # heads are retrained, exactly as M81 did, so the two are commensurable.
    chosen = np.asarray(config["i5_eight"]["classes"], dtype=np.int64)
    remap = {label: position for position, label in enumerate(chosen)}
    fit_mask = np.isin(state["fit_labels"], chosen)
    evaluation_mask = np.isin(state["evaluation_labels"], chosen)
    eight_fit_codes = _subset(fit_codes, np.flatnonzero(fit_mask))
    eight_evaluation_codes = _subset(
        evaluation_codes, np.flatnonzero(evaluation_mask)
    )
    eight_fit_labels = np.array(
        [remap[int(label)] for label in state["fit_labels"][fit_mask]],
        dtype=np.int64,
    )
    eight_evaluation_labels = np.array(
        [
            remap[int(label)]
            for label in state["evaluation_labels"][evaluation_mask]
        ],
        dtype=np.int64,
    )

    arms = [
        _three_arms(
            eight_fit_codes,
            eight_fit_labels,
            eight_evaluation_codes,
            eight_evaluation_labels,
            names,
            class_count=len(chosen),
            atom_budget=budget,
            tag=f"sparse_linear_budget_{budget}",
            config=config,
            seed=seed,
        )
        for budget in config["heads"]["atom_budgets"]
    ]

    return {
        "seed": seed,
        "dictionary": {
            "dictionary_size": int(specification["dictionary_size"]),
            "active_atoms": int(specification["active_atoms"]),
            "final_train_loss": diagnostics["final_train_loss"],
            "loss_decreased": diagnostics["loss_decreased"],
            # N82.6: computed from atom_usage, which counts rows per atom.
            # active_atom_count counts atoms per row, and dividing it by the
            # dictionary size — as M81 and M80 do — yields a per-row array of
            # near-1.0 values that is not a dead-atom fraction at all.
            "dead_atom_fraction": float(
                np.mean(fit_codes.atom_usage() == 0)
            ),
        },
        "naming": channel,
        "i5_eight": arms,
        "elapsed_seconds": time.time() - started,
    }


def _spread(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(max(finite) - min(finite)) if len(finite) > 1 else None


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _gate(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """The verdict, in the order R9 registered it.

    The purity control is checked first and is gating. If the instrument cannot
    name an atom whose exemplars are three-quarters one class, no downstream
    naming number means anything and none is quoted.
    """
    naming = [result["naming"] for result in results]
    control_scores = [entry["purity_control"]["accuracy"] for entry in naming]
    control_nulls = [entry["purity_control_null"]["accuracy"] for entry in naming]
    control_mean = _mean(control_scores)
    floor = float(config["naming"]["purity_control_floor"])
    control_passes = control_mean is not None and control_mean >= floor

    stability_scores = [entry["stability"]["agreement"] for entry in naming]
    stability_nulls = [entry["stability_null"]["agreement"] for entry in naming]
    stability_margins = [entry["stability_margin"] for entry in naming]
    stability_mean = _mean(stability_margins)
    stability_spread = _spread(stability_margins)
    stability_passes = (
        stability_mean is not None
        and stability_mean > 0.0
        and (stability_spread is None or stability_mean > stability_spread)
    )

    carried = config["heads"]["carried_arm"]
    per_arm: dict[str, Any] = {}
    for arm_index, tag in enumerate(
        f"sparse_linear_budget_{budget}"
        for budget in config["heads"]["atom_budgets"]
    ):
        deltas = [result["i5_eight"][arm_index]["naming_delta"] for result in results]
        revelation = [
            result["i5_eight"][arm_index]["revelation_delta"] for result in results
        ]
        per_arm[tag] = {
            "naming_delta_mean": _mean(deltas),
            "naming_delta_spread": _spread(deltas),
            "naming_delta_per_seed": deltas,
            "revelation_delta_mean": _mean(revelation),
            "i5_named_mean": _mean(
                [
                    result["i5_eight"][arm_index]["arms"]["c_revealed_named"]["i5"][
                        "probe_balanced_accuracy"
                    ]
                    for result in results
                ]
            ),
            "i5_matched_random_mean": _mean(
                [
                    result["i5_eight"][arm_index]["arms"][
                        "b2_revealed_matched_random"
                    ]["i5"]["probe_balanced_accuracy"]
                    for result in results
                ]
            ),
            "i5_withheld_mean": _mean(
                [
                    result["i5_eight"][arm_index]["arms"]["a_identity_withheld"][
                        "i5"
                    ]["probe_balanced_accuracy"]
                    for result in results
                ]
            ),
        }

    carried_arm = per_arm[carried]
    delta_mean = carried_arm["naming_delta_mean"]
    delta_spread = carried_arm["naming_delta_spread"]
    naming_passes = (
        delta_mean is not None
        and delta_mean > 0.0
        and (delta_spread is None or delta_mean > delta_spread)
    )

    if not control_passes:
        verdict = "instrument_failed"
    elif not stability_passes:
        verdict = "names_unstable"
    elif naming_passes:
        verdict = "confirmed"
    else:
        verdict = "no_naming_gain"

    return {
        "task_width": config["gate"]["task_width"],
        "carried_arm": carried,
        "purity_control": {
            "is_gating": True,
            "floor": floor,
            "mean": control_mean,
            "per_seed": control_scores,
            "null_mean": _mean(control_nulls),
            "passes": control_passes,
        },
        "stability": {
            "agreement_mean": _mean(stability_scores),
            "null_mean": _mean(stability_nulls),
            "margin_mean": stability_mean,
            "margin_spread": stability_spread,
            "margin_per_seed": stability_margins,
            "passes": stability_passes,
        },
        "naming_delta": per_arm,
        "far_field_rate_mean": _mean(
            [entry["far_field"]["false_naming_rate"] for entry in naming]
        ),
        "verdict": verdict,
        "hypothesis_supported": verdict == "confirmed",
        "notes": config["registration_notes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M82 atom naming")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Override the registered seeds. For smoke runs only; a subset "
        "does not seal.",
    )
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output_dir = arguments.output or _resolve(config["output_dir"])
    registered_seeds = [int(seed) for seed in config["seeds"]]
    seeds = arguments.seeds if arguments.seeds is not None else registered_seeds
    sealed = seeds == registered_seeds

    started = time.time()
    workers = min(int(config["threading"]["workers"]), len(seeds))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(config,)
    ) as pool:
        results = list(pool.map(_worker_run, seeds))
    elapsed = time.time() - started

    gate = _gate(results, config)
    evidence = {
        "schema_version": 1,
        "milestone": "M82",
        "hypothesis": config["registered_hypothesis"],
        "seeds": seeds,
        "sealed": sealed,
        "corpus_sha256": config["corpus"]["sha256"],
        "vocabulary_hash": config["clip"]["vocabulary_hash"],
        "dictionary": config["dictionary"],
        "grouping_rationale": config["grouping_rationale"],
        "per_seed": results,
        "gate": gate,
        "elapsed_seconds": round(elapsed, 1),
    }
    evidence["evidence_hash"] = payload_hash(evidence)

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.json"
    write_canonical_json(evidence_path, evidence)
    build_artifact_index(output_dir)

    print(f"verdict: {gate['verdict']}  sealed: {sealed}")
    print(
        "purity control "
        f"{gate['purity_control']['mean']} (floor {gate['purity_control']['floor']}, "
        f"null {gate['purity_control']['null_mean']}) "
        f"passes {gate['purity_control']['passes']}"
    )
    print(
        f"stability {gate['stability']['agreement_mean']} vs null "
        f"{gate['stability']['null_mean']} margin "
        f"{gate['stability']['margin_mean']} passes {gate['stability']['passes']}"
    )
    for tag, entry in gate["naming_delta"].items():
        print(
            f"{tag}: named {entry['i5_named_mean']} matched-random "
            f"{entry['i5_matched_random_mean']} withheld "
            f"{entry['i5_withheld_mean']} naming delta "
            f"{entry['naming_delta_mean']} spread {entry['naming_delta_spread']}"
        )
    print(f"far-field rate {gate['far_field_rate_mean']}")
    print(f"wrote {evidence_path} in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
