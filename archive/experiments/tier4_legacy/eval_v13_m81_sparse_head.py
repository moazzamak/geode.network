"""M81 — sparse head over the M80 dictionary, and the decisive I5 measurement.

Registered hypothesis H81: a head reading the M80 sparse atoms is substantially
more forward-simulable than a dense head of comparable accuracy.

Registered gate, as amended by R4 (`RESEARCH_IMPLEMENTATION_PLAN_v13.md`
Section 7): the verdict is a **conjunction over two task widths** and neither
width may be reported alone.

* **I5-8** — eight classes, drawn once under seed 8101 and frozen in the
  configuration. Chance 12.5%. The original registered rule is retained
  unchanged: >=40% confirms, 25-40% partial, <=25% refutes.
* **I5-128** — the full corpus width. Chance 0.781%. Must sit strictly above
  the re-measured kNN control on the identical protocol by more than the
  spread across seeds 11/23/37.

The reason both are required is recorded in the plan: the original rule and its
"kNN control level" bound were measured on v12's 8-way CIFAR-10 task and were
about to be applied unchanged to a 128-class corpus, which is precisely the
defect that invalidated L2. The v12 numbers (GEODE 17.737, RBF 22.772, kNN
25.246) are reference points here and never bars; every control is re-measured.

Registration notes carried by this runner:

* **N81.1** — the SHAP arm is GradientExplainer-style expected gradients, not
  KernelSHAP. `shap` is absent from the frozen replay `.venv` and cannot be
  installed without breaking the sealed M73/M77 hashes, and KernelSHAP is not
  tractable at this width. Named as what it is, everywhere.
* **N81.2** — the identity-withheld protocol cannot test whether atoms are
  *nameable*; it tests whether *sparsity* aids simulatability. The nameability
  claim lives at M82. Identity-included I5 is ceiling-artifact-prone for linear
  heads and is not run here.
* **N81.3** — explanations are standardised on the probe-training rows only.
  Unscaled, lbfgs did not converge at max_iter 2000 and I5 measured optimiser
  failure rather than explanation content.
* **N81.4** — the L1 penalty is applied as a proximal soft-threshold. Added to
  the loss and differentiated by Adam it never produced an exact zero: accuracy
  collapsed while atoms cited per decision stayed near k.
* **N81.5** — the hard atom budget ranks candidates by contribution mass, not
  coefficient magnitude. Magnitude alone selects rare atoms, which hold large
  weights because they seldom fire; that head cited 0.6 atoms per decision and
  fell to chance.
* **N81.6** — every arm's explanation, atom heads and dense controls alike, is
  reduced to the identical withheld form of `top_count` sorted magnitudes plus
  sum, max and mean. I5 differences therefore cannot arise from one arm being
  handed a wider explanation vector than another.

Threading contract: single-threaded torch inside parallel worker processes,
carried from M80 where multi-threaded fits were measured non-reproducible.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v13_m81_sparse_head
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
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import NearestNeighbors

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v13_heads import (
    cited_atoms_per_decision,
    expected_gradients,
    fit_decision_list,
    fit_metric_field,
    fit_mlp,
    fit_sparse_linear,
    integrated_gradients,
    withheld_explanation,
)
from experiments.common.v13_i5 import (
    forward_simulation,
    shuffled_explanation_control,
)
from experiments.common.v13_linear_probe import balanced_accuracy
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
    REPO_ROOT / "experiments" / "configs" / "v13" / "m81_sparse_head.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v13" / "m81_sparse_head"

_WORKER_STATE: dict[str, Any] = {}


def _subset(codes: SparseCodes, rows: np.ndarray) -> SparseCodes:
    return SparseCodes(
        codes.indices[rows].copy(), codes.values[rows].copy(), codes.dictionary_size
    )


def _arm(
    name: str,
    *,
    family: str,
    predicted: np.ndarray,
    truth: np.ndarray,
    contributions: np.ndarray,
    class_count: int,
    config: dict[str, Any],
    seed: int,
    active_parameters: int | None,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Score one arm and run its I5 measurement against its own shuffled null."""
    explanations = withheld_explanation(
        contributions, top_count=int(config["explanations"]["top_count"])
    )
    i5_config = config["i5"]
    shared = dict(
        class_count=class_count,
        train_fraction=float(i5_config["train_fraction"]),
        max_iter=int(i5_config["max_iter"]),
        seed=seed,
    )
    simulation = forward_simulation(explanations, predicted, **shared)
    null = shuffled_explanation_control(explanations, predicted, **shared)
    measured = simulation["probe_balanced_accuracy"]
    null_measured = null["probe_balanced_accuracy"]
    margin = (
        None
        if measured is None or null_measured is None
        else float(measured - null_measured)
    )
    return {
        "arm": name,
        "family": family,
        "settings": settings,
        "balanced_accuracy": balanced_accuracy(predicted, truth),
        "active_parameters": active_parameters,
        "explanation_length": cited_atoms_per_decision(
            contributions, budget=int(config["explanations"]["budget"])
        ),
        "i5": simulation,
        "i5_shuffled_null": null,
        "i5_margin_over_null": margin,
        "beats_own_null": bool(margin is not None and margin > 0.0),
    }


def _atom_arms(
    fit_codes: SparseCodes,
    fit_labels: np.ndarray,
    evaluation_codes: SparseCodes,
    evaluation_labels: np.ndarray,
    *,
    class_count: int,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    heads = config["heads"]
    linear = heads["sparse_linear"]
    arms: list[dict[str, Any]] = []

    def add_linear(tag: str, settings: dict[str, Any], **kwargs: Any) -> None:
        head = fit_sparse_linear(
            fit_codes,
            fit_labels,
            class_count=class_count,
            epochs=int(linear["epochs"]),
            batch_size=int(linear["batch_size"]),
            learning_rate=float(linear["learning_rate"]),
            seed=seed,
            **kwargs,
        )
        predicted = head.scores(evaluation_codes).argmax(1)
        arms.append(
            _arm(
                tag,
                family="atoms",
                predicted=predicted,
                truth=evaluation_labels,
                contributions=head.contributions(evaluation_codes, predicted),
                class_count=class_count,
                config=config,
                seed=seed,
                active_parameters=head.active_parameter_count(),
                settings=settings,
            )
        )

    for penalty in linear["l1_penalties"]:
        add_linear(
            f"sparse_linear_l1_{penalty}",
            {"l1_penalty": float(penalty), "atom_budget": None},
            l1_penalty=float(penalty),
        )
    for budget in linear["atom_budgets"]:
        add_linear(
            f"sparse_linear_budget_{budget}",
            {"l1_penalty": 0.0, "atom_budget": int(budget)},
            l1_penalty=0.0,
            atom_budget=int(budget),
        )

    for shrinkage in heads["metric_field"]["variance_shrinkage"]:
        head = fit_metric_field(
            fit_codes,
            fit_labels,
            class_count=class_count,
            variance_shrinkage=float(shrinkage),
        )
        predicted = head.scores(evaluation_codes).argmax(1)
        arms.append(
            _arm(
                f"metric_field_shrinkage_{shrinkage}",
                family="atoms",
                predicted=predicted,
                truth=evaluation_labels,
                contributions=head.contributions(evaluation_codes, predicted),
                class_count=class_count,
                config=config,
                seed=seed,
                active_parameters=head.active_parameter_count(),
                settings={"variance_shrinkage": float(shrinkage)},
            )
        )

    rules = fit_decision_list(
        fit_codes,
        fit_labels,
        class_count=class_count,
        max_rules=int(heads["decision_list"]["max_rules"]),
    )
    predicted = rules.predict(evaluation_codes)
    arms.append(
        _arm(
            "decision_list",
            family="atoms",
            predicted=predicted,
            truth=evaluation_labels,
            contributions=rules.contributions(evaluation_codes),
            class_count=class_count,
            config=config,
            seed=seed,
            active_parameters=rules.active_parameter_count(),
            settings={"rules_kept": int(len(rules.atoms))},
        )
    )
    return arms


def _control_arms(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
    *,
    class_count: int,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    controls = config["controls"]
    arms: list[dict[str, Any]] = []

    # kNN. The explanation a nearest-neighbour model offers is the profile of
    # distances to the neighbours it voted with, identity withheld.
    neighbours = int(controls["knn"]["neighbors"])
    index = NearestNeighbors(n_neighbors=neighbours, n_jobs=1).fit(fit_features)
    distances, found = index.kneighbors(evaluation_features)
    votes = fit_labels[found]
    predicted = np.array(
        [np.bincount(row, minlength=class_count).argmax() for row in votes],
        dtype=np.int64,
    )
    agreeing = (votes == predicted[:, None]).astype(np.float32)
    arms.append(
        _arm(
            "knn",
            family="dense_control",
            predicted=predicted,
            truth=evaluation_labels,
            contributions=(-distances.astype(np.float32)) * agreeing,
            class_count=class_count,
            config=config,
            seed=seed,
            active_parameters=int(fit_features.size),
            settings={"neighbors": neighbours},
        )
    )

    # RBF. Nystroem feature map plus a linear head; see the module docstring for
    # why SVC was not tractable at this width.
    gamma = controls["rbf"]["gamma"]
    if gamma is None:
        gamma = 1.0 / (fit_features.shape[1] * float(fit_features.var()))
    mapping = Nystroem(
        gamma=float(gamma),
        n_components=int(controls["rbf"]["nystroem_components"]),
        random_state=seed,
    ).fit(fit_features)
    fit_mapped = mapping.transform(fit_features).astype(np.float32)
    evaluation_mapped = mapping.transform(evaluation_features).astype(np.float32)
    mlp_config = controls["mlp"]
    rbf_head = fit_mlp(
        fit_mapped,
        fit_labels,
        class_count=class_count,
        hidden=0,
        epochs=int(mlp_config["epochs"]),
        batch_size=int(mlp_config["batch_size"]),
        learning_rate=float(mlp_config["learning_rate"]),
        seed=seed,
    )
    with torch.no_grad():
        weight = rbf_head.stack[0].weight
        scores = rbf_head(torch.from_numpy(evaluation_mapped))
        rbf_predicted = scores.argmax(1).numpy()
        rbf_contributions = (
            weight[torch.from_numpy(rbf_predicted)]
            * torch.from_numpy(evaluation_mapped)
        ).numpy()
    arms.append(
        _arm(
            "rbf_nystroem",
            family="dense_control",
            predicted=rbf_predicted,
            truth=evaluation_labels,
            contributions=rbf_contributions,
            class_count=class_count,
            config=config,
            seed=seed,
            active_parameters=int(weight.numel()),
            settings={
                "nystroem_components": int(controls["rbf"]["nystroem_components"]),
                "gamma": float(gamma),
            },
        )
    )

    # MLP with two post-hoc attribution methods.
    model = fit_mlp(
        fit_features,
        fit_labels,
        class_count=class_count,
        hidden=int(mlp_config["hidden"]),
        epochs=int(mlp_config["epochs"]),
        batch_size=int(mlp_config["batch_size"]),
        learning_rate=float(mlp_config["learning_rate"]),
        seed=seed,
    )
    with torch.no_grad():
        mlp_predicted = (
            model(torch.from_numpy(evaluation_features)).argmax(1).numpy()
        )
    parameters = int(sum(p.numel() for p in model.parameters()))
    explanation_config = config["explanations"]
    arms.append(
        _arm(
            "mlp_integrated_gradients",
            family="dense_control",
            predicted=mlp_predicted,
            truth=evaluation_labels,
            contributions=integrated_gradients(
                model,
                evaluation_features,
                mlp_predicted,
                baseline=fit_features.mean(axis=0),
                steps=int(explanation_config["integrated_gradients"]["steps"]),
            ),
            class_count=class_count,
            config=config,
            seed=seed,
            active_parameters=parameters,
            settings={"steps": int(explanation_config["integrated_gradients"]["steps"])},
        )
    )
    expected_config = explanation_config["expected_gradients"]
    reference_rows = np.random.default_rng(seed).choice(
        len(fit_features), size=min(512, len(fit_features)), replace=False
    )
    arms.append(
        _arm(
            "mlp_expected_gradients",
            family="dense_control",
            predicted=mlp_predicted,
            truth=evaluation_labels,
            contributions=expected_gradients(
                model,
                evaluation_features,
                mlp_predicted,
                reference=fit_features[reference_rows],
                samples=int(expected_config["samples"]),
                seed=int(expected_config["seed"]),
            ),
            class_count=class_count,
            config=config,
            seed=seed,
            active_parameters=parameters,
            settings={
                "samples": int(expected_config["samples"]),
                "estimator": "expected_gradients",
                "not_kernel_shap": True,
            },
        )
    )
    return arms


def _width_result(
    *,
    width: str,
    class_count: int,
    fit_codes: SparseCodes,
    evaluation_codes: SparseCodes,
    fit_features: np.ndarray,
    evaluation_features: np.ndarray,
    fit_labels: np.ndarray,
    evaluation_labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    arms = _atom_arms(
        fit_codes,
        fit_labels,
        evaluation_codes,
        evaluation_labels,
        class_count=class_count,
        config=config,
        seed=seed,
    ) + _control_arms(
        fit_features,
        fit_labels,
        evaluation_features,
        evaluation_labels,
        class_count=class_count,
        config=config,
        seed=seed,
    )
    return {
        "width": width,
        "class_count": class_count,
        "chance_accuracy": 1.0 / class_count,
        "fit_rows": int(len(fit_labels)),
        "evaluation_rows": int(len(evaluation_labels)),
        "arms": arms,
    }


def _worker_initializer(index_path: str, config_text: str) -> None:
    torch.use_deterministic_algorithms(True)
    config = json.loads(config_text)
    torch.set_num_threads(int(config["threading"]["torch_threads_per_worker"]))
    features, labels = _load_corpus(Path(index_path))
    partition = config["partition"]
    fit_rows, evaluation_rows = _partition(
        labels,
        fit_per_class=int(partition["fit_per_class"]),
        evaluation_per_class=int(partition["evaluation_per_class"]),
    )
    _WORKER_STATE["config"] = config
    _WORKER_STATE["fit_features"] = features[fit_rows]
    _WORKER_STATE["fit_labels"] = labels[fit_rows]
    _WORKER_STATE["evaluation_features"] = features[evaluation_rows]
    _WORKER_STATE["evaluation_labels"] = labels[evaluation_rows]


def _worker_run(seed: int) -> dict[str, Any]:
    config = _WORKER_STATE["config"]
    fit_features = _WORKER_STATE["fit_features"]
    fit_labels = _WORKER_STATE["fit_labels"]
    evaluation_features = _WORKER_STATE["evaluation_features"]
    evaluation_labels = _WORKER_STATE["evaluation_labels"]
    started = time.time()

    specification = config["dictionary"]
    dictionary, diagnostics = fit_sparse_dictionary(
        fit_features,
        dictionary_size=int(specification["dictionary_size"]),
        active_atoms=int(specification["active_atoms"]),
        epochs=int(specification["epochs"]),
        batch_size=int(specification["batch_size"]),
        learning_rate=float(specification["learning_rate"]),
        seed=seed,
    )
    fit_codes = dictionary.codes(fit_features)
    evaluation_codes = dictionary.codes(evaluation_features)
    class_count = int(len(np.unique(fit_labels)))

    widths = [
        _width_result(
            width="i5_128",
            class_count=class_count,
            fit_codes=fit_codes,
            evaluation_codes=evaluation_codes,
            fit_features=fit_features,
            evaluation_features=evaluation_features,
            fit_labels=fit_labels,
            evaluation_labels=evaluation_labels,
            config=config,
            seed=seed,
        )
    ]

    # I5-8 reuses the registered 128-class basis: only the heads are retrained.
    # Refitting the dictionary on eight classes would answer a different
    # question, since the basis under test is the one M80 gated.
    chosen = np.asarray(config["i5_eight"]["classes"], dtype=np.int64)
    remap = {label: position for position, label in enumerate(chosen)}
    fit_mask = np.isin(fit_labels, chosen)
    evaluation_mask = np.isin(evaluation_labels, chosen)
    widths.append(
        _width_result(
            width="i5_8",
            class_count=len(chosen),
            fit_codes=_subset(fit_codes, np.flatnonzero(fit_mask)),
            evaluation_codes=_subset(
                evaluation_codes, np.flatnonzero(evaluation_mask)
            ),
            fit_features=fit_features[fit_mask],
            evaluation_features=evaluation_features[evaluation_mask],
            fit_labels=np.array(
                [remap[int(label)] for label in fit_labels[fit_mask]], dtype=np.int64
            ),
            evaluation_labels=np.array(
                [remap[int(label)] for label in evaluation_labels[evaluation_mask]],
                dtype=np.int64,
            ),
            config=config,
            seed=seed,
        )
    )

    result = {
        "seed": seed,
        "dictionary": {
            "dictionary_size": int(specification["dictionary_size"]),
            "active_atoms": int(specification["active_atoms"]),
            "final_train_loss": diagnostics["final_train_loss"],
            "loss_decreased": diagnostics["loss_decreased"],
            "dead_atom_fraction": 1.0
            - fit_codes.active_atom_count() / fit_codes.dictionary_size,
        },
        "widths": widths,
        "elapsed_seconds": time.time() - started,
    }
    result["state_hash"] = payload_hash(
        {"seed": seed, "widths": widths, "dictionary": result["dictionary"]}
    )
    return result


def _i5_of(arm: dict[str, Any]) -> float | None:
    return arm["i5"]["probe_balanced_accuracy"]


def _mean_or_none(values: list[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return float(np.mean(measured)) if measured else None


def _is_degenerate(arm: dict[str, Any], *, class_count: int, floor: float) -> bool:
    """Screen for prediction collapse -- note N81.8.

    A head that emits only a fraction of the label space is easy to simulate
    for a reason that has nothing to do with its explanation: the probe can
    read the skewed marginal instead. The first M81 run was gated by a decision
    list emitting 36 of 128 classes, whose majority baseline was 2.77% against
    every other arm's 0.78%, and whose margin over that baseline was lower than
    arms it appeared to beat. `degenerate_single_prediction` only catches the
    limiting case of one class, which is not enough.
    """
    distinct = arm["i5"].get("distinct_predictions")
    return distinct is None or distinct < floor * class_count


def _comparable_accuracy_floor(
    width: dict[str, Any], *, tolerance_points: float
) -> float:
    """H81 compares against a dense head *of comparable accuracy*. That clause
    is part of the registered hypothesis, so an atom arm far below the best
    dense control cannot carry the result -- note N81.7.
    """
    dense = [
        arm["balanced_accuracy"]
        for arm in width["arms"]
        if arm["family"] == "dense_control"
    ]
    return max(dense) - tolerance_points / 100.0


def _best_atom_arm(
    width: dict[str, Any],
    *,
    respect_budget: bool = True,
    tolerance_points: float,
    degeneracy_floor: float,
) -> dict[str, Any] | None:
    """The best atom arm that is accuracy-comparable, non-degenerate, beats its
    own null, and clears the 10-atom deployment budget.

    Returns `None` when no arm satisfies the conditions, which is itself a
    result: it means the sparse basis cannot meet the registered claim at this
    width rather than that it met it weakly.
    """
    floor = _comparable_accuracy_floor(width, tolerance_points=tolerance_points)
    admissible = [
        arm
        for arm in width["arms"]
        if arm["family"] == "atoms"
        and arm["beats_own_null"]
        and arm["balanced_accuracy"] >= floor
        and not _is_degenerate(
            arm, class_count=width["class_count"], floor=degeneracy_floor
        )
    ]
    if respect_budget:
        admissible = [
            arm
            for arm in admissible
            if arm["explanation_length"]["mean_active_atoms"]
            <= arm["explanation_length"]["budget"]
        ]
    if not admissible:
        return None
    return max(admissible, key=lambda arm: _i5_of(arm) or -1.0)


def _control_level(width: dict[str, Any], name: str) -> float | None:
    for arm in width["arms"]:
        if arm["arm"] == name:
            return _i5_of(arm)
    return None


def _best_dense_control(width: dict[str, Any]) -> dict[str, Any]:
    """The strongest dense control by I5. H81 is a comparative claim, so the
    atom arms have to beat the best explanation any dense model offers, not
    only the one control the gate happens to name.
    """
    dense = [arm for arm in width["arms"] if arm["family"] == "dense_control"]
    return max(dense, key=lambda arm: _i5_of(arm) or -1.0)


def _build_gate(seeds: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    gate_config = config["gate"]

    tolerance = float(gate_config["accuracy_comparability_points"])
    degeneracy_floor = float(gate_config["degeneracy_floor"])

    def collect(width_name: str, *, points: float) -> dict[str, Any]:
        widths = [
            width
            for seed in seeds
            for width in seed["widths"]
            if width["width"] == width_name
        ]
        selection = dict(
            tolerance_points=points, degeneracy_floor=degeneracy_floor
        )
        best = [_best_atom_arm(width, **selection) for width in widths]
        unconstrained = [
            _best_atom_arm(width, respect_budget=False, **selection)
            for width in widths
        ]
        dense = [_best_dense_control(width) for width in widths]
        scores = [_i5_of(arm) if arm else None for arm in best]
        knn = [_control_level(width, "knn") for width in widths]
        nulls = [
            arm["i5_shuffled_null"]["probe_balanced_accuracy"] if arm else None
            for arm in best
        ]
        measured = [score for score in scores if score is not None]
        return {
            "accuracy_comparability_points": points,
            "comparable_accuracy_floor_per_seed": [
                _comparable_accuracy_floor(width, tolerance_points=points)
                for width in widths
            ],
            "best_atom_arm_per_seed": [
                arm["arm"] if arm else None for arm in best
            ],
            "seeds_with_no_admissible_atom_arm": sum(
                1 for arm in best if arm is None
            ),
            "i5_per_seed": scores,
            "i5_mean": _mean_or_none(scores),
            "i5_spread": (
                float(max(measured) - min(measured)) if measured else None
            ),
            "i5_per_seed_meets_eight_way_bar": [
                None
                if score is None
                else bool(score >= float(gate_config["i5_eight"]["confirms_at_or_above"]))
                for score in scores
            ],
            "knn_control_per_seed": knn,
            "knn_control_mean": _mean_or_none(knn),
            "best_dense_control_per_seed": [arm["arm"] for arm in dense],
            "best_dense_control_i5_mean": _mean_or_none(
                [_i5_of(arm) for arm in dense]
            ),
            "shuffled_null_mean": _mean_or_none(nulls),
            "seeds_measurable": len(measured),
            "unconstrained_best_arm_per_seed": [
                arm["arm"] if arm else None for arm in unconstrained
            ],
            "unconstrained_i5_mean": _mean_or_none(
                [_i5_of(arm) if arm else None for arm in unconstrained]
            ),
            "unconstrained_cited_atoms_mean": _mean_or_none(
                [
                    arm["explanation_length"]["mean_active_atoms"]
                    if arm
                    else None
                    for arm in unconstrained
                ]
            ),
            "within_explanation_budget": [
                bool(
                    arm["explanation_length"]["mean_active_atoms"]
                    <= arm["explanation_length"]["budget"]
                )
                if arm
                else False
                for arm in best
            ],
        }

    eight = collect("i5_8", points=tolerance)
    full = collect("i5_128", points=tolerance)
    # The comparability tolerance was fixed after the first run exposed the
    # degenerate arm, so the verdict is also reported across a range of
    # tolerances. A conclusion that survives all of them is not an artefact of
    # where the threshold was placed; one that does not must not be claimed.
    sensitivity = {
        f"{points:g}_points": {
            "i5_8": collect("i5_8", points=points),
            "i5_128": collect("i5_128", points=points),
        }
        for points in gate_config["accuracy_comparability_sensitivity_points"]
    }

    eight_config = gate_config["i5_eight"]
    confirms = float(eight_config["confirms_at_or_above"])
    refutes = float(eight_config["refutes_at_or_below"])
    if eight["i5_mean"] is None:
        eight_verdict = "unmeasurable"
    elif eight["i5_mean"] >= confirms:
        eight_verdict = "confirms"
    elif eight["i5_mean"] <= refutes:
        eight_verdict = "refutes"
    else:
        eight_verdict = "partial"

    full_passes = bool(
        full["i5_mean"] is not None
        and full["knn_control_mean"] is not None
        and full["i5_mean"] > full["knn_control_mean"] + (full["i5_spread"] or 0.0)
    )
    # H81 is comparative, so beating the one named control is necessary but not
    # sufficient. This is reported separately from the registered rule rather
    # than folded into it.
    full_beats_best_dense = bool(
        full["i5_mean"] is not None
        and full["best_dense_control_i5_mean"] is not None
        and full["i5_mean"] > full["best_dense_control_i5_mean"]
    )
    eight_beats_best_dense = bool(
        eight["i5_mean"] is not None
        and eight["best_dense_control_i5_mean"] is not None
        and eight["i5_mean"] > eight["best_dense_control_i5_mean"]
    )

    if eight_verdict == "unmeasurable":
        verdict = "unmeasurable"
    elif eight_verdict == "refutes":
        verdict = "refuted_outcome_h"
    elif eight_verdict == "confirms" and full_passes:
        verdict = "confirmed"
    elif eight_verdict == "confirms" and not full_passes:
        verdict = "task_width_artifact"
    elif full_passes:
        verdict = "partial"
    else:
        verdict = "partial_and_width_limited"

    nulls_collapse = all(
        arm["beats_own_null"]
        for seed in seeds
        for width in seed["widths"]
        for arm in width["arms"]
        if arm["family"] == "atoms"
    )
    eight_seed_agreement = [
        flag for flag in eight["i5_per_seed_meets_eight_way_bar"] if flag is not None
    ]

    return {
        "i5_8": eight
        | {
            "verdict": eight_verdict,
            "chance": 0.125,
            "beats_best_dense_control": eight_beats_best_dense,
        },
        "i5_128": full
        | {
            "passes": full_passes,
            "chance": 1.0 / 128,
            "beats_best_dense_control": full_beats_best_dense,
        },
        "accuracy_comparability_sensitivity": sensitivity,
        "conjunction_verdict": verdict,
        "h81_gate_passed": verdict == "confirmed",
        "dominance_claim_blocked": verdict != "confirmed",
        "all_atom_arms_beat_own_null": nulls_collapse,
        "atoms_beat_best_dense_control_at_both_widths": bool(
            eight_beats_best_dense and full_beats_best_dense
        ),
        "eight_way_seeds_meeting_bar": sum(1 for flag in eight_seed_agreement if flag),
        "eight_way_seeds_measured": len(eight_seed_agreement),
        "eight_way_margin_exceeds_seed_spread": bool(
            eight["i5_mean"] is not None
            and eight["i5_spread"] is not None
            and (eight["i5_mean"] - confirms) > eight["i5_spread"]
        ),
        "v12_reference_points_not_used_as_bars": True,
        "neither_width_reportable_alone": True,
        "final_labels_opened": False,
    }


def run_m81(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)

    index_path = _verify_corpus(
        {
            "path": f"{config['corpus']['root']}/artifact_index.json",
            "sha256": config["corpus"]["index_sha256"],
        }
    )
    features, labels = _load_corpus(index_path)

    seeds = [int(seed) for seed in config["seeds"]]
    worker_count = workers or min(len(seeds), int(config["threading"]["workers"]))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_worker_initializer,
        initargs=(str(index_path), config_text),
    ) as pool:
        results = list(pool.map(_worker_run, seeds))

    evidence = {
        "schema_version": 1,
        "milestone": "M81",
        "program": "v13",
        "registered_hypothesis": "H81",
        "registration_notes": [
            "N81.1", "N81.2", "N81.3", "N81.4", "N81.5", "N81.6"
        ],
        "configuration_hash": sha256_file(config_path),
        "corpus": {
            "name": "v13 DomainNet large",
            "index_sha256": config["corpus"]["index_sha256"],
            "rows": int(len(features)),
            "dimension": int(features.shape[1]),
            "classes": int(len(np.unique(labels))),
        },
        "i5_eight_classes": config["i5_eight"]["classes"],
        "worker_count": int(worker_count),
        "seeds": results,
        "gate": _build_gate(results, config),
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def regate_m81(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Recompute the gate from stored per-arm results, without refitting.

    Every quantity the gate reads is already recorded per arm, so this is exact
    rather than an approximation of a rerun. It exists so that a correction to
    the *selection rule* does not require twenty-seven minutes of identical
    model fitting, and so that the per-seed `state_hash` values stay untouched
    and can be checked against the original run.
    """
    output_dir = Path(output_dir)
    evidence_path = output_dir / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    evidence["gate"] = _build_gate(evidence["seeds"], config)
    evidence["configuration_hash"] = sha256_file(Path(config_path))
    evidence["gate_recomputed_without_refit"] = True
    write_canonical_json(evidence_path, evidence)
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--regate",
        action="store_true",
        help="recompute the gate from stored evidence without refitting",
    )
    arguments = parser.parse_args()
    if arguments.regate:
        evidence = regate_m81(arguments.config, arguments.output)
    else:
        evidence = run_m81(
            arguments.config, arguments.output, workers=arguments.workers
        )

    for width_name in ("i5_8", "i5_128"):
        print(f"\n=== {width_name} ===")
        header = (
            f"{'arm':>34} {'acc':>8} {'I5':>8} {'null':>8} "
            f"{'margin':>8} {'cited':>7} {'<=bud':>6}"
        )
        print(header)
        print("-" * len(header))
        first = evidence["seeds"][0]
        width = next(w for w in first["widths"] if w["width"] == width_name)
        for arm in width["arms"]:
            length = arm["explanation_length"]
            score = _i5_of(arm)
            null = arm["i5_shuffled_null"]["probe_balanced_accuracy"]
            margin = arm["i5_margin_over_null"]
            print(
                f"{arm['arm']:>34} {arm['balanced_accuracy'] * 100:>7.2f}% "
                f"{'  n/a  ' if score is None else format(score * 100, '>7.2f') + '%'} "
                f"{'  n/a  ' if null is None else format(null * 100, '>7.2f') + '%'} "
                f"{'  n/a ' if margin is None else format(margin * 100, '>+7.2f')} "
                f"{length['mean_active_atoms']:>7.1f} "
                f"{length['fraction_of_decisions_within_budget']:>6.2f}"
            )
    print()
    print(json.dumps(evidence["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
