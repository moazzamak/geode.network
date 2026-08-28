"""Absolute-scale boundary supervision for M83.

The defect this corrects
------------------------
v12 placed every synthetic negative at a fixed multiple of the class's *own*
fitted extent — ``center + 4.0 * tangent_scale[axis] * basis[:, axis]``. Under
the model's own Mahalanobis score that probe evaluates to exactly ``4.0`` for
every class, at every extent, forever:

    score = sqrt( (4 sigma)^2 / sigma^2 ) = 4

The supervision was therefore algebraically independent of the quantity it
claimed to supervise, and M77 measured its gradient at zero to within 1e-12.
Eight programs' worth of open-set negatives were collected under a term that
could not move.

M83 places negatives at ``center + multiplier * unit * direction`` where
``unit`` is a **global** feature-space length — the median inter-class centroid
distance — held constant. The same probe now evaluates to
``multiplier * unit / sigma``, which is a function of the extent, so the
gradient exists. That difference is the whole hypothesis, and the degeneracy
report below is the instrument that reads it.

What "the boundary" is
----------------------
Phase A fits per-class geometry in closed form and freezes it (N83.4). The
boundary is a *separate* ellipsoid over that frozen geometry, parameterised by
its own per-axis log radii ``log_beta`` of shape ``(classes, rank + 1)``, and a
point is accepted by class k when its boundary score is at most one. Two
consequences matter for the measurement:

* The boundary has shape, not just size. A scalar threshold would be fully
  determined by any matched-coverage requirement, which would make the
  real-OOD operand vacuous — every arm would be forced to the same number.
  Only the **shape** of ``log_beta`` survives coverage matching, so the shape
  displacement is the operand that can honestly differ between arms.
* ``log_beta`` is literally a set of scale parameters, so the plan's
  degeneracy contract — "the boundary loss has non-zero gradient with respect
  to the scale parameters" — is a statement about the trained parameters
  rather than an analogy.

The degeneracy report evaluates the **probe term alone**, never the total loss.
The known-class term takes real data points, whose scores depend on the radii
under any placement rule, so a total-loss gradient would be non-zero even for
the v12 form and the test would pass everything put to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from src.subspace_primitive import fit_subspace_primitive

#: Probe families whose displacement is a multiple of the fitted extent along
#: the direction they displace. For these the own-class score is algebraically
#: independent of that extent — this is v12's form, retained as the negative
#: end of M83's instrument.
RELATIVE_FAMILIES = ("axis_tangent", "normal", "masking")

#: Families reserved for the held-out generalisation operand. A boundary that
#: has learned a surface rather than the training probes must still reject
#: these.
HELD_OUT_FAMILIES = ("masking", "random_direction")

_EPSILON = 1e-12
#: Floor for the argument of a square root. Clamped rather than added, because
#: an additive epsilon perturbs the v12-form arm's exact cancellation and leaves
#: a spurious ~1e-13 gradient where the contract demands zero — the instrument
#: would then be measuring its own guard term.
_SQRT_FLOOR = 1e-300


def _repair_allocation(
    allocation: np.ndarray,
    availability: np.ndarray,
    slots: np.ndarray,
    *,
    domain: int,
    need: int,
) -> int:
    """Free a blocked slot by moving one already-placed row to another class.

    A greedy fill stalls in one specific way: some class still holds rows of
    the domain we are short of, but has no free slot, because an earlier domain
    took them all — while some other class has a free slot it cannot use. The
    fix is a single exchange. Hand one of the blocked class's earlier rows to
    the class with the spare slot, and spend the slot that frees on the domain
    that is short. Nothing else moves, and no domain total changes except the
    one being filled.
    """
    while need > 0:
        moved = False
        blocked = np.flatnonzero(
            (availability[:, domain] > allocation[:, domain]) & (slots <= 0)
        )
        for source in blocked:
            for other in np.flatnonzero(allocation[source] > 0):
                if other == domain:
                    continue
                receivers = np.flatnonzero(
                    (slots > 0) & (availability[:, other] > allocation[:, other])
                )
                if len(receivers) == 0:
                    continue
                target = int(receivers[0])
                allocation[source, other] -= 1
                allocation[target, other] += 1
                slots[source] += 1
                slots[target] -= 1
                allocation[source, domain] += 1
                slots[source] -= 1
                need -= 1
                moved = True
                break
            if moved or need == 0:
                break
        if not moved:
            break
    return need


def domain_matched_partition(
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    quota: Sequence[int],
    fit_per_class: int,
    domain_count: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Split the corpus so the known evaluation rows match the out-of-set mixture.

    N83.2 requires the known control and the unseen rows to be drawn from the
    same domain profile, or the rejection comparison measures which domains the
    two sets happen to contain. The obvious class-stratified split fails that
    test badly here: the corpus is class-major and every class stores its 350
    quickdraw rows first, so taking the tail of each class as held-out yields an
    evaluation set with *no* quickdraw at all against an out-of-set that is 61
    percent quickdraw.

    So the evaluation rows are allocated by domain instead of by position.
    Domains that every class can supply are pinned at their quota; the scarce
    remainder is water-filled across classes in order of ascending aggregate
    supply, which lets classes holding many paintings cover the classes holding
    none. What is protected is the *aggregate* profile, not the per-class one:
    a class with no paintings ends up carrying an extra row of something else,
    and another class carries its paintings. That trade is the right way round,
    because N83.2 is a statement about the mixture the two evaluation sets are
    drawn from, and nothing in the rejection measurement is read per class.

    Returns the fit rows, the evaluation rows, and a report of what was achieved
    so the deviation is evidence rather than an assumption.
    """
    quota = tuple(int(value) for value in quota)
    if len(quota) != domain_count:
        raise ValueError("M83 partition quota must cover every domain")
    classes = np.unique(labels)
    class_rows = {int(label): np.flatnonzero(labels == label) for label in classes}

    availability = np.zeros((len(classes), domain_count), dtype=np.int64)
    for index, label in enumerate(classes):
        owned = domains[class_rows[int(label)]]
        for domain in range(domain_count):
            availability[index, domain] = int(np.count_nonzero(owned == domain))

    allocation = np.zeros_like(availability)
    slots = np.full(len(classes), sum(quota), dtype=np.int64)
    scarce: list[int] = []
    for domain in range(domain_count):
        if quota[domain] == 0:
            continue
        if quota[domain] <= int(availability[:, domain].min()):
            allocation[:, domain] = quota[domain]
            slots -= quota[domain]
        else:
            scarce.append(domain)

    unmet: dict[str, int] = {}
    order = sorted(scarce, key=lambda d: int(availability[:, d].sum()))
    for domain in order:
        need = quota[domain] * len(classes)
        while need > 0:
            room = np.minimum(availability[:, domain] - allocation[:, domain], slots)
            if int(room.max()) <= 0:
                break
            # One row at a time to the class with the most room. Handing a
            # class everything it can hold starves the classes that come after
            # it, and the slot budget is always exactly tight: each class has
            # as many free slots as the scarce quotas it still owes, so every
            # row given away in the wrong place is a row missing at the end.
            index = int(np.argmax(room))
            allocation[index, domain] += 1
            slots[index] -= 1
            need -= 1
        need = _repair_allocation(
            allocation, availability, slots, domain=domain, need=need
        )
        if need:
            unmet[str(domain)] = int(need)

    fit_rows: list[np.ndarray] = []
    evaluation_rows: list[np.ndarray] = []
    for index, label in enumerate(classes):
        rows = class_rows[int(label)]
        owned = domains[rows]
        taken: list[np.ndarray] = []
        for domain in range(domain_count):
            if allocation[index, domain] == 0:
                continue
            taken.append(rows[owned == domain][: allocation[index, domain]])
        held_out = np.concatenate(taken) if taken else np.empty(0, dtype=np.int64)
        remainder = rows[~np.isin(rows, held_out)]
        if len(remainder) < fit_per_class:
            raise ValueError(
                f"M83 partition leaves class {int(label)} short of fit rows"
            )
        fit_rows.append(remainder[:fit_per_class])
        evaluation_rows.append(np.sort(held_out))

    evaluation = np.concatenate(evaluation_rows)
    achieved = allocation.sum(axis=0)
    profile = achieved / max(int(achieved.sum()), 1)
    requested = np.asarray(quota, dtype=np.float64) / max(sum(quota), 1)
    report = {
        "requested_quota": list(quota),
        "achieved_counts": [int(value) for value in achieved],
        "achieved_profile": [float(value) for value in profile],
        "requested_profile": [float(value) for value in requested],
        "maximum_profile_deviation": float(np.max(np.abs(profile - requested))),
        "unmet_by_domain": unmet,
        "evaluation_rows_per_class": {
            str(int(label)): int(allocation[index].sum())
            for index, label in enumerate(classes)
        },
    }
    return np.concatenate(fit_rows), evaluation, report


def domain_stratified_halves(
    labels: np.ndarray,
    domains: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Halve held-out rows into calibration and report, preserving both mixtures.

    Splitting by position would hand one half the domains that happen to sort
    first, which would undo the matching the partition just bought. The split
    is taken inside every class-and-domain group instead, so both halves carry
    the same profile as the whole.

    Odd groups alternate which half receives the spare row, and the alternation
    is tracked per domain. A single global toggle is not enough: four of the
    five domains hold an odd count in every class, so the parity returns to
    where it started at each class boundary and the first domain wins the spare
    row every single time. Toggling within a domain instead makes the classes
    alternate against each other, and with an even class count both halves come
    out identical in profile as well as in size.
    """
    calibration: list[np.ndarray] = []
    report: list[np.ndarray] = []
    favour_calibration: dict[int, bool] = {}
    for label in np.unique(labels[rows]):
        owned = rows[labels[rows] == label]
        for domain in np.unique(domains[owned]):
            group = owned[domains[owned] == domain]
            cut = len(group) // 2
            if len(group) % 2:
                key = int(domain)
                if favour_calibration.setdefault(key, True):
                    cut += 1
                favour_calibration[key] = not favour_calibration[key]
            calibration.append(group[:cut])
            report.append(group[cut:])
    return (
        np.sort(np.concatenate(calibration)),
        np.sort(np.concatenate(report)),
    )


@dataclass(frozen=True)
class Geometry:
    """Frozen Phase A geometry: one bounded affine subspace per class."""

    centers: np.ndarray
    bases: np.ndarray
    tangent_scales: np.ndarray
    residual_scales: np.ndarray

    @property
    def class_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.centers.shape[1])

    @property
    def rank(self) -> int:
        return int(self.bases.shape[2])


def fit_geometry(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
    class_count: int,
) -> Geometry:
    """Phase A. Closed-form per-class fit, exactly converged (N83.4).

    No optimizer, no schedule, no stopping rule — so none of them can appear
    in the confound set when Phase B is interpreted.
    """
    centers = np.zeros((class_count, features.shape[1]), dtype=np.float64)
    bases = np.zeros((class_count, features.shape[1], rank), dtype=np.float64)
    tangent = np.zeros((class_count, rank), dtype=np.float64)
    residual = np.zeros(class_count, dtype=np.float64)
    for label in range(class_count):
        rows = np.flatnonzero(labels == label)
        if len(rows) < rank + 2:
            raise ValueError(
                f"class {label} has {len(rows)} rows; rank {rank} needs {rank + 2}"
            )
        primitive = fit_subspace_primitive(features[rows], rank, class_label=label)
        centers[label] = primitive.center
        bases[label] = primitive.basis
        tangent[label] = np.sqrt(primitive.tangent_variances)
        residual[label] = float(np.sqrt(primitive.residual_variance))
    return Geometry(centers, bases, tangent, residual)


def absolute_unit(centers: np.ndarray) -> float:
    """The global length scale: median distance between class centroids.

    Global by construction. It does not depend on any class's fitted extent,
    which is precisely why supervision expressed in it cannot cancel against
    that extent.
    """
    if centers.shape[0] < 2:
        raise ValueError("an absolute unit needs at least two classes")
    differences = centers[:, None, :] - centers[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    upper = distances[np.triu_indices(centers.shape[0], k=1)]
    return float(np.median(upper))


def global_scale_unit(geometry: Geometry) -> float:
    """The length probes are actually placed in: one radius for all classes.

    The centroid unit above is the cleaner reference but a poor ruler for
    placement. Class boundaries sit at a few percent of the inter-class
    distance, so a probe one centroid-unit out is so far outside that every
    boundary already rejects it, the hinge is saturated, and a healthy
    objective contributes no gradient at all — it would look as inert as v12's.
    The informative band is where boundaries live.

    This is a **single scalar shared by every class**, which is the property
    that matters. v12's defect was that the length in the displacement was the
    same per-class quantity as the length in the score's denominator, so the
    two cancelled. A global median cannot encode any individual class's extent
    and therefore cannot cancel against it, while still landing probes in the
    band where the supervision does work. The geometry is frozen for the whole
    of Phase B, so this is a constant, not a parameter.
    """
    return float(np.median(geometry.tangent_scales))


def data_scale_unit(
    features: np.ndarray, labels: np.ndarray, geometry: Geometry
) -> float:
    """The length probes must actually be placed in: the data's own extent.

    N83.8. ``global_scale_unit`` above is wrong, and M83.1's evidence is void
    because of it. Its reasoning — that the centroid unit places probes so far
    out that the hinge saturates — is correct. Its conclusion does not follow.
    The median fitted tangent scale is not "where boundaries live"; it is the
    per-direction spread inside a rank-51 subspace of a 384-dimensional space,
    and the distance from a row to its own centroid is dominated by the 333
    residual dimensions it does not describe. On the v13 corpus the tangent
    median is 2.87 while rows sit at a median 30.19 from their own centroid,
    with the closest decile at 24.36. A ladder of multipliers 1--3 in that unit
    put every probe between 2.87 and 8.61 — **inside** the known cloud, by a
    factor of three even at its farthest reach.

    That is not a hard supervision problem, it is an incoherent one. No
    boundary can accept rows at distance 24--30 and reject points at 5--8 along
    the same rays. Every arm of M83.1, including the untrained one, rejected
    exactly zero held-out probes, which is the signature of the defect rather
    than of anything the arms learned.

    The requirement the unit has to meet is only that it not depend on the
    learnable radii, so that v12's cancellation cannot reappear. The extent of
    the frozen fit data satisfies that as completely as the tangent median did,
    while landing the ladder where a negative is actually negative: at the
    median own-centroid distance, probes straddle the cloud's edge, and the
    upper multipliers sit clearly outside it. The geometry and the fit rows are
    both frozen for the whole of Phase B, so this is a constant.
    """
    if len(features) == 0:
        raise ValueError("a data scale unit needs at least one row")
    deltas = features - geometry.centers[labels]
    return float(np.median(np.linalg.norm(deltas, axis=1)))


def probe_validity(
    features: np.ndarray,
    labels: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
    spec: "ProbeSpec",
    *,
    unit: float,
) -> dict[str, object]:
    """Is the probe ladder capable of being a negative at all?

    N83.8. M83.1 spent a sealed run measuring how well four boundaries rejected
    points that lay inside the data those boundaries had to accept. Nothing in
    the gate could see it: every arm agreed, the degeneracy instrument was
    working, and the verdict looked like a clean negative. The check that would
    have caught it costs two medians and no training.

    A probe ladder is admissible only if its outer reach leaves the known
    cloud. The comparison is made in raw distance to the owning centroid, which
    involves no radius, no arm and no outcome, so applying it cannot be a
    rescue of a result that was disliked.
    """
    points = probe_points_numpy(
        geometry, log_beta, spec, placement="absolute", unit=unit
    )
    probe_distance = np.linalg.norm(points - geometry.centers[spec.owners], axis=1)
    known_distance = np.linalg.norm(features - geometry.centers[labels], axis=1)
    known_floor = float(np.percentile(known_distance, 10))
    maximum_probe = float(np.max(probe_distance))
    return {
        "probe_distance_median": float(np.median(probe_distance)),
        "probe_distance_maximum": maximum_probe,
        "known_distance_median": float(np.median(known_distance)),
        "known_distance_tenth_percentile": known_floor,
        "fraction_beyond_known_median": float(
            np.mean(probe_distance > np.median(known_distance))
        ),
        "reaches_past_known_cloud": bool(maximum_probe > known_floor),
    }


def boundary_scores(
    points: Tensor,
    geometry: Geometry,
    log_beta: Tensor,
    *,
    classes: Tensor | None = None,
) -> Tensor:
    """Score points against the boundary ellipsoid of each class.

    ``classes`` selects one class per point, returning a vector; otherwise the
    full ``(points, classes)`` matrix is returned. The per-point form is used
    everywhere a point has an owner, which keeps the memory cost linear rather
    than quadratic in the class count.
    """
    centers = torch.as_tensor(geometry.centers, dtype=points.dtype)
    bases = torch.as_tensor(geometry.bases, dtype=points.dtype)
    beta = torch.exp(log_beta)
    tangent_beta = beta[:, :-1]
    residual_beta = beta[:, -1]

    if classes is None:
        deltas = points[:, None, :] - centers[None, :, :]
        coordinates = torch.einsum("nkd,kdr->nkr", deltas, bases)
        projection = torch.einsum("nkr,kdr->nkd", coordinates, bases)
        residual = deltas - projection
        tangent_term = torch.sum(
            coordinates.square() / tangent_beta[None, :, :].square(), dim=2
        )
        residual_term = torch.sum(residual.square(), dim=2) / residual_beta[
            None, :
        ].square()
    else:
        centers = centers[classes]
        bases = bases[classes]
        deltas = points - centers
        coordinates = torch.einsum("nd,ndr->nr", deltas, bases)
        projection = torch.einsum("nr,ndr->nd", coordinates, bases)
        residual = deltas - projection
        tangent_term = torch.sum(
            coordinates.square() / tangent_beta[classes].square(), dim=1
        )
        residual_term = torch.sum(residual.square(), dim=1) / residual_beta[
            classes
        ].square()
    return torch.sqrt(torch.clamp(tangent_term + residual_term, min=_SQRT_FLOOR))


def owner_scores_numpy(
    features: np.ndarray,
    labels: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
) -> np.ndarray:
    """Boundary score of each row against its own class, without autograd."""
    with torch.no_grad():
        return (
            boundary_scores(
                torch.as_tensor(features, dtype=torch.float64),
                geometry,
                torch.as_tensor(log_beta, dtype=torch.float64),
                classes=torch.as_tensor(labels, dtype=torch.long),
            )
            .cpu()
            .numpy()
        )


def minimum_scores_numpy(
    features: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
    *,
    chunk_rows: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Best-matching class and its boundary score, for points with no owner.

    Out-of-set rows have no true class, so their rejection is decided by the
    class that comes closest to accepting them. Chunked because the full
    ``(rows, classes, dimension)`` delta tensor would be gigabytes.
    """
    best_score = np.empty(features.shape[0], dtype=np.float64)
    best_class = np.empty(features.shape[0], dtype=np.int64)
    beta = torch.as_tensor(log_beta, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, features.shape[0], chunk_rows):
            stop = min(start + chunk_rows, features.shape[0])
            block = torch.as_tensor(features[start:stop], dtype=torch.float64)
            scores = boundary_scores(block, geometry, beta)
            values, indices = torch.min(scores, dim=1)
            best_score[start:stop] = values.cpu().numpy()
            best_class[start:stop] = indices.cpu().numpy()
    return best_class, best_score


# ---------------------------------------------------------------------------
# Probe generation
# ---------------------------------------------------------------------------


def _directions(
    geometry: Geometry,
    class_index: int,
    families: Sequence[str],
    generator: np.random.Generator,
) -> list[np.ndarray]:
    """Unit displacement directions for one class, shared by both placements.

    Both placement rules draw the *same* directions from the *same* generator
    state. Only the distance along them differs, so the v12-form arm and the
    absolute arm are separated by exactly one quantity.
    """
    basis = geometry.bases[class_index]
    directions: list[np.ndarray] = []
    if "axis_tangent" in families:
        for axis in range(geometry.rank):
            for sign in (-1.0, 1.0):
                directions.append(sign * basis[:, axis])
    if "normal" in families:
        normal = generator.normal(size=geometry.dimension)
        normal = normal - basis @ (basis.T @ normal)
        directions.append(normal / max(float(np.linalg.norm(normal)), _EPSILON))
    if "masking" in families and geometry.class_count > 1:
        distances = np.linalg.norm(
            geometry.centers - geometry.centers[class_index], axis=1
        )
        distances[class_index] = np.inf
        toward = geometry.centers[int(np.argmin(distances))] - geometry.centers[
            class_index
        ]
        coordinates = basis.T @ toward
        axis = int(np.argmax(np.abs(coordinates)))
        sign = 1.0 if coordinates[axis] >= 0.0 else -1.0
        directions.append(sign * basis[:, axis])
    if "random_direction" in families:
        direction = generator.normal(size=geometry.dimension)
        directions.append(direction / max(float(np.linalg.norm(direction)), _EPSILON))
    if not directions:
        raise ValueError("at least one probe family is required")
    return directions


def _axis_extent(geometry: Geometry, class_index: int, direction: np.ndarray) -> float:
    """Fitted extent along a direction, from the frozen Phase A geometry.

    Used only for reporting. The v12-form arm does **not** use this: its
    displacement must be a live function of the boundary radii, or the
    cancellation that makes it degenerate never happens. See ``probe_points``.
    """
    basis = geometry.bases[class_index]
    coordinates = basis.T @ direction
    tangent = float(
        np.sum((coordinates * geometry.tangent_scales[class_index]) ** 2)
    )
    residual_component = float(
        np.linalg.norm(direction - basis @ coordinates)
        * geometry.residual_scales[class_index]
    )
    return float(np.sqrt(tangent + residual_component**2 + _EPSILON))


@dataclass(frozen=True)
class ProbeSpec:
    """Synthetic negatives held as directions rather than points.

    The two placement rules share directions, owners and multipliers and differ
    only in the distance travelled along each direction. Keeping the spec
    separate from the points is what lets the v12-form arm rebuild its points
    from the live radii on every evaluation, which is how v12 behaved and the
    only way its degeneracy reproduces.
    """

    directions: np.ndarray
    owners: np.ndarray
    multipliers: np.ndarray

    def __len__(self) -> int:
        return int(self.directions.shape[0])

    def with_owners(self, owners: np.ndarray) -> "ProbeSpec":
        return ProbeSpec(self.directions, owners, self.multipliers)

    def subset(self, rows: np.ndarray) -> "ProbeSpec":
        return ProbeSpec(
            self.directions[rows], self.owners[rows], self.multipliers[rows]
        )


def build_probe_spec(
    geometry: Geometry,
    *,
    families: Sequence[str],
    multipliers: Sequence[float],
    seed: int,
) -> ProbeSpec:
    """Draw displacement directions once, to be shared by both placements."""
    generator = np.random.default_rng(seed)
    directions: list[np.ndarray] = []
    owners: list[int] = []
    scaling: list[float] = []
    for class_index in range(geometry.class_count):
        for direction in _directions(geometry, class_index, families, generator):
            for multiplier in multipliers:
                directions.append(np.asarray(direction, dtype=np.float64))
                owners.append(class_index)
                scaling.append(float(multiplier))
    return ProbeSpec(
        np.stack(directions),
        np.asarray(owners, dtype=np.int64),
        np.asarray(scaling, dtype=np.float64),
    )


def probe_points(
    geometry: Geometry,
    log_beta: Tensor,
    spec: ProbeSpec,
    *,
    placement: str,
    unit: float,
) -> Tensor:
    """Place the negatives. This one function is the entire hypothesis.

    ``absolute`` travels ``multiplier * unit``, a global length that no radius
    appears in, so the resulting score ``multiplier * unit / extent(beta)``
    varies with the radius and can be supervised.

    ``relative`` reproduces v12: it travels ``multiplier * extent(beta)``,
    where ``extent`` is the distance along that direction at which the boundary
    score is one. The score is then exactly ``multiplier`` — the same constant
    for every class, at every radius, forever. Rebuilding the points from the
    live ``log_beta`` is essential; precomputing them as constants would break
    the cancellation and make v12's objective look healthy.
    """
    if placement not in {"absolute", "relative"}:
        raise ValueError(f"unknown probe placement: {placement}")
    owners = torch.as_tensor(spec.owners, dtype=torch.long)
    centers = torch.as_tensor(geometry.centers, dtype=log_beta.dtype)[owners]
    directions = torch.as_tensor(spec.directions, dtype=log_beta.dtype)
    multipliers = torch.as_tensor(spec.multipliers, dtype=log_beta.dtype)

    if placement == "absolute":
        distance = multipliers * float(unit)
    else:
        beta = torch.exp(log_beta)[owners]
        bases = torch.as_tensor(geometry.bases, dtype=log_beta.dtype)[owners]
        coordinates = torch.einsum("pd,pdr->pr", directions, bases)
        perpendicular = directions - torch.einsum(
            "pr,pdr->pd", coordinates, bases
        )
        inverse = torch.sqrt(
            torch.clamp(
                torch.sum(coordinates.square() / beta[:, :-1].square(), dim=1)
                + torch.sum(perpendicular.square(), dim=1) / beta[:, -1].square(),
                min=_SQRT_FLOOR,
            )
        )
        distance = multipliers / inverse
    return centers + distance[:, None] * directions


def probe_points_numpy(
    geometry: Geometry,
    log_beta: np.ndarray,
    spec: ProbeSpec,
    *,
    placement: str,
    unit: float,
) -> np.ndarray:
    with torch.no_grad():
        return (
            probe_points(
                geometry,
                torch.as_tensor(log_beta, dtype=torch.float64),
                spec,
                placement=placement,
                unit=unit,
            )
            .cpu()
            .numpy()
        )


def shuffled_owners(owners: np.ndarray, *, class_count: int, seed: int) -> np.ndarray:
    """N83.1's null: same probes, same count, randomly reassigned classes.

    Preserves the probe set exactly and destroys only the correspondence
    between a negative and the class it was synthesized for. A boundary that
    moves as far under this as under the real assignment has moved for reasons
    unrelated to its own geometry.
    """
    generator = np.random.default_rng(seed)
    return generator.integers(0, class_count, size=len(owners), dtype=np.int64)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def probe_term(
    geometry: Geometry,
    log_beta: Tensor,
    spec: ProbeSpec,
    *,
    placement: str,
    unit: float,
    margin: float,
) -> Tensor:
    """Push synthetic negatives outside their owner's boundary.

    Acceptance is ``score <= 1``, so rejection wants ``score >= 1 + margin``.
    This term alone is what the degeneracy report differentiates.
    """
    points = probe_points(
        geometry, log_beta, spec, placement=placement, unit=unit
    )
    owners = torch.as_tensor(spec.owners, dtype=torch.long)
    scores = boundary_scores(points, geometry, log_beta, classes=owners)
    return torch.mean(torch.relu(1.0 + margin - scores))


def known_term(
    features: Tensor,
    labels: Tensor,
    geometry: Geometry,
    log_beta: Tensor,
    *,
    margin: float,
) -> Tensor:
    """Keep genuine members inside their own boundary."""
    scores = boundary_scores(features, geometry, log_beta, classes=labels)
    return torch.mean(torch.relu(scores - (1.0 - margin)))


def mean_probe_score(
    geometry: Geometry,
    log_beta: Tensor,
    spec: ProbeSpec,
    *,
    placement: str,
    unit: float,
) -> Tensor:
    """The probe objective with its hinge removed.

    This is what the degeneracy report differentiates. The hinge cannot be
    used for that purpose because a probe already far outside its boundary
    contributes exactly zero gradient — correct behaviour for training, fatal
    for an instrument, since a fully saturated healthy objective and a
    genuinely scale-blind one both read zero. Stripping the hinge asks the
    question the contract actually means: does the score at the probe
    locations depend on the radii at all?
    """
    points = probe_points(geometry, log_beta, spec, placement=placement, unit=unit)
    owners = torch.as_tensor(spec.owners, dtype=torch.long)
    return torch.mean(boundary_scores(points, geometry, log_beta, classes=owners))


def degeneracy_report(
    geometry: Geometry,
    log_beta: np.ndarray,
    spec: ProbeSpec,
    *,
    placement: str,
    unit: float,
    margin: float,
    rescale_factors: Sequence[float] = (0.5, 1.0, 2.0),
) -> dict[str, object]:
    """The standing contract generalised from M77, and M83's gating operand.

    Two independent readings of one question — does the probe supervision
    depend on the scale parameters it claims to supervise?

    1. **Gradient.** The norm of the mean probe score's gradient with respect
       to ``log_beta``. v12's placement makes every own-class probe score
       exactly ``multiplier``, a constant, so this is zero to machine
       precision whatever the radii are.
    2. **Rescale.** Multiply every radius by a constant and recompute. A
       scale-blind rule returns the identical scores; the spread across
       factors is a second witness that does not rely on autograd being wired
       correctly.

    Both are read on the probe supervision **alone**. Including the
    known-class term would make every arm pass, because real data points do
    not move when the radii do — which is the exact error M77 exists to
    prevent.

    ``objective_gradient_norm_log_beta`` and ``active_fraction`` are reported
    beside them but are not the contract: they describe the hinge's operating
    point, and a zero there means only that the probes are already outside.
    """
    parameter = torch.tensor(log_beta, dtype=torch.float64, requires_grad=True)
    score = mean_probe_score(
        geometry, parameter, spec, placement=placement, unit=unit
    )
    (gradient,) = torch.autograd.grad(score, [parameter], allow_unused=True)
    gradient_norm = (
        0.0 if gradient is None else float(torch.linalg.vector_norm(gradient))
    )

    objective = torch.tensor(log_beta, dtype=torch.float64, requires_grad=True)
    loss = probe_term(
        geometry, objective, spec, placement=placement, unit=unit, margin=margin
    )
    (objective_gradient,) = torch.autograd.grad(loss, [objective], allow_unused=True)
    objective_gradient_norm = (
        0.0
        if objective_gradient is None
        else float(torch.linalg.vector_norm(objective_gradient))
    )

    scores: list[float] = []
    with torch.no_grad():
        points = probe_points(
            geometry,
            torch.as_tensor(log_beta, dtype=torch.float64),
            spec,
            placement=placement,
            unit=unit,
        )
        raw = boundary_scores(
            points,
            geometry,
            torch.as_tensor(log_beta, dtype=torch.float64),
            classes=torch.as_tensor(spec.owners, dtype=torch.long),
        )
        active_fraction = float(torch.mean((raw < 1.0 + margin).double()))
        for factor in rescale_factors:
            shifted = torch.as_tensor(
                log_beta + float(np.log(factor)), dtype=torch.float64
            )
            scores.append(
                float(
                    mean_probe_score(
                        geometry, shifted, spec, placement=placement, unit=unit
                    )
                )
            )
    return {
        "probe_term": float(loss.detach()),
        "mean_probe_score": float(score.detach()),
        "gradient_norm_log_beta": gradient_norm,
        "objective_gradient_norm_log_beta": objective_gradient_norm,
        "active_fraction": active_fraction,
        "rescale_factors": [float(value) for value in rescale_factors],
        "rescale_scores": scores,
        "rescale_spread": float(max(scores) - min(scores)),
    }


def train_boundary(
    features: np.ndarray,
    labels: np.ndarray,
    geometry: Geometry,
    spec: ProbeSpec,
    *,
    placement: str,
    unit: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
    probe_weight: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Phase B. Train the boundary radii only; the geometry never moves.

    Returns the final ``log_beta`` and a per-epoch history. Initialisation is
    the fitted geometry itself, so the boundary starts as the fitted ellipsoid
    and every reported displacement is displacement away from Phase A.
    """
    torch.manual_seed(seed)
    initial = np.concatenate(
        [
            np.log(geometry.tangent_scales),
            np.log(geometry.residual_scales)[:, None],
        ],
        axis=1,
    )
    parameter = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=learning_rate)

    feature_tensor = torch.as_tensor(features, dtype=torch.float64)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    generator = np.random.default_rng(seed)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        order = generator.permutation(len(features))
        probe_order = generator.permutation(len(spec))
        totals = {"known": 0.0, "probe": 0.0, "total": 0.0}
        batches = 0
        cursor = 0
        for start in range(0, len(order), batch_size):
            rows = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            probe_rows = probe_order[cursor : cursor + batch_size]
            if len(probe_rows) == 0:
                cursor = 0
                probe_rows = probe_order[:batch_size]
            cursor = (cursor + batch_size) % max(len(spec), 1)
            known = known_term(
                feature_tensor[rows],
                label_tensor[rows],
                geometry,
                parameter,
                margin=margin,
            )
            probe = probe_term(
                geometry,
                parameter,
                spec.subset(probe_rows),
                placement=placement,
                unit=unit,
                margin=margin,
            )
            total = known + probe_weight * probe
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            totals["known"] += float(known.detach())
            totals["probe"] += float(probe.detach())
            totals["total"] += float(total.detach())
            batches += 1
        history.append(
            {
                "epoch": epoch,
                "known": totals["known"] / max(batches, 1),
                "probe": totals["probe"] / max(batches, 1),
                "total": totals["total"] / max(batches, 1),
            }
        )
    return parameter.detach().cpu().numpy(), history


# ---------------------------------------------------------------------------
# M84 — real out-group exposure
# ---------------------------------------------------------------------------


def exposure_owners(
    features: np.ndarray, geometry: Geometry, log_beta: np.ndarray
) -> np.ndarray:
    """N84.7. The class whose boundary comes closest to accepting each row.

    A real out-group row has no class of its own, so the only meaningful
    constraint is against whichever boundary would otherwise let it in. This
    is the exact argmin, not a nearest-centroid proxy: the two agree for only
    31% of rows because the ellipsoids are anisotropic enough that Euclidean
    proximity does not predict acceptance.
    """
    best_class, _ = minimum_scores_numpy(features, geometry, log_beta)
    return best_class


def owner_agreement(
    features: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
    owners: np.ndarray,
) -> float:
    """N84.7. How much of the fixed owner assignment survived training.

    Reported, never gated on. Rejection is always read with the exact minimum
    over all classes, so drift costs the supervision efficiency rather than
    costing the measurement its validity.
    """
    if len(features) == 0:
        return 1.0
    return float(np.mean(exposure_owners(features, geometry, log_beta) == owners))


def exposure_validity(
    features: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
    *,
    margin: float,
) -> dict[str, float | bool]:
    """N84.6. Can the negatives move the objective at all?

    The direct analogue of ``probe_validity``. A hinge at ``1 + margin`` that
    every negative already satisfies has exactly zero gradient, and a ladder
    trained there is flat for an instrumental reason that looks identical to a
    real negative result. At the raw fitted radii the active fraction on this
    corpus is 0.0000; at matched coverage it is 0.9833.
    """
    if len(features) == 0:
        return {
            "row_count": 0,
            "active_fraction": 0.0,
            "minimum_score_median": 0.0,
            "already_rejected_fraction": 0.0,
            "term_is_live": False,
        }
    _, minimum = minimum_scores_numpy(features, geometry, log_beta)
    active = float(np.mean(minimum < 1.0 + margin))
    return {
        "row_count": int(len(features)),
        "active_fraction": active,
        "minimum_score_median": float(np.median(minimum)),
        "already_rejected_fraction": float(np.mean(minimum > 1.0)),
        "term_is_live": bool(active > 0.0),
    }


def exposure_term(
    points: Tensor,
    geometry: Geometry,
    log_beta: Tensor,
    *,
    owners: Tensor,
    margin: float,
) -> Tensor:
    """Push real out-group rows outside the boundary that would accept them.

    Structurally identical to ``probe_term`` once the points exist, which is
    the point: the ladder changes what the negatives *are*, not how they are
    used.
    """
    scores = boundary_scores(points, geometry, log_beta, classes=owners)
    return torch.mean(torch.relu(1.0 + margin - scores))


def sample_exposure(
    labels: np.ndarray, *, count: int, diversity: int, seed: int
) -> np.ndarray:
    """Draw ``count`` rows from exactly ``diversity`` exposure classes.

    The ladder's two axes are separated here and nowhere else. Classes are
    drawn without replacement and each contributes an equal share, so a cell
    is a statement about ``count`` and ``diversity`` jointly rather than about
    whichever classes happened to be dense.
    """
    if count == 0:
        return np.empty(0, dtype=np.int64)
    available = np.unique(labels)
    if diversity > len(available):
        raise ValueError(
            f"diversity {diversity} exceeds the {len(available)} filled exposure classes"
        )
    if count % diversity != 0:
        raise ValueError(f"count {count} is not divisible by diversity {diversity}")
    per_class = count // diversity
    generator = np.random.default_rng(seed)
    chosen = generator.choice(available, size=diversity, replace=False)
    rows: list[np.ndarray] = []
    for label in chosen:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(
                f"class {int(label)} holds {len(candidates)} rows, "
                f"below the {per_class} this cell requires"
            )
        rows.append(generator.choice(candidates, size=per_class, replace=False))
    return np.sort(np.concatenate(rows))


def moment_matched_negatives(
    features: np.ndarray, *, count: int, seed: int
) -> np.ndarray:
    """N84.3. Gaussian negatives carrying the sample's mean and covariance.

    R5's null for a ladder whose negatives have no owner to permute. It shares
    the exposure sample's first and second moments exactly and none of its
    content, so beating it means the boundary used the structure of real
    images rather than statistics a covariance estimate would have supplied.

    The covariance is deliberately not shrunk or regularised. At the small
    rungs it is rank-deficient, and a null confined to the span of ten real
    rows is the honest matched control for a cell trained on those ten rows.
    """
    if count == 0:
        return np.empty((0, features.shape[1]), dtype=features.dtype)
    generator = np.random.default_rng(seed)
    mean = np.mean(features, axis=0)
    centred = features - mean
    covariance = centred.T @ centred / max(len(features) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    factor = eigenvectors * np.sqrt(eigenvalues)[None, :]
    noise = generator.standard_normal((count, features.shape[1]))
    return (mean[None, :] + noise @ factor.T).astype(features.dtype)


def train_exposure_boundary(
    features: np.ndarray,
    labels: np.ndarray,
    geometry: Geometry,
    initial: np.ndarray,
    *,
    negatives: np.ndarray,
    owners: np.ndarray,
    epochs: int,
    batch_size: int,
    exposure_batch_size: int,
    learning_rate: float,
    margin: float,
    exposure_weight: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """M84's Phase B. Known rows in, exposure rows out, geometry frozen.

    ``initial`` is passed rather than rebuilt from the geometry because N84.6
    starts training at the matched-coverage point, where the exposure hinge is
    live, instead of at the raw fitted radii, where it is not.

    When ``negatives`` is empty the exposure term is skipped entirely, which is
    the ``known_only`` arm: literally ``N_out = 0`` at the full step budget.
    """
    torch.manual_seed(seed)
    parameter = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=learning_rate)

    feature_tensor = torch.as_tensor(features, dtype=torch.float64)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    negative_tensor = torch.as_tensor(negatives, dtype=torch.float64)
    owner_tensor = torch.as_tensor(owners, dtype=torch.long)
    generator = np.random.default_rng(seed)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        order = generator.permutation(len(features))
        exposure_order = (
            generator.permutation(len(negatives)) if len(negatives) else None
        )
        totals = {"known": 0.0, "exposure": 0.0, "total": 0.0}
        batches = 0
        cursor = 0
        for start in range(0, len(order), batch_size):
            rows = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            known = known_term(
                feature_tensor[rows],
                label_tensor[rows],
                geometry,
                parameter,
                margin=margin,
            )
            if exposure_order is None:
                exposure = torch.zeros((), dtype=torch.float64)
            else:
                # The exposure sample is smaller than the corpus at every rung,
                # so it cycles independently rather than being padded or
                # resampled, and every row is seen the same number of times.
                taken = exposure_order[cursor : cursor + exposure_batch_size]
                if len(taken) == 0:
                    cursor = 0
                    taken = exposure_order[:exposure_batch_size]
                cursor = (cursor + exposure_batch_size) % max(len(negatives), 1)
                selection = torch.as_tensor(taken, dtype=torch.long)
                exposure = exposure_term(
                    negative_tensor[selection],
                    geometry,
                    parameter,
                    owners=owner_tensor[selection],
                    margin=margin,
                )
            total = known + exposure_weight * exposure
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            totals["known"] += float(known.detach())
            totals["exposure"] += float(exposure.detach())
            totals["total"] += float(total.detach())
            batches += 1
        history.append(
            {
                "epoch": epoch,
                "known": totals["known"] / max(batches, 1),
                "exposure": totals["exposure"] / max(batches, 1),
                "total": totals["total"] / max(batches, 1),
            }
        )
    return parameter.detach().cpu().numpy(), history


def tangent_anisotropy(log_beta: np.ndarray) -> float:
    """N84.5's descriptive operand: how unequal are a class's tangent radii?

    The registered prediction is that any improvement exposure buys shows up
    as anisotropy rather than as a smaller mean radius, because a uniform
    radius change is exactly what matched coverage removes. Reported whichever
    way recall goes.
    """
    tangent = log_beta[:, :-1]
    return float(np.mean(np.std(tangent, axis=1)))


# ---------------------------------------------------------------------------
# Displacement, coverage matching, and the rejection operands
# ---------------------------------------------------------------------------


def boundary_displacement(initial: np.ndarray, final: np.ndarray) -> dict[str, float]:
    """Split movement into the part coverage matching destroys and the rest.

    A uniform per-class inflation is pure radius, and matching known coverage
    later removes it entirely. Only the shape term — the per-axis pattern after
    the class mean is subtracted — can still affect a matched-coverage
    measurement, so it is the honest operand for H83.
    """
    delta = final - initial
    radius = np.mean(delta, axis=1)
    shape = delta - radius[:, None]
    return {
        "total": float(np.linalg.norm(delta) / np.sqrt(delta.size)),
        "radius": float(np.linalg.norm(radius) / np.sqrt(radius.size)),
        "shape": float(np.linalg.norm(shape) / np.sqrt(shape.size)),
        "maximum_absolute": float(np.max(np.abs(delta))),
    }


def matched_coverage_offsets(
    features: np.ndarray,
    labels: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
    *,
    coverage: float,
    class_count: int,
) -> np.ndarray:
    """Per-class log offsets placing known acceptance at ``coverage``.

    N83.3. Rejection recall is maximised by rejecting everything, so no recall
    figure may be read at an unmatched operating point. Scaling every radius of
    a class by one factor moves its score by the reciprocal, so the offset is
    the log of the calibrated threshold on that class's own scores.

    The threshold is the split-conformal order statistic, the
    ``ceil((n + 1) * coverage)``-th smallest score, not the empirical quantile.
    The distinction is not pedantic at this sample size. The empirical quantile
    of 32 rows sits at about the 28.9th order statistic, and a fresh row falls
    below the k-th of n with probability k / (n + 1), so it delivers roughly 88
    percent coverage where 90 was asked for — and the gap is a downward bias
    that no amount of averaging over classes removes. The conformal statistic
    is the smallest threshold whose coverage on unseen rows is guaranteed to be
    at least ``coverage`` in finite samples, which is exactly the guarantee
    N83.3 needs, since the whole point of the split is that the threshold is
    read on rows it was not chosen on.
    """
    scores = owner_scores_numpy(features, labels, geometry, log_beta)
    offsets = np.zeros(class_count, dtype=np.float64)
    for label in range(class_count):
        rows = np.flatnonzero(labels == label)
        if len(rows) == 0:
            continue
        owned = np.sort(scores[rows])
        rank = int(np.ceil((len(owned) + 1) * coverage))
        threshold = float(owned[min(rank, len(owned)) - 1])
        offsets[label] = float(np.log(max(threshold, _EPSILON)))
    return offsets


def apply_offsets(log_beta: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return log_beta + offsets[:, None]


def acceptance_rate(
    features: np.ndarray,
    labels: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
) -> float:
    """Fraction of owned rows their own class accepts."""
    if len(features) == 0:
        return float("nan")
    scores = owner_scores_numpy(features, labels, geometry, log_beta)
    return float(np.mean(scores <= 1.0))


def rejection_recall(
    features: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
    *,
    domains: np.ndarray | None = None,
    domain_count: int = 6,
) -> dict[str, object]:
    """Fraction of unowned rows every class rejects, overall and per domain.

    A row is rejected only if no class accepts it, so the operand is decided by
    the closest class. Per-domain figures accompany the aggregate under N83.2,
    because a boundary reading rendering style rather than novelty would show
    it here first.
    """
    if len(features) == 0:
        return {"rejection_recall": None, "row_count": 0, "per_domain": {}}
    _, scores = minimum_scores_numpy(features, geometry, log_beta)
    rejected = scores > 1.0
    per_domain: dict[str, object] = {}
    if domains is not None:
        for domain in range(domain_count):
            rows = np.flatnonzero(domains == domain)
            per_domain[str(domain)] = {
                "row_count": int(len(rows)),
                "rejection_recall": (
                    float(np.mean(rejected[rows])) if len(rows) else None
                ),
            }
    return {
        "rejection_recall": float(np.mean(rejected)),
        "row_count": int(len(features)),
        "per_domain": per_domain,
    }


def probe_rejection(
    geometry: Geometry,
    log_beta: np.ndarray,
    spec: ProbeSpec,
    *,
    placement: str,
    unit: float,
) -> float:
    """Fraction of synthetic negatives their owner rejects.

    Used for the held-out family operand: a boundary that learned a surface
    rather than its training probes must still reject families it never saw.
    """
    if len(spec) == 0:
        return float("nan")
    points = probe_points_numpy(
        geometry, log_beta, spec, placement=placement, unit=unit
    )
    scores = owner_scores_numpy(points, spec.owners, geometry, log_beta)
    return float(np.mean(scores > 1.0))


# ---------------------------------------------------------------------------
# M85 - the threshold-free operand L2 registered and the leg never reported
# ---------------------------------------------------------------------------


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so AUROC is exact rather than approximate.

    Ties are not hypothetical here. A boundary that saturates assigns identical
    scores to many rows, and resolving those ties arbitrarily would move AUROC
    towards whichever side happened to be concatenated second.
    """
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def score_auroc(known: np.ndarray, unseen: np.ndarray) -> float:
    """P(unseen scores above known), with ties counted as half.

    The unseen rows are the positive class because the boundary score rises
    away from the geometry, so a detector that works reads above 0.5.
    """
    if len(known) == 0 or len(unseen) == 0:
        raise ValueError("AUROC needs rows on both sides")
    ranks = average_ranks(np.concatenate([known, unseen]))
    positive = ranks[len(known) :].sum()
    return float(
        (positive - len(unseen) * (len(unseen) + 1) / 2) / (len(known) * len(unseen))
    )


def domain_auroc(
    known: np.ndarray,
    known_domains: np.ndarray,
    unseen: np.ndarray,
    unseen_domains: np.ndarray,
    *,
    domain_count: int,
) -> dict[str, Any]:
    """Aggregate AUROC and the same figure within each rendering domain.

    Both are reported because they answer different questions. The aggregate is
    what L2 asks for. The per-domain split is what tells you whether an
    aggregate near chance is a uniform failure or an average over domains that
    do not behave alike, which on this corpus it is.

    ``within_domain_auroc`` weights the per-domain figures by their unseen row
    counts. It is not a better version of the aggregate and does not replace
    it; the gap between the two is the reportable quantity, because a pooled
    AUROC below every one of its parts means the score separates rendering
    style more strongly than it separates novelty.
    """
    per_domain: dict[str, Any] = {}
    weighted, weight = 0.0, 0
    for domain in range(domain_count):
        known_rows = known[known_domains == domain]
        unseen_rows = unseen[unseen_domains == domain]
        value = (
            score_auroc(known_rows, unseen_rows)
            if len(known_rows) and len(unseen_rows)
            else None
        )
        if value is not None:
            weighted += value * len(unseen_rows)
            weight += len(unseen_rows)
        per_domain[str(domain)] = {
            "auroc": value,
            "known_rows": int(len(known_rows)),
            "unseen_rows": int(len(unseen_rows)),
        }
    return {
        "auroc": score_auroc(known, unseen),
        "within_domain_auroc": weighted / weight if weight else None,
        "known_rows": int(len(known)),
        "unseen_rows": int(len(unseen)),
        "per_domain": per_domain,
    }


def far_field_points(
    features: np.ndarray,
    *,
    count: int,
    multiplier: float,
    seed: int,
) -> np.ndarray:
    """The positive control L2 requires: points that must be detectable.

    Isotropic noise about the data's own mean, scaled to a stated multiple of
    the median distance from a row to that mean. An instrument that cannot
    separate these from real rows is broken, and any AUROC it reports on real
    unseen rows is uninterpretable - which is the check M83.1 lacked.
    """
    if multiplier <= 1.0:
        raise ValueError("a far-field control must sit outside the data")
    generator = np.random.default_rng(seed)
    centre = features.mean(axis=0)
    radius = float(np.median(np.linalg.norm(features - centre, axis=1)))
    directions = generator.normal(size=(count, features.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return centre + directions * radius * multiplier
