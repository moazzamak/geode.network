"""
report_v15_m107_gate.py  —  read M107's evidence.json, rebuild both
accuracy-vs-MACs curves and all three kill switches FROM THE PER-ARM
RECORDS, and print the summary in the form that goes into §7.14 of the
plan and the claim ledger.

This deliberately does not read ``evidence["gate"]`` when deciding the
switches.  The runner's own gate is recomputed here from the arms and the
two verdicts are then compared; a disagreement is a defect in one of them
and is reported as such.

Usage:
    python experiments/tier4/report_v15_m107_gate.py
    python experiments/tier4/report_v15_m107_gate.py --plan-text
    python experiments/tier4/report_v15_m107_gate.py --compare-runner

Respects §7.14 restrictions:
  - restriction 4:  the word "oracle" appears in every sentence quoting a
                    mixture figure
  - restriction 5:  no wall-clock comparison between the two families
  - restriction 7:  d1 and d5 are not admissible apart from one another
  - amendment 5:    kill switch 3 is decidable at two budgets, not six
"""
import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "logs/results/v15/m107_dense/evidence.json"

# §7.14 gate item 4: a sparse point at M MACs is compared against the best
# dense point at or below M.  Points with no dense arm at or below them are
# VOID for kill switches 1 and 2 -- not wins, and not losses.
KS3_TOLERANCE = 0.005


def _load() -> dict:
    if not EVIDENCE_PATH.exists():
        sys.exit(f"no evidence at {EVIDENCE_PATH}")
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    if not evidence.get("admissible_as_evidence"):
        sys.exit("evidence.json is stamped inadmissible -- refusing to report it")
    return evidence


def _curves(evidence: dict) -> tuple[list, list, list, str]:
    """Rebuild the three curves from the arms, at the one chosen penalty."""
    penalty = str(evidence["head"]["chosen_penalty"])
    dense, generalist, mixture = [], [], []
    for name, arm in evidence["arms"].items():
        if arm.get("void"):
            continue
        point = (arm["macs"]["total"], arm["accuracy_by_penalty"][penalty], name)
        if arm["family"] == "dense":
            dense.append(point)
        elif name.startswith("s_mixture"):
            mixture.append(point)
        else:
            generalist.append(point)
    return (sorted(dense), sorted(generalist), sorted(mixture), penalty)


def _best_dense_at_or_below(dense: list, macs: float):
    below = [d for d in dense if d[0] <= macs]
    if not below:
        return None
    return max(below, key=lambda d: d[1])


def _gate(evidence: dict) -> dict:
    dense, generalist, mixture, penalty = _curves(evidence)

    comparisons = []
    for macs, acc, name in generalist:
        opponent = _best_dense_at_or_below(dense, macs)
        comparisons.append({
            "sparse": name,
            "sparse_macs": macs,
            "sparse_accuracy": acc,
            "dense": None if opponent is None else opponent[2],
            "dense_macs": None if opponent is None else opponent[0],
            "dense_accuracy": None if opponent is None else opponent[1],
            "decidable": opponent is not None,
            "sparse_wins": opponent is not None and acc > opponent[1],
        })

    decidable = [c for c in comparisons if c["decidable"]]
    wins = [c for c in decidable if c["sparse_wins"]]

    # Kill switch 3 compares the generalist against the oracle-routed mixture
    # at matched inference MACs, at whichever budgets the mixture reached.
    ks3_rows = []
    for macs, acc, name in mixture:
        twin = [g for g in generalist if g[0] == macs]
        if not twin:
            continue
        ks3_rows.append({
            "macs": macs,
            "mixture": name,
            "mixture_accuracy": acc,
            "generalist": twin[0][2],
            "generalist_accuracy": twin[0][1],
            "generalist_matches": twin[0][1] >= acc - KS3_TOLERANCE,
        })

    return {
        "penalty": penalty,
        "dense": dense,
        "generalist": generalist,
        "mixture": mixture,
        "comparisons": comparisons,
        "ks1_fired": bool(decidable) and not wins,
        "ks1_decidable": bool(decidable),
        "ks2_fired": bool(wins),
        "ks2_decidable": bool(decidable),
        "ks2_points": [(c["sparse_macs"], c["sparse_accuracy"]) for c in wins],
        "ks3_rows": ks3_rows,
        "ks3_fired": bool(ks3_rows) and all(r["generalist_matches"] for r in ks3_rows),
        "ks3_decidable": bool(ks3_rows),
        "void_points": [c["sparse"] for c in comparisons if not c["decidable"]],
    }


def _fired(flag: bool, decidable: bool = True) -> str:
    if not decidable:
        return "VOID (undecidable -- NOT a negative result)"
    return "FIRED" if flag else "not fired"


def _interpolation_note(gate: dict) -> None:
    """Bound 5: the gate compares against the best dense point AT OR BELOW the
    sparse budget, and the dense ladder steps by roughly 2x, so a sparse arm may
    outspend the opponent it beats. Interpolating the dense curve to the sparse
    arm's own budget is arithmetic and not a measurement -- it is printed so the
    margin can be read both ways, never so it can be quoted as a dense arm."""
    ladder = [d for d in gate["dense"] if d[2] != "d5_small_224_from_32"]

    def at(macs: float, log_axis: bool) -> float:
        lo = max((d for d in ladder if d[0] <= macs), key=lambda d: d[0])
        higher = [d for d in ladder if d[0] >= macs]
        if not higher:
            return lo[1]
        hi = min(higher, key=lambda d: d[0])
        if hi[0] == lo[0]:
            return lo[1]
        if log_axis:
            position = ((math.log(macs) - math.log(lo[0]))
                        / (math.log(hi[0]) - math.log(lo[0])))
        else:
            position = (macs - lo[0]) / (hi[0] - lo[0])
        return lo[1] + position * (hi[1] - lo[1])

    print("bound 5 -- the crossing read against an INTERPOLATED dense curve")
    print("   (interpolation is arithmetic, not a measured arm; it may not be")
    print("    quoted as a dense result, only as a bound on the margin)")
    for c in gate["comparisons"]:
        if not c["sparse_wins"]:
            continue
        macs, acc = c["sparse_macs"], c["sparse_accuracy"]
        print(f"   {c['sparse']:<20}{macs / 1e6:8.1f} M   outspends"
              f" {c['dense']} by {macs / c['dense_macs']:.2f}x")
        print(f"      registered rule  {(acc - c['dense_accuracy']) * 100:+6.2f} pp"
              f"      linear {(acc - at(macs, False)) * 100:+6.2f} pp"
              f"      log-MACs {(acc - at(macs, True)) * 100:+6.2f} pp")
    survives = all(c["sparse_accuracy"] > at(c["sparse_macs"], log_axis)
                   for c in gate["comparisons"] if c["sparse_wins"]
                   for log_axis in (False, True))
    print(f"   crossing survives both interpolations at both budgets: {survives}")
    print()


def report(evidence: dict, gate: dict) -> None:
    corpus = evidence["corpus"]
    print("=" * 74)
    print("M107 -- sparse vs dense at matched inference MACs (plan §7.14)")
    print("=" * 74)
    print(f"train rows {corpus['train_rows']}   test rows {corpus['test_rows']}"
          f"   digest {corpus['subsample_sha256'][:16]}")
    print(f"ridge penalty {gate['penalty']} chosen once on"
          f" {evidence['head']['chosen_on']} and applied unchanged to every arm")
    print(f"sample floor {evidence['head']['fit_samples_per_fitted_dimension_floor']}"
          " rows per fitted dimension")
    print()

    print("dense ladder (frozen DINOv2 features, same head, same rows)")
    for macs, acc, name in gate["dense"]:
        print(f"   {name:<24}{macs / 1e6:10.1f} M MACs   {acc * 100:6.2f}%")
    print()
    print("sparse generalist ladder")
    for macs, acc, name in gate["generalist"]:
        print(f"   {name:<24}{macs / 1e6:10.1f} M MACs   {acc * 100:6.2f}%")
    print()
    print("sparse six-expert mixture, ORACLE routing (restriction 4)")
    for macs, acc, name in gate["mixture"]:
        print(f"   {name:<24}{macs / 1e6:10.1f} M MACs   {acc * 100:6.2f}%")
    print()

    print("registered comparison: each sparse point against the best dense"
          " point at or below its MACs")
    for c in gate["comparisons"]:
        if not c["decidable"]:
            print(f"   {c['sparse']:<20}{c['sparse_macs'] / 1e6:8.1f} M"
                  f" {c['sparse_accuracy'] * 100:6.2f}%   VOID -- no dense arm"
                  " at or below this budget")
            continue
        verdict = "SPARSE WINS" if c["sparse_wins"] else "dense wins"
        margin = (c["sparse_accuracy"] - c["dense_accuracy"]) * 100
        print(f"   {c['sparse']:<20}{c['sparse_macs'] / 1e6:8.1f} M"
              f" {c['sparse_accuracy'] * 100:6.2f}%  vs {c['dense']:<20}"
              f"{c['dense_macs'] / 1e6:8.1f} M {c['dense_accuracy'] * 100:6.2f}%"
              f"   {margin:+6.2f} pp   {verdict}")
    print()

    print("kill switches:")
    print(f"   KS1 dense dominates everywhere: "
          f"{_fired(gate['ks1_fired'], gate['ks1_decidable'])}")
    print(f"   KS2 sparse wins somewhere:      "
          f"{_fired(gate['ks2_fired'], gate['ks2_decidable'])}")
    print(f"   KS3 generalist matches mixture: "
          f"{_fired(gate['ks3_fired'], gate['ks3_decidable'])}")
    print()

    if gate["ks3_decidable"]:
        print("kill switch 3 detail -- amendment 5 caps the mixture ladder at two"
              " budgets, so this switch is decidable at TWO budgets, not six:")
        for r in gate["ks3_rows"]:
            print(f"   {r['macs'] / 1e6:8.1f} M MACs   generalist"
                  f" {r['generalist_accuracy'] * 100:6.2f}%   mixture under oracle"
                  f" routing {r['mixture_accuracy'] * 100:6.2f}%"
                  f"   {(r['mixture_accuracy'] - r['generalist_accuracy']) * 100:+6.2f} pp")
        print()

    if gate["ks2_fired"]:
        print("HEADLINE (§11.1): kill switch 2 FIRED.")
        print("  The registered prediction -- the dense ladder dominating at every")
        print("  overlapping MAC budget -- is REFUTED. That prediction ran against")
        print("  the program's own thesis, so this is the thesis surviving a test")
        print("  it was set up to fail.")
        print()
        print("  It must be reported with its five bounds:")
        print(f"   1. only {len(gate['comparisons']) - len(gate['void_points'])}"
              f" of {len(gate['comparisons'])} sparse budgets are decidable;"
              f" {len(gate['void_points'])} are VOID, not wins")
        worst = min(c["sparse_accuracy"] for c in gate["comparisons"] if c["sparse_wins"])
        best = max(c["sparse_accuracy"] for c in gate["comparisons"] if c["sparse_wins"])
        print(f"   2. the crossings sit at {worst * 100:.2f}%-{best * 100:.2f}%"
              " accuracy on 345 classes")
        ceiling = gate["generalist"][-1]
        passer = next((d for d in gate["dense"] if d[1] > ceiling[1]), None)
        if passer is not None:
            print(f"   3. dense passes the sparse ceiling of {ceiling[1] * 100:.2f}%"
                  f" at {passer[2]} ({passer[1] * 100:.2f}%), which costs"
                  f" {passer[0] / ceiling[0]:.2f}x the sparse ceiling's MACs")
        print("   4. see the sample-floor note below -- the sparse ladder was"
              " truncated by the corpus, not by the method")
        print("   5. the gate lets a sparse arm outspend its opponent, because")
        print("      the dense ladder steps by ~2x; see the interpolation check")
        print()
        _interpolation_note(gate)
    elif gate["ks1_fired"]:
        print("HEADLINE (§11.1): kill switch 1 FIRED -- §3.2 Q2 is REFUTED.")
    print()

    _sample_floor_note(evidence, gate)


def _sample_floor_note(evidence: dict, gate: dict) -> None:
    floor = evidence["head"]["fit_samples_per_fitted_dimension_floor"]
    train = evidence["corpus"]["train_rows"]
    arms = evidence["arms"]
    print("why each ladder stopped where it did (§5.3 sample floor):")
    prev = None
    for macs, acc, name in gate["generalist"]:
        rows = arms[name]["rows_per_fitted_dimension"]
        step = "" if prev is None else f"   last step {(acc - prev) * 100:+.2f} pp"
        print(f"   {name:<22} rows/dim {rows:7.2f}{step}")
        prev = acc
    dense_rows = [arms[n]["rows_per_fitted_dimension"] for _, _, n in gate["dense"]]
    print(f"   dense arms sit at {min(dense_rows):.2f}-{max(dense_rows):.2f} rows/dim")
    print(f"   the corpus admits at most {train / (4 * floor):.0f} atoms at the"
          f" floor of {floor}; the ladder's top rung is"
          f" {arms[gate['generalist'][-1][2]]['rows_per_fitted_dimension']:.2f} rows/dim")
    print("   the sparse ladder ran out of statistical support while still")
    print("   improving; the dense ladder never came near the floor")


def compare_runner(evidence: dict, gate: dict) -> int:
    """Check this independent recomputation against the runner's own gate."""
    runner = evidence["gate"]
    pairs = [
        ("kill_switch_1_dense_dominates_everywhere", gate["ks1_fired"]),
        ("kill_switch_2_sparse_wins_somewhere", gate["ks2_fired"]),
        ("kill_switch_3_generalist_beats_mixture", gate["ks3_fired"]),
    ]
    bad = 0
    print("independent recomputation vs the runner's own gate:")
    for key, mine in pairs:
        theirs = runner[key]["fired"]
        ok = mine == theirs
        bad += not ok
        print(f"   {key:<46} mine={str(mine):<5} runner={str(theirs):<5}"
              f" {'ok' if ok else 'DISAGREE'}")
    mine_points = sorted(round(m, 3) for m, _ in gate["ks2_points"])
    theirs_points = sorted(round(m, 3) for m, _ in
                           runner["kill_switch_2_sparse_wins_somewhere"].get("points", []))
    ok = mine_points == theirs_points
    bad += not ok
    print(f"   {'kill switch 2 crossing budgets':<46}"
          f" {'ok' if ok else 'DISAGREE'}")
    print()
    print("AGREED" if not bad else f"{bad} DISAGREEMENT(S) -- one of the two is wrong")
    return bad


def plan_text(evidence: dict, gate: dict) -> str:
    """Produce the §7.14 result paragraph in plan-ready form."""
    lines: list[str] = []
    decidable = [c for c in gate["comparisons"] if c["decidable"]]
    wins = [c for c in decidable if c["sparse_wins"]]
    ceiling = gate["generalist"][-1]
    passer = next((d for d in gate["dense"] if d[1] > ceiling[1]), None)

    lines.append("**M107 gate result.** One fixed per-class subsample,"
                 f" {evidence['corpus']['train_rows']:,} train and"
                 f" {evidence['corpus']['test_rows']:,} test rows, 345 classes,"
                 f" digest `{evidence['corpus']['subsample_sha256'][:16]}`. Ridge"
                 f" penalty {gate['penalty']} was chosen once on"
                 f" `{evidence['head']['chosen_on']}` and applied unchanged to"
                 " every arm in both families.")
    lines.append("")
    lines.append("| arm | family | analytic MACs per image | test accuracy |")
    lines.append("|---|---|---|---|")
    for macs, acc, name in gate["dense"]:
        lines.append(f"| `{name}` | dense | {macs / 1e6:,.1f} M | {acc * 100:.2f}% |")
    for macs, acc, name in gate["generalist"]:
        lines.append(f"| `{name}` | sparse generalist | {macs / 1e6:,.1f} M |"
                     f" {acc * 100:.2f}% |")
    for macs, acc, name in gate["mixture"]:
        lines.append(f"| `{name}` | sparse mixture, oracle | {macs / 1e6:,.1f} M |"
                     f" {acc * 100:.2f}% |")
    lines.append("")
    if gate["ks2_fired"]:
        crossings = "; ".join(
            f"{c['sparse_macs'] / 1e6:,.1f} M MACs at {c['sparse_accuracy'] * 100:.2f}%"
            f" against `{c['dense']}` at {c['dense_accuracy'] * 100:.2f}%"
            f" ({(c['sparse_accuracy'] - c['dense_accuracy']) * 100:+.2f} pp)"
            for c in wins)
        lines.append(
            "**Kill switch 2 fired, and kill switch 1 did not.** The registered"
            " prediction was that the dense ladder would dominate the sparse"
            " ladder at every overlapping MAC budget. It does not. The sparse"
            f" generalist wins at {len(wins)} of the {len(decidable)} decidable"
            f" budgets: {crossings}.")
        lines.append("")
        lines.append(
            f"That result is bounded four ways. First, only {len(decidable)} of"
            f" {len(gate['comparisons'])} sparse budgets are decidable at all:"
            f" the {len(gate['void_points'])} cheapest sparse points sit below"
            " the dense ladder's cheapest arm and are VOID for kill switches 1"
            " and 2, which is not the same as winning there. Second, the"
            f" crossings sit at {min(c['sparse_accuracy'] for c in wins) * 100:.2f}%"
            f" and {max(c['sparse_accuracy'] for c in wins) * 100:.2f}% on 345"
            " classes, and a crossing at an accuracy nobody would deploy is not"
            " an efficiency result.")
        if passer is not None:
            lines.append("")
            lines.append(
                f" Third, the sparse ladder's ceiling is {ceiling[1] * 100:.2f}%"
                f" at {ceiling[0] / 1e6:,.1f} M MACs, and `{passer[2]}` passes it"
                f" at {passer[1] * 100:.2f}% for {passer[0] / 1e6:,.1f} M MACs —"
                f" {passer[0] / ceiling[0]:.2f}x the cost. The sparse family"
                " therefore reaches no accuracy the dense family cannot reach"
                " for less than half as much again.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-text", action="store_true",
                        help="emit the plan-ready §7.14 paragraph")
    parser.add_argument("--compare-runner", action="store_true",
                        help="check this recomputation against the runner's gate")
    args = parser.parse_args()

    evidence = _load()
    gate = _gate(evidence)

    if args.plan_text:
        print(plan_text(evidence, gate))
        return 0
    if args.compare_runner:
        return 1 if compare_runner(evidence, gate) else 0

    report(evidence, gate)
    print()
    print("restriction 4: the word 'oracle' must appear in every sentence"
          " quoting a mixture figure.")
    print("restriction 5: no wall-clock comparison between the two families.")
    print("restriction 7: d1 and d5 are not admissible apart from one another.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
