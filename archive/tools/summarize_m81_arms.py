"""Summarise the M81 evidence: per-width arm tables, seed spread, and the
budget-versus-I5 trade that the R4 conjunction verdict rests on."""

import json
from statistics import mean

d = json.load(open("logs/results/v13/m81_sparse_head/evidence.json"))
seeds = d["seeds"]
print(f"seeds {[s['seed'] for s in seeds]}   "
      f"elapsed {max(s['elapsed_seconds'] for s in seeds)/60:.1f} min (slowest worker)")
print(f"i5-8 classes: {d['i5_eight_classes']}")
print()


def fmt(value, width=7, scale=100.0, suffix="%"):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value * scale:>{width}.2f}{suffix}"


for width_name in ("i5_8", "i5_128"):
    widths = [w for s in seeds for w in s["widths"] if w["width"] == width_name]
    chance = widths[0]["chance_accuracy"]
    print(f"=== {width_name}   {widths[0]['class_count']} classes   "
          f"chance {chance*100:.3f}%   "
          f"eval rows {widths[0]['evaluation_rows']} ===")
    header = (f"{'arm':>34} {'acc':>8} {'I5 mean':>9} {'I5 sd':>7} "
              f"{'null':>8} {'margin':>8} {'cited':>7} {'params':>10}")
    print(header)
    print("-" * len(header))
    names = [a["arm"] for a in widths[0]["arms"]]
    for name in names:
        arms = [a for w in widths for a in w["arms"] if a["arm"] == name]
        scores = [a["i5"]["probe_balanced_accuracy"] for a in arms]
        measured = [s for s in scores if s is not None]
        nulls = [
            a["i5_shuffled_null"]["probe_balanced_accuracy"] for a in arms
        ]
        nulls = [n for n in nulls if n is not None]
        spread = (max(measured) - min(measured)) if len(measured) > 1 else 0.0
        margins = [a["i5_margin_over_null"] for a in arms]
        margins = [m for m in margins if m is not None]
        cited = mean(a["explanation_length"]["mean_active_atoms"] for a in arms)
        flag = "" if not measured else (" *" if cited <= 10 else "")
        print(
            f"{name:>34} "
            f"{mean(a['balanced_accuracy'] for a in arms)*100:>7.2f}% "
            f"{fmt(mean(measured) if measured else None, 8)} "
            f"{spread*100:>6.2f} "
            f"{fmt(mean(nulls) if nulls else None)} "
            f"{(mean(margins)*100 if margins else 0.0):>+7.2f} "
            f"{cited:>7.1f} "
            f"{(arms[0]['active_parameters'] or 0):>10}{flag}"
        )
    print("  * meets the 10-atom deployment budget")
    print()

gate = d["gate"]
for key in ("i5_8", "i5_128"):
    block = gate[key]
    print(f"--- {key} ---")
    print(f"  best arm within budget : {block['best_atom_arm_per_seed']}")
    print(f"  I5 mean                : {fmt(block['i5_mean'], 6)}  "
          f"spread {fmt(block['i5_spread'], 5)}")
    print(f"  kNN control            : {fmt(block['knn_control_mean'], 6)}")
    print(f"  shuffled null          : {fmt(block['shuffled_null_mean'], 6)}")
    print(f"  unconstrained best     : {block['unconstrained_best_arm_per_seed']}")
    print(f"    its I5               : {fmt(block['unconstrained_i5_mean'], 6)}  "
          f"citing {block['unconstrained_cited_atoms_mean']:.1f} atoms")
    print(f"  within budget per seed : {block['within_explanation_budget']}")
    if "verdict" in block:
        print(f"  verdict                : {block['verdict']}")
    if "passes" in block:
        print(f"  passes                 : {block['passes']}")
    print()

print(f"conjunction verdict      : {gate['conjunction_verdict']}")
print(f"H81 gate passed          : {gate['h81_gate_passed']}")
print(f"dominance claim blocked  : {gate['dominance_claim_blocked']}")
print(f"all atom arms beat null  : {gate['all_atom_arms_beat_own_null']}")
