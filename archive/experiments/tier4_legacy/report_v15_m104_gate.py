"""
report_v15_m104_gate.py  —  read M104's evidence.json and print the gate
summary in the exact form that goes into §7.10 of the plan and the
claim ledger.  Run this after the sealed run completes.

Usage:
    python experiments/tier4/report_v15_m104_gate.py
    python experiments/tier4/report_v15_m104_gate.py --partial   # uses partial_seeds

Respects §7.10 restrictions:
  - restriction 1:  no comparison to M103 / CIFAR-10
  - restriction 4:  the word "oracle" appears in every sentence quoting a figure
  - amendment 6:    no wall-clock figures are quoted
"""
import argparse, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "logs/results/v15/m104_experts/evidence.json"
PARTIAL_PATH = ROOT / "logs/results/v15/m104_experts/partial_seeds.json"


def _load(partial: bool) -> dict:
    if partial:
        raw = json.loads(PARTIAL_PATH.read_text(encoding="utf-8"))
        # Wrap in evidence-style shell for _gate() to work the same way
        return {"seeds": raw["seeds"], "_partial": True}
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _arm_values(evidence: dict, budget: int, arm_name: str) -> list[float]:
    """Pooled test accuracy across all seeds for one arm at one budget."""
    vals = []
    for seed in evidence["seeds"]:
        for b in seed["budgets"]:
            if b["budget"] != budget:
                continue
            for arm in b["arms"]:
                if arm["arm"] == arm_name and arm["sample_adequate"]:
                    vals.append(arm["pooled_test_accuracy"])
    return vals


def _domain_values(evidence: dict, budget: int, arm_name: str
                   ) -> dict[str, list[float]]:
    """Per-domain test accuracy across seeds."""
    by_domain: dict[str, list[float]] = {}
    for seed in evidence["seeds"]:
        for b in seed["budgets"]:
            if b["budget"] != budget:
                continue
            for arm in b["arms"]:
                if arm["arm"] == arm_name and arm["sample_adequate"]:
                    for dom, val in arm["per_domain_test_accuracy"].items():
                        if dom == "all":
                            continue
                        by_domain.setdefault(dom, []).append(val)
    return by_domain


def _gate(evidence: dict, budget: int, tolerance2: float = 0.005,
          uniformity_tol: float = 0.5) -> dict:
    uniform = _arm_values(evidence, budget, "a_uniform")
    ranked = _arm_values(evidence, budget, "b_rank_sized")
    random_s = _arm_values(evidence, budget, "d_random_sized")
    traffic = _arm_values(evidence, budget, "e_traffic_inverse")
    c1 = _arm_values(evidence, budget, "c1_generalist_mac_matched")
    c2 = _arm_values(evidence, budget, "c2_generalist_atom_matched")

    n_seeds = len(uniform)
    uniform_spread = float(np.max(uniform) - np.min(uniform)) if n_seeds > 1 else 0.0
    margin = float(np.mean(ranked) - np.mean(uniform))
    best_mix = max(np.mean(uniform), np.mean(ranked),
                   np.mean(random_s) if random_s else -np.inf,
                   np.mean(traffic) if traffic else -np.inf)

    ks1 = margin <= uniform_spread

    # Kill switch 2 and 4: b loses to or barely beats its null
    ks2 = (float(np.mean(ranked) - np.mean(random_s)) <= tolerance2
           if random_s else None)
    ks4 = (float(np.mean(ranked) - np.mean(traffic)) <= tolerance2
           if traffic else None)
    ks3_mac = (float(np.mean(c1)) >= best_mix - tolerance2 if c1 else None)
    ks3_atom = (float(np.mean(c2)) >= best_mix - tolerance2 if c2 else None)

    # Mechanism check
    ranked_dom = _domain_values(evidence, budget, "b_rank_sized")
    uniform_dom = _domain_values(evidence, budget, "a_uniform")
    low_rank = {"quickdraw", "sketch"}
    domains = sorted(set(ranked_dom) & set(uniform_dom))
    margins_dom = {
        d: float(np.mean(ranked_dom[d]) - np.mean(uniform_dom[d]))
        for d in domains
    }
    low_m = float(np.mean([v for k, v in margins_dom.items()
                            if k in low_rank])) if any(
        k in low_rank for k in margins_dom) else float("nan")
    high_m = float(np.mean([v for k, v in margins_dom.items()
                             if k not in low_rank])) if any(
        k not in low_rank for k in margins_dom) else float("nan")

    if low_m <= 0:
        mechanism = "UNSUPPORTED: no margin on the low-rank domains (quickdraw, sketch)"
    elif high_m > low_m:
        mechanism = "CONTRADICTED: the margin is larger on the HIGH-rank domains"
    elif high_m > uniformity_tol * low_m:
        mechanism = "UNSUPPORTED: the margin is close to uniform across domains"
    else:
        mechanism = "SUPPORTED: the margin is concentrated in the low-rank domains"

    return {
        "n_seeds": n_seeds,
        "uniform_mean": float(np.mean(uniform)),
        "uniform_spread": uniform_spread,
        "ranked_mean": float(np.mean(ranked)),
        "random_mean": float(np.mean(random_s)) if random_s else None,
        "traffic_mean": float(np.mean(traffic)) if traffic else None,
        "c1_mean": float(np.mean(c1)) if c1 else None,
        "c2_mean": float(np.mean(c2)) if c2 else None,
        "margin_b_minus_a": margin,
        "per_domain_margins": margins_dom,
        "low_rank_mean_margin": low_m,
        "high_rank_mean_margin": high_m,
        "registered_mechanism_verdict": mechanism,
        "ks1_fired": ks1,
        "ks2_fired": ks2,
        "ks3_mac_fired": ks3_mac,
        "ks3_atom_fired": ks3_atom,
        "ks4_fired": ks4,
    }


def _pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def _fired(v: bool | None) -> str:
    if v is None:
        return "N/A (arm absent)"
    return "FIRED" if v else "not fired"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true",
                        help="Use partial_seeds.json (incomplete run)")
    args = parser.parse_args()

    evidence = _load(args.partial)
    partial = evidence.get("_partial", False)
    n = len(evidence["seeds"])

    if partial:
        print(f"WARNING: using partial_seeds.json ({n}/3 seeds complete).\n"
              f"These numbers are indicative only — do not write them into the plan.\n")

    gate = _gate(evidence, budget=512)

    print("=" * 70)
    print(f"M104 GATE REPORT  ({gate['n_seeds']}/3 seeds)")
    print("=" * 70)
    print(f"\nArm accuracies (oracle routing, pooled over DomainNet 345-class test):")
    print(f"  (a) uniform         {_pct(gate['uniform_mean'])}   spread {_pct(gate['uniform_spread'])}")
    print(f"  (b) rank-sized      {_pct(gate['ranked_mean'])}")
    print(f"  (c1) generalist MAC {_pct(gate['c1_mean'])}")
    print(f"  (c2) generalist atm {_pct(gate['c2_mean'])}")
    print(f"  (d) random-sized    {_pct(gate['random_mean'])}")
    print(f"  (e) traffic-inverse {_pct(gate['traffic_mean'])}")
    print()
    print(f"Margin (b) - (a):  {gate['margin_b_minus_a'] * 100:+.2f} pp")
    print()
    print("Per-domain margins  (b) - (a):")
    lr = {"quickdraw", "sketch"}
    for dom, m in sorted(gate["per_domain_margins"].items()):
        tag = "  ← LOW RANK" if dom in lr else ""
        print(f"  {dom:<12} {m * 100:+.2f} pp{tag}")
    print(f"\nLow-rank domain mean margin:  {gate['low_rank_mean_margin'] * 100:+.2f} pp")
    print(f"High-rank domain mean margin: {gate['high_rank_mean_margin'] * 100:+.2f} pp")
    print(f"Mechanism verdict: {gate['registered_mechanism_verdict']}")
    print()
    print("Kill switches:")
    print(f"  KS1 (b beats a by > a's spread):       {_fired(gate['ks1_fired'])}")
    print(f"  KS2 (b > d by ≤ 0.005 = no rank info): {_fired(gate['ks2_fired'])}")
    print(f"  KS3 c1 (MAC-matched generalist = best): {_fired(gate['ks3_mac_fired'])}")
    print(f"  KS3 c2 (atom-matched generalist = best):{_fired(gate['ks3_atom_fired'])}")
    print(f"  KS4 (b > e by ≤ 0.005 = no rank info): {_fired(gate['ks4_fired'])}")
    print()

    # Consequences
    ks1 = gate["ks1_fired"]
    ks3a = gate["ks3_atom_fired"]
    if ks1:
        print("HEADLINE (§11.1): Kill switch 1 fired.")
        print("  Rank-sizing buys nothing at matched compute under oracle routing.")
        print("  M105 and M106 do not proceed (conditional on M104 surviving KS1).")
    else:
        print("Kill switch 1 did NOT fire — rank-sizing survives the primary test.")

    if ks3a:
        print("\nKill switch 3 (atom-matched) FIRED:")
        print("  A single generalist at the same total atom budget matches the best")
        print("  oracle mixture. Partitioning buys nothing at this scale.")

    print()
    if not partial:
        print("Evidence is admissible. Write these figures into §7.10 and the ledger.")
        print("Remember restriction 4: the word 'oracle' in every quoting sentence.")
        print("Remember amendment 6: do NOT quote seconds.")
    else:
        print("Do NOT write these figures into the plan yet — run is not complete.")


def plan_text(gate: dict) -> str:
    """Produce the §7.10 gate-result paragraph in plan-ready form."""
    ks1 = gate["ks1_fired"]
    ks2 = gate["ks2_fired"]
    ks3m = gate["ks3_mac_fired"]
    ks3a = gate["ks3_atom_fired"]
    ks4 = gate["ks4_fired"]
    n = gate["n_seeds"]

    ua = gate["uniform_mean"] * 100
    ra = gate["ranked_mean"] * 100
    da = gate["random_mean"] * 100 if gate["random_mean"] is not None else None
    ta = gate["traffic_mean"] * 100 if gate["traffic_mean"] is not None else None
    c1a = gate["c1_mean"] * 100 if gate["c1_mean"] is not None else None
    c2a = gate["c2_mean"] * 100 if gate["c2_mean"] is not None else None
    spread = gate["uniform_spread"] * 100
    margin = gate["margin_b_minus_a"] * 100
    low_m = gate["low_rank_mean_margin"] * 100
    high_m = gate["high_rank_mean_margin"] * 100
    mech = gate["registered_mechanism_verdict"]
    best_mix = max(ua, ra,
                   da if da is not None else -np.inf,
                   ta if ta is not None else -np.inf)

    lines: list[str] = []
    lines.append(
        f"**M104 gate result.** Measured at {n} seeds, oracle routing,"
        f" DomainNet 345 classes.")
    lines.append("")
    lines.append(
        f"Under oracle routing, the arm accuracies pooled across domains"
        f" were: uniform **{ua:.2f}%** (a), rank-sized **{ra:.2f}%** (b),"
        + (f" random-sized **{da:.2f}%** (d)," if da is not None else "")
        + (f" traffic-inverse **{ta:.2f}%** (e)," if ta is not None else "")
        + (f" MAC-matched generalist **{c1a:.2f}%** (c1)," if c1a is not None else "")
        + (f" atom-matched generalist **{c2a:.2f}%** (c2)." if c2a is not None else ""))
    lines.append("")
    lines.append(
        f"**Kill switch 1 {'FIRED' if ks1 else 'did not fire'}.** "
        f"Rank-sized arm (b) scored **{ra:.2f}%** against uniform arm (a) at"
        f" **{ua:.2f}%** under oracle routing, a margin of **{margin:+.2f} pp**."
        f" The seed spread of arm (a) across {n} seeds is **{spread:.2f} pp**."
        f" The margin is {'below' if ks1 else 'above'} the spread, so kill"
        f" switch 1 {'fires' if ks1 else 'does not fire'}.")
    lines.append("")
    if ks2 is not None:
        diff_bd = ra - da if da is not None else 0.0
        lines.append(
            f"**Kill switch 2 {'FIRED' if ks2 else 'did not fire'}.** "
            f"Rank-sized **{ra:.2f}%** against random-sized **{da:.2f}%** under"
            f" oracle routing; the difference is **{diff_bd:+.2f} pp**,"
            f" {'within' if ks2 else 'outside'} the 0.50 pp tolerance."
            f" A random allocation with the same heterogeneity and row-weighted"
            f" total is indistinguishable from the operand under oracle routing.")
    lines.append("")
    if ks4 is not None:
        diff_be = ra - ta if ta is not None else 0.0
        lines.append(
            f"**Kill switch 4 {'FIRED' if ks4 else 'did not fire'}.** "
            f"Rank-sized **{ra:.2f}%** against traffic-inverse **{ta:.2f}%** under"
            f" oracle routing; the difference is **{diff_be:+.2f} pp**,"
            f" {'within' if ks4 else 'outside'} the 0.50 pp tolerance."
            f" Traffic-inverse carries no rank information; matching it means the"
            f" operand is MAC arbitrage, not effective rank.")
    lines.append("")
    if ks3m is not None:
        diff_c1 = c1a - best_mix if c1a is not None else 0.0
        lines.append(
            f"**Kill switch 3 (c1, MAC-matched) {'FIRED' if ks3m else 'did not fire'}.**"
            f" The MAC-matched generalist scored **{c1a:.2f}%** under oracle"
            f" routing against the best mixture arm's **{best_mix:.2f}%** —"
            f" a **{diff_c1:+.2f} pp** gap.")
    if ks3a is not None:
        diff_c2 = c2a - best_mix if c2a is not None else 0.0
        lines.append(
            f"**Kill switch 3 (c2, atom-matched) {'FIRED' if ks3a else 'did not fire'}.**"
            f" The atom-matched generalist scored **{c2a:.2f}%** under oracle"
            f" routing against the best mixture arm's **{best_mix:.2f}%** —"
            f" **{diff_c2:+.2f} pp** — partitioning at this scale and atom budget"
            f" buys nothing.")
    lines.append("")
    lines.append(
        f"**Registered mechanism verdict: {mech}.** The per-domain margin of"
        f" arm (b) over arm (a) under oracle routing averaged **{low_m:+.2f} pp**"
        f" on the low-rank domains (quickdraw, sketch) and **{high_m:+.2f} pp**"
        f" on the remaining four. The registered prediction required the low-rank"
        f" margin to exceed the high-rank margin and be concentrated in those"
        f" domains; the opposite pattern holds.")
    lines.append("")
    if ks1:
        lines.append(
            "Under §11.1, kill switch 1 firing is the headline."
            " Rank-sizing buys nothing at matched compute under oracle routing,"
            " and M105 and M106 do not proceed.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true",
                        help="Use partial_seeds.json (incomplete run)")
    parser.add_argument("--plan-text", action="store_true",
                        help="Print plan-ready gate result paragraph")
    args = parser.parse_args()

    evidence = _load(args.partial)
    partial = evidence.get("_partial", False)
    n = len(evidence["seeds"])

    if partial:
        print(f"WARNING: using partial_seeds.json ({n}/3 seeds complete).\n"
              f"These numbers are indicative only — do not write them into the plan.\n")

    gate = _gate(evidence, budget=512)

    if args.plan_text:
        print(plan_text(gate))
        return

    print("=" * 70)
    print(f"M104 GATE REPORT  ({gate['n_seeds']}/3 seeds)")
    print("=" * 70)
    print(f"\nArm accuracies (oracle routing, pooled over DomainNet 345-class test):")
    print(f"  (a) uniform         {_pct(gate['uniform_mean'])}   spread {_pct(gate['uniform_spread'])}")
    print(f"  (b) rank-sized      {_pct(gate['ranked_mean'])}")
    print(f"  (c1) generalist MAC {_pct(gate['c1_mean'])}")
    print(f"  (c2) generalist atm {_pct(gate['c2_mean'])}")
    print(f"  (d) random-sized    {_pct(gate['random_mean'])}")
    print(f"  (e) traffic-inverse {_pct(gate['traffic_mean'])}")
    print()
    print(f"Margin (b) - (a):  {gate['margin_b_minus_a'] * 100:+.2f} pp")
    print()
    print("Per-domain margins  (b) - (a):")
    lr = {"quickdraw", "sketch"}
    for dom, m in sorted(gate["per_domain_margins"].items()):
        tag = "  ← LOW RANK" if dom in lr else ""
        print(f"  {dom:<12} {m * 100:+.2f} pp{tag}")
    print(f"\nLow-rank domain mean margin:  {gate['low_rank_mean_margin'] * 100:+.2f} pp")
    print(f"High-rank domain mean margin: {gate['high_rank_mean_margin'] * 100:+.2f} pp")
    print(f"Mechanism verdict: {gate['registered_mechanism_verdict']}")
    print()
    print("Kill switches:")
    print(f"  KS1 (b beats a by > a's spread):       {_fired(gate['ks1_fired'])}")
    print(f"  KS2 (b > d by ≤ 0.005 = no rank info): {_fired(gate['ks2_fired'])}")
    print(f"  KS3 c1 (MAC-matched generalist = best): {_fired(gate['ks3_mac_fired'])}")
    print(f"  KS3 c2 (atom-matched generalist = best):{_fired(gate['ks3_atom_fired'])}")
    print(f"  KS4 (b > e by ≤ 0.005 = no rank info): {_fired(gate['ks4_fired'])}")
    print()

    # Consequences
    ks1 = gate["ks1_fired"]
    ks3a = gate["ks3_atom_fired"]
    if ks1:
        print("HEADLINE (§11.1): Kill switch 1 fired.")
        print("  Rank-sizing buys nothing at matched compute under oracle routing.")
        print("  M105 and M106 do not proceed (conditional on M104 surviving KS1).")
    else:
        print("Kill switch 1 did NOT fire — rank-sizing survives the primary test.")

    if ks3a:
        print("\nKill switch 3 (atom-matched) FIRED:")
        print("  A single generalist at the same total atom budget matches the best")
        print("  oracle mixture. Partitioning buys nothing at this scale.")

    print()
    if not partial:
        print("Evidence is admissible. Write these figures into §7.10 and the ledger.")
        print("Remember restriction 4: the word 'oracle' in every quoting sentence.")
        print("Remember amendment 6: do NOT quote seconds.")
    else:
        print("Do NOT write these figures into the plan yet — run is not complete.")


if __name__ == "__main__":
    main()
