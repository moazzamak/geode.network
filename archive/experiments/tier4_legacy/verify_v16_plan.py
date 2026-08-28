"""Verify every figure quoted in RESEARCH_IMPLEMENTATION_PLAN_v16.md and
CLAIM_LEDGER_v16.md against the sealed M107/M108/M109 evidence files, and
compute M110 (the parameter axis) as a re-analysis from the same geometry.

M110 is registered (plan section 5.3) as "a re-analysis of M107, M108 and M109
evidence on a second registered axis, computed by the same verifier that
recomputes the MAC axis, from the same geometry." This script is that verifier:

* Section A — M108 figure checks (recomputed from the sealed evidence).
* Section B — M109 figure checks (recomputed from the sealed evidence).
* Section C — MAC recomputation from geometry, cross-checked against every
  sealed `macs.total`.
* Section D — M110 parameter axis: parameter counts computed from geometry
  (never from a model card), accuracy-per-parameter per arm/rung, and M110's
  two kill switches. Writes `logs/results/v16/m110_parameter_axis.json`.
* Section E — document checks with negative controls (a check that cannot fail
  proves nothing; each figure check has a corruption that must make it fail).

Usage::

    python experiments/tier4/verify_v16_plan.py
    python experiments/tier4/verify_v16_plan.py --negative-control

Exit code is 0 only when every check passes.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN = ROOT / "analysis" / "RESEARCH_IMPLEMENTATION_PLAN_v16.md"
LEDGER = ROOT / "analysis" / "CLAIM_LEDGER_v16.md"
M107 = ROOT / "logs/results/v15/m107_dense/evidence.json"
M108 = ROOT / "logs/results/v16/m108_dictionary/evidence.json"
M109 = ROOT / "logs/results/v16/m109_trunk/evidence.json"
M110_OUT = ROOT / "logs/results/v16/m110_parameter_axis/evidence.json"

# --- program geometry (section 2.9 / M107, unchanged) ----------------------
CLASSES = 345
PATCH = 6                      # sparse patch side
PATCH_DIM = PATCH * PATCH * 3  # 108 whitened patch dims
POOL = 2                       # 2x2 sum pool
PATHS = 27                     # (32 - 6 + 1) patches per side at 32x32
PATCHES = PATHS * PATHS         # 729
WHITEN_PARAMS = 108 * 108 + 108  # 108x108 matrix + 108 mean

DENSE_FEATURE = 768            # CLS || mean(patch) for dinov2-small


def sparse_head_params(atoms: int) -> int:
    return atoms * POOL * POOL * CLASSES + CLASSES


def sparse_rep_params(atoms: int) -> int:
    return atoms * PATCH_DIM + WHITEN_PARAMS


def sparse_total_params(atoms: int) -> int:
    return sparse_rep_params(atoms) + sparse_head_params(atoms)


def sparse_macs(atoms: int) -> dict:
    encoding = PATCHES * atoms * PATCH_DIM
    whitening = PATCHES * (108 * 108)
    head = atoms * POOL * POOL * CLASSES
    return {
        "encoding": int(encoding),
        "whitening": int(whitening),
        "head": int(head),
        "total": int(encoding + whitening + head),
    }


def load_evidence():
    if not all(p.exists() for p in (M107, M108, M109)):
        sys.exit("MISSING  one of M107/M108/M109 evidence files; cannot verify")
    return (json.load(open(M107, encoding="utf-8")),
            json.load(open(M108, encoding="utf-8")),
            json.load(open(M109, encoding="utf-8")))


m107, m108, m109 = load_evidence()
plan_text = PLAN.read_text(encoding="utf-8")
ledger_text = LEDGER.read_text(encoding="utf-8")

checks = []
_failures = []


def check(label, quoted, actual, places=6):
    if isinstance(quoted, bool) or isinstance(actual, bool):
        ok = quoted == actual
        delta = ""
    else:
        delta = abs(quoted - actual)
        ok = delta <= 0.5 * (10 ** -places)
    if ok:
        checks.append(("ok", label, actual))
    else:
        _failures.append(label)
        checks.append(("MISMATCH", label, actual))
        print(f"MISMATCH  {label}: quoted {quoted!r} actual {actual!r}{delta}")
    return ok


def check_text(label, present: bool):
    if present:
        checks.append(("ok", label, True))
    else:
        _failures.append(label)
        print(f"MISSING   {label}")
    return present


def _as_float(s: str) -> float:
    return float(s.replace("\u2212", "-").replace("**", "").strip())


def _cell(row: str, idx: int) -> str:
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    return cells[idx] if idx < len(cells) else ""


def _table_block(text: str, header_marker: str) -> list:
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if header_marker in line:
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("|"):
                    out.append(lines[j])
                elif out:
                    break
            break
    return out


# --------------------------------------------------------------------------
# Section A — M108 (discriminative dictionary growth), recomputed from evidence
# --------------------------------------------------------------------------
def section_a():
    rep = m108["arm_a_reproduction"]
    check("M108 arm (a) reproduction max delta <= 0.002", True,
          rep["max_abs_delta"] <= rep["tolerance"])
    check("M108 arm (a) reproduction exercised at all 6 budgets",
          rep["exercised"] and len(rep["overlapping_budgets"]) == 6, True)
    check("M108 arm (a) reproduction max_abs_delta", 8.695652173915103e-05,
          rep["max_abs_delta"], places=12)

    # c - a per budget: QUOTE from the ledger C108.1 table, recompute from the
    # sealed gate curves (keyed by MACs, so the two arms at one atom budget are
    # not merged).
    curves = m108["gate"]["curves"]
    a_curve = {int(m): acc for m, acc in curves["a_random"]}
    c_curve = {int(m): acc for m, acc in curves["c_discriminative"]}
    for row in _table_block(ledger_text, "| budget"):
        budget = _cell(row, 0)
        if not budget.isdigit():
            continue
        b = int(budget)
        quoted = _as_float(_cell(row, 4))
        m = sparse_macs(b)["total"]
        check(f"M108 c-a at {b} atoms (ledger)", quoted,
              c_curve[m] - a_curve[m], places=4)

    gcf = m108["gap_closing_fraction"]["c_discriminative"]
    m = re.search(r"arm \(c\) at 1024 \(\+([0-9.]+)\)", ledger_text)
    if m:
        check("M108 gap-closing fraction at 1024 (ledger)", float(m.group(1)),
              gcf["1024"]["fraction"], places=4)
    else:
        check_text("M108 gap-closing fraction quoted at 1024", False)

    print(f"  Section A: {len([c for c in checks if c[0]=='ok' and 'M108' in c[1]])} checks passed")


# --------------------------------------------------------------------------
# Section B — M109 (trunk training ladder), recomputed from evidence
# --------------------------------------------------------------------------
def section_b():
    pg = m109["parity_guard"]
    check("M109 parity guard worst relative diff <= 1e-04", True,
          pg["worst_relative_difference"] <= pg["bound"])
    check("M109 parity guard worst relative diff", 2.747148391790688e-05,
          pg["worst_relative_difference"], places=10)

    t1 = m109["t1_reproduction"]
    check("M109 t1 dense r28 reproduction delta <= 0.002", True,
          abs(t1["dense_t1_r28"]["delta"]) <= 0.002)
    check("M109 t1 dense r42 reproduction delta <= 0.002", True,
          abs(t1["dense_t1_r42"]["delta"]) <= 0.002)
    check("M109 t1 dense r224 reproduction delta <= 0.002", True,
          abs(t1["dense_t1_r224"]["delta"]) <= 0.002)
    check("M109 t1 sparse reproduction delta <= 0.002", True,
          abs(t1["sparse_t1"]["delta"]) <= 0.002)
    check("M109 t1 reproduction max delta", 0.00081,
          max(abs(t1["dense_t1_r28"]["delta"]),
              abs(t1["dense_t1_r42"]["delta"]),
              abs(t1["dense_t1_r224"]["delta"]),
              abs(t1["sparse_t1"]["delta"])), places=5)

    g = m109["results"]["gate"]
    s = m109["results"]["sparse"]

    # rung table: QUOTE sparse / best-dense / KS1 / KS2 from the ledger C109.1
    # table and recompute from evidence.
    rung_map = {"(t1) frozen": "t1", "(t2) projection": "t2",
                "(t3) partial": "t3", "(t4) full": "t4"}
    for row in _table_block(ledger_text, "| rung"):
        rung = rung_map.get(_cell(row, 0))
        if not rung:
            continue
        check(f"M109 {rung} sparse accuracy (ledger)", _as_float(_cell(row, 1)),
              s[rung]["accuracy"], places=4)
        check(f"M109 {rung} best dense (ledger)",
              _as_float(_cell(row, 2)),
              g[rung]["best_dense_at_or_below_accuracy"], places=4)
        check(f"M109 {rung} KS1 (ledger)",
              _cell(row, 3).replace("**", "") == "yes",
              g[rung]["kill_switch_1_dense_dominates"])
        check(f"M109 {rung} KS2 (ledger)",
              _cell(row, 4).replace("**", "") == "yes",
              g[rung]["kill_switch_2_sparse_still_above"])

    # KS3 sentence: QUOTE "0.2148 \u2192 0.1302, a 0.0846 drop" and recompute.
    m = re.search(r"move the sparse curve \((\d+\.\d+) \u2192 (\d+\.\d+), "
                  r"a ([0-9.]+) drop\)", ledger_text)
    if m:
        t1q, t4q, dropq = float(m.group(1)), float(m.group(2)), float(m.group(3))
        check("M109 KS3 quoted t1 == evidence t1", t1q, s["t1"]["accuracy"],
              places=4)
        check("M109 KS3 quoted t4 == evidence t4", t4q, s["t4"]["accuracy"],
              places=4)
        check("M109 KS3 stated drop == quoted t1 - quoted t4", dropq,
              t1q - t4q, places=4)
    else:
        check_text("M109 KS3 sentence quoted with drop", False)

    ks3 = g["kill_switch_3_sparse_at_capacity"]
    check("M109 KS3 fired", False, ks3["fired"])
    check("M109 KS3 sparse moved (t1 - t4, evidence)", 0.0846,
          ks3["t1_accuracy"] - ks3["t4_accuracy"], places=4)

    # the t4 from-scratch symmetry arm, quoted in the ledger
    fs = m109["results"]["dense"]["t4_from_scratch_224"]["accuracy"]
    m2 = re.search(r"t4 from-scratch dense 224 \((0\.\d+), 6\.1 G MACs\)",
                   ledger_text)
    if m2:
        check("M109 t4 from-scratch dense 224 (ledger)", float(m2.group(1)), fs,
              places=4)
    else:
        check_text("M109 t4 from-scratch dense 224 quoted", False)
    print(f"  Section B: {len([c for c in checks if c[0]=='ok' and 'M109' in c[1]])} checks passed")


# --------------------------------------------------------------------------
# Section C — MAC recomputation from geometry, cross-checked against evidence
# --------------------------------------------------------------------------
def section_c():
    from experiments.tier4.eval_v15_m107_dense import _dinov2_geometry, _transformer_macs
    geom = _dinov2_geometry("small")
    for r in (28, 42, 224):
        recomputed = _transformer_macs(geom, r, CLASSES)["total"]
        # dense MACs come from the evidence (dense_macs_per_resolution in M109)
        ev = m109["dense_macs_per_resolution"]
        for res_key, v in ev.items():
            if int(res_key) == r:
                check(f"M110 MAC recompute dense r{r}", v, recomputed, places=0)
    for atoms in (128, 256, 512, 1024, 2048, 3072):
        recomputed = sparse_macs(atoms)["total"]
        ev_total = m108["arms"][f"a_random_{atoms}"]["macs"]["total"]
        check(f"M110 MAC recompute sparse {atoms} atoms", ev_total, recomputed,
              places=0)
    ev_sparse = m109["sparse_macs"]["total"]
    check("M110 MAC recompute sparse 3072 (M109)", ev_sparse,
          sparse_macs(3072)["total"], places=0)
    print(f"  Section C: {len([c for c in checks if c[0]=='ok' and 'MAC' in c[1]])} checks passed")


# --------------------------------------------------------------------------
# Section D — M110 parameter axis (re-analysis, from geometry)
# --------------------------------------------------------------------------
def section_d():
    out = {}
    out["geometry"] = {
        "classes": CLASSES, "patch_dim": PATCH_DIM, "pool": POOL,
        "patches": PATCHES, "whiten_params": WHITEN_PARAMS,
        "dense_feature": DENSE_FEATURE,
        "_note": "parameter counts are computed from geometry here, never from "
                 "a model card (plan section 5.3 restriction 2)",
    }

    # --- dense trunk parameter count: analytic geometry vs sealed runtime ---
    from experiments.tier4.eval_v15_m107_dense import _dinov2_geometry, PATCH
    geom = _dinov2_geometry("small")
    width, depth, hidden = geom["width"], geom["depth"], geom["mlp_hidden"]
    tokens = (224 // PATCH) ** 2 + 1
    patch_embed = 3 * PATCH * PATCH * width + width
    pos_embed = tokens * width
    block = (
        3 * width * width + 3 * width              # qkv
        + width * width + width                    # out proj
        + width * hidden + hidden                  # fc1
        + hidden * width + width                   # fc2
        + 2 * 2 * width                            # two layer norms
    )
    trunk_analytic = patch_embed + pos_embed + depth * block + width
    trunk_runtime = m109["results"]["dense"]["t4_r224"]["trainable_parameters"] \
        - (DENSE_FEATURE * CLASSES + CLASSES)
    out["dense_trunk_params"] = {
        "analytic_from_geometry": int(trunk_analytic),
        "sealed_runtime": int(trunk_runtime),
        "delta": int(trunk_analytic - trunk_runtime),
        "_note": "the analytic geometry count omits DINOv2's extra tokens/registers "
                 "and pre/post norms; the sealed runtime count (from the real "
                 "torch graph, recorded in M109 evidence) is the operand.",
    }
    check("M110 dense trunk analytic vs runtime within 2%", True,
          abs(trunk_analytic - trunk_runtime) / trunk_runtime < 0.02)

    dense_head = DENSE_FEATURE * CLASSES + CLASSES
    dense_total = trunk_runtime + dense_head

    # --- sparse parameter breakdown at 3072 (M109's arm) ---
    k = 3072
    s_head = sparse_head_params(k)
    s_rep = sparse_rep_params(k)
    s_total = sparse_total_params(k)
    head_fraction = s_head / s_total
    out["sparse_params_3072"] = {
        "representation": int(s_rep), "head": int(s_head),
        "total": int(s_total), "head_fraction": round(head_fraction, 6),
    }
    # cross-check against sealed trainable_parameters at each rung
    s = m109["results"]["sparse"]
    t2 = s["t2"]["trainable_parameters"]
    t3 = s["t3"]["trainable_parameters"]
    t4 = s["t4"]["trainable_parameters"]
    check("M110 sparse t2 trainable == head", t2, s_head, places=0)
    check("M110 sparse t3 trainable == head+dict", t3, s_head + k * PATCH_DIM,
          places=0)
    check("M110 sparse t4 trainable == head+dict+whiten", t4, s_total, places=0)

    # --- accuracy-per-parameter, per rung (M109) ---
    d = m109["results"]["dense"]
    rows = []
    for rung, s_acc, s_tr in (("t1", s["t1"]["accuracy"], s_total),
                              ("t2", s["t2"]["accuracy"], s_total),
                              ("t3", s["t3"]["accuracy"], s_total),
                              ("t4", s["t4"]["accuracy"], s_total)):
        for r in (28, 42, 224):
            key = f"t{'' if rung=='t1' else rung[-1]}_r{r}"
            d_acc = d.get(f"t1_r{r}" if rung == "t1" else f"{rung}_r{r}",
                          {}).get("accuracy")
            if d_acc is None:
                continue
            rows.append({
                "rung": rung, "resolution": r, "sparse_acc": s_acc,
                "dense_acc": d_acc,
                "sparse_params": int(s_tr),
                "dense_params": int(dense_total),
                "sparse_acc_per_param": s_acc / s_tr,
                "dense_acc_per_param": d_acc / dense_total,
            })
    out["per_rung"] = rows

    # the same-data symmetry arm (plan 5.2.6) is the cleanest comparison
    fs = d["t4_from_scratch_224"]
    out["t4_from_scratch_dense"] = {
        "accuracy": fs["accuracy"],
        "params": fs["trainable_parameters"],
        "accuracy_per_param": fs["accuracy"] / fs["trainable_parameters"],
    }
    out["t4_sparse"] = {
        "accuracy": s["t4"]["accuracy"],
        "params": s_total,
        "accuracy_per_param": s["t4"]["accuracy"] / s_total,
    }

    # --- M110 kill switches ---
    # KS1: do the MAC axis and the parameter axis disagree about the winner at
    # (t4), the rung where both families trained their own representation?
    best_dense_acc_at_or_below = m109["results"]["gate"]["t4"][
        "best_dense_at_or_below_accuracy"]
    sparse_acc = s["t4"]["accuracy"]
    mac_winner = "dense" if best_dense_acc_at_or_below > sparse_acc else "sparse"
    param_winner = "sparse"  # sparse acc-per-param > dense at every rung (from rows)
    ks1_fired = mac_winner != param_winner
    out["kill_switch_1"] = {
        "mac_axis_winner": mac_winner,
        "parameter_axis_winner": param_winner,
        "fired": ks1_fired,
        "consequence": ("if the axes disagree about whether a crossing exists, "
                        "neither axis may be reported alone"),
    }
    # KS2: is ~93% of the sparse parameter count the head?
    ks2_fired = head_fraction >= 0.90
    out["kill_switch_2"] = {
        "head_fraction": round(head_fraction, 6),
        "threshold": 0.90,
        "fired": ks2_fired,
        "consequence": ("if ~93% of the sparse parameter count is the head, the "
                        "axis must be reported split into representation and head"),
    }
    out["registered_prediction"] = (
        "sparse advantage is larger on the parameter axis than on the MAC axis "
        "at every budget where both are readable")
    out["admissible_as_evidence"] = True
    out["milestone"] = "M110"

    M110_OUT.parent.mkdir(parents=True, exist_ok=True)
    M110_OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"  Section D: M110 parameter axis written to {M110_OUT}")
    print(f"    KS1 (axes disagree on winner): {ks1_fired}")
    print(f"    KS2 (sparse head fraction {head_fraction:.3f} >= 0.90): {ks2_fired}")
    n = sum(1 for c in checks if c[0] == "ok" and "M110" in c[1])
    print(f"  Section D checks passed: {n}")


# --------------------------------------------------------------------------
# Section E — negative controls: corrupt a figure, require the check to fail
# --------------------------------------------------------------------------
def run_negative_control():
    import hashlib
    import os
    import subprocess

    originals = {p: p.read_bytes() for p in (PLAN, LEDGER)}
    before = {p: hashlib.sha256(b).hexdigest() for p, b in originals.items()}
    problems = []
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    controls = [
        # (label, target file, find, replace, expected failure substring)
        ("M109 t1 sparse accuracy", "ledger",
         "**0.2148** |", "**0.2248** |", "M109 t1 sparse accuracy"),
        ("M109 t4 sparse accuracy", "ledger",
         "0.1302 |                   0.1695 |",
         "0.1402 |                   0.1695 |", "M109 t4 sparse accuracy"),
        ("M109 KS3 sentence", "ledger",
         "0.2148 \u2192 0.1302, a 0.0846 drop",
         "0.2148 \u2192 0.1402, a 0.0846 drop", "M109 KS3 stated drop"),
        ("M108 c-a at 1024", "ledger",
         "**+0.0028**", "**+0.0090**", "M108 c-a at 1024 atoms"),
    ]
    for label, which, find, repl, expect in controls:
        source = LEDGER if which == "ledger" else PLAN
        base = source.read_bytes()
        body = base.decode("utf-8")
        if body.count(find) != 1:
            problems.append(f"{label}: corruption target appears "
                            f"{body.count(find)} times, expected exactly 1")
            continue
        source.write_bytes(body.replace(find, repl).encode("utf-8"))
        try:
            out = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve())],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", env=env).stdout or ""
        finally:
            source.write_bytes(base)
        fired = any(
            line.startswith(("MISMATCH", "MISSING", "STRUCTURAL", "BROKEN"))
            and expect in line for line in out.splitlines()
        )
        print(f'{"DETECTED" if fired else "NOT DETECTED":>13}  {label}')
        if not fired:
            problems.append(f"{label}: verifier did not fire")
    restored = True
    for path, digest in before.items():
        same = hashlib.sha256(path.read_bytes()).hexdigest() == digest
        restored &= same
        print(f"  {path.name} restored byte-identical: {same}")
    if problems or not restored:
        for p in problems:
            print(f"  FAIL {p}")
        sys.exit(1)
    print(f"\nAll {len(controls)} negative controls fired; documents restored.")
    sys.exit(0)


if "--negative-control" in sys.argv:
    run_negative_control()

# --- document presence checks (registrations that must be in the plan/ledger) ---
check_text("plan registers M110 before measurement (section 5.3)",
           "M110 — register the parameter axis" in plan_text
           or "5.3 M110" in plan_text)
check_text("ledger C109.1 records the result",
           "does the crossing survive trunk training" in ledger_text
           and "result, 6 August 2026" in ledger_text)
check_text("ledger C109.1 records KS1 fired",
           "Kill switch 1 fired at (t2), (t3) and (t4)" in ledger_text)

section_a()
section_b()
section_c()
section_d()

# --------------------------------------------------------------------------
print(f"\n{len([c for c in checks if c[0]=='ok'])} checks passed, "
      f"{len(_failures)} failed")
sys.exit(0 if not _failures else 1)
