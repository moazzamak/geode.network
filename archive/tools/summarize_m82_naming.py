import json
import pathlib

evidence = json.loads(
    pathlib.Path("logs/results/v13/m82_atom_naming/evidence.json").read_text()
)
gate = evidence["gate"]
print("verdict:", gate["verdict"], " seeds", evidence["seeds"], " sealed", evidence["sealed"])
print()
print("PURITY CONTROL (gating)")
control = gate["purity_control"]
print(f"  floor {control['floor']}  mean {control['mean']}  null {control['null_mean']}  passes {control['passes']}")
print(f"  per seed {control['per_seed']}")
print()
print("STABILITY")
stability = gate["stability"]
print(f"  agreement {stability['agreement_mean']}  null {stability['null_mean']}")
print(f"  margin {stability['margin_mean']}  spread {stability['margin_spread']}  passes {stability['passes']}")
print(f"  per seed {stability['margin_per_seed']}")
print()
print("NAMING DELTA")
for tag, entry in gate["naming_delta"].items():
    print(f"  {tag}")
    for key, value in entry.items():
        print(f"    {key}: {value}")
print()
print("far field rate mean:", gate["far_field_rate_mean"])
print()
for seed_result in evidence["per_seed"]:
    naming = seed_result["naming"]
    print(f"SEED {seed_result['seed']}  {seed_result['elapsed_seconds']:.0f}s")
    print(
        f"  live {naming['live_atoms']}/{naming['atoms']}  named {naming['named_atoms']}"
        f"  distinct {naming['distinct_names']}  style-named {naming['style_named_atoms']}"
        f"  pure {naming['pure_atoms']}  mean score {naming['mean_name_score']}"
    )
    print(f"  dead atom fraction {seed_result['dictionary']['dead_atom_fraction']}")
    print(f"  far field {naming['far_field']}")
    print(f"  purity control {naming['purity_control']}")
    print(f"  purity null    {naming['purity_control_null']}")
    print(f"  stability {naming['stability']} null {naming['stability_null']}")
    print("  per domain:")
    for domain, entry in naming["per_domain"].items():
        print(f"    {domain}: {entry}")
    for arm in seed_result["i5_eight"]:
        print(
            f"  {arm['arm']}: acc {arm['balanced_accuracy']:.4f}"
            f"  active {arm['active_atoms_in_head']}  named {arm['named_atoms_in_head']}"
            f"  distinct names {arm['distinct_names_in_head']}"
        )
        for name, sub in arm["arms"].items():
            i5 = sub["i5"]["probe_balanced_accuracy"]
            null = sub["i5_shuffled_null"]["probe_balanced_accuracy"]
            print(
                f"      {name:32s} width {sub['explanation_width']:5d}  I5 {i5}  null {null}"
            )
        print(
            f"      naming delta {arm['naming_delta']}  revelation {arm['revelation_delta']}"
            f"  per-atom {arm['per_atom_delta']}"
        )
    print()
