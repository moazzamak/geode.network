import json

d = json.load(open('logs/results/v13/m80_sparse_dictionary/evidence.json'))
raw = d['gate']['raw_feature_probe_balanced_accuracy']
print(f"raw feature probe: {raw*100:.3f}%")
print()
head = f"{'m':>6} {'k':>4} {'probe':>9} {'rnd':>9} {'margin':>8} {'deficit':>8} {'bits':>6} {'shuf':>6} {'dead':>6}"
print(head)
print('-' * len(head))
for c in d['cells']:
    p = c['codes_probe_balanced_accuracy'] * 100
    r = c['random_control_probe_balanced_accuracy'] * 100
    print(
        f"{c['dictionary_size']:>6} {c['active_atoms']:>4} {p:>8.3f}% {r:>8.3f}% "
        f"{p - r:>+7.2f} {raw*100 - p:>7.2f} "
        f"{c['mean_atom_label_entropy_bits']:>6.2f} {c['shuffled_label_entropy_bits']:>6.2f} "
        f"{c['dead_atom_fraction']*100:>5.1f}%"
    )
print()
print("cells passing tolerance AND beating their own random control:")
for c in d['cells']:
    p = c['codes_probe_balanced_accuracy'] * 100
    r = c['random_control_probe_balanced_accuracy'] * 100
    if raw * 100 - p <= 3.0 and p > r:
        print(f"  m={c['dictionary_size']} k={c['active_atoms']}  "
              f"deficit {raw*100-p:.3f} pt, margin over random +{p-r:.2f} pt, "
              f"bits {c['mean_atom_label_entropy_bits']:.2f}")
print()
print("convergence check (first/last 3 of loss trace):")
for c in d['cells']:
    t = c['epoch_loss_trace']
    print(f"  m={c['dictionary_size']:>5} k={c['active_atoms']:>3}  "
          f"{t[0]:.4f} {t[1]:.4f} {t[2]:.4f} ... {t[-3]:.4f} {t[-2]:.4f} {t[-1]:.4f}  "
          f"last-10 drop {t[-10]-t[-1]:.5f}")
