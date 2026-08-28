"""Summarize the M78 sample-adequacy grid as a rank-by-sample-count table.

M78 recorded one cell per (sample count, rank, seed). This collapses the seed
dimension and prints the geometric head against its logistic control so the
effect of sample adequacy on the comparison can be read directly.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "logs/results/v13/m78_sample_adequacy/evidence.json"
)


def main() -> None:
    cells = json.loads(EVIDENCE.read_text(encoding="utf-8"))["cells"]

    grouped: dict[tuple[object, int], list[dict]] = defaultdict(list)
    for cell in cells:
        count = cell.get("geometry_per_class", cell.get("samples_per_class"))
        grouped[(count, cell["requested_rank"])].append(cell)

    header = (
        f"{'n/class':>8} {'rank':>5} {'fit':>4} {'n/dim':>7} {'ident':>6} "
        f"{'geo_acc':>8} {'logi_acc':>9} {'delta':>7} {'geo_unk':>8} {'logi_unk':>9}"
    )
    print(header)
    print("-" * len(header))

    for key in sorted(grouped, key=lambda k: (k[0] or 0, k[1])):
        group = grouped[key]
        average = lambda selector: mean(selector(cell) for cell in group)

        geometric = average(lambda c: c["known_balanced_accuracy"])
        logistic = average(lambda c: c["logistic_known_balanced_accuracy"])
        print(
            f"{str(key[0]):>8} {key[1]:>5} {group[0]['fitted_rank']:>4} "
            f"{average(lambda c: c['samples_per_fitted_dimension']):>7.2f} "
            f"{average(lambda c: c['subspace_stability']['identifiability']):>6.3f} "
            f"{geometric * 100:>7.2f}% {logistic * 100:>8.2f}% "
            f"{(geometric - logistic) * 100:>+6.2f} "
            f"{average(lambda c: c['unknown_recall']) * 100:>7.2f}% "
            f"{average(lambda c: c['logistic_unknown_recall']) * 100:>8.2f}%"
        )


if __name__ == "__main__":
    main()
