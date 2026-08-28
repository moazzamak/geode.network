"""Fill the M123 certificate verdict note into the sealed evidence (measurement
numbers untouched) and rebuild the artifact index. One-shot helper."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common.v5_artifacts import (  # noqa: E402
    build_artifact_index,
    write_canonical_json,
)

OUT = Path(__file__).resolve().parents[2] / "logs" / "results" / "v16" / "m123_margin_certificate"

NOTE = (
    "SEALED 9 Aug: the per-class Gaussian margin certificate FAILS to predict the "
    "measured M116 crossing (KS1 fired: no crossing at any lambda; KS2 fired: "
    "predicted dense 0.9998 > sparse 0.0035 at n_max, measured 0.2153 > 0.1972). "
    "The model predicts sparse ~ chance at every data size and dense -> ~1.0, both "
    "contradicted by measurement. Per the registered fallback (v19 5.2), the "
    "certificate is CLOSED as a prediction tool and kept as an explanatory "
    "diagnostic only. Diagnostic content: (1) the spectral facts reproduce M121 "
    "(sparse captures 3.4x more label power, eff-rank 8 vs 30); (2) the failure "
    "mode is instructive: the Canatar per-class score variance makes dense's score "
    "noise vanish (its few captured modes are learned instantly), yet the MEASURED "
    "dense accuracy (0.1972) is far from the model's near-perfect prediction - the "
    "345-channel argmax is far more sensitive than the population-spectrum Gaussian "
    "model captures, confirming M121's conclusion that the accuracy crossing is an "
    "argmax phenomenon no spectral MSE/margin proxy built from the Gram spectrum + "
    "label projections alone tracks."
)

LICENSE = (
    "M123 fires under the registered gate; the firing is about the spectral "
    "prediction TOOL, not about the existence of learnable label power in the "
    "sparse span (11.8% vs 3.5%, reproduced)."
)

e = json.loads((OUT / "evidence.json").read_text(encoding="utf-8"))
e["certificate_verdict"] = {
    "note": NOTE,
    "registered_conclusion_licenses": LICENSE,
}
write_canonical_json(OUT / "evidence.json", e)
build_artifact_index(OUT)
print("verdict note filled + index rebuilt")
