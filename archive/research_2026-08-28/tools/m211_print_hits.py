"""Print the M211 search hits for the anchor and discovery queries."""
import json

e = json.load(open("logs/results/v25/m211_zkml_llm_search/evidence.json",
                   encoding="utf-8"))
print("anchor gate:", e["anchor_gate"], "void:", e["void"])
for stage in ("and", "or"):
    r = e["results"].get(stage, {})
    for qid in ("anchor_zkllm", "anchor_zen", "anchor_zkcnn",
                "disc_optimistic_ml", "disc_quantized_zk_llm",
                "disc_zkvm_ml", "disc_mpc_llm", "disc_fhe_llm",
                "disc_verifiable_llm", "disc_proof_of_inference"):
        if qid not in r:
            continue
        res = r[qid]
        print(f"[{stage}] {qid} n={res['n_hits']} status={res['status']}")
        for h in res["hits"][:6]:
            print("   -", h["title"][:120])
