import json

c = json.load(open("coverage.json", encoding="utf-8"))
for f, d in c.items():
    print("===", f)
    b = d.get("b", {})
    for k, v in b.items():
        if isinstance(v, dict):
            if v.get("count", 1) == 0:
                print("  zero branch", k, "line", v["locations"][0]["line"])
        elif isinstance(v, list):
            for br in v:
                if isinstance(br, dict) and br.get("count", 1) == 0:
                    print("  zero branch", k, "line", br["locations"][0]["line"])
