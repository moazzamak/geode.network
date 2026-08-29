"""Static structural validation of the GEODE whitepaper source.

There is no LaTeX toolchain on this machine, so the paper cannot
be compiled. This checks the failure modes that a compile would
have caught: unresolved references, orphaned or duplicated
bibliography entries, unbalanced braces, and unbalanced
environments. It is a lint, not a substitute for a build.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

TEX = Path("docs/WHITEPAPER_GEODE.tex")


def main() -> int:
    src = TEX.read_text(encoding="utf-8")
    failures: list[str] = []

    bibitems = re.findall(r"\\bibitem\{([^}]+)\}", src)
    cited: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", src):
        cited.update(k.strip() for k in group.split(","))
    dupes = [k for k, n in Counter(bibitems).items() if n > 1]
    missing = sorted(cited - set(bibitems))
    uncited = sorted(set(bibitems) - cited)
    if dupes:
        failures.append(f"duplicate bibitem keys: {dupes}")
    if missing:
        failures.append(f"cited but not defined: {missing}")
    if uncited:
        failures.append(f"defined but never cited: {uncited}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs: set[str] = set()
    for group in re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}",
                            src):
        refs.update(k.strip() for k in group.split(","))
    dangling = sorted(refs - labels)
    if dangling:
        failures.append(f"unresolved refs: {dangling}")

    stripped = re.sub(r"(?<!\\)%.*", "", src)
    stripped = re.sub(r"\\[{}]", "", stripped)
    opens, closes = stripped.count("{"), stripped.count("}")
    if opens != closes:
        failures.append(f"brace imbalance: {opens} open, "
                        f"{closes} close")

    begins = Counter(re.findall(r"\\begin\{([^}]+)\}", stripped))
    ends = Counter(re.findall(r"\\end\{([^}]+)\}", stripped))
    for env in sorted(set(begins) | set(ends)):
        if begins[env] != ends[env]:
            failures.append(f"environment {env!r}: "
                            f"{begins[env]} begin, {ends[env]} end")

    print(f"bibitems: {len(bibitems)}  cited keys: {len(cited)}  "
          f"labels: {len(labels)}  refs: {len(refs)}")
    print(f"braces: {opens}/{closes}  environments: {len(begins)}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: no structural defects found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
