# Build the arXiv paper

The submission source is `analysis/FINAL_RESEARCH_PAPER.tex`. It is a
self-contained LaTeX article with an inline bibliography and no external
figures.

Before submission, replace the placeholder author, affiliation, email, public
repository URL, and release DOI.

The recommended compiler is
[Tectonic](https://tectonic-typesetting.github.io/), which does not require a
separate `pdflatex` installation. After installing it, open a new terminal so
the updated `PATH` is visible, then run:

```powershell
Set-Location analysis
tectonic FINAL_RESEARCH_PAPER.tex
```

If Tectonic is not yet visible in the current terminal on Windows, use its
installed path directly:

```powershell
Set-Location analysis
& "$env:LOCALAPPDATA\Programs\Tectonic\tectonic.exe" FINAL_RESEARCH_PAPER.tex
```

A full TeX distribution remains an optional alternative:

```powershell
Set-Location analysis
pdflatex -interaction=nonstopmode -halt-on-error FINAL_RESEARCH_PAPER.tex
pdflatex -interaction=nonstopmode -halt-on-error FINAL_RESEARCH_PAPER.tex
```

For arXiv, upload `FINAL_RESEARCH_PAPER.tex`. The generated PDF and LaTeX
auxiliary files are build products and do not need to be included in the source
archive.

`FINAL_RESEARCH_PAPER.md` is the historical v7 manuscript and is not the v8
submission source.

## The v19 findings paper

The v19 findings are written up as a self-contained technical report in
`analysis/RESEARCH_REPORT_v19.md` (abstract, literature review with
references, methodology, results, conclusion, references; architecture and
reproduction sections aimed at a technically capable non-expert). It is the
authoritative write-up of the joint-budget scaling measurements and the v20
engineering track. To submit it, convert the markdown to LaTeX (e.g.
`pandoc -s RESEARCH_REPORT_v19.md -o research_report_v19.tex`) and replace
the placeholder author/affiliation/email/repository fields; every numeric
claim carries a sealed evidence path under `logs/results/v16/`.

`FINAL_RESEARCH_PAPER.tex` remains the earlier (v8-era) manuscript; the v19
findings are appended to it as a self-contained "Recent extension: the
joint-budget scaling frontier" section. The registered literature surveys
are `analysis/HTN_ROUTING_LITERATURE_REVIEW.md` (M127) and
`analysis/PROGRAMMATIC_PRIMITIVES_LITERATURE_REVIEW.md` (M129).
