# Build the MS thesis-style report

The thesis source is `analysis/MS_THESIS_REPORT.tex`. It is separate from the
arXiv paper and expands the research into a thesis structure with front matter,
research questions, background, system design, methodological evolution, a
structured experiment/test ledger, final protocols, results, discussion,
limitations, reproducibility, and appendices.

Before university submission, replace the provisional:

- university, department, city, and degree wording;
- supervisor and committee information;
- approval and declaration text;
- acknowledgments;
- formatting rules required by the institution; and
- repository URL or DOI if required.

Build from the `analysis` directory:

```powershell
tectonic .\MS_THESIS_REPORT.tex
```

If the generated PDF is open and Windows locks it, either close the PDF or
compile to another directory:

```powershell
$out = Join-Path $env:TEMP "geode-thesis-build"
New-Item -ItemType Directory -Force $out | Out-Null
tectonic --outdir $out .\MS_THESIS_REPORT.tex
```

The expected output is `MS_THESIS_REPORT.pdf`. Pass the `.tex` source—not the
PDF—to Tectonic.

`MS_THESIS_REPORT_REVISED.pdf` is the checked-in 65-page build containing the
complete methodology chronology from the first recorded milestones onward. It
was emitted under a separate name because the earlier
`MS_THESIS_REPORT.pdf` was open and locked during regeneration. Closing the old
PDF and running the standard command will regenerate that conventional output
from the same revised source.
