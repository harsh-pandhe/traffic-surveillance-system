# Research Paper

`paper.pdf` — generic two-column academic format, built by
`scripts/build_research_paper.py` directly from `results/phase*/*.json`
(the same tracked files used by every other report in this project).

## Rebuilding

```bash
python scripts/build_research_paper.py
```

## Retargeting for a specific venue

The current layout (A4, two-column, Helvetica 9.3pt) is generic. To target a
specific venue (IEEE, ACM, a conference's own LaTeX template, etc.):

1. Update page size and margins to the venue's template (most use US Letter,
   ~1.9cm/0.75in margins, Times 10pt).
2. Swap the citation style if the venue mandates a specific format (current
   references are numbered IEEE-style).
3. Check the venue's page limit — the paper currently runs to a compact
   5 pages including 6 figures and 4 tables; expand or trim sections in
   `scripts/build_research_paper.py` accordingly.
4. If the venue requires a LaTeX submission rather than a camera-ready PDF,
   this script's structure (title/abstract → 7 sections → references) maps
   directly onto a standard `article`/`IEEEtran` class; the prose can be
   copied in directly since it contains no PDF-specific formatting.

All reported numbers are sourced from `results/phase1..4/*.json` — retarget
formatting, not content, unless new experiments are run.
