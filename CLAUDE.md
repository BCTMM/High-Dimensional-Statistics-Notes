# Daily Concepts — working notes for Claude

A MkDocs Material site of daily study notes (RMT, high-dim probability, covariance/quant, regression & shrinkage,
uncertainty quantification), deployed to GitHub Pages by `.github/workflows/deploy.yml` on push to `main`.

## Layout
- `curriculum.yml` — single source of truth: tracks, note slugs, titles, prereqs, ★ priorities.
- `hooks/curriculum.py` — builds the sidebar nav, the interleaved daily order (topological, track-balanced),
  the Roadmap table, the "Builds on"/"Leads to" boxes, and `site/curriculum.json`. Links to unpublished notes
  are rendered as plain "coming soon" text automatically, so forward links are fine to write.
- A note is published when `docs/notes/<slug>.md` exists.
- `docs/javascripts/daily.js` — daily-note logic (localStorage). `docs/javascripts/widgets/` — `lib.js`
  (RNG, eigen solvers, canvas Plot, controls) and `widgets.js` (one function per widget, mounted from
  `<div class="widget" data-widget="NAME"></div>`).
- `scripts/figures/<slug>.py` — matplotlib figures → `docs/notes/img/*.svg` (import `_style`; set `FIG_PREVIEW=dir` for PNG copies).

## Commands
- Serve: `.venv/bin/mkdocs serve`  ·  Build check: `.venv/bin/mkdocs build --strict`
- Figures: `cd scripts/figures && ../../.venv/bin/python <slug>.py`

## Note template (keep consistent)
H1 title → `!!! tldr` → `## Why care?` → `## Building blocks` → `## The main result` (use `!!! theorem "Theorem (...)"`,
then derivation/proof sketch) → `## Examples` (Python/NumPy snippet + figure; widget where it helps) →
`## Exercises` (4–5, warm-up → stretch; `!!! question` with nested `??? success "Solution"`) →
`## Where it shows up` (modern ML / quant applications) → `## Further reading`.
Do NOT write a "Builds on" section by hand — the hook injects it from `prereqs`.

## Conventions
- Audience: multivariable calculus, linear algebra, basic probability/statistics. Balance intuition, pedagogy, rigor.
- Eigenvalues in decreasing order λ₁ ≥ … ≥ λₙ unless stated. GOE: off-diagonal N(0,1), diagonal N(0,2); H/√n → semicircle on [−2, 2].
  Sample covariance E = XXᵀ/T with q = N/T.
- Every printed output in a code block must come from actually running the code (fixed seeds). Verify every numeric claim.
- Math: `$...$` / `$$...$$` (KaTeX via arithmatex). Macros available: \R \E \P \Var \Cov \tr \diag.
- Figures: `![alt](img/name.svg){ .fig }` — transparent background, dark ink; CSS inverts them in dark mode.
- Widgets must work on a phone (390px wide) and in dark mode; test with Playwright (`.venv` has it).
