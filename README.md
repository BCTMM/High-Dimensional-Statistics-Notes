# When p Meets n

**Daily notes on random matrices, shrinkage & uncertainty** — read them at
**<https://bctmm.github.io/When-p-meets-n/>**.

Classical statistics assumes many more samples than variables. Modern data rarely cooperates: a covariance
matrix of 500 stocks from two years of returns, a regression with as many features as observations, a
neural network with more parameters than training points. When the dimension *p* is comparable to the
sample size *n*, sample eigenvalues spread out, estimators overfit, and textbook confidence intervals lie.
These notes build, one concept a day, the theory that explains what goes wrong and how to fix it.

## What's inside

53 notes across six tracks, ordered so that every note's prerequisites come first:

| Track | Notes | Highlights |
|---|---|---|
| Toolbox | 8 | Courant–Fischer & Weyl, Davis–Kahan, Fisher information, KKT, proximal methods |
| High-dimensional probability | 5 | Geometry of high dimensions, sub-Gaussian tails, matrix Bernstein, Johnson–Lindenstrauss |
| Random matrix theory | 10 | Semicircle & Marchenko–Pastur laws, free probability, Tracy–Widom, BBP transition |
| Covariance estimation & quant | 6 | Markowitz estimation error, James–Stein, Ledoit–Wolf, eigenvalue clipping, rotationally invariant & nonlinear shrinkage |
| Regression & shrinkage | 11 | Ridge & double descent, LASSO & LARS, robust and quantile regression, debiased LASSO, double ML |
| Uncertainty quantification | 13 | Bootstrap, sandwich & HAC errors, conformal prediction, FDR & knockoffs, post-selection inference, high-dimensional MLE |

Each note follows the same arc: **why it matters → building blocks → the main result** (stated precisely,
with a derivation or proof sketch) **→ worked examples** with reproducible NumPy simulations
**→ exercises** with hidden solutions **→ where it shows up** in modern ML and quantitative finance.
Interactive widgets let you watch eigenvalues repel, push a spike through the BBP transition, trace a LASSO
path, or drive a ridge model through double descent.

**Background assumed:** multivariable calculus, linear algebra, basic probability & statistics.

## How the daily note works

- **Daily note** opens the next unread note, alternating between tracks while respecting prerequisites.
- You get the same note all day; a new one is picked tomorrow. After every note has been read, a new cycle begins.
- Opening a note from the sidebar also marks it read. Progress lives in your browser's localStorage,
  so each device keeps its own.

## Repository layout

| Path | Contents |
|---|---|
| `curriculum.yml` | Tracks, note slugs, titles and prerequisites (single source of truth) |
| `docs/notes/` | The notes (Markdown + KaTeX) and their figures |
| `hooks/curriculum.py` | Builds the sidebar, daily order, roadmap and "Builds on" boxes from `curriculum.yml` |
| `docs/javascripts/` | Daily-note logic and the interactive widgets |
| `scripts/figures/` | Matplotlib scripts that generate every figure |

Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages
on every push to `main`.
