import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(1)
d = 500
spectra = {"Σ = I  (r = 500)": (np.ones(d), BLUE), "λ_i = 1/i  (r ≈ 6.8)": (1.0 / np.arange(1, d + 1), ORANGE)}
ns = np.unique(np.logspace(2, 4.3, 10).astype(int))
fig, ax = plt.subplots(figsize=(7, 3.3))
for name, (lam, c) in spectra.items():
    r = lam.sum() / lam.max()
    errs = [np.mean([np.linalg.norm((X := rng.standard_normal((n, d)) * np.sqrt(lam)).T @ X / n - np.diag(lam), 2) for _ in range(3)]) for n in ns]
    ax.loglog(ns, errs, "o", color=c, label=f"{name}: observed")
    ax.loglog(ns, 2 * np.sqrt(r / ns) + r / ns, color=c, lw=1.5, ls="--", label=r"$2\sqrt{r/n} + r/n$")
ax.set_xlabel("sample size n"); ax.set_ylabel(r"$\|S - \Sigma\|_{op}$  (with $\|\Sigma\| = 1$)")
ax.set_title(f"d = {d}: error is governed by the effective rank, not d", fontsize=10)
ax.legend(fontsize=8)
save(fig, "mb-cov")
