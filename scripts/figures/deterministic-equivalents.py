import numpy as np
from scipy.optimize import brentq
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
p = 400
s = 1.0 / np.arange(1, p + 1) ** 1.2; s *= p / s.sum()
def kappa(lam, n):
    f = lambda k: k - lam - k * np.sum(s / (s + k)) / n
    if lam == 0 and p <= n: return 0.0
    return brentq(f, max(lam, 1e-12), lam + 1e3 * s.sum())
lams = np.logspace(-4, 1, 200)
fig, ax = plt.subplots(figsize=(7, 3.2))
for n, c in [(800, GREEN), (400, BLUE), (200, ORANGE)]:
    ax.loglog(lams, [kappa(l, n) for l in lams], color=c, lw=2, label=f"p/n = {p/n:g}")
ax.loglog(lams, lams, color=GREY, ls="--", lw=1.2, label=r"$\kappa = \lambda$ (no extra regularization)")
ax.set_xlabel(r"explicit ridge penalty $\lambda$"); ax.set_ylabel(r"effective regularization $\kappa(\lambda)$")
ax.set_title("power-law covariance spectrum, p = 400", fontsize=10); ax.legend(fontsize=8.5)
save(fig, "de-kappa")
