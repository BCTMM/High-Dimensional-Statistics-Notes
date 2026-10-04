import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(7, 3.2))
norms = np.linspace(0, 8, 41)
for d, c in [(3, GREEN), (10, BLUE), (50, ORANGE)]:
    risk = []
    for r in norms:
        theta = np.zeros(d); theta[0] = r
        X = theta + rng.standard_normal((20000, d))
        js = (1 - (d - 2) / np.sum(X**2, 1))[:, None] * X
        risk.append(np.mean(np.sum((js - theta) ** 2, 1)) / d)
    ax.plot(norms, risk, color=c, lw=2, label=f"James–Stein, d = {d}")
ax.axhline(1, color=GREY, ls="--", lw=1.5, label="MLE (all d)")
ax.set_xlabel(r"$\|\theta\|$ (in units of $\sigma$)")
ax.set_ylabel(r"risk / $(d\sigma^2)$")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8.5, loc="lower right")
save(fig, "js-risk")
