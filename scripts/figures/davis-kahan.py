import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save

rng = np.random.default_rng(1)
p, n = 50, 500
u = np.ones(p) / np.sqrt(p)
gaps = np.linspace(0.05, 3, 40)
sins, bounds = [], []
for g in gaps:
    s_list, b_list = [], []
    for _ in range(20):
        Sigma = np.eye(p) + g * np.outer(u, u)
        X = np.linalg.cholesky(Sigma) @ rng.standard_normal((p, n))
        S = X @ X.T / n
        uhat = np.linalg.eigh(S)[1][:, -1]
        s_list.append(np.sqrt(max(0.0, 1 - (uhat @ u) ** 2)))
        b_list.append(min(1.0, 2 * np.linalg.norm(S - Sigma, 2) / g))
    sins.append(np.mean(s_list)); bounds.append(np.mean(b_list))
fig, ax = plt.subplots(figsize=(7, 3.1))
ax.plot(gaps, bounds, color=ORANGE, lw=2, label=r"Davis–Kahan bound $\min(1,\, 2\|E\|/\mathrm{gap})$")
ax.plot(gaps, sins, "o", color=BLUE, ms=4, label=r"actual $\sin\angle(u,\hat u)$")
ax.set_xlabel("eigengap of the true covariance")
ax.set_ylabel("sine of angle")
ax.set_ylim(0, 1.05)
ax.legend()
save(fig, "dk-gap")
