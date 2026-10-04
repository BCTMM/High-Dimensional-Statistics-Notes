import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(0)
p, n = 100, 200
Sigma = 0.7 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
X = rng.standard_normal((n, p)) @ np.linalg.cholesky(Sigma).T
S = X.T @ X / n; m = np.trace(S) / p
d2 = np.sum((S - m * np.eye(p)) ** 2) / p
b2 = min(sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / p / n**2, d2)
rho = b2 / d2
lam_S = np.sort(np.linalg.eigvalsh(S))[::-1]
fig, ax = plt.subplots(figsize=(7, 3.3))
ax.plot(np.sort(np.linalg.eigvalsh(Sigma))[::-1], color="#222", lw=2.2, label="true eigenvalues")
ax.plot(lam_S, color=ORANGE, lw=1.8, label="sample eigenvalues")
ax.plot((1 - rho) * lam_S + rho * m, color=BLUE, lw=1.8, label=f"Ledoit–Wolf (intensity {rho:.2f})")
ax.axhline(m, color=GREY, ls=":", lw=1.2, label="grand mean")
ax.set_yscale("log"); ax.set_xlabel("rank"); ax.set_ylabel("eigenvalue (log scale)")
ax.set_title(f"AR(1) covariance, p = {p}, n = {n}", fontsize=10)
ax.legend(fontsize=8.5)
save(fig, "lw-eigs")
