import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(1)
N, T = 200, 300; q = N / T
beta = 1 + 0.25 * rng.standard_normal(N); sector = rng.integers(0, 8, N)
load = np.zeros((N, 9)); load[:, 0] = 0.3 * beta; load[np.arange(N), 1 + sector] = 0.25
Sigma = load @ load.T + np.diag(rng.uniform(0.6, 1.0, N)); d = np.sqrt(np.diag(Sigma)); Sigma /= np.outer(d, d)
X = rng.multivariate_normal(np.zeros(N), Sigma, size=T); C = X.T @ X / T; dd = np.sqrt(np.diag(C)); C /= np.outer(dd, dd)
lam = np.sort(np.linalg.eigvalsh(C))[::-1]; edge = (1 + np.sqrt(q)) ** 2
lam_c = np.where(lam > edge, lam, lam[lam <= edge].mean())
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.semilogy(np.arange(1, N + 1), np.sort(np.linalg.eigvalsh(Sigma))[::-1], color="#222", lw=2, label="true correlation eigenvalues")
ax.semilogy(np.arange(1, N + 1), lam, color=ORANGE, lw=1.6, label="sample eigenvalues")
ax.semilogy(np.arange(1, N + 1), lam_c, color=BLUE, lw=1.8, label="clipped")
ax.axhline(edge, color=GREEN, ls=":", lw=1.5, label=f"MP edge (1+√q)² = {edge:.2f}")
ax.set_xlabel("rank"); ax.set_ylabel("eigenvalue (log)"); ax.set_title(f"N = {N}, T = {T}: market + 8 sectors", fontsize=10)
ax.legend(fontsize=8)
save(fig, "clip-eigs")
