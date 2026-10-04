import numpy as np
from _style import BLUE, ORANGE, GREY, plt, save

rng = np.random.default_rng(3)
N, T = 200, 400                                     # q = 0.5
X = rng.standard_normal((N, T))                     # true covariance = I
E = X @ X.T / T
lam = np.linalg.eigvalsh(E)
q = N / T
lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
x = np.linspace(lo, hi, 400)
mp = np.sqrt(np.maximum((hi - x) * (x - lo), 0)) / (2 * np.pi * q * x)
fig, ax = plt.subplots(figsize=(7, 3.1))
ax.hist(lam, bins=50, density=True, alpha=0.45, color=BLUE, label="sample eigenvalues")
ax.plot(x, mp, color=ORANGE, lw=2, label="Marchenko–Pastur density (a later note)")
ax.axvline(1, color=GREY, ls="--", lw=1.5, label="true eigenvalues (all = 1)")
ax.set_xlabel("eigenvalue")
ax.set_ylabel("density")
ax.legend(fontsize=8.5)
save(fig, "mk-eigs")

# Weights of the plug-in minimum-variance portfolio vs the truth (w* = 1/N).
w = np.linalg.solve(E, np.ones(N))
w /= w.sum()
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.hist(w * N, bins=50, color=BLUE, alpha=0.6)
ax.axvline(1, color=ORANGE, lw=2, label="true optimal weight (= 1/N)")
ax.set_xlabel("weight × N")
ax.set_ylabel("count")
ax.legend()
save(fig, "mk-weights")
print("gross leverage sum|w| =", np.abs(w).sum(), " fraction short =", (w < 0).mean())
