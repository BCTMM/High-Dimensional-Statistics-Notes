import numpy as np
from scipy import stats
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
n, p = 10000, 0.01
t = np.linspace(0, 0.006, 300)                     # deviation of the sample mean
exact = stats.binom.sf(np.ceil(n * (p + t)) - 1, n, p)
hoeff = np.exp(-2 * n * t**2)
bern = np.exp(-n * t**2 / (2 * (p * (1 - p) + t / 3)))
clt = stats.norm.sf(t / np.sqrt(p * (1 - p) / n))
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.semilogy(t, exact, color="#222", lw=2.2, label="exact binomial tail")
ax.semilogy(t, clt, color=GREY, ls="--", lw=1.5, label="normal approximation")
ax.semilogy(t, bern, color=BLUE, lw=2, label="Bernstein bound")
ax.semilogy(t, hoeff, color=ORANGE, lw=2, label="Hoeffding bound")
ax.set_ylim(1e-10, 1.5); ax.set_xlabel(r"deviation $t$ of the sample mean above $p = 0.01$  (n = 10,000)")
ax.set_ylabel(r"$P(\hat p - p \geq t)$"); ax.legend(fontsize=8.5, loc="lower left")
save(fig, "hb-tails")
