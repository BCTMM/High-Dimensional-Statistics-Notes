import numpy as np
from scipy import stats
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
t = np.linspace(0, 20, 600)
fig, ax = plt.subplots(figsize=(7, 3.3))
ax.semilogy(t, 2 * stats.norm.sf(t), color=BLUE, lw=2, label="Gaussian  (sub-Gaussian)")
ax.semilogy(t, np.exp(-t * np.sqrt(2)), color=GREEN, lw=2, label="Laplace  (sub-exponential)")
ax.semilogy(t, 2 * stats.t.sf(t * np.sqrt(3), 3), color=ORANGE, lw=2, label="Student-t, 3 d.o.f.  (heavy, polynomial)")
ax.semilogy(t, stats.chi2.sf(1 + np.sqrt(2) * t, 1), color=PINK, lw=2, ls="--",
            label=r"$(Z^2-1)/\sqrt{2}$, upper tail  (sub-exponential)")
ax.set_ylim(1e-12, 1.5)
ax.set_xlabel("t   (all variables standardized to variance 1)")
ax.set_ylabel(r"$P(|X| \geq t)$")
ax.legend(fontsize=8.5, loc="lower left")
save(fig, "sg-tails")
