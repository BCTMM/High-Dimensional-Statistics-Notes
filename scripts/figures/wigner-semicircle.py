import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(0)
N = 2000
def wigner(A):
    X = np.triu(A, 1); X = X + X.T + np.diag(np.diag(A)); return X / np.sqrt(N)
x = np.linspace(-2, 2, 400); sc = np.sqrt(4 - x**2) / (2 * np.pi)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
lam_r = np.linalg.eigvalsh(wigner(rng.choice([-1.0, 1.0], size=(N, N))))
axs[0].hist(lam_r, bins=60, density=True, color=BLUE, alpha=0.5, label="±1 entries, N = 2000")
axs[0].plot(x, sc, color=ORANGE, lw=2, label="semicircle")
axs[0].set_title("light tails: semicircle, edge at 2", fontsize=10); axs[0].legend(fontsize=8); axs[0].set_xlabel("eigenvalue")
lam_t = np.linalg.eigvalsh(wigner(rng.standard_t(2.5, size=(N, N)) / np.sqrt(5.0)))
axs[1].hist(lam_t[np.abs(lam_t) < 3], bins=60, density=True, color=GREEN, alpha=0.5, label="Student-t(2.5) entries")
axs[1].set_ylim(0, 0.55)
axs[1].plot(x, sc, color=ORANGE, lw=2, label="semicircle")
out = lam_t[np.abs(lam_t) > 2.3]
axs[1].plot(np.clip(out, -3, 3), np.full_like(out, 0.01), "|", color=PINK, ms=14, mew=2, label=f"{len(out)} outliers (largest {lam_t.max():.1f})")
axs[1].set_xlim(-3.1, 3.1)
axs[1].set_title("heavy tails: outliers escape, bulk shrinks (slow convergence)", fontsize=10); axs[1].legend(fontsize=7.5, loc="upper right"); axs[1].set_xlabel("eigenvalue")
save(fig, "wig-universality")
