import numpy as np
from scipy import stats
from scipy.linalg import eigvalsh_tridiagonal
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
from _tw import F1, F2, density
rng = np.random.default_rng(2)
g = np.linspace(-6, 4, 600)
_, d1 = density(F1, g); _, d2 = density(F2, g)

def beta_lmax(beta, N, reps):                    # Dumitriu–Edelman tridiagonal model, eigenvalues scaled to [-2, 2]
    out = []
    for _ in range(reps):
        dg = rng.standard_normal(N) * np.sqrt(2 / beta)
        off = np.sqrt(rng.chisquare(beta * np.arange(N - 1, 0, -1))) / np.sqrt(beta)
        out.append(eigvalsh_tridiagonal(dg, off, select="i", select_range=(N - 1, N - 1))[0] / np.sqrt(N))
    return (np.array(out) - 2) * N ** (2 / 3)
N = 1000
z1 = beta_lmax(1, N, 4000); z2 = beta_lmax(2, N, 4000)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.hist(z1, bins=60, density=True, color=BLUE, alpha=0.35, label="GOE (β=1), N = 1000")
ax.hist(z2, bins=60, density=True, color=GREEN, alpha=0.35, label="GUE (β=2), N = 1000")
ax.plot(g, d1, color=BLUE, lw=2, label="Tracy–Widom $F_1$")
ax.plot(g, d2, color=GREEN, lw=2, label="Tracy–Widom $F_2$")
m, s = np.trapezoid(g * d1, g), np.sqrt(np.trapezoid((g - np.trapezoid(g * d1, g)) ** 2 * d1, g))
ax.plot(g, stats.norm.pdf(g, m, s), color=GREY, ls="--", lw=1.5, label="Gaussian with $F_1$'s mean, sd")
ax.set_xlabel(r"$N^{2/3}(\lambda_{\max} - 2)$"); ax.set_ylabel("density"); ax.legend(fontsize=8)
save(fig, "tw-hist")
print("GOE sim mean/sd", z1.mean().round(3), z1.std().round(3), " GUE", z2.mean().round(3), z2.std().round(3))
