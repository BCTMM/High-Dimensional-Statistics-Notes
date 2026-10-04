import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(0)

def mp_density(x, q, s2=1.0):
    lo, hi = s2 * (1 - np.sqrt(q)) ** 2, s2 * (1 + np.sqrt(q)) ** 2
    return np.where((x > lo) & (x < hi), np.sqrt(np.maximum((hi - x) * (x - lo), 0)) / (2 * np.pi * q * s2 * x), 0)

fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3))
for q, c in [(0.1, GREEN), (0.5, BLUE)]:
    N = 400; T = int(N / q)
    X = rng.standard_normal((N, T))
    lam = np.linalg.eigvalsh(X @ X.T / T)
    axs[0].hist(lam, bins=60, density=True, color=c, alpha=0.4, label=f"q = {q}")
    x = np.linspace(0.001, 3.2, 800)
    axs[0].plot(x, mp_density(x, q), color=c, lw=2)
axs[0].axvline(1, color=GREY, ls="--", lw=1.2, label="true eigenvalues (all 1)")
axs[0].set_xlabel("eigenvalue"); axs[0].set_ylabel("density"); axs[0].legend(fontsize=8)
axs[0].set_title("pure noise: Marchenko–Pastur", fontsize=10)

# Factor model "stock returns": market + 4 sectors + idiosyncratic noise, N = 400 stocks, T = 1000 days.
N, T = 400, 1000
market = rng.standard_normal(T); sectors = rng.standard_normal((4, T))
beta = 1 + 0.2 * rng.standard_normal(N); sec = rng.integers(0, 4, N)
R = 0.35 * np.outer(beta, market) + 0.3 * sectors[sec] + rng.standard_normal((N, T))
R = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
lam = np.linalg.eigvalsh(R @ R.T / T)
q = N / T
s2 = 1 - lam[lam > (1 + np.sqrt(q)) ** 2 * 1.2].sum() / N         # variance left for the noise bulk
axs[1].hist(lam[lam < 4], bins=50, density=True, color=BLUE, alpha=0.45, label="correlation eigenvalues")
x = np.linspace(0.01, 4, 800)
axs[1].plot(x, mp_density(x, q, s2), color=ORANGE, lw=2, label=f"MP fit (σ² = {s2:.2f})")
big = lam[lam > 4]
axs[1].plot(np.full_like(big, 3.9), np.linspace(0.15, 0.45, len(big)), ">", color=PINK, ms=6,
            label="outliers: " + ", ".join(f"{v:.0f}" for v in big[::-1]))
axs[1].set_xlim(0, 4.1); axs[1].set_xlabel("eigenvalue"); axs[1].legend(fontsize=8)
axs[1].set_title("market + 4 sectors + noise (N=400, T=1000)", fontsize=10)
save(fig, "mp-spectra")
print("outliers:", np.round(lam[lam > (1 + np.sqrt(q))**2 * s2 + 0.05][::-1], 2), "lambda+ =", round(s2 * (1 + np.sqrt(q))**2, 3))
