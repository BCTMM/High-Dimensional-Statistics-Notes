import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save

rng = np.random.default_rng(2)

# 2x2 GOE: H = [[a, b], [b, c]], a, c ~ N(0, 2), b ~ N(0, 1)
m = 200_000
a, c = rng.normal(0, np.sqrt(2), m), rng.normal(0, np.sqrt(2), m)
b = rng.normal(0, 1, m)
s_goe = np.sqrt((a - c) ** 2 + 4 * b**2)           # eigenvalue gap
s_diag = np.abs(a - c)                             # same, but with b = 0
s = np.linspace(0, 10, 400)
fig, ax = plt.subplots(figsize=(7, 3.1))
ax.hist(s_goe, bins=100, range=(0, 10), density=True, alpha=0.45, color=BLUE, label="2×2 GOE gap")
ax.hist(s_diag, bins=100, range=(0, 10), density=True, histtype="step", lw=1.5, color=GREEN, label="diagonal only (b = 0)")
ax.plot(s, s / 4 * np.exp(-s**2 / 8), color=ORANGE, lw=2, label=r"$\frac{s}{4}e^{-s^2/8}$")
ax.plot(s, np.exp(-s**2 / 8) / np.sqrt(2 * np.pi), color=GREY, lw=1.5, ls="--", label=r"$\frac{1}{\sqrt{2\pi}}e^{-s^2/8}$")
ax.set_xlabel("gap s")
ax.set_ylabel("density")
ax.legend()
save(fig, "ge-2x2-gap")

# GOE eigenvalues vs independent points, both unfolded to unit mean spacing.
n = 1000
G = rng.standard_normal((n, n))
H = (G + G.T) / np.sqrt(2)
x = np.linalg.eigvalsh(H) / np.sqrt(n)                       # semicircle on [-2, 2]
F = lambda x: 0.5 + (x * np.sqrt(4 - x**2) / 4 + np.arcsin(x / 2)) / np.pi   # semicircle CDF
u = n * F(np.clip(x, -2, 2))
mid = u[(u > n / 2) & (u < n / 2 + 60)]
pois = np.sort(rng.uniform(0, 60, 60))
fig, axs = plt.subplots(2, 1, figsize=(7, 1.8), sharex=True)
axs[0].vlines(pois, 0, 1, color=GREEN, lw=1.3)
axs[0].set_ylabel("Poisson", rotation=0, ha="right", va="center")
axs[1].vlines(mid - mid[0], 0, 1, color=BLUE, lw=1.3)
axs[1].set_ylabel("GOE", rotation=0, ha="right", va="center")
for ax in axs:
    ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
axs[1].set_xlabel("position (units of mean spacing)")
save(fig, "ge-rug")
