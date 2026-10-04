import numpy as np
from scipy.special import gammaln
from _style import BLUE, ORANGE, GREEN, PINK, plt, save

# Panel 1: volume of the unit ball relative to the cube [-1,1]^d, and mass in the outer 1% shell.
d = np.arange(1, 101)
ball_over_cube = np.exp(d / 2 * np.log(np.pi) - gammaln(d / 2 + 1) - d * np.log(2))
shell = 1 - 0.99 ** d
fig, axs = plt.subplots(1, 2, figsize=(8, 3.1))
axs[0].semilogy(d, ball_over_cube, color=BLUE)
axs[0].set_xlabel("dimension d")
axs[0].set_title("vol(unit ball) / vol([−1,1]$^d$)", fontsize=10)
d2 = np.arange(1, 1001)
axs[1].plot(d2, 1 - 0.99 ** d2, color=ORANGE)
axs[1].set_xlabel("dimension d")
axs[1].set_title("fraction of ball volume in outer 1% shell", fontsize=10)
save(fig, "hdg-volume")

# Panel 2: distances between random points concentrate.
rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(7, 3.1))
for dim, c in zip([2, 10, 100, 1000], [BLUE, GREEN, ORANGE, PINK]):
    X = rng.standard_normal((400, dim))
    sq = (X**2).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    dist = np.sqrt(np.maximum(D2[np.triu_indices(400, 1)], 0))
    ax.hist(dist / np.sqrt(2 * dim), bins=120, range=(0, 2), density=True, histtype="step", lw=1.8, color=c, label=f"d = {dim}")
ax.set_xlabel(r"pairwise distance $\|x_i-x_j\|\,/\,\sqrt{2d}$")
ax.set_ylabel("density")
ax.legend()
save(fig, "hdg-distances")
