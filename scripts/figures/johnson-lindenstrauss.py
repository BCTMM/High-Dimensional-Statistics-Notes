import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(1)
def pw(Y):
    sq = (Y**2).sum(1); D = sq[:, None] + sq[None, :] - 2 * Y @ Y.T; return D[np.triu_indices(len(Y), 1)]
n, d = 500, 5000
X = rng.standard_normal((n, d)); D0 = pw(X)
ks = np.unique(np.logspace(1.7, 3.6, 12).astype(int)); mx = []
for k in ks:
    r = pw(X @ rng.standard_normal((d, k)) / np.sqrt(k)) / D0; mx.append(np.max(np.abs(r - 1)))
    if k == ks[7]: r_show, k_show = r, k
eps = np.linspace(1e-3, 0.999, 4000); kk = 4 * np.log(n) / (eps**2 / 2 - eps**3 / 3)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2))
axs[0].loglog(ks, mx, "o", color=BLUE, label="observed max distortion")
axs[0].loglog(kk[kk < 5000], eps[kk < 5000], color=ORANGE, lw=2, label=r"guarantee: $k = 4\ln n/(\varepsilon^2/2-\varepsilon^3/3)$")
axs[0].set_xlabel("target dimension k"); axs[0].set_ylabel(r"max over all pairs of |ratio − 1|")
axs[0].set_title(f"n = {n} points, original d = {d}", fontsize=10); axs[0].legend(fontsize=8)
axs[1].hist(r_show, bins=80, density=True, color=GREEN, alpha=0.6)
axs[1].set_xlabel("squared-distance ratio after projection"); axs[1].set_title(f"all {len(r_show):,} pairs, k = {k_show}", fontsize=10)
save(fig, "jl-distortion")
