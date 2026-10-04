import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(1)
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.4))
N = 1000
for nu, c in [(8.0, BLUE), (3.0, GREEN), (1.5, ORANGE)]:
    A = np.triu(rng.standard_t(nu, size=(N, N)), 1); A = A + A.T
    lam = np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1]
    lam /= np.median(lam)
    axs[0].loglog(lam, np.arange(1, N + 1) / N, color=c, lw=2, label=f"Student-t({nu:g}) entries")
axs[0].set_xlim(0.5, 40); axs[0].set_xlabel("|eigenvalue| / median"); axs[0].set_ylabel("fraction of eigenvalues above")
axs[0].set_title("Wigner matrices: heavy-tailed entries → heavy-tailed spectra", fontsize=9.5); axs[0].legend(fontsize=8)
n_assets, T = 200, 1000; q = n_assets / T
R = (1 / np.sqrt(rng.chisquare(3, size=(T, 1)) / 3)) * rng.standard_normal((T, n_assets))
ev_p = np.linalg.eigvalsh(np.corrcoef(R, rowvar=False))
S = R / np.linalg.norm(R, axis=1, keepdims=True); ev_s = np.linalg.eigvalsh(np.corrcoef(S, rowvar=False))
lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
x = np.linspace(lo, hi, 300); mp = np.sqrt((hi - x) * (x - lo)) / (2 * np.pi * q * x)
bins = np.linspace(0, 4, 60)
axs[1].hist(np.minimum(ev_p, 3.99), bins=bins, density=True, color=ORANGE, alpha=0.45, label=f"Pearson (largest = {ev_p[-1]:.1f})")
axs[1].hist(ev_s, bins=bins, density=True, histtype="step", color=BLUE, lw=2, label="spatial-sign correlation")
axs[1].plot(x, mp, color="#222", lw=1.8, label="Marchenko–Pastur")
axs[1].set_xlabel("eigenvalue (values above 4 piled at the right edge)"); axs[1].legend(fontsize=8)
axs[1].set_title("uncorrelated returns with a common heavy-tailed volatility", fontsize=9.5)
save(fig, "ht-spectra")
