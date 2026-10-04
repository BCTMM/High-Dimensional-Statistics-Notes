import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save

rng = np.random.default_rng(0)
N = 1000
G = rng.standard_normal((N, N))
H = (G + G.T) / np.sqrt(2 * N)
lam = np.linalg.eigvalsh(H)
dens = lambda x, eta: np.array([-np.mean(1 / (xx + 1j * eta - lam)).imag / np.pi for xx in x])
sc = lambda x: np.sqrt(np.maximum(4 - x**2, 0)) / (2 * np.pi)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"width_ratios": [1.4, 1]})
x = np.linspace(-2.6, 2.6, 800)
axs[0].plot(x, sc(x), color="#222", lw=2.2, label="semicircle")
for eta, c in [(0.5, GREEN), (0.05, BLUE)]:
    axs[0].plot(x, dens(x, eta), color=c, lw=1.6, label=rf"$\eta = {eta}$")
axs[0].set_ylim(0, 0.45); axs[0].set_xlabel("x"); axs[0].set_ylabel("density")
axs[0].set_title(r"$-\frac{1}{\pi}\,\mathrm{Im}\,g_N(x+i\eta)$, whole spectrum", fontsize=10)
axs[0].legend(fontsize=8.5)

x = np.linspace(-0.05, 0.05, 1500)
axs[1].plot(x, dens(x, 0.0005), color=ORANGE, lw=1.4, label=r"$\eta = 0.0005$")
axs[1].plot(x, dens(x, 0.05), color=BLUE, lw=1.6, label=r"$\eta = 0.05$")
inwin = lam[np.abs(lam) < 0.05]
axs[1].plot(inwin, np.zeros_like(inwin), "|", color="#222", ms=14, mew=1.5, label="eigenvalues")
axs[1].set_ylim(0, 2.0); axs[1].set_xlabel("x")
axs[1].set_title("zoom: tiny η resolves single eigenvalues", fontsize=10)
axs[1].legend(fontsize=8.5, loc="upper right")
save(fig, "stj-inversion")
