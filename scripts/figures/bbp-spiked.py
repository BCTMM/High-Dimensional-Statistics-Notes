import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(3)
N, T = 600, 1200; q = N / T
ths = np.linspace(0, 3, 300)
lam_th = np.where(ths > np.sqrt(q), (1 + ths) * (1 + q / np.maximum(ths, 1e-9)), (1 + np.sqrt(q)) ** 2)
ov_th = np.where(ths > np.sqrt(q), (1 - q / np.maximum(ths, 1e-9) ** 2) / (1 + q / np.maximum(ths, 1e-9)), 0)
sim_t, sim_l, sim_o = [], [], []
for th in np.linspace(0.1, 3, 16):
    Z = rng.standard_normal((N, T)); Z[0] *= np.sqrt(1 + th)
    w, V = np.linalg.eigh(Z @ Z.T / T)
    sim_t.append(th); sim_l.append(w[-1]); sim_o.append(V[0, -1] ** 2)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3))
axs[0].plot(ths, lam_th, color=ORANGE, lw=2, label="BBP prediction")
axs[0].plot(ths, 1 + ths, color=GREY, ls="--", lw=1.5, label="true spike eigenvalue 1 + θ")
axs[0].plot(sim_t, sim_l, "o", color=BLUE, ms=5, label=f"simulation (N = {N}, q = {q})")
axs[0].axvline(np.sqrt(q), color=GREEN, ls=":", lw=1.5)
axs[0].set_xlabel("θ"); axs[0].set_ylabel("top sample eigenvalue"); axs[0].legend(fontsize=8)
axs[1].plot(ths, ov_th, color=ORANGE, lw=2, label="BBP / Paul prediction")
axs[1].plot(sim_t, sim_o, "o", color=BLUE, ms=5, label="simulation")
axs[1].axvline(np.sqrt(q), color=GREEN, ls=":", lw=1.5, label=r"threshold $\sqrt{q}$")
axs[1].set_xlabel("θ"); axs[1].set_ylabel(r"$|\langle \hat u, v\rangle|^2$"); axs[1].legend(fontsize=8)
save(fig, "bbp-curves")
