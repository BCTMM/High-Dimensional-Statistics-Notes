import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(1)
N = 400
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for L, c in [(1, GREEN), (5, BLUE), (20, ORANGE)]:
    J = np.eye(N)
    for _ in range(L):
        J = rng.standard_normal((N, N)) / np.sqrt(N) @ J
    s = np.linalg.svd(J, compute_uv=False)
    ax.hist(np.log10(s), bins=np.linspace(-8, 1, 70), density=True, histtype="step", lw=2, color=c, label=f"Gaussian layers, depth {L}")
ax.axvline(0, color=PINK, lw=2.5, label="orthogonal layers (any depth): all = 1")
ax.set_xlabel(r"$\log_{10}$ singular value of the input–output Jacobian $J = W_L\cdots W_1$")
ax.set_xlim(-8, 1); ax.set_ylabel("density"); ax.legend(fontsize=8.5, loc="upper left")
save(fig, "fps-jacobian")
