import numpy as np
from _style import BLUE, ORANGE, GREEN, plt, save

rng = np.random.default_rng(0)
n, sigma = 200, 0.3
G = rng.standard_normal((n, n))
E = sigma * (G + G.T) / np.sqrt(2 * n)             # GOE noise, ||E||_op ≈ 2·sigma
op, fro = np.linalg.norm(E, 2), np.linalg.norm(E, "fro")

fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
for ax, (name, A) in zip(axs, [("A with spread-out eigenvalues", np.diag(np.linspace(-3, 3, n))),
                               ("A = I  (all eigenvalues equal)", np.eye(n))]):
    shift = np.abs(np.linalg.eigvalsh(A + E) - np.linalg.eigvalsh(A))   # both sorted ascending
    print(f"{name}: ||E||_op = {op:.3f}, max shift = {shift.max():.3f}, RMS = {np.sqrt((shift**2).mean()):.3f}")
    ax.plot(np.arange(1, n + 1), shift, ".", color=BLUE, ms=4, label=r"$|\lambda_k(A+E)-\lambda_k(A)|$")
    ax.axhline(op, color=ORANGE, lw=2, label=r"Weyl: $\|E\|_{op}$")
    ax.axhline(fro / np.sqrt(n), color=GREEN, lw=2, ls="--", label=r"Hoffman–Wielandt RMS: $\|E\|_F/\sqrt{n}$")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("index k")
    ax.set_ylim(0, op * 1.35)
axs[0].set_ylabel("eigenvalue shift")
axs[0].legend(loc="upper left", fontsize=8)
save(fig, "cfw-weyl-shifts")
