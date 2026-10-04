import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(0)
def enet_cd(X, y, lam, alpha, iters=300):
    """(1/2n)||y - Xb||^2 + lam * (alpha ||b||_1 + (1-alpha)/2 ||b||^2); alpha = 1 is the LASSO."""
    n, p = X.shape; b = np.zeros(p); r = y.copy(); col = (X**2).sum(0) / n
    for _ in range(iters):
        for j in range(p):
            r += X[:, j] * b[j]; z = X[:, j] @ r / n
            b[j] = np.sign(z) * max(abs(z) - lam * alpha, 0) / (col[j] + lam * (1 - alpha))
            r -= X[:, j] * b[j]
    return b


n, p = 100, 40
Z = rng.standard_normal((n, 3))
X = np.hstack([Z[:, [g]] + 0.23 * rng.standard_normal((n, 5)) for g in range(3)] + [rng.standard_normal((n, p - 15))])
X = (X - X.mean(0)) / X.std(0)
y = X[:, :5].sum(1) * 0.4 + rng.standard_normal(n)
lams = np.logspace(0.5, -2, 50)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
for ax, (name, alpha) in zip(axs, [("LASSO (α = 1)", 1.0), ("elastic net (α = 0.3)", 0.3)]):
    P = np.array([enet_cd(X, y, l, alpha, iters=150) for l in lams])
    for j in range(p):
        col = BLUE if j < 5 else (GREEN if j < 15 else GREY)
        ax.plot(lams, P[:, j], color=col, lw=1.8 if j < 5 else 0.8)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_title(name, fontsize=10); ax.set_xlabel(r"$\lambda$ (decreasing →)")
axs[0].set_ylabel("coefficient")
axs[1].plot([], [], color=BLUE, label="relevant correlated group"); axs[1].plot([], [], color=GREEN, label="irrelevant correlated groups")
axs[1].plot([], [], color=GREY, label="independent nulls"); axs[1].legend(fontsize=8, loc="upper left")
save(fig, "enet-paths")
