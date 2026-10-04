import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(0)
soft = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0)
n, p, k = 200, 500, 10
X = rng.standard_normal((n, p)) / np.sqrt(n); b0 = np.zeros(p); b0[:k] = rng.choice([-3, 3], k)
y = X @ b0 + 0.1 * rng.standard_normal(n); lam = 0.3; L = np.linalg.norm(X, 2) ** 2
F = lambda b: 0.5 * np.sum((y - X @ b) ** 2) + lam * np.abs(b).sum()
def run(fast, iters):
    b = z = np.zeros(p); t = 1.0; h = []
    for _ in range(iters):
        bn = soft(z - X.T @ (X @ z - y) / L, lam / L)
        if fast:
            tn = (1 + np.sqrt(1 + 4 * t * t)) / 2; z = bn + (t - 1) / tn * (bn - b); t = tn
        else:
            z = bn
        b = bn; h.append(F(b))
    return np.array(h)
Fs = run(True, 5000)[-1]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
v = np.linspace(-3, 3, 400)
axs[0].plot(v, v, color=GREY, ls="--", lw=1.2, label="identity")
axs[0].plot(v, soft(v, 1), color=BLUE, lw=2, label=r"prox of $|\cdot|$ (soft threshold, $\lambda=1$)")
axs[0].plot(v, v / 2, color=GREEN, lw=2, label=r"prox of $\frac{1}{2}x^2$ (shrink by half)")
axs[0].plot(v, np.clip(v, -1, 1), color=ORANGE, lw=2, label="prox of indicator of [−1,1] (projection)")
axs[0].legend(fontsize=7.5); axs[0].set_xlabel("v"); axs[0].set_title("proximal maps in 1-D", fontsize=10)
i = np.arange(1, 301)
axs[1].semilogy(i, np.maximum(run(False, 300) - Fs, 1e-16), color=ORANGE, lw=2, label="ISTA")
axs[1].semilogy(i, np.maximum(run(True, 300) - Fs, 1e-16), color=BLUE, lw=2, label="FISTA")
axs[1].set_ylim(1e-12, 1e2); axs[1].set_xlabel("iteration"); axs[1].set_ylabel("objective − optimum")
axs[1].set_title("LASSO, n = 200, p = 500", fontsize=10); axs[1].legend()
save(fig, "prox-maps")
