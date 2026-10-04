import numpy as np
from scipy import stats
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(3)
sig = lambda t: 1 / (1 + np.exp(-t))

def logistic_mle(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(X @ b); W = p * (1 - p)
        step = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-8: break
    eta = X @ b; p = sig(eta)
    ll = np.sum(y * eta - np.logaddexp(0, eta))                      # stable log-likelihood
    return b, np.linalg.inv(X.T @ ((p * (1 - p))[:, None] * X)), ll


n, p = 1000, 200
beta = np.r_[np.full(p // 2, np.sqrt(5 * n / (p // 2))), np.zeros(p // 2)]
nz, z0 = [], []
for rep in range(30):
    X = rng.standard_normal((n, p)) / np.sqrt(n); y = (rng.uniform(size=n) < sig(X @ beta)).astype(float)
    b, V, _ = logistic_mle(X, y); se = np.sqrt(np.diag(V))
    nz.extend(b[: p // 2]); z0.extend(b[p // 2:] / se[p // 2:])
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2))
axs[0].hist(nz, bins=50, density=True, color=ORANGE, alpha=0.5, label="MLE of the nonzero coefficients")
axs[0].axvline(beta[0], color="#222", lw=2, ls="--", label=f"true value {beta[0]:.2f}")
axs[0].axvline(np.mean(nz), color=ORANGE, lw=2, label=f"mean of MLE {np.mean(nz):.2f}")
axs[0].set_title("κ = p/n = 0.2: the MLE is inflated", fontsize=10); axs[0].legend(fontsize=8)
x = np.linspace(-5, 5, 300)
axs[1].hist(z0, bins=60, density=True, color=BLUE, alpha=0.5, label="null coefs: MLE / classical SE")
axs[1].plot(x, stats.norm.pdf(x), color="#222", lw=2, label="N(0, 1) (classical theory)")
axs[1].set_title(f"…and more variable than the Fisher information says (sd {np.std(z0):.2f})", fontsize=10); axs[1].legend(fontsize=8)
save(fig, "hdmle-hist")
