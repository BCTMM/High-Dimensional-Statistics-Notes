import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(5)
def lasso_cd(X, y, lam, iters=100):
    n, p = X.shape; b = np.zeros(p); r = y.copy(); col = (X**2).sum(0) / n
    for _ in range(iters):
        for j in range(p):
            r += X[:, j] * b[j]; z = X[:, j] @ r / n
            b[j] = np.sign(z) * max(abs(z) - lam, 0) / col[j]; r -= X[:, j] * b[j]
    return b

def gaussian_knockoffs(X, Sigma):
    """Equicorrelated model-X knockoffs for rows X_i ~ N(0, Sigma) (Sigma a correlation matrix)."""
    p = len(Sigma); s = min(2 * np.linalg.eigvalsh(Sigma)[0], 1.0) * 0.999
    D = s * np.eye(p); Si_D = np.linalg.solve(Sigma, D)
    mean = X - X @ Si_D                                        # E[X~ | X] = X - X Σ^{-1} D
    cov = 2 * D - D @ Si_D                                     # Cov[X~ | X] = 2D - D Σ^{-1} D
    return mean + rng.standard_normal(X.shape) @ np.linalg.cholesky(cov).T

def knockoff_plus_threshold(W, q):
    for t in np.sort(np.abs(W[W != 0])):
        if (1 + np.sum(W <= -t)) / max(1, np.sum(W >= t)) <= q:
            return t
    return np.inf


n, p, k, q = 600, 150, 25, 0.1
Sigma = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
X = rng.standard_normal((n, p)) @ np.linalg.cholesky(Sigma).T
beta = np.zeros(p); S = rng.choice(p, k, replace=False); beta[S] = rng.choice([-1, 1], k) * 0.18
y = X @ beta + rng.standard_normal(n)
Xk = gaussian_knockoffs(X, Sigma); b = lasso_cd(np.hstack([X, Xk]), y, lam=0.05)
a, ak = np.abs(b[:p]), np.abs(b[p:]); W = a - ak; t = knockoff_plus_threshold(W, q)
sig = np.isin(np.arange(p), S)
fig, ax = plt.subplots(figsize=(5.6, 4.2))
m = max(a.max(), ak.max()) * 1.05
ax.plot([0, m], [0, m], color=GREY, lw=1)
ax.fill_between([t, m], [0, m - t], [0, 0], color=GREEN, alpha=0.12, label=f"selected: |β| − |β̃| ≥ τ = {t:.3f}")
ax.plot(a[~sig], ak[~sig], "o", color=GREY, ms=4, alpha=0.7, label="null variables")
ax.plot(a[sig], ak[sig], "o", color=BLUE, ms=5, label="true signals")
ax.set_xlabel(r"$|\hat\beta_j|$ (original variable)"); ax.set_ylabel(r"$|\hat\beta_{\tilde j}|$ (its knockoff)")
ax.set_xlim(0, m); ax.set_ylim(0, m); ax.legend(fontsize=8, loc="upper right")
ax.set_title("nulls are symmetric about the diagonal; signals are not", fontsize=9.5)
save(fig, "knockoff-scatter")
