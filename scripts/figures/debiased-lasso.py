import numpy as np
from scipy import stats
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(1)
def lasso_cd(X, y, lam, iters=100):
    n, p = X.shape; b = np.zeros(p); r = y.copy(); col = (X**2).sum(0) / n
    for _ in range(iters):
        for j in range(p):
            r += X[:, j] * b[j]; z = X[:, j] @ r / n
            b[j] = np.sign(z) * max(abs(z) - lam, 0) / col[j]; r -= X[:, j] * b[j]
    return b


n, p, s, rho = 300, 500, 10, 0.5
Sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p))); Theta = np.linalg.inv(Sigma); L = np.linalg.cholesky(Sigma)
beta = np.zeros(p); beta[:s] = 1.0; lam = 2 * np.sqrt(2 * np.log(p) / n)
act_l, act_d, znull = [], [], []
for rep in range(40):
    X = rng.standard_normal((n, p)) @ L.T; y = X @ beta + rng.standard_normal(n)
    b = lasso_cd(X, y, lam); r = y - X @ b; sh = np.sqrt(r @ r / (n - np.sum(b != 0)))
    bd = b + Theta @ X.T @ r / n; se = sh * np.sqrt(np.einsum("ij,jk,ik->i", Theta, X.T @ X / n, Theta) / n)
    act_l.extend(b[:s]); act_d.extend(bd[:s]); znull.extend(((bd - beta) / se)[s:])
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2))
bins = np.linspace(0.4, 1.4, 40)
axs[0].hist(act_l, bins=bins, density=True, color=ORANGE, alpha=0.5, label="LASSO")
axs[0].hist(act_d, bins=bins, density=True, color=BLUE, alpha=0.5, label="debiased LASSO")
axs[0].axvline(1, color="#222", lw=1.5, ls="--", label="true value")
axs[0].set_title("estimates of the active coefficients (true = 1)", fontsize=10); axs[0].legend(fontsize=8)
x = np.linspace(-4, 4, 300)
axs[1].hist(znull, bins=60, density=True, color=GREEN, alpha=0.5, label="debiased, standardized (null coefs)")
axs[1].plot(x, stats.norm.pdf(x), color="#222", lw=2, label="N(0, 1)")
axs[1].set_title(f"p = {p} > n = {n}: Gaussian, correctly scaled", fontsize=10); axs[1].legend(fontsize=8)
save(fig, "debiased-hist")
