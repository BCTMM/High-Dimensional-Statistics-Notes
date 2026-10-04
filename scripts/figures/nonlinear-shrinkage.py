import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(2)
S5 = np.sqrt(5)

def nonlinear_shrinkage(E, n):
    """Analytical nonlinear shrinkage (Ledoit & Wolf, 2020), case p <= n."""
    lam, U = np.linalg.eigh(E); p = len(lam); q = p / n
    h = n ** (-1 / 3); hj = lam * h                                   # locally adaptive bandwidths
    x = (lam[:, None] - lam[None, :]) / hj[None, :]                   # x[i, j] = (λ_i - λ_j) / h_j
    K = np.where(np.abs(x) < S5, 3 / (4 * S5) * (1 - x**2 / 5), 0.0)  # Epanechnikov kernel
    with np.errstate(divide="ignore", invalid="ignore"):
        HK = -3 * x / (10 * np.pi) + 3 / (4 * S5 * np.pi) * (1 - x**2 / 5) * np.log(np.abs((S5 - x) / (S5 + x)))
    HK = np.nan_to_num(HK)
    f = (K / hj[None, :]).mean(1)                                     # density of sample eigenvalues at λ_i
    Hf = (HK / hj[None, :]).mean(1)                                   # its Hilbert transform
    d = lam / ((np.pi * q * lam * f) ** 2 + (1 - q - np.pi * q * lam * Hf) ** 2)
    return (U * d) @ U.T, lam, d, U

def ledoit_wolf_linear(E, X):
    n, p = X.shape; m = np.trace(E) / p; d2 = np.sum((E - m * np.eye(p)) ** 2) / p
    b2 = min(sum(np.sum((np.outer(x, x) - E) ** 2) for x in X) / p / n**2, d2)
    return (1 - b2 / d2) * E + b2 / d2 * m * np.eye(p)


p, n = 200, 400
t = np.r_[np.ones(100), 3 * np.ones(60), 10 * np.ones(40)]
O = np.linalg.qr(rng.standard_normal((p, p)))[0]; Sigma = (O * t) @ O.T
X = rng.standard_normal((n, p)) @ np.linalg.cholesky(Sigma).T; E = X.T @ X / n
NL, lam, d, U = nonlinear_shrinkage(E, n)
orc = np.einsum("ij,ik,kj->j", U, Sigma, U)
LW = ledoit_wolf_linear(E, X); dlw = np.einsum("ij,ik,kj->j", U, LW, U)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(lam, orc, "o", color=GREY, ms=3.5, alpha=0.7, label=r"oracle $u_i^\top\Sigma u_i$")
ax.plot(lam, d, color=BLUE, lw=2.2, label="analytical nonlinear shrinkage")
ax.plot(lam, dlw, color=GREEN, lw=1.6, ls="--", label="linear Ledoit–Wolf")
ax.plot(lam, lam, color=PINK, lw=1, label="sample eigenvalues (no shrinkage)")
for v in [1, 3, 10]: ax.axhline(v, color=ORANGE, lw=0.8, ls=":")
ax.set_xlabel(r"sample eigenvalue $\lambda_i$"); ax.set_ylabel("shrunk eigenvalue")
ax.set_title("true spectrum: 1 (50%), 3 (30%), 10 (20%);  p = 200, n = 400", fontsize=10); ax.legend(fontsize=8)
save(fig, "nls-curve")
