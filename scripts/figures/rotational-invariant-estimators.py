import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(0)
def rie(E, q):
    """Rotationally invariant estimator (Ledoit–Péché / Bouchaud–Potters) with the Bun et al. small-eigenvalue fix."""
    lam, U = np.linalg.eigh(E); N = len(lam)
    out = lam > 1.05 * lam.mean() * (1 + np.sqrt(q)) ** 2              # isolated outliers (above the bulk)
    z = np.where(out, lam, lam - 1j / np.sqrt(N))                      # bulk: just below the real axis, eta = N^{-1/2}
    D = z[:, None] - lam[None, :]
    D[out, np.flatnonzero(out)] = np.inf                               # outliers: drop the self-term, evaluate on the real axis
    g = (1 / D).mean(1)                                                # sample Stieltjes transform g(z_i)
    xi = lam / np.abs(1 - q + q * z * g) ** 2                          # RIE: estimate of u_i' Σ u_i
    s2 = lam[0] / (1 - np.sqrt(q)) ** 2                                # MP "null" matched at the bottom edge
    lo, hi = s2 * (1 - np.sqrt(q)) ** 2, s2 * (1 + np.sqrt(q)) ** 2
    g0 = ((z - s2 * (1 - q)) - np.sqrt(z - lo) * np.sqrt(z - hi)) / (2 * q * s2 * z)
    xi = np.where(out, xi, xi * np.maximum(1, s2 * np.abs(1 - q + q * z * g0) ** 2 / lam))  # undo the bias at small λ
    xi *= lam.sum() / xi.sum()                                         # keep the trace
    return (U * xi) @ U.T, lam, xi, U


N, T = 300, 600; q = N / T
t = np.sort(1 / np.linspace(0.05, 1, N)); t /= t.mean()
O = np.linalg.qr(rng.standard_normal((N, N)))[0]; Sigma = (O * t) @ O.T
X = rng.standard_normal((T, N)) @ np.linalg.cholesky(Sigma).T; E = X.T @ X / T
R, lam, xi, U = rie(E, q)
oracle = np.einsum("ij,ik,kj->j", U, Sigma, U)
n, p = X.shape; m = lam.mean(); d2 = np.sum((E - m * np.eye(p)) ** 2) / p
b2 = min(sum(np.sum((np.outer(x, x) - E) ** 2) for x in X) / p / n**2, d2); rho = b2 / d2
edge = (1 + np.sqrt(q)) ** 2; keep = lam > edge
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(lam, oracle, "o", color=GREY, ms=3.5, alpha=0.7, label=r"oracle $u_i^\top\Sigma u_i$")
ax.plot(lam, xi, color=BLUE, lw=2.2, label="RIE (from the sample alone)")
ax.plot(lam, (1 - rho) * lam + rho * m, color=GREEN, lw=1.6, ls="--", label=f"Ledoit–Wolf (linear, ρ = {rho:.2f})")
ax.plot(lam, np.where(keep, lam, lam[~keep].mean()), color=ORANGE, lw=1.6, ls=":", label="clipping")
ax.plot(lam, lam, color=PINK, lw=1, label="no cleaning (identity)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"sample eigenvalue $\lambda_i$"); ax.set_ylabel("cleaned eigenvalue")
ax.set_title(f"continuous power-law spectrum, N = {N}, T = {T}", fontsize=10); ax.legend(fontsize=8)
save(fig, "rie-curve")
