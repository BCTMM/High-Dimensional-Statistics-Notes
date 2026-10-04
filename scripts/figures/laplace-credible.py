import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(2)
sig = lambda t: 1 / (1 + np.exp(-t))
tau = 3.0                                                   # prior: (a, b) ~ N(0, tau^2 I)

def log_post(a, b, x, y):                                   # intercept a, slope b (vectorized over grids)
    eta = a[..., None] + b[..., None] * x
    return np.sum(y * eta - np.logaddexp(0, eta), -1) - (a**2 + b**2) / (2 * tau**2)

def laplace(x, y):
    """MAP by Newton, covariance = inverse Hessian of the negative log-posterior at the MAP."""
    X = np.column_stack([np.ones_like(x), x]); w = np.zeros(2)
    for _ in range(50):
        p = sig(X @ w)
        H = X.T @ ((p * (1 - p))[:, None] * X) + np.eye(2) / tau**2
        w = w + np.linalg.solve(H, X.T @ (y - p) - w / tau**2)
    p = sig(X @ w); H = X.T @ ((p * (1 - p))[:, None] * X) + np.eye(2) / tau**2
    return w, np.linalg.inv(H)


fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6))
for ax, n in zip(axs, [20, 200]):
    x = rng.standard_normal(n); y = (rng.uniform(size=n) < sig(-0.5 + 2 * x)).astype(float)
    w, S = laplace(x, y); sa, sb = np.sqrt(np.diag(S))
    A, B = np.meshgrid(np.linspace(w[0] - 5 * sa, w[0] + 5 * sa, 250), np.linspace(w[1] - 5 * sb, w[1] + 6 * sb, 250), indexing="ij")
    lp = log_post(A, B, x, y); post = np.exp(lp - lp.max())
    d = np.stack([A - w[0], B - w[1]], -1); Si = np.linalg.inv(S)
    gauss = np.exp(-0.5 * np.einsum("...i,ij,...j->...", d, Si, d))
    lev = [np.exp(-0.5 * c**2) for c in (2.45, 1.67)][::-1]
    ax.contour(A, B, post, levels=sorted(lev), colors=BLUE, linewidths=2)
    ax.contour(A, B, gauss, levels=sorted(lev), colors=ORANGE, linewidths=1.5, linestyles="--")
    ax.plot(-0.5, 2, "k*", ms=10)
    ax.set_xlabel("intercept a"); ax.set_ylabel("slope b"); ax.set_title(f"n = {n}", fontsize=10)
axs[0].plot([], [], color=BLUE, lw=2, label="exact posterior"); axs[0].plot([], [], color=ORANGE, ls="--", label="Laplace approximation")
axs[0].plot([], [], "k*", label="true parameters"); axs[0].legend(fontsize=8)
save(fig, "laplace-contours")
