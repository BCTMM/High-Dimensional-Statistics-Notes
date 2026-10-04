import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(0)
r = np.linspace(-6, 6, 600); k, c = 1.345, 4.685
rho = {"squared": r**2 / 2, "absolute": np.abs(r),
       "Huber": np.where(np.abs(r) <= k, r**2 / 2, k * np.abs(r) - k**2 / 2),
       "Tukey": np.where(np.abs(r) <= c, c**2 / 6 * (1 - (1 - (r / c) ** 2) ** 3), c**2 / 6)}
psi = {"squared": r, "absolute": np.sign(r), "Huber": np.clip(r, -k, k),
       "Tukey": np.where(np.abs(r) <= c, r * (1 - (r / c) ** 2) ** 2, 0)}
cols = {"squared": GREY, "absolute": GREEN, "Huber": BLUE, "Tukey": ORANGE}
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1))
for name in rho:
    axs[0].plot(r, rho[name], color=cols[name], lw=2, label=name); axs[1].plot(r, psi[name], color=cols[name], lw=2, label=name)
axs[0].set_ylim(0, 8); axs[0].set_title(r"loss $\rho(r)$", fontsize=10); axs[0].legend(fontsize=8)
axs[1].set_ylim(-3, 3); axs[1].set_title(r"influence $\psi(r) = \rho'(r)$", fontsize=10)
def m_reg(X, y, kind, iters=100):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    if kind == "tukey": b = m_reg(X, y, "huber")
    for _ in range(iters):
        res = y - X @ b; s = np.median(np.abs(res - np.median(res))) / 0.6745; a = np.abs(res / s)
        w = np.minimum(1, k / np.maximum(a, 1e-12)) if kind == "huber" else np.where(a < c, (1 - (a / c) ** 2) ** 2, 0)
        b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    return b
n = 200; x = rng.uniform(0, 10, n); X = np.column_stack([np.ones(n), x])
y = 1 + 0.5 * x + rng.standard_normal(n); bad = rng.choice(n, 20, replace=False); y[bad] = 30 + 5 * rng.standard_normal(20)
axs[2].plot(x, y, ".", color=GREY, ms=4); xs = np.array([0, 10])
for kind, col, lab in [("ols", GREY, "OLS"), ("huber", BLUE, "Huber"), ("tukey", ORANGE, "Tukey")]:
    b = np.linalg.lstsq(X, y, rcond=None)[0] if kind == "ols" else m_reg(X, y, kind)
    axs[2].plot(xs, b[0] + b[1] * xs, color=col, lw=2, ls="--" if kind == "ols" else "-", label=lab)
axs[2].set_title("10% gross outliers", fontsize=10); axs[2].legend(fontsize=8)
save(fig, "rob-functions")
