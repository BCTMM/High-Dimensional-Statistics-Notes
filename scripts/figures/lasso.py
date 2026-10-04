import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(1)
def lasso_cd(X, y, lam, b=None, iters=100):
    n, p = X.shape; b = np.zeros(p) if b is None else b.copy(); col = (X**2).sum(0) / n; r = y - X @ b
    for _ in range(iters):
        for j in range(p):
            r += X[:, j] * b[j]; z = X[:, j] @ r / n
            b[j] = np.sign(z) * max(abs(z) - lam, 0) / col[j]; r -= X[:, j] * b[j]
    return b
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
# geometry: elliptical loss contours around the OLS point and the l1 / l2 balls
ax = axs[0]; t = np.linspace(0, 2 * np.pi, 400); c = np.array([1.8, 0.35])
A = np.array([[1.0, 0.6], [0.6, 1.0]])
d0 = c - np.array([1.0, 0.0]); lev0 = d0 @ A @ d0           # contour through the corner (1, 0)
for lev in [0.15 * lev0, 0.5 * lev0, lev0]:
    w, V = np.linalg.eigh(A); pts = (V @ (np.sqrt(lev / w)[:, None] * np.vstack([np.cos(t), np.sin(t)]))).T + c
    ax.plot(pts[:, 0], pts[:, 1], color=GREY, lw=1)
ax.fill([1, 0, -1, 0], [0, 1, 0, -1], color=BLUE, alpha=0.25, label=r"$\|\beta\|_1 \leq 1$")
ax.plot(np.cos(t), np.sin(t), color=ORANGE, lw=1.8, label=r"$\|\beta\|_2 \leq 1$")
ax.plot(*c, "k+", ms=10); ax.text(c[0] + 0.05, c[1] + 0.05, "OLS", fontsize=9)
ax.plot(1, 0, "o", color=BLUE, ms=7); ax.text(1.05, -0.25, "LASSO: on a corner\n(β₂ = 0)", fontsize=8, color=BLUE)
ax.set_aspect("equal"); ax.set_xlim(-1.4, 2.6); ax.set_ylim(-1.4, 2.0); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel(r"$\beta_1$"); ax.set_ylabel(r"$\beta_2$"); ax.set_title("why the ℓ1 ball gives zeros", fontsize=10)
# path
n, p, s = 100, 200, 8
X = rng.standard_normal((n, p)); beta = np.zeros(p); beta[:s] = [3, -3, 2, -2, 1.5, -1.5, 1, -1]
y = X @ beta + rng.standard_normal(n)
lams = np.logspace(0.6, -2.5, 60); path = []; b = None
for lam in lams:
    b = lasso_cd(X, y, lam, b); path.append(b.copy())
path = np.array(path); ax = axs[1]
for j in range(p):
    ax.plot(lams, path[:, j], color=(BLUE if j < s else GREY), lw=1.6 if j < s else 0.6, alpha=1 if j < s else 0.6)
ax.axvline(np.sqrt(2 * np.log(p) / n), color=GREEN, ls="--", lw=1.4, label=r"$\lambda = \sigma\sqrt{2\log p/n}$")
ax.axvline(0.070, color=ORANGE, ls=":", lw=1.6, label=r"$\lambda_{CV}$")
ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel(r"$\lambda$ (decreasing →)"); ax.set_ylabel("coefficient")
ax.plot([], [], color=BLUE, label="true signals"); ax.plot([], [], color=GREY, label="nulls")
ax.legend(fontsize=8, loc="upper left"); ax.set_title("LASSO path (n = 100, p = 200)", fontsize=10)
save(fig, "lasso-geometry-path")
