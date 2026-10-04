import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save

# Figure 1: projection geometry in R^3.
fig = plt.figure(figsize=(5.2, 4.2))
ax = fig.add_subplot(projection="3d")
x1, x2 = np.array([1.0, 0.0, 0.0]), np.array([0.3, 1.0, 0.0])
y = np.array([0.9, 0.8, 0.9])
X = np.column_stack([x1, x2])
yhat = X @ np.linalg.solve(X.T @ X, X.T @ y)
s, t = np.meshgrid(np.linspace(-0.2, 1.3, 2), np.linspace(-0.2, 1.1, 2))
P = s[..., None] * x1 + t[..., None] * x2
ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], alpha=0.15, color=BLUE)
for v, c, lab in [(x1, BLUE, r"$x_1$"), (x2, BLUE, r"$x_2$"), (y, "#222", r"$y$"), (yhat, ORANGE, r"$\hat y = Py$")]:
    ax.quiver(0, 0, 0, *v, color=c, arrow_length_ratio=0.08, lw=2)
    ax.text(*(v * 1.08), lab, color=c, fontsize=11)
ax.plot(*np.column_stack([yhat, y]), color=GREEN, lw=2, ls="--")
ax.text(*((y + yhat) / 2 + np.array([0.05, 0, 0])), r"$e = My$", color=GREEN, fontsize=11)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.1); ax.set_zlim(0, 1)
ax.view_init(elev=22, azim=-60)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
ax.text2D(0.02, 0.92, r"col$(X)$", transform=ax.transAxes, color=BLUE)
save(fig, "ols-projection")

# Figure 2: excess prediction error of OLS vs p/n.
rng = np.random.default_rng(0)
n, sigma = 200, 1.0
ps = np.arange(10, 191, 10)
emp = []
for p in ps:
    errs = []
    for _ in range(40):
        X = rng.standard_normal((n, p)); beta = rng.standard_normal(p) / np.sqrt(p)
        y = X @ beta + sigma * rng.standard_normal(n)
        bhat = np.linalg.lstsq(X, y, rcond=None)[0]
        errs.append(np.sum((bhat - beta) ** 2))          # = E[(x'(bhat - beta))^2] for x ~ N(0, I)
    emp.append(np.mean(errs))
q = np.linspace(0.02, 0.96, 300)
fig, ax = plt.subplots(figsize=(7, 3.0))
ax.plot(ps / n, emp, "o", color=BLUE, label="simulation (n = 200)")
ax.plot(q, sigma**2 * q / (1 - q), color=ORANGE, lw=2, label=r"$\sigma^2\, q/(1-q)$")
ax.set_xlabel("q = p / n")
ax.set_ylabel("excess test error")
ax.set_ylim(0, 25)
ax.legend()
save(fig, "ols-variance-blowup")
