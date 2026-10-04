import numpy as np
from scipy.optimize import linprog
from scipy import stats
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(0)
def qreg(X, y, tau):
    n, p = X.shape
    c = np.r_[np.zeros(p), tau * np.ones(n), (1 - tau) * np.ones(n)]
    res = linprog(c, A_eq=np.hstack([X, np.eye(n), -np.eye(n)]), b_eq=y, bounds=[(None, None)] * p + [(0, None)] * (2 * n), method="highs")
    return res.x[:p]
n = 500
x = rng.uniform(0, 3, n); X = np.column_stack([np.ones(n), x])
y = 1 + 2 * x + (0.5 + x) * rng.standard_normal(n)
xs = np.linspace(0, 3, 50)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(x, y, ".", color=GREY, ms=3, alpha=0.6)
for tau, c in [(0.1, BLUE), (0.5, GREEN), (0.9, ORANGE)]:
    b = qreg(X, y, tau); z = stats.norm.ppf(tau)
    ax.plot(xs, b[0] + b[1] * xs, color=c, lw=2.2, label=f"τ = {tau}: fitted")
    ax.plot(xs, (1 + 0.5 * z) + (2 + z) * xs, color=c, lw=1.2, ls="--")
b = np.linalg.lstsq(X, y, rcond=None)[0]
ax.plot(xs, b[0] + b[1] * xs, color=PINK, lw=1.5, ls=":", label="OLS (conditional mean)")
ax.plot([], [], color="#222", ls="--", lw=1.2, label="true conditional quantiles")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8, loc="upper left")
save(fig, "qr-fan")
