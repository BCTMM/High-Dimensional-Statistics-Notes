import numpy as np
from scipy.optimize import linprog
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(0)
def qreg(F, y, tau):
    n, p = F.shape
    c = np.r_[np.zeros(p), tau * np.ones(n), (1 - tau) * np.ones(n)]
    res = linprog(c, A_eq=np.hstack([F, np.eye(n), -np.eye(n)]), b_eq=y, bounds=[(None, None)] * p + [(0, None)] * (2 * n), method="highs")
    return res.x[:p]
def make(n):
    x = rng.uniform(0, 4, n); return x, np.sin(2 * x) + (0.2 + 0.4 * x) * (rng.exponential(1.0, n) - 1.0)
centers = np.linspace(0, 4, 20)
feats = lambda x: np.column_stack([np.ones_like(x)] + [np.exp(-(x - c) ** 2 / 0.08) for c in centers])
x_tr, y_tr = make(200); x_cal, y_cal = make(1000); x_te, y_te = make(600)
alpha = 0.1
b_lo, b_hi = qreg(feats(x_tr), y_tr, alpha / 2), qreg(feats(x_tr), y_tr, 1 - alpha / 2)
b_mid = qreg(feats(x_tr), y_tr, 0.5)
k = int(np.ceil(1001 * 0.9))
Q = np.sort(np.maximum(feats(x_cal) @ b_lo - y_cal, y_cal - feats(x_cal) @ b_hi))[k - 1]
Qa = np.sort(np.abs(y_cal - feats(x_cal) @ b_mid))[k - 1]
xs = np.linspace(0, 4, 400); F = feats(xs)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
for ax, title, L, U, c in [(axs[0], "split conformal around the median (constant width)", F @ b_mid - Qa, F @ b_mid + Qa, ORANGE),
                           (axs[1], f"CQR (Q = {Q:+.2f}): adaptive and asymmetric", F @ b_lo - Q, F @ b_hi + Q, BLUE)]:
    ax.plot(x_te, y_te, ".", color=GREY, ms=3, alpha=0.6)
    ax.fill_between(xs, L, U, color=c, alpha=0.25)
    if c == BLUE:
        ax.plot(xs, F @ b_lo, color=c, lw=1, ls="--", label="raw quantile fits (5%, 95%)"); ax.plot(xs, F @ b_hi, color=c, lw=1, ls="--")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_title(title, fontsize=9.5); ax.set_xlabel("x")
save(fig, "cqr-bands")
