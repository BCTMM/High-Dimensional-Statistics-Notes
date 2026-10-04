import numpy as np
from scipy.optimize import minimize_scalar
from _style import BLUE, ORANGE, GREEN, GREY, plt, save

rng = np.random.default_rng(0)
sigmoid = lambda z: 1 / (1 + np.exp(-z))
n = 20_000
s = rng.normal(0, 1.5, n)
y = rng.uniform(size=n) < sigmoid(s)
z = 2.5 * s + rng.normal(0, 1.0, n)
val, test = slice(0, n // 2), slice(n // 2, n)
nll = lambda p, y: -np.mean(np.where(y, np.log(p), np.log(1 - p)))
T = minimize_scalar(lambda T: nll(sigmoid(z[val] / T), y[val]), bounds=(0.05, 20), method="bounded").x

def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    m = np.array([p[idx == b].mean() for b in range(bins)])
    f = np.array([y[idx == b].mean() for b in range(bins)])
    w = np.array([np.mean(idx == b) for b in range(bins)])
    return m, f, w

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
for ax, (title, p, c) in zip(axs, [("raw model (overconfident)", sigmoid(z[test]), ORANGE),
                                    (f"after temperature scaling (T = {T:.2f})", sigmoid(z[test] / T), BLUE)]):
    m, f, w = reliability(p, y[test])
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.2, label="perfect calibration")
    ax.plot(m, f, "o-", color=c, lw=2, label="observed frequency")
    ax.bar(np.linspace(0.05, 0.95, 10), w, width=0.09, color=c, alpha=0.2, label="share of predictions")
    ax.set_xlabel("predicted probability"); ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axs[0].set_ylabel("fraction of positives")
axs[0].legend(fontsize=8, loc="upper left")
save(fig, "calib-reliability")
