import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(0)
def make(n):
    x = rng.uniform(0, 5, n)
    return x, np.sin(x) * x + rng.normal(0, 0.1 + 0.3 * x, n)
x_tr, y_tr = make(1000); x_cal, y_cal = make(1000); x_te, y_te = make(400)
mu = np.poly1d(np.polyfit(x_tr, y_tr, 5))
sig = np.poly1d(np.polyfit(x_tr, np.abs(y_tr - mu(x_tr)), 2))
k = int(np.ceil(1001 * 0.9))
q_abs = np.sort(np.abs(y_cal - mu(x_cal)))[k - 1]
q_nrm = np.sort(np.abs(y_cal - mu(x_cal)) / sig(x_cal))[k - 1]
xs = np.linspace(0, 5, 300)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for ax, title, half in [(axs[0], "absolute-residual score (constant width)", lambda x: q_abs + 0 * x),
                        (axs[1], "normalized score (adaptive width)", lambda x: q_nrm * sig(x))]:
    ax.fill_between(xs, mu(xs) - half(xs), mu(xs) + half(xs), color=ORANGE, alpha=0.25, label="90% conformal band")
    ax.plot(xs, mu(xs), color=ORANGE, lw=2, label="model")
    ax.plot(x_te, y_te, ".", color=BLUE, ms=3.5, alpha=0.7, label="test points")
    ax.set_title(title, fontsize=10); ax.set_xlabel("x")
axs[0].legend(fontsize=8, loc="lower left")
save(fig, "conf-bands")
