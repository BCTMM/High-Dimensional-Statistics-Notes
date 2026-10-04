import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
k = np.arange(1, 7)
def fit(target):
    th = 0.0
    for _ in range(50):
        p = np.exp(th * k); p /= p.sum(); m = p @ k; v = p @ k**2 - m**2
        th -= (m - target) / v
    p = np.exp(th * k); return p / p.sum(), th
fig, ax = plt.subplots(figsize=(7, 3.0))
for j, (t, c) in enumerate([(2.5, GREEN), (3.5, GREY), (4.5, BLUE), (5.5, ORANGE)]):
    p, th = fit(t)
    ax.bar(k + (j - 1.5) * 0.2, p, width=0.2, color=c, label=f"E[X] = {t}  (θ = {th:+.2f})")
ax.set_xticks(k); ax.set_xlabel("face"); ax.set_ylabel("probability")
ax.set_title(r"maximum-entropy die with a given mean: $p_k \propto e^{\theta k}$", fontsize=10)
ax.legend(fontsize=8.5)
save(fig, "ef-dice")
