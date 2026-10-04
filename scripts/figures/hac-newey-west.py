import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(0)
T, h = 400000, 12
r = rng.standard_normal(T + h)
y = np.convolve(r, np.ones(h), mode="valid")[1:T + 1]            # overlapping 12-month sums
y = y - y.mean()
lags = np.arange(0, 31)
acf = [np.mean(y[k:] * y[:len(y) - k]) / np.var(y) for k in lags]
fig, ax = plt.subplots(figsize=(7, 3.0))
ax.bar(lags, acf, color=BLUE, alpha=0.6, label="sample autocorrelation of overlapping 12-month returns")
ax.plot(lags, np.maximum(1 - lags / h, 0), color="#222", lw=1.5, ls="--", label=r"theory: $(h-j)/h$")
for L, c in [(12, ORANGE), (24, GREEN)]:
    ax.plot(lags, np.maximum(1 - lags / (L + 1), 0), color=c, lw=2, label=f"Bartlett weights, L = {L}")
ax.set_xlabel("lag j (months)"); ax.set_ylim(-0.1, 1.05); ax.legend(fontsize=8)
save(fig, "hac-acf")
