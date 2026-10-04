import numpy as np
from scipy import stats
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(1)
m = 20; mu = np.linspace(-1, 1, m)
Y = mu + rng.standard_normal((50000, m)); j = Y.argmax(1)
err = Y[np.arange(len(Y)), j] - mu[j]
x = np.linspace(-4, 5, 400)
fig, ax = plt.subplots(figsize=(7, 3.0))
ax.hist(err, bins=80, density=True, color=ORANGE, alpha=0.45, label=f"selected winner: reported − true  (mean {err.mean():+.2f})")
ax.plot(x, stats.norm.pdf(x), color=BLUE, lw=2, label="a pre-specified strategy: N(0, 1)")
ax.axvline(0, color=GREY, lw=1)
ax.set_xlabel("estimation error (units of standard error)"); ax.set_ylabel("density"); ax.legend(fontsize=8.5)
ax.set_title("picking the best of 20 noisy estimates biases it upward", fontsize=10)
save(fig, "psi-winners-curse")
