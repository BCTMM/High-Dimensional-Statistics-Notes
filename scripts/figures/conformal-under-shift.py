import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(0)
T = 3000
vol = np.where((np.arange(T) // 500) % 2 == 0, 0.01, 0.03)
r = vol * rng.standard_t(5, T) / np.sqrt(5 / 3)
alpha, window, gamma = 0.1, 250, 0.01
def quantile(scores, level):
    if level >= 1: return np.inf
    if level <= 0: return 0.0
    n = len(scores); k = int(np.ceil((n + 1) * level))
    return np.inf if k > n else np.sort(scores)[k - 1]
es, ea, alphas, a_t = [], [], [], alpha
for t in range(window, T):
    past = np.abs(r[t - window:t])
    es.append(abs(r[t]) > quantile(past, 1 - alpha))
    e = abs(r[t]) > quantile(past, 1 - a_t); ea.append(e); alphas.append(a_t)
    a_t += gamma * (alpha - e)
tt = np.arange(window, T)
roll = lambda e: np.convolve(np.array(e, float), np.ones(100) / 100, mode="same")
fig, axs = plt.subplots(2, 1, figsize=(8.5, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for k in range(T // 500):
    if k % 2 == 1:
        for ax in axs: ax.axvspan(500 * k, 500 * (k + 1), color=GREY, alpha=0.12, lw=0)
axs[0].plot(tt, roll(es), color=ORANGE, lw=1.6, label="rolling-window split conformal")
axs[0].plot(tt, roll(ea), color=BLUE, lw=1.6, label="adaptive conformal inference (ACI)")
axs[0].axhline(alpha, color="#222", ls="--", lw=1.2, label="target α = 0.1")
axs[0].set_ylabel("100-day miscoverage"); axs[0].legend(fontsize=8, loc="upper left"); axs[0].set_ylim(0, 0.5)
axs[0].set_title("shaded = high-volatility regime", fontsize=10)
axs[1].plot(tt, alphas, color=BLUE, lw=1.4)
axs[1].axhline(alpha, color="#222", ls="--", lw=1.2)
axs[1].set_ylabel(r"ACI's $\alpha_t$"); axs[1].set_xlabel("day")
save(fig, "aci-coverage")
