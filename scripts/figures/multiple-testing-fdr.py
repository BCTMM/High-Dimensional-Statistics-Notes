import numpy as np
from scipy import stats
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(3)
m, m1 = 1000, 100
z = np.r_[rng.normal(3.0, 1, m1), rng.standard_normal(m - m1)]
p = stats.norm.sf(z); is_sig = np.r_[np.ones(m1, bool), np.zeros(m - m1, bool)]
order = np.argsort(p); ps = p[order]; sig = is_sig[order]
k = np.arange(1, m + 1); q = 0.1
below = ps <= q * k / m; kmax = np.max(np.nonzero(below)[0]) + 1
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.loglog(k[sig], ps[sig], "o", ms=3.5, color=BLUE, label="true effects")
ax.loglog(k[~sig], ps[~sig], "o", ms=3.5, color=GREY, alpha=0.6, label="nulls")
ax.loglog(k, q * k / m, color=ORANGE, lw=2, label=f"BH line  q·k/m  (q = {q})")
ax.axhline(q / m, color=GREEN, ls="--", lw=1.5, label="Bonferroni  q/m")
ax.axvline(kmax, color=ORANGE, ls=":", lw=1.2)
ax.text(kmax * 1.1, 1e-6, f"BH rejects\nthe first {kmax}", color=ORANGE, fontsize=9)
ax.set_xlabel("rank k"); ax.set_ylabel("sorted p-value  p(k)"); ax.set_ylim(1e-7, 1.5)
ax.legend(fontsize=8, loc="upper left")
save(fig, "bh-plot")
print("BH rejections", kmax, "false among them", np.sum(~sig[:kmax]), " Bonferroni rejections", np.sum(ps <= q / m))
