import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save

rng = np.random.default_rng(0)
n, B = 50, 5000

fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.1))
# (a) Mean of skewed (exponential) data: bootstrap vs the true sampling distribution.
true_m = rng.exponential(size=(20000, n)).mean(axis=1) - 1.0
x = rng.exponential(size=n)
boot_m = rng.choice(x, size=(B, n), replace=True).mean(axis=1) - x.mean()
bins = np.linspace(-0.6, 0.8, 50)
axs[0].hist(true_m, bins=bins, density=True, alpha=0.45, color=BLUE, label="true: mean − 1")
axs[0].hist(boot_m, bins=bins, density=True, histtype="step", lw=1.8, color=ORANGE, label="bootstrap: mean* − mean")
axs[0].set_title("mean of skewed data, n = 50: bootstrap works", fontsize=10)
axs[0].legend(fontsize=8)

# (b) Maximum of uniform data: the bootstrap fails.
true_max = n * (1 - rng.uniform(size=(20000, n)).max(axis=1))
u = rng.uniform(size=n)
boot_max = n * (u.max() - rng.choice(u, size=(B, n), replace=True).max(axis=1))
bins = np.linspace(0, 6, 40)
axs[1].hist(true_max, bins=bins, density=True, alpha=0.45, color=BLUE, label="true: n(1 − max)")
axs[1].hist(boot_max, bins=bins, density=True, histtype="step", lw=1.8, color=ORANGE, label="bootstrap: n(max − max*)")
axs[1].set_title(f"maximum: {np.mean(boot_max == 0):.0%} of bootstrap mass at 0", fontsize=10)
axs[1].legend(fontsize=8)
save(fig, "boot-works-fails")
