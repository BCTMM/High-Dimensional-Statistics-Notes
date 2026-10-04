import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
p0 = 0.3
p = np.linspace(0.01, 0.99, 400)
kl = p0 * np.log(p0 / p) + (1 - p0) * np.log((1 - p0) / (1 - p))
quad = 0.5 * (p - p0) ** 2 / (p0 * (1 - p0))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
axs[0].plot(p, kl, color=BLUE, lw=2, label=r"KL(Bern(0.3) ‖ Bern($p$))")
axs[0].plot(p, quad, color=ORANGE, lw=2, ls="--", label=r"$\frac{1}{2} I(0.3)\,(p-0.3)^2$")
axs[0].set_ylim(0, 1.2); axs[0].set_xlabel("p"); axs[0].legend(fontsize=8.5)
axs[0].set_title("Fisher information = curvature of KL", fontsize=10)
theta = np.linspace(-6, 6, 400)                 # logit parametrization
pp = 1 / (1 + np.exp(-theta)); th0 = np.log(p0 / (1 - p0))
kl2 = p0 * np.log(p0 / pp) + (1 - p0) * np.log((1 - p0) / (1 - pp))
axs[1].plot(theta, kl2, color=BLUE, lw=2, label="same KL, logit coordinates")
axs[1].plot(theta, 0.5 * p0 * (1 - p0) * (theta - th0) ** 2, color=ORANGE, lw=2, ls="--", label=r"$\frac{1}{2}\,p_0(1-p_0)\,(\theta-\theta_0)^2$")
axs[1].set_ylim(0, 1.2); axs[1].set_xlabel(r"$\theta = \mathrm{logit}\,p$"); axs[1].legend(fontsize=8.5)
axs[1].set_title("Fisher information transforms as a metric", fontsize=10)
save(fig, "fisher-kl")
