import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
x = np.linspace(-2, 2, 400)
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.1))
axs[0].plot(x, np.abs(x), color="#222", lw=2.2, label=r"$f(x) = |x|$")
for s, c in [(-0.8, BLUE), (0.0, GREEN), (0.5, ORANGE)]:
    axs[0].plot(x, s * x, color=c, lw=1.5, ls="--", label=f"supporting line, slope {s}")
axs[0].set_ylim(-1, 2); axs[0].legend(fontsize=8); axs[0].set_title("every slope in [−1, 1] supports |x| at 0", fontsize=10)
axs[1].plot([-2, 0], [-1, -1], color="#222", lw=2.2); axs[1].plot([0, 2], [1, 1], color="#222", lw=2.2)
axs[1].plot([0, 0], [-1, 1], color=ORANGE, lw=3, label=r"$\partial f(0) = [-1, 1]$")
axs[1].set_ylim(-1.6, 1.6); axs[1].set_xlabel("x"); axs[1].legend(fontsize=8.5, loc="upper left")
axs[1].set_title(r"the subdifferential $\partial|x|$ is set-valued", fontsize=10)
save(fig, "kkt-subgradient")
