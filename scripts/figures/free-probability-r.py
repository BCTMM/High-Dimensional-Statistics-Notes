import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, PINK, plt, save
rng = np.random.default_rng(0)
N = 2000
A = np.diag(np.where(np.arange(N) < N // 2, -1.0, 1.0))
G = rng.standard_normal((N, N)); W = (G + G.T) / np.sqrt(2 * N)
Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
sc = lambda x: np.sqrt(np.maximum(4 - x**2, 0)) / (2 * np.pi)
def g_free(z):
    g = 1 / z
    for _ in range(3000):
        w = z - g; g = 0.5 * g + 0.25 * (1 / (w - 1) + 1 / (w + 1))
    return g
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3))
x = np.linspace(-3.6, 3.6, 500)
axs[0].hist(np.linalg.eigvalsh(A + W), bins=70, density=True, color=BLUE, alpha=0.4, label="eigenvalues of A + W")
axs[0].plot(x, [-g_free(xx + 0.003j).imag / np.pi for xx in x], color=ORANGE, lw=2, label="free convolution")
axs[0].plot(x, 0.5 * (sc(x - 1) + sc(x + 1)), color=GREY, ls="--", lw=1.5, label="classical convolution")
axs[0].set_title("A = ±1 (half each), W = Wigner", fontsize=10); axs[0].legend(fontsize=8); axs[0].set_xlabel("eigenvalue")
x = np.linspace(-2.2, 2.2, 500)
axs[1].hist(np.linalg.eigvalsh(A + Q @ A @ Q.T), bins=70, density=True, color=GREEN, alpha=0.4, label="eigenvalues of A + QAQᵀ")
axs[1].plot(x, np.where(np.abs(x) < 2, 1 / (np.pi * np.sqrt(np.maximum(4 - x**2, 1e-9))), 0), color=ORANGE, lw=2, label="free: arcsine law")
for v, h in [(-2, 0.25), (0, 0.5), (2, 0.25)]:
    axs[1].annotate("", xy=(v, 1.2 * h + 0.3), xytext=(v, 0), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.5))
axs[1].plot([], [], color=GREY, lw=1.5, label="classical: atoms at −2, 0, 2")
axs[1].set_ylim(0, 1.3); axs[1].set_title("two free copies of ±1", fontsize=10); axs[1].legend(fontsize=8, loc="upper center"); axs[1].set_xlabel("eigenvalue")
save(fig, "free-sums")
