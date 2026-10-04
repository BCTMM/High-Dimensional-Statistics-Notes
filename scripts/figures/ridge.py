import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save

rng = np.random.default_rng(0)
n, p, sigma = 60, 8, 1.0
# Correlated design: features share a common factor.
F = rng.standard_normal((n, 1))
X = 0.9 * F + 0.45 * rng.standard_normal((n, p))
X = (X - X.mean(0)) / X.std(0)
beta = np.array([3, -2, 1.5, 0, 0, 1, 0, -1.0])
y = X @ beta + sigma * rng.standard_normal(n)

U, d, Vt = np.linalg.svd(X, full_matrices=False)
lams = np.logspace(-2, 4, 200)
path = np.array([Vt.T @ (d / (d**2 + l) * (U.T @ y)) for l in lams])
df = np.array([np.sum(d**2 / (d**2 + l)) for l in lams])

fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
for j, c in zip(range(p), [BLUE, ORANGE, GREEN, PINK, GREY, BLUE, ORANGE, GREEN]):
    axs[0].plot(df, path[:, j], color=c, lw=1.6, ls="-" if j < 5 else "--")
axs[0].axhline(0, color="#444", lw=0.8)
axs[0].set_xlabel(r"effective degrees of freedom  df($\lambda$)")
axs[0].set_ylabel("coefficient")
axs[0].set_title("ridge path (λ decreases →)", fontsize=10)

# Bias^2, variance and test error vs lambda (exact formulas, fixed design, new noise),
# for a noisier problem with more features: n = 40, p = 30, sigma = 2.5.
n, p, sigma = 40, 30, 2.5
F = rng.standard_normal((n, 1))
X = 0.9 * F + 0.45 * rng.standard_normal((n, p))
X = (X - X.mean(0)) / X.std(0)
beta = rng.standard_normal(p)
y = X @ beta + sigma * rng.standard_normal(n)
U, d, Vt = np.linalg.svd(X, full_matrices=False)
theta = Vt @ beta
bias2 = np.array([np.sum((l / (d**2 + l) * theta) ** 2 * d**2) / n for l in lams])   # in-sample prediction bias^2
var = np.array([sigma**2 * np.sum((d**2 / (d**2 + l)) ** 2) / n for l in lams])
# GCV on the observed data
gcv = []
for l in lams:
    H_diag_mean = np.sum(d**2 / (d**2 + l)) / n
    yhat = U @ (d**2 / (d**2 + l) * (U.T @ y))
    gcv.append(np.mean((y - yhat) ** 2) / (1 - H_diag_mean) ** 2)
axs[1].semilogx(lams, bias2, color=BLUE, label="bias²")
axs[1].semilogx(lams, var, color=GREEN, label="variance")
axs[1].semilogx(lams, bias2 + var + sigma**2, color=ORANGE, lw=2, label="expected error (bias² + variance + σ²)")
axs[1].set_xlabel("λ")
axs[1].set_ylim(0, 3 * sigma**2)
axs[1].legend(fontsize=8)
axs[1].set_title("bias–variance trade-off (n = 40, p = 30)", fontsize=10)
save(fig, "ridge-path-tradeoff")
