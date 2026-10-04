import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(1)
n, p = 2000, 5
X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
beta_true = np.array([-0.5, 1.0, -1.0, 0.5, 0.0])
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta_true))).astype(float)
sig = lambda e: 1 / (1 + np.exp(-e))
def nll(b): e = X @ b; return np.sum(np.logaddexp(0, e) - y * e)
# reference optimum
b = np.zeros(p)
for _ in range(50):
    m = sig(X @ b); W = m * (1 - m); b = b + np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (y - m))
best = nll(b)
gaps_irls, gaps_gd = [], []
b = np.zeros(p)
for _ in range(8):
    gaps_irls.append(nll(b) - best)
    m = sig(X @ b); W = m * (1 - m); b = b + np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (y - m))
b = np.zeros(p); L = 0.25 * np.linalg.eigvalsh(X.T @ X).max()
for _ in range(200):
    gaps_gd.append(nll(b) - best); b = b - X.T @ (sig(X @ b) - y) / L
fig, ax = plt.subplots(figsize=(7, 3.0))
ax.semilogy(np.maximum(gaps_irls, 1e-16), "o-", color=BLUE, label="IRLS (= Newton = Fisher scoring)")
ax.semilogy(np.maximum(gaps_gd, 1e-16), color=ORANGE, label="gradient descent, step 1/L")
ax.set_xlabel("iteration"); ax.set_ylabel("negative log-likelihood − optimum")
ax.set_ylim(1e-13, 2e3); ax.set_xlim(-1, 50); ax.legend(); ax.set_title("logistic regression, n = 2000, p = 5", fontsize=10)
save(fig, "glm-convergence")
