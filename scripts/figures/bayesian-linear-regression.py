import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(0)
def features(x, degree=9):                         # polynomial features on [-1, 1]
    return np.vander(x, degree + 1, increasing=True)

def posterior(Phi, y, alpha, beta):
    """Prior w ~ N(0, alpha^{-1} I), noise precision beta. Returns posterior mean and covariance."""
    S = np.linalg.inv(alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi)
    return beta * S @ Phi.T @ y, S

def log_evidence(Phi, y, alpha, beta):
    n, p = Phi.shape; m, S = posterior(Phi, y, alpha, beta)
    E = beta / 2 * np.sum((y - Phi @ m) ** 2) + alpha / 2 * m @ m
    return p / 2 * np.log(alpha) + n / 2 * np.log(beta) - E + 0.5 * np.linalg.slogdet(S)[1] - n / 2 * np.log(2 * np.pi)

f = lambda x: np.sin(3 * x)
x = rng.uniform(-1, 0.4, 30); y = f(x) + 0.2 * rng.standard_normal(30)   # note: no data on (0.4, 1]
Phi = features(x)

# Empirical Bayes: maximize the evidence over (alpha, beta) by MacKay's fixed-point updates.
alpha, beta = 1.0, 1.0
for _ in range(200):
    m, S = posterior(Phi, y, alpha, beta)
    lam = np.linalg.eigvalsh(beta * Phi.T @ Phi)
    gamma = np.sum(lam / (lam + alpha))            # effective number of well-determined parameters
    alpha = gamma / (m @ m)
    beta = (len(y) - gamma) / np.sum((y - Phi @ m) ** 2)
_ = (f"evidence-optimal: alpha = {alpha:.3f}, noise sd = {beta**-0.5:.3f} (true 0.2), "
      f"effective parameters = {gamma:.2f} of {Phi.shape[1]}, log evidence = {log_evidence(Phi, y, alpha, beta):.2f}")


m, S = posterior(Phi, y, alpha, beta)
xs = np.linspace(-1, 1, 400); P = features(xs)
mean = P @ m; sd = np.sqrt(1 / beta + np.einsum("ij,jk,ik->i", P, S, P))
fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.fill_between(xs, mean - 1.96 * sd, mean + 1.96 * sd, color=BLUE, alpha=0.18, label="95% posterior predictive")
for w in rng.multivariate_normal(m, S, size=5):
    ax.plot(xs, P @ w, color=BLUE, lw=0.8, alpha=0.6)
ax.plot(xs, mean, color=BLUE, lw=2, label="posterior mean (= ridge)")
ax.plot(xs, f(xs), color="#222", lw=1.5, ls="--", label="true function")
ax.plot(x, y, "o", color=ORANGE, ms=4, label="data")
ax.axvspan(0.4, 1, color=GREY, alpha=0.1)
ax.text(0.55, -2.6, "no data here", fontsize=9, color=GREY)
ax.set_ylim(-3, 3); ax.set_xlabel("x"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("degree-9 polynomial with evidence-tuned prior; thin lines: posterior samples", fontsize=9.5)
save(fig, "blr-band")
