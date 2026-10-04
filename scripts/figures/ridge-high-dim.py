import numpy as np
from _style import BLUE, ORANGE, GREEN, PINK, GREY, plt, save
rng = np.random.default_rng(1)
sigma, r = 1.0, 1.0
kap = lambda lam, g: ((lam + g - 1) + np.sqrt((lam + g - 1) ** 2 + 4 * lam)) / 2
def risk(lam, g):
    k = kap(lam, g); df2 = g / (1 + k) ** 2
    return (k**2 * r**2 / (1 + k) ** 2 + sigma**2 * df2) / (1 - df2)
def sim(lam, n, p, reps=10):
    out = []
    for _ in range(reps):
        X = rng.standard_normal((n, p)); beta = rng.standard_normal(p) * r / np.sqrt(p); y = X @ beta + sigma * rng.standard_normal(n)
        b = X.T @ np.linalg.solve(X @ X.T + n * lam * np.eye(n), y) if p > n else np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)
        out.append(np.sum((b - beta) ** 2))
    return np.mean(out)
g = np.linspace(0.05, 4, 800); n = 100
fig, ax = plt.subplots(figsize=(7.4, 3.5))
for lam, c, lab in [(1e-10, ORANGE, "ridgeless (min-norm) "), (0.1, BLUE, "λ = 0.1"), (None, GREEN, "optimal λ = σ²γ/r²")]:
    curve = [risk(lam if lam is not None else sigma**2 * gg / r**2, gg) for gg in g]
    ax.plot(g, curve, color=c, lw=2, label=lab)
    gs = np.array([0.3, 0.6, 0.8, 1.25, 1.6, 2.5, 3.5])
    ax.plot(gs, [sim(lam if lam is not None else sigma**2 * gg / r**2, n, int(gg * n)) for gg in gs], "o", color=c, ms=4)
ax.axhline(r**2, color=GREY, ls=":", lw=1.2, label="null predictor (β̂ = 0)")
ax.axvline(1, color=GREY, lw=0.8)
ax.set_ylim(0, 4); ax.set_xlabel("γ = p / n  (parameters per sample)"); ax.set_ylabel("excess test error")
ax.set_title("isotropic features, SNR = 1; dots: simulation with n = 100", fontsize=10); ax.legend(fontsize=8)
save(fig, "dd-curves")
