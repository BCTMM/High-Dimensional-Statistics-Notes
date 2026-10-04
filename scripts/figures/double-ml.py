import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(1)
def knn_predict(Xtr, ytr, Xte, k=10):
    """k-nearest-neighbour regression (a simple flexible ML learner)."""
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    return ytr[idx].mean(1)

def dml(Y, D, X, cross_fit=True, folds=2, k=10):
    """Partially linear model Y = theta*D + g(X) + eps: residualize Y and D on X, then regress residual on residual."""
    n = len(Y); rY, rD = np.empty(n), np.empty(n)
    if cross_fit:
        fold = np.arange(n) % folds
        for f in range(folds):
            tr, te = fold != f, fold == f                        # nuisances fitted on the OTHER folds
            rY[te] = Y[te] - knn_predict(X[tr], Y[tr], X[te], k)
            rD[te] = D[te] - knn_predict(X[tr], D[tr], X[te], k)
    else:                                                        # same data for fitting and residualizing
        rY = Y - knn_predict(X, Y, X, k); rD = D - knn_predict(X, D, X, k)
    theta = rD @ rY / (rD @ rD)
    psi = (rY - theta * rD) * rD                                 # orthogonal score -> sandwich standard error
    se = np.sqrt(np.mean(psi**2) / np.mean(rD**2) ** 2 / n)
    return theta, se


theta0, n = 1.0, 1000
ols, dmlc = [], []
for rep in range(120):
    X = rng.uniform(-1, 1, (n, 2))
    m = np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2; g = np.cos(np.pi * X[:, 0]) + 2 * np.abs(X[:, 1])
    D = m + 0.5 * rng.standard_normal(n); Y = theta0 * D + g + rng.standard_normal(n)
    Z = np.column_stack([np.ones(n), D, X]); ols.append(np.linalg.lstsq(Z, Y, rcond=None)[0][1])
    dmlc.append(dml(Y, D, X, cross_fit=True)[0])
fig, ax = plt.subplots(figsize=(7, 3.0))
bins = np.linspace(0.7, 1.5, 50)
ax.hist(ols, bins=bins, density=True, color=ORANGE, alpha=0.5, label="OLS with linear controls (confounding not removed)")
ax.hist(dmlc, bins=bins, density=True, color=BLUE, alpha=0.5, label="double ML (k-NN nuisances, cross-fitted)")
ax.axvline(theta0, color="#222", lw=2, ls="--", label="true effect θ = 1")
ax.set_xlabel("estimated treatment effect"); ax.set_ylabel("density"); ax.legend(fontsize=8)
save(fig, "dml-hist")
