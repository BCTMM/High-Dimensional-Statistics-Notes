import numpy as np
from _style import BLUE, ORANGE, GREEN, GREY, plt, save
rng = np.random.default_rng(0)
def lars_lasso(X, y):
    """Exact LASSO path for 0.5||y - Xb||^2 + lam ||b||_1 (homotopy / LARS with the lasso modification).
    Returns breakpoints lams[k] and coefficients B[k]."""
    n, p = X.shape
    b = np.zeros(p); c = X.T @ y; lam = np.max(np.abs(c))
    A = [int(np.argmax(np.abs(c)))]
    lams, B = [lam], [b.copy()]
    while lam > 1e-10:
        XA = X[:, A]; s = np.sign(c[A])
        d = np.linalg.solve(XA.T @ XA, s)                # move so that all active correlations shrink together
        a = X.T @ (XA @ d)                               # rate at which every correlation changes
        gam, event = lam, None                           # default: run all the way to lam = 0
        for j in range(p):                               # next variable to enter
            if j in A: continue
            for g in ((lam - c[j]) / (1 - a[j]), (lam + c[j]) / (1 + a[j])):
                if 1e-12 < g < gam: gam, event = g, ("in", j)
        for k, j in enumerate(A):                        # next active coefficient to hit zero (lasso modification)
            g = -b[j] / d[k]
            if 1e-12 < g < gam: gam, event = g, ("out", j)
        b[A] += gam * d; c -= gam * a; lam -= gam
        lams.append(lam); B.append(b.copy())
        if event is None or len(A) == min(n, p) and event[0] == "in": break
        A.append(event[1]) if event[0] == "in" else A.remove(event[1])
    return np.array(lams), np.array(B)


n, p = 50, 20
X = rng.standard_normal((n, p)); X /= np.linalg.norm(X, axis=0)
beta = np.zeros(p); beta[:4] = [4, -3, 2, 1.5]
y = X @ beta + 0.5 * rng.standard_normal(n)
lams, B = lars_lasso(X, y)
# forward stagewise: tiny steps on the most correlated variable
eps, b, r, FS = 0.005, np.zeros(p), y.copy(), []
for _ in range(4000):
    c = X.T @ r; j = np.argmax(np.abs(c)); delta = eps * np.sign(c[j])
    b[j] += delta; r -= delta * X[:, j]; FS.append(b.copy())
FS = np.array(FS)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
l1 = np.abs(B).sum(1)
for j in range(p):
    col = BLUE if j < 4 else GREY
    axs[0].plot(l1, B[:, j], color=col, lw=1.6 if j < 4 else 0.8)
    axs[1].plot(np.abs(FS).sum(1), FS[:, j], color=col, lw=1.6 if j < 4 else 0.8)
for x in l1: axs[0].axvline(x, color=GREY, lw=0.4, alpha=0.5)
axs[0].set_title(f"exact LASSO path by LARS ({len(lams) - 1} linear pieces)", fontsize=10)
axs[1].set_title("forward stagewise, step 0.005", fontsize=10)
for ax in axs: ax.set_xlabel(r"$\|\beta\|_1$"); ax.set_xlim(0, l1[-1])
axs[0].set_ylabel("coefficient")
save(fig, "lars-path")
