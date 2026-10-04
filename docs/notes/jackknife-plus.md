# Jackknife+ & CV+

!!! tldr "TL;DR"
    Split conformal throws away half the data for calibration. The obvious fix, the **jackknife** (center the interval at the full-data fit and use leave-one-out residuals), has **no guarantee**
    and can fail badly. **Jackknife+** fixes it with one change: center each residual at the corresponding *leave-one-out* prediction at the test point. Then, for **any** algorithm that treats
    training points symmetrically, coverage is at least $1-2\alpha$ in the worst case, and typically about $1-\alpha$. **CV+** does the same with $K$ folds instead of $n$ refits.

## Why care?

[Split conformal](conformal-prediction.md) is simple and exactly valid, but it fits the model on only part of the data and calibrates on the rest. With small datasets this hurts twice:
the model is worse (wider intervals) and the calibration set is small (noisier coverage). This is common in clinical studies, drug discovery, materials science and backtests with few independent periods.

Cross-validation-style methods use every point for both fitting and calibration, which is what practitioners naturally try. It turns out that the natural version can be invalid. Barber,
Candès, Ramdas & Tibshirani (2021) found the minimal modification that restores a distribution-free guarantee, and it is cheap when leave-one-out (or $K$-fold) fits are cheap, which they
often are for linear models, kernels and ensembles.

## Building blocks

**Setup.** Data $(X_i,Y_i)_{i=1}^n$ exchangeable with a test point $(X_{n+1},Y_{n+1})$. A regression algorithm $\mathcal A$ maps a dataset to a predictor. We assume $\mathcal A$ is **symmetric**: it doesn't care
about the order of the training points (true for almost all algorithms, or after randomizing the order).

**Leave-one-out models and residuals.** $\hat\mu_{-i} = \mathcal A\big(\{(X_j,Y_j)\}_{j\neq i}\big)$ and $R_i = |Y_i - \hat\mu_{-i}(X_i)|$. Each $R_i$ is an honest out-of-sample error.

**Quantile notation.** For values $v_1,\dots,v_n$: $\hat q^+_\alpha\{v_i\}$ is the $\lceil(1-\alpha)(n+1)\rceil$-th smallest, and $\hat q^-_\alpha\{v_i\} = -\hat q^+_\alpha\{-v_i\}$ is the $\lfloor\alpha(n+1)\rfloor$-th smallest.

**The naive jackknife.**

$$
C^{\text{jack}}(x) = \Big[\hat\mu(x) - \hat q^+_\alpha\{R_i\},\;\hat\mu(x) + \hat q^+_\alpha\{R_i\}\Big],
$$

where $\hat\mu$ is fitted on all $n$ points. It looks reasonable: the LOO residuals estimate the out-of-sample error distribution. **But the interval is centred at a different model** ($\hat\mu$, trained on $n$ points)
from the ones that produced the residuals ($\hat\mu_{-i}$, trained on $n-1$). If the algorithm is unstable, so that adding one point changes it a lot, the residual quantile can badly misjudge $\hat\mu$'s error.

## The main result

**Jackknife+.**

$$
C^{\text{J+}}(x) = \Big[\hat q^-_\alpha\big\{\hat\mu_{-i}(x) - R_i\big\},\;\hat q^+_\alpha\big\{\hat\mu_{-i}(x) + R_i\big\}\Big].
$$

Each LOO model predicts at the test point and is padded by its own LOO residual. The interval endpoints are quantiles of these $n$ padded predictions.

!!! theorem "Theorem (Barber, Candès, Ramdas & Tibshirani, 2021)"
    If the $n+1$ data points are exchangeable and $\mathcal A$ is symmetric, then for **any** distribution and **any** algorithm,

    $$
    \P\big(Y_{n+1}\in C^{\text{J+}}(X_{n+1})\big)\ge1 - 2\alpha .
    $$

    The factor 2 cannot be removed in the worst case, but for stable algorithms coverage is $\approx1-\alpha$.

**Proof sketch (the tournament argument).** Imagine fitting on the $n+1$ points with *two* left out. For $i\neq j$ let $\tilde\mu_{-ij}$ be trained without $i$ and $j$, and let $R_{ij} = |Y_i - \tilde\mu_{-ij}(X_i)|$
be point $i$'s residual under that model. Say **$i$ beats $j$** if $R_{ij} > R_{ji}$, meaning $i$ looks stranger than $j$ to the model that saw neither.

*Step 1: few points can be very strange.* Call $i$ **strange** if it beats at least $(1-\alpha)(n+1)$ others. Let $S$ be the set of strange points. Each win involves a pair, and strange points can rack up wins only against
each other (at most $\binom{|S|}{2}$ such wins in total) or against non-strange points (at most $|S|(n+1-|S|)$). So

$$
|S|\,(1-\alpha)(n+1)\le\binom{|S|}{2} + |S|(n+1-|S|)\quad\Longrightarrow\quad|S|\le2\alpha(n+1).
$$

*Step 2: exchangeability.* The tournament is a symmetric function of the data, so by exchangeability every point, the test point included, is equally likely to be strange:
$\P(n+1\in S)\le\frac{2\alpha(n+1)}{n+1} = 2\alpha$.

*Step 3: miscoverage implies strangeness.* $\tilde\mu_{-(n+1)i}$ is exactly the LOO model $\hat\mu_{-i}$. If $Y_{n+1}$ lies above the upper endpoint, then $Y_{n+1} > \hat\mu_{-i}(X_{n+1}) + R_i$ for at least $(1-\alpha)(n+1)$ indices $i$,
i.e. $R_{n+1,i} > R_{i,n+1}$, so the test point beats all of them. Same for the lower endpoint. So miscoverage $\subseteq\{n+1\in S\}$. $\square$

The naive jackknife has no such argument. Its centre $\hat\mu$ is not one of the models in the tournament, and examples exist where its coverage is arbitrarily low.

### Variants

- **CV+.** Split into $K$ folds and fit $K$ models $\hat\mu_{-S_k}$. Use $\hat\mu_{-S_{k(i)}}(x)\pm R_i$, where $k(i)$ is $i$'s fold. Coverage is $\ge1 - 2\alpha - O(\sqrt{1/n})$ (exactly
  $1-2\alpha-\min\{\frac{2(1-1/K)}{n/K+1},\frac{1-K/n}{K+1}\}$). It costs $K$ fits instead of $n$.
- **Jackknife+-after-bootstrap (J+aB).** With bagged ensembles (random forests), aggregate for each $i$ only the bootstrap models that didn't contain $i$ (out-of-bag). This gives jackknife+ intervals essentially for free.
- **Jackknife-minmax.** Use $\min_i\hat\mu_{-i}(x) - \hat q^+\{R_i\}$ and $\max_i\hat\mu_{-i}(x) + \hat q^+\{R_i\}$. This guarantees $1-\alpha$ but is conservative.

## Examples

### When the jackknife breaks

Least squares with almost no regularization is extremely unstable when the number of features $p$ is close to the number of training points. Near $p = n$ the fit interpolates, and its variance explodes
(the peak of [double descent](ridge-high-dim.md)).

```python
import numpy as np
rng = np.random.default_rng(0)
alpha, n, n_test = 0.1, 100, 500

def fit(X, y, lam=1e-6):                       # (nearly) ridgeless least squares: unstable when p ≈ n
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def q_hi(v): return np.sort(v)[int(np.ceil((1 - alpha) * (len(v) + 1))) - 1]
def q_lo(v): return -q_hi(-v)

def one_trial(p):
    beta = rng.standard_normal(p) / np.sqrt(p)
    X = rng.standard_normal((n, p)); y = X @ beta + rng.standard_normal(n)
    Xt = rng.standard_normal((n_test, p)); yt = Xt @ beta + rng.standard_normal(n_test)
    # split conformal: train on half, calibrate on half
    b = fit(X[:50], y[:50]); R = np.abs(y[50:] - X[50:] @ b); q = q_hi(R)
    split = np.mean(np.abs(yt - Xt @ b) <= q); w_split = 2 * q
    # leave-one-out models
    loo = [fit(np.delete(X, i, 0), np.delete(y, i)) for i in range(n)]
    R_loo = np.array([abs(y[i] - X[i] @ loo[i]) for i in range(n)])
    full = Xt @ fit(X, y)
    naive = np.mean(np.abs(yt - full) <= q_hi(R_loo))                       # jackknife: full model ± LOO quantile
    M = np.stack([Xt @ bi for bi in loo], axis=1)                            # n_test x n LOO predictions
    lo = np.array([q_lo(M[t] - R_loo) for t in range(n_test)])
    hi = np.array([q_hi(M[t] + R_loo) for t in range(n_test)])
    jplus = np.mean((yt >= lo) & (yt <= hi))
    return split, naive, jplus, w_split, np.median(hi - lo)

for p in [20, 50, 100, 200]:
    res = np.array([one_trial(p) for _ in range(20)]).mean(axis=0)
    print(f"p = {p:3d}   coverage: split {res[0]:.3f}  naive jackknife {res[1]:.3f}  jackknife+ {res[2]:.3f}   "
          f"width: split {res[3]:.2f}  jackknife+ {res[4]:.2f}")
# p =  20   coverage: split 0.900  naive jackknife 0.894  jackknife+ 0.892   width: split 4.30  jackknife+ 3.71
# p =  50   coverage: split 0.906  naive jackknife 0.925  jackknife+ 0.925   width: split 34.65  jackknife+ 5.01
# p = 100   coverage: split 0.885  naive jackknife 0.558  jackknife+ 0.939   width: split 4.99  jackknife+ 49.48
# p = 200   coverage: split 0.907  naive jackknife 0.901  jackknife+ 0.910   width: split 4.98  jackknife+ 5.50
```

There is a lot in this table:

- **$p = 100 = n$:** the full model sits exactly at the interpolation peak, while the LOO models (trained on 99 points) are slightly past it. The LOO residuals badly underestimate the full model's error,
  and the naive jackknife covers only **56%**. Jackknife+ stays valid at 94%, because it never uses the full model.
- **Widths reveal the same peak.** Each method blows up when *its own* training size equals $p$. Split conformal trains on 50 points, so it explodes at $p = 50$ (width 35). Jackknife+'s LOO models train on 99, so they
  explode at $p = 100$ (width 49). Validity never breaks, only efficiency.
- **Away from the peak** (for example $p = 20$), jackknife+ is about 15% narrower than split conformal, because each model saw twice as much data.

## Exercises

!!! question "Exercise 1 · warm-up: costs"
    Compare the number of model fits and the number of calibration residuals for split conformal (50/50), jackknife+, and 10-fold CV+, with $n = 1000$. For ridge regression, how can jackknife+ be computed with a single fit?

    ??? success "Solution"
        Split: 1 fit, 500 residuals. Jackknife+: 1000 fits, 1000 residuals. CV+ with $K = 10$: 10 fits, 1000 residuals. For ridge (any linear smoother), the [LOO shortcut](ridge.md) gives the residuals $R_i = |e_i|/(1 - H_{ii})$ from one fit.
        The LOO *predictions at the test point* are also available in closed form via Sherman–Morrison, $\hat\beta_{-i} = \hat\beta - \frac{(X^\top X+\lambda I)^{-1}x_ie_i}{1 - H_{ii}}$, so jackknife+ costs the same as one fit.

!!! question "Exercise 2 · the counting bound"
    Fill in the algebra: show that $|S|(1-\alpha)(n+1)\le\binom{|S|}{2} + |S|(n+1-|S|)$ implies $|S|\le2\alpha(n+1)$.

    ??? success "Solution"
        Divide by $|S|$: $(1-\alpha)(n+1)\le\frac{|S|-1}{2} + n + 1 - |S| = n + \frac12 - \frac{|S|}{2}$. So $\frac{|S|}{2}\le n + \frac12 - (1-\alpha)(n+1) = \alpha(n+1) - \frac12$, i.e. $|S|\le2\alpha(n+1) - 1$.
        That is slightly better than the claimed $2\alpha(n+1)$, since each point plays only $n$ games, not $n+1$.

!!! question "Exercise 3 · the sample mean"
    Let $\mathcal A$ output the constant predictor $\hat\mu = $ sample mean. Write out $\hat\mu_{-i}$, $R_i$, and the jackknife+ interval. Show that jackknife+ and the naive jackknife are nearly identical here. Why?

    ??? success "Solution"
        $\hat\mu_{-i} = \frac{n\bar Y - Y_i}{n-1}$, so $\hat\mu_{-i} - \bar Y = \frac{\bar Y - Y_i}{n-1}$, which is $O(1/n)$. $R_i = |Y_i - \hat\mu_{-i}| = \frac{n}{n-1}|Y_i - \bar Y|$. The J+ endpoints are quantiles of $\hat\mu_{-i}\pm R_i$, which
        differ from $\bar Y\pm R_i$ by $O(1/n)$. The mean is very **stable** (leaving one point out moves it by $O(1/n)$), so the jackknife's centring problem disappears. Instability, as in the $p\approx n$ least-squares example, is what separates the methods.

!!! question "Exercise 4 · why $2\alpha$ is the worst case"
    Explain informally why the guarantee can't be $1-\alpha$ for jackknife+ in general, while split conformal gets $1-\alpha$ exactly. What property of split conformal's calibration residuals is missing in jackknife+?

    ??? success "Solution"
        In split conformal, the calibration scores and the test score are computed by the **same** fixed model, so they are exchangeable and the rank lemma applies directly. In jackknife+, residual $R_i$ comes from model
        $\hat\mu_{-i}$, while the comparison at the test point uses $\hat\mu_{-i}(X_{n+1})$, and every model is different. There is no single exchangeable list of scores, only the pairwise comparisons of the tournament,
        and pairwise exchangeability gives the weaker $2\alpha$ bound. Barber et al. construct an (adversarial, unstable) algorithm whose coverage approaches $1-2\alpha$.

!!! question "Exercise 5 · stretch: stability gives $1-\alpha$"
    Suppose the algorithm is **$\varepsilon$-stable**: $|\hat\mu(x) - \hat\mu_{-i}(x)|\le\varepsilon$ for all $i$ and $x$ (with probability 1). Show that the jackknife+ interval contains the split-conformal-like interval
    $[\hat\mu(x) - \hat q^+\{R_i\} + \varepsilon,\ \hat\mu(x) + \hat q^+\{R_i\} - \varepsilon]$ and is contained in $[\hat\mu(x) - \hat q^+\{R_i\} - \varepsilon,\ \hat\mu(x) + \hat q^+\{R_i\} + \varepsilon]$. Discuss what this says about coverage for stable algorithms
    such as ridge with large $\lambda$ or bagged ensembles.

    ??? success "Solution"
        For each $i$, $\hat\mu_{-i}(x) + R_i\in[\hat\mu(x) + R_i - \varepsilon,\ \hat\mu(x) + R_i + \varepsilon]$. Quantiles are monotone and shift-equivariant, so $\hat q^+\{\hat\mu_{-i}(x) + R_i\}\in[\hat\mu(x) + \hat q^+\{R_i\} - \varepsilon,\ \hat\mu(x) + \hat q^+\{R_i\} + \varepsilon]$.
        The lower endpoint is analogous. So for stable algorithms, jackknife+ ≈ the naive jackknife ≈ full-data-model ± LOO quantile, up to $\varepsilon$. The LOO residuals are then nearly exchangeable with the test residual of
        $\hat\mu$ (each is an out-of-sample error of a model trained on $\approx n$ points), so coverage is $\approx1-\alpha$, up to terms controlled by $\varepsilon$ (Barber et al. make this precise). Ridge with substantial
        regularization and bagging are stable, which explains the good practical behaviour.

## Where it shows up

- **Small-data science.** In drug discovery, materials and clinical prediction, every labelled point is expensive. Jackknife+ and CV+ give valid intervals without sacrificing half the data. They are implemented in
  standard libraries (for example MAPIE in Python).
- **Ensembles for free.** Jackknife+-after-bootstrap turns the out-of-bag predictions that random forests and bagged models already compute into valid predictive intervals at negligible cost.
- **Time series and quant forecasting.** EnbPI (Xu & Xie, 2021) adapts jackknife+-after-bootstrap ensembles to sequential data, producing prediction intervals for energy demand, prices and returns that
  update as residuals arrive. The guarantees then rely on weaker, approximate assumptions.
- **Stability as a design goal.** The analysis shows that algorithmic stability, a classic route to generalization bounds (Bousquet & Elisseeff, 2002), also buys tight predictive inference. This is one
  more reason to prefer regularized or ensembled models when uncertainty matters.

## Further reading

- R. F. Barber, E. Candès, A. Ramdas & R. Tibshirani, "Predictive inference with the jackknife+" (*Ann. Stat.*, 2021).
- B. Kim, C. Xu & R. F. Barber, "Predictive inference is free with the jackknife+-after-bootstrap" (NeurIPS 2020).
- C. Xu & Y. Xie, "Conformal prediction interval for dynamic time-series" (ICML 2021).
- A. Angelopoulos & S. Bates, "A gentle introduction to conformal prediction" (2021/2023), §5.
