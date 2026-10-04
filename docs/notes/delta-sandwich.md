# The delta method & sandwich standard errors

!!! tldr "TL;DR"
    Two tools give standard errors for almost anything. The **delta method**: if $\sqrt n(\hat\theta - \theta)\Rightarrow N(0,\Sigma)$ then $\sqrt n\big(g(\hat\theta) - g(\theta)\big)\Rightarrow N(0,\nabla g^\top\Sigma\nabla g)$, i.e. linearize and propagate. The **sandwich**: an estimator defined by
    $\sum_i\psi(x_i,\hat\theta) = 0$ has asymptotic covariance $A^{-1}BA^{-1}/n$, with "bread" $A = -\E\nabla\psi$ and "meat" $B = \E\psi\psi^\top$. It reduces to $I^{-1}$ for a correct likelihood, and it stays valid when the model's variance assumptions are wrong.
    For OLS it gives White's heteroskedasticity-robust standard errors, now the default in econometrics and finance.

## Why care?

Point estimates are easy. Honest error bars are hard, for two common reasons:

1. **The quantity is a nonlinear function of simple estimates**: a Sharpe ratio ($\hat\mu/\hat\sigma$), a click-through rate defined as total clicks over total impressions across users, an elasticity, an odds ratio, the ratio of two model accuracies. You know how the inputs fluctuate. How does the output fluctuate?
2. **The model's noise assumptions are wrong**: regression errors are heteroskedastic (variance changes with $x$), counts are overdispersed, observations are clustered. The textbook formula $\sigma^2(X^\top X)^{-1}$ can then be badly off, usually too small.

The delta method solves the first and the sandwich the second. Together they are the default machinery behind regression tables in econometrics, metric analysis in online experimentation (Deng et al., 2018), and robust inference for misspecified models.

## Building blocks

**Asymptotic normality.** Many estimators satisfy $\sqrt n(\hat\theta - \theta)\Rightarrow N(0,\Sigma)$, by the CLT plus Slutsky's lemma ($X_n\Rightarrow X$ and $Y_n\to c$ in probability imply $X_nY_n\Rightarrow cX$).

**M-estimators / estimating equations.** $\hat\theta$ solves $\frac1n\sum_i\psi(x_i,\hat\theta) = 0$ for some function $\psi$ with $\E\psi(X,\theta_0) = 0$. Examples:

- MLE: $\psi$ is the score $\nabla\log p_\theta$.
- OLS: $\psi(x,y;\beta) = x(y - x^\top\beta)$.
- Quantiles, Huber regression and GLMs: $\psi$ is the derivative of the respective loss ([robust regression](robust-m-estimators.md)).
- Moments: $\psi = x - \theta$.

## The main results

### The delta method

!!! theorem "Theorem (delta method)"
    If $\sqrt n(\hat\theta - \theta)\Rightarrow N(0,\Sigma)$ and $g:\R^k\to\R^m$ is differentiable at $\theta$ with Jacobian $J = \nabla g(\theta)^\top$, then

    $$
    \sqrt n\big(g(\hat\theta) - g(\theta)\big)\Rightarrow N\big(0,\ J\Sigma J^\top\big).
    $$

**Proof.** By Taylor's theorem, $g(\hat\theta) - g(\theta) = J(\hat\theta - \theta) + o(\|\hat\theta - \theta\|)$. Multiply by $\sqrt n$. The remainder is $o_p(1)$ because $\sqrt n\|\hat\theta - \theta\| = O_p(1)$, and $J\sqrt n(\hat\theta - \theta)\Rightarrow N(0, J\Sigma J^\top)$. $\square$

If $J = 0$ (for example $g(\theta) = \theta^2$ at $\theta = 0$), the first-order term vanishes, and a second-order delta method gives a non-Gaussian, chi-square-type limit at rate $1/n$.

**Example: the Sharpe ratio.** With $\hat\theta = (\hat\mu,\hat\sigma^2)$ and $g = \mu/\sigma$: $\nabla g = (1/\sigma,\ -\mu/(2\sigma^3))$. For general returns with skewness $\gamma_3$ and kurtosis $\gamma_4$, $\Sigma = \begin{pmatrix}\sigma^2 & \gamma_3\sigma^3\\ \gamma_3\sigma^3 & (\gamma_4-1)\sigma^4\end{pmatrix}$, giving

$$
n\Var(\widehat{\mathrm{SR}})\approx1 - \gamma_3\,\mathrm{SR} + \frac{\gamma_4 - 1}{4}\mathrm{SR}^2,
$$

which reduces to Lo's $1 + \mathrm{SR}^2/2$ for Gaussian returns ($\gamma_3 = 0$, $\gamma_4 = 3$). **Negative skewness inflates the uncertainty**, as the example below shows.

### The sandwich

!!! theorem "Theorem (asymptotics of M-estimators)"
    Suppose $\hat\theta\to\theta_0$ solves $\frac1n\sum_i\psi(x_i,\hat\theta) = 0$, with $\E\psi(X,\theta_0) = 0$, $A = -\E\big[\nabla_\theta\psi(X,\theta_0)\big]$ invertible, and $B = \E\big[\psi(X,\theta_0)\psi(X,\theta_0)^\top\big]$ finite. Under regularity conditions,

    $$
    \sqrt n(\hat\theta - \theta_0)\Rightarrow N\big(0,\ A^{-1}BA^{-\top}\big).
    $$

    The plug-in estimate $\hat A^{-1}\hat B\hat A^{-\top}/n$, with sample averages at $\hat\theta$, is the **sandwich estimator**.

**Proof sketch.** Expand the estimating equation around $\theta_0$: $0 = \frac1n\sum\psi(x_i,\hat\theta)\approx\frac1n\sum\psi(x_i,\theta_0) - \hat A(\hat\theta - \theta_0)$. So $\sqrt n(\hat\theta - \theta_0)\approx\hat A^{-1}\cdot\frac{1}{\sqrt n}\sum_i\psi(x_i,\theta_0)$. The CLT gives $\frac1{\sqrt n}\sum\psi\Rightarrow N(0,B)$, the LLN gives $\hat A\to A$, and Slutsky finishes. $\square$

The term $\psi(x_i,\theta_0)$ scaled by $A^{-1}$ is the **influence function** of observation $i$: its first-order effect on $\hat\theta$. The sandwich is the variance of a sum of influence functions.

**Correct likelihood.** For a well-specified MLE, $\psi$ is the score, and the information equality gives $A = B = I$, so the sandwich collapses to $I^{-1}$ ([Fisher information](fisher-kl-cramer-rao.md)). Under misspecification $A\neq B$, $I^{-1}$ is wrong, and the sandwich is still right for the "pseudo-true" parameter (Huber 1967, White 1982).

### OLS with heteroskedasticity

For OLS, $\psi = x(y - x^\top\beta)$, $A = \E[xx^\top]$ and $B = \E[\varepsilon^2xx^\top]$. The sample version is

$$
\widehat\Cov(\hat\beta) = (X^\top X)^{-1}\Big(\sum_i\hat e_i^2x_ix_i^\top\Big)(X^\top X)^{-1}\qquad\text{(HC0, White 1980)}.
$$

If $\Var(\varepsilon\mid x) = \sigma^2$ is constant, $B = \sigma^2A$ and this reduces to $\sigma^2(X^\top X)^{-1}$. Otherwise the classical formula is wrong, typically too small when the noise is largest where $x$ is extreme. Small-sample refinements rescale the squared residuals: HC1 by $\frac{n}{n-p}$, and HC3 by $\frac{1}{(1-h_{ii})^2}$ using
[leverages](ols-gauss-markov.md). **Cluster-robust** versions sum $\psi$ within clusters (days, firms, users, test questions sharing a passage) before forming $B$. Time-series dependence needs [HAC estimators](hac-newey-west.md).

## Examples

### Sharpe ratio error bars and robust regression standard errors

```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)

# 1) Delta method for the Sharpe ratio of a "short-volatility" strategy: small steady gains, rare crashes (monthly data).
def draw(size):
    crash = rng.uniform(size=size) < 0.05
    return np.where(crash, rng.normal(-0.15, 0.05, size), rng.normal(0.02, 0.02, size))
pop = draw(10_000_000)                                              # population moments by brute force
mu, sd = pop.mean(), pop.std()
sr, g3, g4 = mu / sd, stats.skew(pop), stats.kurtosis(pop, fisher=False)
n, reps = 120, 20000                                                # 10 years of monthly returns
r = draw((reps, n)); sr_hat = r.mean(1) / r.std(1, ddof=1)
se_normal = np.sqrt((1 + sr**2 / 2) / n)                            # Lo (2002): assumes normal returns
se_delta = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr**2) / n)        # delta method with skewness and kurtosis
print(f"monthly SR {sr:.3f}, skewness {g3:.2f}, kurtosis {g4:.1f}")
print(f"SE of estimated SR: simulated {sr_hat.std():.4f}   delta (normal) {se_normal:.4f}   delta (skew+kurtosis) {se_delta:.4f}")

# 2) Classical vs sandwich (heteroskedasticity-robust) standard errors for an OLS slope.
def ols_ses(x, y):
    X = np.column_stack([np.ones_like(x), x]); XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y; e = y - X @ b; n, p = X.shape
    classical = np.sqrt(np.diag(XtX_inv) * (e @ e) / (n - p))
    meat = X.T @ (X * (e**2)[:, None])
    hc0 = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))                # White (1980)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)                     # leverages
    meat3 = X.T @ (X * (e**2 / (1 - h) ** 2)[:, None])
    hc3 = np.sqrt(np.diag(XtX_inv @ meat3 @ XtX_inv))               # HC3: small-sample correction
    return b[1], classical[1], hc0[1], hc3[1]

for n in [50, 500]:
    cover = np.zeros(3)
    for _ in range(4000):
        x = rng.standard_normal(n)
        y = 1.0 + 0.5 * x + np.abs(x) * rng.standard_normal(n)      # noise sd grows with |x|
        b, *ses = ols_ses(x, y)
        cover += [abs(b - 0.5) <= 1.96 * s for s in ses]
    print(f"n = {n:3d}  95% CI coverage:  classical {cover[0]/4000:.3f}   HC0 {cover[1]/4000:.3f}   HC3 {cover[2]/4000:.3f}")
# monthly SR 0.265, skewness -3.20, kurtosis 15.6
# SE of estimated SR: simulated 0.1510   delta (normal) 0.0929   delta (skew+kurtosis) 0.1325
# n =  50  95% CI coverage:  classical 0.743   HC0 0.907   HC3 0.937
# n = 500  95% CI coverage:  classical 0.745   HC0 0.946   HC3 0.949
```

**The Sharpe ratio.** The strategy looks great (monthly Sharpe 0.27, about 0.92 annualized), but its returns are strongly negatively skewed. The normal-theory standard error understates the uncertainty by 40%. The delta method with the right moments recovers most of the gap.
The rest is because 120 observations of a kurtosis-15 distribution are still far from asymptopia, which is a good reason to also bootstrap.

**The regression.** Classical intervals cover only 74%, **regardless of sample size**: the formula is wrong, not merely noisy. White's HC0 is right asymptotically, and HC3 already works at $n = 50$.

## Exercises

!!! question "Exercise 1 · warm-up: log of a mean"
    If $\bar X$ is the mean of $n$ i.i.d. positive variables with mean $\mu$ and variance $\sigma^2$, find the asymptotic variance of $\log\bar X$. Why is $\log$ a "variance-stabilizing" transform for Poisson counts?

    ??? success "Solution"
        $g = \log$ and $g' = 1/\mu$, so $\Var(\log\bar X)\approx\frac{\sigma^2}{n\mu^2}$, the squared coefficient of variation over $n$. For Poisson, $\sigma^2 = \mu$, giving $\frac{1}{n\mu}$, which still depends on $\mu$. The variance-stabilizing transform for Poisson is $\sqrt x$: $g' = \frac{1}{2\sqrt\mu}$ gives $\Var\approx\frac{\mu}{4n\mu} = \frac{1}{4n}$, constant. (For the log-normal, or multiplicative noise, $\log$ is the stabilizer.)

!!! question "Exercise 2 · ratio metrics in A/B tests"
    A metric is defined as total clicks over total page views, $\hat R = \bar C/\bar V$, where $(C_u, V_u)$ are clicks and views of user $u$ (users i.i.d., but clicks and views within a user correlated). Use the delta method to find $\Var(\hat R)$. Why is the naive binomial formula $\hat R(1-\hat R)/\sum_uV_u$ wrong?

    ??? success "Solution"
        $g(c,v) = c/v$ with gradient $(1/\mu_V,\ -\mu_C/\mu_V^2)$. So $\Var(\hat R)\approx\frac{1}{n\mu_V^2}\big[\Var(C) - 2R\Cov(C,V) + R^2\Var(V)\big]$ with $R = \mu_C/\mu_V$.
        The binomial formula treats every page view as independent, but the randomization unit is the *user*, and views from the same user are correlated (heavy users behave differently). The delta method on user-level totals respects the randomization unit. Deng et al. (2018) show the naive formula can understate variance severalfold.

!!! question "Exercise 3 · the sandwich collapses under homoskedasticity"
    Show that if $\E[\varepsilon^2\mid x] = \sigma^2$, then $A^{-1}BA^{-1} = \sigma^2(\E xx^\top)^{-1}$, matching the classical OLS covariance.

    ??? success "Solution"
        $B = \E[\varepsilon^2xx^\top] = \E\big[\E[\varepsilon^2\mid x]xx^\top\big] = \sigma^2\E[xx^\top] = \sigma^2A$. So $A^{-1}BA^{-1} = \sigma^2A^{-1}$, and the sample analogue $\sigma^2(X^\top X)^{-1}\cdot n/n$ gives the textbook formula.

!!! question "Exercise 4 · Sharpe ratio under normality"
    For i.i.d. $N(\mu,\sigma^2)$ returns, $\hat\mu$ and $\hat\sigma^2$ are asymptotically independent with variances $\sigma^2/n$ and $2\sigma^4/n$. Derive Lo's formula $n\Var(\widehat{\mathrm{SR}})\approx1 + \mathrm{SR}^2/2$. How many years of monthly data are needed for a standard error of $0.1$ on a monthly Sharpe of $0.3$?

    ??? success "Solution"
        $\nabla g = (1/\sigma,\ -\mu/(2\sigma^3))$, so $n\Var\approx\frac{1}{\sigma^2}\sigma^2 + \frac{\mu^2}{4\sigma^6}2\sigma^4 = 1 + \frac{\mu^2}{2\sigma^2} = 1 + \frac{\mathrm{SR}^2}{2}$.
        For SE $0.1$: $n = \frac{1 + 0.045}{0.01}\approx105$ months, about 9 years. And that's under normality. With realistic negative skew, as in the example, it takes considerably longer.

!!! question "Exercise 5 · stretch: misspecified Poisson regression"
    Fit a Poisson GLM (canonical link) to counts with correct mean $\mu_i = e^{x_i^\top\beta}$ but $\Var(y_i\mid x_i) = v_i$ arbitrary. Show that the sandwich is $(X^\top WX)^{-1}\big(X^\top\diag(v_i)X\big)(X^\top WX)^{-1}$ with $W = \diag(\mu_i)$, and that it reduces to $\phi(X^\top WX)^{-1}$ when $v_i = \phi\mu_i$ (quasi-Poisson).
    Which one would you report if you don't know the variance structure?

    ??? success "Solution"
        $\psi_i = x_i(y_i - \mu_i)$, so $A = \frac1n\sum_i\mu_ix_ix_i^\top = \frac1nX^\top WX$ and $B = \frac1n\sum_i\E[(y_i - \mu_i)^2]x_ix_i^\top = \frac1nX^\top\diag(v_i)X$. The sandwich follows. With $v_i = \phi\mu_i$, $B = \phi A$, so $A^{-1}BA^{-1} = \phi A^{-1}$, the quasi-Poisson scaling from the [GLM note](glm.md).
        If you don't know the variance structure, report the sandwich (estimating $v_i$ by $\hat e_i^2$). It is valid whatever $\Var(y\mid x)$ is, at a small efficiency cost when a simpler model would have been right.

## Where it shows up

- **Experimentation platforms.** Ratio metrics (CTR, revenue per session), percent changes ($\frac{\bar Y_T - \bar Y_C}{\bar Y_C}$) and variance-reduced estimators (CUPED) all get their standard errors from the delta method, computed at the level of the randomization unit.
- **Econometrics and empirical finance.** Robust (HC) and cluster-robust standard errors are the default in regression tables. Factor-model alphas, event studies and Fama–MacBeth regressions add [Newey–West](hac-newey-west.md) corrections for time dependence.
- **ML evaluation.** Benchmark questions often come in clusters (several questions per passage or per document), so clustered standard errors are needed (Miller, 2024). Differences between two models on the same questions are paired, and the delta method handles ratios such as relative improvement.
- **Influence functions and data attribution.** The sandwich's ingredient $A^{-1}\psi(x_i,\hat\theta)$ is the influence function used to estimate how much each training point affects a model's parameters or predictions, from robust statistics to modern data-attribution methods for deep networks.
- **Quant performance analysis.** Standard errors of Sharpe, Sortino and information ratios, and their adjustments for skewness, kurtosis and autocorrelation (Lo 2002; Mertens 2002; Opdyke 2007), are delta-method calculations.

## Further reading

- A. W. van der Vaart, *Asymptotic Statistics* (1998), Ch. 3 (delta method) and Ch. 5 (M-estimators).
- H. White, "A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity" (*Econometrica*, 1980).
- A. Deng, U. Knoblich & J. Lu, "Applying the delta method in metric analytics: a practical guide with novel ideas" (KDD 2018).
- A. Lo, "The statistics of Sharpe ratios" (*Financial Analysts Journal*, 2002).
