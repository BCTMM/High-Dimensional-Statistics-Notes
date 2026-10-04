# Matrix calculus & trace tricks

!!! tldr "TL;DR"
    Don't differentiate matrix expressions entry by entry. Compute the **differential** $df$ using a handful of rules ($d(XY) = dX\,Y + X\,dY$, $d(X^{-1}) = -X^{-1}dX\,X^{-1}$,
    $d\log\det X = \tr(X^{-1}dX)$), then massage it into the form $df = \tr(G^\top dX)$ using **cyclicity of the trace**. The gradient is $G$. This one recipe gives the Gaussian
    covariance MLE, the normal equations, backpropagation through linear layers, and the gradients of Gaussian-process marginal likelihoods, with no index gymnastics and no layout confusion.

## Why care?

Matrix-valued parameters are everywhere: weight matrices, covariance matrices, kernel matrices, portfolio weights, attention projections. Deriving their gradients by writing out
$\partial/\partial X_{ij}$ is slow and error-prone. Different textbooks also disagree on whether the gradient of a scalar with respect to a column vector is a row or a column ("numerator vs denominator layout").

The differential-plus-trace method sidesteps all of that. It is also exactly what **reverse-mode automatic differentiation** does under the hood. Each operation knows how to pull back
$\tr(G^\top dY)$ to $\tr(\tilde G^\top dX)$. Knowing the method lets you sanity-check autodiff, write custom backward passes, and derive closed-form estimators that later notes rely on
(Gaussian MLE, [Fisher information](fisher-kl-cramer-rao.md), [GLM](glm.md) Hessians, Ledoit–Wolf, RIE).

## Building blocks

**Frobenius inner product.** $\langle A, B\rangle = \tr(A^\top B) = \sum_{ij}A_{ij}B_{ij}$. It is the ordinary dot product of the matrices viewed as long vectors.

**Trace identities.**

- *Cyclicity:* $\tr(ABC) = \tr(CAB) = \tr(BCA)$ (for compatible shapes). You can rotate but not reorder arbitrarily.
- *Transpose:* $\tr(A) = \tr(A^\top)$.
- *Scalars are traces:* $a^\top Xb = \tr(a^\top Xb) = \tr(ba^\top X)$. This is how you turn quadratic forms into traces.

**The differential.** $df$ is the linear part of $f(X + dX) - f(X)$. For scalar $f$ of a matrix $X$:

$$
df = \sum_{ij}\frac{\partial f}{\partial X_{ij}}dX_{ij} = \tr\Big(\big(\nabla_Xf\big)^\top dX\Big).
$$

**Identification rule:** *if you can write $df = \tr(G^\top dX)$ for some $G$, then $\nabla_Xf = G$* (same shape as $X$). Everything else is getting $df$ into this form.

## The main results

!!! theorem "Theorem (the core differentials)"
    For matrices of compatible sizes (and invertible where needed):

    | expression | differential |
    |---|---|
    | $AXB$ ($A, B$ constant) | $A\,dX\,B$ |
    | $XY$ | $dX\,Y + X\,dY$ |
    | $X^\top$ | $(dX)^\top$ |
    | $X^{-1}$ | $-X^{-1}\,dX\,X^{-1}$ |
    | $\tr X$ | $\tr(dX)$ |
    | $\log\det X$ | $\tr(X^{-1}dX)$ |
    | $\det X$ | $\det X\,\tr(X^{-1}dX)$ |
    | $\lambda_k(X)$ ($X$ symmetric, $\lambda_k$ simple) | $u_k^\top dX\,u_k$ |

**Proofs of the non-obvious ones.**

*Inverse.* Differentiate $XX^{-1} = I$: $dX\,X^{-1} + X\,d(X^{-1}) = 0$, so $d(X^{-1}) = -X^{-1}dX\,X^{-1}$. This is the matrix version of $d(1/x) = -dx/x^2$, and the order matters.

*Log-determinant.* $\det(X + dX) = \det X\,\det(I + X^{-1}dX)$, and $\det(I + E) = 1 + \tr E + O(\|E\|^2)$, because the eigenvalues of $I + E$ are $1 + \mu_i$ and their product is $1 + \sum\mu_i + \dots$. So
$\log\det(X + dX) - \log\det X = \tr(X^{-1}dX) + O(\|dX\|^2)$.

*Eigenvalue.* This is the first-order perturbation formula from the [Davis–Kahan](davis-kahan.md) note. $\lambda_k = u_k^\top Xu_k$, and the term coming from $du_k$ vanishes because $u_k^\top du_k = 0$.

**Corollary (gradients).** By the identification rule, $\nabla_X\tr(AX) = A^\top$, $\nabla_X\log\det X = X^{-\top}$, $\nabla_X\,a^\top X^{-1}b = -X^{-\top}ab^\top X^{-\top}$, and $\nabla_X\lambda_k(X) = u_ku_k^\top$.

### Worked derivation 1: the Gaussian covariance MLE

For $x_1,\dots,x_n\sim N(0,\Sigma)$ i.i.d., with $S = \frac1n\sum_ix_ix_i^\top$, the log-likelihood is, up to a constant,

$$
\ell(\Sigma) = -\frac n2\log\det\Sigma - \frac12\sum_ix_i^\top\Sigma^{-1}x_i = -\frac n2\Big[\log\det\Sigma + \tr\big(\Sigma^{-1}S\big)\Big].
$$

(The quadratic forms were turned into a trace: $\sum_ix_i^\top\Sigma^{-1}x_i = \tr(\Sigma^{-1}\sum_ix_ix_i^\top)$.) Differentiate with the table:

$$
d\ell = -\frac n2\Big[\tr(\Sigma^{-1}d\Sigma) - \tr(\Sigma^{-1}d\Sigma\,\Sigma^{-1}S)\Big] = \frac n2\tr\Big(\big(\Sigma^{-1}S\Sigma^{-1} - \Sigma^{-1}\big)d\Sigma\Big).
$$

So $\nabla_\Sigma\ell = \frac n2(\Sigma^{-1}S\Sigma^{-1} - \Sigma^{-1})$, which vanishes iff $\Sigma = S$. The sample covariance is the MLE. The same computation in "precision" coordinates $\Theta = \Sigma^{-1}$
gives $\ell = \frac n2[\log\det\Theta - \tr(\Theta S)]$, a concave function. That is the starting point of the graphical LASSO.

### Worked derivation 2: backprop through a linear layer

Let $Y = XW$ (batch $X\in\R^{B\times d_{in}}$, weights $W\in\R^{d_{in}\times d_{out}}$) and suppose the upstream gradient is $G = \nabla_YL$, i.e. $dL = \tr(G^\top dY)$. Then:

- with respect to $W$: $dY = X\,dW$, so $dL = \tr(G^\top X\,dW) = \tr\big((X^\top G)^\top dW\big)$, hence $\nabla_WL = X^\top G$;
- with respect to $X$: $dY = dX\,W$, so $dL = \tr(G^\top dX\,W) = \tr(WG^\top dX) = \tr\big((GW^\top)^\top dX\big)$, hence $\nabla_XL = GW^\top$.

Those two lines are the backward pass of every dense layer. The weight gradient is a sum over the batch of rank-one outer products $x_b g_b^\top$, a structure that K-FAC and Shampoo exploit.

### Vectorization and Kronecker products

Sometimes you need the full Jacobian of a matrix-valued map, for example for Fisher information or second-order methods. Stack columns with $\operatorname{vec}$. The key identity is

$$
\operatorname{vec}(AXB) = (B^\top\otimes A)\operatorname{vec}(X).
$$

For instance, $\operatorname{vec}(d\Sigma^{-1}) = -(\Sigma^{-1}\otimes\Sigma^{-1})\operatorname{vec}(d\Sigma)$, so the Jacobian of matrix inversion is $-\Sigma^{-1}\otimes\Sigma^{-1}$. Kronecker structure makes huge Jacobians cheap
to store and invert: $(A\otimes B)^{-1} = A^{-1}\otimes B^{-1}$.

## Examples

### Check your gradients numerically

Always test a derived gradient against finite differences. It takes five lines and catches transposes and missing factors of 2.

```python
import numpy as np
rng = np.random.default_rng(0)
p, n = 5, 200
A = rng.standard_normal((p, p)); Sigma = A @ A.T + p * np.eye(p)          # a covariance
X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
S = X.T @ X / n

def loglik(Sig):                                    # Gaussian log-likelihood (mean 0), up to a constant
    _, logdet = np.linalg.slogdet(Sig)
    return -0.5 * n * (logdet + np.trace(np.linalg.solve(Sig, S)))

def grad_loglik(Sig):                               # from d loglik = (n/2) tr[(Σ⁻¹ S Σ⁻¹ − Σ⁻¹) dΣ]
    Si = np.linalg.inv(Sig)
    return 0.5 * n * (Si @ S @ Si - Si)

def numerical_grad(f, M, h=1e-6):
    G = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            E = np.zeros_like(M); E[i, j] = h
            G[i, j] = (f(M + E) - f(M - E)) / (2 * h)
    return G

Sig0 = Sigma + 0.5 * np.eye(p)
err = np.max(np.abs(grad_loglik(Sig0) - numerical_grad(loglik, Sig0)))
print(f"max |analytic - numerical| gradient of the log-likelihood: {err:.2e}")
print(f"gradient at the MLE Sigma = S (should vanish): {np.max(np.abs(grad_loglik(S))):.2e}")

# d log det X = tr(X^{-1} dX) and d X^{-1} = -X^{-1} dX X^{-1}, checked along a random direction
dX = rng.standard_normal((p, p)) * 1e-6
lhs = np.linalg.slogdet(Sig0 + dX)[1] - np.linalg.slogdet(Sig0)[1]
print(f"log det: finite difference {lhs:.6e}   tr(X^-1 dX) {np.trace(np.linalg.solve(Sig0, dX)):.6e}")
Si = np.linalg.inv(Sig0)
print(f"inverse: max error {np.max(np.abs((np.linalg.inv(Sig0 + dX) - Si) - (-Si @ dX @ Si))):.2e}   (size of dX^2 terms ~1e-14)")
# max |analytic - numerical| gradient of the log-likelihood: 1.94e-07
# gradient at the MLE Sigma = S (should vanish): 2.78e-15
# log det: finite difference 3.003731e-07   tr(X^-1 dX) 3.003732e-07
# inverse: max error 2.31e-14   (size of dX^2 terms ~1e-14)
```

The finite-difference check perturbs each entry independently, so it tests the gradient over *all* matrices. For symmetric-matrix parameters you may also see $2G - \operatorname{diag}(G)$ in the literature,
which comes from treating $\Sigma_{ij}$ and $\Sigma_{ji}$ as one variable. Both are correct for their own parametrizations. Just be consistent.

## Exercises

!!! question "Exercise 1 · warm-up"
    Find $\nabla_X\tr(AXB)$ and $\nabla_X\tr(X^\top AX)$.

    ??? success "Solution"
        $d\tr(AXB) = \tr(A\,dX\,B) = \tr(BA\,dX)$, so $\nabla = (BA)^\top = A^\top B^\top$.
        $d\tr(X^\top AX) = \tr(dX^\top AX) + \tr(X^\top A\,dX) = \tr(X^\top A^\top dX) + \tr(X^\top A\,dX)$, so $\nabla = (A + A^\top)X$, which is $2AX$ for symmetric $A$. This is the matrix analogue of $\frac{d}{dx}ax^2 = 2ax$.

!!! question "Exercise 2 · normal equations in one line"
    Derive the gradient of $f(\beta) = \|y - X\beta\|^2 + \lambda\|\beta\|^2$ with differentials and recover the [ridge](ridge.md) solution.

    ??? success "Solution"
        $f = (y - X\beta)^\top(y - X\beta) + \lambda\beta^\top\beta$, so $df = -2(y - X\beta)^\top X\,d\beta + 2\lambda\beta^\top d\beta$, hence $\nabla f = -2X^\top(y - X\beta) + 2\lambda\beta$. Setting it to zero gives
        $(X^\top X + \lambda I)\beta = X^\top y$.

!!! question "Exercise 3 · logistic regression's Hessian"
    For $L(\beta) = \sum_i[\log(1 + e^{x_i^\top\beta}) - y_ix_i^\top\beta]$, show $\nabla L = X^\top(p - y)$ with $p_i = \sigma(x_i^\top\beta)$, and that the Hessian is $X^\top WX$ with $W = \diag(p_i(1-p_i))$. Conclude that $L$ is convex.

    ??? success "Solution"
        $dL = \sum_i[\sigma(x_i^\top\beta) - y_i]x_i^\top d\beta = (p - y)^\top X\,d\beta$, so $\nabla L = X^\top(p - y)$. Differentiating again with $dp_i = p_i(1-p_i)x_i^\top d\beta$, i.e. $dp = WX\,d\beta$:
        $d(\nabla L) = X^\top WX\,d\beta$, so the Hessian is $X^\top WX\succeq0$, since $W$ has positive diagonal. This is the matrix inside Newton's method (IRLS) for [GLMs](glm.md).

!!! question "Exercise 4 · Gaussian-process marginal likelihood"
    The log marginal likelihood of a GP is $\ell(\theta) = -\frac12y^\top K_\theta^{-1}y - \frac12\log\det K_\theta + \text{const}$. Show that

    $$
    \frac{\partial\ell}{\partial\theta} = \frac12\tr\Big(\big(\alpha\alpha^\top - K_\theta^{-1}\big)\frac{\partial K_\theta}{\partial\theta}\Big),\qquad\alpha = K_\theta^{-1}y .
    $$

    ??? success "Solution"
        $d(y^\top K^{-1}y) = -y^\top K^{-1}dK\,K^{-1}y = -\alpha^\top dK\,\alpha = -\tr(\alpha\alpha^\top dK)$, and $d\log\det K = \tr(K^{-1}dK)$. So $d\ell = \frac12\tr(\alpha\alpha^\top dK) - \frac12\tr(K^{-1}dK)$, with $dK = \frac{\partial K}{\partial\theta}d\theta$.
        This formula is how GP libraries tune kernel hyperparameters by gradient ascent.

!!! question "Exercise 5 · stretch: Fisher information of a Gaussian covariance"
    For one observation $x\sim N(0,\Sigma)$ with log-likelihood $\ell(\Sigma) = -\frac12\log\det\Sigma - \frac12x^\top\Sigma^{-1}x$, compute the second differential $d^2\ell$ and show that the Fisher information, viewed as a quadratic form in $d\Sigma$, is

    $$
    -\E[d^2\ell] = \frac12\tr\big(\Sigma^{-1}d\Sigma\,\Sigma^{-1}d\Sigma\big),\qquad\text{i.e.}\qquad\mathcal I = \tfrac12\,\Sigma^{-1}\otimes\Sigma^{-1}\ \text{(in vec coordinates)}.
    $$

    ??? success "Solution"
        First differential: $d\ell = -\frac12\tr(\Sigma^{-1}d\Sigma) + \frac12x^\top\Sigma^{-1}d\Sigma\,\Sigma^{-1}x$. Differentiate again (with $d^2\Sigma = 0$), using $d(\Sigma^{-1}) = -\Sigma^{-1}d\Sigma\,\Sigma^{-1}$:

        $$
        d^2\ell = \frac12\tr(\Sigma^{-1}d\Sigma\,\Sigma^{-1}d\Sigma) - x^\top\Sigma^{-1}d\Sigma\,\Sigma^{-1}d\Sigma\,\Sigma^{-1}x .
        $$

        Taking expectations with $\E[xx^\top] = \Sigma$, the second term becomes $-\tr(\Sigma^{-1}d\Sigma\,\Sigma^{-1}d\Sigma)$. So $-\E d^2\ell = \frac12\tr(\Sigma^{-1}d\Sigma\,\Sigma^{-1}d\Sigma)$. In vec form, using $\tr(A^\top B) = \operatorname{vec}(A)^\top\operatorname{vec}(B)$
        and the Kronecker identity, this is $\frac12\operatorname{vec}(d\Sigma)^\top(\Sigma^{-1}\otimes\Sigma^{-1})\operatorname{vec}(d\Sigma)$. This Kronecker-structured metric is the Fisher–Rao geometry on covariance matrices, invariant under
        $\Sigma\mapsto A\Sigma A^\top$.

## Where it shows up

- **Automatic differentiation.** Reverse-mode AD is the trace trick applied mechanically. Each primitive implements a "vector–Jacobian product" mapping the upstream $G$ to $\tilde G$ such that $\tr(G^\top dY) = \tr(\tilde G^\top dX)$.
  Writing custom backward passes (for fused attention kernels, say, or for numerically stable log-det or Cholesky layers) needs exactly this.
- **Second-order and preconditioned optimizers.** K-FAC approximates a layer's Fisher by a Kronecker product $A\otimes G$ of input and gradient covariances, using $\nabla_WL = X^\top G$ from above.
  Shampoo maintains Kronecker-factored preconditioners. Both rely on $(A\otimes B)^{-1} = A^{-1}\otimes B^{-1}$.
- **Normalizing flows and Gaussian models.** Log-likelihoods of flows need $\log|\det J|$ and its gradient $\tr(J^{-1}dJ)$, which motivates architectures with triangular Jacobians. Gaussian MLE, the graphical
  LASSO and Kalman filters use the log-det and inverse differentials constantly.
- **Covariance estimation in quant.** Deriving the Ledoit–Wolf optimal intensity, the minimum-variance portfolio, or the sensitivity of portfolio risk to covariance errors ($dR^2 = w^\top d\Sigma\,w$) is
  routine with differentials.
- **Gaussian processes and kernel methods.** Hyperparameter learning by marginal likelihood (Exercise 4) and leave-one-out formulas are matrix-calculus exercises.

## Further reading

- J. R. Magnus & H. Neudecker, *Matrix Differential Calculus with Applications in Statistics and Econometrics* (3rd ed., 2019). The systematic treatment of the differential method.
- K. B. Petersen & M. S. Pedersen, *The Matrix Cookbook*. An invaluable lookup table.
- T. Minka, "Old and new matrix algebra useful for statistics" (2000). Short, practical notes on the differential approach.
