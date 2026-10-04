"""Tracy–Widom CDFs via Bornemann's Nyström discretization of Fredholm determinants."""
import numpy as np
from scipy.special import airy

def _nodes(s, m=60, length=14.0):
    x, w = np.polynomial.legendre.leggauss(m)
    return s + (x + 1) * length / 2, w * length / 2

def F1(s):
    """GOE: det(I - K1) on L^2(s, inf), K1(x, y) = Ai((x + y)/2) / 2."""
    x, w = _nodes(s)
    K = 0.5 * airy((x[:, None] + x[None, :]) / 2)[0]
    sw = np.sqrt(w)
    return np.linalg.det(np.eye(len(x)) - sw[:, None] * K * sw[None, :])

def F2(s):
    """GUE: det(I - K_Airy) on L^2(s, inf)."""
    x, w = _nodes(s)
    ai, aip, _, _ = airy(x)
    X, Y = np.meshgrid(x, x, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        K = (ai[:, None] * aip[None, :] - aip[:, None] * ai[None, :]) / (X - Y)
    K[np.diag_indices_from(K)] = aip**2 - x * ai**2
    sw = np.sqrt(w)
    return np.linalg.det(np.eye(len(x)) - sw[:, None] * K * sw[None, :])

def density(F, grid):
    c = np.array([F(s) for s in grid])
    return c, np.gradient(c, grid)
