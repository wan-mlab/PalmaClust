import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------- exact helpers (no sampling, no densification) ----------

def gini_index_sparse_exact(nonzero: np.ndarray, n_cells: int, unbiased: bool = True) -> float:
    """
    Exact Gini from your formula, computed using ONLY the sorted nonzeros and zero count.
    Returns NaN if mu*N == 0 (e.g., all zeros), matching your original behavior.
    """
    v = np.asarray(nonzero, dtype=np.float64)
    n = int(n_cells)
    m = v.size
    z = n - m
    if n <= 0 or z < 0:
        return np.nan

    # sort positives, compute weighted sum
    if m:
        y = np.sort(v)                   # ascending
        S = y.sum()                      # total sum
        mu = S / n
        j = np.arange(1, m + 1, dtype=np.float64)
        Jy = (j * y).sum()
        dsum = (2.0 * z - n - 1.0) * S + 2.0 * Jy
    else:
        S = 0.0
        mu = 0.0
        dsum = 0.0

    N = float(n * (n - 1) if unbiased else n * n)
    # identical semantics to your dense function (NaN if mu*N == 0)
    return dsum / (mu * N)