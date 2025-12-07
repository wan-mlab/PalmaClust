import numpy as np

def fano_factor_from_sparse_nonzeros(
    nonzero: np.ndarray,
    n_cells: int,
    ddof: int = 1,                 # 1 → sample variance; 0 → population variance
) -> float:
    """
    Exact Fano factor for a single gene from sparse non-zeros.

    Parameters
    ----------
    nonzero : 1D array-like of floats/ints
        Strictly positive counts for this gene (zeros omitted).
    n_cells : int
        Total number of cells (including zeros).
    ddof : int
        Degrees of freedom for variance. Use 1 for sample variance (default),
        0 for population variance.

    Returns
    -------
    fano : float
        Variance / mean over all cells (zeros included).
        NaN if mean==0 and `nan_for_zero_mean=True`, or if n_cells <= ddof.
    """
    v = np.asarray(nonzero, dtype=np.float64)
    n = int(n_cells)
    if n <= ddof:
        return np.nan

    S  = v.sum()
    S2 = (v * v).sum()
    mu = S / n

    # Var = (sum(x^2) - n*mu^2) / (n - ddof)
    var_num = S2 - n * (mu * mu)
    var = var_num / (n - ddof)
    if var < 0.0:  # numerical guard
        var = 0.0

    if mu <= 0.0:
        raise ZeroDivisionError("Has mean of zero for FANO FACTOR")

    return var / mu