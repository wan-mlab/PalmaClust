import numpy as np

def idf_from_sparse_nonzeros(
    nonzero: np.ndarray,
    n_cells: int,
    smooth: bool = True,
    add_one: bool = True
) -> float:
    """
    Inverse Document Frequency (IDF) for one gene over ALL cells.

    Classic IR form: idf = log(n / df).
    Smoothed variant (default): idf = log((1+n)/(1+df)) + 1  (when add_one=True).

    Parameters
    ----------
    nonzero : 1D array-like
        Strictly positive counts of this gene (zeros omitted). Only its length matters.
    n_cells : int
        Total number of cells (documents).
    smooth : bool
        If True, apply smoothing to avoid inf when df=0.
    add_one : bool
        If True with smoothing, add +1 to the final idf (common scikit-like variant).
    log_base : float
        Base of the logarithm (default: e). For base 10, set log_base=10.0.

    Returns
    -------
    idf : float
        Inverse document frequency.
        If smooth=False and df=0, returns +inf.
    """
    m = int(np.asarray(nonzero).size)  # df = #cells with count>0
    n = int(n_cells)
    if n <= 0:
        return np.nan

    if smooth:
        # log((1+n)/(1+m)) [+ 1]
        val = np.log((1.0 + n) / (1.0 + m))
        if add_one:
            val += 1.0
        return float(val)
    else:
        # log(n/m) with edge cases
        if m == 0:
            return np.inf
        val = np.log(n / m)
        return float(val)