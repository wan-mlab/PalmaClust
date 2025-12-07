import numpy as np


def _nth_order_stat(a: np.ndarray, k: int) -> float:
    aa = np.asarray(a, dtype=np.float64).copy()
    return float(np.partition(aa, k)[k])

def _quantile_linear_with_zeros(nonzero: np.ndarray, zeros_count: int, q: float) -> float:
    """
    Exact q-quantile (NumPy 'linear' method) of the multiset {0×zeros_count} ∪ nonzero.
    """
    m = int(nonzero.size)
    z = int(zeros_count)
    n = z + m
    if n == 0: return np.nan
    r = q * (n - 1)
    a = int(np.floor(r)); b = int(np.ceil(r)); gamma = r - a
    xa = 0.0 if a < z else _nth_order_stat(nonzero, a - z)
    xb = 0.0 if b < z else _nth_order_stat(nonzero, b - z)
    return (1.0 - gamma) * xa + gamma * xb

def palma_ratio_from_sparse_nonzeros(
    nonzero: np.ndarray,
    n_cells: int,
    upper: float,
    lower: float,
    alpha: float,
    winsor: float = 0.0,
) -> float:
    """
    Exact Palma = (upper_share + alpha) / (lower_share + alpha),
    with exact winsorization and exact top/bottom tails, treating zeros analytically.
    Matches the dense function you posted (including tail size rules).
    """
    v = np.asarray(nonzero, dtype=np.float64)
    n = int(n_cells); m = v.size; z = n - m
    if n <= 0 or z < 0:
        return np.nan

    # No winsorization → direct tails
    if not (0.0 < winsor < 0.5):
        k_lower = max(1, int(np.floor(lower * n)))
        k_upper = max(1, int(np.ceil(upper * n)))

        # lower tail: zeros first, then smallest positives
        if k_lower <= z:
            lower_sum = 0.0
        else:
            s = k_lower - z
            lower_sum = float(np.partition(v.copy(), s - 1)[:s].sum())

        # upper tail: largest positives
        if k_upper <= m:
            idx = m - k_upper
            upper_sum = float(np.partition(v.copy(), idx)[idx:].sum())
        else:
            upper_sum = float(v.sum())

        total = float(v.sum())
        if not (total > 0):  # all zeros → denominator zero in your original; treat as +inf
            return np.inf
        lower_share = lower_sum / total
        upper_share = upper_sum / total
        return (upper_share + alpha) / (lower_share + alpha)

    # Winsorization path (exact)
    lo = _quantile_linear_with_zeros(v, z, winsor)
    hi = _quantile_linear_with_zeros(v, z, 1.0 - winsor)
    if lo > hi: lo, hi = hi, lo

    le_lo = v <= lo
    ge_hi = v >= hi
    mid_mask = (~le_lo) & (~ge_hi)
    pos_le_lo = int(le_lo.sum())
    pos_ge_hi = int(ge_hi.sum())
    mid_vals = v[mid_mask]
    m_mid = int(mid_vals.size)

    c_le_lo = z + pos_le_lo
    c_ge_hi = pos_ge_hi
    sum_mid_all = float(mid_vals.sum())

    total = float(c_le_lo * lo + sum_mid_all + c_ge_hi * hi)
    if not (total > 0):
        return np.inf

    k_lower = max(1, int(np.floor(lower * n)))
    k_upper = max(1, int(np.ceil(upper * n)))

    # bottom tail on winsorized values
    if k_lower <= c_le_lo:
        lower_sum = k_lower * lo
    elif k_lower <= c_le_lo + m_mid:
        s = k_lower - c_le_lo
        lower_sum = c_le_lo * lo + float(np.partition(mid_vals.copy(), s - 1)[:s].sum())
    else:
        s_hi = k_lower - c_le_lo - m_mid
        lower_sum = c_le_lo * lo + sum_mid_all + s_hi * hi

    # top tail on winsorized values
    if k_upper <= c_ge_hi:
        upper_sum = k_upper * hi
    elif k_upper <= c_ge_hi + m_mid:
        s = k_upper - c_ge_hi
        idx = m_mid - s
        upper_sum = c_ge_hi * hi + float(np.partition(mid_vals.copy(), idx)[idx:].sum())
    else:
        s_lo = k_upper - c_ge_hi - m_mid
        upper_sum = c_ge_hi * hi + sum_mid_all + s_lo * lo

    lower_share = lower_sum / total
    upper_share = upper_sum / total
    return (upper_share + alpha) / (lower_share + alpha)