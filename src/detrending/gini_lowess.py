import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

def lowess_twopass_detrending(gini: np.ndarray,
                              log2max: np.ndarray,
                              outlier:float = 0.75,
                              span: float=0.9):
    assert len(gini) == len(log2max)
    x = log2max
    y = gini

    f1 = lowess(endog=y, exog=x, frac=span, it=0, return_sorted=False)
    r1 = y - f1

    # 2) inlier mask via upper-quantile of positive residuals
    pos = r1[r1 > 0]
    thresh = np.quantile(pos, outlier) if pos.size else np.inf
    inlier = r1 < thresh
    outlier = ~inlier

    # 3) second fit on inliers
    x2, y2 = x[inlier], y[inlier]
    f2_in = lowess(endog=y2, exog=x2, frac=span, it=0, return_sorted=False)

    # 4) predict for outliers by linear interp / edge extrapolation
    order = np.argsort(x2, kind="mergesort")
    xs, fs = x2[order], f2_in[order]
    i = np.searchsorted(xs, x[outlier], side="left")

    # exact hits
    inb = (i < xs.size)  # in-bounds for indexing
    hit = np.zeros_like(i, dtype=bool)
    hit[inb] = (xs[i[inb]] == x[outlier][inb])
    yhat_out = np.empty(i.size, dtype=float)
    if hit.any():
        yhat_out[hit] = fs[i[hit]]

    # interpolate: 0<i<n
    mid = (~hit) & (i > 0) & (i < xs.size)
    if mid.any():
        i0, i1 = i[mid] - 1, i[mid]
        x0, x1 = xs[i0], xs[i1]
        y0, y1 = fs[i0], fs[i1]
        yhat_out[mid] = y0 + (y1 - y0) * (x[outlier][mid] - x0) / (x1 - x0)

    # left extrapolation: i==0
    left = (~hit) & (i == 0)
    if left.any():
        x0, x1 = xs[0], xs[1]
        y0, y1 = fs[0], fs[1]
        yhat_out[left] = y1 - (y1 - y0) * (x1 - x[outlier][left]) / (x1 - x0)

    # right extrapolation: i>=n
    right = (~hit) & (i >= xs.size)
    if right.any():
        x0, x1 = xs[-2], xs[-1]
        y0, y1 = fs[-2], fs[-1]
        yhat_out[right] = y0 + (y1 - y0) * (x[outlier][right] - x0) / (x1 - x0)

    # 5) residual2 assembly
    r2 = np.zeros_like(y, dtype=float)
    r2[inlier] = y[inlier] - f2_in
    r2[outlier] = y[outlier] - yhat_out
    return r2

