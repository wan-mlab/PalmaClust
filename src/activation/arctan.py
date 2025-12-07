import numpy as np
import pandas as pd
import scipy.sparse as sp

# --- helper: duplicate-safe mapping (uses first occurrence if genes_filtered has duplicates) ---
def _map_genes_to_rows(gf, gq):
    gf = pd.Index(gf) if not isinstance(gf, pd.Index) else gf
    gq = pd.Index(gq) if not isinstance(gq, pd.Index) else gq
    if gf.is_unique:
        idx = gf.get_indexer(gq)
        keep = idx >= 0
        return idx[keep].astype(np.int64), gq[keep]
    # duplicate-safe path: take first occurrence
    pos = pd.Series(np.arange(len(gf), dtype=np.int64), index=gf)
    first = pos.groupby(level=0).nth(0)
    idx_s = first.reindex(gq)
    keep = idx_s.notna().to_numpy()
    return idx_s[keep].astype(np.int64).to_numpy(), gq[keep]

# --- helper: compute expM for one gene from its nonzero counts (row), with the exact control flow of your code ---
def _expM_from_row_nonzeros(v_nonzero_sorted_desc: np.ndarray, n_cells: int) -> float:
    """
    v_nonzero_sorted_desc: 1D float array of this gene's nonzero counts, sorted descending.
    n_cells: total #cells (including zeros).
    Returns expM exactly per your algorithm.
    """
    S = float(v_nonzero_sorted_desc.sum())
    if S <= 0.0:
        return 0.0  # all zeros -> expM=0

    m = v_nonzero_sorted_desc.size
    csum = np.cumsum(v_nonzero_sorted_desc)          # length m
    th = 0.8 * S
    binCellNum = int(n_cells // 1000)

    if binCellNum <= 9:
        # fine search: first j with csum[j-1] > 0.8 * S
        j_star = int(np.searchsorted(csum, th, side="right")) + 1  # 1-based
        j_end = j_star
    else:
        # coarse steps of binCellNum: end = bin, 2*bin, ...
        loopNum = int((n_cells - binCellNum) // binCellNum)
        if loopNum <= 0:
            j_end = int(np.searchsorted(csum, th, side="right")) + 1
        else:
            ends = (np.arange(loopNum, dtype=np.int64) + 1) * binCellNum   # array of step endpoints
            # clamp to m when indexing csum; sums beyond m equal S
            clamp = np.minimum(ends, m)
            sums_at_end = csum[clamp - 1]
            hit = sums_at_end / S > 0.8
            if np.any(hit):
                j_end = int(ends[int(np.flatnonzero(hit)[0])])
            else:
                # extremely unlikely (would mean 0.8 never reached by ends); fall back to last end
                j_end = int(ends[-1])

    # expM = mean of the "top j_end" entries of the *full* sorted vector (positives + zeros)
    # if j_end <= m: mean over j_end positives; if j_end > m: S / j_end (the rest are zeros)
    if j_end <= m:
        expM = float(csum[j_end - 1] / j_end)
    else:
        expM = float(S / j_end)
    return expM

def arctan_transform(  # name kept as requested; now returns arctan-transformed features (continuous, not binary)
    X_sel
):
    """
    Arctan feature transform (sparse, memory-lean), per your arctanTransform():
      - Slice to qualified genes first.
      - For each selected gene, compute expM = mean of the minimal top counts
        whose cumulative sum > 80% of this gene's total expression mass
        (stepping by binCellNum when n_cells >= 10k).
      - Apply:  feat = 10 * [ arctan(x - expM) + arctan(expM) ].
        (Zeros stay zeros, so sparsity is preserved.)
      - Return B (cells x features) CSR float32 and zero_cells mask.
    """

    X_sel.sort_indices()

    n_feat, n_cells = X_sel.shape
    indptr, data = X_sel.indptr, X_sel.data

    # per-gene arctan transform in place
    for r in range(n_feat):
        s, e = indptr[r], indptr[r + 1]
        if s == e:
            # all zeros for this gene -> remains zero; expM=0 => transform of zeros is 0
            continue
        v = data[s:e].astype(np.float64, copy=False)   # strictly positive counts
        v_sorted_desc = np.sort(v)[::-1]
        expM = _expM_from_row_nonzeros(v_sorted_desc, n_cells)
        # apply arctan: 10*(atan(x - expM) + atan(expM))
        data[s:e] = 10.0 * (np.arctan(v - expM) + np.arctan(expM))

    X_sel.eliminate_zeros()

    # cells x features
    B = X_sel.T.tocsr(copy=False)
    # store as float32 to save memory (optional; comment out if you want float64)
    B.data = B.data.astype(np.float32, copy=False)

    zero_cells = (B.getnnz(axis=1) == 0)
    return B, zero_cells