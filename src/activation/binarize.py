import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings

def _rawcounts_cutoff_rows(X_rows_csr: sp.csr_matrix, gamma: float) -> int:
    """
    Exact per-gene threshold selection on *just the provided rows* (genes x cells, CSR).
    Returns one integer cutoff using your Gamma heuristic.
    """
    n_cells = X_rows_csr.shape[1]
    indptr, data = X_rows_csr.indptr, X_rows_csr.data

    bc_low, bc_high = [], []
    for r in range(X_rows_csr.shape[0]):
        s, e = indptr[r], indptr[r + 1]
        v = data[s:e].astype(np.int64, copy=False)
        m = e - s
        z = n_cells - m  # zeros

        if m > 0:
            u, freq_pos = np.unique(v, return_counts=True)  # ascending
            c = np.concatenate([np.array([0], dtype=np.int64), u])
            f = np.concatenate([np.array([z], dtype=np.int64), freq_pos])
        else:
            c = np.array([0], dtype=np.int64)
            f = np.array([z], dtype=np.int64)

        denom = int(v.sum())
        if denom == 0:
            csum = np.zeros_like(c, dtype=float)
            warnings.warn("Gene used for cutoff has all zero counts.")
        else:
            contrib = (c * f).astype(np.int64, copy=False)
            tail_ge = np.cumsum(contrib[::-1], dtype=np.int64)[::-1]
            csum = tail_ge / float(denom)

        order_desc = np.argsort(c)[::-1]
        c_desc = c[order_desc]
        csum_desc = csum[order_desc]

        hits = np.where(csum_desc > gamma)[0]
        n_idx = int(hits[0]) if hits.size else 0
        n_idx = max(2, n_idx)
        if n_idx >= len(c_desc) - 1:
            n_idx = max(0, len(c_desc) - 2)

        hi = float(c_desc[n_idx]) if len(c_desc) else 0.0
        lo = float(c_desc[n_idx + 1]) if (len(c_desc) >= 2) else hi

        bc_high.append(hi)
        bc_low.append(lo)

    bc_med = 0.5 * (np.asarray(bc_high) + np.asarray(bc_low))
    top_n_gene = max(int(len(bc_med) * 0.10), 10)
    cutoff = int(np.floor(np.mean(bc_med[:top_n_gene])))
    return cutoff

def jaccard_binary(
    X_sel,
    gamma
):
    """
    Memory-efficient:
      - Never binarizes all genes. Only slices the qualified genes first.
      - Computes cutoff on the selected rows only, then binarizes just those rows.
      - Returns:
          B: (cells x features) CSR binary
          obj: dense distance (np.ndarray or memmap) or sparse ε-graph (CSR)
          meta: {'cutoff', 'feature_names', 'zero_cells'}
    """


    # --- cutoff on selected rows only ---
    cutoff = _rawcounts_cutoff_rows(X_sel, gamma=gamma)

    # --- binarize ONLY the selected rows (>= cutoff) ---
    # safe to modify in place: X_sel is a slice (new object)
    X_sel.data = (X_sel.data >= cutoff).astype(np.int8, copy=False)
    X_sel.eliminate_zeros()

    # --- build B = (cells x features) ---
    B = X_sel.T.tocsr(copy=False)             # cells x selected-features
    zero_cells = (B.getnnz(axis=1) == 0)

    return B, zero_cells