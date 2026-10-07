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

def _cutoff_rows_generic(X_rows_csr: sp.csr_matrix, gamma: float, per_gene: bool = False):
    """
    Float-safe version of the same Gamma-heuristic cutoff, without the integer casts
    _rawcounts_cutoff_rows relies on. Needed once counts stop being exact integers
    (cp10k-normalized data), and used for the per-gene cutoff (one cutoff per row
    instead of one global cutoff for every selected gene).

    per_gene=False mirrors _rawcounts_cutoff_rows's single collapsed cutoff (as a float);
    per_gene=True returns one cutoff per row.
    """
    n_cells = X_rows_csr.shape[1]
    indptr, data = X_rows_csr.indptr, X_rows_csr.data

    bc_low, bc_high = [], []
    for r in range(X_rows_csr.shape[0]):
        s, e = indptr[r], indptr[r + 1]
        v = data[s:e].astype(np.float64, copy=False)
        m = e - s
        z = n_cells - m  # zeros

        if m > 0:
            u, freq_pos = np.unique(v, return_counts=True)  # ascending
            c = np.concatenate([np.array([0.0]), u])
            f = np.concatenate([np.array([z], dtype=np.int64), freq_pos])
        else:
            c = np.array([0.0])
            f = np.array([z], dtype=np.int64)

        denom = float(v.sum())
        if denom == 0:
            csum = np.zeros_like(c, dtype=float)
            warnings.warn("Gene used for cutoff has all zero counts.")
        else:
            contrib = c * f
            tail_ge = np.cumsum(contrib[::-1])[::-1]
            csum = tail_ge / denom

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
    if per_gene:
        return bc_med
    top_n_gene = max(int(len(bc_med) * 0.10), 10)
    return float(np.mean(bc_med[:top_n_gene]))

def jaccard_binary(
    X_sel,
    gamma,
    cutoff_mode: str = "global",   # "global" | "per_gene"
    as_int: bool = True             # False once counts are no longer exact integers (cp10k)
):
    """
    Memory-efficient:
      - Never binarizes all genes. Only slices the qualified genes first.
      - Computes cutoff on the selected rows only, then binarizes just those rows.
      - Returns:
          B: (cells x features) CSR binary
          obj: dense distance (np.ndarray or memmap) or sparse ε-graph (CSR)
          meta: {'cutoff', 'feature_names', 'zero_cells'}

    cutoff_mode="global" (default) collapses every selected gene's cutoff into one
    number, exactly as before. cutoff_mode="per_gene" uses each gene's own cutoff,
    so a single deeply-sequenced gene can't push every other gene's threshold out of
    reach (the collapsed cutoff is what made most cells lose all their active features
    on deep droplet datasets -- see the "graph degenerated" warnings in the benchmark).
    """
    if as_int and cutoff_mode == "global":
        # exact original path -- bit-identical to the pre-existing behaviour
        cutoff = _rawcounts_cutoff_rows(X_sel, gamma=gamma)
        X_sel.data = (X_sel.data >= cutoff).astype(np.int8, copy=False)
    elif cutoff_mode == "per_gene":
        cutoffs = _cutoff_rows_generic(X_sel, gamma=gamma, per_gene=True)
        indptr, data = X_sel.indptr, X_sel.data
        for r in range(X_sel.shape[0]):
            s, e = indptr[r], indptr[r + 1]
            data[s:e] = (data[s:e] >= cutoffs[r]).astype(np.int8, copy=False)
        X_sel.data = X_sel.data.astype(np.int8, copy=False)
    else:
        # global cutoff, but data is no longer exact integers (cp10k)
        cutoff = _cutoff_rows_generic(X_sel, gamma=gamma, per_gene=False)
        X_sel.data = (X_sel.data >= cutoff).astype(np.int8, copy=False)
    X_sel.eliminate_zeros()

    # --- build B = (cells x features) ---
    B = X_sel.T.tocsr(copy=False)             # cells x selected-features
    zero_cells = (B.getnnz(axis=1) == 0)

    return B, zero_cells