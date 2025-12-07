import numpy as np
import scipy.sparse as sp
from typing import Iterable, Tuple, List

def _as_csr(G):
    if sp.isspmatrix_csr(G):
        return G
    return G.tocsr(copy=False)

def _prune_topk_csr(W: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """
    Keep top-k weights per row (ties broken arbitrarily but deterministically).
    """
    if k is None:
        return W
    k = int(k)
    if k <= 0:
        return sp.csr_matrix(W.shape, dtype=W.dtype)

    W = _as_csr(W)
    W.sort_indices()
    indptr, indices, data = W.indptr, W.indices, W.data
    n_rows = W.shape[0]

    # First pass: count how many entries we keep per row
    keep_counts = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        s, e = indptr[i], indptr[i+1]
        cnt = e - s
        keep_counts[i] = cnt if cnt <= k else k

    new_indptr = np.empty(n_rows + 1, dtype=np.int64)
    new_indptr[0] = 0
    np.cumsum(keep_counts, out=new_indptr[1:])
    nnz_new = int(new_indptr[-1])

    new_indices = np.empty(nnz_new, dtype=indices.dtype)
    new_data = np.empty(nnz_new, dtype=data.dtype)

    # Second pass: select top-k per row
    out_pos = 0
    for i in range(n_rows):
        s, e = indptr[i], indptr[i+1]
        cnt = e - s
        if cnt == 0:
            continue
        if cnt <= k:
            new_indices[out_pos:out_pos+cnt] = indices[s:e]
            new_data[out_pos:out_pos+cnt] = data[s:e]
            out_pos += cnt
        else:
            # pick k largest weights (descending)
            row_w = data[s:e]
            row_j = indices[s:e]
            sel = np.argpartition(row_w, -k)[-k:]
            # sort the selected by weight descending for stability
            ord_sel = sel[np.argsort(row_w[sel])[::-1]]
            t = ord_sel.size
            new_indices[out_pos:out_pos+t] = row_j[ord_sel]
            new_data[out_pos:out_pos+t] = row_w[ord_sel]
            out_pos += t

    Wk = sp.csr_matrix((new_data, new_indices, new_indptr), shape=W.shape)
    Wk.eliminate_zeros()
    return Wk

def mix_knn_graphs(
    G_list: Iterable[sp.spmatrix],
    weights: Iterable[float],
    is_distance: bool = True,       # True if inputs store distances (e.g., Jaccard distance in [0,1])
    per_graph_normalize: str = "none",  # 'none' | 'row_l1' | 'max'
    symmetrize: str = "none",         # 'none' | 'max' | 'mean' | 'min' | 'sum'
    drop_self_loops: bool = True,
    clip_range: Tuple[float,float] | None = None,   # clip mixed weights to a range
    prune_topk: int | None = None,   # keep top-k neighbors per row after mixing (optional)
    dtype = np.float64
) -> sp.csr_matrix:
    """
    Mix multiple (kNN) graphs with given weights into a single sparse graph.

    Parameters
    ----------
    G_list : iterable of sparse matrices (n x n)
        Each is a kNN graph (similarity or distance). Shapes must match.
        They can be directed; we will symmetrize as requested.
    weights : iterable of floats
        Non-negative weights; they need not sum to 1 (absolute scale only affects magnitudes).
    is_distance : bool
        If True, convert each G_i to similarity via (1 - d) before mixing (and clip to [0,1]).
    per_graph_normalize : {'none','row_l1','max'}
        Optional pre-normalization for comparability across graphs:
          - 'none'   : no normalization
          - 'row_l1' : make each row sum to 1 (row-stochastic) when possible
          - 'max'    : divide by per-graph max edge weight
    symmetrize : {'none','max','mean','min','sum'}
        How to make the final graph undirected:
          - 'none' : leave as is
          - 'max'  : W = max(W, W.T)
          - 'mean' : W = 0.5*(W + W.T)
          - 'min'  : W = min(W, W.T)
          - 'sum'  : W = W + W.T
    drop_self_loops : bool
        If True, zero the diagonal.
    clip_range : (lo, hi) or None
        Clip weights after mixing (e.g., to [0,1]). Set None to skip.
    prune_topk : int or None
        If set, keep only the top-k edges per row after mixing + symmetrization.
        Useful to maintain sparsity comparable to an input kNN.
    dtype : numpy dtype
        Output data type.

    Returns
    -------
    W : csr_matrix (n x n)
        Mixed, (optionally) symmetrized graph of similarities (not distances).
    """
    G_list = list(G_list)
    weights = np.asarray(list(weights), dtype=np.float64)

    if len(G_list) == 0:
        raise ValueError("G_list is empty.")
    if weights.size != len(G_list):
        raise ValueError("weights length must match G_list length.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")

    n = _as_csr(G_list[0]).shape[0]
    shape = _as_csr(G_list[0]).shape
    for i, Gi in enumerate(G_list):
        if _as_csr(Gi).shape != shape:
            raise ValueError(f"All graphs must have the same shape; G[{i}] has {Gi.shape}, expected {shape}.")

    # Build a weighted union via COO concatenation (single pass over nnz)
    rows_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    data_all: List[np.ndarray] = []

    for w, Gi in zip(weights, G_list):
        if w == 0.0:
            continue
        Gc = _as_csr(Gi).copy()  # shallow copy header, data view is fine (we alter .data)
        Gc.eliminate_zeros()
        Gc = Gc.tocoo(copy=False)

        # To similarity if needed
        if is_distance:
            di = Gc.data.astype(np.float64, copy=False)
            si = 1.0 - di
            if clip_range is not None:
                lo, hi = clip_range
                np.clip(si, lo, hi, out=si)
            Gc.data = si

        # Per-graph normalization to put graphs on comparable scales
        if per_graph_normalize == "row_l1":
            # row normalize the COO: we need row sums
            # Convert to CSR to get row sums cheaply
            tmp = sp.csr_matrix((Gc.data, (Gc.row, Gc.col)), shape=shape)
            rs = np.asarray(tmp.sum(axis=1)).ravel()
            # Guard divide-by-zero
            nz = rs[tmp.row] > 0
            if nz.any():
                Gc.data[nz] = Gc.data[nz] / rs[tmp.row[nz]]
            # else leave weights as-is for zero-sum rows
        elif per_graph_normalize == "max":
            m = float(Gc.data.max()) if Gc.data.size else 0.0
            if m > 0:
                Gc.data = Gc.data / m
        elif per_graph_normalize == "none":
            pass
        else:
            raise ValueError("per_graph_normalize must be 'none', 'row_l1', or 'max'.")

        if w != 1.0:
            Gc.data = (Gc.data.astype(np.float64, copy=False) * w)

        rows_all.append(Gc.row.astype(np.int64, copy=False))
        cols_all.append(Gc.col.astype(np.int64, copy=False))
        data_all.append(Gc.data.astype(dtype, copy=False))

    if not data_all:
        # everything was zero-weight
        return sp.csr_matrix(shape, dtype=dtype)

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    dat  = np.concatenate(data_all)

    # Combine duplicates by summing (COO -> CSR merges duplicates)
    W = sp.csr_matrix((dat, (rows, cols)), shape=shape, dtype=dtype)
    if drop_self_loops:
        W.setdiag(0.0)
    W.eliminate_zeros()

    # Symmetrize
    if symmetrize != "none":
        WT = W.T.tocsr(copy=False)
        if symmetrize == "max":
            W = W.maximum(WT)
        elif symmetrize == "mean":
            W = (W + WT) * 0.5
        elif symmetrize == "min":
            W = W.minimum(WT)
        elif symmetrize == "sum":
            W = W + WT
        else:
            raise ValueError("symmetrize must be 'none','max','mean','min','sum'")
        if drop_self_loops:
            W.setdiag(0.0)
        W.eliminate_zeros()

    # Optional clipping
    if clip_range is not None:
        lo, hi = clip_range
        if (lo is not None) or (hi is not None):
            if lo is None: lo = -np.inf
            if hi is None: hi =  np.inf
            if W.nnz:
                np.clip(W.data, lo, hi, out=W.data)
                W.eliminate_zeros()

    # Optional top-k pruning per row to keep kNN‑like sparsity
    if prune_topk is not None:
        W = _prune_topk_csr(W, int(prune_topk))

    W.sort_indices()
    return W