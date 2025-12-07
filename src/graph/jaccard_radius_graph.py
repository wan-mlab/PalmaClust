import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
from numpy.lib.format import open_memmap

def jaccard_radius_graph_blockwise(B: sp.csr_matrix, eps: float, block: int = 2048) -> sp.csr_matrix:
    """
    Exact ε-radius graph for binary CSR (cells x features), built blockwise.
    Returns CSR with edge weights = Jaccard distance for pairs with dist <= eps.
    """
    B = B.tocsr(copy=False); B.sort_indices()
    n = B.shape[0]
    r = np.diff(B.indptr).astype(np.int64)
    zero_rows = (r == 0)

    rows_all, cols_all, dist_all = [], [], []
    BT = B.T.tocsc(copy=False)

    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        Bi = B[i0:i1, :]
        C = Bi.dot(BT).tocsr(copy=False)  # intersections for this block
        C.setdiag(0); C.eliminate_zeros()
        indptr, indices, data = C.indptr, C.indices, C.data
        r_block = r[i0:i1]

        for ii in range(i1 - i0):
            beg, end = indptr[ii], indptr[ii + 1]
            if beg == end:
                continue
            js = indices[beg:end]
            inter = data[beg:end].astype(np.int64, copy=False)
            unions = r_block[ii] + r[js] - inter
            d = 1.0 - (inter / unions)
            keep = d <= eps
            if np.any(keep):
                rows_all.append(np.full(keep.sum(), i0 + ii, dtype=np.int64))
                cols_all.append(js[keep])
                dist_all.append(d[keep])


    if rows_all:
        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        dist = np.concatenate(dist_all)
    else:
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)
        dist = np.array([], dtype=float)

    G = sp.csr_matrix((dist, (rows, cols)), shape=(n, n))
    return G
