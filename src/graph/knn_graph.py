import numpy as np
import scipy.sparse as sp

def knn_graph_from_binary_B(
    B_features_x_cells: sp.csr_matrix,
    k: int = 30,
    mode: str = "union",               # 'union' (kNN ∪ kNN^T) or 'mutual' (kNN ∩ kNN^T)
    sym_weight: str = "max",           # how to combine weights on symmetrization: 'max' | 'mean' | 'min'
    block_size: int = 4096,            # cells processed per block (for Jaccard; ignored by Euclidean)
    ensure_binary: bool = True,        # only used for Jaccard; ignored by Euclidean
    return_distance: bool = False,     # False -> similarity (for Leiden); True -> distance
    metric: str = "jaccard",           # 'jaccard' | 'euclidean'
    euclidean_similarity: str = "reciprocal",  # when metric='euclidean' and return_distance=False: 'reciprocal' | 'gaussian'
    euclidean_sigma: float | None = None,      # sigma for gaussian similarity; if None, auto (median k-NN distance)
    n_jobs: int | None = None          # passed to sklearn NearestNeighbors when metric='euclidean'
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build a sparse kNN graph from a feature matrix (features x cells).

    Parameters
    ----------
    B_features_x_cells : csr_matrix
        Shape (n_features, n_cells). For 'jaccard', nonzeros indicate presence (will be binarized if ensure_binary=True).
        For 'euclidean', values are arbitrary nonnegative/real features.
    k : int
        Number of neighbors per cell (directed); after symmetrization you get an undirected graph.
    mode : {'union','mutual'}
        'union' -> edges that appear in either direction; 'mutual' -> only edges present in both directions.
    sym_weight : {'max','mean','min'}
        How to combine weights when symmetrizing (applied in the *similarity* domain).
    block_size : int
        Batch size for Jaccard blocked multiplication.
    ensure_binary : bool
        If True (default), Jaccard treats any nonzero as 1 (without densifying).
    return_distance : bool
        If False (default), return a similarity-weighted graph (Leiden expects similarities).
        If True, return a distance-weighted graph.
        - Jaccard: distance = 1 - similarity.
        - Euclidean: distance = actual Euclidean distance (L2).
    metric : {'jaccard','euclidean'}
        Distance metric to drive neighbor selection / weighting.
    euclidean_similarity : {'reciprocal','gaussian'}
        Similarity transform used when metric='euclidean' and return_distance=False.
    euclidean_sigma : float or None
        Sigma for the gaussian similarity. If None, uses the median k-NN distance across cells.
    n_jobs : int or None
        Passed to sklearn.neighbors.NearestNeighbors for Euclidean kNN.

    Returns
    -------
    W : csr_matrix
        (n_cells x n_cells) symmetric kNN graph. Data are similarities by default, or distances if return_distance=True.
    zero_cells : np.ndarray (bool)
        Mask of cells with zero active features (for Jaccard) or zero-norm (for Euclidean).
    """
    if not sp.isspmatrix_csr(B_features_x_cells):
        B_features_x_cells = B_features_x_cells.tocsr(copy=False)
    B_features_x_cells.eliminate_zeros()
    metric = metric.lower()

    # Orient as cells x features for neighbor computations
    X = B_features_x_cells.T.tocsr(copy=False)  # (n_cells x n_features)
    X.sort_indices()
    n_cells = X.shape[0]

    # Track zero-cells (all-zero rows)
    nnz_per_cell = np.diff(X.indptr).astype(np.int64)
    zero_cells = (nnz_per_cell == 0)

    # ---------- Branch 1: Jaccard (set-overlap on binary detection) ----------
    if metric == "jaccard":
        B = X.copy()
        if ensure_binary and B.data.size:
            B.data[:] = 1  # in-place binarization of data without densifying
        # Precompute transpose for intersections
        BT = B.T.tocsc(copy=False)

        rows_all, cols_all, w_all = [], [], []

        for i0 in range(0, n_cells, block_size):
            i1 = min(n_cells, i0 + block_size)
            Bi = B[i0:i1, :]                  # (b x f)
            # Intersections (b x n_cells); sparse
            C = Bi.dot(BT).tocsr(copy=False)
            C.eliminate_zeros()
            indptr, indices, data = C.indptr, C.indices, C.data
            r_block = nnz_per_cell[i0:i1]

            for off in range(i1 - i0):
                i = i0 + off
                s, e = indptr[off], indptr[off + 1]
                if s == e:
                    continue
                js = indices[s:e]
                inter = data[s:e].astype(np.int64, copy=False)

                # exclude self
                mask = (js != i)
                if not np.any(mask):
                    continue
                js = js[mask]
                inter = inter[mask]

                # Jaccard similarity: |A∩B| / (|A| + |B| - |A∩B|)
                denom = r_block[off] + nnz_per_cell[js] - inter
                valid = denom > 0
                if not np.any(valid):
                    continue
                js = js[valid]
                inter = inter[valid]
                denom = denom[valid]
                sim = inter / denom.astype(np.float64, copy=False)

                # Top-k by similarity
                k_eff = min(k, sim.size)
                if k_eff <= 0:
                    continue
                top_idx = np.argpartition(sim, -k_eff)[-k_eff:]
                order = top_idx[np.argsort(sim[top_idx])[::-1]]

                rows_all.append(np.full(order.size, i, dtype=np.int64))
                cols_all.append(js[order])
                w_all.append(sim[order].astype(np.float64, copy=False))

        if rows_all:
            rows_dir = np.concatenate(rows_all)
            cols_dir = np.concatenate(cols_all)
            w_dir = np.concatenate(w_all)
        else:
            rows_dir = np.array([], dtype=np.int64)
            cols_dir = np.array([], dtype=np.int64)
            w_dir = np.array([], dtype=np.float64)

        A = sp.csr_matrix((w_dir, (rows_dir, cols_dir)), shape=(n_cells, n_cells))

        # ---------- Branch 2: Euclidean (exact kNN via sklearn) ----------
    elif metric in ("euclidean", "l2"):
        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception as e:
            raise ImportError(
                "metric='euclidean' requires scikit-learn. Install with `pip install scikit-learn`."
            ) from e

        # n_neighbors includes self so we can drop it after querying
        n_neighbors = min(k + 1, max(n_cells, 1))
        # If n_cells == 0 or 1, handle trivially
        if n_cells <= 1 or n_neighbors <= 1:
            W = sp.csr_matrix((n_cells, n_cells), dtype=np.float64)
            return W, zero_cells

        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="euclidean",
            algorithm="auto",
            n_jobs=n_jobs
        )
        nn.fit(X)  # CSR is supported
        dists, nbrs = nn.kneighbors(return_distance=True)

        # Optional sigma for gaussian similarity
        if not return_distance and euclidean_similarity == "gaussian":
            # Use the k-th neighbor distance (excluding self) across all cells
            kth_idx = min(k, dists.shape[1] - 1)
            # If we included self, the k-th neighbor is at index kth_idx
            kth_d = dists[:, kth_idx]
            sigma = float(np.median(kth_d[kth_d > 0])) if euclidean_sigma is None else float(euclidean_sigma)
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0  # safe fallback
        else:
            sigma = None

        rows_all, cols_all, w_all = [], [], []

        for i in range(n_cells):
            idx = nbrs[i]
            di = dists[i].astype(np.float64, copy=False)

            # Remove self if present at position 0
            if idx.size and idx[0] == i:
                idx = idx[1:]
                di = di[1:]

            if idx.size == 0:
                continue
            if idx.size > k:
                idx = idx[:k]
                di = di[:k]

            if return_distance:
                w = di  # distances as weights
            else:
                if euclidean_similarity == "reciprocal":
                    w = 1.0 / (1.0 + di)
                elif euclidean_similarity == "gaussian":
                    # sigma already set above
                    w = np.exp(-(di * di) / (2.0 * sigma * sigma))
                else:
                    raise ValueError("euclidean_similarity must be 'reciprocal' or 'gaussian'")

            rows_all.append(np.full(idx.size, i, dtype=np.int64))
            cols_all.append(idx.astype(np.int64))
            w_all.append(w)

        if rows_all:
            rows_dir = np.concatenate(rows_all)
            cols_dir = np.concatenate(cols_all)
            w_dir = np.concatenate(w_all)
        else:
            rows_dir = np.array([], dtype=np.int64)
            cols_dir = np.array([], dtype=np.int64)
            w_dir = np.array([], dtype=np.float64)

        A = sp.csr_matrix((w_dir, (rows_dir, cols_dir)), shape=(n_cells, n_cells))

    else:
        raise ValueError("metric must be 'jaccard' or 'euclidean'")

    A.eliminate_zeros()
    AT = A.T.tocsr(copy=False)

    # ---- Symmetrize (operate in whatever domain A is currently in: similarity if return_distance=False, else distance) ----
    # We standardize by always symmetrizing in the *similarity* domain:
    #   - For Jaccard: A holds similarity now.
    #   - For Euclidean:
    #       * if return_distance=False, A holds similarity (reciprocal/gaussian) -> OK
    #       * if return_distance=True, A holds distance; convert to similarity temporarily.
    if return_distance and metric in ("euclidean", "l2"):
        # Temporary similarity for combining: use reciprocal map to [0,1], monotonic in distance
        A_tmp = A.copy()
        A_tmp.data = 1.0 / (1.0 + A_tmp.data)
        AT_tmp = A_tmp.T.tocsr(copy=False)
        base_sim_A, base_sim_AT = A_tmp, AT_tmp
    else:
        base_sim_A, base_sim_AT = A, AT

    if mode == "union":
        if sym_weight == "max":
            W_sim = base_sim_A.maximum(base_sim_AT)
        elif sym_weight == "mean":
            W_sim = (base_sim_A + base_sim_AT) * 0.5
        elif sym_weight == "min":
            W_sim = base_sim_A.minimum(base_sim_AT)
        else:
            raise ValueError("sym_weight must be 'max', 'mean', or 'min'")
    elif mode == "mutual":
        both = base_sim_A.multiply(base_sim_AT.sign())  # keep edges present in both directions
        if sym_weight == "max":
            W_sim = both.maximum(both.T)
        elif sym_weight == "mean":
            W_sim = (both + both.T) * 0.5
        elif sym_weight == "min":
            W_sim = both.minimum(both.T)
        else:
            raise ValueError("sym_weight must be 'max', 'mean', or 'min'")
    else:
        raise ValueError("mode must be 'union' or 'mutual'")

    W_sim.setdiag(0.0)
    W_sim.eliminate_zeros()
    W_sim.sort_indices()

    # Convert back to the desired output domain
    if return_distance:
        if metric == "jaccard":
            W = W_sim.tocsr(copy=False)
            W.data = 1.0 - W.data  # Jaccard distance in [0,1]
            np.clip(W.data, 0.0, 1.0, out=W.data)
        else:
            # We symmetrized via a temporary similarity (reciprocal). Invert that map:
            # sim = 1/(1+d)  ->  d = 1/sim - 1
            W = W_sim.tocsr(copy=False)
            # guard against zeros from numerics
            eps = 1e-12
            W.data = (1.0 / np.maximum(W.data, eps)) - 1.0
    else:
        W = W_sim

    return W#, zero_cells