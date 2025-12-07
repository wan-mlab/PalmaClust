import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# ---------- utilities ----------

def _indexer_first(gf: pd.Index, gq) -> np.ndarray:
    """Map gq -> row positions in gf (duplicate-safe: keep first occurrence)."""
    gq = pd.Index(gq)
    if gf.is_unique:
        idx = gf.get_indexer(gq)
        return idx[idx >= 0].astype(np.int64)
    pos = pd.Series(np.arange(len(gf), dtype=np.int64), index=gf)
    first = pos.groupby(level=0).nth(0)
    idx_s = first.reindex(gq)
    return idx_s[idx_s.notna()].astype(np.int64).to_numpy()

def _zscore_cols(X: sp.csr_matrix) -> sp.csr_matrix:
    """Z-score each gene (row) across cells (columns) inside the cluster (zeros included)."""
    if not sp.isspmatrix_csr(X):
        X = X.tocsr(copy=False)
    X = X.astype(np.float32, copy=False)
    n_cells = X.shape[1]
    # mean per row (include zeros via sum / n)
    sums = np.asarray(X.sum(axis=1)).ravel()
    mu = sums / float(n_cells)
    # sum of squares of nonzeros, variance across all cells (ddof=0)
    Xsq = X.copy()
    Xsq.data **= 2
    sums2 = np.asarray(Xsq.sum(axis=1)).ravel()
    var = (sums2 / float(n_cells)) - (mu ** 2)
    var[var < 0] = 0
    std = np.sqrt(var, dtype=np.float32)
    # subtract mean & divide by std in-place for nonzero entries; zeros -> (0 - mu)/std (we ignore to keep sparsity)
    # To keep sparsity, we center only nonzeros, which is fine for rank/AUC/logFC and graph building.
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr))
    X.data -= mu[rows]
    nz = std[rows] > 0
    X.data[nz] /= std[rows[nz]]
    # prune explicit zeros after centering
    X.eliminate_zeros()
    return X

def _arctan_center_transform_rows(X_rows_csr: sp.csr_matrix) -> sp.csr_matrix:
    """
    Per-gene arctan centering/soft-clipping (zeros stay zeros).
    For each row g: expM = mean of top counts covering >80% mass (with binning >=10k cells),
    then x -> 10*(atan(x - expM) + atan(expM)).
    """
    if not sp.isspmatrix_csr(X_rows_csr):
        X_rows_csr = X_rows_csr.tocsr(copy=False)
    X_rows_csr.sort_indices()
    n_feat, n_cells = X_rows_csr.shape
    indptr, data = X_rows_csr.indptr, X_rows_csr.data

    for r in range(n_feat):
        s, e = indptr[r], indptr[r + 1]
        if s == e:
            continue
        v = data[s:e].astype(np.float64, copy=False)
        vs = np.sort(v)[::-1]
        S = float(vs.sum())
        if S <= 0.0:
            continue
        th = 0.8 * S
        binCellNum = int(n_cells // 1000)
        if binCellNum <= 9:
            cs = np.cumsum(vs)
            j_end = int(np.searchsorted(cs, th, side="right")) + 1
        else:
            loopNum = int((n_cells - binCellNum) // binCellNum)
            if loopNum <= 0:
                cs = np.cumsum(vs)
                j_end = int(np.searchsorted(cs, th, side="right")) + 1
            else:
                ends = (np.arange(loopNum, dtype=np.int64) + 1) * binCellNum
                clamp = np.minimum(ends, vs.size)
                sums_at_end = np.cumsum(vs)[clamp - 1]
                hit = sums_at_end > th
                if np.any(hit):
                    j_end = int(ends[int(np.flatnonzero(hit)[0])])
                else:
                    j_end = int(ends[-1])
        # mean over first j_end values (zeros if j_end > nnz)
        if j_end <= vs.size:
            expM = float(np.cumsum(vs)[j_end - 1] / j_end)
        else:
            expM = float(S / j_end)
        data[s:e] = 10.0 * (np.arctan(v - expM) + np.arctan(expM))
    X_rows_csr.eliminate_zeros()
    return X_rows_csr

def _pca_from_rows(X_rows: sp.csr_matrix, n_components: int = 25, random_state: int = 0) -> np.ndarray:
    """
    PCA on genes×cells sparse: compute SVD on cells×genes (transpose).
    Returns dense (cells × n_components), L2-normalized rows.
    """
    X_cg = X_rows.T  # cells × genes
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    Z = svd.fit_transform(X_cg)  # dense
    Z = normalize(Z, norm="l2", axis=1, copy=False)
    return Z

def _knn_graph_cosine(X: np.ndarray, k: int = 20, mode: str = "union") -> sp.csr_matrix:
    """
    Build cosine kNN graph from dense features X (cells × d).
    Returns symmetric CSR similarity graph with weights = 1 - cosine distance.
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), metric="cosine", algorithm="auto", n_jobs=-1)
    nbrs.fit(X)
    dist, idx = nbrs.kneighbors(X, return_distance=True)
    # drop self (first neighbor)
    dist = dist[:, 1:]; idx = idx[:, 1:]
    sim = 1.0 - dist
    rows = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    cols = idx.ravel()
    dat  = sim.ravel()
    A = sp.csr_matrix((dat, (rows, cols)), shape=(n, n))
    if mode == "union":
        A = A.maximum(A.T)
    elif mode == "mutual":
        A = A.minimum(A.T)
    else:
        raise ValueError("mode must be 'union' or 'mutual'")
    A.setdiag(0.0); A.eliminate_zeros()
    # Row-normalize (SNN-like smoothing can be added if desired)
    rs = np.asarray(A.sum(axis=1)).ravel()  # row sums
    inv = 1.0 / np.maximum(rs, 1e-12)  # avoid divide-by-zero
    Dinv = sp.diags(inv)
    A = Dinv @ A
    return A

def _leiden_from_similarity(A: sp.csr_matrix, resolution: float = 1.5, seed: int = 0) -> np.ndarray:
    """Leiden on a symmetric similarity graph A; returns labels."""
    import igraph as ig, leidenalg as la
    C = sp.triu(A, k=1, format="coo")
    g = ig.Graph(n=A.shape[0], edges=list(zip(C.row.tolist(), C.col.tolist())), directed=False)
    g.es["weight"] = C.data.tolist()
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                             resolution_parameter=resolution, n_iterations=-1, seed=seed)
    return np.array(part.membership, dtype=int)

def _internal_connectivity(A: sp.csr_matrix, S_idx: np.ndarray) -> float:
    """frac_internal_edges(S) = sum_{i in S, j in S} Aij / sum_{i in S, j} Aij."""
    if S_idx.size == 0:
        return 0.0
    As = A[S_idx][:, :]            # rows S
    denom = float(As.sum())        # edges from S to all
    if denom <= 0:
        return 0.0
    num = float(As[:, S_idx].sum())  # edges inside S
    return num / denom

def _marker_auc_logfc(X_rows: sp.csr_matrix, genes_idx: np.ndarray, S_mask: np.ndarray, eps: float = 1e-9):
    """
    Compute per-gene AUC (S vs not S) and logFC (meanS - meanRest on log scale) for rows[genes_idx].
    X_rows: genes×cells (not log-transformed; can be z-scored or arctan'd).
    """
    Xg = X_rows[genes_idx, :]  # small (G × n_cells)
    y = S_mask.astype(np.uint8)
    n_cells = Xg.shape[1]
    # means
    sums = np.asarray(Xg.sum(axis=1)).ravel()
    mean_all = sums / n_cells
    # positive/rest means
    sel = np.where(S_mask)[0]
    rest = np.where(~S_mask)[0]
    if sel.size == 0 or rest.size == 0:
        auc = np.zeros(len(genes_idx), dtype=float)
        lfc = np.zeros(len(genes_idx), dtype=float)
        return auc, lfc
    mean_pos = np.asarray(Xg[:, sel].mean(axis=1)).ravel()
    mean_neg = np.asarray(Xg[:, rest].mean(axis=1)).ravel()
    lfc = np.log2((mean_pos + eps) / (mean_neg + eps))

    # AUC per gene
    auc = np.zeros(len(genes_idx), dtype=float)
    # to keep memory low, compute gene by gene (G for B2 is modest)
    for i in range(len(genes_idx)):
        xi = np.asarray(Xg[i, :].todense()).ravel() if sp.issparse(Xg) else Xg[i, :]
        try:
            auc[i] = roc_auc_score(y, xi)
        except Exception:
            auc[i] = 0.5
    return auc, lfc

def _bootstrap_stability(A_full: sp.csr_matrix, base_S_mask: np.ndarray,
                         n_boot: int = 20, frac: float = 0.8,
                         resolution: float = 1.5, seed: int = 0) -> float:
    """
    Fast stability: subgraph bootstraps on A_full.
    For each bootstrap, cluster subgraph; take cluster with max Jaccard vs S∩U; average Jaccard.
    """
    rng = np.random.default_rng(seed)
    n = A_full.shape[0]
    base_idx = np.flatnonzero(base_S_mask)
    Jacc = []
    for b in range(n_boot):
        m = int(max(5, np.ceil(frac * n)))
        U = np.sort(rng.choice(n, size=m, replace=False))
        A_sub = A_full[U][:, U]
        labs = _leiden_from_similarity(A_sub, resolution=resolution, seed=seed + b + 1)
        # map cluster with best Jaccard to S∩U
        S_U_mask = np.isin(U, base_idx)
        if not S_U_mask.any():
            Jacc.append(0.0); continue
        # For each cluster id, compute Jaccard with S_U
        best = 0.0
        for c in np.unique(labs):
            T_mask = (labs == c)
            inter = (S_U_mask & T_mask).sum()
            union = (S_U_mask | T_mask).sum()
            if union > 0:
                j = inter / union
                if j > best:
                    best = j
        Jacc.append(best)
    return float(np.mean(Jacc)) if len(Jacc) else 0.0

# ---------- main: B2 rare detection ----------

def detect_rare_B2(
    X_gc: sp.csr_matrix,          # genes × cells, CPM/CP10k, no log1p
    genes_index: pd.Index,        # gene names aligned to rows
    labels_major: np.ndarray,     # parent (major) cluster labels for the same cells (length = n_cells)
    B2_genes,                     # iterable of gene names in the 3–1% band
    *,
    B2_weights: pd.Series | None = None,  # optional per-gene weights (aligned by gene name), scaled [0,1]
    use_arctan: bool = False,     # apply per-gene arctan centering before z-score/PCA
    n_pcs: int = 25,
    k_knn: int = 25,
    resolution: float = 1.5,      # tune 1.2–2.0
    A_global: sp.csr_matrix | None = None,  # optional global fused graph (cells×cells, similarity)
    mix_alpha: float = 0.7,       # A_local = alpha*A_B2 + (1-alpha)*A_global[C,C]
    size_min_abs: int | None = None,  # override absolute min size; default max(10, ceil(0.003*N_total))
    size_min_frac_parent: float = 0.008,     # ≥2% of parent (stricter with absolute)
    conn_min: float = 0.5,        # internal connectivity threshold
    auc_min: float = 0.80,        # marker AUC threshold
    n_markers_min: int = 1,       # require >=2 markers with AUC>=auc_min and logFC>0
    stab_bootstrap: int = 20,
    stab_frac: float = 0.8,
    stab_min: float = 0.5,        # mean bootstrap Jaccard ≥ 0.5
    random_state: int = 0,
    output_path: str = '.'
):
    """
    B2-band rare-pass A inside each major cluster (3–1% band).
    Returns updated labels and a report DataFrame of candidate children.
    """
    if not sp.isspmatrix_csr(X_gc):
        X_gc = X_gc.tocsr(copy=False)
    X_gc.eliminate_zeros()
    n_genes, n_cells = X_gc.shape
    if len(labels_major) != n_cells:
        raise ValueError("labels_major length must equal #cells.")

    # map B2 genes to row indices
    idx_B2 = _indexer_first(genes_index, B2_genes)
    if idx_B2.size == 0:
        raise ValueError("No B2 genes found in genes_index.")

    # Decide absolute min size
    N_total = n_cells
    size_abs = max(10, int(np.ceil(0.003 * N_total))) if size_min_abs is None else int(size_min_abs)

    # current labels to update
    labels_out = pd.Series(labels_major.copy(), index=np.arange(n_cells), dtype=object)

    rows_report = []

    # process each parent cluster C
    parents = pd.Index(pd.Series(labels_major, dtype=object).unique())
    rng = np.random.default_rng(random_state)
    child_counter = 0

    for parent in parents:
        cell_idx = np.flatnonzero(labels_major == parent)
        if cell_idx.size < max(size_abs, 2):  # too small to split
            continue

        # local matrix: (B2 genes × cells_in_C)
        X_loc = X_gc[idx_B2, :][:, cell_idx].tocsr(copy=False)

        # optional arctan centering
        if use_arctan:
            X_loc = _arctan_center_transform_rows(X_loc)

        # z-score genes across cells in C
        X_loc = _zscore_cols(X_loc)

        # optional per-gene weights (column multiplier after transpose to cells×genes)
        if B2_weights is not None:
            # align weights to idx_B2 order
            w = pd.Series(1.0, index=genes_index[idx_B2], dtype=float)
            w.loc[B2_weights.index.intersection(w.index)] = B2_weights.astype(float)
            wv = w.to_numpy().astype(np.float32)
            # multiply rows by weights (genes dimension)
            X_loc = sp.diags(wv) @ X_loc

        # PCA (cells × n_pcs) with L2 norm
        Z = _pca_from_rows(X_loc, n_components=n_pcs, random_state=random_state)

        # local graph from Z (cosine kNN, union)
        A_loc = _knn_graph_cosine(Z, k=k_knn, mode="union")

        # optional hybrid with global graph restricted to C
        if A_global is not None:
            if not sp.isspmatrix_csr(A_global):
                A_global = A_global.tocsr(copy=False)
            A_gl = A_global[cell_idx][:, cell_idx]
            al = float(mix_alpha)
            A_loc = (al * A_loc) + ((1.0 - al) * A_gl)
            A_loc.setdiag(0.0); A_loc.eliminate_zeros()

        np.maximum(A_loc.data, 0, out=A_loc.data)  # set negatives to 0 (in-place)
        A_loc.eliminate_zeros()
        # Local Leiden to propose splits
        lab_local = _leiden_from_similarity(A_loc, resolution=resolution, seed=random_state)
        parts = np.unique(lab_local)

        # evaluate each candidate child S ⊂ C (skip the largest if desired? we keep all and test)
        parent_size = cell_idx.size
        size_thresh = max(size_abs, int(np.ceil(size_min_frac_parent * parent_size)))

        for c in parts:
            S_local = np.flatnonzero(lab_local == c)          # indices in local (0..|C|-1)
            S_cells = cell_idx[S_local]                       # global cell indices
            S_size = S_cells.size
            if S_size < size_thresh:
                # too small -> reject
                rows_report.append({
                    "parent": parent, "child_label": None, "size": S_size,
                    "accepted": False, "reason": f"size<{size_thresh}"
                })
                continue

            # internal connectivity on local graph
            f_int = _internal_connectivity(A_loc, S_local)
            if f_int < conn_min:
                rows_report.append({
                    "parent": parent, "child_label": None, "size": S_size,
                    "accepted": False, "reason": f"connectivity<{conn_min:.2f}", "frac_internal": f_int
                })
                continue

            # marker coherence on B2 genes (use the same X_loc used for PCA/graph)
            S_mask_local = np.zeros(parent_size, dtype=bool)
            S_mask_local[S_local] = True
            auc, lfc = _marker_auc_logfc(X_loc, np.arange(X_loc.shape[0]), S_mask_local)
            n_markers = int(((auc >= auc_min) & (lfc > 0)).sum())
            if n_markers < n_markers_min:
                rows_report.append({
                    "parent": parent, "child_label": None, "size": S_size,
                    "accepted": False, "reason": f"markers<{n_markers_min}",
                    "frac_internal": f_int, "n_markers": n_markers
                })
                continue

            # bootstrap stability on the same local graph
            stab = _bootstrap_stability(A_loc, S_mask_local,
                                        n_boot=stab_bootstrap, frac=stab_frac,
                                        resolution=resolution, seed=random_state)
            if stab < stab_min:
                rows_report.append({
                    "parent": parent, "child_label": None, "size": S_size,
                    "accepted": False, "reason": f"stability<{stab_min:.2f}",
                    "frac_internal": f_int, "n_markers": n_markers, "stability": stab
                })
                continue

            # accept: label as child of parent
            child_counter += 1
            child_name = f"{parent}_B2_{child_counter}"
            labels_out.iloc[S_cells] = child_name

            rows_report.append({
                "parent": parent, "child_label": child_name, "size": S_size,
                "accepted": True, "reason": "ok",
                "frac_internal": f_int, "n_markers": n_markers, "stability": stab
            })


    report = pd.DataFrame(rows_report)
    report.to_csv(f"{output_path}/refine_b2.csv", index=True, header=True)
    return labels_out.to_numpy(dtype=object), report