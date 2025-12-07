import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

# ----------------------- helpers reused/adapted -----------------------

def _indexer_first(gf: pd.Index, gq) -> np.ndarray:
    gq = pd.Index(gq)
    if gf.is_unique:
        idx = gf.get_indexer(gq)
        return idx[idx >= 0].astype(np.int64)
    pos = pd.Series(np.arange(len(gf), dtype=np.int64), index=gf)
    first = pos.groupby(level=0).nth(0)
    idx_s = first.reindex(gq)
    return idx_s[idx_s.notna()].astype(np.int64).to_numpy()

def _zscore_cols(X: sp.csr_matrix) -> sp.csr_matrix:
    if not sp.isspmatrix_csr(X):
        X = X.tocsr(copy=False)
    X = X.astype(np.float32, copy=False)
    n_cells = X.shape[1]
    sums = np.asarray(X.sum(axis=1)).ravel()
    mu = sums / float(n_cells)
    Xsq = X.copy(); Xsq.data **= 2
    sums2 = np.asarray(Xsq.sum(axis=1)).ravel()
    var = (sums2 / float(n_cells)) - (mu ** 2)
    var[var < 0] = 0
    std = np.sqrt(var, dtype=np.float32)
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr))
    X.data -= mu[rows]
    nz = std[rows] > 0
    X.data[nz] /= std[rows[nz]]
    X.eliminate_zeros()
    return X

def _arctan_center_transform_rows(X_rows_csr: sp.csr_matrix) -> sp.csr_matrix:
    if not sp.isspmatrix_csr(X_rows_csr):
        X_rows_csr = X_rows_csr.tocsr(copy=False)
    X_rows_csr.sort_indices()
    n_feat, n_cells = X_rows_csr.shape
    indptr, data = X_rows_csr.indptr, X_rows_csr.data

    for r in range(n_feat):
        s, e = indptr[r], indptr[r + 1]
        if s == e: continue
        v = data[s:e].astype(np.float64, copy=False)
        vs = np.sort(v)[::-1]
        S = float(vs.sum())
        if S <= 0.0: continue
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
                cs = np.cumsum(vs)
                clamp = np.minimum(ends, vs.size)
                sums_at_end = cs[clamp - 1]
                hit = sums_at_end > th
                j_end = int(ends[int(np.flatnonzero(hit)[0])]) if np.any(hit) else int(ends[-1])
        if j_end <= vs.size:
            expM = float(np.cumsum(vs)[j_end - 1] / j_end)
        else:
            expM = float(S / j_end)
        data[s:e] = 10.0 * (np.arctan(v - expM) + np.arctan(expM))
    X_rows_csr.eliminate_zeros()
    return X_rows_csr

def _pca_from_rows(X_rows: sp.csr_matrix, n_components: int = 15, random_state: int = 0) -> np.ndarray:
    X_cg = X_rows.T
    svd = TruncatedSVD(n_components=n_components, random_state=1453)#, random_state=random_state)
    Z = svd.fit_transform(X_cg)
    Z = normalize(Z, norm="l2", axis=1, copy=False)
    return Z

def _knn_graph_cosine(X: np.ndarray, k: int = 20, mode: str = "union") -> sp.csr_matrix:
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), metric="cosine", n_jobs=-1)
    nbrs.fit(X)
    dist, idx = nbrs.kneighbors(X, return_distance=True)
    dist = dist[:, 1:]; idx = idx[:, 1:]
    sim = 1.0 - dist
    rows = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    cols = idx.ravel(); dat = sim.ravel()
    A = sp.csr_matrix((dat, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T) if mode == "union" else A.minimum(A.T)
    A.setdiag(0.0); A.eliminate_zeros()
    # row-normalize
    rs = np.asarray(A.sum(axis=1)).ravel()
    Dinv = sp.diags(1.0 / np.maximum(rs, 1e-12))
    A = Dinv @ A
    return A

def _internal_connectivity(A: sp.csr_matrix, S_idx: np.ndarray) -> float:
    if S_idx.size == 0: return 0.0
    As = A[S_idx][:, :]
    denom = float(As.sum())
    if denom <= 0: return 0.0
    num = float(As[:, S_idx].sum())
    return num / denom

def _marker_auc_logfc(X_rows: sp.csr_matrix, genes_idx: np.ndarray, S_mask: np.ndarray, eps: float = 1e-9):
    Xg = X_rows[genes_idx, :]
    y = S_mask.astype(np.uint8)
    n_cells = Xg.shape[1]
    sel = np.where(S_mask)[0]; rest = np.where(~S_mask)[0]
    if sel.size == 0 or rest.size == 0:
        return np.zeros(len(genes_idx)), np.zeros(len(genes_idx))
    mean_pos = np.asarray(Xg[:, sel].mean(axis=1)).ravel()
    mean_neg = np.asarray(Xg[:, rest].mean(axis=1)).ravel()
    lfc = np.log2((mean_pos + eps) / (mean_neg + eps))
    auc = np.zeros(len(genes_idx), dtype=float)
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
    import igraph as ig, leidenalg as la
    rng = np.random.default_rng(seed)
    n = A_full.shape[0]
    base_idx = np.flatnonzero(base_S_mask)
    J = []
    for b in range(n_boot):
        m = int(max(5, np.ceil(frac * n)))
        U = np.sort(rng.choice(n, size=m, replace=False))
        A_sub = A_full[U][:, U]
        C = sp.triu(A_sub, k=1, format="coo")
        g = ig.Graph(n=A_sub.shape[0], edges=list(zip(C.row.tolist(), C.col.tolist())), directed=False)
        g.es["weight"] = C.data.tolist()
        #la.set_rng_seed(seed + b + 1)
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                 resolution_parameter=resolution, n_iterations=-1, seed=seed)
        labs = np.array(part.membership, dtype=int)
        S_U_mask = np.isin(U, base_idx)
        if not S_U_mask.any():
            J.append(0.0); continue
        best = 0.0
        for cid in np.unique(labs):
            T_mask = (labs == cid)
            inter = (S_U_mask & T_mask).sum()
            union = (S_U_mask | T_mask).sum()
            if union > 0:
                best = max(best, inter / union)
        J.append(best)
    return float(np.mean(J)) if J else 0.0

# ----------------------- main B1 detector -----------------------

def detect_ultrarare_B1(
    X_gc: sp.csr_matrix,           # genes × cells (CPM/CP10k), no log1p
    genes_index: pd.Index,         # gene names aligned to rows
    labels_major: np.ndarray,      # parent cluster labels per cell
    B1_genes,                      # iterable of 0.1–1% band genes
    B1_weights: pd.Series | None = None,  # optional per-gene weights in [0,1], index=gene
    use_arctan: bool = False,      # per-gene arctan centering before z-score/PCA
    n_pcs: int = 15,
    k_knn: int = 20,
    seed_top_frac: float = 0.005,  # 0.5% (use 0.01 for 1.0%)
    seed_mad_k: float = 3.0,
    grow_q: float = 0.95,          # grow threshold quantile for s_smooth
    min_comp_size: int = 5,
    s_mad_accept: float = 1.0,     # median(s) ≥ median(parent)+s_mad_accept*MAD to accept
    conn_min: float = 0.20,
    auc_min: float = 0.85,
    n_markers_min: int = 1,
    stab_bootstrap: int = 20,
    stab_frac: float = 0.8,
    stab_min: float = 0.30,
    propagate_border: bool = False,
    prop_steps: int = 2,           # number of random-walk steps for label propagation
    prop_pmin: float = 0.80,       # minimum probability to attach a border cell
    random_state: int = 0,
    output_path: str = '.'
):
    """
    B1 seed-and-grow rare detector (0.1–1% band) inside each major cluster.
    Returns updated labels and a report DataFrame.
    """
    if not sp.isspmatrix_csr(X_gc):
        X_gc = X_gc.tocsr(copy=False)
    X_gc.eliminate_zeros()
    n_genes, n_cells = X_gc.shape
    if len(labels_major) != n_cells:
        raise ValueError("labels_major length must equal #cells.")

    # map B1 genes
    idx_B1 = _indexer_first(genes_index, B1_genes)
    if idx_B1.size == 0:
        raise ValueError("No B1 genes found in genes_index.")

    # weights aligned to idx_B1 (default = 1)
    if B1_weights is not None:
        w = pd.Series(1.0, index=genes_index[idx_B1], dtype=float)
        w.loc[B1_weights.index.intersection(w.index)] = B1_weights.astype(float)
        wv = w.to_numpy().astype(np.float32)
    else:
        wv = np.ones(idx_B1.size, dtype=np.float32)

    labels_out = pd.Series(labels_major.copy(), index=np.arange(n_cells), dtype=object)
    report_rows = []
    rng = np.random.default_rng(random_state)
    parents = pd.Index(pd.Series(labels_major, dtype=object).unique())
    child_counter = 0

    for parent in parents:
        C = np.flatnonzero(labels_major == parent)
        if C.size < max(20, min_comp_size + 5):
            continue

        # local matrix: B1 genes × |C|
        X_loc = X_gc[idx_B1, :][:, C].tocsr(copy=False)

        # optional arctan
        if use_arctan:
            X_loc = _arctan_center_transform_rows(X_loc)

        # z-score across cells in C
        X_loc = _zscore_cols(X_loc)

        # local embedding for graph (cells × n_pcs)
        Z = _pca_from_rows(X_loc, n_components=n_pcs, random_state=random_state)

        # local kNN cosine graph, row-normalized (this is A_B1)
        A = _knn_graph_cosine(Z, k=k_knn, mode="union")  # similarity & row-stochastic

        # rare score s = sum_g w_g * z_ig  (z_ig lives in X_loc; genes×cells)
        # compute efficiently: s = (w^T * X_loc)  -> shape (cells,)

        wv = np.asarray(wv, dtype=np.float64)
        if wv.ndim != 1 or wv.shape[0] != X_loc.shape[0]:
            raise ValueError(f"Weight vector shape {wv.shape} must equal #rows of X_loc={X_loc.shape[0]}.")
        s = X_loc.T.dot(wv)  # shape (C_size,)
        s = np.asarray(s, dtype=np.float64)  # ensure 1-D
        # smooth once: s_tilde = A * s
        s_tilde = A.dot(s)

        # thresholds in parent C
        med = float(np.median(s_tilde))
        mad = float(np.median(np.abs(s_tilde - med))) + 1e-12
        seed_thr = med + seed_mad_k * mad
        q_thr = np.quantile(s_tilde, 1.0 - float(seed_top_frac))

        seeds_mask = (s_tilde >= seed_thr) & (s_tilde >= q_thr)
        if not seeds_mask.any():
            report_rows.append({"parent": parent, "child_label": None, "accepted": False,
                                "reason": "no_seeds"})
            continue

        # grow threshold
        grow_thr = float(np.quantile(s_tilde, float(grow_q)))
        grow_nodes = np.flatnonzero(s_tilde >= grow_thr)
        if grow_nodes.size == 0:
            report_rows.append({"parent": parent, "child_label": None, "accepted": False,
                                "reason": "no_grow_nodes"})
            continue

        # connected components on thresholded subgraph
        mask_all = np.zeros(C.size, dtype=bool); mask_all[grow_nodes] = True
        A_thr = A[mask_all][:, mask_all]
        n_comp, comp_labels = connected_components(A_thr, directed=False, return_labels=True)
        # map back to local indices
        comps = []
        for k in range(n_comp):
            loc = np.flatnonzero(comp_labels == k)
            if loc.size == 0: continue
            comps.append(grow_nodes[loc])

        # candidate filtering and validation
        med_s = med
        for comp in comps:
            if comp.size < min_comp_size:
                report_rows.append({"parent": parent, "child_label": None, "size": int(comp.size),
                                    "accepted": False, "reason": f"size<{min_comp_size}"})
                continue

            s_med = float(np.median(s[comp]))
            s_mad = float(np.median(np.abs(s - np.median(s)))) + 1e-12
            if s_med < (np.median(s) + s_mad_accept * s_mad):
                report_rows.append({"parent": parent, "child_label": None, "size": int(comp.size),
                                    "accepted": False, "reason": f"score_median_below_{s_mad_accept}MAD"})
                continue

            # internal connectivity (on A)
            f_int = _internal_connectivity(A, comp)
            if f_int < conn_min:
                report_rows.append({"parent": parent, "child_label": None, "size": int(comp.size),
                                    "accepted": False, "reason": f"connectivity<{conn_min:.2f}",
                                    "frac_internal": f_int})
                continue

            # marker coherence (on B1 genes used for X_loc)
            S_mask_local = np.zeros(C.size, dtype=bool); S_mask_local[comp] = True
            auc, lfc = _marker_auc_logfc(X_loc, np.arange(X_loc.shape[0]), S_mask_local)
            n_mark = int(((auc >= auc_min) & (lfc > 0)).sum())
            if n_mark < n_markers_min:
                report_rows.append({"parent": parent, "child_label": None, "size": int(comp.size),
                                    "accepted": False, "reason": f"markers<{n_markers_min}",
                                    "frac_internal": f_int, "n_markers": n_mark})
                continue

            # bootstrap stability (stricter)
            stab = _bootstrap_stability(A, S_mask_local,
                                        n_boot=stab_bootstrap, frac=stab_frac,
                                        resolution=1.5, seed=random_state)
            if stab < stab_min:
                report_rows.append({"parent": parent, "child_label": None, "size": int(comp.size),
                                    "accepted": False, "reason": f"stability<{stab_min:.2f}",
                                    "frac_internal": f_int, "n_markers": n_mark, "stability": stab})
                continue

            # ACCEPT: create child label on global indices
            child_counter += 1
            child_name = f"{parent}_B1_{child_counter}"
            labels_out.iloc[C[comp]] = child_name

            # Optional: border propagation by random walks
            if propagate_border:
                # Start prob vector y (one-hot over comp / normalized)
                y0 = np.zeros(C.size, dtype=np.float64)
                y0[comp] = 1.0 / comp.size
                P = A  # already row-stochastic
                y = y0.copy()
                for _ in range(int(prop_steps)):
                    y = P @ y
                # assign unlabeled neighbors with high probability and close to the component (not in comp)
                attach = np.flatnonzero((y >= float(prop_pmin)) & (~S_mask_local))
                if attach.size:
                    labels_out.iloc[C[attach]] = child_name

            report_rows.append({
                "parent": parent, "child_label": child_name, "size": int(comp.size),
                "accepted": True, "reason": "ok",
                "frac_internal": f_int, "n_markers": n_mark, "stability": stab
            })
    report = pd.DataFrame(report_rows)
    report.to_csv(f"{output_path}/refine_b1.csv", index=True, header=True)
    return labels_out.to_numpy(dtype=object), report