import random
import numpy as np
import scipy.sparse as sp
import igraph as ig

def louvain_from_radius_graph_inplace(
    G: sp.csr_matrix,
    is_distance: bool = True,          # G.data in [0,1] distances -> convert to similarity in place
    assume_symmetric: bool = True,     # set False only if builder may leave asymmetric edges
    force_symmetrize: bool = False,    # if True & not symmetric, do G = G.maximum(G.T)
    batch_edges: int = 1_000_000,      # edges per batch pushed to igraph
    resolution_like: float | None = None,  # None or >0; 1.0 ≈ neutral; >1 finer, <1 coarser (heuristic via loops)
    loops_strategy: str = "degree",    # 'degree' (loops ~ node strength) or 'uniform' (same loop weight)
    seed: int = 0,
    return_levels: bool = False        # set True to obtain Louvain hierarchy
):
    """
    Louvain clustering from a sparse ε-radius graph with minimal copies.
    MUTATES G IN PLACE (distance→similarity, zeroing diagonal).

    Returns
    -------
    labels : np.ndarray[int] (n_vertices,)
    part   : igraph.clustering.VertexClustering
             (or list of VertexClustering if return_levels=True)
    """
    # --- RNG (igraph has its own RNG depending on version) ---
    random.seed(seed)
    try:
        ig.random.seed(seed)  # igraph >=0.10
    except Exception:
        try:
            ig.set_random_number_generator(ig.RandomState(seed))  # older igraph
        except Exception:
            pass

    # --- ensure CSR & sorted indices ---
    if not sp.isspmatrix_csr(G):
        G = G.tocsr(copy=False)
    G.sort_indices()

    # 1) distance -> similarity IN PLACE
    if is_distance:
        G.data = 1.0 - G.data
        np.clip(G.data, 0.0, 1.0, out=G.data)

    # drop self loops if any
    G.setdiag(0.0)
    G.eliminate_zeros()

    # 2) Optional symmetrization
    if not assume_symmetric and force_symmetrize:
        G = G.maximum(G.T)
        G.setdiag(0.0)
        G.eliminate_zeros()
        G.sort_indices()

    n = G.shape[0]

    # 3) Build igraph from UPPER TRIANGLE only
    g = ig.Graph(n=n, directed=False)
    indptr, indices, data = G.indptr, G.indices, G.data

    def _flush_batch(es, ed, ew):
        if not es:
            return
        m = len(es)
        g.add_edges(list(zip(es, ed)))
        g.es[-m:]["weight"] = ew[:]     # copy to igraph attribute
        es.clear(); ed.clear(); ew.clear()

    es, ed, ew = [], [], []
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        js = indices[s:e]; ws = data[s:e]
        if js.size == 0:
            continue
        mask = js > i       # upper triangle
        if not np.any(mask):
            continue
        jj = js[mask]; ww = ws[mask]
        es.extend([i] * jj.size)
        ed.extend(jj.tolist())
        ew.extend(ww.tolist())
        if len(es) >= batch_edges:
            _flush_batch(es, ed, ew)
    _flush_batch(es, ed, ew)

    # --- Optional resolution-like control via self-loops (AFG trick) ---
    if resolution_like is not None and resolution_like != 1.0:
        # Add loops to bias community scale:
        #   >1.0 → more (finer) communities; <1.0 → fewer (coarser).
        # Strategy:
        #   'degree'  : loop_i = (resolution_like - 1)*strength_i
        #   'uniform' : loop_i = (resolution_like - 1)*mean_edge_weight
        if "weight" not in g.es.attributes():
            g.es["weight"] = [1.0] * g.ecount()
        if loops_strategy == "degree":
            strength = np.asarray(g.strength(weights="weight"))
            loop_w = (resolution_like - 1.0) * strength
        elif loops_strategy == "uniform":
            mean_w = float(np.mean(g.es["weight"])) if g.ecount() > 0 else 1.0
            loop_w = np.full(n, (resolution_like - 1.0) * mean_w, dtype=float)
        else:
            raise ValueError("loops_strategy must be 'degree' or 'uniform'.")

        # add self-loops
        g.add_edges([(i, i) for i in range(n)])
        g.es[-n:]["weight"] = loop_w.tolist()

    # 4) Louvain (multilevel modularity); weights are used if present
    # return_levels=True returns the hierarchy (list of VertexClustering)
    vc = g.community_multilevel(weights="weight" if "weight" in g.es.attributes() else None,
                                return_levels=return_levels)

    if return_levels:
        # final partition is the last level
        labels = np.array(vc[-1].membership, dtype=int)
        part = vc
    else:
        labels = np.array(vc.membership, dtype=int)
        part = vc

    return labels, part