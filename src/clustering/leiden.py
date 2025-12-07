import random
import numpy as np
import scipy.sparse as sp
import igraph as ig
import leidenalg as la

def leiden_from_radius_graph_inplace(
    G: sp.csr_matrix,
    is_distance: bool = True,         # True if G.data are distances in [0,1]; converts to similarity in place
    assume_symmetric: bool = True,    # set False only if your builder may leave asymmetric edges
    force_symmetrize: bool = False,   # if True and not assume_symmetric, do one W = G.maximum(G.T) (uses extra mem)
    batch_edges: int = 1_000_000,     # edges per batch pushed to igraph
    resolution: float = 1.0,
    n_iterations: int = -1,           # -1 → until convergence
    seed: int = 0
):
    """
    Leiden clustering from a sparse ε-radius graph with minimal copies.
    MUTATES G IN PLACE (distance→similarity, zeroing diagonal).

    Returns
    -------
    labels : np.ndarray[int] (n_vertices,)
    part   : leidenalg VertexPartition
    """
    # --- deps local to the function ---
    if not sp.isspmatrix_csr(G):
        G = G.tocsr(copy=False)  # view; minimal overhead
    G.sort_indices()

    # 1) Convert distance → similarity IN PLACE (if needed)
    if is_distance:
        G.data = 1.0 - G.data
        # clamp numerical jitter
        np.clip(G.data, 0.0, 1.0, out=G.data)

    # ensure no self loops
    G.setdiag(0.0)
    G.eliminate_zeros()

    # 2) Optional symmetrization (avoid unless you truly need it)
    #    If your radius builder produced both directions, you can skip this.
    if not assume_symmetric and force_symmetrize:
        # This makes one extra sparse matrix then frees the old ref when G is rebound.
        G = G.maximum(G.T)
        G.setdiag(0.0)
        G.eliminate_zeros()
        G.sort_indices()

    n = G.shape[0]

    # 3) Build igraph streaming the UPPER TRIANGLE only (no duplicate edges)
    g = ig.Graph(n=n, directed=False)
    # Prepare access to CSR internals once
    indptr, indices, data = G.indptr, G.indices, G.data

    def _flush_batch(edges_src, edges_dst, weights):
        if not edges_src:
            return
        # igraph add in one call; then set weights on the last batch
        m = len(edges_src)
        g.add_edges(list(zip(edges_src, edges_dst)))
        g.es[-m:]["weight"] = weights[:]  # slice copy into igraph’s attribute array
        edges_src.clear(); edges_dst.clear(); weights.clear()

    edges_src, edges_dst, weights = [], [], []
    budget = batch_edges

    # Stream rows; keep only entries with j > i (upper triangle)
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        js = indices[s:e]
        ws = data[s:e]
        if js.size == 0:
            continue
        mask = js > i
        if not np.any(mask):
            continue
        jj = js[mask]; ww = ws[mask]

        # Append to batch
        edges_src.extend([i] * jj.size)
        edges_dst.extend(jj.tolist())
        weights.extend(ww.tolist())

        # Flush if batch is large
        if len(edges_src) >= budget:
            _flush_batch(edges_src, edges_dst, weights)

    # Flush remaining
    _flush_batch(edges_src, edges_dst, weights)
    #print(g.es.attribute_names())  # should include 'weight' or 'weights'
    #print(len(g.es))  # number of edges > 0

    # 4) Leiden (RBConfiguration with weights)
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed = seed
    )
    labels = np.array(part.membership, dtype=int)
    return labels