from .parameters import Parameters
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .activation.binarize import jaccard_binary
from .activation.arctan import arctan_transform

def make_binary(
    params: Parameters,
    X_filtered: sp.csr_matrix,
    genes_filtered,                 # Index/Series/array aligned to rows of X_filtered
    genes_qualified                 # Index/Series/array (~50–200)
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
    if not sp.isspmatrix_csr(X_filtered):
        X_filtered = X_filtered.tocsr(copy=False)
    X_filtered.eliminate_zeros()

    gf = pd.Index(genes_filtered) if not isinstance(genes_filtered, pd.Index) else genes_filtered
    gq = pd.Index(genes_qualified) if not isinstance(genes_qualified, pd.Index) else genes_qualified


    # order-preserving indexer in the order of genes_qualified
    idx = gf.get_indexer(gq)
    keep = idx >= 0
    if not np.any(keep):
        raise ValueError("None of genes_qualified are present in genes_filtered.")
    idx_sel = idx[keep]
    feat_names = gq[keep]  # preserve caller's order

    # --- slice rows FIRST (only selected genes) ---
    X_sel = X_filtered[idx_sel, :]            # (n_feat x n_cells), new object; no full copy

    if params.activation == 'binarize':
        B, zero_cells = jaccard_binary(X_sel, params.jaccard_gamma)
    elif params.activation == 'arctan':
        B, zero_cells = arctan_transform(X_sel)
    elif params.activation == 'none':
        X_sel.eliminate_zeros()

        # cells x features
        B = X_sel.T.tocsr(copy=False)
        # store as float32 to save memory (optional; comment out if you want float64)
        B.data = B.data.astype(np.float32, copy=False)

        zero_cells = (B.getnnz(axis=1) == 0)

    else:
        raise NotImplementedError(params.activation)

    return B, zero_cells