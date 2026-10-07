import numpy as np
import scipy.sparse as sp
from .parameters import Parameters


def cp10k_normalize(X_gc: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """
    Scale each cell to `target_sum` total counts (CP10K by default), keeping the
    matrix sparse. X_gc is genes x cells, matching filter_counts's output orientation.

    Gini/Palma/Fano, binarization and refinement all ran on raw counts before this
    (see Parameters.normalize); sequencing depth then leaked directly into gene
    selection and features. This is opt-in (params.normalize == "cp10k") so the
    default pipeline is unaffected.
    """
    if not sp.isspmatrix_csc(X_gc):
        X_gc = X_gc.tocsc(copy=False)
    sums = np.asarray(X_gc.sum(axis=0)).ravel()
    scale = np.where(sums > 0, target_sum / sums, 0.0)
    X_gc = X_gc @ sp.diags(scale)
    X_gc = X_gc.tocsr()
    X_gc.data = X_gc.data.astype(np.float64, copy=False)
    return X_gc


def maybe_normalize(params: Parameters, X_gc: sp.csr_matrix) -> sp.csr_matrix:
    if params.normalize == "cp10k":
        return cp10k_normalize(X_gc)
    if params.normalize == "none":
        return X_gc
    raise NotImplementedError(f"normalize={params.normalize!r} not implemented")
