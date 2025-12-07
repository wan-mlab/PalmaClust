import numpy as np
import pandas as pd
from .parameters import Parameters
from .blend.concatenate import blend_select_features_concat
from .blend.weighted_sum import blend_select_weighted_sum

def select_genes(
    params : Parameters,
    gene_stats : pd.DataFrame,
    blend_weights: dict,
    nfeatures: int,
    idf_bar: float
):
    """
    Simplified single-metric selector (no NaNs).
    Returns a ranked pd.Index of selected genes.
      - standard="top": pick top-N (or top-ratio) by score (descending).
      - standard="pval": pick genes with one-sided right-tail normal p < alpha,
                         ranked by ascending p-value.
    """

        # k from ratio or absolute N

        #s = np.asarray(gene_stats[metric+"_final"], dtype=float)
        #top = nfeatures
        #k = int(np.ceil(top * n)) if (isinstance(top, float) and 0 < top <= 1.0) else int(top)
        #k = max(1, min(n, k))
        #idx_k = np.argpartition(s, -k)[-k:]               # top-k (unordered)
        #idx_sorted = idx_k[np.argsort(s[idx_k])[::-1]]    # rank by score (desc)
    selected_idx = blend_select_weighted_sum(gene_stats, nfeatures, blend_weights, idf_bar=idf_bar)

    return selected_idx, None#s[idx_sorted]

