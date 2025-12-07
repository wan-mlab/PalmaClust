import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple

def blend_select_weighted_sum(
    gene_stat: pd.DataFrame,
    nfeatures: int,
    blend_weights: Dict[str, float],
    idf_bar: float | None = None,           # IDF gate: only idf_final > idf_bar are eligible
    column_suffix: str = "_final",
    metrics_allowed = ("fano","gini","palma","theil","idf","coverage"),
    strict: bool = False,                   # if True, error when a weighted metric column is missing
    return_scores: bool = False             # if True, return (selected_genes, combined_score_series)
) -> Tuple[pd.Index, pd.Series] | pd.Index:
    """
    Weighted-sum feature blending over normalized metrics with an IDF gate.

    Parameters
    ----------
    gene_stat : pd.DataFrame
        Index = gene IDs. Columns expected: '<metric><column_suffix>'.
        e.g., 'fano_final', 'gini_final', 'palma_final', 'theil_final', 'idf_final'.
    nfeatures : int
        Number of top genes to select (unique).
    blend_weights : dict
        Metric -> weight (nonnegative). Metrics with 0 weight or missing columns are ignored
        unless strict=True.
    idf_bar : float | None
        If provided, only genes with idf_final > idf_bar are eligible.
    column_suffix, metrics_allowed, strict, return_scores :
        See docs above.

    Returns
    -------
    selected : pd.Index
        Top-nfeatures genes by weighted-sum score (descending) among IDF-eligible genes.
    combined_scores : pd.Series (optional)
        Weighted-sum score for all genes (ineligible set to -inf), same index as gene_stat.
    """
    if nfeatures <= 0:
        raise ValueError("nfeatures must be a positive integer.")

    # ---------------- 1) IDF gate ----------------
    if idf_bar is not None:
        idf_col = f"idf"
        if idf_col not in gene_stat.columns:
            raise KeyError(f"IDF gate requested but column '{idf_col}' not found in gene_stat.")
        idf_vals = gene_stat[idf_col].to_numpy(dtype=np.float64, copy=False)
        mask_idf = np.isfinite(idf_vals) & (idf_vals > float(idf_bar))
        eligible_idx = gene_stat.index[mask_idf]
        if eligible_idx.size == 0:
            warnings.warn(f"No genes pass IDF gate: {idf_col} > {idf_bar}. Returning 0 features.")
            if return_scores:
                scores_full = pd.Series(-np.inf, index=gene_stat.index, name="weighted_sum_score")
                return pd.Index([]), scores_full
            return pd.Index([])
        df = gene_stat.loc[eligible_idx]
    else:
        df = gene_stat
        eligible_idx = gene_stat.index

    # ------------- 2) Collect usable metric columns -------------
    cols, wts = [], []
    for m, w in blend_weights.items():
        if m not in metrics_allowed or w <= 0:
            continue
        col = f"{m}{column_suffix}"
        if col not in df.columns:
            if strict:
                raise KeyError(f"Required column '{col}' not found in gene_stat.")
            else:
                continue
        cols.append(col)
        wts.append(float(w))

    if not cols:
        raise ValueError("No usable metrics found (check weights and column names).")

    # ------------- 3) Weighted-sum score on ELIGIBLE genes -------------
    M = df[cols].to_numpy(dtype=np.float64, copy=False)
    M[~np.isfinite(M)] = -np.inf
    weights = np.asarray(wts, dtype=np.float64)  # scaling doesn’t affect ranks
    combined_eligible = pd.Series(M @ weights, index=df.index, name="weighted_sum_score")

    # ------------- 4) Pick top-k, with warning if fewer than requested -------------
    k_avail = combined_eligible.shape[0]
    if k_avail < nfeatures:
        warnings.warn(
            f"Only {k_avail} genes pass the IDF gate; returning {k_avail} features (< nfeatures={nfeatures})."
        )
        k = k_avail
    else:
        k = int(nfeatures)

    order = np.argsort(-combined_eligible.to_numpy(), kind="mergesort")
    selected = combined_eligible.index[order][:k]

    if return_scores:
        # Make a full-length score vector; ineligible genes get -inf
        scores_full = pd.Series(-np.inf, index=gene_stat.index, name="weighted_sum_score")
        scores_full.loc[combined_eligible.index] = combined_eligible.values
        return selected, scores_full
    return selected