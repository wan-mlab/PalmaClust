import numpy as np
import pandas as pd
from typing import Dict, Tuple

def blend_select_features_concat(
    gene_stat: pd.DataFrame,
    nfeatures: int,
    blend_weights: Dict[str, float],
    column_suffix: str = "_final",
    metrics_allowed = ("fano","gini","palma","theil","idf"),
    return_counts: bool = True
) -> Tuple[pd.Index, dict] | pd.Index:
    """
    Concatenate blending feature selection with overlap handling.

    Parameters
    ----------
    gene_stat : pd.DataFrame
        Index = gene IDs. Must contain columns like 'fano_final', 'gini_final', ...
    nfeatures : int
        Total number of unique genes to select.
    blend_weights : dict
        Mapping metric -> weight (nonnegative). Metrics not in DataFrame or with 0 weight are ignored.
        Example: {"fano":0.3, "gini":0.2, "palma":0.2, "theil":0.3, "idf":0.0}
    column_suffix : str
        Suffix appended to metric name for the column in gene_stat.
    metrics_allowed : iterable
        Whitelist of metric names allowed.
    return_counts : bool
        If True, return (selected_genes, per_metric_counts). Else return selected_genes only.

    Returns
    -------
    selected : pd.Index
        Selected gene IDs (length == nfeatures).
    counts : dict (optional)
        Actual counts contributed by each metric after overlap resolution.
    """
    if nfeatures <= 0:
        raise ValueError("nfeatures must be a positive integer.")

    # ---- 1) Normalize and validate weights over available metrics ----
    # Keep only allowed metrics with positive weight and existing columns
    w_items = []
    for m, w in blend_weights.items():
        if m not in metrics_allowed:
            continue
        if w <= 0:
            continue
        col = f"{m}{column_suffix}"
        if col in gene_stat.columns:
            # ensure some finite values exist
            if np.isfinite(gene_stat[col].to_numpy()).any():
                w_items.append((m, float(w)))
    if not w_items:
        raise ValueError("No valid metrics with positive weight and available columns.")

    metrics, weights = zip(*w_items)
    weights = np.array(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    wsum = weights.sum()
    if wsum <= 0:
        raise ValueError("Sum of positive weights must be > 0.")
    weights = weights / wsum  # normalize

    # ---- 2) Build per-metric ranked lists (descending score, stable) ----
    ranked = {}
    for m in metrics:
        col = f"{m}{column_suffix}"
        s = gene_stat[col]
        # drop NaN / +/-inf, sort descending by value then by index (stable mergesort)
        mask = np.isfinite(s.to_numpy())
        if not mask.any():
            ranked[m] = np.array([], dtype=gene_stat.index.dtype)
            continue
        order = np.argsort(-s.to_numpy()[mask], kind="mergesort")
        ranked[m] = s.index[mask][order].to_numpy()

    # ---- 3) Quotas via largest-remainder (Hamilton) rounding ----
    raw = weights * nfeatures
    base = np.floor(raw).astype(int)
    remainder = nfeatures - int(base.sum())
    if remainder > 0:
        frac = raw - base
        add_order = np.argsort(-frac, kind="mergesort")  # largest remainders first
        for i in add_order[:remainder]:
            base[i] += 1
    quotas = dict(zip(metrics, base.tolist()))

    # ---- 4) Fill with overlap handling to keep ratios ----
    selected = []
    selected_set = set()
    contrib = {m: 0 for m in metrics}
    ptr = {m: 0 for m in metrics}              # pointer into each ranked list
    exhausted = {m: False for m in metrics}

    def pick_metric():
        # choose metric with largest deficit (quota - contrib); break ties by weight then name
        deficits = []
        for idx, m in enumerate(metrics):
            if exhausted[m]:
                continue
            deficits.append((quotas[m] - contrib[m], -weights[idx], m))  # max deficit, then larger weight
        if not deficits:
            return None
        deficits.sort(reverse=True)  # max first
        return deficits[0][2]

    # main loop: keep selecting until we reach nfeatures or everything is exhausted
    while len(selected) < nfeatures:
        m = pick_metric()
        if m is None:
            break  # nothing left to draw (should be rare if nfeatures <= #genes)
        # advance pointer until we find a non-duplicate
        arr = ranked[m]
        p = ptr[m]
        while p < arr.size and (arr[p] in selected_set):
            p += 1
        if p >= arr.size:
            exhausted[m] = True
            continue
        g = arr[p]
        ptr[m] = p + 1
        # add
        selected.append(g)
        selected_set.add(g)
        contrib[m] += 1

    # ---- 5) If still short (due to exhausted lists / heavy overlaps), fill by blended score ----
    if len(selected) < nfeatures:
        # compute a blended score as weighted sum over available metrics (missing -> 0)
        # normalize each column to [0,1] by rank (robust) to avoid scale issues
        blend_score = pd.Series(0.0, index=gene_stat.index)
        total_w = 0.0
        for m, w in zip(metrics, weights):
            col = f"{m}{column_suffix}"
            s = gene_stat[col]
            mask = np.isfinite(s)
            if not mask.any():
                continue
            # rank descending, then scale to [0,1]
            r = s.rank(method="average", ascending=False)
            r = (r.max() - r) / (r.max() - r.min() + 1e-12)
            blend_score = blend_score.add(w * r.fillna(0.0), fill_value=0.0)
            total_w += w
        if total_w > 0:
            blend_score = blend_score / total_w

        # take top of the remaining genes by blended score
        remaining = blend_score.index.difference(pd.Index(selected))
        order = np.argsort(-blend_score.loc[remaining].to_numpy(), kind="mergesort")
        needed = nfeatures - len(selected)
        selected.extend(remaining[order][:needed].tolist())

    # ---- 6) Finalize and report counts actually used (after overlap resolution) ----
    selected_idx = pd.Index(selected)

    if return_counts:
        # recompute actual contributions by first-appearance provenance:
        # Walk the selection again, attribute each gene to the metric where it first appears in its ranked list.
        # This is optional; you may prefer to keep 'contrib' from the loop (close to quotas).
        actual = {m: 0 for m in metrics}
        posmap = {m: {g: i for i, g in enumerate(ranked[m])} for m in metrics}
        # assign each selected gene to the metric where it has the *best* rank (smallest index),
        # but only among metrics with positive weight. This mirrors the spirit of "taken from"
        # while being deterministic even for the blended fill.
        for g in selected:
            best_m, best_rank = None, np.inf
            for m in metrics:
                r = posmap[m].get(g, np.inf)
                if r < best_rank:
                    best_m, best_rank = m, r
            if best_m is not None:
                actual[best_m] += 1
        return selected_idx, actual

    return selected_idx