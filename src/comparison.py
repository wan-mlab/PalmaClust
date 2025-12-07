import numpy as np
import pandas as pd
from .parameters import Parameters
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def compare_clusters_filtered(
    params: Parameters,
    labels_f: np.ndarray,           # predicted cluster ids for filtered cells (e.g., ints, may include -1)
    labels_all: np.ndarray,         # ground-truth labels for ALL cells (same order as `cells`)
    cells_f: pd.Index,              # filtered cell IDs (subset of `cells`)
    cells: pd.Index,                # ALL cell IDs (same order as `labels_all`)
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    if not isinstance(cells, pd.Index):
        cells = pd.Index(cells)
    if not isinstance(cells_f, pd.Index):
        cells_f = pd.Index(cells_f)

    # All filtered cells must be present in `cells`
    pos = cells.get_indexer(cells_f)
    if np.any(pos < 0):
        missing = cells_f[np.flatnonzero(pos < 0)[:5]]
        raise ValueError(f"Some filtered cells not found in `cells`, e.g. {list(missing)} ...")

    y_true_f = labels_all[pos].astype(object)  # ground truth for filtered cells
    y_pred_f = np.asarray(labels_f, dtype=str)
    n_f = y_pred_f.shape[0]
    if y_true_f.shape[0] != n_f:
        raise ValueError("Length mismatch between filtered predictions and mapped ground truth.")

    # ---------------------------
    # 1) Per-predicted-cluster table (filtered cells only)
    # ---------------------------
    pred, counts = np.unique(y_pred_f, return_counts=True)
    tab = pd.DataFrame(
        {"count": counts},
        index=pd.Index(pred, dtype=object, name="cluster.ID")
    )
    tab["frequency"] = tab["count"] / float(n_f)
    tab["mapped_label"] = "N/A"
    tab["hit"] = 0
    tab["precision"] = 0.0
    tab["recall"] = 0.0
    tab["entropy"] = 0.0
    tab["f1"] = 0.0


    # Ground-truth counts among filtered cells
    gt_counts = pd.Series(y_true_f, index=cells_f, dtype="object").value_counts()

    # Majority-vote mapping and stats per predicted cluster
    y_true_series = pd.Series(y_true_f, index=cells_f, dtype="object")
    y_pred_series = pd.Series(y_pred_f, index=cells_f, dtype="object")

    for c in tab.index:
        members_mask = (y_pred_series.values == c)
        m = int(members_mask.sum())
        if m == 0:
            continue
        gt_sub = y_true_series.values[members_mask]
        pool = pd.Series(gt_sub, dtype="object").value_counts()

        p = (pool / pool.sum()).to_numpy(dtype=float)
        entropy = float(-(p * np.log2(p)).sum()) if p.size else 0.0

        winner = pool.idxmax()
        hit = int(pool.loc[winner])
        precision = hit / m
        denom = int(gt_counts.get(winner, 0))
        recall = (hit / denom) if denom > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        tab.loc[c, "mapped_label"] = str(winner)
        tab.loc[c, "hit"] = hit
        tab.loc[c, "precision"] = precision
        tab.loc[c, "recall"] = recall
        tab.loc[c, "entropy"] = entropy
        tab.loc[c, "f1"] = f1


    # ---------------------------
    # 2) Overall metrics on filtered cells
    # ---------------------------
    ari = adjusted_rand_score(y_true_f, y_pred_f)
    nmi = normalized_mutual_info_score(y_true_f, y_pred_f, average_method="arithmetic")

    # ---------------------------
    # 3) GT-centric breakdown (filtered cells only)
    # ---------------------------
    # Global predicted cluster sizes (filtered only), for tie-breaking
    pred_sizes = tab["count"]

    def _label_sort_key_for_ties(lbl):
        # numeric labels (e.g., -1, 0, 1, ...) rank before non-numeric; then ascending
        try:
            v = int(lbl)
            return (0, v)
        except Exception:
            return (1, str(lbl))

    rows = []
    for gt_label, cnt_gt in gt_counts.items():
        # predicted clusters used by this GT label
        mask = (y_true_f == gt_label)
        preds_for_gt = y_pred_f[mask]
        vc = pd.Series(preds_for_gt, dtype="object").value_counts()  # counts within this GT label
        num_clusters = int(vc.size)

        # top1 by count; tie -> smaller *cluster size*; tie again -> smaller cluster ID
        top_count = vc.max()
        cands = vc[vc == top_count].index

        if cands.size == 1:
            top_cluster = cands[0]
        else:
            # break tie by smaller global cluster size
            sizes = pred_sizes.reindex(cands).astype(int)
            min_size = sizes.min()
            cands2 = cands[sizes == min_size]
            if cands2.size == 1:
                top_cluster = cands2[0]
            else:
                # final tie-break by cluster ID
                top_cluster = sorted(cands2, key=_label_sort_key_for_ties)[0]

        tp = int(vc.loc[top_cluster])
        cluster_size = int(pred_sizes.loc[top_cluster]) if top_cluster in pred_sizes.index else tp
        precision = tp / cluster_size if cluster_size > 0 else 0.0
        recall = tp / int(cnt_gt) if cnt_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        mapped_match = False
        # Compare with per-cluster mapping; both as strings
        try:
            mapped_match = (str(tab.loc[top_cluster, "mapped_label"]) == str(gt_label))
        except KeyError:
            mapped_match = False

        rows.append({
            "gt_label": str(gt_label),
            "count": int(cnt_gt),
            "frequency": float(cnt_gt / n_f),
            "top1_cluster": top_cluster,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_clusters": num_clusters,
            "mapped_match": bool(mapped_match),
        })

    gt_breakdown = pd.DataFrame(rows).set_index("gt_label")

    tab.to_csv(f"{params.output_folder}/cluster_analysis.csv", index=True, header=True)
    gt_breakdown.to_csv(f"{params.output_folder}/groundtruth_breakdown.csv", index=True, header=True)
    with open(f"{params.output_folder}/overall_performance.txt", "w") as f:
        f.write(f"ARI: {ari:0.4f}, NMI: {nmi:0.4f}\n")

    return tab, gt_breakdown, ari, nmi
