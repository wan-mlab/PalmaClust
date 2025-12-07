import os
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import DBSCAN
from .clustering.leiden import leiden_from_radius_graph_inplace
from .clustering.louvain import louvain_from_radius_graph_inplace
from .parameters import Parameters



def generate_clusters(params: Parameters, G: sp.csr_matrix, cells):
    if params.clustering == 'dbscan':
        labels = DBSCAN(metric="precomputed", eps=params.dbscan_eps, min_samples=params.dbscan_minpts).fit_predict(G)
    elif params.clustering == 'leiden':
        labels = leiden_from_radius_graph_inplace(G, seed=1453, resolution=1.0)
    elif params.clustering == 'louvain':
        labels, _ = louvain_from_radius_graph_inplace(G)
    else:
        raise NotImplementedError("CLUSTERING METHOD NOT IMPLEMENTED")
    csv_path = os.path.join(params.output_folder, "cluster_output.csv")
    pd.DataFrame({"labels": labels}, index=cells).to_csv(csv_path, index=True, header=True)
    return labels

