from src.graph.jaccard_radius_graph import jaccard_radius_graph_blockwise
from src.graph.knn_graph import knn_graph_from_binary_B
from src.parameters import Parameters
import scipy.sparse as sp


def make_graph(params: Parameters, B: sp.csr_matrix):

    if params.graph_method == "radius":
        obj = jaccard_radius_graph_blockwise(B, eps=params.dbscan_eps)
    elif params.graph_method == "knn":
        obj = knn_graph_from_binary_B(B.T,metric=params.distance_metric, return_distance=True)
    else:
        raise NotImplementedError(params.graph_method+" Graph Not Implemented")

    print(obj.shape)
    return obj