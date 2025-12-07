import pandas as pd
import scipy.sparse as sp
from .select_genes import select_genes
from .activate import make_binary
from .graph.make_graph import make_graph
from .graph.graph_mix import mix_knn_graphs
from .parameters import Parameters

def make_channel_graphs(params: Parameters, gene_stats:pd.DataFrame, matrix:sp.csr_matrix,
                        genes_f, labels, cells_f, cells, b5=None, b4=None, b3=None, b2=None, b1=None, bands=None,
                        band_weights = None):
    if b5 is None: b5 = {"fano":1.0}
    if b4 is None: b4 = {"gini":1.0}
    if b3 is None: b3 = {"palma":1.0}
    if b2 is None: b2 = {"palma":params.b2_palma, "gini":params.b2_gini}
    if b1 is None: b1 = {"palma":params.b1_palma, "gini":params.b1_gini}



    if bands is None:
        bands = [
            ("50-30", b5, 0.0, params.fano_nfeatures),
            ("30-10", b4, 0.0, params.gini_nfeatures),
            ("10-3", b3, 3.5, params.palma_nfeatures),
            ("3-1", b2, 1.0, params.b2_nfeatures),
            ("1-0.1", b1, 3.5, params.b1_nfeatures)
        ]
    band_graphs = []
    if band_weights is None: band_weights = [params.fano_balance, params.gini_balance, params.palma_balance, 0.0, 0.0]
    #print(bands,band_weights)



    band_genes = []
    for band in bands:
        #print(band[0])
        qualifiers, weighted_ranked_p = select_genes(params, gene_stats, band[1], band[3], band[2])
        binarized, zero_cells = make_binary(params, matrix, genes_f, qualifiers)
        #print(zero_cells.sum())
        graph = make_graph(params, binarized)
        band_graphs.append(graph)
        band_genes.append(qualifiers)
    G = mix_knn_graphs(band_graphs, band_weights)
    return G, band_genes

