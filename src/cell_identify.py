import pandas as pd

import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from .parameters import Parameters
from .preprocess import preprocess
from .filter import filter_counts
from .calc_stat import calc_gene_stats
from .detrend import detrend
from .generate_cluster import generate_clusters
from .comparison import compare_clusters_filtered
from .channels import make_channel_graphs
from .refine_cluster import refine_cluster

def cell_identify(config_filename:str):
    params = Parameters(config_filename)
    matrix, cells, genes, labels = preprocess(params)
    matrix_f, genes_f, cells_f = filter_counts(params, matrix, genes, cells, False)
    gene_stats = calc_gene_stats(params, matrix_f, genes_f)
    gene_stats = detrend(params, gene_stats)


    graph, band_genes = make_channel_graphs(params, gene_stats, matrix_f, genes_f, labels, cells_f, cells)
    labels_f = generate_clusters(params, graph, cells_f)
    labels_rf = refine_cluster(params, matrix_f, genes_f, labels_f, band_genes, graph)
    result = pd.DataFrame({"label":labels_rf}, index=cells_f)
    tab, gt_breakdown, ari, nmi = compare_clusters_filtered(params, labels_rf, labels, cells_f, cells)




