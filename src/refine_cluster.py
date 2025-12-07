import scipy.sparse as sp
from .parameters import Parameters
from .refine.band2_refine import detect_rare_B2
from .refine.band1_refine import detect_ultrarare_B1

def refine_cluster(
        params: Parameters,
        mat: sp.csr_matrix,
        genes,
        labels,
        band_genes,
        global_graph
):
    _labels, report = detect_rare_B2(mat, genes, labels, band_genes[3],A_global=global_graph,
                                     output_path=params.output_folder, conn_min=params.conn_b2, stab_min=params.stab_b2,
                                     random_state=12277, n_pcs=params.npca_b2, k_knn=params.npca_b2,
                                     size_min_frac_parent=params.size_min_b2, mix_alpha=params.mix_alpha_b2)
    _labels, report = detect_ultrarare_B1(mat, genes, _labels, band_genes[4], output_path=params.output_folder,
                                          conn_min=params.conn_b1, stab_min=params.stab_b1)
    return _labels