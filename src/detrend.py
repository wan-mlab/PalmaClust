import os
import pandas as pd
import numpy as np
from .parameters import Parameters
from .detrending.gini_lowess import lowess_twopass_detrending

def _rankp(col: pd.Series) -> np.ndarray:
    G = len(col)
    p_df = (col.rank(axis=0, method='average', ascending=True) - 0.5) / G
    return p_df.to_numpy()

def detrend(params: Parameters,gene_stats: pd.DataFrame) -> pd.DataFrame:
    log2max = np.asarray(gene_stats['log2max'], dtype=float)
    gini = np.asarray(gene_stats['gini'], dtype=float)
    palma = np.asarray(gene_stats['palma'], dtype=float)
    #idf = np.asarray(gene_stats['idf'], dtype=float)
    gini_d = lowess_twopass_detrending(log2max, gini)
    palma_d = lowess_twopass_detrending(log2max, palma)
    gene_stats['gini_final'] = _rankp(pd.Series(gini_d))
    gene_stats['palma_final'] = _rankp(pd.Series(palma_d))
    gene_stats['fano_final'] = _rankp(gene_stats['fano'])
    gene_stats['palma_d'] = palma_d

    csv_path = os.path.join(params.output_folder, "gene_stats_detrend.csv")
    gene_stats.to_csv(csv_path,index=True, header=True)
    return gene_stats
