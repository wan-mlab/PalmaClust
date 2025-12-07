import os
import shutil
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from .parameters import Parameters

def preprocess(params: Parameters):
    """
    Read a raw .h5ad into AnnData, ensure X is sparse CSR (cells x genes),
    set bookkeeping, and write a clean 'raw.h5ad' copy to the output folder.
    """
    # --- read ---
    try:
        adata = ad.read_h5ad(params.raw_filename, backed=None)  # load into memory
    except Exception as e:
        raise IOError(f"Failed to read h5ad file: {e}")

    # Ensure unique names (AnnData best practice)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()

    # --- ensure sparse CSR (cells x genes) & compact dtype ---
    X = adata.X
    if sp.issparse(X):
        # Prefer CSR for fast row ops; ensure compact dtypes
        X = X.tocsr(copy=False)
    else:
        # Convert dense to sparse (this can be large if X is huge and dense)
        X = sp.csr_matrix(X)
    # Use int32 if counts are integers; otherwise keep original dtype
    if np.issubdtype(X.dtype, np.integer):
        X.data = X.data.astype(np.int32, copy=False)
    adata.X = X

    # --- reset output folder ---
    if os.path.exists(params.output_folder):
        warnings.warn(f"Output folder {params.output_folder} already exists; deleting it.")
        shutil.rmtree(params.output_folder)
    os.makedirs(params.output_folder, exist_ok=True)

    print(f"Counts of {adata.n_vars:,} genes and {adata.n_obs:,} cells read from {params.raw_filename}")
    return (adata.X, adata.obs[params.cell_type_column].index,
            adata.var.index, adata.obs[params.cell_type_column])