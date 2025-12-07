import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from .parameters import Parameters

def filter_counts(params: Parameters, X, genes, cells, save=True, save_prefix="filtered"):
    """
    Sparse filtering on AnnData.X (CSR, cells x genes) using the same logic as your original code:
      1) remove MIR/Mir genes
      2) keep genes with >= min_cells cells having count > expression_cutoff
      3) keep cells with >= min_genes genes expressed (>0) among the kept genes
      4) keep genes whose log2(max(count)+0.1) is in [log2_cutoffl, log2_cutoffh]
    Saves a filtered .h5ad and gene/cell lists.

    Requires on `self`:
      - self.adata  (AnnData with X as CSR)
      - self.expression_cutoff, self.min_cells, self.min_genes
      - self.log2_cutoffl, self.log2_cutoffh
      - self.output_folder
    """

    X = X.T
    X = X.tocsr(copy=False)

    X.eliminate_zeros()
    n_genes, n_cells = X.shape

    genes_idx = pd.Index(genes) if not isinstance(genes, pd.Index) else genes
    cells_idx = pd.Index(cells) if not isinstance(cells, pd.Index) else cells

    if len(genes_idx) != n_genes:
        raise ValueError("len(genes) must equal X.shape[0]")
    if len(cells_idx) != n_cells:
        raise ValueError("len(cells) must equal X.shape[1]")

    expr_cutoff = getattr(params, "expression_cutoff", 0)
    min_cells = getattr(params, "min_cells", 1)
    min_genes = getattr(params, "min_genes", 1)
    log2_lo = getattr(params, "log2_cutoffl", -np.inf)
    log2_hi = getattr(params, "log2_cutoffh", np.inf)
    out_dir = getattr(params, "output_folder", ".")
    mir_regex = getattr(params, "mir_regex", r"MIR|Mir")

    # --- Step 1: gene pre-filter (MIR removal + min_cells threshold) ---
    # Count of cells per gene with expression > expression_cutoff (O(nnz), no densify)
    starts = X.indptr[:-1]
    ends = X.indptr[1:]

    if expr_cutoff <= 0:
        # number of stored nonzeros per gene
        ncells_per_gene = (ends - starts).astype(np.int64, copy=False)
    else:
        mask = (X.data > expr_cutoff)
        # robust per-row sum without building row indices (avoid reduceat empty-row pitfall)
        ncells_per_gene = np.empty(n_genes, dtype=np.int64)
        for i in range(n_genes):
            s, e = starts[i], ends[i]
            ncells_per_gene[i] = int(mask[s:e].sum())
    # MIR/Mir removal (preserve your original regex semantics)
    # Ensure string dtype for .str.contains
    gene_str = genes_idx.astype("string")
    is_mir = gene_str.str.contains(mir_regex, regex=True, na=False).to_numpy()

    keep_gene1 = (~is_mir) & (ncells_per_gene >= min_cells)
    gene_ix1 = np.flatnonzero(keep_gene1)
    if gene_ix1.size == 0:
        raise RuntimeError("All genes filtered out at step 1 (MIR removal + min_cells).")

    Xg = X[gene_ix1, :]  # (genes_kept1 x n_cells)

    # --- Step 2: cell filter (min_genes with >0 among kept genes) ---
    # Count genes >0 per cell via bincount on CSR column indices (exact, O(nnz))
    ngene_per_cell = np.bincount(Xg.indices, minlength=n_cells)
    cell_keep = (ngene_per_cell >= min_genes)
    cell_ix = np.flatnonzero(cell_keep)
    if cell_ix.size == 0:
        raise RuntimeError("All cells filtered out at step 2 (min_genes).")

    Xgc = Xg[:, cell_ix].tocsr(copy=False)
    Xgc.eliminate_zeros()

    # --- Step 3: intensity gene filter on kept cells (log2(max+0.1) window) ---
    # Row-wise max without densifying the matrix; only a length=#genes vector
    row_max = np.zeros(Xgc.shape[0], dtype=np.float64)
    st2, en2 = Xgc.indptr[:-1], Xgc.indptr[1:]
    d2 = Xgc.data
    for i in range(Xgc.shape[0]):
        s, e = st2[i], en2[i]
        row_max[i] = float(d2[s:e].max()) if e > s else 0.0

    log2maxs = np.log2(row_max + 0.1)
    keep_gene2_within = (log2maxs >= log2_lo) & (log2maxs <= log2_hi)
    gene_ix2_within = np.flatnonzero(keep_gene2_within)
    if gene_ix2_within.size == 0:
        raise RuntimeError("All genes filtered out at step 3 (log2 max thresholds).")

    X_final = Xgc[gene_ix2_within, :].tocsr(copy=False)
    X_final.eliminate_zeros()

    # Map back to original indices
    gene_ix_final = gene_ix1[gene_ix2_within]

    # Preserve input types (Index or Series)
    def _take_like(obj, idx):
        # pd.Index.take / pd.Series.take are positional and fast.
        return obj.take(idx)

    genes_filtered = _take_like(genes, gene_ix_final) if hasattr(genes, "take") else genes_idx.take(gene_ix_final)
    cells_filtered = _take_like(cells, cell_ix) if hasattr(cells, "take") else cells_idx.take(cell_ix)

    # --- Report & optional save ---
    dropped_genes = n_genes - X_final.shape[0]
    dropped_cells = n_cells - X_final.shape[1]
    print(
        f"Kept {X_final.shape[0]:,} genes and {X_final.shape[1]:,} cells "
        f"(dropped {dropped_genes:,} genes, {dropped_cells:,} cells)."
    )

    if save:
        os.makedirs(out_dir, exist_ok=True)
        sp.save_npz(os.path.join(out_dir, f"{save_prefix}_counts.npz"), X_final)
        pd.Series(genes_filtered, copy=False).to_csv(
            os.path.join(out_dir, f"{save_prefix}_genes.txt"), index=False, header=False
        )
        pd.Series(cells_filtered, copy=False).to_csv(
            os.path.join(out_dir, f"{save_prefix}_cells.txt"), index=False, header=False
        )

    return X_final, genes_filtered, cells_filtered