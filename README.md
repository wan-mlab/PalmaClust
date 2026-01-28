# PalmaClust

**PalmaClust** is a graph-fusion clustering framework for **robust rare / ultra-rare cell type detection** in scRNA-seq data. It leverages the **Palma ratio** (a tail-sensitive inequality metric) for feature selection and combines it with complementary gene-selection views (Gini index and Fano factor) through **multi-view kNN graph fusion**, followed by **local refinement** to resolve rare subpopulations.

> Paper: *PalmaClust: A graph-fusion framework leveraging the Palma ratio for robust ultra-rare cell type detection in scRNA-seq data* (Bioinformatics, 2026). DOI: **TBD**

---

## Why PalmaClust?

Rare cell populations (<1%) can be missed by standard clustering pipelines. PalmaClust is designed to improve rare-cell sensitivity while preserving global clustering structure by:

- **Tail-sensitive feature selection** via the Palma ratio (emphasizes heavy-tailed “rare marker” patterns)
- **Multi-view kNN graph construction** (Palma / Gini / Fano feature views)
- **Weighted graph fusion** to balance rare-cell separability and global manifold stability
- **Local refinement** within major clusters to split rare / ultra-rare subtypes

---

## Installation

### 1) Clone the repository
```bash
git clone https://github.com/wan-mlab/PalmaClust.git
cd PalmaClust
```

### 2) Create an environment and install dependencies
Using `venv`:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Tip: if you hit build issues with scientific Python packages, consider using conda/mamba and then `pip install -r requirements.txt`.

---

## Quickstart

PalmaClust runs from a JSON config file.

### Run a single config
```bash
python main.py --config cfg/config_102580.json
```

### Run multiple configs (batch / glob)
```bash
python main.py --config cfg/*.json
```

### Override config values without editing JSON
You can override any config key using `--set key=value`. Values are parsed as JSON when possible.
```bash
python main.py \
  --config cfg/config_102580.json \
  --set output_folder="results/run1" palma_upper=0.1 palma_lower=0.8 palma_alpha=1e-6
```

### Print the effective config and exit (debug)
```bash
python main.py --config cfg/config_102580.json --set min_genes=200 --dry-run
```

---

## Configuration

Parameters are loaded from `src/parameters.py`. Common keys include:

- `raw_filename`: input dataset path
- `output_folder`: output directory
- `cell_type_column`: column name for ground-truth labels (for evaluation)
- `gene_name_column`: column name for gene IDs/names (if applicable to your input format)
- Palma feature scoring: `palma_alpha`, `palma_upper`, `palma_lower`
- Feature counts: `palma_nfeatures`, `gini_nfeatures`, `fano_nfeatures`
- Graph fusion weights: `palma_balance`, `gini_balance`, `fano_balance`
- Graph similarity: `distance_metric`, `jaccard_gamma`, `activation`, `graph_method`
- Refinement: `npca_b2`, `mix_alpha_b2`, etc.

### Minimal config (example)
At minimum, set `raw_filename` and `output_folder`:

```json
{
  "raw_filename": "PATH/TO/YOUR_DATA",
  "output_folder": "results/run1"
}
```

### Recommended “paper-style” starting config (example)

The paper reports the following robust defaults across benchmarks:
- QC filters: keep cells with ≥200 detected genes; keep genes expressed in ≥3 cells
- Palma hyperparameters: (pt, pb) = (0.1, 0.8) with α = 1e-6
- Feature set sizes: 1000 per metric
- Graph fusion weights (Palma, Gini, Fano): (0.5, 0.1, 0.4)
- Local refinement: truncated SVD to 20 dims; hybrid mixing λ = 0.7

```json
{
  "raw_filename": "PATH/TO/YOUR_DATA",
  "output_folder": "results/palmaclust_run",

  "cell_type_column": "cell_type",
  "gene_name_column": "Gene",

  "clustering": "leiden",

  "min_cells": 3,
  "min_genes": 200,

  "method": "palma",

  "palma_alpha": 1e-6,
  "palma_upper": 0.1,
  "palma_lower": 0.8,

  "fano_nfeatures": 1000,
  "gini_nfeatures": 1000,
  "palma_nfeatures": 1000,

  "activation": "binarize",
  "graph_method": "knn",
  "distance_metric": "jaccard",
  "jaccard_gamma": 0.9,

  "palma_balance": 0.5,
  "gini_balance": 0.1,
  "fano_balance": 0.4,

  "npca_b2": 20,
  "mix_alpha_b2": 0.7
}
```

> Notes:
> - QC thresholds and Palma fractions can be dataset-dependent.
> - Feature counts (`*_nfeatures`) are a performance/robustness knob.
> - Fusion weights (`*_balance`) control the trade-off between rare-cell sensitivity and global structure.

For the full list of keys and defaults, see `src/parameters.py`.

---

## Input data expectations

At a conceptual level, PalmaClust operates on a **filtered raw count matrix** (scRNA-seq UMI/read counts), plus:
- **cell IDs**
- **gene names**
- optional **ground-truth labels** (for evaluation only), referenced by `cell_type_column`

The exact file format/loader depends on `src/preprocess.py` in this repository. Use the configs in `cfg/` as templates for how the input is expected to be specified.

---

## Outputs

Outputs are written under `output_folder` (created if missing). Depending on your configuration and enabled steps, outputs may include:

- final cluster labels (rare/ultra-rare populations may appear as small clusters)
- optional evaluation summaries (if ground-truth labels are available), e.g. ARI/NMI and per-type breakdowns
- intermediate artifacts/plots (if enabled in your pipeline)

> Implementation note: in `src/cell_identify.py`, a `result = pd.DataFrame({"label": labels_rf}, index=cells_f)` is created.
> If you want a guaranteed label file, you can add:
> `result.to_csv(os.path.join(params.output_folder, "labels.csv"))`

---

## Citation

If you use PalmaClust in your research, please cite:

**Xingzhi Niu, Jieqiong Wang, Shibiao Wan.**  
*PalmaClust: A graph-fusion framework leveraging the Palma ratio for robust ultra-rare cell type detection in scRNA-seq data.*  
Bioinformatics, 2026. DOI: **TBD**

BibTeX (update DOI when available):
```bibtex
@article{Niu2026PalmaClust,
  title   = {PalmaClust: A graph-fusion framework leveraging the Palma ratio for robust ultra-rare cell type detection in scRNA-seq data},
  author  = {Niu, Xingzhi and Wang, Jieqiong and Wan, Shibiao},
  journal = {Bioinformatics},
  year    = {2026},
  doi     = {TBD}
}
```

---

## Contact

- **Corresponding author / maintainer:** Shibiao Wan — swan@unmc.edu
- For bugs and feature requests, please open a GitHub Issue.
