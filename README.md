# Mitoxyperilysis–mTORC2–RhoA–actin axis in human sepsis: transcriptomic reanalysis

Reanalysis code and derived results for the manuscript:

> **The mTORC2–RhoA–actin axis in mitoxyperilysis: transcriptomic reanalysis of murine
> macrophages and validation in a human sepsis cohort**
> Dong Yang, Huozhao Ruan, Jinhai Liu, Shiyu Geng, Xin Kang
> Department of Emergency Medicine, The Fifth Affiliated Hospital of Southern Medical University

---

## What this repository contains

| Path | Contents |
|---|---|
| `scripts/m1_ssgsea_null.py` | M1 — expression- and variance-matched permutation null + competitive control (ssGSEA scoring) |
| `scripts/permutation_specificity.py` | M1 — original all-in-one script (z-mean / ssGSEA selectable) |
| `scripts/adjust_cell_composition.py` | M4 — cell-composition adjustment, marker-based (PanglaoDB) deconvolution |
| `scripts/adjust_cell_composition_xcell.py` | M4 — cell-composition adjustment using official xCell output |
| `scripts/run_xcell.R` | Runs the official xCell R package on a linear-scale matrix |
| `data_derived/signatures/` | Gene sets (mouse-derived modules after human ortholog mapping; Hallmark sets; mTOR complexes) |
| `data_derived/meta_65682.csv` | Sample annotation for GSE65682 (group, age, sex, endotype, 28-day mortality) |
| `data_derived/xcell_results.csv` | xCell abundance scores (67 cell types × 802 samples) |
| `data_derived/cell_composition.csv` | Marker-based deconvolution scores (13 cell types × 802 samples) |
| `results/` | All numeric outputs and figures reported in the manuscript |

## Data sources (public)

- **GSE235046** — murine BMDM RNA-seq, 5 groups × n = 3 (Media, LPS, CS, LPS+CS, LPS+CS+Torin-1), 9 h.
- **GSE65682** — MARS cohort, human whole blood (Affymetrix HG-U219), 479 sepsis / 42 healthy / 281 other ICU.

Both are downloaded directly from NCBI GEO; raw matrices are **not** redistributed here.

## Reproducing the analysis

### 0. Environment

```bash
python3 -m pip install pydeseq2 gseapy numpy pandas scipy statsmodels scikit-learn matplotlib
# for the xCell branch:
sudo apt-get install -y r-base r-base-dev
Rscript -e 'install.packages(c("GSVA","GSEABase","pracma","MASS","digest","curl","quadprog"))'
# xCell itself (official package) — see scripts/run_xcell.R header
```

### 1. M1 — specificity against a matched permutation null

```bash
python3 scripts/m1_ssgsea_null.py \
  --expr GSE65682_gene_matrix_log2.csv \
  --meta data_derived/meta_65682.csv \
  --module data_derived/signatures/death_promoting_module.txt \
  --group_col group --sepsis sepsis --control healthy \
  --n_perm 5000 --seed 2026 --score_method ssgsea --outdir M1_ssgsea
```

Output: `M1_ssgsea_results.json`, `Fig_M1_ssgsea_null.png/.pdf`, null distribution `.npy`.

**Headline result.** Observed Cohen d = 1.85 (P = 2.3 × 10⁻²¹) lies **within** the
expression/variance-matched null (mean 0.59, SD 1.03, 95th percentile 2.29);
empirical P = 0.117, z = 1.22. The module's elevation is therefore not
distinguishable from what a matched random gene set shows.

### 2. M4 — cell-composition adjustment

```bash
# (a) marker-based deconvolution (PanglaoDB)
python3 scripts/adjust_cell_composition_xcell.py   # or adjust_cell_composition.py with --cells cell_composition.csv

# (b) official xCell
Rscript scripts/run_xcell.R GSE65682_gene_matrix_linear.csv xcell_results.csv
python3 scripts/adjust_cell_composition_xcell.py \
  --expr GSE65682_gene_matrix_log2.csv \
  --meta data_derived/meta_65682.csv \
  --sigdir data_derived/signatures \
  --cells data_derived/xcell_results.csv \
  --covars age,sex --score_method ssgsea --outdir M4_xCell_out
```

**Headline result.** With the four marker-based proxy scores the death-promoting
module survives adjustment (β = 0.66, FDR = 4.5 × 10⁻⁴) and only mTORC1 loses
significance. With **xCell** covariates the death-module association drops to
β = 0.33 (P = 0.07) under a four-cell-type model and is non-significant under a
larger model — because several xCell T-cell/B-cell scores are themselves almost
collinear with case–control status (Cohen d up to −4.3), so adjustment removes a
large fraction of the very contrast being tested.

## Notes on scoring method

All signature scores in the manuscript use **ssGSEA** (gseapy, `sample_norm_method="rank"`).
Reproducing the published Table 3 (d values) requires ssGSEA; a z-score-of-mean
approach gives a reversed sign for the death-promoting module and should not be used.

## License

Code: MIT (see `LICENSE`). Derived gene sets and result tables: CC BY 4.0.
