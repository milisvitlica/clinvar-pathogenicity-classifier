# ClinVar Pathogenicity Classifier

Predict variant pathogenicity (**pathogenic** vs **benign**) from curated ClinVar
expert-reviewed variants, enriched with UniProt protein feature annotations
(domains, sites, distance to nearest feature, etc.).

Evaluation is **gene-grouped**: the same gene never appears in both train and test
partitions, so reported metrics estimate performance on held-out genes.

## Pipeline

```
ingest_*.py                         -> data/raw/
clean_*.py                          -> data/processed/
join_clinvar_uniprot.py             -> clinvar_uniprot_joined.parquet
position_matching_clinvar_uniprot.py -> clinvar_uniprot_position_matched.parquet
features.py / cv_train_eval          -> matrices, splits, nested CV metrics
final_catboost.py                    -> deployable CatBoost model + threshold

# model feature (conservation; PP3/BP4-style, not BA1):
ingest_phylop.py                     -> phylop_clean.parquet

# optional EDA (not used for training):
ingest_gnomad.py / clean_gnomad.py / join_clinvar_gnomad.py
```

## Getting started

```bash
python3 -m venv venv
source venv/bin/activate          # Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env              # set PROJECT_ROOT to this folder's absolute path
```

Register the venv as a Jupyter kernel:

```bash
python -m ipykernel install --user --name clinvar-clf --display-name "Python (clinvar-clf)"
```

## Build the dataset

Run in order (ingest needs network):

```bash
python src/ingest_clinvar.py      # -> data/raw/clinvar_reliable_grch38.parquet
python src/clean_clinvar.py       # -> clinvar_clean.parquet (P/B) + clinvar_clean_vus.parquet

python src/ingest_uniprot.py      # -> data/raw/uniprot_human_reviewed.parquet
python src/clean_uniprot.py       # -> data/processed/uniprot_clean.parquet

python src/join_clinvar_uniprot.py              # gene-level join (labelled)
python src/position_matching_clinvar_uniprot.py # AA position + UniProt feature context
```

Optional **EDA** population frequencies (not model features — ClinVar labels already
use ACMG BA1/BS1/PM2, so AF is circular with the target):

```bash
python src/ingest_gnomad.py         # gnomAD v4 AF for ClinVar SNVs (GraphQL; skips IDs already saved)
python src/clean_gnomad.py          # -> data/processed/gnomad_clean.parquet
python src/join_clinvar_gnomad.py   # left-join AF onto the position-matched table
```

Optional VUS inference table (same QC filters; not used for training):

```bash
python src/join_clinvar_uniprot.py --vus
python src/position_matching_clinvar_uniprot.py --vus
# -> data/processed/clinvar_uniprot_position_matched_vus.parquet
python src/join_clinvar_gnomad.py --vus   # optional EDA only
```

PhyloP conservation (UCSC hg38 phyloP100way) **is** a model feature:

```bash
python src/ingest_phylop.py       # -> data/processed/phylop_clean.parquet
```

Joined at train/inference time in `prepare_modeling_frame` / `prepare_inference_frame`.
PhyloP is sometimes ACMG **PP3/BP4**, as supporting/moderate computational evidence,
not stand-alone like **BA1**. Milder label leakage than gnomAD AF; gene-holdout
does not remove it (constraint is per-site). Re-run nested CV / `final_catboost.py`
after ingest so the shipped model includes `phylop_100way`.

## Modeling

### Features

Structured features from the position-matched table (gene-proxy / high-cardinality
identity fields such as Chromosome, Length, and free-text domain notes are
**excluded**). Includes protein position, distance to closest UniProt feature,
overlap flags (`in_domain`, …), `closest_feature_type`, alleles, and
`phylop_100way` (UCSC 100-way vertebrate conservation at the GRCh38 reference base).

gnomAD allele frequencies are available for EDA (`notebooks/eda/gnomad_eda.ipynb`)
but are **not** in the model matrix: ClinVar P/B labels already use ACMG BA1/BS1/PM2.

PhyloP sometimes PP3/BP4, as supporting/moderate computational evidence, not
stand-alone like BA1.

XGBoost and CatBoost consume categoricals natively. Elastic-net logistic and
random forest use `encode_for_sklearn()`: median-impute + scale numerics, one-hot
cats fitted on the training frame only (`handle_unknown=ignore`).

### Two-stage protocol

| Stage | What | Purpose |
|-------|------|---------|
| **1. Nested gene CV** | Outer test folds + inner tune | **Honest KPIs** on held-out genes |
| **2. Single-loop gene CV** on all labelled data | Pick one `θ*`, `n_estimators*`, threshold | **Deploy hyperparameters** |
| then | Fit on **all** labelled rows | Ship `models/catboost_final_*.cbm` |

Nested CV produces a *different* hyperparam set per outer fold — it is for
evaluation, not a single production config. Stage 2 re-tunes once on the full
labelled table so the shipped model has one coherent setting. The stage-2 tune
score (e.g. inner mean ROC-AUC ≈ 0.67) is a **selection diagnostic only** — do
not report it as model performance.

### Honest performance (nested CV)

```bash
python src/cv_train_eval.py          # write data/processed/cv_folds.parquet
# then open notebooks/modeling/cv/cv_xgboost.ipynb
```

Gene-grouped **nested CV** (`src/cv_train_eval.py` / `notebooks/modeling/cv/cv_xgboost.ipynb`):

- Outer folds → test metrics (report these)
- Inner CV (on the outer-train pool only) → hyperparameters, tree budget
  (`n_estimators` = median of inner `best_iteration+1`), and decision threshold
- Fits use `scale_pos_weight` from training label counts
- Threshold: **Youden's J**, median of per-inner-fold cutoffs (not pooled scores;
  plain F1 often collapses to near-all-pathogenic on path-heavy folds)

**Report:**

- **ROC-AUC** and **PR-AUC** as mean ± std across outer folds (threshold-free)
- **Precision / recall / accuracy / F1 / confusion matrix** at the Youden cutoff

Do not use pooled OOF ROC as the headline number (fold score scales differ).

### Holdout baseline (optional)

```bash
python src/features.py               # optional train/valid/test parquets
```

### Final classifier (single-loop CV tune → fit all)

After nested CV, choose deploy settings with a **standard (single-loop) gene CV**
on the full labelled table, then fit on every labelled row:

```bash
python src/final_catboost.py
# or notebooks/modeling/final/train_final_catboost.ipynb
```

Writes:

- `models/catboost_final_pathogenicity.cbm` — CatBoost model
- `models/catboost_final_pathogenicity_meta.json` — threshold, params, feature schema,
  `tune_inner_mean_roc_auc` (diagnostic only)

XGBoost stage-2 notebooks remain under `notebooks/modeling/final/deprecated/`
(`python src/final_model.py`).

### Inference

```bash
# notebooks/modeling/final/infer_catboost.ipynb
```

Scores **ClinVar VUS** from `clinvar_uniprot_position_matched_vus.parquet` with the
saved CatBoost model + Youden threshold (writes `data/processed/vus_inference_predictions.parquet`).
API: `predict_pathogenicity()` / `prepare_inference_frame()` in `src/`.

## Notebooks

### EDA (`notebooks/eda/`)

- `clinvar_eda.ipynb`
- `uniprot_eda.ipynb`
- `joined_clinvar_uniprot_eda.ipynb`
- `position_matching_eda.ipynb`
- `gnomad_eda.ipynb`
- `phylop_eda.ipynb`

### Modeling (`notebooks/modeling/`)

Stage 1 lives in `cv/` (compare algorithms on the same gene folds).
The winner is the only model that goes to `final/` for full-data train + inference.

- `cv/cv_xgboost.ipynb` — nested CV (XGBoost reference)
- `cv/cv_logistic.ipynb` — elastic-net logistic (one-hot cats)
- `cv/cv_random_forest.ipynb` — random forest (one-hot cats)
- `cv/cv_catboost.ipynb` — CatBoost (native categoricals)
- `cv/conclusions.md` — nested-CV comparison → **CatBoost** for inference
- `final/train_final_catboost.ipynb` — full-data CV tune → deploy model
- `final/infer_catboost.ipynb` — score new rows (e.g. VUS)
- `final/deprecated/` — previous XGBoost train / infer notebooks

## Project layout

```
src/
  ingest_clinvar.py / ingest_uniprot.py
  clean_clinvar.py / clean_uniprot.py
  join_clinvar_uniprot.py
  position_matching_clinvar_uniprot.py
  ingest_gnomad.py / clean_gnomad.py / join_clinvar_gnomad.py  # EDA only
  ingest_phylop.py           # UCSC phyloP100way (model feature)
  features.py                # matrices, encoding, fit helpers; optional holdout split
  cv_train_eval.py           # gene CV folds + nested CV (honest KPIs)
  cv_baselines.py            # nested CV for logistic / RF / CatBoost
  final_catboost.py          # single-loop full-data tune + fit all + inference
  final_model.py             # XGBoost stage 2 (kept; notebooks deprecated)
notebooks/
  eda/
    clinvar_eda.ipynb
    uniprot_eda.ipynb
    joined_clinvar_uniprot_eda.ipynb
    position_matching_eda.ipynb
    gnomad_eda.ipynb             # optional; AF not used in training
    phylop_eda.ipynb             # phyloP100way (model feature; PP3/BP4 caveat)
  modeling/
    cv/
      cv_xgboost.ipynb           # stage 1: nested CV (XGBoost)
      cv_logistic.ipynb          # elastic-net logistic
      cv_random_forest.ipynb     # random forest
      cv_catboost.ipynb          # CatBoost
      conclusions.md             # pick winner for inference (CatBoost)
    final/                       # winner only (CatBoost)
      train_final_catboost.ipynb # stage 2: full-data CV tune → deploy model
      infer_catboost.ipynb       # score new rows (e.g. VUS)
      deprecated/                # previous XGBoost stage-2 notebooks
data/
  raw/          # gitignored
  processed/    # gitignored
models/         # gitignored (*.json model + meta)
```

## Notes

- Target: ClinVar `label` ∈ {pathogenic, benign} (expert-reviewed SNVs after cleaning).
- Splits are by **gene**, not by random variants.
- Mega-genes (e.g. BRCA1/2) can dominate some folds; interpret fold-wise metrics with that in mind.
