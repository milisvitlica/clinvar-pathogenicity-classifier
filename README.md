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
holdout_train_eval / cv_train_eval  -> splits, nested CV metrics
final_model.py                      -> deployable model + threshold
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

Optional VUS inference table (same QC filters; not used for training):

```bash
python src/join_clinvar_uniprot.py --vus
python src/position_matching_clinvar_uniprot.py --vus
# -> data/processed/clinvar_uniprot_position_matched_vus.parquet
```

## Modeling

### Features

Structured features from the position-matched table (gene-proxy / high-cardinality
identity fields such as Chromosome, Length, and free-text domain notes are
**excluded**). Includes protein position, distance to closest UniProt feature,
overlap flags (`in_domain`, …), `closest_feature_type`, and alleles.

### Two-stage protocol

| Stage | What | Purpose |
|-------|------|---------|
| **1. Nested gene CV** | Outer test folds + inner tune | **Honest KPIs** on held-out genes |
| **2. Single-loop gene CV** on all labelled data | Pick one `θ*`, `n_estimators*`, threshold | **Deploy hyperparameters** |
| then | Fit on **all** labelled rows | Ship `models/xgb_final_*.json` |

Nested CV produces a *different* hyperparam set per outer fold — it is for
evaluation, not a single production config. Stage 2 re-tunes once on the full
labelled table so the shipped model has one coherent setting. The stage-2 tune
score (e.g. inner mean ROC-AUC ≈ 0.67) is a **selection diagnostic only** — do
not report it as model performance.

### Honest performance (nested CV)

```bash
python src/cv_train_eval.py          # write data/processed/cv_folds.parquet
# then open notebooks/modeling/xgboost/cv_xgboost.ipynb
```

Gene-grouped **nested CV** (`src/cv_train_eval.py` / `notebooks/modeling/xgboost/cv_xgboost.ipynb`):

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
python src/holdout_train_eval.py     # train/valid/test parquets
```

### Final classifier (single-loop CV tune → fit all)

After nested CV, choose deploy settings with a **standard (single-loop) gene CV**
on the full labelled table, then fit on every labelled row:

```bash
python src/final_model.py
# or notebooks/modeling/xgboost/train_final_xgboost.ipynb
```

Writes:

- `models/xgb_final_pathogenicity.json` — XGBoost model
- `models/xgb_final_pathogenicity_meta.json` — threshold, params, feature schema,
  `tune_inner_mean_roc_auc` (diagnostic only)

### Inference

```bash
# notebooks/modeling/xgboost/infer_xgboost.ipynb
```

Scores **ClinVar VUS** from `clinvar_uniprot_position_matched_vus.parquet` with the
saved model + Youden threshold (writes `data/processed/vus_inference_predictions.parquet`).
API: `predict_pathogenicity()` / `prepare_inference_frame()` in `src/`.

## Notebooks

### EDA (`notebooks/eda/`)

- `clinvar_eda.ipynb`
- `uniprot_eda.ipynb`
- `joined_clinvar_uniprot_eda.ipynb`
- `position_matching_eda.ipynb`

### Modeling (`notebooks/modeling/xgboost/`)

- `cv_xgboost.ipynb` — stage 1: nested CV evaluation
- `train_final_xgboost.ipynb` — stage 2: full-data CV tune → deploy model
- `infer_xgboost.ipynb` — score new rows (e.g. VUS)

## Project layout

```
src/
  ingest_clinvar.py / ingest_uniprot.py
  clean_clinvar.py / clean_uniprot.py
  join_clinvar_uniprot.py
  position_matching_clinvar_uniprot.py
  holdout_train_eval.py      # features, holdout split, fit helpers
  cv_train_eval.py           # gene CV folds + nested CV (honest KPIs)
  final_model.py             # single-loop full-data tune + fit all + inference
notebooks/
  eda/
    clinvar_eda.ipynb
    uniprot_eda.ipynb
    joined_clinvar_uniprot_eda.ipynb
    position_matching_eda.ipynb
  modeling/
    xgboost/
      cv_xgboost.ipynb           # stage 1: nested CV evaluation
      train_final_xgboost.ipynb  # stage 2: full-data CV tune → deploy model
      infer_xgboost.ipynb        # score new rows (e.g. VUS)
data/
  raw/          # gitignored
  processed/    # gitignored
models/         # gitignored (*.json model + meta)
```

## Notes

- Target: ClinVar `label` ∈ {pathogenic, benign} (expert-reviewed SNVs after cleaning).
- Splits are by **gene**, not by random variants.
- Mega-genes (e.g. BRCA1/2) can dominate some folds; interpret fold-wise metrics with that in mind.
