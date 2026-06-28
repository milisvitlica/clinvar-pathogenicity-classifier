# ClinVar Pathogenicity Classifier

Predict variant pathogenicity (pathogenic vs. benign) from curated **ClinVar**
expert-reviewed variants, enriched with **UniProt** human protein annotations.

## Pipeline

```
ingest_*.py             -> data/raw/         (download ClinVar + UniProt)
clean_*.py              -> data/processed/   (filter, label, tidy)
build_joined_dataset.py -> data/processed/joined.parquet (gene-level join)
*_eda.ipynb             -> exploration + feature ideas
```

## Getting started

```bash
python3 -m venv venv
source venv/bin/activate          # Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env              # then set PROJECT_ROOT to this folder's absolute path
```

Register the venv as a Jupyter kernel for the notebooks:

```bash
python -m ipykernel install --user --name clinvar-clf --display-name "Python (clinvar-clf)"
```

## Build the dataset

Run in dependency order (ingest downloads from NCBI/UniProt and needs network):

```bash
python src/ingest_clinvar.py      # -> data/raw/clinvar_reliable_grch38.parquet
python src/clean_clinvar.py       # -> data/processed/clinvar_clean.parquet (adds `label`)

python src/ingest_uniprot.py      # -> data/raw/uniprot_human_reviewed.parquet
python src/clean_uniprot.py       # -> data/processed/uniprot_clean.parquet

python src/build_joined_dataset.py  # -> data/processed/joined.parquet
```

## EDA

Open with JupyterLab (`jupyter lab`) and run top to bottom:

- `notebooks/clinvar_eda.ipynb` - ClinVar value counts, labels, scope
- `notebooks/uniprot_eda.ipynb` - UniProt fields, disease coverage
- `notebooks/joined_eda.ipynb`  - join fan-out, match types, overlap

## Project layout

```
src/
  ingest_clinvar.py        # download ClinVar -> raw parquet
  ingest_uniprot.py        # download UniProt -> raw parquet
  clean_clinvar.py         # filter + label -> clinvar_clean.parquet
  clean_uniprot.py         # filter + tidy  -> uniprot_clean.parquet
  build_joined_dataset.py  # outer-join cleaned parquets -> joined.parquet
notebooks/
  clinvar_eda.ipynb
  uniprot_eda.ipynb
  joined_eda.ipynb
data/
  raw/          # downloaded parquets (gitignored)
  processed/    # cleaned + joined parquets (gitignored)
docs/
```

## Next steps (classifier)

The repo currently produces a clean, labelled, joined dataset. Toward a model:

- **Feature engineering** from `joined.parquet` (gene, variant type, position, protein context); target = `label`.
- **Split by gene** (group split) to avoid the same gene in train and test.
- **Baseline** logistic regression / XGBoost, evaluated with stratified CV (ROC-AUC, PR-AUC).
- **Persist** the fitted model (`*.joblib`, already gitignored).
