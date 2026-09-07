# Nested CV comparison — which model to ship

Four algorithms were tuned and scored independently on the **same gene-grouped outer folds** (`cv_folds.parquet`, seed 42). Headline numbers are outer-fold **mean ± std**. Thresholded metrics use the inner-chosen **Youden** cutoff (the same rule inference would apply).

Sources: `cv_xgboost.ipynb`, `cv_logistic.ipynb`, `cv_random_forest.ipynb`, `cv_catboost.ipynb`.

## Summary


| Model                | ROC-AUC           | PR-AUC            | Precision         | Recall            | F1                | Accuracy          |
| -------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Random forest        | **0.698 ± 0.034** | **0.714 ± 0.109** | **0.713 ± 0.094** | 0.615 ± 0.143     | 0.652 ± 0.109     | **0.669 ± 0.026** |
| CatBoost             | 0.697 ± 0.040     | 0.711 ± 0.110     | 0.693 ± 0.110     | **0.668 ± 0.036** | **0.677 ± 0.057** | 0.666 ± 0.028     |
| XGBoost              | 0.682 ± 0.028     | 0.698 ± 0.120     | 0.697 ± 0.104     | 0.577 ± 0.227     | 0.608 ± 0.191     | 0.655 ± 0.015     |
| Elastic-net logistic | 0.617 ± 0.056     | 0.654 ± 0.147     | 0.700 ± 0.105     | 0.513 ± 0.215     | 0.577 ± 0.191     | 0.647 ± 0.022     |




## Fold-wise ROC-AUC (paired)


| Fold                         | XGBoost   | Logistic | Random forest | CatBoost  |
| ---------------------------- | --------- | -------- | ------------- | --------- |
| 0 (19 genes, 36% pathogenic) | 0.687     | 0.534    | **0.737**     | **0.737** |
| 1                            | **0.722** | 0.664    | 0.723         | **0.738** |
| 2                            | 0.691     | 0.665    | 0.689         | **0.693** |
| 3 (74% pathogenic)           | **0.659** | 0.588    | 0.649         | 0.651     |
| 4                            | 0.651     | 0.634    | **0.691**     | 0.664     |


Mean Δ ROC vs XGBoost: random forest **+0.016**, CatBoost **+0.015**, logistic **−0.065**.

The tree-model gaps are smaller than fold-to-fold std (~0.03–0.04). Logistic loses on every fold.

## What the numbers say

**Logistic is out.** A linear additive model on one-hot UniProt/allele features is clearly weaker, especially on fold 0 (ROC 0.534). Tree models pick up interactions that elastic-net does not.

**XGBoost, random forest, and CatBoost are in a dead heat on ranking.** Mean ROC/PR differ by ~0.015. That is not a strong enough gap, with only five gene folds, to treat “best mean AUC” as a scientific winner.

**The operating point is not a dead heat.** Inference will apply one Youden threshold and emit hard labels (and VUS scores). There XGBoost is brittle: fold-0 recall **0.179** (F1 0.278) vs CatBoost **0.620** on the same genes. CatBoost recall std is **0.036** vs XGBoost **0.227**. Random forest sits in between (fold-0 recall 0.366). For a deploy cutoff that has to survive mega-gene / prevalence swings, CatBoost is the most usable.

## Decision for inference

**Use CatBoost** for stage 2 (full-data tune → fit all → VUS inference).

Reasons, in order:

1. Same ranking quality as random forest (and a small lift over XGBoost).
2. Most stable Youden behaviour across gene pools — the quantity inference actually uses.
3. Native categoricals, same feature matrix as XGBoost (no one-hot).

Do **not** report CatBoost’s nested-CV score as if model selection had a third nested loop. The honest claim is: among four pre-specified families, CatBoost matched the best AUCs and failed less at the cutoff.

Random forest is the runner-up if a sklearn-only stack is required. Keep XGBoost as the documented GBDT baseline; do not ship logistic.

## Next

Stage 2 is CatBoost: `notebooks/modeling/final/train_final_catboost.ipynb` then
`infer_catboost.ipynb` (`src/final_catboost.py`). XGBoost train/infer notebooks
are under `notebooks/modeling/final/deprecated/`.