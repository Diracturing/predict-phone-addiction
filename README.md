# Predict Phone Addiction

Kaggle Playground Series S6E8: predict a binary `addicted_label` (phone addiction) from screen-time, sleep, notification and demographic columns. Metric: AUC.

## Approach

1. EDA: missing data, distributions, and which features carry signal (screen-time features dominate).
2. Feature pruning and engineering: drop noise features, add share and ratio features, tune XGBoost with Optuna.
3. Target and frequency encoding versus native dtypes, tuned with Optuna.
4. Blending and stacking over a library of 47 out-of-fold models.

Best XGBoost CV AUC: about 0.9667, from the full engineered plus target/frequency encoding pipeline.

## Files

| File | Purpose |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis |
| `02_feature_pruning_xgb.ipynb` | Raw features versus pruned plus engineered features (Optuna XGBoost) |
| `03_target_encoding_xgb.ipynb` | Target/frequency encoding pipeline versus native dtypes (Optuna XGBoost) |
| `04_blend_and_stacking.ipynb` | Rank blending versus logistic stacking on the OOF library |
| `summary.txt` | Notes on strategies from competition write-ups |


