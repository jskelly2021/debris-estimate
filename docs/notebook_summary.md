# Notebook Summary: notebooks/xgboost.ipynb

This document summarizes the current behavior of `notebooks/xgboost.ipynb`.

The notebook has no saved cell outputs. The behavior below is derived from the code in the notebook, not from persisted result tables or figures.

## Overall Workflow

The notebook trains a two-layer debris volume model for target `VolBoth_sum` using XGBoost:

1. Load `../data/GrideH9_v1.xlsx`.
2. Preprocess predictors, remove leakage columns, clip numeric feature outliers, log-transform selected columns, and one-hot encode predictors.
3. For several target clipping percentiles, create stratified train/test splits and store them in `clip_results`.
4. Train a layer-1 zero-vs-positive classifier at clip `99`, then reuse its positive-sample split for every target clipping level through `clf_results`.
5. Train layer-2 low-vs-high tier classifiers and separate low/high regressors on log-transformed positive target values.
6. Run threshold sweeps, learning curves, a grid search, full metric summaries, and actual-vs-predicted scatter plots.

## Shared Data And Variable Dependencies

- `df_raw`: raw Excel data loaded from `../data/GrideH9_v1.xlsx`.
- `df`: mutable feature frame used during preprocessing.
- `y_reg_raw`: raw target series, `df["VolBoth_sum"]`.
- `df_model`: one-hot encoded feature matrix.
- `clip_percentiles`: `[50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]`.
- `clip_results`: created by Cell 2 and required by Cells 3-9.
- `expand_features`: created by Cell 3 and required by Cells 4-9.
- `clf_results`: created by Cell 3 and required by Cells 4-9.
- `calc_metrics`: first created by Cell 4 with six metrics, then replaced by Cell 8 with nine metrics. Cells after Cell 8 expect the nine-value version.
- `best_thr_by_clip`: redefined in multiple cells with different values. Each cell uses the local definition in that cell.

## Cell 1: Initial Data Peek

Purpose:
- Imports common data and plotting libraries.
- Loads the Excel file and prints basic dataset information.

Inputs:
- `../data/GrideH9_v1.xlsx`.

Outputs or important variables:
- `df`: raw DataFrame from Excel.
- Prints dataset shape and column names.

Hardcoded values:
- Input path: `../data/GrideH9_v1.xlsx`.

Dependencies:
- Independent. Later cells reload the data and do not rely on this cell's `df`.

## Cell 2: Preprocessing And Split

Notebook label:
- `Cell 1: Preprocessing + Split`

Purpose:
- Builds the reusable modeling baseline for all later cells.
- Loads data, defines the target, removes leakage features, applies feature clipping/log transforms, creates distance flags, one-hot encodes predictors, clips target values at multiple percentiles, and creates stratified train/test splits.

Inputs:
- `../data/GrideH9_v1.xlsx`.
- Target column: `VolBoth_sum`.

Outputs or important variables:
- `df_raw`: raw Excel data.
- `df`: preprocessed feature data before dummy encoding.
- `y_reg_raw`: raw `VolBoth_sum`.
- `clip_df`: summary of feature clipping by feature.
- `thr_coast`, `thr_res`: 10th percentile thresholds among positive `dist_coast` and `dist_reservoir` values.
- `near_coast`, `near_reservoir`: binary distance indicators added to `df`.
- `df_model`: dummy-encoded model matrix.
- `clip_percentiles`: `[50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]`.
- `clip_results`: dictionary keyed by clip percentile. Each entry includes:
  - `y_reg`: target clipped at that percentile.
  - `y_upper`: positive target quantile used as the upper clipping value.
  - `n_clipped`: count of positive targets above `y_upper`.
  - `threshold`: initial balanced low/high threshold around the positive median.
  - `X_train`, `X_test`: stratified feature splits.
  - `y_train_reg`, `y_test_reg`: clipped target aligned to the split.

Leakage columns dropped when present:
- `VolCD`, `VolVG`, `VolCD_sum`, `VolVG_sum`, `VolBoth_sum`
- `Bin_CD`, `Bin_VG`, `Bin_Both`
- `GRID_ID`
- `age_sum`, `age_med`, `num_story_sum`, `num_story_med`
- `sqft_sum`, `sqft_med`
- `found_ht_sum`, `found_ht_med`
- `val_struct_sum`, `val_struct_med`
- `val_cont_sum`, `val_cont_med`
- `landuse`

Feature clipping behavior:
- `log_cols`: `["sqft", "val_struct", "val_cont", "fld_pct"]`; these are excluded from numeric clipping and log-transformed later.
- `cont_cols`: numeric columns excluding names starting with `occtype` or `bldg`, excluding `dist_coast`, `dist_reservoir`, and excluding `log_cols`.
- For each `cont_col`, compute skew and set an upper clipping quantile:
  - skew `> 3`: `0.80`
  - skew `> 2`: `0.90`
  - skew `> 1`: `0.98`
  - otherwise: `0.995`
- Values are clipped with `np.clip(df[col], 0, upper)`, so negative values are also floored to `0`.

Target clipping and split behavior:
- For each `cp` in `clip_percentiles`, `y_upper` is the `cp / 100` quantile of positive target values.
- `y_reg_cp = y_reg_raw.clip(upper=y_upper)`.
- Candidate initial thresholds are centered around the median of positive clipped target values:
  - `median_val = int(y_pos_cp.median())`
  - `range(max(50, median_val - 200), median_val + 225, 25)`
- The chosen `best_thr` minimizes `abs(low_ratio - 0.5)`.
- Stratification labels are:
  - `zero` for target `== 0`
  - `low` for target `> 0` and `<= best_thr`
  - `high` for target `> best_thr`
- Split uses `train_test_split(df_model, y_stratify, test_size=0.1, random_state=42, stratify=y_stratify)`.

Plots:
- A 2 x 6 histogram grid of positive clipped target distributions with median and selected threshold.
- A 2 x 6 bar chart grid comparing train/test counts for zero, low, and high classes.

Dependencies:
- Later cells depend on `clip_results`, `clip_percentiles`, and the preprocessed/dummy-encoded feature columns stored inside each split.

## Cell 3: Layer 1 Zero-Vs-Positive Classifier

Notebook label:
- `Cell 2-A: Layer 1 - Zero vs Positive Classification`

Purpose:
- Trains the layer-1 classifier that predicts whether target volume is zero or positive.
- Selects top feature groups by XGBoost feature importance, expands grouped categorical families, balances training data with SMOTE, evaluates classification, and stores positive train/test subsets for every target clipping level.

Inputs:
- `clip_results[99]` from Cell 2.

Outputs or important variables:
- `XGB_PARAMS_CLF`
- `bldg_cols`: columns starting with `bldg`.
- `occtype_cols`: columns starting with `occtype`.
- `grouped_cols`: union of `bldg_cols` and `occtype_cols`.
- `grouped_imp_df`: feature importances with all `bldg*` columns collapsed into `bldg_type` and all `occtype*` columns collapsed into `occtype`.
- `top20_groups`: top 20 grouped features.
- `expand_features(group_list, bldg_cols, occtype_cols)`: expands `bldg_type` and `occtype` groups back into their dummy columns.
- `top_features`: expanded feature list used by the classifier.
- `clf_l1`: trained zero-vs-positive classifier.
- `train_acc_l1`, `test_acc_l1`, `auc_l1`, `report_df`.
- `clf_results`: dictionary keyed by clip percentile. Each entry includes the classifier, selected features, grouped column metadata, layer-1 metrics, and positive samples for layer-2 modeling.

Model parameters/constants:
- `XGBClassifier` for layer 1:
  - `n_estimators=50`
  - `max_depth=10`
  - `min_child_weight=10`
  - `colsample_bynode=0.8`
  - `random_state=42`
  - `n_jobs=-1`
- Fixed classification clip: `cp = 99`.
- SMOTE: `SMOTE(random_state=42)`.
- Top feature groups: first 20 rows of `grouped_imp_df`.

Metrics calculated:
- Train accuracy on SMOTE-resampled training data.
- Test accuracy on the original test split.
- ROC AUC.
- Precision, recall, and F1 for `Zero`, `Positive`, and macro average.

Plots:
- ROC curve for layer 1.

Dependencies:
- Requires `clip_results`.
- Creates `expand_features` and `clf_results`, which later cells require.
- `X_train` and `X_test` are taken from `clip_results[99]`. The same indices are reused for all `cp_reg` values inside `clf_results`.
- `X_train_pos` is based on actual positive training rows.
- `X_test_pos` is based on predicted-positive test rows, not actual-positive test rows.

## Cell 4: Layer 2 Regression Threshold Sweep

Notebook label:
- `Cell 2-B: Layer 2 - Regression`

Purpose:
- For each target clipping percentile, sweeps low/high threshold candidates and trains a tier classifier plus separate low/high regressors.
- Selects the best threshold per clip by lowest test COV.

Inputs:
- `clf_results` from Cell 3.
- `clip_results` from Cell 2.
- `expand_features` from Cell 3.

Outputs or important variables:
- `calc_metrics`: six-metric version returning `rmse`, `mae`, `r2`, `nrmse`, `cov`, `agg`.
- `XGB_PARAMS_REG`
- `all_reg_results`: list of threshold-sweep result dictionaries.
- `all_reg_df`: DataFrame of all accepted threshold results.
- `best_df`: best row per clip by `COV_Test`.
- `best_overall`: single best row by `COV_Test`.

Model parameters/constants:
- `XGBClassifier` and `XGBRegressor` use:
  - `n_estimators=50`
  - `max_depth=6`
  - `min_child_weight=1`
  - `colsample_bynode=0.8`
  - `random_state=42`
  - `n_jobs=-1`
- `clip_percentiles = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]`.
- `thr_candidates = list(range(100, 1500, 25))`.
- Skip threshold when `thr >= y_upper`.
- Skip threshold when either low or high train group has fewer than 5 samples.
- Skip threshold unless train low ratio is between `0.30` and `0.70`, inclusive.
- Tier SMOTE uses `SMOTE(random_state=42)` with a broad exception fallback to unbalanced training data.

Training behavior:
- For each clip, regression feature importance is fit once using `X_train_pos` and `np.log1p(y_tr_pos)`.
- Top 20 grouped regression features are expanded with `expand_features`.
- Tier labels are `0` when `y_tr_pos <= thr`, else `1`.
- Low and high regressors are trained on `np.log1p(y_tr_pos)` and predictions are inverted with `np.expm1`.
- Train predictions use actual low/high masks.
- Test predictions route rows by predicted tier from the tier classifier.
- Negative predictions are floored with `np.clip(..., 0, None)`.

Metrics calculated:
- Tier train/test accuracy and gap.
- Train/test R2 and R2 gap.
- Train/test NRMSE, where `nrmse = rmse / (y_true.max() - y_true.min()) * 100`.
- Train/test COV, where `cov = rmse / y_true.mean()`.
- Train/test aggregate percent error, where `agg = abs(y_true - y_pred).sum() / y_true.sum() * 100`.

Plots:
- A 2 x 3 comparison of best-per-clip train/test metrics.

Dependencies:
- Requires `clf_results`, `clip_results`, `expand_features`, `SMOTE`, `pd`, `np`, `plt`, and sklearn metrics imported earlier.
- Creates `calc_metrics`; later cells use this function until Cell 8 replaces it.

## Cell 5: Train-Only Learning Curve

Notebook label:
- `Cell 2-C: Learning Curve(Training)`

Purpose:
- For each clip, uses hardcoded thresholds and sweeps `n_estimators` to evaluate training-only behavior.

Inputs:
- `clf_results`, `clip_results`, `clip_percentiles`, `expand_features`, and six-value `calc_metrics`.

Outputs or important variables:
- `best_thr_by_clip`: hardcoded threshold map for this cell.
- `n_estimators_list`: `[5, 10, ..., 100] + [200, 300, ..., 2000]`.
- `lc_results`: dictionary keyed by clip; each value is a DataFrame of train-only learning-curve metrics.

Hardcoded thresholds:
- `50: 325`
- `55: 225`
- `60: 200`
- `65: 625`
- `70: 725`
- `75: 200`
- `80: 150`
- `85: 175`
- `90: 150`
- `95: 150`
- `99: 175`

Model parameters/constants:
- Feature-importance model is fixed at `n_estimators=50`, `max_depth=6`, `min_child_weight=1`, `colsample_bynode=0.8`, `random_state=42`, `n_jobs=-1`.
- Sweep models use the same parameters except `n_estimators=n`.
- Tier SMOTE uses `SMOTE(random_state=42)` with a broad exception fallback.

Metrics calculated:
- Tier training accuracy.
- Low-tier train R2.
- High-tier train R2.
- Overall train R2.
- Train NRMSE, COV, and aggregate percent error.

Plots:
- A 3 x 2 learning-curve plot across clips for train-only metrics.

Dependencies:
- Requires `calc_metrics` to return six values.
- Reuses the `lc_results` name; Cell 6 overwrites it.

## Cell 6: Train-And-Test Learning Curve

Notebook label:
- `Cell 2-C: Learning Curve: Train + Test`

Purpose:
- Similar to Cell 5, but evaluates both train and test metrics for `n_estimators` from 5 to 100.

Inputs:
- `clf_results`, `clip_results`, `clip_percentiles`, `expand_features`, and six-value `calc_metrics`.

Outputs or important variables:
- `best_thr_by_clip`: same threshold map as Cell 5.
- `n_estimators_list`: `list(range(5, 105, 5))`.
- `lc_results`: dictionary keyed by clip; this replaces the train-only `lc_results` from Cell 5.

Hardcoded thresholds:
- `50: 325`
- `55: 225`
- `60: 200`
- `65: 625`
- `70: 725`
- `75: 200`
- `80: 150`
- `85: 175`
- `90: 150`
- `95: 150`
- `99: 175`

Model parameters/constants:
- Feature-importance model is fixed at `n_estimators=50`, `max_depth=6`, `min_child_weight=1`, `colsample_bynode=0.8`, `random_state=42`, `n_jobs=-1`.
- Sweep models use the same parameters except `n_estimators=n`.
- Tier SMOTE uses `SMOTE(random_state=42)` with a broad exception fallback.

Metrics calculated:
- Tier train/test accuracy and gap.
- Train/test R2 and R2 gap.
- Train/test NRMSE.
- Train/test COV.
- Train/test aggregate percent error.

Plots:
- Separate figures for R2, NRMSE, COV, aggregate percent error, tier accuracy, and R2 gap.
- Train lines are thin; test lines are thicker with markers.
- R2 gap plot includes horizontal reference lines at `15` and `10`.

Dependencies:
- Requires six-value `calc_metrics`.
- Overwrites `lc_results` from Cell 5.

## Cell 7: Grid Search For max_depth And min_child_weight

Purpose:
- Runs a small grid search over `max_depth` and `min_child_weight` for one fixed clip/threshold setting.

Inputs:
- `clf_results`, `clip_results`, `expand_features`, and six-value `calc_metrics`.

Outputs or important variables:
- `gs_results`: list of grid-search result dictionaries.
- `gs_df`: DataFrame of grid-search results.

Hardcoded values:
- `cp = 90`.
- `thr = 100`.
- `n_estimators = 50`.
- `param_grid["max_depth"] = [5, 6, 8, 10, 15, 20]`.
- `param_grid["min_child_weight"] = [1, 2, 3, 5, 10]`.
- Total combinations: `30`.

Model parameters/constants:
- All models use `n_estimators=50`, selected `max_depth`, selected `min_child_weight`, `colsample_bynode=0.8`, `random_state=42`, and `n_jobs=-1`.
- Feature-importance model remains fixed at `max_depth=6`, `min_child_weight=1`.
- Tier SMOTE uses `SMOTE(random_state=42)` with a broad exception fallback.

Metrics calculated:
- Train/test R2 and R2 gap.
- Train/test NRMSE.
- Train/test COV.
- Train/test aggregate percent error.
- Tier train/test accuracy and tier gap.

Plots:
- 2 x 3 heatmap grid for `R2_Test`, `NRMSE_Test`, `COV_Test`, `Err_Test`, `R2_Gap`, and `Tier_Gap`.

Dependencies:
- Requires six-value `calc_metrics`.

## Cell 8: Full Metrics Summary With Updated calc_metrics

Notebook label:
- `calc_metrics update`

Purpose:
- Replaces `calc_metrics` with an expanded metric function and evaluates fixed `n_estimators=50` models per clip using a different hardcoded threshold map.

Inputs:
- `clf_results`, `clip_results`, `clip_percentiles`, and `expand_features`.

Outputs or important variables:
- Updated `calc_metrics`: returns nine values.
- `best_thr_by_clip`: new threshold map for this cell.
- `full_results`: list of full metric summaries.
- `full_df`: DataFrame of full metric summaries.

Hardcoded thresholds:
- `50: 100`
- `55: 100`
- `60: 100`
- `65: 100`
- `70: 100`
- `75: 100`
- `80: 100`
- `85: 100`
- `90: 100`
- `95: 175`
- `99: 150`

Model parameters/constants:
- All tier/regression models use:
  - `n_estimators=50`
  - `max_depth=6`
  - `min_child_weight=1`
  - `colsample_bynode=0.8`
  - `random_state=42`
  - `n_jobs=-1`
- Feature-importance model uses the same parameters.
- Tier SMOTE uses `SMOTE(random_state=42)` with a broad exception fallback.

Metrics calculated:
- Existing metrics:
  - RMSE
  - MAE
  - R2
  - NRMSE as percent of target range
  - COV as `rmse / mean(y_true)`
  - aggregate percent error
- Added metrics:
  - ARE: mean absolute relative error in percent, only where `y_true > 1.0`.
  - `cc`: Pearson correlation coefficient calculated manually with `1e-9` added to denominator.
  - MSA: `100 * (exp(median(abs(log(y_pred / y_true)))) - 1)`, only where both true and predicted values are positive.
- Also reports tier train/test accuracy and tier gap.

Dependencies:
- Replaces `calc_metrics`; any later cell must unpack nine return values.
- Uses a threshold map that intentionally differs from Cells 5, 6, and 9.

## Cell 9: Actual-Vs-Predicted Scatter Plots

Purpose:
- Rebuilds final per-clip tier/regressor models and stores train/test predictions for actual-vs-predicted scatter plots.

Inputs:
- `clf_results`, `clip_results`, `clip_percentiles`, `expand_features`, and the nine-value `calc_metrics` from Cell 8.

Outputs or important variables:
- `best_thr_by_clip`: same threshold map as Cells 5 and 6, different from Cell 8.
- `n_estimators = 50`.
- `scatter_results`: dictionary keyed by clip. Each entry includes actual positive target values, train/test predictions, predicted low/high test masks, and train/test R2, NRMSE, COV, and aggregate percent error.

Hardcoded thresholds:
- `50: 325`
- `55: 225`
- `60: 200`
- `65: 625`
- `70: 725`
- `75: 200`
- `80: 150`
- `85: 175`
- `90: 150`
- `95: 150`
- `99: 175`

Model parameters/constants:
- All models use:
  - `n_estimators=50`
  - `max_depth=6`
  - `min_child_weight=1`
  - `colsample_bynode=0.8`
  - `random_state=42`
  - `n_jobs=-1`
- Tier SMOTE uses `SMOTE(random_state=42)` with a broad exception fallback.

Metrics calculated:
- Train/test R2.
- Train/test NRMSE.
- Train/test COV.
- Train/test aggregate percent error.
- The cell calls the nine-value `calc_metrics` from Cell 8 but ignores MAE, ARE, cc, and MSA in the scatter summary.

Plots:
- An `n_clips x 2` grid of actual-vs-predicted scatter plots.
- Left column: train predictions.
- Right column: test predictions, colored by predicted low/high tier.
- Each plot includes a red dashed perfect-fit line.

Dependencies:
- Requires the Cell 8 version of `calc_metrics`; running this cell before Cell 8 would fail because earlier `calc_metrics` returns only six values.

## Extraction Targets

Reusable `src/` modules:

- `src/data.py`
  - Load Excel data from a configurable path.
  - Validate required columns such as `VolBoth_sum`, `dist_coast`, and `dist_reservoir`.

- `src/preprocessing.py`
  - Leakage-column removal.
  - Numeric feature clipping based on skew thresholds.
  - Log transforms for `sqft`, `val_struct`, `val_cont`, and `fld_pct`.
  - `near_coast` and `near_reservoir` feature construction.
  - Dummy encoding.
  - Target clipping and stratification-label creation.

- `src/splitting.py`
  - Stratified train/test split behavior using zero/low/high labels.
  - Preservation of index alignment between features and target.

- `src/features.py`
  - Group `bldg*` and `occtype*` feature importances.
  - Select top grouped features.
  - Expand grouped feature names back into concrete columns.

- `src/metrics.py`
  - Six original metrics from Cell 4.
  - Expanded metrics from Cell 8.
  - Explicit naming for NRMSE-range, COV, aggregate percent error, ARE, Pearson correlation, and MSA.

- `src/models.py`
  - Layer-1 zero-vs-positive classifier training and prediction.
  - Layer-2 tier classifier training.
  - Low/high log-target regressor training and prediction.
  - Shared XGBoost parameter defaults.

- `src/evaluation.py`
  - Train/test prediction assembly.
  - Positive-sample extraction logic.
  - Summary DataFrame construction.

- `src/plotting.py`
  - Target distribution plots.
  - Train/test distribution plots.
  - ROC curve.
  - Metric comparison plots.
  - Learning-curve plots.
  - Grid-search heatmaps.
  - Actual-vs-predicted scatter plots.

Separate experiment scripts:

- `scripts/recreate_notebook_baseline.py`
  - Runs the Cell 2 and Cell 3 baseline setup end to end and emits validation summaries.

- `scripts/threshold_sweep.py`
  - Extracts Cell 4 behavior: clip percentile x low/high threshold sweep.

- `scripts/learning_curve_train.py`
  - Extracts Cell 5 train-only learning curve behavior.

- `scripts/learning_curve_train_test.py`
  - Extracts Cell 6 train/test learning curve behavior.

- `scripts/grid_search_depth_child.py`
  - Extracts Cell 7 grid search over `max_depth` and `min_child_weight`.

- `scripts/full_metrics_summary.py`
  - Extracts Cell 8 full metric summary behavior.

- `scripts/scatter_eval.py`
  - Extracts Cell 9 actual-vs-predicted evaluation and plotting.

Extraction notes:

- The notebook contains duplicated model-building code across Cells 4-9. During extraction, shared layer-2 training/prediction should be centralized first.
- Threshold maps differ by cell. Preserve each map in experiment-specific configuration until there is a deliberate decision to consolidate them.
- `X_test_pos` is based on layer-1 predicted positives, not actual positives. This should be preserved in a baseline recreation and documented in any later behavior change.
- Cell 8 changes the signature of `calc_metrics`. Extraction should avoid a mutable global function name by giving the two metric sets distinct names.
- No generated artifacts are currently saved by the notebook. Extracted scripts should write result tables and figures under `outputs/runs/` or `outputs/experiments/`.
