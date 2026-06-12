This is a historical notebook workflow, not the current packaged pipeline. See README for current behavior.

# GridH9v3 Combined Notebook Summary

This notebook builds and evaluates random forest models for predicting debris volume from the `GrideH9_v3.xlsx` dataset. The target variable is `VolBoth_sum`.

The workflow has two main modeling layers:

- Layer 1 predicts whether a grid has zero or positive debris volume.
- Layer 2 predicts the debris volume for positive-volume grids.

## Data Loading

The notebook starts by loading the Excel file with pandas and printing the dataset shape and column names. It then reloads the same file for preprocessing and keeps a copy of the raw data so some engineered features can still use original columns.

## Preprocessing

The target is stored as `y_reg_raw = VolBoth_sum`.

Potential leakage and unused columns are removed before modeling. These include raw and aggregate volume fields, bin labels, `GRID_ID`, and several building/property summary fields that the notebook excludes from the feature set.

The notebook then engineers a smaller set of hazard and distance features:

- `dist_htrack` is the minimum of `dist_htrack_M` and `dist_htrack_H`.
- `windgust` is the maximum of `windgust_M` and `windgust_H`.
- `rainfall` is the maximum of `rainfall_M` and `rainfall_H`.
- `near_coast` is `1` when `dist_coast < 3`, otherwise `0`.
- `near_reservoir` is `1` when `dist_reservoir < 5`, otherwise `0`.

Numeric predictor outliers are clipped using upper quantiles chosen by each feature's skewness. Highly skewed predictors are clipped more aggressively. The notebook logs a clipping summary showing each feature's skew, clipping threshold, and number of clipped values.

Several skewed continuous columns are transformed with `log1p`: `sqm_sum`, `val_struct_sum`, `val_cont_sum`, and `fld_pct`.

Categorical/ordinal fields are encoded as follows:

- `evac_degree`: `none=0`, `low=1`, `med=2`, `high=3`
- `fld_zone`: `X=0`, `A/AO=1`, `AE=2`, `VE=3`
- Remaining categorical fields are one-hot encoded with `pd.get_dummies`.

## Target Clipping And Splits

The notebook evaluates several target clipping percentiles: `50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99`.

For each clipping level:

1. Positive target values above the selected percentile are clipped.
2. A low/high threshold is chosen near the clipped positive-target median to balance the positive samples.
3. Samples are assigned to `zero`, `low`, or `high` strata.
4. A stratified 85/15 train/test split is created.
5. The split metadata is saved in `clip_results`.

The notebook plots target distributions by clipping level and compares zero/low/high counts across train and test splits.

## Layer 1: Zero Vs Positive Classification

Layer 1 uses the 99% clipped split, but the binary target is simply whether `VolBoth_sum > 0`.

Before training, the notebook checks for near-perfect feature-target correlations and drops any feature with absolute correlation above `0.95`.

Feature selection is based on a random forest classifier trained on the original training data. Building-type dummy columns are grouped as `bldg_type`, and occupancy-type dummy columns are grouped as `occtype` when calculating feature importance. The top 20 feature groups are selected, then expanded back to their underlying dummy columns for modeling.

The classifier is trained with SMOTE-balanced training data and these random forest settings:

- `n_estimators=50`
- `max_depth=10`
- `min_samples_leaf=10`
- `max_features=7`
- `random_state=42`

Predictions are made on the original train set and the held-out test set. The notebook evaluates accuracy, precision, recall, F1, AUC, and the confusion matrix. It also plots the ROC curve and confusion matrix.

Layer 1 results are stored in `clf_results`, including the trained classifier, selected features, classification metrics, and positive train/test subsets for each clipping level.

## Layer 2: Positive-Volume Regression

Layer 2 is an independent regression workflow for rows where `VolBoth_sum > 0`. It does not directly chain from the Layer 1 predictions during the main regression sweep.

For each target clipping percentile, the notebook creates a separate stratified 85/15 split of positive-volume samples. The stratification is based on the low/high threshold associated with that clipping level.

Regression feature selection is computed once using the 99% clipped positive training data. A random forest regressor is trained on `log1p(target)`, feature importances are grouped the same way as Layer 1, and the top 20 groups are expanded into the final regression feature list.

The regression experiment sweeps:

- target clipping percentiles from 50% through 99%
- low/high thresholds from 100 to 1475 in steps of 25

For each clipping and threshold combination:

1. Training rows are split into low and high target tiers.
2. Thresholds are skipped if either tier is too small or if the low/high split is too imbalanced.
3. A random forest tier classifier predicts whether a positive sample belongs to the low or high tier. SMOTE is used when possible.
4. Separate low-tier and high-tier random forest regressors are trained on `log1p(target)`.
5. SMOGN oversampling is attempted for each tier's regression data; if it fails, the original tier data is used.
6. Test samples are routed through the tier classifier, then predicted by the corresponding low or high regressor.
7. Predictions are transformed back with `expm1` and clipped to be non-negative.

The notebook records train and test metrics for each run: RMSE, MAE, normalized MAE, R2, normalized RMSE, coefficient of variation, aggregate percent error, tier-classifier accuracy, and low/high subgroup R2.

It then selects the best threshold for each clipping level by highest test R2, plots metric trends across clipping levels, and identifies the overall best result.

## Prediction Visualizations

For the best regression configuration, the notebook retrains the tier classifier and low/high regressors, predicts the held-out positive test set, and plots actual vs predicted distributions for:

- all positive test samples
- samples predicted as low tier
- samples predicted as high tier

The plots include R2 and MAE in their titles.

## Target Distribution Check

The notebook separately plots the distribution of positive `VolBoth_sum` values using:

- a standard histogram
- a log-scale histogram
- a boxplot

It also prints selected percentiles from 50% through 100%. This section is used to understand the target distribution and the effect of high-volume outliers.

## Fixed-Cap Regression Experiment

The final regression section repeats the Layer 2 idea with a fixed cap instead of percentile clipping:

- It keeps only positive samples with `VolBoth_sum <= 2700`.
- Samples above 2700 are excluded rather than clipped.
- It uses a stratified 85/15 split based on the median positive target value.
- It selects top regression features from this fixed-cap training set.
- It sweeps thresholds from 100 up to 2675.

For each threshold, the notebook trains a SMOTE-balanced low/high tier classifier and separate random forest regressors for the low and high tiers. Unlike the earlier regression sweep, this fixed-cap version does not use SMOGN.

It selects the best threshold by test R2, plots threshold-sweep metrics, and creates actual vs predicted distribution plots for the best fixed-cap model.

## Overall Purpose

The notebook explores how well random forest models can estimate debris volume while handling three practical issues in the data: many zero-volume rows, a skewed positive-volume target, and high-volume outliers. It uses classification to separate zero from positive cases, then tests tiered regression strategies for positive debris volumes.
