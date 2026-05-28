"""owns staged model training logic and prediction logic."""

import pandas as pd
import numpy as np

from dataclasses import dataclass
from xgboost import XGBClassifier, XGBRegressor
from debris_estimate.logger import Log
from debris_estimate.resample import apply_smote
from debris_estimate.split import Split

log = Log()

XGB_CLF_PARAMS_DEFAULT = dict(
    n_estimators     = 50,
    max_depth        = 10,
    min_child_weight = 10,
    colsample_bynode = 0.8,
    random_state     = 42,
    n_jobs           = -1
)

XGB_REG_PARAMS_DEFAULT = dict(
    n_estimators     = 50,
    max_depth        = 6,
    min_child_weight = 1,
    colsample_bynode = 0.8,
    random_state     = 42,
    n_jobs           = -1
)


@dataclass
class PredictionResults:
    zero_pos_y_true: pd.Series
    zero_pos_y_pred: pd.Series
    zero_pos_y_prob: pd.Series
    tier_y_true: pd.Series
    tier_y_pred: pd.Series
    tier_y_prob: pd.Series
    low_y_true: pd.Series
    low_y_pred: pd.Series
    high_y_true: pd.Series
    high_y_pred: pd.Series
    final_y_true: pd.Series
    final_y_pred: pd.Series


def train_zero_vs_positive_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = XGB_CLF_PARAMS_DEFAULT
):
    # Get binary labels for classification
    y_train = (y_train > 0).astype(int)

    X_resampled, y_resampled = apply_smote(X_train, y_train)

    log.info("Training zero vs positive classfier...")

    model = XGBClassifier(**params)
    model.fit(X_resampled, y_resampled)

    log.info("Zero vs positive classifier training complete.")

    return model


def train_tier_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float,
    params: dict = XGB_CLF_PARAMS_DEFAULT
):
    y_tier = (y_train > threshold).astype(int)

    X_resampled, y_resampled = apply_smote(X_train, y_tier)

    log.info("Training tier classfier...")

    model = XGBClassifier(**params)
    model.fit(X_resampled, y_resampled)

    log.info("Tier classifier training complete.")

    return model


def train_low_high_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float,
    params: dict = XGB_REG_PARAMS_DEFAULT
):
    low_mask = y_train <= threshold
    high_mask = y_train > threshold

    X_low = X_train.loc[low_mask]
    y_low = np.log1p(y_train.loc[low_mask])

    X_high = X_train.loc[high_mask]
    y_high = np.log1p(y_train.loc[high_mask])

    log.info("Training low regressor...")
    model_low = XGBRegressor(**params)
    model_low.fit(X_low, y_low)

    log.info("Training high regressor...")
    model_high = XGBRegressor(**params)
    model_high.fit(X_high, y_high)

    log.info("Low/high regressor training complete.")

    return model_low, model_high


def train_staged_model(
    split: Split,
    threshold: float
):
    X_train = split.X_train
    y_train = split.y_train

    zero_vs_positive_model = train_zero_vs_positive_classifier(X_train, y_train)

    pos_mask = y_train > 0
    X_train_pos = X_train.loc[pos_mask]
    y_train_pos = y_train.loc[pos_mask]

    tier_model = train_tier_classifier(
        X_train_pos,
        y_train_pos,
        threshold=threshold,
    )

    low_regressor, high_regressor = train_low_high_regressors(
        X_train_pos,
        y_train_pos,
        threshold=threshold,
    )

    return zero_vs_positive_model, tier_model, low_regressor, high_regressor


def predict_staged_model(
    X: pd.DataFrame,
    zero_vs_positive_model,
    tier_model,
    low_regressor,
    high_regressor
) -> PredictionResults:
    final_preds = np.zeros(X.shape[0])

    positive_mask = zero_vs_positive_model.predict(X) == 1

    if positive_mask.sum() == 0:
        return final_preds

    X_pos = X.loc[positive_mask]
    tier_preds = tier_model.predict(X_pos)

    low_mask = tier_preds == 0
    high_mask = tier_preds == 1

    pos_preds = np.zeros(len(X_pos))

    if low_mask.sum() > 0:
        pos_preds[low_mask] = np.expm1(low_regressor.predict(X_pos.loc[low_mask]))

    if high_mask.sum() > 0:
        pos_preds[high_mask] = np.expm1(high_regressor.predict(X_pos.loc[high_mask]))

    final_preds[positive_mask] = np.clip(pos_preds, 0, None)

    return PredictionResults(
        zero_pos_y_true=None,
        zero_pos_y_pred=None,
        zero_pos_y_prob=None,
        tier_y_true=None,
        tier_y_pred=None,
        tier_y_prob=None,
        low_y_true=None,
        low_y_pred=None,
        high_y_true=None,
        high_y_pred=None,
        final_true=None,
        final_pred=final_preds
    )
