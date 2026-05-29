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
    zero_pos_pred: pd.Series
    zero_pos_prob: pd.Series
    tier_pred: pd.Series
    tier_prob: pd.Series
    low_pred: pd.Series
    high_pred: pd.Series
    reg_pred: pd.Series
    final_pred: pd.Series


def train_zero_pos_classifier(
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
    X_train_pos: pd.DataFrame,
    y_train_pos: pd.Series,
    threshold: float,
    params: dict = XGB_CLF_PARAMS_DEFAULT
):
    y_tier = (y_train_pos > threshold).astype(int)

    X_resampled, y_resampled = apply_smote(X_train_pos, y_tier)

    log.info("Training tier classfier...")

    model = XGBClassifier(**params)
    model.fit(X_resampled, y_resampled)

    log.info("Tier classifier training complete.")

    return model


def train_low_high_regressors(
    X_train_pos: pd.DataFrame,
    y_train_pos: pd.Series,
    threshold: float,
    params: dict = XGB_REG_PARAMS_DEFAULT
):
    low_mask = y_train_pos <= threshold
    high_mask = y_train_pos > threshold

    X_low = X_train_pos.loc[low_mask]
    y_low = np.log1p(y_train_pos.loc[low_mask])

    X_high = X_train_pos.loc[high_mask]
    y_high = np.log1p(y_train_pos.loc[high_mask])

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

    zero_pos_classifier = train_zero_pos_classifier(X_train, y_train)

    pos_mask = y_train > 0
    X_train_pos = X_train.loc[pos_mask]
    y_train_pos = y_train.loc[pos_mask]

    tier_classifier = train_tier_classifier(
        X_train_pos,
        y_train_pos,
        threshold=threshold,
    )

    low_regressor, high_regressor = train_low_high_regressors(
        X_train_pos,
        y_train_pos,
        threshold=threshold,
    )

    return zero_pos_classifier, tier_classifier, low_regressor, high_regressor


def predict_staged_model(
    X: pd.DataFrame,
    zero_pos_classifier,
    tier_classifier,
    low_regressor,
    high_regressor
) -> PredictionResults:
    # Initialize prediction series
    n = X.shape[0]
    zero_pos_pred = np.full(n, np.nan)
    zero_pos_prob = np.full(n, np.nan)
    tier_pred = np.full(n, np.nan)
    tier_prob = np.full(n, np.nan)
    low_pred = np.full(n, np.nan)
    high_pred = np.full(n, np.nan)
    reg_pred = np.full(n, np.nan)
    final_pred = np.zeros(n)

    # Predict zero vs positive
    zero_pos_pred = zero_pos_classifier.predict(X)
    zero_pos_prob = zero_pos_classifier.predict_proba(X)[:, 1]

    positive_mask = zero_pos_pred == 1

    if positive_mask.sum() > 0:
        # Predict low vs high tier, only for positive-predicted samples
        tier_pred[positive_mask] = tier_classifier.predict(X.loc[positive_mask])
        tier_prob[positive_mask] = tier_classifier.predict_proba(X.loc[positive_mask])[:, 1]

        low_mask = positive_mask & (tier_pred == 0)  # positive and low tier predicted
        high_mask = positive_mask & (tier_pred == 1) # positive and high tier predicted

        # Predict regression values only for positive, low tier samples
        if low_mask.sum() > 0:
            low_pred[low_mask] = np.expm1(low_regressor.predict(X.loc[low_mask]))
            low_pred[low_mask] = np.clip(low_pred[low_mask], 0, None)

        # Predict regression values only for positive, high tier samples
        if high_mask.sum() > 0:
            high_pred[high_mask] = np.expm1(high_regressor.predict(X.loc[high_mask]))
            high_pred[high_mask] = np.clip(high_pred[high_mask], 0, None)

        # Combine low/high predictions
        reg_pred[low_mask] = low_pred[low_mask]
        reg_pred[high_mask] = high_pred[high_mask]

        # Final prediction is zero for zero-predicted samples
        # and regression prediction for positive-predicted samples
        final_pred[positive_mask] = reg_pred[positive_mask]

    return PredictionResults(
        zero_pos_pred=zero_pos_pred,
        zero_pos_prob=zero_pos_prob,
        tier_pred=tier_pred,
        tier_prob=tier_prob,
        low_pred=low_pred,
        high_pred=high_pred,
        reg_pred=reg_pred,
        final_pred=final_pred
    )
