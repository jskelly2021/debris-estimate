"""owns staged model training logic and prediction logic."""

import pandas as pd
import numpy as np

from xgboost import XGBClassifier, XGBRegressor
from debris_estimate.logger import Log
from debris_estimate.resample import apply_smote

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
    params: dict = XGB_CLF_PARAMS_DEFAULT
):
    X_resampled, y_resampled = apply_smote(X_train, y_train)

    log.info("Training tier classfier...")

    model = XGBClassifier(**params)
    model.fit(X_resampled, y_resampled)

    log.info("Tier classifier training complete.")

    return model


def train_low_high_regressors(
    X_train: pd.DataFrame,
    y_train_low: pd.Series,
    y_train_high: pd.Series,
    params: dict = XGB_REG_PARAMS_DEFAULT
):
    y_train_log_low = np.log1p(y_train_low)
    y_train_log_high = np.log1p(y_train_high)

    log.info("Training low regressor...")

    model_low = XGBRegressor(**params)
    model_low.fit(X_train, y_train_log_low)

    log.info("Low regressor training complete.")

    log.info("Training high regressor...")

    model_high = XGBRegressor(**params)
    model_high.fit(X_train, y_train_log_high)

    log.info("High regressor training complete.")

    return model_low, model_high
