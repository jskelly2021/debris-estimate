"""Training functions for internal models."""

import pandas as pd
import numpy as np

from xgboost import XGBClassifier, XGBRegressor
from debris_estimate.logger import Log
from debris_estimate.resample import apply_smote

log = Log()


def train_zero_pos_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict
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
    params: dict
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
    low_params: dict,
    high_params: dict,
):
    low_mask = y_train_pos <= threshold
    high_mask = y_train_pos > threshold

    if low_mask.sum() == 0:
        raise ValueError("Cannot train low regressor: no low-tier samples.")

    if high_mask.sum() == 0:
        raise ValueError("Cannot train high regressor: no high-tier samples.")

    X_low = X_train_pos.loc[low_mask]
    y_low = np.log1p(y_train_pos.loc[low_mask])

    X_high = X_train_pos.loc[high_mask]
    y_high = np.log1p(y_train_pos.loc[high_mask])

    log.info("Training low/high regressors...")

    model_low = XGBRegressor(**low_params)
    model_low.fit(X_low, y_low)

    model_high = XGBRegressor(**high_params)
    model_high.fit(X_high, y_high)

    log.info("Low/high regressor training complete.")

    return model_low, model_high
