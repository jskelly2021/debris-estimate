"""owns staged model training logic and prediction logic."""

import pandas as pd

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from debris_estimate.logger import Log

log = Log()

XGB_CLF_PARAMS_DEFAULT = dict(
    n_estimators     = 50,
    max_depth        = 10,
    min_child_weight = 10,
    colsample_bynode = 0.8,
    random_state     = 42,
    n_jobs           = -1
)


def apply_smote(X: pd.DataFrame, y: pd.Series):

    before_pos = int((y == 1).sum())
    before_neg = int((y == 0).sum())

    if before_pos < 2 or before_neg < 2:
        log.warn("Skipping SMOTE: not enough minority samples.")
        return X, y

    log.info("Applying SMOTE. Class distribution before: %s", {0: before_neg, 1: before_pos})

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    log.info(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

    return X_resampled, y_resampled


def train_zero_vs_positive_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
):
    # Get binary labels for classification
    y_train = (y_train > 0).astype(int)

    X_resampled, y_resampled = apply_smote(X_train, y_train)

    log.info("Training zero vs positive classfier...")

    model = XGBClassifier(**XGB_CLF_PARAMS_DEFAULT)
    model.fit(X_resampled, y_resampled)

    log.info("Zero vs positive classifier training complete.")

    return model
