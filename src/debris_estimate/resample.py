"""owns the logic for resampling the training data to address class imbalance."""

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from debris_estimate.logger import Log

log = Log()


def apply_smote(X: pd.DataFrame, y: pd.Series):

    before_pos = int((y == 1).sum())
    before_neg = int((y == 0).sum())

    if before_pos < 2 or before_neg < 2:
        log.warn("Skipping SMOTE: not enough minority samples.")
        return X, y

    log.info("Applying SMOTE. Class distribution before: %s", {0: before_neg, 1: before_pos})

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    log.info(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()})")

    return X_resampled, y_resampled
