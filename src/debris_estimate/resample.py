"""owns the logic for resampling the training data to address class imbalance."""

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from debris_estimate.logger import Log

log = Log()


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    k_neighbors: int = 5
):
    counts = y.value_counts()

    if len(counts) < 2:
        log.warn(
            "Skipping SMOTE: only one class present: %s",
            counts.to_dict()
        )
        return X, y

    min_count = counts.min()

    if min_count <= k_neighbors:
        log.warn(
            "Skipping SMOTE: minority class has %s samples, need at least %s.",
            min_count,
            k_neighbors + 1
        )
        return X, y

    log.info("Applying SMOTE. Class distribution before: %s", counts.to_dict())

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    log.info("Resampled class distribution: %s", pd.Series(y_resampled).value_counts().to_dict())

    return X_resampled, y_resampled
