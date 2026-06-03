"""Feature and target clipping module"""

import pandas as pd
import numpy as np

from dataclasses import dataclass
from debris_estimate.logger import Log

log = Log()


def _is_binary_col(s: pd.Series) -> bool:
    values = s.dropna().unique()
    return len(values) <= 2 and set(values).issubset({0, 1})


def fit_numeric_feature_clip_caps(
    X_train: pd.DataFrame,
    percentile: float,
    exclude_cols: list[str] | None = None,
) -> dict[str, float]:
    if not 0 < percentile <= 1:
        raise ValueError("feature clip percentile must be between 0 and 1.")

    exclude_cols = exclude_cols or []
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    return {
        col: X_train[col].quantile(percentile)
        for col in numeric_cols
        if col not in exclude_cols
        and not _is_binary_col(X_train[col])
    }


def apply_numeric_feature_clip_caps(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    caps: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col, upper in caps.items():
        if col in X_train.columns:
            X_train[col] = np.clip(X_train[col], 0, upper)

        if col in X_test.columns:
            X_test[col] = np.clip(X_test[col], 0, upper)

    return X_train, X_test


def clip_target(
    y: pd.Series,
    percentile: float,
    positive_only: bool = True,
) -> tuple[pd.Series, float, int, float]:
    if not 0 < percentile <= 1:
        raise ValueError("target clip percentile must be between 0 and 1.")

    y_clipped = y.copy()
    upper = float("nan")
    n_clipped = 0
    percent_clipped = 0.0

    quantile_values = y[y > 0] if positive_only else y

    if not quantile_values.empty:
        log.info(f"Clipping training targets at {percentile:.3f}")
        upper = quantile_values.quantile(percentile)

        clip_mask = y > upper
        n_clipped = int(clip_mask.sum())
        percent_clipped = n_clipped / len(y) * 100

        y_clipped = y_clipped.clip(upper=upper)

    return y_clipped, upper, n_clipped, percent_clipped
