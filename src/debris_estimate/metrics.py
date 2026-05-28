"""Metrics module for evaluating model performance."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)


def _to_numpy(y):
    return np.asarray(y, dtype=float)


def _safe_divide(numerator, denominator):
    return np.nan if denominator == 0 else numerator / denominator


# Regression metrics
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def nrmse_range(y_true, y_pred):
    y_true = _to_numpy(y_true)
    denominator = np.max(y_true) - np.min(y_true)
    return _safe_divide(rmse(y_true, y_pred), denominator)


def nrmse_mean(y_true, y_pred):
    y_true = _to_numpy(y_true)
    return _safe_divide(rmse(y_true, y_pred), np.mean(y_true))


def nrmse_std(y_true, y_pred):
    y_true = _to_numpy(y_true)
    return _safe_divide(rmse(y_true, y_pred), np.std(y_true))


def cov(y_true, y_pred):
    return np.cov(y_true, y_pred)[0, 1]


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


def aggregate_percent_error(y_true, y_pred):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    denominator = np.sum(y_true)
    return _safe_divide(np.sum(np.abs(y_true - y_pred)), denominator) * 100


def pearson_correlation(y_true, y_pred):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return np.corrcoef(y_true, y_pred)[0, 1]


def msa(y_true, y_pred):
    """
    Median symmetric accuracy.

    Only computed where both true and predicted values are positive.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    mask = (y_true > 0) & (y_pred > 0)

    if not np.any(mask):
        return np.nan

    log_ratio_error = np.abs(np.log(y_pred[mask] / y_true[mask]))

    return 100 * (np.exp(np.median(log_ratio_error)) - 1)


# Classification metrics
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)
