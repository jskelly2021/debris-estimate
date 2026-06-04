"""Metrics module for evaluating model performance."""

import numpy as np

from dataclasses import dataclass
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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: np.ndarray

    @classmethod
    def nan(cls):
        return cls(
            accuracy=np.nan,
            precision=np.nan,
            recall=np.nan,
            f1=np.nan,
            roc_auc=np.nan,
            pr_auc=np.nan,
            confusion_matrix=np.array([]),
        )

@dataclass
class RegressionMetrics:
    mae: float
    mse: float
    rmse: float
    r2: float
    nrmse_range: float
    nrmse_mean: float
    nrmse_std: float
    cov: float
    mape: float
    ape: float
    pcc: float
    msa: float

    @classmethod
    def nan(cls):
        return cls(
            mae=np.nan,
            mse=np.nan,
            rmse=np.nan,
            r2=np.nan,
            nrmse_range=np.nan,
            nrmse_mean=np.nan,
            nrmse_std=np.nan,
            cov=np.nan,
            mape=np.nan,
            ape=np.nan,
            pcc=np.nan,
            msa=np.nan,
        )


def _to_numpy(y):
    return np.asarray(y, dtype=float)


def _safe_divide(numerator, denominator):
    return np.nan if denominator == 0 else numerator / denominator


# Regression metrics
def mae(y_true, y_pred):
    """Mean absolute error."""
    return mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    """Mean squared error."""
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return root_mean_squared_error(y_true, y_pred)


def r2(y_true, y_pred):
    """R-squared."""
    return r2_score(y_true, y_pred)


def nrmse_range(y_true, y_pred):
    """Normalized RMSE by range."""
    y_true = _to_numpy(y_true)
    denominator = np.max(y_true) - np.min(y_true)
    return _safe_divide(rmse(y_true, y_pred), denominator)


def nrmse_mean(y_true, y_pred):
    """Normalized RMSE by mean."""
    y_true = _to_numpy(y_true)
    return _safe_divide(rmse(y_true, y_pred), np.mean(y_true))


def nrmse_std(y_true, y_pred):
    """Normalized RMSE by standard deviation."""
    y_true = _to_numpy(y_true)
    return _safe_divide(rmse(y_true, y_pred), np.std(y_true))


def cov(y_true, y_pred):
    """Covariance"""
    return np.cov(y_true, y_pred)[0, 1]


def mape(y_true, y_pred):
    """Mean absolute percentage error."""
    return mean_absolute_percentage_error(y_true, y_pred)


def ape(y_true, y_pred):
    """Aggregate percent error."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    denominator = np.sum(y_true)
    return _safe_divide(np.sum(np.abs(y_true - y_pred)), denominator) * 100


def pcc(y_true, y_pred):
    """Pearson correlation coefficient."""
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


def roc_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)


def pr_auc(y_true, y_prob):
    return average_precision_score(y_true, y_prob)


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
