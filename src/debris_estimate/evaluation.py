"""owns model evaluation logic."""

import pandas as pd
import numpy as np
import debris_estimate.metrics as metrics

from dataclasses import dataclass
from debris_estimate.logger import Log
from debris_estimate.metrics import ClassificationMetrics, RegressionMetrics
from debris_estimate.model import PredictionResults

log = Log()


@dataclass
class EvaluationResults:
    system_metrics: RegressionMetrics
    zero_pos_classifier_metrics: ClassificationMetrics
    tier_classifier_metrics: ClassificationMetrics
    low_regressor_metrics: RegressionMetrics
    high_regressor_metrics: RegressionMetrics
    full_regressor_metrics: RegressionMetrics


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series
) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=metrics.accuracy(y_true, y_pred),
        precision=metrics.precision(y_true, y_pred),
        recall=metrics.recall(y_true, y_pred),
        f1=metrics.f1(y_true, y_pred),
        roc_auc=metrics.roc_auc(y_true, y_prob),
        pr_auc=metrics.pr_auc(y_true, y_prob),
        confusion_matrix=metrics.confusion(y_true, y_pred),
    )


def evaluate_regressor(
    y_true: pd.Series,
    y_pred: pd.Series
) -> RegressionMetrics:
    return RegressionMetrics(
        mae=metrics.mae(y_true, y_pred),
        mse=metrics.mse(y_true, y_pred),
        rmse=metrics.rmse(y_true, y_pred),
        r2=metrics.r2(y_true, y_pred),
        nrmse_range=metrics.nrmse_range(y_true, y_pred),
        nrmse_mean=metrics.nrmse_mean(y_true, y_pred),
        nrmse_std=metrics.nrmse_std(y_true, y_pred),
        cov=metrics.cov(y_true, y_pred),
        mape=metrics.mape(y_true, y_pred),
        ape=metrics.ape(y_true, y_pred),
        pcc=metrics.pcc(y_true, y_pred),
        msa=metrics.msa(y_true, y_pred),
    )


def evaluate_system(
    y_true: pd.Series,
    preds: PredictionResults,
    threshold: float
):
    system_metrics = evaluate_regressor(y_true, preds.final_pred)

    zero_pos_metrics = evaluate_classifier(
        (y_true > 0).astype(int),
        preds.zero_pos_pred,
        preds.zero_pos_prob
    )

    positive_mask = preds.zero_pos_pred == 1

    tier_metrics = evaluate_classifier(
        (y_true > threshold).astype(int)[positive_mask],
        preds.tier_pred[positive_mask],
        preds.tier_prob[positive_mask]
    )

    low_mask = positive_mask & (preds.tier_pred == 0)
    high_mask = positive_mask & (preds.tier_pred == 1)
    reg_mask = low_mask | high_mask

    low_metrics = evaluate_regressor(
        y_true[low_mask],
        preds.low_pred[low_mask]
    )

    high_metrics = evaluate_regressor(
        y_true[high_mask],
        preds.high_pred[high_mask]
    )

    full_metrics = evaluate_regressor(
        y_true[reg_mask],
        preds.reg_pred[reg_mask]
    )

    return EvaluationResults(
        system_metrics=system_metrics,
        zero_pos_classifier_metrics=zero_pos_metrics,
        tier_classifier_metrics=tier_metrics,
        low_regressor_metrics=low_metrics,
        high_regressor_metrics=high_metrics,
        full_regressor_metrics=full_metrics
    )
