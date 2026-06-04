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
    if len(y_true) == 0:
        return ClassificationMetrics.nan()

    unique_classes = pd.Series(y_true).dropna().unique()

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    cm = metrics.confusion_matrix(y_true, y_pred)

    if len(unique_classes) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
        pr_auc = metrics.average_precision_score(y_true, y_prob)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm,
    )


def evaluate_regressor(
    y_true: pd.Series,
    y_pred: pd.Series
) -> RegressionMetrics:
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    valid_mask = y_true.notna() & y_pred.notna()
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return RegressionMetrics.nan()

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
    y_tier_true, y_tier_pred, y_tier_prob = preds.tier_pairs(y_true)
    y_low_true, y_low_pred = preds.low_pairs(y_true)
    y_high_true, y_high_pred = preds.high_pairs(y_true)
    y_reg_true, y_reg_pred = preds.reg_pairs(y_true)

    system_metrics = evaluate_regressor(
        y_true=y_true,
        y_pred=preds.final_pred
    )

    zero_pos_metrics = evaluate_classifier(
        y_true=(y_true > 0).astype(int),
        y_pred=preds.zero_pos_pred,
        y_prob=preds.zero_pos_prob
    )

    tier_metrics = evaluate_classifier(
        y_true=(y_tier_true > threshold).astype(int),
        y_pred=y_tier_pred,
        y_prob=y_tier_prob
    )

    low_metrics = evaluate_regressor(
        y_true=y_low_true,
        y_pred=y_low_pred
    )

    high_metrics = evaluate_regressor(
        y_true=y_high_true,
        y_pred=y_high_pred
    )

    full_metrics = evaluate_regressor(
        y_true=y_reg_true,
        y_pred=y_reg_pred
    )

    return EvaluationResults(
        system_metrics=system_metrics,
        zero_pos_classifier_metrics=zero_pos_metrics,
        tier_classifier_metrics=tier_metrics,
        low_regressor_metrics=low_metrics,
        high_regressor_metrics=high_metrics,
        full_regressor_metrics=full_metrics
    )
