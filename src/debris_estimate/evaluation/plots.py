"""Save model evaluation plots."""

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from debris_estimate.evaluation.results import EvaluationResults
from debris_estimate.model import PredictionResults, FeatureImportanceResults

CONFUSION_MATRIX = "confusion_matrix"
CLASSIFICATION_CURVES = "classification_curves"
ACTUAL_VS_PREDICTED = "actual_vs_predicted"
RESIDUAL = "residual"
FEATURE_IMPORTANCE = "feature_importance"


def _create_confusion_matrix(
    confusion_matrix: pd.DataFrame,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=labels,
    )

    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)

    fig.tight_layout()
    return fig


def _create_roc_curve(
    y_true: pd.Series,
    y_prob: pd.Series,
    title: str = "ROC Curve",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
        ax=ax,
    )

    ax.set_title(title)

    fig.tight_layout()
    return fig


def _create_pr_curve(
    y_true: pd.Series,
    y_prob: pd.Series,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_prob,
        ax=ax,
    )

    ax.set_title(title)

    fig.tight_layout()
    return fig


def _create_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Actual vs Predicted"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.6)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    fig.tight_layout()
    return fig


def _create_residual(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Residual Plot",
) -> plt.Figure:
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")

    fig.tight_layout()
    return fig


def _create_feature_importance(
    df: pd.DataFrame,
    title: str = "Feature Importance Plot",
    top_n: int = 25,
) -> plt.Figure:
    df = (
        df.sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    fig_height = max(5, len(df) * 0.3)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    ax.barh(df["feature"], df["importance"])

    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    return fig


def _create_confusion_plots(
    eval: EvaluationResults,
) -> dict[str, plt.Figure]:
    figs = {}

    # Zero vs Positive Confusion Matrix
    figs["zero_pos"] = _create_confusion_matrix(
        confusion_matrix=eval.zero_pos.confusion_matrix,
        labels=["Zero", "Positive"],
        title="Zero vs Positive Confusion Matrix"
    )

    # Tier Confusion Matrix
    cm = eval.tier.confusion_matrix
    if cm is not None and cm.size > 0:
        figs["tier"] = _create_confusion_matrix(
            confusion_matrix=cm,
            labels=["Low", "High"],
            title="Tier Confusion Matrix"
        )

    return figs


def _create_classification_curve_plots(
    y_true: pd.Series,
    pred_results: PredictionResults,
    threshold: float,
) -> dict[str, plt.Figure]:
    figs = {}

    # Zero vs Positive ROC Curve
    y_true_pos = (y_true > 0).astype(int)
    if pred_results.zero_pos_prob is not None and y_true_pos.nunique() >= 2:
        figs["zero_pos_roc"] = _create_roc_curve(
            y_true=y_true_pos,
            y_prob=pred_results.zero_pos_prob,
            title="Zero vs Positive ROC Curve",
        )

        # Zero vs Positive Precision-Recall Curve
        figs["zero_pos_pr"] = _create_pr_curve(
            y_true=y_true_pos,
            y_prob=pred_results.zero_pos_prob,
            title="Zero vs Positive Precision-Recall Curve",
        )

    # Tier ROC Curve
    y_tier_true, _, y_tier_prob = pred_results.tier_pairs(y_true)

    if y_tier_prob is not None and not y_tier_true.empty and y_tier_true.nunique() >= 2:
        tier_true = (y_tier_true > threshold).astype(int)

        figs["tier_roc"] = _create_roc_curve(
            y_true=tier_true,
            y_prob=y_tier_prob,
            title="Tier ROC Curve",
        )

        # Tier Precision-Recall Curve
        figs["tier_pr"] = _create_pr_curve(
            y_true=tier_true,
            y_prob=y_tier_prob,
            title="Tier Precision-Recall Curve",
        )

    return figs


def _create_actual_vs_predicted_plots(
    y_true: pd.Series,
    pred_results: PredictionResults,
) -> dict[str, plt.Figure]:
    figs = {}

    y_low_true, y_low_pred = pred_results.low_pairs(y_true)
    y_high_true, y_high_pred = pred_results.high_pairs(y_true)
    y_reg_true, y_reg_pred = pred_results.reg_pairs(y_true)
    
    # System Actual vs Predicted
    figs["system"] = _create_actual_vs_predicted(
        y_true=y_true,
        y_pred=pred_results.final_pred,
        title="System Actual vs Predicted"
    )

    # Low Tier Actual vs Predicted
    figs["low"] = _create_actual_vs_predicted(
        y_true=y_low_true,
        y_pred=y_low_pred,
        title="Low Tier Actual vs Predicted"
    )

    # High Tier Actual vs Predicted
    figs["high"] = _create_actual_vs_predicted(
        y_true=y_high_true,
        y_pred=y_high_pred,
        title="High Tier Actual vs Predicted"
    )

    # Full Regression Actual vs Predicted
    figs["reg"] = _create_actual_vs_predicted(
        y_true=y_reg_true,
        y_pred=y_reg_pred,
        title="Full Regression Actual vs Predicted"
    )

    return figs


def _create_residual_plots(
    y_true: pd.Series,
    pred_results: PredictionResults,
) -> dict[str, plt.Figure]:
    figs = {}

    y_low_true, y_low_pred = pred_results.low_pairs(y_true)
    y_high_true, y_high_pred = pred_results.high_pairs(y_true)
    y_reg_true, y_reg_pred = pred_results.reg_pairs(y_true)

    # System Residual Plot
    figs["system"] = _create_residual(
        y_true=y_true,
        y_pred=pred_results.final_pred,
        title="System Residual Plot"
    )

    # Low Tier Residual Plot
    figs["low"] = _create_residual(
        y_true=y_low_true,
        y_pred=y_low_pred,
        title="Low Tier Residual Plot"
    )

    # High Tier Residual Plot
    figs["high"] = _create_residual(
        y_true=y_high_true,
        y_pred=y_high_pred,
        title="High Tier Residual Plot"
    )

    # Full Regression Residual Plot
    figs["reg"] = _create_residual(
        y_true=y_reg_true,
        y_pred=y_reg_pred,
        title="Full Regression Residual Plot"
    )

    return figs


def _create_feature_importance_plots(
    feature_importance_results: FeatureImportanceResults,
) -> dict[str, plt.Figure]:
    return {
        "zero_pos": _create_feature_importance(
            feature_importance_results.zero_pos,
            "Zero vs Positive Feature Importance",
        ),
        "tier": _create_feature_importance(
            feature_importance_results.tier,
            "Tier Classifier Feature Importance",
        ),
        "low": _create_feature_importance(
            feature_importance_results.low,
            "Low Regressor Feature Importance",
        ),
        "high": _create_feature_importance(
            feature_importance_results.high,
            "High Regressor Feature Importance",
        ),
    }


def create_evaluation_figures(
    y_true: pd.Series,
    pred_results: PredictionResults | None = None,
    eval_results: EvaluationResults | None = None,
    feature_importance_results: FeatureImportanceResults | None = None,
    threshold: float | None = 1.0,
) -> dict[str, dict[str, plt.Figure]]:
    figure_groups: dict[str, dict[str, plt.Figure]] = {}

    if eval_results is not None:
        figure_groups[CONFUSION_MATRIX] = _create_confusion_plots(eval_results)

    if pred_results is not None:
        figure_groups[ACTUAL_VS_PREDICTED] = _create_actual_vs_predicted_plots(y_true, pred_results)
        figure_groups[RESIDUAL] = _create_residual_plots(y_true, pred_results)

        if threshold is not None:
            figure_groups[CLASSIFICATION_CURVES] = _create_classification_curve_plots(y_true, pred_results, threshold)

    if feature_importance_results is not None:
        figure_groups[FEATURE_IMPORTANCE] = _create_feature_importance_plots(feature_importance_results)

    return figure_groups
