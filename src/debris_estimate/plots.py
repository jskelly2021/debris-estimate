"""Save model evaluation plots."""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from pathlib import Path
from debris_estimate.evaluation import EvaluationResults
from debris_estimate.model import PredictionResults


def _save_confusion_matrix(
    confusion_matrix: pd.DataFrame,
    output_path: Path,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix"
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=labels,
    )

    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_roc_curve(
    y_true: pd.Series,
    y_prob: pd.Series,
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
        ax=ax,
    )

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_pr_curve(
    y_true: pd.Series,
    y_prob: pd.Series,
    output_path: Path,
    title: str = "Precision-Recall Curve",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_prob,
        ax=ax,
    )

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
    title: str = "Actual vs Predicted"
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.6)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_residual(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
    title: str = "Residual Plot",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_plots(eval: EvaluationResults, confusion_path: Path) -> None:
    # Zero vs Positive Confusion Matrix
    _save_confusion_matrix(
        confusion_matrix=eval.zero_pos_classifier_metrics.confusion_matrix,
        output_path=confusion_path / "zero_pos_confusion.png",
        labels=["Zero", "Positive"],
        title="Zero vs Positive Confusion Matrix"
    )

    # Tier Confusion Matrix
    _save_confusion_matrix(
        confusion_matrix=eval.tier_classifier_metrics.confusion_matrix,
        output_path=confusion_path / "tier_confusion.png",
        labels=["Low", "High"],
        title="Tier Confusion Matrix"
    )


def save_classification_curve_plots(y_true: pd.Series, preds: PredictionResults, threshold: float, classification_curve_path: Path) -> None:
    y_tier_true, _, y_tier_prob = preds.tier_pairs(y_true)

    # Zero vs Positive ROC Curve
    _save_roc_curve(
        y_true=(y_true > 0).astype(int),
        y_prob=preds.zero_pos_prob,
        output_path=classification_curve_path / "zero_pos_roc.png",
        title="Zero vs Positive ROC Curve",
    )

    # Tier ROC Curve
    _save_roc_curve(
        y_true=(y_tier_true > threshold).astype(int),
        y_prob=y_tier_prob,
        output_path=classification_curve_path / "tier_roc.png",
        title="Tier ROC Curve",
    )

    # Zero vs Positive Precision-Recall Curve
    _save_pr_curve(
        y_true=(y_true > 0).astype(int),
        y_prob=preds.zero_pos_prob,
        output_path=classification_curve_path / "zero_pos_pr.png",
        title="Zero vs Positive Precision-Recall Curve",
    )

    # Tier Precision-Recall Curve
    _save_pr_curve(
        y_true=(y_tier_true > threshold).astype(int),
        y_prob=y_tier_prob,
        output_path=classification_curve_path / "tier_pr.png",
        title="Tier Precision-Recall Curve",
    )


def save_actual_vs_predicted_plots(y_true: pd.Series, preds: PredictionResults, actual_vs_pred_path: Path) -> None:
    y_low_true, y_low_pred = preds.low_pairs(y_true)
    y_high_true, y_high_pred = preds.high_pairs(y_true)
    y_reg_true, y_reg_pred = preds.reg_pairs(y_true)
    
    # System Actual vs Predicted
    _save_actual_vs_predicted(
        y_true=y_true,
        y_pred=preds.final_pred,
        output_path=actual_vs_pred_path / "system_actual_vs_pred.png",
        title="System Actual vs Predicted"
    )

    # Low Tier Actual vs Predicted
    _save_actual_vs_predicted(
        y_true=y_low_true,
        y_pred=y_low_pred,
        output_path=actual_vs_pred_path / "low_actual_vs_pred.png",
        title="Low Tier Actual vs Predicted"
    )

    # High Tier Actual vs Predicted
    _save_actual_vs_predicted(
        y_true=y_high_true,
        y_pred=y_high_pred,
        output_path=actual_vs_pred_path / "high_actual_vs_pred.png",
        title="High Tier Actual vs Predicted"
    )

    # Full Regression Actual vs Predicted
    _save_actual_vs_predicted(
        y_true=y_reg_true,
        y_pred=y_reg_pred,
        output_path=actual_vs_pred_path / "reg_actual_vs_pred.png",
        title="Full Regression Actual vs Predicted"
    )


def save_residual_plots(y_true: pd.Series, preds: PredictionResults, residual_path: Path) -> None:
    y_low_true, y_low_pred = preds.low_pairs(y_true)
    y_high_true, y_high_pred = preds.high_pairs(y_true)
    y_reg_true, y_reg_pred = preds.reg_pairs(y_true)

    # System Residual Plot
    _save_residual(
        y_true=y_true,
        y_pred=preds.final_pred,
        output_path=residual_path / "system_residual.png",
        title="System Residual Plot"
    )

    # Low Tier Residual Plot
    _save_residual(
        y_true=y_low_true,
        y_pred=y_low_pred,
        output_path=residual_path / "low_residual.png",
        title="Low Tier Residual Plot"
    )

    # High Tier Residual Plot
    _save_residual(
        y_true=y_high_true,
        y_pred=y_high_pred,
        output_path=residual_path / "high_residual.png",
        title="High Tier Residual Plot"
    )

    # Full Regression Residual Plot
    _save_residual(
        y_true=y_reg_true,
        y_pred=y_reg_pred,
        output_path=residual_path / "reg_residual.png",
        title="Full Regression Residual Plot"
    )
