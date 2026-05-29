"""Save model evaluation plots."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from pathlib import Path
from debris_estimate.logger import Log
from debris_estimate.evaluation import EvaluationResults
from debris_estimate.model import PredictionResults

log = Log()

PLOT_DIR = "plots"
CONFUSION_MATRIX_PLOT_DIR = "confusion"
CLASSIFICATION_CURVE_PLOT_DIR = "classification_curves"
ACTUAL_VS_PREDICTED_PLOT_DIR = "actual_vs_pred"
RESIDUAL_PLOT_DIR = "residuals"


def save_confusion_matrix(
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


def save_roc_curve(
    roc_curve: pd.DataFrame,
    output_dir: Path
):
    pass


def save_pr_curve(
    pr_curve: pd.DataFrame,
    output_dir: Path
):
    pass


def save_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
    title: str = "Actual vs Predicted"
):
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


def save_residual_plot(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_dir: Path
):
    pass


def save_all_plots(
    y_true: pd.Series,
    preds: PredictionResults,
    eval: EvaluationResults,
    output_dir: Path
):
    plots_dir = output_dir / PLOT_DIR

    # Zero vs Positive Confusion Matrix
    save_confusion_matrix(
        confusion_matrix=eval.zero_pos_classifier_metrics.confusion_matrix,
        output_path=plots_dir / CONFUSION_MATRIX_PLOT_DIR / "zero_pos_confusion.png",
        labels=["Zero", "Positive"],
        title="Zero vs Positive Confusion Matrix"
    )

    # Tier Confusion Matrix
    save_confusion_matrix(
        confusion_matrix=eval.tier_classifier_metrics.confusion_matrix,
        output_path=plots_dir / CONFUSION_MATRIX_PLOT_DIR / "tier_confusion.png",
        labels=["Low", "High"],
        title="Tier Confusion Matrix"
    )

    # System Actual vs Predicted
    save_actual_vs_predicted(
        y_true=y_true,
        y_pred=preds.final_pred,
        output_path=plots_dir / ACTUAL_VS_PREDICTED_PLOT_DIR / "system_actual_vs_pred.png",
        title="System Actual vs Predicted"
    )
