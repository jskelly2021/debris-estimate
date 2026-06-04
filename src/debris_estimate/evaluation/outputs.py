"""Module for saving model predictions and evaluation results."""

import json
import pandas as pd
import numpy as np

from pathlib import Path
from dataclasses import asdict, is_dataclass
from debris_estimate.logger import Log
from debris_estimate.config import RunConfig
from debris_estimate.evaluation.evaluation import EvaluationResults
from debris_estimate.model import PredictionResults
from debris_estimate.evaluation.plots import (
    save_confusion_plots,
    save_classification_curve_plots,
    save_actual_vs_predicted_plots,
    save_residual_plots
)

log = Log()

METRICS_FILENAME = "metrics.json"
PREDICTIONS_FILENAME = "predictions.csv"
CONFIG_FILENAME = "config.json"
PLOT_DIR = "plots"
CONFUSION_MATRIX_PLOT_DIR = "confusion"
CLASSIFICATION_CURVE_PLOT_DIR = "classification_curves"
ACTUAL_VS_PREDICTED_PLOT_DIR = "actual_vs_pred"
RESIDUAL_PLOT_DIR = "residuals"


def create_output_dir(
    output_path: str | Path,
    run_name: str | None = None
) -> Path:
    if run_name is None:
        run_name = "run_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(output_path) / run_name

    if output_dir.exists():
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def _to_serializable(obj):
    if is_dataclass(obj):
        return _to_serializable(asdict(obj))

    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}

    if isinstance(obj, list | tuple):
        return [_to_serializable(value) for value in obj]

    if isinstance(obj, pd.Series):
        return obj.tolist()

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def _save_metrics_json(
    eval: EvaluationResults,
    file_path: Path
):
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(eval), f, indent=4)


def _save_predictions_csv(
    y_true: pd.Series,
    preds: PredictionResults,
    file_path: Path
):
    pred_df = pd.DataFrame({
        "y_true": y_true,
        "final_pred": preds.final_pred,
        "zero_pos_pred": preds.zero_pos_pred,
        "zero_pos_prob": preds.zero_pos_prob,
        "tier_pred": preds.tier_pred,
        "tier_prob": preds.tier_prob,
        "low_pred": preds.low_pred,
        "high_pred": preds.high_pred,
        "reg_pred": preds.reg_pred,
    })

    pred_df.to_csv(file_path, index=True, index_label="index")


def _save_plots(
    y_true: pd.Series,
    preds: PredictionResults,
    eval: EvaluationResults,
    threshold: float,
    plots_path: Path
) -> None:
    confusion_path = plots_path / CONFUSION_MATRIX_PLOT_DIR
    classification_curve_path = plots_path / CLASSIFICATION_CURVE_PLOT_DIR
    actual_vs_pred_path = plots_path / ACTUAL_VS_PREDICTED_PLOT_DIR
    residual_path = plots_path / RESIDUAL_PLOT_DIR

    save_confusion_plots(eval, confusion_path)
    save_classification_curve_plots(y_true, preds, threshold, classification_curve_path)
    save_actual_vs_predicted_plots(y_true, preds, actual_vs_pred_path)
    save_residual_plots(y_true, preds, residual_path)


def _save_config_json(
    config: RunConfig,
    file_path: Path
) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(config), f, indent=4)


def save_run_outputs(
    y_true: pd.Series,
    preds: PredictionResults,
    eval: EvaluationResults,
    config: RunConfig,
    output_path: Path,
    save_metrics: bool = True,
    save_predictions: bool = True,
    save_plots: bool = True,
    save_config: bool = True,
):
    output_dir_path = create_output_dir(output_path, run_name=config.run_name)

    metrics_file_path = output_dir_path / METRICS_FILENAME
    predictions_file_path = output_dir_path / PREDICTIONS_FILENAME
    plots_dir_path = output_dir_path / PLOT_DIR
    config_file_path = output_dir_path / CONFIG_FILENAME

    if save_metrics:
        _save_metrics_json(eval=eval, file_path=metrics_file_path)
    if save_predictions:
        _save_predictions_csv(y_true=y_true, preds=preds, file_path=predictions_file_path)
    if save_plots:
        _save_plots(
            y_true=y_true,
            preds=preds,
            eval=eval,
            threshold=config.model.threshold,
            plots_path=plots_dir_path
        )
    if save_config:
        _save_config_json(config=config, file_path=config_file_path)
