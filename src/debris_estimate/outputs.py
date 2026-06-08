"""Module for saving model predictions and evaluation results."""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import asdict, is_dataclass
from debris_estimate.logger import Log
from debris_estimate.config import RunConfig, ExperimentConfig
from debris_estimate.evaluation.results import EvaluationResults
from debris_estimate.model import PredictionResults

log = Log()

EXPERIMENT_CONFIG_FILENAME = "experiment.json"
METRICS_FILENAME = "metrics.json"
PREDICTIONS_FILENAME = "predictions.csv"
CONFIG_FILENAME = "config.json"
PLOT_DIR = "plots"


def _create_output_dir(
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


def _to_serializable(obj) -> object:
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
    eval_results: EvaluationResults,
    file_path: Path
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(eval_results), f, indent=4)


def _save_predictions_csv(
    y_true: pd.Series,
    pred_results: PredictionResults,
    file_path: Path
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    pred_df = pd.DataFrame({
        "y_true": y_true,
        "final_pred": pred_results.final_pred,
        "zero_pos_pred": pred_results.zero_pos_pred,
        "zero_pos_prob": pred_results.zero_pos_prob,
        "tier_pred": pred_results.tier_pred,
        "tier_prob": pred_results.tier_prob,
        "low_pred": pred_results.low_pred,
        "high_pred": pred_results.high_pred,
        "reg_pred": pred_results.reg_pred,
    })

    pred_df.to_csv(file_path, index=True, index_label="index")


def _save_plots(
    figure_groups: dict[str, dict[str, plt.Figure]],
    output_dir_path: Path
) -> None:
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for group_name, figures in figure_groups.items():
        group_dir_path = output_dir_path / group_name
        group_dir_path.mkdir(parents=True, exist_ok=True)

        for name, fig in figures.items():
            fig_path = group_dir_path / f"{name}.png"
            fig.savefig(fig_path)
            plt.close(fig)


def _save_config_json(
    config: RunConfig,
    file_path: Path
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(config), f, indent=4)


def save_run_outputs(
    output_path: Path  | None = None,
    run_name: str | None = None,
    eval_results: EvaluationResults  | None = None,
    y_true: pd.Series | None = None,
    pred_results: PredictionResults | None = None,
    run_config: RunConfig  | None = None,
    figure_groups: dict[str, dict[str, plt.Figure]] | None = None,
) -> None:
    output_dir_path = _create_output_dir(output_path, run_name=run_name)

    metrics_file_path = output_dir_path / METRICS_FILENAME
    predictions_file_path = output_dir_path / PREDICTIONS_FILENAME
    config_file_path = output_dir_path / CONFIG_FILENAME
    plots_dir_path = output_dir_path / PLOT_DIR

    if eval_results is not None:
        _save_metrics_json(eval_results=eval_results, file_path=metrics_file_path)

    if pred_results is not None and y_true is not None:
        _save_predictions_csv(y_true=y_true, pred_results=pred_results, file_path=predictions_file_path)

    if figure_groups is not None:
        _save_plots(figure_groups=figure_groups, output_dir_path=plots_dir_path)

    if run_config is not None:
        _save_config_json(config=run_config, file_path=config_file_path)


def save_experiment_config(
    output_path: Path,
    experiment_config: ExperimentConfig,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / EXPERIMENT_CONFIG_FILENAME

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(experiment_config), f, indent=4)
