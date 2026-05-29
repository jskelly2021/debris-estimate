"""Module for saving model predictions and evaluation results."""

import json
import pandas as pd
import numpy as np

from pathlib import Path
from dataclasses import asdict, is_dataclass
from debris_estimate.logger import Log
from debris_estimate.evaluation import EvaluationResults
from debris_estimate.model import PredictionResults

log = Log()

METRICS_FILENAME = "metrics.json"
PREDICTIONS_FILENAME = "predictions.csv"


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


def create_output_dir(
    base_dir: str | Path,
    run_name: str | None = None
) -> Path:
    if run_name is None:
        run_name = "run_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(base_dir) / run_name

    if output_dir.exists():
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def save_metrics_json(
    eval: EvaluationResults,
    output_dir: Path
):
    output_path = output_dir / METRICS_FILENAME

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(eval), f, indent=4)


def save_predictions(
    y_true: pd.Series,
    preds: PredictionResults,
    output_dir: Path
):
    output_path = output_dir / PREDICTIONS_FILENAME

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

    pred_df.to_csv(output_path, index=True, index_label="index")

    return output_path


def save_run_outputs(
    eval: EvaluationResults,
    preds: PredictionResults,
    y_true: pd.Series,
    output_dir: Path,
    write_predictions: bool = True
):
    save_metrics_json(eval, output_dir)
    if write_predictions:
        save_predictions(y_true, preds, output_dir)
