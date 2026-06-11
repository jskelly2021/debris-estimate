"""Threshold sweep for staged_model."""

import pandas as pd

from pathlib import Path
from copy import deepcopy
from debris_estimate.logger import setup_logger, Log
from debris_estimate.config import RunConfig, ExperimentConfig
from debris_estimate.data import (
    load_dataset,
    preprocess_features,
    split_data,
    clip_features,
    clip_targets,
)
from debris_estimate.model import StagedModel
from debris_estimate.evaluation import create_evaluation_figures, evaluate_staged_model
from debris_estimate.outputs import save_run_outputs, save_experiment_config
from debris_estimate.sweep import analyze_sweep
from presets.baseline import (
    H9_V6_RUN_CONFIG,
    H8_V3_RUN_CONFIG,
    H9_STP_V3_RUN_CONFIG,
    GH9_V3_RUN_CONFIG,
    GH9_STP_V3_RUN_CONFIG,
    GH8_V3_RUN_CONFIG,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs"
RUN_OUTPUT_DIR = "runs"
ANALYSIS_OUTPUT_DIR = "analysis"
EXPERIMENT_NAME = "threshold_sweep"

DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig(
    experiment_name=EXPERIMENT_NAME,
    primary_metric="system_r2",
    primary_metric_mode="max",
    swept_fields=["model.threshold"],
)


DEFAULT_THRESHOLDS = [100, 200, 300, 400, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000]


setup_logger(verbose=False)
log = Log()


def is_valid_threshold(y_train: pd.Series, threshold: float, min_samples: int = 5) -> bool:
    y_pos = y_train[y_train > 0]

    low_count = int((y_pos <= threshold).sum())
    high_count = int((y_pos > threshold).sum())

    return low_count >= min_samples and high_count >= min_samples


def run_threshold_sweep(
    base_config: RunConfig | None = GH8_V3_RUN_CONFIG,
    experiment_config: ExperimentConfig | None = DEFAULT_EXPERIMENT_CONFIG,
    thresholds: list[int] | None = DEFAULT_THRESHOLDS,
    output_dir: str | Path = OUTPUT_DIR,
):
    output_path = PROJECT_ROOT / output_dir / experiment_config.experiment_name
    runs_output_path = output_path / RUN_OUTPUT_DIR

    data_path = PROJECT_ROOT / base_config.data.dataset
    df = load_dataset(path=data_path)

    ### Preprocessing ###
    X = preprocess_features(
        df=df,
        config=base_config.data.preprocess,
    )

    ### Splitting ###
    y = df["VolBoth_sum"]

    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        config=base_config.data.split,
    )

    ### Clipping ###
    X_train_clipped, X_test_clipped = clip_features(
        X_train=X_train,
        X_test=X_test,
        exclude_cols=base_config.data.preprocess.exclude_clip_cols,
        config=base_config.data.clip,
    )

    y_train_clipped, _, _, _ = clip_targets(
        y=y_train,
        config=base_config.data.clip,
    )

    ### Threshold Sweep ###
    for threshold in thresholds:
        if not is_valid_threshold(y_train_clipped, threshold, min_samples=5):
            log.warn("Skipping threshold=%d: not enough low/high samples.", threshold)
            continue

        log.info("Training theshold=%d", threshold)

        try:
            config = deepcopy(base_config)
            config.run_name = f"threshold_{threshold}"
            config.model.threshold = threshold

            ### Training ###
            staged_model = StagedModel(config=config.model)
            staged_model.fit(X_train=X_train_clipped, y_train=y_train_clipped)

            ### Prediction ###
            pred_results = staged_model.predict_details(X=X_test_clipped)

            ### Evaluation ###
            eval_results = evaluate_staged_model(
                y_true=y_test,
                pred_results=pred_results,
                threshold=config.model.threshold,
            )

            figure_groups = create_evaluation_figures(
                y_true=y_test,
                pred_results=pred_results,
                eval_results=eval_results,
                threshold=config.model.threshold,
            )

            ### Output ###
            save_run_outputs(
                output_path=runs_output_path,
                run_name=config.run_name,
                eval_results=eval_results,
                y_true=y_test,
                pred_results=pred_results,
                run_config=config,
                figure_groups=figure_groups,
            )

        except ValueError as e:
            log.warn("Skipping threshold=%d: %s", threshold, e)
            continue

    log.info(f"outputting to {output_path}")

    save_experiment_config(
        output_path=output_path,
        experiment_config=experiment_config
    )

    analyze_sweep(experiment_path=output_path)


def main() -> int:
    try:
        run_threshold_sweep()
        log.info(f"{EXPERIMENT_NAME} completed successfully.")
        return 0

    except Exception as e:
        log.error(f"{EXPERIMENT_NAME} failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
