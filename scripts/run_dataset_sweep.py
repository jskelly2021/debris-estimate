"""Dataset sweep for staged_model."""

import pandas as pd

from pathlib import Path
from debris_estimate.logger import setup_logger, Log
from debris_estimate.config import ExperimentConfig, RunConfig
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

from config_presets import ALL_PRESETS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "dataset_sweep"
RUN_OUTPUT_DIR = "runs"
ANALYSIS_OUTPUT_DIR = "analysis"
OUTPUT_PATH = PROJECT_ROOT / OUTPUT_DIR / EXPERIMENT_NAME
RUNS_OUTPUT_PATH = OUTPUT_PATH / RUN_OUTPUT_DIR
ANALYSIS_OUTPUT_PATH = OUTPUT_PATH / ANALYSIS_OUTPUT_DIR

setup_logger()
log = Log()

DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig(
    experiment_name=EXPERIMENT_NAME,
    primary_metric="system_r2",
    primary_metric_mode="max",
    swept_fields=["data.dataset"],
)


def is_valid_threshold(y_train: pd.Series, threshold: float, min_samples: int = 5) -> bool:
    y_pos = y_train[y_train > 0]

    low_count = int((y_pos <= threshold).sum())
    high_count = int((y_pos > threshold).sum())

    return low_count >= min_samples and high_count >= min_samples


def run_once(X: pd.Dataframe, y: pd.Series, run_config: RunConfig):
        ### Splitting ###
        X_train, X_test, y_train, y_test = split_data(
            X=X,
            y=y,
            config=run_config.data.split,
        )

        ### Clipping ###
        X_train_clipped, X_test_clipped = clip_features(
            X_train=X_train,
            X_test=X_test,
            exclude_cols=run_config.data.preprocess.exclude_clip_cols,
            config=run_config.data.clip,
        )

        y_train_clipped, _, _, _ = clip_targets(
            y=y_train,
            config=run_config.data.clip,
        )

        if not is_valid_threshold(y_train_clipped, run_config.model.threshold, min_samples=5):
            raise ValueError(
                f"Skipping | threshold={run_config.model.threshold} | "\
                f"feature clip={run_config.data.clip.feature_clip_percentile} | "\
                f"target clip={run_config.data.clip.target_clip_percentile} | "\
                f"not enough low/high samples."
            )

        ### Training ###
        staged_model = StagedModel(config=run_config.model)
        staged_model.fit(X_train=X_train_clipped, y_train=y_train_clipped)

        ### Prediction ###
        pred_results = staged_model.predict_details(X=X_test_clipped)

        ### Evaluation ###
        eval_results = evaluate_staged_model(
            y_true=y_test,
            pred_results=pred_results,
            threshold=run_config.model.threshold,
        )

        figure_groups = create_evaluation_figures(
            y_true=y_test,
            pred_results=pred_results,
            eval_results=eval_results,
            threshold=run_config.model.threshold,
        )

        ### Output ###
        save_run_outputs(
            output_path=RUNS_OUTPUT_PATH,
            run_name=run_config.run_name,
            eval_results=eval_results,
            y_true=y_test,
            pred_results=pred_results,
            run_config=run_config,
            figure_groups=figure_groups,
        )


def run_dataset_sweep(
    experiment_config: ExperimentConfig | None = DEFAULT_EXPERIMENT_CONFIG,
):
    ### Dataset Sweep ###
    for config in ALL_PRESETS:
        run_config = config.build_run_config()
        data_path = PROJECT_ROOT / run_config.data.dataset
        df = load_dataset(path=data_path)

        ### Preprocessing ###
        X = preprocess_features(
            df=df,
            config=run_config.data.preprocess,
        )

        y = df[run_config.data.preprocess.target_col]

        try:
            run_once(X=X, y=y, run_config=run_config)
        except Exception as e:
            log.error(f"{run_config.run_name}: {e}")
            continue

    save_experiment_config(
        output_path=OUTPUT_PATH,
        experiment_config=experiment_config
    )

    analyze_sweep(experiment_path=OUTPUT_PATH)


def main() -> int:
    try:
        run_dataset_sweep()
        log.info(f"{EXPERIMENT_NAME} completed successfully.")
        return 0

    except Exception as e:
        log.error(f"{EXPERIMENT_NAME} failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
