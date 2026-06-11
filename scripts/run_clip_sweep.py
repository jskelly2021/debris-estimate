"""Feature/Target clip sweep for staged_model."""

import pandas as pd

from pathlib import Path
from copy import deepcopy
from debris_estimate.logger import setup_logger, Log
from debris_estimate.config import RunConfig, ExperimentConfig
from config_presets.baseline import (
    H9_V6_DATA_CONFIG,
    BASELINE_MODEL_CONFIG,
)
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "clip_sweep"
RUN_OUTPUT_DIR = "runs"
ANALYSIS_OUTPUT_DIR = "analysis"

setup_logger()
log = Log()

DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig(
    experiment_name=EXPERIMENT_NAME,
    primary_metric="system_r2",
    primary_metric_mode="max",
    swept_fields=["data.clip.feature_clip_percentile", "data.clip.target_clip_percentile"],
)

DEFAULT_RUN_CONFIG = RunConfig(
    run_name="base",
    data=H9_V6_DATA_CONFIG,
    model=BASELINE_MODEL_CONFIG,
)

clips = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 1.0]

setup_logger(verbose=False)
log = Log()


def is_valid_threshold(y_train: pd.Series, threshold: float, min_samples: int = 5) -> bool:
    y_pos = y_train[y_train > 0]

    low_count = int((y_pos <= threshold).sum())
    high_count = int((y_pos > threshold).sum())

    return low_count >= min_samples and high_count >= min_samples


def run_clip_sweep(
    base_config: RunConfig | None = DEFAULT_RUN_CONFIG,
    experiment_config: ExperimentConfig | None = DEFAULT_EXPERIMENT_CONFIG,
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

    ### Feature/Target Clip Sweep ###
    for feature_clip in clips:
        for target_clip in clips:
            ### Clipping ###
            log.info(f"Training feature clip={feature_clip} target clip={target_clip}")
            config = deepcopy(base_config)
            config.run_name = f"clip_{feature_clip:.2f}_{target_clip:.2f}"
            config.data.clip.feature_clip_percentile = feature_clip
            config.data.clip.target_clip_percentile = target_clip

            X_train_clipped, X_test_clipped = clip_features(
                X_train=X_train,
                X_test=X_test,
                exclude_cols=config.data.preprocess.exclude_clip_cols,
                config=config.data.clip,
            )

            y_train_clipped, _, _, _ = clip_targets(
                y=y_train,
                config=config.data.clip,
            )

            if not is_valid_threshold(y_train_clipped, config.model.threshold, min_samples=5):
                log.warn(f"Skipping feature clip={feature_clip} target clip={target_clip} not enough low/high samples.")
                continue

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

    save_experiment_config(
        output_path=output_path,
        experiment_config=experiment_config
    )

    analyze_sweep(experiment_path=output_path)


def main() -> int:
    try:
        run_clip_sweep()
        log.info(f"{EXPERIMENT_NAME} sweep completed successfully.")
        return 0

    except Exception as e:
        log.error(f"{EXPERIMENT_NAME} sweep failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
