"""Simple smoke test for staged_model. Performs a single run of the model."""

import argparse

from pathlib import Path
from debris_estimate.logger import setup_logger, Log
from debris_estimate.config import RunConfig
from debris_estimate.presets import (
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
from debris_estimate.outputs import save_run_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "smoke_test"
OUTPUT_PATH = PROJECT_ROOT / OUTPUT_DIR / EXPERIMENT_NAME

setup_logger()
log = Log()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a smoke test for the staged model.")
    parser.add_argument("--data_path", type=str, default="data/h9_debrisv6.csv", help="Path to the input dataset.")
    return parser.parse_args()


def run_smoke_test(args):
    config = RunConfig(
        experiment_name=EXPERIMENT_NAME,
        run_name="run",
        data=H9_V6_DATA_CONFIG,
        model=BASELINE_MODEL_CONFIG,
    )

    data_path = PROJECT_ROOT / args.data_path
    df = load_dataset(path=data_path)

    ### Preprocessing ###
    X = preprocess_features(
        df=df,
        config=config.data.preprocess,
    )

    ### Splitting ###
    y = df["VolBoth_sum"]

    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        config=config.data.split,
    )

    ### Clipping ###
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
    log.info(f"Saving run outputs to {OUTPUT_PATH}...")

    save_run_outputs(
        output_path=OUTPUT_PATH,
        run_name=config.run_name,
        eval_results=eval_results,
        y_true=y_test,
        pred_results=pred_results,
        run_config=config,
        figure_groups=figure_groups,
    )


def main() -> int:
    args = parse_args()

    try:
        run_smoke_test(args)
        log.info("Smoke test completed successfully.")
        return 0

    except Exception as e:
        log.error(f"Smoke test failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
