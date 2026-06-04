"""Simple smoke test for staged_model. Performs a single run of the model."""

import argparse
import pandas as pd

from pathlib import Path
from debris_estimate.evaluation import evaluate_system
from debris_estimate.logger import setup_logger, Log
from debris_estimate.data import load_dataset
from debris_estimate.preprocessing import preprocess_features
from debris_estimate.split import split_data
from debris_estimate.model import StagedModel
from debris_estimate.clipping import clip_features, clip_targets
from debris_estimate.outputs import save_run_outputs
from debris_estimate.config import RunConfig
from debris_estimate.presets import (
    H9_V6_PREPROCESS_CONFIG,
    BASELINE_SPLIT_CONFIG,
    BASELINE_CLIP_CONFIG,
    BASELINE_MODEL_CONFIG,
)

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
        preprocess=H9_V6_PREPROCESS_CONFIG,
        split=BASELINE_SPLIT_CONFIG,
        clip=BASELINE_CLIP_CONFIG,
        model=BASELINE_MODEL_CONFIG,
    )

    data_path = PROJECT_ROOT / args.data_path
    df = load_dataset(path=data_path)

    ### Preprocessing ###
    X = preprocess_features(
        df=df,
        config=config.preprocess,
    )

    ### Splitting ###
    y = df["VolBoth_sum"]

    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        test_size=config.split.test_size,
        random_state=config.split.random_state
    )

    ### Clipping ###
    exclude_cols = (
        config.preprocess.log_cols
        + config.preprocess.distance_cols
        + config.preprocess.categorical_cols
    )

    X_train_clipped, X_test_clipped = clip_features(
        X_train=X_train,
        X_test=X_test,
        exclude_cols=exclude_cols,
        config=config.clip,
    )

    y_train_clipped, _, _, _ = clip_targets(
        y=y_train,
        config=config.clip,
    )

    ### Training ###
    staged_model = StagedModel(config=config.model)
    staged_model.fit(X_train=X_train_clipped, y_train=y_train_clipped)

    ### Prediction ###
    preds = staged_model.predict_details(X=X_test_clipped)

    ### Evaluation ###
    model_eval = evaluate_system(
        y_true=y_test,
        preds=preds,
        threshold=config.model.threshold
    )

    ### Output ###
    log.info(f"Saving run outputs to {OUTPUT_PATH}...")

    save_run_outputs(
        y_true=y_test,
        preds=preds,
        eval=model_eval,
        config=config,
        output_path=OUTPUT_PATH,
        save_metrics=True,
        save_predictions=True,
        save_plots=True,
        save_config=True,
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
