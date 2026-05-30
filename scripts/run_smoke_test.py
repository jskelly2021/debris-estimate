"""Simple smoke test for staged_model. Performs a single run of the model."""

import argparse
import pandas as pd

from pathlib import Path
from debris_estimate.evaluation import evaluate_system
from debris_estimate.logger import setup_logger, Log
from debris_estimate.data import load_dataset
from debris_estimate.preprocessing import preprocess_features, DEFAULT_PREPROCESS_CONFIG
from debris_estimate.split import split_data
from debris_estimate.model import train_staged_model, predict_staged_model
from debris_estimate.clipping import fit_numeric_feature_clip_caps, apply_numeric_feature_clip_caps, clip_target
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


def run_smoke_test(args=None):
    data_path = PROJECT_ROOT / args.data_path

    df = load_dataset(data_path)
    X = preprocess_features(df)
    y = df["VolBoth_sum"]

    split = split_data(X, y, test_size=0.2, random_state=42)

    exclude_cols = (
        DEFAULT_PREPROCESS_CONFIG.log_cols
        + DEFAULT_PREPROCESS_CONFIG.distance_cols
        + DEFAULT_PREPROCESS_CONFIG.categorical_cols
    )

    feature_clip_caps = fit_numeric_feature_clip_caps(
        X_train=split.X_train,
        percentile=0.99,
        exclude_cols=exclude_cols
    )

    X_train = apply_numeric_feature_clip_caps(
        X=split.X_train,
        caps=feature_clip_caps,
    )

    X_test = apply_numeric_feature_clip_caps(
        X=split.X_test,
        caps=feature_clip_caps,
    )

    target_clip_result = clip_target(y=split.y_train, percentile=1.0)
    y_train = target_clip_result.y_clipped

    zero_vs_positive_model, tier_model, low_regressor, high_regressor = train_staged_model(
        X_train=X_train,
        y_train=y_train,
        threshold=300
    )

    preds = predict_staged_model(
        X=X_test,
        zero_pos_classifier=zero_vs_positive_model,
        tier_classifier=tier_model,
        low_regressor=low_regressor,
        high_regressor=high_regressor
    )

    model_eval = evaluate_system(
        y_true=split.y_test,
        preds=preds,
        threshold=300
    )

    log.info(f"Saving run outputs to {OUTPUT_PATH}...")

    save_run_outputs(
        y_true=split.y_test,
        preds=preds,
        eval=model_eval,
        threshold=300,
        output_path=OUTPUT_PATH,
        run_name="run",
        save_metrics=True,
        save_predictions=True,
        save_plots=True
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
