"""Simple smoke test for staged_model. Performs a single run of the model."""

import argparse

from pathlib import Path
from debris_estimate.evaluation import evaluate_system
from debris_estimate.logger import setup_logger, Log
from debris_estimate.data import load_dataset
from debris_estimate.preprocessing import preprocess_features
from debris_estimate.split import split_data
from debris_estimate.model import train_staged_model, predict_staged_model
from debris_estimate.outputs import create_output_dir, save_run_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

setup_logger()
log = Log()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a smoke test for the staged model.")
    parser.add_argument("--data_path", type=str, default="data/h9_debrisv5.csv", help="Path to the input dataset.")
    return parser.parse_args()


def run_smoke_test(args=None):
    data_path = PROJECT_ROOT / args.data_path

    df = load_dataset(data_path)
    X = preprocess_features(df)
    y = df["VolBoth_sum"]

    split = split_data(X, y, test_size=0.2, random_state=42)

    zero_vs_positive_model, tier_model, low_regressor, high_regressor = train_staged_model(split, threshold=300)

    preds = predict_staged_model(
        split.X_test,
        zero_vs_positive_model,
        tier_model,
        low_regressor,
        high_regressor
    )

    model_eval = evaluate_system(split.y_test, preds, threshold=300)

    output_dir = create_output_dir(OUTPUT_DIR, run_name="smoke_test")
    log.info(f"Saving run outputs to {output_dir}...")
    save_run_outputs(model_eval, preds, split.y_test, output_dir, save_predictions=True)


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
