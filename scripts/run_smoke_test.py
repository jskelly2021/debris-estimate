"""Simple smoke test for staged_model. Performs a single run of the model."""

import sys
import argparse

from pathlib import Path
from debris_estimate.logger import setup_logger, Log
from debris_estimate.data import load_dataset
from debris_estimate.preprocessing import preprocess_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

setup_logger()
log = Log()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a smoke test for the staged model.")
    parser.add_argument("--data_path", type=str, default="data/h9_debrisv5.csv", help="Path to the input dataset.")
    return parser.parse_args()


def run_smoke_test(args=None):
    data_path = PROJECT_ROOT / args.data_path

    df = load_dataset(data_path)
    df = preprocess_data(df)

    log.info("Smoke test ran successfully with preprocessed data shape: %s", df.shape)


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
