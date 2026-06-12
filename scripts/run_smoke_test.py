"""Simple smoke test for staged_model. Performs a single run of the model."""

from pathlib import Path
from debris_estimate.logger import setup_logger, Log
from debris_estimate.run import run_model
from config_presets import baseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "smoke_test"

setup_logger(verbose=True)
log = Log()


def run_smoke_test():
    run_config = baseline.build_run_config()
    run_dir = PROJECT_ROOT / OUTPUT_DIR / EXPERIMENT_NAME
    run_model(config=run_config, run_dir=run_dir)


def main() -> int:
    try:
        run_smoke_test()
        log.info(f"Smoke test completed successfully.")
        return 0

    except Exception as e:
        log.error(f"Smoke test failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
