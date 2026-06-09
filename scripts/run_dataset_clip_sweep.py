"""Sweep feature and target clip for each dataset for staged_model."""

from pathlib import Path
from copy import deepcopy
from run_clip_sweep import run_clip_sweep
from debris_estimate.logger import setup_logger, Log
from debris_estimate.config import RunConfig, ExperimentConfig
from debris_estimate.presets import (
    H9_V6_DATA_CONFIG,
    H8_V3_DATA_CONFIG,
    H9_STP_V3_DATA_CONFIG,
    GH9_V3_DATA_CONFIG,
    GH9_STP_V3_DATA_CONFIG,
    GH8_V3_DATA_CONFIG,
    BASELINE_MODEL_CONFIG,
)

data_configs = [
    H9_V6_DATA_CONFIG,
    H8_V3_DATA_CONFIG,
    H9_STP_V3_DATA_CONFIG,
    GH9_V3_DATA_CONFIG,
    GH9_STP_V3_DATA_CONFIG,
    GH8_V3_DATA_CONFIG,
]

DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig(
    experiment_name="base",
    primary_metric="system_r2",
    primary_metric_mode="max",
    swept_fields=["data.clip.feature_clip_percentile", "data.clip.target_clip_percentile"],
)

DEFAULT_RUN_CONFIG = RunConfig(
    run_name="base",
    data=H9_V6_DATA_CONFIG,
    model=BASELINE_MODEL_CONFIG,
)

setup_logger(verbose=False)
log = Log()


def run_dataset_clip_sweep():
    for data_config in data_configs:
        dataset_name = Path(data_config.dataset).stem
        log.info(f"{dataset_name}")

        run_config = deepcopy(DEFAULT_RUN_CONFIG)
        run_config.run_name = f"{dataset_name}"
        run_config.data = data_config

        experiment_config = deepcopy(DEFAULT_EXPERIMENT_CONFIG)
        experiment_config.experiment_name = f"clip_sweep_{dataset_name}"

        run_clip_sweep(
            base_config=run_config,
            experiment_config=experiment_config,
            output_dir="outputs/clip_sweeps"
        )


def main() -> int:
    try:
        run_dataset_clip_sweep()
        log.info("dataset feature/target clip sweep completed successfully.")
        return 0

    except Exception as e:
        log.error(f"{e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
