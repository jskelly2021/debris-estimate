"""Sweep thresholds for each dataset for staged_model."""

from pathlib import Path
from copy import deepcopy
from run_threshold_sweep import run_threshold_sweep
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
    swept_fields=["model.threshold"],
)

DEFAULT_RUN_CONFIG = RunConfig(
    run_name="base",
    data=H9_V6_DATA_CONFIG,
    model=BASELINE_MODEL_CONFIG,
)

setup_logger(verbose=False)
log = Log()

GrideH8_v3      = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 15000]
GrideH9_StP_v3  = [100, 200, 400, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
GrideH9_v3      = [100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 1800, 2000, 3000, 4000, 4500]
h8_debrisv3     = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 15000]
h9_debrisv6     = [100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 1800, 2000]
h9_StP_debrisv3 = [100, 200, 400, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]

thresholds = {
    "GrideH8_v3": GrideH8_v3,
    "GrideH9_StP_v3": GrideH9_StP_v3,
    "GrideH9_v3": GrideH9_v3,
    "h8_debrisv3": h8_debrisv3,
    "h9_debrisv6": h9_debrisv6,
    "h9_StP_debrisv3": h9_StP_debrisv3,
}


def run_dataset_threshold_sweep():
    for data_config in data_configs:
        dataset_name = Path(data_config.dataset).stem
        log.info(f"{dataset_name}")

        run_config = deepcopy(DEFAULT_RUN_CONFIG)
        run_config.run_name = f"{dataset_name}"
        run_config.data = data_config

        experiment_config = deepcopy(DEFAULT_EXPERIMENT_CONFIG)
        experiment_config.experiment_name = f"threshold_sweep_{dataset_name}"

        run_threshold_sweep(
            base_config=run_config,
            experiment_config=experiment_config,
            thresholds=thresholds[dataset_name],
        )


def main() -> int:
    try:
        run_dataset_threshold_sweep()
        log.info("dataset threshold sweep completed successfully.")
        return 0

    except Exception as e:
        log.error(f"{e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
