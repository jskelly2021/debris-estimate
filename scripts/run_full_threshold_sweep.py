
from pathlib import Path
from debris_estimate.logger import setup_logger, Log
from debris_estimate.config import ExperimentConfig

from run_sweep import run_sweep
from config_presets import ALL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs/threshold_sweeps"


setup_logger()
log = Log()


thresholds = {
    "gh8_v3_both": [100, 200, 300, 500, 800, 1000],
    "gh8_v3_cd":   [100, 200, 300, 500, 800, 1000],
    "gh8_v3_vg":   [100, 200, 300, 500, 800, 1000],

    "gh9_v3_both": [100, 200, 300, 500, 800, 1000],
    "gh9_v3_cd":   [100, 200, 300, 500, 800, 1000],
    "gh9_v3_vg":   [100, 200, 300, 500, 800, 1000],

    "h8_v3_both":  [100, 200, 300, 500, 800, 1000],
    "h8_v3_cd":    [100, 200, 300, 500, 800, 1000],
    "h8_v3_vg":    [100, 200, 300, 500, 800, 1000],

    "h9_v6_both":  [100, 200, 300, 500, 800, 1000],
    "h9_v6_cd":    [100, 200, 300, 500, 800, 1000],
    "h9_v6_vg":    [100, 200, 300, 500, 800, 1000],
}


def run_full_threshold_sweep() -> tuple[int, int]:
    count = 0
    success = 0
    for preset in ALL:
        name = preset.CONFIG_NAME

        if name not in thresholds:
            log.warn(f"Skipping {name}: no thresholds defined.")
            continue

        count += 1

        experiment_config = ExperimentConfig(
            experiment_name=preset.CONFIG_NAME,
            output_dir=OUTPUT_DIR,
            base_run_config=preset.build_run_config(),
            swept_fields={
                "model.threshold": thresholds[name],
            },
        )

        try:
            run_sweep(experiment_config=experiment_config)
        except Exception as e:
            log.error(f"Experiment {experiment_config.experiment_name} failed: {e}")
            continue
        success += 1

    return count, success


def main() -> int:
    try:
        count, success = run_full_threshold_sweep()
        log.info(f"Finished threshold sweeps: {success}/{count} succeeded")
        return 0

    except Exception as e:
        log.error(f"Failed to complete sweeps: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
