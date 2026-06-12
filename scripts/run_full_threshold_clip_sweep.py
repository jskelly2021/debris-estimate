
from pathlib import Path
from debris_estimate.logger import setup_logger, Log
from debris_estimate.config import ExperimentConfig

from run_sweep import run_sweep
from config_presets import ALL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs/threshold_clip_sweeps"


setup_logger()
log = Log()


thresholds = {
    "gh8_v3_both": [300, 500, 750, 800, 850, 1000, 1250, 2000, 2500, 4000, 5000, 7500],
    "gh8_v3_cd":   [100, 200, 300, 500, 800, 1000, 1500, 2000, 2250, 2500, 2750, 3000],
    "gh8_v3_vg":   [100, 200, 250, 300, 350, 400, 500, 800, 1000, 1500, 2000],

    "gh9_v3_both": [450, 500, 550, 600, 800, 1000, 1500, 1750, 2000, 2250],
    "gh9_v3_cd":   [150, 175, 200, 300, 400, 450, 500, 550, 600, 800],
    "gh9_v3_vg":   [100, 200, 300, 350, 450, 500, 550, 800, 1000, 1500, 1750],

    "h8_v3_both":  [100, 200, 300, 500, 800, 1000, 1500, 2000, 2500, 3000, 5000, 10000],
    "h8_v3_cd":    [500, 800, 1000, 1250, 1500, 2000, 2225, 2500, 2750, 3000],
    "h8_v3_vg":    [100, 200, 300, 500, 800, 1000, 12500, 2000, 4000, 5000, 6000, 8000],

    "h9_v6_both":  [100, 150, 200, 225, 250, 275, 300, 325, 350, 400],
    "h9_v6_cd":    [100, 125, 150, 200, 250, 300, 350, 400],
    "h9_v6_vg":    [100, 150, 200, 225, 250, 275, 300, 350],
}

clips = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 1.0]


def run_full_threshold_clip_sweep() -> tuple[int, int]:
    count = 0
    success = 0
    for preset in ALL:
        name = preset.CONFIG_NAME

        if name not in thresholds:
            log.warn(f"Skipping {name}: no thresholds defined.")
            continue

        count += 1

        experiment_config = ExperimentConfig(
            experiment_name=name,
            output_dir=OUTPUT_DIR,
            base_run_config=preset.build_run_config(),
            swept_fields={
                "model.threshold": thresholds[name],
                "data.clip.fclip": clips,
                "data.clip.tclip": clips,
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
        count, success = run_full_threshold_clip_sweep()
        log.info(f"Finished sweeps: {success}/{count} succeeded")
        return 0

    except Exception as e:
        log.error(f"Failed to complete sweeps: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
