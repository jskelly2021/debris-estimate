
from pathlib import Path
from debris_estimate.logger import setup_logger, Log
from debris_estimate.sweep import analyze_sweep
from debris_estimate.run import run_model
from debris_estimate.config import ExperimentConfig
from debris_estimate.outputs import save_experiment_metadata
from config_presets import ALL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "best_runs"
RUNS_DIR = "runs"
ANALYSIS_DIR = "analysis"


setup_logger()
log = Log()


def run_full(experiment_config: ExperimentConfig) -> tuple[int, int]:
    experiment_dir = PROJECT_ROOT / OUTPUT_DIR / EXPERIMENT_NAME
    runs_dir = experiment_dir / RUNS_DIR
    analysis_dir = experiment_dir /ANALYSIS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    save_experiment_metadata(
        experiment_config=experiment_config,
        output_path=experiment_dir,
    )

    count = 0
    success = 0
    for preset in ALL:
        name = preset.CONFIG_NAME
        count += 1

        try:
            log.info(f"{name}")
            run_model(config=preset.build_run_config(), run_dir=runs_dir / name)
        except Exception as e:
            log.error(f"{name} failed: {e}")
            continue
        success += 1

    analyze_sweep(
        experiment_dir=experiment_dir,
        runs_dir=runs_dir,
        output_path=analysis_dir
    )

    return count, success


def main() -> int:

    experiment_config = ExperimentConfig(
        experiment_name=EXPERIMENT_NAME,
        output_dir=OUTPUT_DIR,
        base_run_config=None,
    )

    try:
        count, success = run_full(experiment_config=experiment_config)
        log.info(f"Finished runs: {success}/{count} succeeded")
        return 0

    except Exception as e:
        log.error(f"Failed to complete runs: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
