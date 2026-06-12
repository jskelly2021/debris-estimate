from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from itertools import product
from typing import Any

from debris_estimate.config import ExperimentConfig, RunConfig
from debris_estimate.outputs import save_experiment_metadata
from debris_estimate.logger import setup_logger, Log
from debris_estimate.sweep import analyze_sweep
from debris_estimate.run import run_model

setup_logger(verbose=False)
log = Log()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_OUTPUT_DIR = "runs"
ANALYSIS_OUTPUT_DIR = "analysis"


def format_value(value: object) -> str:
    return str(value).replace(".", "p").replace("/", "-")


def make_run_name(i: int, swept_values: dict[str, object]) -> str:
    parts = [f"{i:03d}"]

    for field, value in swept_values.items():
        short_field = field.split(".")[-1]
        parts.append(f"{short_field}_{format_value(value)}")

    return "_".join(parts)


def set_nested_attr(obj: object, path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj

    for part in parts[:-1]:
        target = getattr(target, part)

    setattr(target, parts[-1], value)


def generate_run_configs(exp: ExperimentConfig) -> list[RunConfig]:
    fields = list(exp.swept_fields.keys())
    values = list(exp.swept_fields.values())

    if not fields:
        config = deepcopy(exp.base_run_config)
        config.run_name = "000"
        return [config]

    configs = []

    for i, combo in enumerate(product(*values)):
        config = deepcopy(exp.base_run_config)

        swept_values = dict(zip(fields, combo))

        for field, value in swept_values.items():
            set_nested_attr(config, field, value)

        config.run_name = make_run_name(i, swept_values)
        configs.append(config)

    return configs


def run_sweep(experiment_config: ExperimentConfig) -> None:
    experiment_dir = Path(experiment_config.output_dir) / experiment_config.experiment_name
    runs_dir = experiment_dir / RUN_OUTPUT_DIR
    analysis_dir = experiment_dir / ANALYSIS_OUTPUT_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    save_experiment_metadata(
        experiment_config=experiment_config,
        output_path=experiment_dir,
    )

    log.info(f"Starting {experiment_config.experiment_name} sweep")

    run_configs = generate_run_configs(exp=experiment_config)

    for run_config in run_configs:
        try:
            log.info(f"{run_config.run_name}")
            run_model(
                config=run_config,
                run_dir=runs_dir / run_config.run_name,
            )
        except Exception as e:
            log.error(f"{run_config.run_name}: {e}")
            continue

    analyze_sweep(
        experiment_dir=experiment_dir,
        runs_dir=runs_dir,
        output_path=analysis_dir
    )


def main() -> int:
    import config_presets.baseline as baseline

    THRESHOLD_SWEEP = ExperimentConfig(
        experiment_name="baseline_threshold_sweep",
        output_dir="outputs/threshold_sweeps",
        base_run_config=baseline.build_run_config(),
        swept_fields={
            "model.threshold": [100, 250, 500, 750, 1000],
        },
    )

    try:
        run_sweep(experiment_config=THRESHOLD_SWEEP)
        log.info(f"Sweep completed successfully.")
        return 0

    except Exception as e:
        log.error(f"Sweep failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
