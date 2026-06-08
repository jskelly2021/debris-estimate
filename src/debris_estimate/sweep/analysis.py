import pandas as pd
import json

from pathlib import Path
from debris_estimate.config import ExperimentConfig
from debris_estimate.sweep.summary import build_summary, write_summary
from debris_estimate.sweep.plots import plot_metric_vs_param


PLOTS_OUTPUT_DIR = "plots"


def collect_run_dirs(runs_path: Path) -> list[Path]:
    return sorted(
        path
        for path in runs_path.iterdir()
        if path.is_dir()
    )


def load_experiment_config(file_path: Path) -> ExperimentConfig: 
    with open(file_path, "r") as f:
        experiment = json.load(f)

    return ExperimentConfig(**experiment)


def analyze_sweep(
    runs_path: Path,
    analysis_path: Path,
    experiment_config: ExperimentConfig,
) -> pd.DataFrame:
    plots_path = analysis_path / PLOTS_OUTPUT_DIR

    run_paths = collect_run_dirs(runs_path=runs_path)

    # Summary
    summary = build_summary(run_paths=run_paths, swept_fields=experiment_config.swept_fields)
    write_summary(
        summary=summary,
        analysis_path=analysis_path,
    )

    # Leaderboard

    # Best Run

    # Plots
