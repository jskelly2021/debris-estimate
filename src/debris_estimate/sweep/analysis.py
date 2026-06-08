import pandas as pd

from pathlib import Path
from debris_estimate.sweep.summary import build_summary, write_summary
from debris_estimate.sweep.plots import plot_metric_vs_param


PLOTS_OUTPUT_DIR = "plots"


def collect_run_dirs(
    runs_path: Path,
) -> list[Path]:
    return sorted(
        path
        for path in runs_path.iterdir()
        if path.is_dir()
    )


def analyze_sweep(
    runs_path: Path,
    analysis_path: Path,
    primary_metric: str = "system_r2",
) -> pd.DataFrame:
    plots_path = analysis_path / PLOTS_OUTPUT_DIR

    run_paths = collect_run_dirs(runs_path=runs_path)

    # Summary
    summary = build_summary(run_paths=run_paths)
    write_summary(
        summary=summary,
        analysis_path=analysis_path,
    )

    # Leaderboard

    # Best Run

    # Plots
