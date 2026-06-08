
import pandas as pd

from pathlib import Path


def rank_runs(
    df: pd.DataFrame,
    metric_col: str,
) -> pd.DataFrame:
    pass


def write_leaderboard(
    leaderboard: pd.DataFrame,
    analysis_file_path: Path,
    metric_col: str,
    top_n: int | None,
) -> None:
    pass


def write_best_run(
    leaderboard: pd.DataFrame,
    best_run_file_path: Path,
) -> None:
    pass
