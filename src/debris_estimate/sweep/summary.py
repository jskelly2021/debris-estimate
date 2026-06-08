import json
import pandas as pd

from pathlib import Path

DEFAULT_SUMMARY_FILENAME = "summary.csv"


def collect_run_dirs(
    runs_path: Path,
) -> list[Path]:
    return sorted(
        path
        for path in runs_path.iterdir()
        if path.is_dir()
    )


def load_run_record(
    run_path: Path,
) -> dict:
    with open(run_path / "config.json", "r") as f:
        config = json.load(f)

    with open(run_path / "metrics.json", "r") as f:
        metrics = json.load(f)

    return {
        "run_name": run_path.name,
        "config": config,
        "metrics": metrics,
    }


def flatten_run_record(
    record: dict
) -> dict:
    row = {
        "run_name": record["run_name"],
    }

    def flatten_dict(
        d: dict,
        prefix: str,
    ) -> None:
        for key, value in d.items():
            column_name = f"{prefix}_{key}" if prefix else key

            if isinstance(value, dict):
                flatten_dict(value, column_name)
            else:
                row[column_name] = value

    flatten_dict(record["config"], "config")
    flatten_dict(record["metrics"], "metrics")

    return row


def build_summary_df(
    runs_path: Path,
) -> pd.DataFrame:
    rows = []

    for run_path in collect_run_dirs(runs_path):
        record = load_run_record(run_path)
        row = flatten_run_record(record)
        rows.append(row)

    return pd.DataFrame(rows)


def write_summary(
    summary_df: pd.DataFrame,
    analysis_path: Path,
    summary_filename: str = DEFAULT_SUMMARY_FILENAME,
) -> None:
    analysis_path.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        analysis_path / summary_filename,
        index=False,
    )
