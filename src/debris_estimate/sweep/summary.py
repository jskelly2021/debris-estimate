import json
import pandas as pd

from pathlib import Path

DEFAULT_SUMMARY_FILENAME = "summary.csv"


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


def flatten_run_record(record: dict) -> dict:
    row = {
        "run_name": record["run_name"],
    }

    def flatten_dict(d: dict, prefix: str = ""):
        for key, value in d.items():
            column = f"{prefix}_{key}" if prefix else key

            if isinstance(value, dict):
                flatten_dict(value, column)
            else:
                row[column] = value

    flatten_dict(record["config"])
    flatten_dict(record["metrics"])

    return row


def build_summary(
    run_paths: Path,
) -> pd.DataFrame:
    rows = []

    for run_path in run_paths:
        record = load_run_record(run_path)
        row = flatten_run_record(record)
        rows.append(row)

    return pd.DataFrame(rows)


def write_summary(
    summary: pd.DataFrame,
    analysis_path: Path,
    summary_filename: str = DEFAULT_SUMMARY_FILENAME,
) -> None:
    analysis_path.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        analysis_path / summary_filename,
        index=False,
    )
