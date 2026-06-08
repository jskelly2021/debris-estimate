import json
import pandas as pd

from pathlib import Path

CONFIG_FILENAME= "config.json"
METRICS_FILENAME = "metrics.json"
DEFAULT_SUMMARY_FILENAME = "summary.csv"


def load_run_record(run_path: Path) -> dict:
    with open(run_path / CONFIG_FILENAME, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(run_path / METRICS_FILENAME, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "run_name": run_path.name,
        "config": config,
        "metrics": metrics,
    }


def get_nested_value(d: dict, field_path: str):
    value = d

    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]

    return value


def flatten_run_record(
    record: dict,
    swept_fields: list[str] | None,
) -> dict:
    row = {
        "run_name": record["run_name"],
    }

    # Include only selected config fields
    for field in swept_fields or []:
        row[field.replace(".", "_")] = get_nested_value(record["config"], field)

    # Include all metrics
    def flatten_metrics(d: dict, prefix: str = ""):
        for key, value in d.items():
            column = f"{prefix}_{key}" if prefix else key

            if isinstance(value, dict):
                flatten_metrics(value, column)
            else:
                row[column] = value

    flatten_metrics(record["metrics"])

    return row


def build_summary(
    run_paths: Path,
    swept_fields: list[str] | None
) -> pd.DataFrame:
    rows = []

    for run_path in run_paths:
        record = load_run_record(run_path)
        row = flatten_run_record(record, swept_fields=swept_fields)
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
