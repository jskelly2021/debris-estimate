"""Build experiment summary tables from run outputs."""

import json
from pathlib import Path

import pandas as pd

CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(d: dict, field: str):
    cur = d
    for part in field.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _flatten_metrics(metrics: dict) -> dict:
    row = {}

    for group_name, group_metrics in metrics.items():
        if isinstance(group_metrics, dict):
            for metric_name, value in group_metrics.items():
                row[f"{group_name}_{metric_name}"] = value
        else:
            row[group_name] = group_metrics

    return row


def build_summary(
    runs_path: Path,
    swept_fields: list[str] | None = None
) -> pd.DataFrame:
    swept_fields = swept_fields or []

    rows = []

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / CONFIG_FILENAME
        metrics_path = run_dir / METRICS_FILENAME

        if not config_path.exists() or not metrics_path.exists():
            continue

        config = _load_json(config_path)
        metrics = _load_json(metrics_path)

        row = {
            "run_id": run_dir.name,
        }

        for field in swept_fields:
            row[field] = _get_nested(config, field)

        row.update(_flatten_metrics(metrics))
        rows.append(row)

    return pd.DataFrame(rows)
