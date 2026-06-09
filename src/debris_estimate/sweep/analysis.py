"""Orchestrate sweep analysis outputs."""

import json
from pathlib import Path

from debris_estimate.sweep.summary import build_summary
from debris_estimate.config import ExperimentConfig


RUNS_DIR = "runs"
ANALYSIS_DIR = "analysis"
EXPERIMENT_CONFIG_FILENAME = "experiment.json"
SUMMARY_FILENAME = "summary.csv"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_sweep(experiment_path: Path) -> None:
    experiment_config = ExperimentConfig(**_load_json(experiment_path / EXPERIMENT_CONFIG_FILENAME))

    runs_path = experiment_path / RUNS_DIR
    analysis_path = experiment_path / ANALYSIS_DIR
    analysis_path.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        runs_path=runs_path,
        swept_fields=experiment_config.swept_fields,
    )

    summary_file_path = analysis_path / SUMMARY_FILENAME
    summary.to_csv(summary_file_path, index=False)
