"""Build sweep analysis plots."""

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from pandas.api.types import is_numeric_dtype


def is_numeric_param(summary: pd.DataFrame, param: str) -> bool:
    series = summary[param].dropna()

    if is_numeric_dtype(series):
        return True

    try:
        pd.to_numeric(series)
        return True
    except (ValueError, TypeError):
        return False


def plot_category_vs_metric(
    summary: pd.DataFrame,
    param: str,
    metric: str,
    output_path: str | Path,
) -> None:
    plot_df = (
        summary[[param, metric]]
        .dropna()
        .groupby(param)[metric]
        .mean()
        .sort_values(ascending=False)
    )

    if plot_df.empty:
        return

    plt.figure()
    plot_df.plot(kind="bar")
    plt.ylabel(metric)
    plt.title(f"{param} vs {metric}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_param_vs_metric(
    summary: pd.DataFrame,
    param: str,
    metric: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = (
        summary[[param, metric]]
        .dropna()
        .sort_values(param)
    )

    if plot_df.empty:
        return

    plt.figure()
    plt.plot(plot_df[param], plot_df[metric], marker="o")
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.title(f"{param} vs {metric}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_sweep_plots(
    summary: pd.DataFrame,
    swept_fields: list[str] | None,
    output_path: Path,
    metrics: list[str] | None = None,
) -> None:
    swept_fields = swept_fields or []
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = metrics or [
        "system_r2",
        "system_mae",
        "system_rmse",
    ]

    available_metrics = [
        metric for metric in metrics
        if metric in summary.columns
    ]

    for param in swept_fields:
        if param not in summary.columns:
            continue

        for metric in available_metrics:
            filename = f"{param}_vs_{metric}.png".replace(".", "_")

            if is_numeric_param(summary=summary, param=param):
                plot_param_vs_metric(
                    summary=summary,
                    param=param,
                    metric=metric,
                    output_path=output_path / filename,
                )
            else:
                plot_category_vs_metric(
                    summary=summary,
                    param=param,
                    metric=metric,
                    output_path=output_path / filename,
                )
