"""Build sweep analysis plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
            plot_param_vs_metric(
                summary=summary,
                param=param,
                metric=metric,
                output_path=output_path / filename,
            )
