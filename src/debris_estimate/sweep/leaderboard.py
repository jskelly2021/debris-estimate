"""Build leaderboard tables from sweep summaries."""

import pandas as pd


def build_leaderboard(
    summary: pd.DataFrame,
    swept_fields: list[str] | None,
    primary_metric: str,
    primary_metric_mode: str,
) -> pd.DataFrame:
    swept_fields = swept_fields or []

    system_cols = [
        col for col in summary.columns
        if col.startswith("system_")
    ]

    columns = ["run_id", *swept_fields, *system_cols]

    ascending = primary_metric_mode == "min"

    leaderboard = (
        summary[columns]
        .sort_values(primary_metric, ascending=ascending)
        .reset_index(drop=True)
    )

    leaderboard.insert(0, "rank", leaderboard.index + 1)

    return leaderboard
