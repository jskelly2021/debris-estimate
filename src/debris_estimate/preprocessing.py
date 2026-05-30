"""owns the logic for processing features."""

import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from debris_estimate.logger import Log

log = Log()


@dataclass
class PreprocessConfig:
    drop_cols: list[str] = field(default_factory=list)
    log_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    distance_cols: list[str] = field(default_factory=list)
    feature_clip_percentile: float | None = 1.0


# Default config based on h9_debrisv6 dataset.
DEFAULT_PREPROCESS_CONFIG = PreprocessConfig(
    drop_cols = [
        "GRID_ID",
        "age_med",
        "num_story_sum", "num_story_med",
        "sqm_sum", "sqm_med",
        "found_ht_med", "found_ht_sum",
        "val_struct_sum", "val_struct_med",
        "val_cont_med", "val_cont_sum",
        "VolCD", "VolVG",
        "VolCD_sum", "VolVG_sum", "VolBoth_sum",
        "Bin_CD", "Bin_VG", "Bin_Both"
    ],
    log_cols  = ["sqm", "val_struct", "val_cont", "fld_pct"],
    categorical_cols = ["evac_degree", "fld_zone", "landcover", "landuse"],
    distance_cols = ["dist_coast", "dist_reservoir", "dist_htrack_M", "dist_htrack_H"],
    feature_clip_percentile = 0.99,
)


def _remove_leakage_columns(df, columns) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=columns, errors="ignore")
    return df


def _clip_numeric_columns(
    df: pd.DataFrame,
    percentile: float = 0.99,
    exclude_cols: list[str] | None = None
) -> pd.DataFrame:
    df = df.copy()

    exclude_cols = exclude_cols or []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    clip_cols = [
        col for col in numeric_cols
        if col not in exclude_cols
    ]

    log.info("Clipping numeric features...")

    for col in clip_cols:
        upper = df[col].quantile(percentile)
        df[col] = np.clip(df[col], 0, upper)

    log.info(f"Clipped {len(clip_cols)} features at {percentile:.3f}")

    return df


def _log_transform_columns(df, columns) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col])
        else:
            log.warn("Column %s not found for log transformation.", col)
    return df


def _one_hot_encode_columns(df, columns, drop_first=True) -> pd.DataFrame:
    df = df.copy()
    existing_cols = [col for col in columns if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cols, drop_first=drop_first)
    return df


def preprocess_features(
    df: pd.DataFrame,
    config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG
) -> pd.DataFrame:
    log.info("Starting data preprocessing...")
    df = df.copy()

    df = _remove_leakage_columns(df, config.drop_cols)

    if config.feature_clip_percentile is not None:
        df = _clip_numeric_columns(
            df=df,
            percentile=config.feature_clip_percentile,
            exclude_cols=config.log_cols + config.distance_cols + config.categorical_cols,
        )

    df = _log_transform_columns(df, config.log_cols)

    #TODO: Convert distance features to binary indicators.

    df = _one_hot_encode_columns(df, config.categorical_cols)

    log.info("Data preprocessing completed.")
    return df
