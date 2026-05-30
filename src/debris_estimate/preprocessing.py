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
    distance_cols = ["dist_coast", "dist_reservoir"],
)


def _remove_leakage_columns(df, columns) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=columns, errors="ignore")
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
    df = _log_transform_columns(df, config.log_cols)

    #TODO: Convert distance features to binary indicators.

    df = _one_hot_encode_columns(df, config.categorical_cols)

    log.info("Data preprocessing completed.")
    return df
