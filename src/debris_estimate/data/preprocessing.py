"""owns the logic for preprocessing features."""

import pandas as pd
import numpy as np

from debris_estimate.logger import Log
from debris_estimate.config import PreprocessConfig

log = Log()


def _remove_leakage_columns(
    df: pd.DataFrame,
    drop_cols: list[str] | None
) -> pd.DataFrame:
    if not drop_cols:
        return df

    df = df.copy()
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def _log_transform_columns(
    df: pd.DataFrame,
    log_cols: list[str] | None
) -> pd.DataFrame:
    if not log_cols:
        return df

    df = df.copy()
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
        else:
            log.warn("Column %s not found for log transformation.", col)
    return df


def _one_hot_encode_columns(
    df: pd.DataFrame,
    categorical_cols: list[str] | None,
    drop_first=True
) -> pd.DataFrame:
    if not categorical_cols:
        return df

    df = df.copy()
    existing_cols = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cols, drop_first=drop_first)
    return df


def _convert_distance_to_binary(
    df: pd.DataFrame,
    distance_col_threshold_map: dict[str, float] | None,
) -> pd.DataFrame:
    if not distance_col_threshold_map:
        return df

    df = df.copy()
    for col, threshold in distance_col_threshold_map.items():
        if col in df.columns:
            threshold_str = str(threshold).replace(".", "_")
            binary_col_name = f"{col}_within_{threshold_str}"
            df[binary_col_name] = (df[col] <= threshold).astype(int)
        else:
            log.warn("Column %s not found for distance to binary conversion.", col)

    df = df.drop(columns=list(distance_col_threshold_map), errors="ignore")

    return df


def preprocess_features(
    df: pd.DataFrame,
    config: PreprocessConfig
) -> pd.DataFrame:
    log.info("Starting data preprocessing...")
    df = df.copy()

    df = _remove_leakage_columns(df=df, drop_cols=config.drop_cols)
    df = _log_transform_columns(df=df, log_cols=config.log_cols)

    df = _convert_distance_to_binary(
        df=df,
        distance_col_threshold_map=config.distance_col_threshold_map
    )

    df = _one_hot_encode_columns(df=df, categorical_cols=config.categorical_cols)

    log.info("Data preprocessing completed.")
    return df
