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


def _convert_distance_to_binary(
    df: pd.DataFrame,
    distance_thresholds: dict[str, float] | None,
) -> pd.DataFrame:
    if not distance_thresholds:
        return df

    df = df.copy()
    for col, threshold in distance_thresholds.items():
        if col in df.columns:
            threshold_str = str(threshold).replace(".", "_")
            binary_col_name = f"{col}_within_{threshold_str}"
            df[binary_col_name] = (df[col] <= threshold).astype(int)
        else:
            log.warn("Column %s not found for distance to binary conversion.", col)

    df = df.drop(columns=list(distance_thresholds), errors="ignore")
    return df


def _encode_ordinals(
    df: pd.DataFrame,
    ordinal_maps: dict[str, dict[str, int]] | None,
) -> pd.DataFrame:
    if not ordinal_maps:
        return df

    df = df.copy()
    for feature, value_map in ordinal_maps.items():
        if feature in df.columns:
            df[f"{feature}_ord"] = df[feature].map(value_map)

            unmapped = df.loc[df[f"{feature}_ord"].isna() & df[feature].notna(), feature].unique()
            if len(unmapped) > 0:
                log.warn(
                    "Unmapped values found for %s during ordinal encoding: %s",
                    feature,
                    list(unmapped),
                )
        else:
            log.warn("Column %s not found for ordinal encoding.", feature)

    df = df.drop(columns=list(ordinal_maps), errors="ignore")
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


def preprocess_features(
    df: pd.DataFrame,
    config: PreprocessConfig
) -> pd.DataFrame:
    log.info("Starting data preprocessing...")
    df = df.copy()

    df = _remove_leakage_columns(df=df, drop_cols=config.drop_cols)
    df = _log_transform_columns(df=df, log_cols=config.log_cols)
    df = _convert_distance_to_binary(df=df, distance_thresholds=config.distance_thresholds)
    df = _encode_ordinals(df=df, ordinal_maps=config.ordinal_maps)
    df = _one_hot_encode_columns(df=df, categorical_cols=config.categorical_cols)

    log.info("Data preprocessing completed.")
    return df
