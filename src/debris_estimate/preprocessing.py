"""owns the logic for processing features."""

import pandas as pd
import numpy as np

from debris_estimate.logger import Log

log = Log()


def _remove_leakage_columns(
    df: pd.DataFrame,
    drop_columns: list[str] | None
) -> pd.DataFrame:
    if not drop_columns:
        return df

    df = df.copy()
    df = df.drop(columns=drop_columns, errors="ignore")
    return df


def _log_transform_columns(
    df: pd.DataFrame,
    log_columns
) -> pd.DataFrame:
    if not log_columns:
        return df

    df = df.copy()
    for col in log_columns:
        if col in df.columns:
            df[col] = np.log1p(df[col])
        else:
            log.warn("Column %s not found for log transformation.", col)
    return df


def _one_hot_encode_columns(
    df: pd.DataFrame,
    categorical_columns: list[str] | None,
    drop_first=True
) -> pd.DataFrame:
    if not categorical_columns:
        return df

    df = df.copy()
    existing_cols = [col for col in categorical_columns if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cols, drop_first=drop_first)
    return df


#TODO: Convert distance features to binary indicators. 
def _convert_distance_to_binary(
    df: pd.DataFrame,
    distance_columns: list[str] | None,
    binary_distance_threshold: float,
) -> pd.DataFrame:
    pass


def preprocess_features(
    df: pd.DataFrame,
    drop_cols: list[str] = None,
    log_cols: list[str] = None,
    categorical_cols: list[str] = None,
    distance_cols: list[str] = None,
    binary_distance_threshold: float | None = None
) -> pd.DataFrame:
    log.info("Starting data preprocessing...")
    df = df.copy()

    df = _remove_leakage_columns(df=df, drop_columns=drop_cols)
    df = _log_transform_columns(df=df, log_columns=log_cols)

    #TODO: Convert distance features to binary indicators.
    # df = _convert_distance_to_binary(
    #     df=df,
    #     distance_columns=distance_cols,
    #     binary_distance_threshold=binary_distance_threshold
    # )

    df = _one_hot_encode_columns(df=df, categorical_columns=categorical_cols)

    log.info("Data preprocessing completed.")
    return df
