"""owns the logic for preprocessing features."""

import pandas as pd
import numpy as np

from debris_estimate.logger import Log

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


#TODO: Convert distance features to binary indicators. 
def _convert_distance_to_binary(
    df: pd.DataFrame,
    distance_cols: list[str] | None,
    binary_distance_threshold: float | None,
) -> pd.DataFrame:
    pass


def preprocess_features(
    df: pd.DataFrame,
    drop_cols: list[str] | None = None,
    log_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    distance_cols: list[str] | None = None,
    binary_distance_threshold: float | None = None
) -> pd.DataFrame:
    log.info("Starting data preprocessing...")
    df = df.copy()

    df = _remove_leakage_columns(df=df, drop_cols=drop_cols)
    df = _log_transform_columns(df=df, log_cols=log_cols)

    #TODO: Convert distance features to binary indicators.
    # df = _convert_distance_to_binary(
    #     df=df,
    #     distance_cols=distance_cols,
    #     binary_distance_threshold=binary_distance_threshold
    # )

    df = _one_hot_encode_columns(df=df, categorical_cols=categorical_cols)

    log.info("Data preprocessing completed.")
    return df
