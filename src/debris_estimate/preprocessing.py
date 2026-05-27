"""owns the logic for turning a dataframe into model-ready inputs."""

import pandas as pd
import numpy as np

from debris_estimate.logger import Log

log = Log()

drop_cols = [
    "VolCD","VolVG","VolCD_sum","VolVG_sum","VolBoth_sum",
    "Bin_CD","Bin_VG","Bin_Both",
    "GRID_ID",
    "age_sum","age_med","num_story_sum","num_story_med",
    "sqft_sum","sqft_med",
    "found_ht_sum","found_ht_med",
    "val_struct_sum","val_struct_med",
    "val_cont_sum","val_cont_med","landuse"
]

def remove_leakage_columns(df):
    df = df.copy()
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

def log_transform_columns(df, columns) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = np.log1p(df[col])
    return df


def one_hot_encode_columns(df, columns) -> pd.DataFrame:
    df = df.copy()
    return pd.get_dummies(df, columns=columns, drop_first=True)


def preprocess_data(df):
    log.info("Starting data preprocessing.")
    df = df.copy()

    # Remove columns that leak information about the target variable
    df = remove_leakage_columns(df)

    # Log-transform skewed numerical features
    skewed_cols = ["feature1", "feature2"]  # Replace with actual column names
    df = log_transform_columns(df, skewed_cols)

    # One-hot encode categorical features
    categorical_cols = ["category1", "category2"]  # Replace with actual column names
    df = one_hot_encode_columns(df, categorical_cols)

    log.info("Data preprocessing completed.")
    return df
