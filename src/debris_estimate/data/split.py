"""owns the train/test split logic."""

import pandas as pd

from typing import Union
from sklearn.model_selection import train_test_split
from debris_estimate.logger import Log
from debris_estimate.config import SplitConfig

log = Log()

# Target can be a Series or DataFrame
Target = Union[pd.Series, pd.DataFrame]


def split_data(
    X: pd.DataFrame,
    y: Target,
    config: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, Target, Target]:
    train_idx, temp_idx = train_test_split(
        X.index,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[temp_idx]
    y_test = y.loc[temp_idx]

    log.debug(f"Split data into train and test sets with test size {config.test_size}.")
    log.debug(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}.")

    return X_train, X_test, y_train, y_test
