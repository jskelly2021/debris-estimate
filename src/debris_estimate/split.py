"""owns the train/test split logic."""

import pandas as pd

from dataclasses import dataclass
from typing import Union
from sklearn.model_selection import train_test_split
from debris_estimate.logger import Log

log = Log()

# Target can be a Series or DataFrame
Target = Union[pd.Series, pd.DataFrame]


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: Target
    y_test: Target


def split_data(
    X: pd.DataFrame,
    y: Target,
    test_size: float = 0.2,
    random_state: int = 42
) -> Split:
    train_idx, temp_idx = train_test_split(
        X.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[temp_idx]
    y_test = y.loc[temp_idx]

    log.info(f"Split data into train and test sets with test size {test_size}.")
    log.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}.")

    return Split(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
