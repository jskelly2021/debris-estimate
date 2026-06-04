"""owns staged model training logic and prediction logic."""

import pandas as pd
import numpy as np

from debris_estimate.logger import Log
from debris_estimate.config import ModelConfig
from debris_estimate.model.predict import PredictionResults
from debris_estimate.model.train import (
    train_zero_pos_classifier,
    train_tier_classifier,
    train_low_high_regressors,
)

log = Log()


class StagedModel:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.zero_pos_classifier = None
        self.tier_classifier = None
        self.low_regressor = None
        self.high_regressor = None
        self.is_fitted: bool = False


    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> None:
        self.zero_pos_classifier = train_zero_pos_classifier(
            X_train=X_train,
            y_train=y_train,
            params=self.config.zero_pos_params
        )

        pos_mask = y_train > 0
        X_train_pos = X_train.loc[pos_mask]
        y_train_pos = y_train.loc[pos_mask]

        self.tier_classifier = train_tier_classifier(
            X_train_pos=X_train_pos,
            y_train_pos=y_train_pos,
            threshold=self.config.threshold,
            params=self.config.tier_params,
        )

        self.low_regressor, self.high_regressor = train_low_high_regressors(
            X_train_pos=X_train_pos,
            y_train_pos=y_train_pos,
            threshold=self.config.threshold,
            low_params=self.config.low_params,
            high_params=self.config.high_params,
        )

        self.is_fitted = True


    def predict(
        self,
        X: pd.DataFrame
    ) -> PredictionResults:
        if not self.is_fitted:
            raise ValueError("StagedModel must be fitted before prediction.")

        # Initialize prediction series
        idx = X.index
        zero_pos_pred = pd.Series(np.nan, index=idx, name="zero_pos_pred")
        zero_pos_prob = pd.Series(np.nan, index=idx, name="zero_pos_prob")
        tier_pred = pd.Series(np.nan, index=idx, name="tier_pred")
        tier_prob = pd.Series(np.nan, index=idx, name="tier_prob")
        low_pred = pd.Series(np.nan, index=idx, name="low_pred")
        high_pred = pd.Series(np.nan, index=idx, name="high_pred")
        reg_pred = pd.Series(np.nan, index=idx, name="reg_pred")
        final_pred = pd.Series(0.0, index=idx, name="final_pred")

        # Predict zero vs positive
        zero_pos_pred = self.zero_pos_classifier.predict(X)
        zero_pos_prob = self.zero_pos_classifier.predict_proba(X)[:, 1]

        positive_mask = zero_pos_pred == 1

        if positive_mask.sum() > 0:
            # Predict low vs high tier, only for positive-predicted samples
            tier_pred[positive_mask] = self.tier_classifier.predict(X.loc[positive_mask])
            tier_prob[positive_mask] = self.tier_classifier.predict_proba(X.loc[positive_mask])[:, 1]

            low_mask = positive_mask & (tier_pred == 0)  # positive and low tier predicted
            high_mask = positive_mask & (tier_pred == 1) # positive and high tier predicted

            # Predict regression values only for positive, low tier samples
            if low_mask.sum() > 0:
                low_pred[low_mask] = np.expm1(self.low_regressor.predict(X.loc[low_mask]))
                low_pred[low_mask] = np.clip(low_pred[low_mask], 0, None)

            # Predict regression values only for positive, high tier samples
            if high_mask.sum() > 0:
                high_pred[high_mask] = np.expm1(self.high_regressor.predict(X.loc[high_mask]))
                high_pred[high_mask] = np.clip(high_pred[high_mask], 0, None)

            # Combine low/high predictions
            reg_pred[low_mask] = low_pred[low_mask]
            reg_pred[high_mask] = high_pred[high_mask]

            # Final prediction is zero for zero-predicted samples
            # and regression prediction for positive-predicted samples
            final_pred[positive_mask] = reg_pred[positive_mask]

        return PredictionResults(
            zero_pos_pred=zero_pos_pred,
            zero_pos_prob=zero_pos_prob,
            tier_pred=tier_pred,
            tier_prob=tier_prob,
            low_pred=low_pred,
            high_pred=high_pred,
            reg_pred=reg_pred,
            final_pred=final_pred
        )
