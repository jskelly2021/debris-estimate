"""owns staged model training logic and prediction logic."""

import pandas as pd
import numpy as np

from dataclasses import dataclass
from xgboost import XGBClassifier, XGBRegressor
from debris_estimate.logger import Log
from debris_estimate.resample import apply_smote
from debris_estimate.config import ModelConfig

log = Log()


@dataclass
class StagedModel:
    zero_pos_classifier: XGBClassifier
    tier_classifier: XGBClassifier
    low_regressor: XGBRegressor
    high_regressor: XGBRegressor


@dataclass
class PredictionResults:
    zero_pos_pred: pd.Series
    zero_pos_prob: pd.Series
    tier_pred: pd.Series
    tier_prob: pd.Series
    low_pred: pd.Series
    high_pred: pd.Series
    reg_pred: pd.Series
    final_pred: pd.Series

    def positive_mask(self) -> pd.Series:
        return self.zero_pos_pred == 1
    
    def tier_mask(self) -> pd.Series:
        return self.tier_pred.notna()

    def low_mask(self) -> pd.Series:
        return self.low_pred.notna()
    
    def high_mask(self) -> pd.Series:
        return self.high_pred.notna()
    
    def reg_mask(self) -> pd.Series:
        return self.reg_pred.notna()
    
    def tier_pairs(self, y_true: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        mask = self.tier_mask()
        return y_true.loc[mask], self.tier_pred.loc[mask], self.tier_prob.loc[mask]

    def low_pairs(self, y_true: pd.Series) -> tuple[pd.Series, pd.Series]:
        mask = self.low_mask()
        return y_true.loc[mask], self.low_pred.loc[mask]

    def high_pairs(self, y_true: pd.Series) -> tuple[pd.Series, pd.Series]:
        mask = self.high_mask()
        return y_true.loc[mask], self.high_pred.loc[mask]

    def reg_pairs(self, y_true: pd.Series) -> tuple[pd.Series, pd.Series]:
        mask = self.reg_mask()
        return y_true.loc[mask], self.reg_pred.loc[mask]


def train_zero_pos_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict
):
    # Get binary labels for classification
    y_train = (y_train > 0).astype(int)

    X_resampled, y_resampled = apply_smote(X_train, y_train)

    log.info("Training zero vs positive classfier...")

    model = XGBClassifier(**params)
    model.fit(X_resampled, y_resampled)

    log.info("Zero vs positive classifier training complete.")

    return model


def train_tier_classifier(
    X_train_pos: pd.DataFrame,
    y_train_pos: pd.Series,
    threshold: float,
    params: dict
):
    y_tier = (y_train_pos > threshold).astype(int)

    X_resampled, y_resampled = apply_smote(X_train_pos, y_tier)

    log.info("Training tier classfier...")

    model = XGBClassifier(**params)
    model.fit(X_resampled, y_resampled)

    log.info("Tier classifier training complete.")

    return model


def train_low_high_regressors(
    X_train_pos: pd.DataFrame,
    y_train_pos: pd.Series,
    threshold: float,
    low_params: dict,
    high_params: dict,
):
    low_mask = y_train_pos <= threshold
    high_mask = y_train_pos > threshold

    if low_mask.sum() == 0:
        raise ValueError("Cannot train low regressor: no low-tier samples.")

    if high_mask.sum() == 0:
        raise ValueError("Cannot train high regressor: no high-tier samples.")

    X_low = X_train_pos.loc[low_mask]
    y_low = np.log1p(y_train_pos.loc[low_mask])

    X_high = X_train_pos.loc[high_mask]
    y_high = np.log1p(y_train_pos.loc[high_mask])

    log.info("Training low regressor...")
    model_low = XGBRegressor(**low_params)
    model_low.fit(X_low, y_low)

    log.info("Training high regressor...")
    model_high = XGBRegressor(**high_params)
    model_high.fit(X_high, y_high)

    log.info("Low/high regressor training complete.")

    return model_low, model_high


def train_staged_model(
    X_train: pd.Series,
    y_train: pd.Series,
    config: ModelConfig
) -> StagedModel:
    zero_pos_classifier = train_zero_pos_classifier(
        X_train=X_train,
        y_train=y_train,
        params=config.zero_pos_params
    )

    pos_mask = y_train > 0
    X_train_pos = X_train.loc[pos_mask]
    y_train_pos = y_train.loc[pos_mask]

    tier_classifier = train_tier_classifier(
        X_train_pos=X_train_pos,
        y_train_pos=y_train_pos,
        threshold=config.threshold,
        params=config.tier_params,
    )

    low_regressor, high_regressor = train_low_high_regressors(
        X_train_pos=X_train_pos,
        y_train_pos=y_train_pos,
        threshold=config.threshold,
        low_params=config.low_params,
        high_params=config.high_params,
    )

    return StagedModel(
        zero_pos_classifier=zero_pos_classifier,
        tier_classifier=tier_classifier,
        low_regressor=low_regressor,
        high_regressor=high_regressor
    )


def predict_staged_model(
    X: pd.DataFrame,
    model: StagedModel
) -> PredictionResults:
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
    zero_pos_pred = model.zero_pos_classifier.predict(X)
    zero_pos_prob = model.zero_pos_classifier.predict_proba(X)[:, 1]

    positive_mask = zero_pos_pred == 1

    if positive_mask.sum() > 0:
        # Predict low vs high tier, only for positive-predicted samples
        tier_pred[positive_mask] = model.tier_classifier.predict(X.loc[positive_mask])
        tier_prob[positive_mask] = model.tier_classifier.predict_proba(X.loc[positive_mask])[:, 1]

        low_mask = positive_mask & (tier_pred == 0)  # positive and low tier predicted
        high_mask = positive_mask & (tier_pred == 1) # positive and high tier predicted

        # Predict regression values only for positive, low tier samples
        if low_mask.sum() > 0:
            low_pred[low_mask] = np.expm1(model.low_regressor.predict(X.loc[low_mask]))
            low_pred[low_mask] = np.clip(low_pred[low_mask], 0, None)

        # Predict regression values only for positive, high tier samples
        if high_mask.sum() > 0:
            high_pred[high_mask] = np.expm1(model.high_regressor.predict(X.loc[high_mask]))
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
