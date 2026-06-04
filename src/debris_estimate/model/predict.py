"""Prediction routing logic for the staged model."""

import numpy as np
import pandas as pd

from debris_estimate.model.results import PredictionResults


def predict_staged_model(
    X: pd.DataFrame,
    zero_pos_classifier,
    tier_classifier,
    low_regressor,
    high_regressor,
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
    zero_pos_pred = zero_pos_classifier.predict(X)
    zero_pos_prob = zero_pos_classifier.predict_proba(X)[:, 1]

    positive_mask = zero_pos_pred == 1

    if positive_mask.sum() > 0:
        # Predict low vs high tier, only for positive-predicted samples
        tier_pred[positive_mask] = tier_classifier.predict(X.loc[positive_mask])
        tier_prob[positive_mask] = tier_classifier.predict_proba(X.loc[positive_mask])[:, 1]

        low_mask = positive_mask & (tier_pred == 0)  # positive and low tier predicted
        high_mask = positive_mask & (tier_pred == 1) # positive and high tier predicted

        # Predict regression values only for positive, low tier samples
        if low_mask.sum() > 0:
            low_pred[low_mask] = np.expm1(low_regressor.predict(X.loc[low_mask]))
            low_pred[low_mask] = np.clip(low_pred[low_mask], 0, None)

        # Predict regression values only for positive, high tier samples
        if high_mask.sum() > 0:
            high_pred[high_mask] = np.expm1(high_regressor.predict(X.loc[high_mask]))
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
