"""Orchestration of the staged model."""

import pandas as pd

from debris_estimate.logger import Log
from debris_estimate.config import ModelConfig
from debris_estimate.model.predict import predict_staged_model
from debris_estimate.model.results import PredictionResults, FeatureImportanceResults
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
        log.debug("Training model...")
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
        log.debug("Model training complete.")


    def predict_details(
        self,
        X: pd.DataFrame
    ) -> PredictionResults:
        if not self.is_fitted:
            raise ValueError("StagedModel must be fitted before prediction.")

        return predict_staged_model(
            X=X,
            zero_pos_classifier=self.zero_pos_classifier,
            tier_classifier=self.tier_classifier,
            low_regressor=self.low_regressor,
            high_regressor=self.high_regressor,
        )


    def predict(
        self,
        X: pd.DataFrame
    ) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("StagedModel must be fitted before prediction.")

        pred_details = self.predict_details(X)
        return pred_details.final_pred


    def feature_importance(
        self,
        importance_type: str = "gain",
    ) -> FeatureImportanceResults:
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before extracting feature importances."
            )

        return FeatureImportanceResults(
            zero_pos=_get_feature_importance_df(
                self.zero_pos_classifier,
                importance_type,
            ),
            tier=_get_feature_importance_df(
                self.tier_classifier,
                importance_type,
            ),
            low=_get_feature_importance_df(
                self.low_regressor,
                importance_type,
            ),
            high=_get_feature_importance_df(
                self.high_regressor,
                importance_type,
            ),
            importance_type=importance_type,
        )


def _get_feature_importance_df(
    model,
    importance_type: str,
) -> pd.DataFrame:
    scores = model.get_booster().get_score(
        importance_type=importance_type
    )

    total = sum(scores.values())

    rows = [
        {
            "feature": feature,
            "importance": importance / total if total > 0 else importance,
        }
        for feature, importance in scores.items()
    ]

    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
