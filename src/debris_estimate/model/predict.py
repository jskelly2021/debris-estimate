
import pandas as pd

from dataclasses import dataclass

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

    def tier_pairs(self, y_true: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        mask = self.tier_pred.notna()
        return y_true.loc[mask], self.tier_pred.loc[mask], self.tier_prob.loc[mask]

    def low_pairs(self, y_true: pd.Series) -> tuple[pd.Series, pd.Series]:
        mask = self.low_pred.notna()
        return y_true.loc[mask], self.low_pred.loc[mask]

    def high_pairs(self, y_true: pd.Series) -> tuple[pd.Series, pd.Series]:
        mask = self.high_pred.notna()
        return y_true.loc[mask], self.high_pred.loc[mask]

    def reg_pairs(self, y_true: pd.Series) -> tuple[pd.Series, pd.Series]:
        mask = self.reg_pred.notna()
        return y_true.loc[mask], self.reg_pred.loc[mask]
