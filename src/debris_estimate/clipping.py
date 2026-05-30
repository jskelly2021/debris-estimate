"""Target clipping module"""

import pandas as pd

from dataclasses import dataclass
from debris_estimate.logger import Log

log = Log()

@dataclass
class TargetClipResult:
    y_clipped: pd.Series
    upper: float
    n_clipped: int
    percent_clipped: float


def clip_target(
    y: pd.Series,
    percentile: float = 0.99,
    positive_only: bool = True,
) -> TargetClipResult:
    if not 0 < percentile <= 1:
        raise ValueError("target clip percentile must be between 0 and 1.")

    y_clipped = y.copy()
    upper = float("nan")
    n_clipped = 0
    percent_clipped = 0.0

    quantile_values = y[y > 0] if positive_only else y

    if not quantile_values.empty:
        log.info(f"Clipping training targets at {percentile:.3f}")
        upper = quantile_values.quantile(percentile)

        clip_mask = y > upper
        n_clipped = int(clip_mask.sum())
        percent_clipped = n_clipped / len(y) * 100

        y_clipped = y_clipped.clip(upper=upper)

    return TargetClipResult(
        y_clipped=y_clipped,
        upper=float(upper),
        n_clipped=n_clipped,
        percent_clipped=float(percent_clipped),
    )
