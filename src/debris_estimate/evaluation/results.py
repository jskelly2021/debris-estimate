
from dataclasses import dataclass
from debris_estimate.evaluation.metrics import ClassificationMetrics, RegressionMetrics


@dataclass
class EvaluationResults:
    system_metrics: RegressionMetrics
    zero_pos_classifier_metrics: ClassificationMetrics
    tier_classifier_metrics: ClassificationMetrics
    low_regressor_metrics: RegressionMetrics
    high_regressor_metrics: RegressionMetrics
    full_regressor_metrics: RegressionMetrics
