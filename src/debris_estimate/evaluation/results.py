
from dataclasses import dataclass
from debris_estimate.evaluation.metrics import ClassificationMetrics, RegressionMetrics


@dataclass
class EvaluationResults:
    system: RegressionMetrics
    zero_pos: ClassificationMetrics
    tier: ClassificationMetrics
    low: RegressionMetrics
    high: RegressionMetrics
    full: RegressionMetrics
