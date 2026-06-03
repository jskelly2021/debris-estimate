""""""

from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    drop_cols: list[str] = field(default_factory=list)
    log_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    distance_cols: list[str] = field(default_factory=list)
    feature_clip_percentile: float | None = 1.0
