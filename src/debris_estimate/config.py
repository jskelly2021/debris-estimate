""""""

from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    drop_cols: list[str] = field(default_factory=list)
    log_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    distance_cols: list[str] = field(default_factory=list)
    binary_distance_threshold: float | None


@dataclass
class SplitConfig:
    test_size: float | None
    random_state: int | None


@dataclass
class ClipConfig:
    feature_clip_percentile: float | None
    target_clip_percentile: float | None


@dataclass
class ExperimentConfig:
    preprocess: PreprocessConfig
    split: SplitConfig
    clip: ClipConfig
