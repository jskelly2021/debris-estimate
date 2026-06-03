""""""

from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    drop_cols: list[str] = field(default_factory=list)
    log_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    distance_cols: list[str] = field(default_factory=list)
    binary_distance_threshold: float | None = None


@dataclass
class SplitConfig:
    test_size: float | None = None
    random_state: int | None = None


@dataclass
class ClipConfig:
    feature_clip_percentile: float | None = None
    target_clip_percentile: float | None = None


@dataclass
class ModelConfig:
    zero_pos_params: dict = field(default_factory=dict)
    tier_params: dict = field(default_factory=dict)
    low_params: dict = field(default_factory=dict)
    high_params: dict = field(default_factory=dict)
    threshold: float | None = None


@dataclass
class ExperimentConfig:
    name: str | None = None
    preprocess: PreprocessConfig
    split: SplitConfig
    clip: ClipConfig
    model: ModelConfig
