"""Define configuration dataclasses for the debris estimation pipeline"""

from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    drop_cols: list[str] = field(default_factory=list)
    log_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    distance_col_threshold_map: dict[str, float] = field(default_factory=dict)


@dataclass
class SplitConfig:
    test_size: float | None = None
    random_state: int | None = None


@dataclass
class ClipConfig:
    feature_clip_percentile: float | None = None
    target_clip_percentile: float | None = None
    positive_only_target_clip: bool = True


@dataclass
class DataConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)


@dataclass
class ModelConfig:
    zero_pos_params: dict = field(default_factory=dict)
    tier_params: dict = field(default_factory=dict)
    low_params: dict = field(default_factory=dict)
    high_params: dict = field(default_factory=dict)
    threshold: float | None = None


@dataclass
class RunConfig:
    experiment_name: str | None = None
    run_name: str | None = "run"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
