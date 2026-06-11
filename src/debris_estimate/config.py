"""Define configuration dataclasses for the debris estimation pipeline"""

from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    target_col: str | None = None
    drop_cols: list[str] = field(default_factory=list)
    log_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    distance_thresholds: dict[str, float] = field(default_factory=dict)
    ordinal_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    hazard_features: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    exclude_clip_cols: list[str] = field(default_factory=list)


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
    dataset: str | None = None
    dataset_name: str | None = None
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
    run_name: str | None = "run"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class ExperimentConfig:
    experiment_name: str | None = None
    primary_metric: str = "system_r2"
    primary_metric_mode: str = "max"
    swept_fields: list[str] | None = None
