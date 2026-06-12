"""Define configuration dataclasses for the debris estimation pipeline"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

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
    fclip: float | None = None
    tclip: float | None = None
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
    run_name: str
    output_dir: str | Path = "outputs/runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class ExperimentConfig:
    experiment_name: str
    output_dir: str | Path = "outputs"
    base_run_config: RunConfig = field(default_factory=RunConfig)
    primary_metric: str = "system_r2"
    primary_metric_mode: str = "max"
    swept_fields: dict[str, list[Any]] = field(default_factory=dict)
