import config_presets.baseline as baseline

from pathlib import Path
from debris_estimate.config import (
    PreprocessConfig,
    SplitConfig,
    ClipConfig,
    DataConfig,
    ModelConfig,
    RunConfig,
)

CONFIG_NAME             = "h8_v3_cd"

### Dataset ###
DATASET                 = "data/h8_debrisv3.csv"
DATASET_NAME            = Path(DATASET).stem
RUN_NAME                = f"{DATASET_NAME}_cd_baseline"

### Preprocessing ###
TARGET_COL              = "VolCD_sum"
DROP_COLS               = baseline.DROP_COLS
LOG_COLS                = baseline.LOG_COLS
CATEGORICAL_COLS        = baseline.CATEGORICAL_COLS
DISTANCE_THRESHOLDS     = baseline.DISTANCE_THRESHOLDS
ORDINAL_MAPS            = baseline.ORDINAL_MAPS
HAZARD_FEATURES         = baseline.HAZARD_FEATURES
EXCLUDE_CLIP_COLS       = baseline.EXCLUDE_CLIP_COLS

### Splitting ###
TEST_SIZE               = 0.2
SPLIT_RANDOM_STATE      = 42

### Clipping ###
FCLIP = 0.80
TCLIP  = 1.0

### Model Params ###
ZERO_POS_PARAMS         = baseline.ZERO_POS_PARAMS
TIER_PARAMS             = baseline.TIER_PARAMS
LOW_PARAMS              = baseline.LOW_PARAMS
HIGH_PARAMS             = baseline.HIGH_PARAMS
THRESHOLD               = 500


### Factories ###
def build_preprocess_config() -> PreprocessConfig:
    return PreprocessConfig(
        target_col=TARGET_COL,
        drop_cols=DROP_COLS,
        log_cols=LOG_COLS,
        categorical_cols=CATEGORICAL_COLS,
        distance_thresholds=DISTANCE_THRESHOLDS,
        ordinal_maps=ORDINAL_MAPS,
        hazard_features=HAZARD_FEATURES,
        exclude_clip_cols=EXCLUDE_CLIP_COLS,
    )

def build_split_config() -> SplitConfig:
    return SplitConfig(
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
    )

def build_clip_config() -> ClipConfig:
    return ClipConfig(
        fclip=FCLIP,
        tclip=TCLIP,
    )

def build_data_config() -> DataConfig:
    return DataConfig(
        dataset=DATASET,
        dataset_name=DATASET_NAME,
        preprocess=build_preprocess_config(),
        split=build_split_config(),
        clip=build_clip_config(),
    )

def build_model_config() -> ModelConfig:
    return ModelConfig(
        zero_pos_params=ZERO_POS_PARAMS,
        tier_params=TIER_PARAMS,
        low_params=LOW_PARAMS,
        high_params=HIGH_PARAMS,
        threshold=THRESHOLD,
    )

def build_run_config() -> RunConfig:
    return RunConfig(
        run_name=RUN_NAME,
        data=build_data_config(),
        model=build_model_config(),
    )

