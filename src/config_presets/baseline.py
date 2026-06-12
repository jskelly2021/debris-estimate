"""Presets for BASELINE configurations of the debris estimation pipeline."""

from pathlib import Path
from debris_estimate.config import (
    PreprocessConfig,
    SplitConfig,
    ClipConfig,
    DataConfig,
    ModelConfig,
    RunConfig,
)

### Dataset ###
DATASET                 = "data/GrideH8_v3.csv"
DATASET_NAME            = Path(DATASET).stem
RUN_NAME                = f"{DATASET_NAME}_baseline"

### Preprocessing ###
TARGET_COL              = "VolBoth_sum"
DROP_COLS               = [
                            "GRID_ID",
                            "age_med",
                            "num_story_sum", "num_story_med",
                            "sqm_sum", "sqm_med",
                            "found_ht_med", "found_ht_sum",
                            "val_struct_sum", "val_struct_med",
                            "val_cont_med", "val_cont_sum",
                            "VolCD", "VolVG",
                            "VolCD_sum", "VolVG_sum", "VolBoth_sum",
                            "Bin_CD", "Bin_VG", "Bin_Both"
                        ]
LOG_COLS                = ["sqm", "val_struct", "val_cont", "fld_pct"]
CATEGORICAL_COLS        = ["landcover", "landuse"]
DISTANCE_THRESHOLDS     = {
                            "dist_coast": 5.0,
                            "dist_reservoir": 3.0,
                        }
ORDINAL_MAPS            = {
                            "evac_degree": {"none": 0, "low": 1, "med": 2, "high": 3},
                            "fld_zone": {"OPEN WATER": 0, "X": 0, "A": 1, "AO": 1, "AE": 2, "VE": 3},
                        }
HAZARD_FEATURES         = {
                            "dist_htrack": ("min", ["dist_htrack_M", "dist_htrack_H"]),
                            "windgust": ("max", ["windgust_M", "windgust_H"]),
                            "rainfall": ("max", ["rainfall_M", "rainfall_H"]),
                        }
EXCLUDE_CLIP_COLS       = ["sqm", "val_struct", "val_cont", "fld_pct", "evac_degree", "fld_zone"]

### Splitting ###
TEST_SIZE               = 0.2
SPLIT_RANDOM_STATE      = 42

### Clipping ###
FCLIP = 0.99
TCLIP  = 0.95

### Model Params ###
ZERO_POS_PARAMS         = dict(
                            n_estimators     = 50,
                            max_depth        = 10,
                            min_child_weight = 10,
                            colsample_bynode = 0.8,
                            random_state     = 42,
                            n_jobs           = -1
                        )
TIER_PARAMS             = dict(
                            n_estimators     = 50,
                            max_depth        = 10,
                            min_child_weight = 10,
                            colsample_bynode = 0.8,
                            random_state     = 42,
                            n_jobs           = -1
                        )
LOW_PARAMS              = dict(
                            n_estimators     = 50,
                            max_depth        = 6,
                            min_child_weight = 1,
                            colsample_bynode = 0.8,
                            random_state     = 42,
                            n_jobs           = -1
                        )
HIGH_PARAMS             = dict(
                            n_estimators     = 50,
                            max_depth        = 6,
                            min_child_weight = 1,
                            colsample_bynode = 0.8,
                            random_state     = 42,
                            n_jobs           = -1
                        )
THRESHOLD               = 800


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
