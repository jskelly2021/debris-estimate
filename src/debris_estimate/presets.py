"""Presets for default configurations of the debris estimation pipeline."""

from debris_estimate.config import (
    PreprocessConfig,
    SplitConfig,
    ClipConfig,
    DataConfig,
    ModelConfig,
)


BASELINE_DROP_COLS = [
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
BASELINE_LOG_COLS = ["sqm", "val_struct", "val_cont", "fld_pct"]
BASELINE_CATEGORICAL_COLS = ["landcover", "landuse"]
BASELINE_DISTANCE_THRESHOLDS = {
    "dist_coast": 5.0,
    "dist_reservoir": 3.0,
}
BASELINE_ORDINAL_MAPS = {
    "evac_degree": {"none": 0, "low": 1, "med": 2, "high": 3},
    "fld_zone": {"OPEN WATER": 0, "X": 0, "A": 1, "AO": 1, "AE": 2, "VE": 3},
}
BASELINE_HAZARD_FEATURES = {
    "dist_htrack": ("min", ["dist_htrack_M", "dist_htrack_H"]),
    "windgust": ("max", ["windgust_M", "windgust_H"]),
    "rainfall": ("max", ["rainfall_M", "rainfall_H"]),
}
BASELINE_EXCLUDE_CLIP_COLS = ["sqm", "val_struct", "val_cont", "fld_pct", "evac_degree", "fld_zone"]


BASELINE_PREPROCESS_CONFIG = PreprocessConfig(
    drop_cols=BASELINE_DROP_COLS,
    log_cols=BASELINE_LOG_COLS,
    categorical_cols=BASELINE_CATEGORICAL_COLS,
    distance_thresholds=BASELINE_DISTANCE_THRESHOLDS,
    ordinal_maps=BASELINE_ORDINAL_MAPS,
    hazard_features=BASELINE_HAZARD_FEATURES,
    exclude_clip_cols=BASELINE_EXCLUDE_CLIP_COLS,
)


H8_V3_PREPROCESS_CONFIG = BASELINE_PREPROCESS_CONFIG
H9_V6_PREPROCESS_CONFIG = BASELINE_PREPROCESS_CONFIG
H9_STP_V3_PREPROCESS_CONFIG = BASELINE_PREPROCESS_CONFIG
GH8_V3_PREPROCESS_CONFIG = BASELINE_PREPROCESS_CONFIG
GH9_V3_PREPROCESS_CONFIG = BASELINE_PREPROCESS_CONFIG
GH9_STP_V3_PREPROCESS_CONFIG = BASELINE_PREPROCESS_CONFIG


BASELINE_SPLIT_CONFIG = SplitConfig(
    test_size=0.2,
    random_state=42
)


BASELINE_CLIP_CONFIG = ClipConfig(
    feature_clip_percentile=0.95,
    target_clip_percentile=0.98
)


H9_V6_DATA_CONFIG = DataConfig(
    dataset = "data/h9_debrisv6.csv",
    preprocess = H9_V6_PREPROCESS_CONFIG,
    split = BASELINE_SPLIT_CONFIG,
    clip = BASELINE_CLIP_CONFIG,
)


H8_V3_DATA_CONFIG = DataConfig(
    dataset="data/h8_debrisv3.csv",
    preprocess = H8_V3_PREPROCESS_CONFIG,
    split = BASELINE_SPLIT_CONFIG,
    clip = BASELINE_CLIP_CONFIG,
)


H9_STP_V3_DATA_CONFIG = DataConfig(
    dataset="data/h9_StP_debrisv3.csv",
    preprocess = H9_STP_V3_PREPROCESS_CONFIG,
    split = BASELINE_SPLIT_CONFIG,
    clip = BASELINE_CLIP_CONFIG,
)


GH9_V3_DATA_CONFIG = DataConfig(
    dataset="data/GrideH9_v3.csv",
    preprocess = GH9_V3_PREPROCESS_CONFIG,
    split = BASELINE_SPLIT_CONFIG,
    clip = BASELINE_CLIP_CONFIG,
)


GH9_STP_V3_DATA_CONFIG = DataConfig(
    dataset = "data/GrideH9_StP_v3.csv",
    preprocess = GH9_STP_V3_PREPROCESS_CONFIG,
    split = BASELINE_SPLIT_CONFIG,
    clip = BASELINE_CLIP_CONFIG,
)


GH8_V3_DATA_CONFIG = DataConfig(
    dataset = "data/GrideH8_v3.csv",
    preprocess = GH8_V3_PREPROCESS_CONFIG,
    split = BASELINE_SPLIT_CONFIG,
    clip = BASELINE_CLIP_CONFIG,
)


DEFAULT_CLF_PARAMS_XGB = dict(
    n_estimators     = 50,
    max_depth        = 10,
    min_child_weight = 10,
    colsample_bynode = 0.8,
    random_state     = 42,
    n_jobs           = -1
)


DEFAULT_REG_PARAMS_XGB = dict(
    n_estimators     = 50,
    max_depth        = 6,
    min_child_weight = 1,
    colsample_bynode = 0.8,
    random_state     = 42,
    n_jobs           = -1
)


BASELINE_MODEL_CONFIG = ModelConfig(
    zero_pos_params = DEFAULT_CLF_PARAMS_XGB,
    tier_params = DEFAULT_CLF_PARAMS_XGB,
    low_params = DEFAULT_REG_PARAMS_XGB,
    high_params = DEFAULT_REG_PARAMS_XGB,
    threshold = 300
)
