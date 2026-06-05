"""Presets for default configurations of the debris estimation pipeline."""

from debris_estimate.config import (
    PreprocessConfig,
    SplitConfig,
    ClipConfig,
    DataConfig,
    ModelConfig,
)


H9_V6_PREPROCESS_CONFIG = PreprocessConfig(
    drop_cols = [
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
    ],
    log_cols  = [ "sqm", "val_struct", "val_cont", "fld_pct"],
    categorical_cols = ["landcover", "landuse", "evac_degree", "fld_zone",],
    distance_thresholds = {
        "dist_coast": 5.0,
        "dist_reservoir": 3.0,
    },
    ordinal_maps = {
        "evac_degree": {"none": 0, "low": 1, "med": 2, "high": 3},
        "fld_zone": {"OPEN WATER": 0, "X": 0, "A": 1, "AO": 1, "AE": 2, "VE": 3},
    },
    hazard_features = {
        "dist_htrack": ("min", ["dist_htrack_M", "dist_htrack_H"]),
        "windgust": ("max", ["windgust_M", "windgust_H"]),
        "rainfall": ("max", ["rainfall_M", "rainfall_H"]),
    },
    exclude_clip_cols = [
        "sqm", "val_struct", "val_cont", "fld_pct", "evac_degree", "fld_zone",
    ],
)


BASELINE_SPLIT_CONFIG = SplitConfig(
    test_size=0.2,
    random_state=42
)


BASELINE_CLIP_CONFIG = ClipConfig(
    feature_clip_percentile=0.99,
    target_clip_percentile=0.99
)


H9_V6_DATA_CONFIG = DataConfig(
    dataset = "data/h9_debrisv6.csv",
    preprocess = H9_V6_PREPROCESS_CONFIG,
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
