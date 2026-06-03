"""Presets for default configurations of the debris estimation pipeline."""

from debris_estimate.config import PreprocessConfig


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
    log_cols  = ["sqm", "val_struct", "val_cont", "fld_pct"],
    categorical_cols = ["evac_degree", "fld_zone", "landcover", "landuse"],
    distance_cols = ["dist_coast", "dist_reservoir", "dist_htrack_M", "dist_htrack_H"],
)
