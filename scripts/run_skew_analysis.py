"""Script to analyze distribution in features."""

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from debris_estimate.logger import Log
from debris_estimate.data import load_dataset
from debris_estimate.config import PreprocessConfig
import config_presets.baseline as baseline

log = Log()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "skew_analysis"
PLOTS_DIR = "histograms"
OUTPUT_PATH = PROJECT_ROOT / OUTPUT_DIR / EXPERIMENT_NAME

DATASETS = [
    "data/GrideH8_v3.csv",
    "data/GrideH9_v3.csv",
    "data/h8_debrisv3.csv",
    "data/h9_debrisv6.csv",
]

def get_numeric_clip_candidates(
    df: pd.DataFrame,
    config: PreprocessConfig
)-> list[str]:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    return [
        col for col in numeric_cols
        if col not in config.categorical_cols
    ]


def analyze_skew(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []

    for col in cols:
        s = df[col].dropna()

        rows.append({
            "col": col,
            "skew": s.skew(),
            "min": s.min(),
            "max": s.max(),
            "avg": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "n_unique": s.nunique(),
            "p50": s.quantile(0.50),
            "p90": s.quantile(0.90),
            "p95": s.quantile(0.95),
            "p99": s.quantile(0.99),
        })

    return pd.DataFrame(rows).sort_values("skew", ascending=False)


def plot_histograms(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in summary["col"]:
        plt.figure()
        df[col].dropna().hist(bins=50)
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}.png")
        plt.close()


def run_skew_analysis():
    config = baseline.build_preprocess_config()

    for dataset in DATASETS:
        data_path = PROJECT_ROOT / dataset

        df = load_dataset(data_path)

        output_dir = OUTPUT_PATH / Path(data_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        cols = get_numeric_clip_candidates(df=df, config=config)
        summary = analyze_skew(df, cols)
        summary.to_csv(output_dir / "skew_summary.csv", index=False)

        plot_histograms(df=df, summary=summary, output_dir=output_dir / PLOTS_DIR)


def main() -> int:
    try:
        run_skew_analysis()
        log.info("Feature skew analysis completed successfully.")
        return 0

    except Exception as e:
        log.error(f"Feature skew analysis failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
