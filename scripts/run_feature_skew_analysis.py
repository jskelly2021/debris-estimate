"""Script to analyze distribution in features."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from debris_estimate.logger import Log
from debris_estimate.data import load_dataset
from debris_estimate.config import PreprocessConfig
from presets.baseline import BASELINE_PREPROCESS_CONFIG

log = Log()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "feature_skew"
OUTPUT_PATH = PROJECT_ROOT / OUTPUT_DIR / EXPERIMENT_NAME


def get_numeric_clip_candidates(
    df: pd.DataFrame,
    config: PreprocessConfig
)-> list[str]:
    # df = df.drop(columns=config.drop_cols, errors="ignore")
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
            "feature": col,
            "skew": s.skew(),
            "min": s.min(),
            "p50": s.quantile(0.50),
            "p90": s.quantile(0.90),
            "p95": s.quantile(0.95),
            "p99": s.quantile(0.99),
            "max": s.max(),
            "avg": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "n_unique": s.nunique(),
        })

    return pd.DataFrame(rows).sort_values("skew", ascending=False)


def plot_feature_histograms(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature in summary["feature"]:
        plt.figure()
        df[feature].dropna().hist(bins=50)
        plt.title(f"{feature} distribution")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"{feature}.png")
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze distribution of features.")
    parser.add_argument("--data_path", type=str, default="data/h9_debrisv6.csv", help="Path to the input dataset.")
    return parser.parse_args()


def run_feature_skew_analysis(args):
    data_path = PROJECT_ROOT / args.data_path

    df = load_dataset(data_path)

    output_path = OUTPUT_PATH / Path(args.data_path).stem

    output_path.mkdir(parents=True, exist_ok=True)

    cols = get_numeric_clip_candidates(df, BASELINE_PREPROCESS_CONFIG)
    summary = analyze_skew(df, cols)
    summary.to_csv(output_path / "feature_skew_summary.csv", index=False)


    plot_feature_histograms(df, summary, output_path / "histograms")

    print(summary)


def main() -> int:
    args = parse_args()

    try:
        run_feature_skew_analysis(args)
        log.info("Feature skew analysis completed successfully.")
        return 0

    except Exception as e:
        log.error(f"Feature skew analysis failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
