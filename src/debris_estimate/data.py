"""Load dataset from a given csv file path."""

import pandas as pd

from pathlib import Path
from debris_estimate.logger import Log

log = Log()


def load_dataset(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    log.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)

    log.info("Dataset loaded with shape %s", df.shape)
    return df
