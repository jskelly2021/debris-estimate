"""Simple, dumb file loading"""

from pathlib import Path

import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    data_path = Path(path)
    return pd.read_excel(data_path)