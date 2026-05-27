# Debris Estimation

## Project Structure

```text
src/
  data.py            # dataset loading
  preprocessing.py   # features and target clipping
  splits.py          # train/test splits
  metrics.py         # metric formulas
  model.py           # staged XGBoost model only
  outputs.py         # standardized output saving
scripts/
  run_smoke_test.py  # one-clip, one-threshold staged model smoke check
docs/
  experiments.md     # future experiment ideas and notes
  roadmap.md         # implementation direction
notebooks/           # legacy; core logic extracted
```

## Setup

Create a python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn openpyxl
```
