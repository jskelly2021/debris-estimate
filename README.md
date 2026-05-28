# Debris Estimation

## Project Structure

```text
src/
  debris_estimate/
    data.py            # dataset loading
    preprocessing.py   # feature preprocessing
    split.py           # train/test splits
    metrics.py         # metric formulas
    model.py           # staged XGBoost model only
    outputs.py         # standardized output saving
    logger.py          # process wide logging
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
pip install -e .
```
