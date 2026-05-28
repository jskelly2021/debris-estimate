# Debris Estimation

## Project Structure

```text
src/
  debris_estimate/
    data.py            # dataset loading
    evaluation.py      # model evaluation
    logger.py          # process wide logging
    metrics.py         # metric formulas
    model.py           # staged XGBoost model only
    outputs.py         # standardized output saving
    preprocessing.py   # feature preprocessing
    split.py           # train/test splits
scripts/
  run_smoke_test.py    # one-clip, one-threshold staged model smoke check
docs/
  experiments.md       # future experiment ideas and notes
  roadmap.md           # implementation direction
notebooks/             # legacy; core logic extracted
```

## Pipeline
```text
Stage 1:
    classify zero vs positive
    (SMOTE applied)

Stage 2:
    among positives, classify low vs high
    (SMOTE applied)

Stage 3:
    regress exact amount within low/high tier
    (NO SMOTE)
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
