
import pandas as pd

from pathlib import Path
from debris_estimate.config import RunConfig
from debris_estimate.data import (
    load_dataset,
    preprocess_features,
    split_data,
    clip_features,
    clip_targets,
)
from debris_estimate.model import StagedModel
from debris_estimate.evaluation import create_evaluation_figures, evaluate_staged_model
from debris_estimate.outputs import save_run_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_valid_threshold(y_train: pd.Series, threshold: float, min_samples: int = 5) -> bool:
    y_pos = y_train[y_train > 0]

    low_count = int((y_pos <= threshold).sum())
    high_count = int((y_pos > threshold).sum())

    return low_count >= min_samples and high_count >= min_samples


def run_model(config: RunConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    ### Data ###
    data_path = PROJECT_ROOT / config.data.dataset
    df = load_dataset(path=data_path)

    ### Preprocessing ###
    X = preprocess_features(
        df=df,
        config=config.data.preprocess,
    )

    ### Splitting ###
    y = df[config.data.preprocess.target_col]

    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        config=config.data.split,
    )

    ### Clipping ###
    X_train_clipped, X_test_clipped = clip_features(
        X_train=X_train,
        X_test=X_test,
        exclude_cols=config.data.preprocess.exclude_clip_cols,
        config=config.data.clip,
    )

    y_train_clipped, _, _, _ = clip_targets(
        y=y_train,
        config=config.data.clip,
    )

    if not is_valid_threshold(y_train_clipped, config.model.threshold, min_samples=5):
        raise ValueError(
            f"Skipping | threshold={config.model.threshold} | "\
            f"feature clip={config.data.clip.fclip} | "\
            f"target clip={config.data.clip.tclip} | "\
            f"not enough low/high samples."
        )

    ### Training ###
    staged_model = StagedModel(config=config.model)
    staged_model.fit(X_train=X_train_clipped, y_train=y_train_clipped)

    ### Feature Importance ###
    feature_importance_results = staged_model.feature_importance(importance_type="gain")

    ### Prediction ###
    pred_results = staged_model.predict_details(X=X_test_clipped)

    ### Evaluation ###
    eval_results = evaluate_staged_model(
        y_true=y_test,
        pred_results=pred_results,
        threshold=config.model.threshold,
    )

    figure_groups = create_evaluation_figures(
        y_true=y_test,
        pred_results=pred_results,
        eval_results=eval_results,
        feature_importance_results=feature_importance_results,
        threshold=config.model.threshold,
    )

    ### Output ###
    save_run_outputs(
        output_path=run_dir,
        eval_results=eval_results,
        y_true=y_test,
        pred_results=pred_results,
        feature_importance_results=feature_importance_results,
        run_config=config,
        figure_groups=figure_groups,
    )
