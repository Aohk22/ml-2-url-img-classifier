from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from .constants import DATASET_ID
from .data import DatasetSplits, infer_feature_columns, load_hf_splits, map_labels, split_xy
from .train import ModelResult, evaluate_on_test, train_and_select


@dataclass(frozen=True)
class ExperimentResult:
    dataset_id: str
    feature_cols: list[str]
    best_model: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]


def run_experiment(
    splits: DatasetSplits,
    random_state: int = 42,
    val_size: float = 0.2,
) -> tuple[ExperimentResult, ModelResult, list[ModelResult]]:
    train_df = map_labels(splits.train)
    test_df = map_labels(splits.test)

    feature_cols = infer_feature_columns(train_df)
    X_train, y_train = split_xy(train_df, feature_cols)
    X_test, y_test = split_xy(test_df, feature_cols)

    best, candidates = train_and_select(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        val_size=val_size,
    )

    # Refit best model on the full training split before final test eval.
    best.pipeline.fit(X_train, y_train)
    test_metrics = evaluate_on_test(best.pipeline, X_test, y_test)

    exp = ExperimentResult(
        dataset_id=DATASET_ID,
        feature_cols=feature_cols,
        best_model=best.name,
        val_metrics=best.metrics,
        test_metrics=test_metrics,
    )
    return exp, best, candidates


def save_bundle(
    out_path: str | Path,
    model_result: ModelResult,
    experiment: ExperimentResult,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "dataset_id": experiment.dataset_id,
        "feature_cols": experiment.feature_cols,
        "best_model": experiment.best_model,
        "val_metrics": experiment.val_metrics,
        "test_metrics": experiment.test_metrics,
        "model": model_result.pipeline,
    }
    joblib.dump(bundle, out_path)
    return out_path


def load_default_splits(dataset_id: str = DATASET_ID) -> DatasetSplits:
    return load_hf_splits(dataset_id=dataset_id)
