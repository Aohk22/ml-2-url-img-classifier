'''
This file contains model pipeline definitions and 
code for training.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]


def _evaluate_binary(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def _build_candidates(random_state: int) -> list[tuple[str, Pipeline]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    # Keep preprocessing simple: numeric features only.

    return [
        (
            "logreg",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ]
            ),
        ),
        (
            "rf",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=400,
                            random_state=random_state,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
        ),
        (
            "hgb",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            random_state=random_state,
                            max_depth=None,
                            learning_rate=0.1,
                        ),
                    ),
                ]
            ),
        ),
    ]


def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    val_size: float = 0.2,
) -> tuple[ModelResult, list[ModelResult]]:
    """
    Train multiple models, pick best by validation ROC-AUC.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    results: list[ModelResult] = []
    for name, pipe in _build_candidates(random_state=random_state):
        pipe.fit(X_tr, y_tr)

        # Prefer predict_proba when available; otherwise use decision_function.
        y_pred = pipe.predict(X_val)
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_val)[:, 1]
        else:
            scores = pipe.decision_function(X_val)
            y_score = 1 / (1 + np.exp(-scores))

        metrics = _evaluate_binary(y_val, y_pred, y_score)
        results.append(ModelResult(name=name, pipeline=pipe, metrics=metrics))

    best = max(results, key=lambda r: r.metrics["roc_auc"])
    return best, results


def evaluate_on_test(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_score = 1 / (1 + np.exp(-scores))
    return _evaluate_binary(y_test, y_pred, y_score)
