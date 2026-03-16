from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .feature_extraction import UrlFeatureExtractor


@dataclass(frozen=True)
class UrlPrediction:
    url: str
    phishing_proba: float
    is_phishing: bool


def load_url_pipeline(path: str | Path) -> Any:
    """
    Load a sklearn model (typically a Pipeline) saved via `joblib.dump`.

    Models in this repo's `models/` folder were created by notebooks:
    - `models/url_clf_features_only.joblib`
    - `models/url_clf_w_embedding.joblib`
    """
    import joblib

    return joblib.load(Path(path))


def predict_urls(
    pipeline: Any,
    urls: Iterable[str],
    *,
    threshold: float = 0.5,
) -> list[UrlPrediction]:
    """
    Predict phishing probability for URLs using a loaded sklearn pipeline.

    This function rebuilds the training-time feature frame with
    `UrlFeatureExtractor` so you can pass raw URL strings at inference time.
    """
    extractor = UrlFeatureExtractor()
    urls_list = [str(u) for u in urls]
    features = extractor.extract_many(urls_list)

    # Two saved variants exist:
    # 1) Features-only model: expects numeric columns only (no "url").
    # 2) Embedding+features model: expects a "url" column + numeric feature columns.
    uses_embedding = _pipeline_uses_embedding(pipeline)
    X = features if uses_embedding else features.drop(columns=["url"], errors="ignore")

    phishing_proba = _predict_phishing_proba(pipeline, X)
    out: list[UrlPrediction] = []
    for u, p in zip(urls_list, phishing_proba, strict=False):
        p = float(p)
        out.append(UrlPrediction(url=u, phishing_proba=p, is_phishing=p >= float(threshold)))
    return out


def _pipeline_uses_embedding(pipeline: Any) -> bool:
    """Heuristic for the embedding+features pipeline variant."""
    named_steps = getattr(pipeline, "named_steps", None)
    if isinstance(named_steps, dict) and "features" in named_steps:
        return True
    return False


def _predict_phishing_proba(pipeline: Any, X: Any) -> np.ndarray:
    """Return p(y=phishing) for each row."""
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        return np.asarray(proba)[:, 1]

    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))

    raise TypeError("Model does not expose predict_proba or decision_function.")
