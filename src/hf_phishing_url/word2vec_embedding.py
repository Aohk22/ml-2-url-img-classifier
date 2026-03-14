from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .feature_extraction import UrlTokenizer

TokenSource = Literal["raw", "host", "path", "subdomain"]
Pooling = Literal["mean", "sum", "max"]


class UrlWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Word2Vec vector embedding for URLs.

    Trains a Word2Vec model over URL tokens (via `UrlTokenizer`) and converts each
    URL into a fixed-size dense vector by pooling token vectors.

    Requires `gensim` at runtime.
    """

    def __init__(
        self,
        *,
        sources: Sequence[TokenSource] = ("raw", "host", "path", "subdomain"),
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,
        negative: int = 5,
        sample: float = 1e-3,
        epochs: int = 10,
        workers: int = 1,
        seed: int = 42,
        pooling: Pooling = "mean",
        normalize: bool = False,
        feature_prefix: str = "w2v__",
    ) -> None:
        self.sources = tuple(sources)
        self.vector_size = int(vector_size)
        self.window = int(window)
        self.min_count = int(min_count)
        self.sg = int(sg)
        self.negative = int(negative)
        self.sample = float(sample)
        self.epochs = int(epochs)
        self.workers = int(workers)
        self.seed = int(seed)
        self.pooling = pooling
        self.normalize = bool(normalize)
        self.feature_prefix = feature_prefix

        self._tokenizer = UrlTokenizer()
        self._model = None

    def fit(self, X: Iterable[str], y=None):  # noqa: ANN001
        try:
            from gensim.models import Word2Vec  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "UrlWord2VecVectorizer requires `gensim`. "
                "Install it (e.g. `pip install gensim`) to use Word2Vec embeddings."
            ) from e

        urls = list(X)
        sentences = [self._tokens(u) for u in urls]

        self._model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            negative=self.negative,
            sample=self.sample,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
        )
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("UrlWord2VecVectorizer is not fitted. Call fit() first.")

        urls = list(X)
        out = np.zeros((len(urls), self.vector_size), dtype=np.float32)

        for i, u in enumerate(urls):
            toks = self._tokens(u)
            vecs = [self._model.wv[t] for t in toks if t in self._model.wv]
            if not vecs:
                continue

            mat = np.asarray(vecs, dtype=np.float32)
            if self.pooling == "mean":
                v = mat.mean(axis=0)
            elif self.pooling == "sum":
                v = mat.sum(axis=0)
            elif self.pooling == "max":
                v = mat.max(axis=0)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            if self.normalize:
                n = float(np.linalg.norm(v))
                if n > 0:
                    v = v / n

            out[i] = v

        return out

    def fit_transform(self, X, y=None, **fitparams) -> np.ndarray:
        self.fit(X, y=y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        return np.asarray(
            [f"{self.feature_prefix}{i}" for i in range(self.vector_size)],
            dtype=object,
        )

    def _tokens(self, url: str) -> list[str]:
        tokens = self._tokenizer.tokenize(url or "")

        parts: list[str] = []
        for src in self.sources:
            if src == "raw":
                parts.extend(tokens["raw_tokens"])
            elif src == "host":
                parts.extend(tokens["host_tokens"])
            elif src == "path":
                parts.extend(tokens["path_tokens"])
            elif src == "subdomain":
                parts.extend(tokens["subdomain_tokens"])
            else:
                raise ValueError(f"Unknown token source: {src}")
        return parts

