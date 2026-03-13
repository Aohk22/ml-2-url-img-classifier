from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .constants import DATASET_ID, LABEL_COL, LABEL_MAP, URL_COL


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    test: pd.DataFrame


def load_hf_splits(dataset_id: str = DATASET_ID) -> DatasetSplits:
    """
    Load the dataset from Hugging Face.

    Prefers `datasets.load_dataset` when available, but falls back to
    reading Parquet via pandas if needed.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        load_dataset = None

    if load_dataset is not None:
        ds = load_dataset(dataset_id)
        return DatasetSplits(
            train=ds["train"].to_pandas(),
            test=ds["test"].to_pandas(),
        )

    # Fallback: pandas + parquet from HF.
    # This requires network access when the user runs it.
    train_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/data/train.parquet"
    test_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/data/test.parquet"
    return DatasetSplits(
        train=pd.read_parquet(train_url),
        test=pd.read_parquet(test_url),
    )


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected label column `{LABEL_COL}` in dataframe.")

    out = df.copy()
    out[LABEL_COL] = out[LABEL_COL].astype(str).str.lower().map(LABEL_MAP)
    if out[LABEL_COL].isna().any():
        bad = sorted(set(df[LABEL_COL].astype(str).str.lower()) - set(LABEL_MAP.keys()))
        raise ValueError(f"Unknown labels in `{LABEL_COL}`: {bad[:10]}")
    out[LABEL_COL] = out[LABEL_COL].astype(int)
    return out


def infer_feature_columns(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
    drop_cols: Iterable[str] = (URL_COL,),
) -> list[str]:
    drop = set(drop_cols) | {label_col}
    feature_cols: list[str] = []
    for c in df.columns:
        if c in drop:
            continue
        # Keep numeric-ish columns only; the dataset has many int/float columns.
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after dropping label/url.")
    return feature_cols


def split_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = LABEL_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols]
    y = df[label_col]
    return X, y
