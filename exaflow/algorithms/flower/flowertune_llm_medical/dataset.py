"""Dataset adapter for flowertune_llm_medical Phase 1 runtime."""

from __future__ import annotations

import os
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd


class DatasetLoadError(RuntimeError):
    """Raised when explicit dataset loading fails."""


def _synthetic_partition(seed: int, size: int = 256, n_features: int = 16):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(size, n_features)).astype(np.float32)
    true_w = rng.normal(size=(n_features, 1)).astype(np.float32)
    logits = x @ true_w + 0.1 * rng.normal(size=(size, 1)).astype(np.float32)
    y = (logits[:, 0] > 0).astype(np.float32)
    return x, y


def _split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float):
    n = x.shape[0]
    if n < 2:
        return x, y, x, y
    val_size = max(1, int(n * val_ratio))
    if val_size >= n:
        val_size = max(1, n - 1)
    split = n - val_size
    return x[:split], y[:split], x[split:], y[split:]


def _split_text_train_val(
    texts: List[str], val_ratio: float
) -> Tuple[List[str], List[str]]:
    n = len(texts)
    if n < 2:
        return texts, texts
    val_size = max(1, int(n * val_ratio))
    if val_size >= n:
        val_size = max(1, n - 1)
    split = n - val_size
    return texts[:split], texts[split:]


def _to_numeric_frame(df: pd.DataFrame) -> np.ndarray:
    # Convert mixed types to numeric matrix quickly for tiny smoke training.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes
    return df.astype(float).to_numpy(dtype=np.float32)


def _row_to_text(row: pd.Series, x_vars: List[str], y_var: str) -> str:
    feature_pairs = [f"{name}={row[name]}" for name in x_vars]
    return f"Patient data: {', '.join(feature_pairs)}; target={row[y_var]}"


def load_partition(
    inputdata: Dict,
    *,
    seed: int,
    val_split_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a local partition from worker CSVs or fallback to synthetic data."""

    csv_paths = [p for p in os.getenv("CSV_PATHS", "").split(",") if p]
    x_vars = inputdata.get("x") or []
    y_vars = inputdata.get("y") or []

    if csv_paths:
        if not x_vars or not y_vars:
            raise DatasetLoadError(
                "CSV_PATHS provided but inputdata.x/y are missing; cannot load dataset."
            )
        try:
            frames = [pd.read_csv(path) for path in csv_paths]
            full_df = pd.concat(frames, ignore_index=True)

            y_col = y_vars[0]
            missing_cols = [
                col for col in [*x_vars, y_col] if col not in full_df.columns
            ]
            if missing_cols:
                raise DatasetLoadError(
                    f"Dataset columns not found in CSV input: {missing_cols}"
                )

            features = _to_numeric_frame(full_df[x_vars].copy())
            target_raw = full_df[y_col]
            if target_raw.dtype == "object":
                target = target_raw.astype("category").cat.codes.to_numpy(
                    dtype=np.float32
                )
            else:
                target = target_raw.to_numpy(dtype=np.float32)
            target = (target > np.median(target)).astype(np.float32)

            return _split_train_val(features, target, val_split_ratio)
        except DatasetLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DatasetLoadError(f"Failed to load CSV dataset: {exc}") from exc

    # Contract fallback for Phase 1 smoke runs.
    x, y = _synthetic_partition(seed=seed)
    return _split_train_val(x, y, val_split_ratio)


def load_text_partition(
    inputdata: Dict,
    *,
    seed: int,
    val_split_ratio: float,
) -> Tuple[List[str], List[str]]:
    """Load local text partition for hf_peft backend."""

    csv_paths = [p for p in os.getenv("CSV_PATHS", "").split(",") if p]
    x_vars = inputdata.get("x") or []
    y_vars = inputdata.get("y") or []

    if csv_paths:
        if not x_vars or not y_vars:
            raise DatasetLoadError(
                "CSV_PATHS provided but inputdata.x/y are missing; cannot build text dataset."
            )
        try:
            frames = [pd.read_csv(path) for path in csv_paths]
            full_df = pd.concat(frames, ignore_index=True)
            y_col = y_vars[0]
            missing_cols = [
                col for col in [*x_vars, y_col] if col not in full_df.columns
            ]
            if missing_cols:
                raise DatasetLoadError(
                    f"Dataset columns not found in CSV input: {missing_cols}"
                )
            texts = [
                _row_to_text(row, x_vars=x_vars, y_var=y_col)
                for _, row in full_df.iterrows()
            ]
            return _split_text_train_val(texts, val_split_ratio)
        except DatasetLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DatasetLoadError(f"Failed to load CSV text dataset: {exc}") from exc

    rng = np.random.default_rng(seed)
    texts = []
    for i in range(256):
        age = int(rng.integers(18, 90))
        bmi = float(rng.normal(27.0, 4.5))
        label = "high_risk" if (age > 60 and bmi > 28) else "low_risk"
        texts.append(
            f"Patient profile #{i}: age={age}, bmi={bmi:.2f}. predicted_outcome={label}"
        )
    return _split_text_train_val(texts, val_split_ratio)
