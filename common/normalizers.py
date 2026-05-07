"""Observation / feature normalizers used across surrogate + RL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np


class FlatNormalizer:
    """Z-score with clipping. mean/std are 1D arrays per feature."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, obs_clip: float = 5.0):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
        self.obs_clip = float(obs_clip)

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float32) - self.mean) / self.std
        z = np.nan_to_num(z, nan=0.0, posinf=self.obs_clip, neginf=-self.obs_clip)
        z = np.clip(z, -self.obs_clip, self.obs_clip)
        return z.astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, dtype=np.float32) * self.std + self.mean).astype(np.float32)

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "mean": self.mean.tolist(),
                    "std": self.std.tolist(),
                    "obs_clip": self.obs_clip,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def from_json(path: Union[str, Path]) -> "FlatNormalizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return FlatNormalizer(d["mean"], d["std"], d["obs_clip"])

    @staticmethod
    def fit(x: np.ndarray, obs_clip: float = 5.0) -> "FlatNormalizer":
        x = np.asarray(x, dtype=np.float32)
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        return FlatNormalizer(mean, std, obs_clip)


class IdentityNormalizer:
    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float32)
