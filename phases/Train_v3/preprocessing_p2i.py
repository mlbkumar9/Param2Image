"""
Tabular preprocessing for Param2Image v3.

Standardizes the four input parameters (velocity, projectile radius, x, y),
and log-standardizes the three numerical responses (max von Mises, max PEEQ,
max deflection). Stats are saved as a JSON file alongside the checkpoint so
inference can invert the transform exactly.

This is a self-contained copy of the Train_v2 scaler, kept in this folder so
the v3 stack can be uploaded to Colab as a single directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PARAM_COLUMNS = [
    "velocity_ms",
    "proj_radius_mm",
    "impact_x_mm",
    "impact_y_mm",
]

OUTPUT_COLUMNS = [
    "max_mises_mpa",
    "max_peeq",
    "max_deflection_mm",
]

OUTPUT_SHORT_NAMES = ["mises", "peeq", "deflection"]
OUTPUT_DISPLAY_NAMES = ["Max Mises stress", "Max PEEQ", "Max deflection"]
OUTPUT_UNITS = ["MPa", "-", "mm"]

SCALER_FILENAME = "normalization_stats.json"


def _safe_std(values: np.ndarray, eps: float) -> np.ndarray:
    std = values.std(axis=0)
    return np.where(std < eps, 1.0, std)


class ImpactTabularScaler:
    def __init__(self, output_transform: str = "log1p", eps: float = 1e-8):
        if output_transform not in {"log1p", "identity"}:
            raise ValueError("output_transform must be either 'log1p' or 'identity'")
        self.output_transform = output_transform
        self.eps = eps
        self.param_mean = None
        self.param_std = None
        self.output_mean = None
        self.output_std = None

    @property
    def fitted(self) -> bool:
        return all(v is not None for v in [self.param_mean, self.param_std, self.output_mean, self.output_std])

    def fit(self, df: pd.DataFrame) -> "ImpactTabularScaler":
        missing = [c for c in PARAM_COLUMNS + OUTPUT_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns required for scaling: {missing}")
        params = df[PARAM_COLUMNS].astype(float).to_numpy()
        outputs = df[OUTPUT_COLUMNS].astype(float).to_numpy()
        outputs_t = self._forward_outputs_np(outputs)
        self.param_mean = params.mean(axis=0)
        self.param_std = _safe_std(params, self.eps)
        self.output_mean = outputs_t.mean(axis=0)
        self.output_std = _safe_std(outputs_t, self.eps)
        return self

    def transform_params_array(self, values) -> np.ndarray:
        self._require_fit()
        arr = np.atleast_2d(np.asarray(values, dtype=np.float32))
        return ((arr - self.param_mean) / self.param_std).astype(np.float32)

    def transform_outputs_array(self, values) -> np.ndarray:
        self._require_fit()
        arr = np.atleast_2d(np.asarray(values, dtype=np.float32))
        arr_t = self._forward_outputs_np(arr)
        return ((arr_t - self.output_mean) / self.output_std).astype(np.float32)

    def inverse_transform_outputs_array(self, values) -> np.ndarray:
        self._require_fit()
        arr = np.atleast_2d(np.asarray(values, dtype=np.float32))
        restored = arr * self.output_std + self.output_mean
        return self._inverse_outputs_np(restored).astype(np.float32)

    def inverse_transform_outputs_tensor(self, values: torch.Tensor) -> torch.Tensor:
        self._require_fit()
        mean = torch.as_tensor(self.output_mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.output_std, dtype=values.dtype, device=values.device)
        restored = values * std + mean
        if self.output_transform == "log1p":
            restored = torch.clamp(restored, max=60.0)
            return torch.expm1(restored).clamp_min(0.0)
        return restored

    def save(self, path) -> None:
        self._require_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ImpactTabularScaler":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        scaler = cls(output_transform=data.get("output_transform", "log1p"), eps=float(data.get("eps", 1e-8)))
        scaler.param_mean = np.asarray(data["param_mean"], dtype=np.float32)
        scaler.param_std = np.asarray(data["param_std"], dtype=np.float32)
        scaler.output_mean = np.asarray(data["output_mean"], dtype=np.float32)
        scaler.output_std = np.asarray(data["output_std"], dtype=np.float32)
        return scaler

    def to_dict(self) -> dict:
        self._require_fit()
        return {
            "version": 1,
            "param_columns": PARAM_COLUMNS,
            "output_columns": OUTPUT_COLUMNS,
            "output_transform": self.output_transform,
            "eps": self.eps,
            "param_mean": self.param_mean.tolist(),
            "param_std": self.param_std.tolist(),
            "output_mean": self.output_mean.tolist(),
            "output_std": self.output_std.tolist(),
        }

    def _forward_outputs_np(self, values: np.ndarray) -> np.ndarray:
        if self.output_transform == "log1p":
            return np.log1p(np.clip(values, 0.0, None))
        return values.astype(np.float32)

    def _inverse_outputs_np(self, values: np.ndarray) -> np.ndarray:
        if self.output_transform == "log1p":
            return np.maximum(np.expm1(np.clip(values, None, 60.0)), 0.0)
        return values.astype(np.float32)

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("ImpactTabularScaler must be fitted or loaded before use")
