from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


GLOBAL_SEED = 42


def _validate_matrix(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix; got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return x


class IsolationForestBaseline:
    """CPU-only target-commissioning Isolation Forest baseline.

    The model is fitted only on target healthy fit cycles. Calibration remains
    external and must use the same frozen healthy calibration episodes as the
    other P0.7 methods. Larger returned scores mean more anomalous.
    """

    def __init__(self, random_state: int = GLOBAL_SEED) -> None:
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.fitted = False

    def fit(self, target_features: np.ndarray) -> "IsolationForestBaseline":
        x = _validate_matrix(target_features, "target_features")
        z = self.scaler.fit_transform(x)
        self.model.fit(z)
        self.fitted = True
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit must be called before score_samples")
        x = _validate_matrix(features, "features")
        z = self.scaler.transform(x)
        # sklearn uses larger values for more normal observations.
        return -self.model.score_samples(z)


class ConformalKNNBaseline:
    """Target-only Euclidean k-NN nonconformity on cycle feature vectors."""

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.scaler = StandardScaler()
        self.nn: NearestNeighbors | None = None
        self.effective_k_: int | None = None

    def fit(self, target_features: np.ndarray) -> "ConformalKNNBaseline":
        x = _validate_matrix(target_features, "target_features")
        z = self.scaler.fit_transform(x)
        effective_k = min(self.k, len(z))
        self.nn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
        self.nn.fit(z)
        self.effective_k_ = int(effective_k)
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        if self.nn is None or self.effective_k_ is None:
            raise RuntimeError("fit must be called before score_samples")
        x = _validate_matrix(features, "features")
        z = self.scaler.transform(x)
        d, _ = self.nn.kneighbors(z, return_distance=True)
        return d[:, self.effective_k_ - 1].astype(np.float64)


def _linear_resample(values: np.ndarray, length: int) -> np.ndarray:
    values = _validate_matrix(values, "cycle")
    if length < 2:
        raise ValueError("length must be >= 2")
    if len(values) == length:
        return values.copy()
    old_t = np.linspace(0.0, 1.0, num=len(values))
    new_t = np.linspace(0.0, 1.0, num=length)
    out = np.empty((length, values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(new_t, old_t, values[:, j])
    return out


def _align_to_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """FastDTW phase alignment returning a fixed-length reference-grid cycle."""
    values = _validate_matrix(values, "cycle")
    reference = _validate_matrix(reference, "reference")
    if values.shape[1] != reference.shape[1]:
        raise ValueError("cycle/reference channel dimensions differ")

    _, path = fastdtw(reference, values, dist=euclidean)
    buckets: list[list[int]] = [[] for _ in range(len(reference))]
    for i_ref, i_val in path:
        if 0 <= i_ref < len(reference) and 0 <= i_val < len(values):
            buckets[i_ref].append(i_val)

    aligned = np.empty_like(reference, dtype=np.float64)
    last = 0
    for i, idxs in enumerate(buckets):
        if idxs:
            aligned[i] = values[np.asarray(idxs, dtype=int)].mean(axis=0)
            last = int(idxs[-1])
        else:
            aligned[i] = values[last]
    return aligned


@dataclass
class RawCycleKNNBaseline:
    """Raw multivariate cycle k-NN with optional DTW phase alignment.

    Each channel is standardized from target healthy fit cycles only. Unaligned
    mode linearly resamples cycles to the first fit cycle length before Euclidean
    k-NN. Phase-aligned mode first FastDTW-aligns each cycle to that reference.
    """

    k: int = 10
    phase_align: bool = False

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be positive")
        self.reference_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.nn_: NearestNeighbors | None = None
        self.effective_k_: int | None = None

    def _standardize_cycle(self, values: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("baseline is not fitted")
        return (values - self.center_) / self.scale_

    def _embed_one(self, values: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("baseline is not fitted")
        z = self._standardize_cycle(_validate_matrix(values, "cycle"))
        if self.phase_align:
            fixed = _align_to_reference(z, self.reference_)
        else:
            fixed = _linear_resample(z, len(self.reference_))
        return fixed.reshape(-1)

    def fit(self, cycles: Sequence[np.ndarray]) -> "RawCycleKNNBaseline":
        if len(cycles) == 0:
            raise ValueError("at least one fit cycle is required")
        checked = [_validate_matrix(v, "fit_cycle") for v in cycles]
        p = checked[0].shape[1]
        if any(v.shape[1] != p for v in checked):
            raise ValueError("fit cycles have inconsistent channel counts")
        stacked = np.vstack(checked)
        self.center_ = stacked.mean(axis=0)
        self.scale_ = stacked.std(axis=0, ddof=1)
        self.scale_ = np.where(np.isfinite(self.scale_) & (self.scale_ > 1e-12), self.scale_, 1.0)
        self.reference_ = self._standardize_cycle(checked[0])
        train = np.vstack([self._embed_one(v) for v in checked])
        effective_k = min(int(self.k), len(train))
        self.nn_ = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
        self.nn_.fit(train)
        self.effective_k_ = effective_k
        return self

    def score_cycles(self, cycles: Sequence[np.ndarray]) -> np.ndarray:
        if self.nn_ is None or self.effective_k_ is None:
            raise RuntimeError("fit must be called before score_cycles")
        if len(cycles) == 0:
            return np.empty(0, dtype=np.float64)
        x = np.vstack([self._embed_one(v) for v in cycles])
        d, _ = self.nn_.kneighbors(x, return_distance=True)
        return d[:, self.effective_k_ - 1].astype(np.float64)
