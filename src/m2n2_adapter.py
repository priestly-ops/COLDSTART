from __future__ import annotations

"""M2N2-style reconstruction and test-time adaptation for COLDSTART.

This module is an explicit robotics adaptation of Kim, Park & Choo (AAAI 2024),
"When Model Meets New Normals".  It preserves the core M2N2 ingredients used
by the official MLP implementation:

* an MLP reconstruction model over fixed windows;
* optional EMA detrending;
* a fixed high-quantile normal-score threshold;
* masked self-training that updates only locations predicted normal;
* SGD at test/adaptation time.

The original work is timestep-level continuous TSAD.  COLDSTART evaluates whole
robot executions.  We therefore aggregate timestep reconstruction error by the
mean to obtain one execution-level anomaly score.  No anomaly labels are used
for model fitting, adaptation, calibration, or threshold selection.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence
import copy
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("E6 M2N2 requires PyTorch.") from exc


@dataclass(frozen=True)
class M2N2Config:
    window_size: int = 12
    stride: int = 12
    latent_dim: int = 128
    train_batch_size: int = 64
    train_epochs: int = 10
    train_lr: float = 1e-3
    tt_lr: float = 5e-3
    gamma: float = 0.99999
    source_mask_quantile: float = 0.996
    use_detrend: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.train_epochs < 1:
            raise ValueError("train_epochs must be positive")
        if not 0.0 < self.source_mask_quantile < 1.0:
            raise ValueError("source_mask_quantile must lie in (0, 1)")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1)")


def set_m2n2_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class Detrender(nn.Module):
    """EMA mean detrending matching the official M2N2 Normalizer."""

    def __init__(self, num_channels: int, gamma: float) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("mean", torch.zeros(1, 1, num_channels))

    @torch.no_grad()
    def update_statistics(self, x: torch.Tensor) -> None:
        mu = torch.mean(x, dim=tuple(range(0, x.ndim - 1)), keepdim=True)
        self.mean.lerp_(mu, 1.0 - self.gamma)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.mean

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mean


class M2N2MLP(nn.Module):
    """MLP autoencoder topology matching the official M2N2 MLP backbone."""

    def __init__(
        self,
        window_size: int,
        num_channels: int,
        latent_dim: int,
        gamma: float,
        use_detrend: bool,
    ) -> None:
        super().__init__()
        input_size = int(window_size * num_channels)
        half = max(input_size // 2, 4)
        quarter = max(input_size // 4, 2)
        self.window_size = int(window_size)
        self.num_channels = int(num_channels)
        self.use_detrend = bool(use_detrend)
        self.detrender = Detrender(num_channels, gamma)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, half), nn.ReLU(),
            nn.Linear(half, quarter), nn.ReLU(),
            nn.Linear(quarter, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, quarter), nn.ReLU(),
            nn.Linear(quarter, half), nn.ReLU(),
            nn.Linear(half, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        if l != self.window_size or c != self.num_channels:
            raise ValueError(
                f"Expected (*,{self.window_size},{self.num_channels}), got {tuple(x.shape)}"
            )
        work = self.detrender.normalize(x) if self.use_detrend else x
        z = self.encoder(work.reshape(b, l * c))
        out = self.decoder(z).reshape(b, l, c)
        return self.detrender.denormalize(out) if self.use_detrend else out


@dataclass(frozen=True)
class ChannelStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, cycles: Sequence) -> "ChannelStandardizer":
        if not cycles:
            raise ValueError("cycles cannot be empty")
        values = np.concatenate(
            [np.asarray(c.values, dtype=np.float64) for c in cycles], axis=0
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        return ((x - self.mean) / self.std).astype(np.float32)


def cycle_windows(
    values: np.ndarray,
    *,
    window_size: int,
    stride: int,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("cycle values must be a 2D array")
    if len(x) < window_size:
        pad = np.repeat(x[-1:, :], window_size - len(x), axis=0)
        return np.concatenate([x, pad], axis=0)[None, :, :]
    starts = list(range(0, len(x) - window_size + 1, stride))
    if starts[-1] != len(x) - window_size:
        starts.append(len(x) - window_size)
    return np.stack([x[s : s + window_size] for s in starts], axis=0)


def windows_from_cycles(
    cycles: Sequence,
    standardizer: ChannelStandardizer,
    config: M2N2Config,
) -> np.ndarray:
    pieces = [
        cycle_windows(
            standardizer.transform(c.values),
            window_size=config.window_size,
            stride=config.stride,
        )
        for c in cycles
    ]
    if not pieces:
        raise ValueError("No cycles supplied")
    return np.concatenate(pieces, axis=0).astype(np.float32)


class M2N2Adapter:
    def __init__(
        self,
        num_channels: int,
        standardizer: ChannelStandardizer,
        config: M2N2Config,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.standardizer = standardizer
        self.device = torch.device(device)
        set_m2n2_seed(config.seed)
        self.model = M2N2MLP(
            window_size=config.window_size,
            num_channels=num_channels,
            latent_dim=config.latent_dim,
            gamma=config.gamma,
            use_detrend=config.use_detrend,
        ).to(self.device)
        self.source_mask_threshold_: float | None = None

    def clone(self) -> "M2N2Adapter":
        cloned = M2N2Adapter(
            num_channels=self.model.num_channels,
            standardizer=self.standardizer,
            config=self.config,
            device=str(self.device),
        )
        cloned.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        cloned.source_mask_threshold_ = self.source_mask_threshold_
        return cloned

    def _loader(self, windows: np.ndarray, *, shuffle: bool) -> DataLoader:
        tensor = torch.as_tensor(windows, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            TensorDataset(tensor),
            batch_size=self.config.train_batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=generator,
        )

    def fit_source(self, source_cycles: Sequence) -> "M2N2Adapter":
        windows = windows_from_cycles(source_cycles, self.standardizer, self.config)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.train_lr, weight_decay=0.0
        )
        self.model.train()
        for _ in range(self.config.train_epochs):
            for (x,) in self._loader(windows, shuffle=True):
                x = x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                xhat = self.model(x)
                loss = F.mse_loss(xhat, x)
                loss.backward()
                optimizer.step()
        # Official M2N2 uses a high quantile of normal training anomaly scores
        # as the initial pseudo-label threshold for online adaptation.
        timestep_scores = self.timestep_scores_for_windows(windows)
        self.source_mask_threshold_ = float(
            np.quantile(timestep_scores, self.config.source_mask_quantile)
        )
        return self

    @torch.no_grad()
    def timestep_scores_for_windows(self, windows: np.ndarray) -> np.ndarray:
        self.model.eval()
        scores: list[np.ndarray] = []
        for (x,) in self._loader(windows, shuffle=False):
            x = x.to(self.device)
            err = (self.model(x) - x) ** 2
            a = err.mean(dim=2)  # official M2N2 channel-mean anomaly score
            scores.append(a.detach().cpu().numpy().reshape(-1))
        return np.concatenate(scores).astype(np.float64)

    @torch.no_grad()
    def score_cycle(self, cycle) -> float:
        x_np = cycle_windows(
            self.standardizer.transform(cycle.values),
            window_size=self.config.window_size,
            stride=self.config.stride,
        )
        scores = self.timestep_scores_for_windows(x_np)
        return float(np.mean(scores))

    def score_cycles(self, cycles: Sequence) -> np.ndarray:
        return np.asarray([self.score_cycle(c) for c in cycles], dtype=np.float64)

    def adapt_cycles(self, cycles: Sequence) -> dict[str, float]:
        """M2N2 masked self-training over a target execution stream.

        Cycles are processed in ascending episode ID as the available acquisition
        order proxy. Labels are never consulted.  Statistics are updated before
        inference and SGD is applied only to locations whose current score is
        below the source-derived normal threshold, matching official M2N2 logic.
        """
        if self.source_mask_threshold_ is None:
            raise RuntimeError("fit_source must be called before adaptation")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.tt_lr)
        selected = 0
        total = 0
        losses: list[float] = []
        self.model.train()
        for cycle in sorted(cycles, key=lambda c: int(c.episode_id)):
            windows = cycle_windows(
                self.standardizer.transform(cycle.values),
                window_size=self.config.window_size,
                stride=self.config.stride,
            )
            for (x,) in self._loader(windows, shuffle=False):
                x = x.to(self.device)
                if self.config.use_detrend:
                    self.model.detrender.update_statistics(x)
                xhat = self.model(x)
                a = ((xhat - x) ** 2).mean(dim=2)
                mask = (a < self.source_mask_threshold_).float().detach()
                optimizer.zero_grad(set_to_none=True)
                # Keep the official mean-over-all-elements convention; masked
                # anomaly locations contribute zero rather than renormalizing.
                loss = (a * mask).mean()
                loss.backward()
                optimizer.step()
                selected += int(mask.sum().detach().cpu().item())
                total += int(mask.numel())
                losses.append(float(loss.detach().cpu().item()))
        return {
            "adaptation_selected_fraction": float(selected / total) if total else float("nan"),
            "mean_adaptation_loss": float(np.mean(losses)) if losses else float("nan"),
        }

    def online_score_and_adapt(self, cycles: Sequence) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Faithful continued online adaptation; returns IDs and pre-update scores.

        Exact fixed-detector certification is intentionally not provided for
        this mode because model parameters/statistics change through evaluation.
        """
        if self.source_mask_threshold_ is None:
            raise RuntimeError("fit_source must be called before adaptation")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.tt_lr)
        ids: list[int] = []
        cycle_scores: list[float] = []
        selected = 0
        total = 0
        self.model.train()
        for cycle in sorted(cycles, key=lambda c: int(c.episode_id)):
            windows = cycle_windows(
                self.standardizer.transform(cycle.values),
                window_size=self.config.window_size,
                stride=self.config.stride,
            )
            per_cycle: list[np.ndarray] = []
            for (x,) in self._loader(windows, shuffle=False):
                x = x.to(self.device)
                if self.config.use_detrend:
                    self.model.detrender.update_statistics(x)
                xhat = self.model(x)
                a = ((xhat - x) ** 2).mean(dim=2)
                per_cycle.append(a.detach().cpu().numpy().reshape(-1))
                mask = (a < self.source_mask_threshold_).float().detach()
                optimizer.zero_grad(set_to_none=True)
                loss = (a * mask).mean()
                loss.backward()
                optimizer.step()
                selected += int(mask.sum().detach().cpu().item())
                total += int(mask.numel())
            ids.append(int(cycle.episode_id))
            cycle_scores.append(float(np.mean(np.concatenate(per_cycle))))
        return (
            np.asarray(ids, dtype=np.int64),
            np.asarray(cycle_scores, dtype=np.float64),
            {"online_selected_fraction": float(selected / total) if total else float("nan")},
        )
