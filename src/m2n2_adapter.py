from __future__ import annotations

"""M2N2-style reconstruction and test-time adaptation for COLDSTART.

Robotics adaptation of Kim, Park & Choo (AAAI 2024), preserving the official
MLP reconstruction + detrending + masked test-time SGD mechanism. The original
method is timestep-level TSAD; COLDSTART requires one score per robot execution.
The primary execution score is the predeclared 99th percentile of channel-mean
timestep reconstruction errors, preserving localized faults without anomaly
labels. Set ``cycle_score_quantile=None`` for a mean-score sensitivity run.
"""

from dataclasses import dataclass
from typing import Sequence
import copy
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
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
    cycle_score_quantile: float | None = 0.99
    use_detrend: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.window_size < 2 or self.stride < 1:
            raise ValueError("Invalid window_size/stride")
        if self.latent_dim < 1 or self.train_epochs < 1:
            raise ValueError("Invalid latent_dim/train_epochs")
        if not 0.0 < self.source_mask_quantile < 1.0:
            raise ValueError("source_mask_quantile must lie in (0,1)")
        if self.cycle_score_quantile is not None and not 0.0 < self.cycle_score_quantile <= 1.0:
            raise ValueError("cycle_score_quantile must lie in (0,1] or be None")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must lie in [0,1)")


def set_m2n2_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class Detrender(nn.Module):
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
    def __init__(self, window_size: int, num_channels: int, latent_dim: int, gamma: float, use_detrend: bool) -> None:
        super().__init__()
        d = int(window_size * num_channels)
        d2, d4 = max(d // 2, 4), max(d // 4, 2)
        self.window_size = int(window_size)
        self.num_channels = int(num_channels)
        self.use_detrend = bool(use_detrend)
        self.detrender = Detrender(num_channels, gamma)
        self.encoder = nn.Sequential(
            nn.Linear(d, d2), nn.ReLU(), nn.Linear(d2, d4), nn.ReLU(),
            nn.Linear(d4, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d4), nn.ReLU(), nn.Linear(d4, d2), nn.ReLU(),
            nn.Linear(d2, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        if l != self.window_size or c != self.num_channels:
            raise ValueError(f"Expected (*,{self.window_size},{self.num_channels}), got {tuple(x.shape)}")
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
        x = np.concatenate([np.asarray(c.values, dtype=np.float64) for c in cycles], axis=0)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        return cls(mean.astype(np.float64), std.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        return ((x - self.mean) / self.std).astype(np.float32)


def cycle_windows(values: np.ndarray, *, window_size: int, stride: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("cycle values must be nonempty 2D")
    if len(x) < window_size:
        pad = np.repeat(x[-1:, :], window_size - len(x), axis=0)
        return np.concatenate([x, pad], axis=0)[None, :, :]
    starts = list(range(0, len(x) - window_size + 1, stride))
    if starts[-1] != len(x) - window_size:
        starts.append(len(x) - window_size)
    return np.stack([x[s:s + window_size] for s in starts], axis=0)


def windows_from_cycles(cycles: Sequence, standardizer: ChannelStandardizer, config: M2N2Config) -> np.ndarray:
    if not cycles:
        raise ValueError("No cycles supplied")
    return np.concatenate([
        cycle_windows(standardizer.transform(c.values), window_size=config.window_size, stride=config.stride)
        for c in cycles
    ], axis=0).astype(np.float32)


class M2N2Adapter:
    def __init__(self, num_channels: int, standardizer: ChannelStandardizer, config: M2N2Config, device: str = "cpu") -> None:
        self.config = config
        self.standardizer = standardizer
        self.device = torch.device(device)
        set_m2n2_seed(config.seed)
        self.model = M2N2MLP(config.window_size, num_channels, config.latent_dim, config.gamma, config.use_detrend).to(self.device)
        self.source_mask_threshold_: float | None = None

    def clone(self) -> "M2N2Adapter":
        out = M2N2Adapter(self.model.num_channels, self.standardizer, self.config, str(self.device))
        out.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        out.source_mask_threshold_ = self.source_mask_threshold_
        return out

    def _loader(self, windows: np.ndarray, *, shuffle: bool) -> DataLoader:
        g = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            TensorDataset(torch.as_tensor(windows, dtype=torch.float32)),
            batch_size=self.config.train_batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=g,
        )

    def fit_source(self, source_cycles: Sequence) -> "M2N2Adapter":
        windows = windows_from_cycles(source_cycles, self.standardizer, self.config)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.train_lr, weight_decay=0.0)
        self.model.train()
        for _ in range(self.config.train_epochs):
            for (x,) in self._loader(windows, shuffle=True):
                x = x.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(self.model(x), x)
                loss.backward()
                opt.step()
        normal_scores = self.timestep_scores_for_windows(windows)
        self.source_mask_threshold_ = float(np.quantile(normal_scores, self.config.source_mask_quantile))
        return self

    @torch.no_grad()
    def timestep_scores_for_windows(self, windows: np.ndarray) -> np.ndarray:
        self.model.eval()
        chunks: list[np.ndarray] = []
        for (x,) in self._loader(windows, shuffle=False):
            x = x.to(self.device)
            a = ((self.model(x) - x) ** 2).mean(dim=2)
            chunks.append(a.detach().cpu().numpy().reshape(-1))
        return np.concatenate(chunks).astype(np.float64)

    def _aggregate_cycle_scores(self, scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=np.float64)
        q = self.config.cycle_score_quantile
        return float(np.mean(scores) if q is None else np.quantile(scores, q))

    @torch.no_grad()
    def score_cycle(self, cycle) -> float:
        windows = cycle_windows(
            self.standardizer.transform(cycle.values),
            window_size=self.config.window_size,
            stride=self.config.stride,
        )
        return self._aggregate_cycle_scores(self.timestep_scores_for_windows(windows))

    def score_cycles(self, cycles: Sequence) -> np.ndarray:
        return np.asarray([self.score_cycle(c) for c in cycles], dtype=np.float64)

    def adapt_cycles(self, cycles: Sequence) -> dict[str, float]:
        if self.source_mask_threshold_ is None:
            raise RuntimeError("fit_source must be called before adaptation")
        opt = torch.optim.SGD(self.model.parameters(), lr=self.config.tt_lr)
        selected = total = 0
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
                a = ((self.model(x) - x) ** 2).mean(dim=2)
                mask = (a < self.source_mask_threshold_).float().detach()
                opt.zero_grad(set_to_none=True)
                loss = (a * mask).mean()
                loss.backward()
                opt.step()
                selected += int(mask.sum().detach().cpu().item())
                total += int(mask.numel())
                losses.append(float(loss.detach().cpu().item()))
        return {
            "adaptation_selected_fraction": float(selected / total) if total else float("nan"),
            "mean_adaptation_loss": float(np.mean(losses)) if losses else float("nan"),
        }

    def online_score_and_adapt(self, cycles: Sequence) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        if self.source_mask_threshold_ is None:
            raise RuntimeError("fit_source must be called before adaptation")
        opt = torch.optim.SGD(self.model.parameters(), lr=self.config.tt_lr)
        ids: list[int] = []
        cycle_scores: list[float] = []
        selected = total = 0
        self.model.train()
        for cycle in sorted(cycles, key=lambda c: int(c.episode_id)):
            windows = cycle_windows(
                self.standardizer.transform(cycle.values),
                window_size=self.config.window_size,
                stride=self.config.stride,
            )
            all_scores: list[np.ndarray] = []
            for (x,) in self._loader(windows, shuffle=False):
                x = x.to(self.device)
                if self.config.use_detrend:
                    self.model.detrender.update_statistics(x)
                a = ((self.model(x) - x) ** 2).mean(dim=2)
                all_scores.append(a.detach().cpu().numpy().reshape(-1))
                mask = (a < self.source_mask_threshold_).float().detach()
                opt.zero_grad(set_to_none=True)
                loss = (a * mask).mean()
                loss.backward()
                opt.step()
                selected += int(mask.sum().detach().cpu().item())
                total += int(mask.numel())
            ids.append(int(cycle.episode_id))
            cycle_scores.append(self._aggregate_cycle_scores(np.concatenate(all_scores)))
        return (
            np.asarray(ids, dtype=np.int64),
            np.asarray(cycle_scores, dtype=np.float64),
            {"online_selected_fraction": float(selected / total) if total else float("nan")},
        )
