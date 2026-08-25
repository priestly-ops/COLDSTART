from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.m2n2_adapter import (
    ChannelStandardizer,
    M2N2Adapter,
    M2N2Config,
    cycle_windows,
)


@dataclass(frozen=True)
class DummyCycle:
    episode_id: int
    values: np.ndarray
    columns: tuple[str, ...] = ("a", "b")
    anomaly: bool = False
    category: int = 0
    setting: int = 72


def _cycles(n: int, *, shift: float = 0.0) -> list[DummyCycle]:
    rng = np.random.default_rng(123)
    return [
        DummyCycle(
            episode_id=i,
            values=(rng.normal(size=(24, 2)) + shift).astype(np.float64),
        )
        for i in range(n)
    ]


def test_cycle_windows_pads_short_cycle_and_covers_tail() -> None:
    short = np.arange(10, dtype=np.float32).reshape(5, 2)
    out = cycle_windows(short, window_size=6, stride=6)
    assert out.shape == (1, 6, 2)
    np.testing.assert_array_equal(out[0, -1], short[-1])

    long = np.arange(34, dtype=np.float32).reshape(17, 2)
    out = cycle_windows(long, window_size=6, stride=6)
    assert out.shape[1:] == (6, 2)
    np.testing.assert_array_equal(out[-1, -1], long[-1])


def test_standardizer_is_source_only_and_finite() -> None:
    source = _cycles(3)
    scaler = ChannelStandardizer.fit(source)
    stacked = np.concatenate([c.values for c in source], axis=0)
    transformed = scaler.transform(stacked)
    assert np.all(np.isfinite(transformed))
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)


def test_source_fit_and_commissioning_adaptation_produce_finite_scores() -> None:
    source = _cycles(4)
    target = _cycles(2, shift=0.25)
    scaler = ChannelStandardizer.fit(source)
    cfg = M2N2Config(
        window_size=6,
        stride=6,
        latent_dim=4,
        train_batch_size=8,
        train_epochs=1,
        train_lr=1e-3,
        tt_lr=1e-3,
        gamma=0.99,
        source_mask_quantile=0.9,
        use_detrend=True,
        seed=7,
    )
    model = M2N2Adapter(
        num_channels=2,
        standardizer=scaler,
        config=cfg,
        device="cpu",
    ).fit_source(source)
    assert np.isfinite(model.source_mask_threshold_)
    before = model.score_cycles(target)
    info = model.adapt_cycles(target)
    after = model.score_cycles(target)
    assert np.all(np.isfinite(before))
    assert np.all(np.isfinite(after))
    assert 0.0 <= info["adaptation_selected_fraction"] <= 1.0


def test_clone_is_independent() -> None:
    source = _cycles(4)
    scaler = ChannelStandardizer.fit(source)
    cfg = M2N2Config(
        window_size=6, stride=6, latent_dim=4,
        train_batch_size=8, train_epochs=1, seed=11,
    )
    original = M2N2Adapter(2, scaler, cfg, device="cpu").fit_source(source)
    clone = original.clone()
    p0 = next(original.model.parameters()).detach().clone()
    with torch.no_grad():
        next(clone.model.parameters()).add_(1.0)
    assert torch.allclose(next(original.model.parameters()), p0)
    assert not torch.allclose(next(original.model.parameters()), next(clone.model.parameters()))
