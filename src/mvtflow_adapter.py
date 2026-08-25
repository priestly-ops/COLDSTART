from __future__ import annotations

import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.base_detector import deterministic_conformal_threshold
from src.certification import certify_operating_point
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority
from src.voraus_loader import RobotCycle


OFFICIAL_CONFIG = {
    "epochs": 70,
    "batch_size": 32,
    "n_coupling_blocks": 4,
    "clamp": 1.2,
    "learning_rate": 8e-4,
    "n_hidden_layers": 0,
    "scale": 2,
    "kernel_size_1": 13,
    "dilation_1": 2,
    "kernel_size_2": 1,
    "dilation_2": 1,
    "kernel_size_3": 1,
    "dilation_3": 1,
    "milestones": [11, 61],
    "gamma": 0.1,
}


@dataclass(frozen=True)
class MVTFlowPreparedData:
    train: np.ndarray
    calibration: np.ndarray
    healthy_eval: np.ndarray
    anomaly_eval: np.ndarray
    target_length: int
    n_signals: int
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class MVTFlowRunResult:
    detector: str
    commissioning_size: int
    seed: int
    model_seed: int
    device: str
    threshold: float
    calibration_alpha: float
    conformal_rank: int
    conformal_regime: str
    calibration_size: int
    tp: int
    fn: int
    fp: int
    tn: int
    recall: float
    false_positive_rate: float
    recall_lower: float
    fpr_upper: float
    empirical_success: bool
    certified_success: bool
    auroc: float
    oracle_recall_at_fpr_budget: float
    oracle_fpr_at_max_recall: float
    oracle_threshold: float
    oracle_empirically_feasible: bool
    bottleneck_label: str
    target_length: int
    n_signals: int


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "E2 requires PyTorch. Install `requirements-mvtflow.txt` first."
        ) from exc
    return torch


def _load_official_modules(project_root: Path):
    official_root = project_root / "external" / "voraus-ad-dataset"
    required = [
        official_root / "configuration.py",
        official_root / "normalizing_flow.py",
        official_root / "coupling_layers.py",
        official_root / "graph_inn.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Official voraus-AD submodule is not initialized. Run `git submodule "
            "update --init --recursive`. Missing: " + ", ".join(missing)
        )

    root_string = str(official_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    configuration = importlib.import_module("configuration")
    normalizing_flow = importlib.import_module("normalizing_flow")
    return configuration, normalizing_flow


def _check_columns(groups: Sequence[Sequence[RobotCycle]]) -> tuple[str, ...]:
    columns: tuple[str, ...] | None = None
    for group in groups:
        for cycle in group:
            if columns is None:
                columns = tuple(cycle.columns)
            elif tuple(cycle.columns) != columns:
                raise ValueError("MVT-Flow cycles do not share an identical signal schema.")
    if columns is None:
        raise ValueError("No cycles supplied to MVT-Flow preprocessing.")
    return columns


def _fit_standardizer(cycles: Sequence[RobotCycle]) -> tuple[np.ndarray, np.ndarray]:
    if not cycles:
        raise ValueError("Commissioning cycles cannot be empty.")
    stacked = np.vstack([np.asarray(c.values, dtype=np.float64) for c in cycles])
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


def _standardize_and_pad(
    cycles: Sequence[RobotCycle],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    target_length: int,
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for cycle in cycles:
        values = (np.asarray(cycle.values, dtype=np.float64) - mean) / scale
        # Match the official loader: truncate to training max length, then right-pad.
        values = values[:target_length]
        if values.shape[0] < target_length:
            padding = np.zeros(
                (target_length - values.shape[0], values.shape[1]),
                dtype=np.float64,
            )
            values = np.vstack((values, padding))
        arrays.append(values.astype(np.float32, copy=False))
    if not arrays:
        return np.empty((0, target_length, mean.size), dtype=np.float32)
    return np.stack(arrays, axis=0)


def prepare_mvtflow_data(
    commissioning: Sequence[RobotCycle],
    calibration: Sequence[RobotCycle],
    healthy_eval: Sequence[RobotCycle],
    anomaly_eval: Sequence[RobotCycle],
) -> MVTFlowPreparedData:
    columns = _check_columns([commissioning, calibration, healthy_eval, anomaly_eval])
    mean, scale = _fit_standardizer(commissioning)
    target_length = max(len(c.values) for c in commissioning)
    if target_length <= 0:
        raise ValueError("MVT-Flow target length must be positive.")

    return MVTFlowPreparedData(
        train=_standardize_and_pad(
            commissioning, mean=mean, scale=scale, target_length=target_length
        ),
        calibration=_standardize_and_pad(
            calibration, mean=mean, scale=scale, target_length=target_length
        ),
        healthy_eval=_standardize_and_pad(
            healthy_eval, mean=mean, scale=scale, target_length=target_length
        ),
        anomaly_eval=_standardize_and_pad(
            anomaly_eval, mean=mean, scale=scale, target_length=target_length
        ),
        target_length=target_length,
        n_signals=len(columns),
        mean=mean,
        scale=scale,
    )


def _configuration(configuration_module, seed: int):
    return configuration_module.Configuration(
        columns="machine",
        epochs=OFFICIAL_CONFIG["epochs"],
        frequencyDivider=1,
        trainGain=1.0,
        seed=seed,
        batchsize=OFFICIAL_CONFIG["batch_size"],
        nCouplingBlocks=OFFICIAL_CONFIG["n_coupling_blocks"],
        clamp=OFFICIAL_CONFIG["clamp"],
        learningRate=OFFICIAL_CONFIG["learning_rate"],
        normalize=True,
        pad=True,
        nHiddenLayers=OFFICIAL_CONFIG["n_hidden_layers"],
        scale=OFFICIAL_CONFIG["scale"],
        kernelSize1=OFFICIAL_CONFIG["kernel_size_1"],
        dilation1=OFFICIAL_CONFIG["dilation_1"],
        kernelSize2=OFFICIAL_CONFIG["kernel_size_2"],
        dilation2=OFFICIAL_CONFIG["dilation_2"],
        kernelSize3=OFFICIAL_CONFIG["kernel_size_3"],
        dilation3=OFFICIAL_CONFIG["dilation_3"],
        milestones=OFFICIAL_CONFIG["milestones"],
        gamma=OFFICIAL_CONFIG["gamma"],
    )


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def train_mvtflow(
    prepared: MVTFlowPreparedData,
    *,
    project_root: Path,
    seed: int,
    device_name: str | None = None,
):
    torch = _require_torch()
    configuration_module, normalizing_flow = _load_official_modules(project_root)
    _seed_everything(seed, torch)

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    config = _configuration(configuration_module, seed=seed)
    model = normalizing_flow.NormalizingFlow(
        (prepared.n_signals, prepared.target_length), config
    ).float().to(device)

    train_tensor = torch.from_numpy(prepared.train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor),
        batch_size=OFFICIAL_CONFIG["batch_size"],
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=OFFICIAL_CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=OFFICIAL_CONFIG["milestones"],
        gamma=OFFICIAL_CONFIG["gamma"],
    )

    for _epoch in range(OFFICIAL_CONFIG["epochs"]):
        model.train()
        for (batch,) in loader:
            batch = batch.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            latent_z, jacobian = model.forward(batch.transpose(2, 1))
            jacobian = torch.sum(jacobian, dim=tuple(range(1, jacobian.dim())))
            loss = normalizing_flow.get_loss(latent_z, jacobian)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model, normalizing_flow, device


def score_mvtflow(model, normalizing_flow_module, device, arrays: np.ndarray) -> np.ndarray:
    torch = _require_torch()
    if arrays.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(arrays))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=OFFICIAL_CONFIG["batch_size"], shuffle=False
    )
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.float().to(device)
            latent_z, jacobian = model.forward(batch.transpose(2, 1))
            jacobian = torch.sum(jacobian, dim=tuple(range(1, jacobian.dim())))
            batch_scores = normalizing_flow_module.get_loss_per_sample(latent_z, jacobian)
            scores.append(batch_scores.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(scores)


def classify_e2_bottleneck(
    *,
    oracle_feasible: bool,
    empirical_success: bool,
    certified_success: bool,
) -> str:
    if not oracle_feasible:
        return "representation_limited"
    if certified_success:
        return "certified"
    if empirical_success:
        return "certification_limited"
    return "calibration_limited"


def run_mvtflow_replicate(
    *,
    commissioning: Sequence[RobotCycle],
    calibration: Sequence[RobotCycle],
    healthy_eval: Sequence[RobotCycle],
    anomaly_eval: Sequence[RobotCycle],
    project_root: Path,
    commissioning_size: int,
    seed: int,
    false_alert_budget: float = 0.01,
    recall_target: float = 0.90,
    joint_confidence: float = 0.95,
    device_name: str | None = None,
) -> MVTFlowRunResult:
    prepared = prepare_mvtflow_data(
        commissioning=commissioning,
        calibration=calibration,
        healthy_eval=healthy_eval,
        anomaly_eval=anomaly_eval,
    )
    model_seed = 42_000 + seed * 1_000 + commissioning_size
    model, nf_module, device = train_mvtflow(
        prepared, project_root=project_root, seed=model_seed, device_name=device_name
    )

    calibration_scores = score_mvtflow(model, nf_module, device, prepared.calibration)
    healthy_scores = score_mvtflow(model, nf_module, device, prepared.healthy_eval)
    anomaly_scores = score_mvtflow(model, nf_module, device, prepared.anomaly_eval)

    threshold, rank, regime = deterministic_conformal_threshold(
        calibration_scores, alpha=false_alert_budget
    )
    healthy_pred = healthy_scores > threshold
    anomaly_pred = anomaly_scores > threshold

    fp = int(healthy_pred.sum())
    tn = int((~healthy_pred).sum())
    tp = int(anomaly_pred.sum())
    fn = int((~anomaly_pred).sum())
    fpr = float(fp / (fp + tn))
    recall = float(tp / (tp + fn))

    certification = certify_operating_point(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall_target=recall_target,
        fpr_budget=false_alert_budget,
        joint_confidence=joint_confidence,
    )
    empirical_success = bool(recall >= recall_target and fpr <= false_alert_budget)
    oracle = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=false_alert_budget,
        recall_target=recall_target,
    )

    return MVTFlowRunResult(
        detector="TargetOnly-MVTFlow",
        commissioning_size=int(commissioning_size),
        seed=int(seed),
        model_seed=int(model_seed),
        device=str(device),
        threshold=float(threshold),
        calibration_alpha=float(false_alert_budget),
        conformal_rank=int(rank),
        conformal_regime=str(regime),
        calibration_size=int(calibration_scores.size),
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall=recall,
        false_positive_rate=fpr,
        recall_lower=float(certification.recall_lower),
        fpr_upper=float(certification.fpr_upper),
        empirical_success=empirical_success,
        certified_success=bool(certification.certified),
        auroc=float(probability_of_superiority(healthy_scores, anomaly_scores)),
        oracle_recall_at_fpr_budget=float(oracle.max_recall_at_fpr_budget),
        oracle_fpr_at_max_recall=float(oracle.fpr_at_max_recall),
        oracle_threshold=float(oracle.threshold_at_fpr_budget),
        oracle_empirically_feasible=bool(oracle.empirically_feasible),
        bottleneck_label=classify_e2_bottleneck(
            oracle_feasible=bool(oracle.empirically_feasible),
            empirical_success=empirical_success,
            certified_success=bool(certification.certified),
        ),
        target_length=int(prepared.target_length),
        n_signals=int(prepared.n_signals),
    )
