"""Aggregate checkpointed Original RACE component-ablation seed folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROTOCOL_VERSION = "original-race-component-ablation-aggregate-v1"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "original_race_component_ablation_seed0_4_aggregate"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip())
    except Exception:
        return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_file(input_root: Path, seed: int, filename: str) -> Path:
    path = input_root / f"original_race_component_ablation_seed{seed}" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing seed-{seed} artifact: {path}")
    return path


def _read_seed_frames(input_root: Path, seeds: tuple[int, ...], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = _seed_file(input_root, seed, filename)
        frame = pd.read_csv(path)
        frame["run_seed_folder"] = int(seed)
        frame["source_artifact"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_seed_folder", "source_pair_id", "source_group", "target_group", "N", "seed"]
    baseline = results[results["detector"] == "TargetOnly"].set_index(keys)
    rows: list[dict[str, object]] = []
    for detector in sorted(results["detector"].unique()):
        if detector == "TargetOnly":
            continue
        current = results[results["detector"] == detector].set_index(keys)
        for key in current.index.intersection(baseline.index):
            row = dict(zip(keys, key))
            row["candidate_detector"] = detector
            for metric in ["recall", "FPR", "AUROC", "AUPRC", "success", "threshold"]:
                row[f"delta_{metric}"] = float(current.loc[key, metric] - baseline.loc[key, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_original_race_component_ablation(
    *,
    input_root: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = _read_seed_frames(input_root, seeds, "original_race_component_ablation.csv")
    direction = _read_seed_frames(input_root, seeds, "original_race_direction_audit.csv")
    score_equivalence = _read_seed_frames(input_root, seeds, "original_race_score_equivalence.csv")
    partitions = _read_seed_frames(input_root, seeds, "original_race_partition_audit.csv")
    source_compatibility = _read_seed_frames(input_root, seeds, "original_race_source_compatibility.csv")

    summary = results.groupby(["detector", "N"], as_index=False).agg(
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        FPR_mean=("FPR", "mean"),
        FPR_std=("FPR", "std"),
        AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", "std"),
        AUPRC_mean=("AUPRC", "mean"),
        success_rate=("success", "mean"),
        runs=("recall", "size"),
    )
    deltas = _paired_deltas(results)
    delta_summary = deltas.groupby(["candidate_detector"], as_index=False).agg(
        delta_recall_mean=("delta_recall", "mean"),
        delta_recall_std=("delta_recall", "std"),
        delta_FPR_mean=("delta_FPR", "mean"),
        delta_FPR_std=("delta_FPR", "std"),
        delta_AUROC_mean=("delta_AUROC", "mean"),
        delta_AUROC_std=("delta_AUROC", "std"),
        runs=("delta_recall", "size"),
    )
    score_equivalence_eval = score_equivalence[
        (score_equivalence["reference_detector"] == "TargetOnly")
        & (score_equivalence["candidate_detector"] == "OriginalRACE")
        & (score_equivalence["score_split"] == "eval")
    ].copy()
    equivalence_summary = score_equivalence_eval.groupby(["N"], as_index=False).agg(
        affine_r2_mean=("affine_r2", "mean"),
        affine_r2_max=("affine_r2", "max"),
        spearman_mean=("spearman_score_corr", "mean"),
        changed_predictions_mean=("number_changed_predictions", "mean"),
        equivalence_flags=("score_equivalence_flag", lambda s: int((s == "STRUCTURAL_SCORE_EQUIVALENCE").sum())),
        runs=("affine_r2", "size"),
    )
    top_posthoc_directions = direction.sort_values(
        "posthoc_direction_separation_change",
        ascending=False,
    ).head(50)

    paths = {
        "manifest": output_dir / "original_race_aggregate_manifest.json",
        "component_ablation": output_dir / "original_race_component_ablation_all_seeds.csv",
        "summary": output_dir / "original_race_component_summary.csv",
        "paired_deltas": output_dir / "original_race_paired_deltas_all_seeds.csv",
        "delta_summary": output_dir / "original_race_delta_summary.csv",
        "score_equivalence": output_dir / "original_race_score_equivalence_all_seeds.csv",
        "score_equivalence_summary": output_dir / "original_race_score_equivalence_summary.csv",
        "direction_audit": output_dir / "original_race_direction_audit_all_seeds.csv",
        "top_posthoc_directions": output_dir / "original_race_top_posthoc_directions.csv",
        "source_compatibility": output_dir / "original_race_source_compatibility_all_seeds.csv",
        "partition_audit": output_dir / "original_race_partition_audit_all_seeds.csv",
    }
    results.to_csv(paths["component_ablation"], index=False)
    summary.to_csv(paths["summary"], index=False)
    deltas.to_csv(paths["paired_deltas"], index=False)
    delta_summary.to_csv(paths["delta_summary"], index=False)
    score_equivalence.to_csv(paths["score_equivalence"], index=False)
    equivalence_summary.to_csv(paths["score_equivalence_summary"], index=False)
    direction.to_csv(paths["direction_audit"], index=False)
    top_posthoc_directions.to_csv(paths["top_posthoc_directions"], index=False)
    source_compatibility.to_csv(paths["source_compatibility"], index=False)
    partitions.to_csv(paths["partition_audit"], index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "seeds": list(seeds),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_version": platform.python_version(),
        "input_files": [
            {
                "seed": int(seed),
                "component_ablation": str(_seed_file(input_root, seed, "original_race_component_ablation.csv")),
                "component_ablation_sha256": _sha256(_seed_file(input_root, seed, "original_race_component_ablation.csv")),
            }
            for seed in seeds
        ],
        "outputs": {name: str(path) for name, path in paths.items()},
        "interpretation_boundary": (
            "Aggregate is diagnostic-only. Anomaly outcomes are post-hoc and must not tune "
            "SS-RACE thresholds, compatibility, safe gates, or transfer weights."
        ),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = aggregate_original_race_component_ablation(
        input_root=args.input_root,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
    )
    print(f"Wrote aggregate Original RACE component-ablation outputs to {paths['manifest'].parent}")


if __name__ == "__main__":
    main()
