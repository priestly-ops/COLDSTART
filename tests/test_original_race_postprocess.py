import pandas as pd

from experiments.postprocess_original_race_component_ablation import (
    aggregate_original_race_component_ablation,
)


def _write_seed(root, seed: int) -> None:
    folder = root / f"original_race_component_ablation_seed{seed}"
    folder.mkdir()
    rows = [
        {
            "source_pair_id": f"near_shift_N10_seed{seed}",
            "source_group": "near",
            "target_group": "target",
            "N": 10,
            "seed": seed,
            "detector": "TargetOnly",
            "recall": 0.2,
            "FPR": 0.01,
            "AUROC": 0.8,
            "AUPRC": 0.9,
            "success": 0.0,
            "threshold": 10.0,
        },
        {
            "source_pair_id": f"near_shift_N10_seed{seed}",
            "source_group": "near",
            "target_group": "target",
            "N": 10,
            "seed": seed,
            "detector": "OriginalRACE",
            "recall": 0.5,
            "FPR": 0.01,
            "AUROC": 0.9,
            "AUPRC": 0.95,
            "success": 0.0,
            "threshold": 5.0,
        },
    ]
    pd.DataFrame(rows).to_csv(folder / "original_race_component_ablation.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_pair_id": f"near_shift_N10_seed{seed}",
                "source_group": "near",
                "target_group": "target",
                "N": 10,
                "seed": seed,
                "direction": 0,
                "posthoc_direction_separation_change": 1.0 + seed,
            }
        ]
    ).to_csv(folder / "original_race_direction_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_pair_id": f"near_shift_N10_seed{seed}",
                "N": 10,
                "seed": seed,
                "reference_detector": "TargetOnly",
                "candidate_detector": "OriginalRACE",
                "score_split": "eval",
                "affine_r2": 0.5,
                "spearman_score_corr": 0.7,
                "number_changed_predictions": 3,
                "score_equivalence_flag": "",
            }
        ]
    ).to_csv(folder / "original_race_score_equivalence.csv", index=False)
    pd.DataFrame([{"source_pair_id": f"near_shift_N10_seed{seed}", "seed": seed, "no_overlap": True}]).to_csv(
        folder / "original_race_partition_audit.csv",
        index=False,
    )
    pd.DataFrame([{"source_pair_id": f"near_shift_N10_seed{seed}", "seed": seed, "projector_similarity": 0.2}]).to_csv(
        folder / "original_race_source_compatibility.csv",
        index=False,
    )


def test_original_race_postprocess_writes_aggregate_outputs(tmp_path):
    _write_seed(tmp_path, 0)
    _write_seed(tmp_path, 1)
    output_dir = tmp_path / "aggregate"

    paths = aggregate_original_race_component_ablation(
        input_root=tmp_path,
        output_dir=output_dir,
        seeds=(0, 1),
    )

    assert paths["manifest"].exists()
    delta_summary = pd.read_csv(paths["delta_summary"])
    original = delta_summary[delta_summary["candidate_detector"] == "OriginalRACE"].iloc[0]
    assert original["delta_recall_mean"] == 0.3
    assert original["runs"] == 2
    equivalence = pd.read_csv(paths["score_equivalence_summary"])
    assert equivalence["equivalence_flags"].iloc[0] == 0
