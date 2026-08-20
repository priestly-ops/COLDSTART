"""P0.8a AURSAD acquisition-session audit.

This audit uses only sample numbers, timestamps, and labels to characterize the
recording structure before defining any source/target commissioning split.
It does not fit anomaly detectors and does not use anomaly performance.

A new acquisition segment is inferred whenever the first timestamp of the next
sample is lower than the previous sample's first timestamp. This is deliberately
simple and transparent: sample_nr is monotone in the stored file, while the
observed timestamp clock clearly resets between recording blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "aursad" / "AURSAD.h5"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p08a_aursad_session_audit"
PROTOCOL_VERSION = "p08a-aursad-session-audit-v1"

LABEL_NAMES = {
    0: "normal",
    1: "damaged_screw",
    2: "extra_component",
    3: "missing_screw",
    4: "damaged_thread",
    5: "loosening_picking",
}


def _read_sample_metadata(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        g = f["complete_data"]
        b1 = [x.decode("utf-8") for x in g["block1_items"][:]]
        b2 = [x.decode("utf-8") for x in g["block2_items"][:]]
        b3 = [x.decode("utf-8") for x in g["block3_items"][:]]

        ts = g["block1_values"][:, b1.index("timestamp")]
        label = g["block2_values"][:, b2.index("label")]
        sample = g["block3_values"][:, b3.index("sample_nr")]

    row = pd.DataFrame({
        "sample_nr": sample.astype(np.int64),
        "timestamp": ts.astype(np.float64),
        "label": label.astype(np.int64),
    })

    sample_df = (
        row.groupby("sample_nr", sort=True)
        .agg(
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            n_rows=("timestamp", "size"),
            label=("label", "first"),
            label_nunique=("label", "nunique"),
        )
        .reset_index()
        .sort_values("sample_nr")
        .reset_index(drop=True)
    )

    if not (sample_df.label_nunique == 1).all():
        raise RuntimeError("Mixed labels detected within at least one sample")
    return sample_df


def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    s = _read_sample_metadata(dataset)

    prev_first = s.first_timestamp.shift(1)
    reset = (s.first_timestamp < prev_first).fillna(False)
    s["timestamp_reset"] = reset.astype(bool)
    s["segment_id"] = reset.cumsum().astype(int)
    s["label_name"] = s.label.map(LABEL_NAMES).fillna("unknown")

    s.to_csv(output / "p08a_sample_metadata.csv", index=False)

    segment_rows = []
    for seg, g in s.groupby("segment_id", sort=True):
        counts = g.label.value_counts().to_dict()
        segment_rows.append({
            "segment_id": int(seg),
            "sample_nr_min": int(g.sample_nr.min()),
            "sample_nr_max": int(g.sample_nr.max()),
            "samples": int(len(g)),
            "first_timestamp": float(g.first_timestamp.iloc[0]),
            "last_timestamp": float(g.last_timestamp.iloc[-1]),
            "normal_n": int(counts.get(0, 0)),
            "damaged_screw_n": int(counts.get(1, 0)),
            "extra_component_n": int(counts.get(2, 0)),
            "missing_screw_n": int(counts.get(3, 0)),
            "damaged_thread_n": int(counts.get(4, 0)),
            "loosening_picking_n": int(counts.get(5, 0)),
            "primary_screwdriving_n": int(sum(counts.get(k, 0) for k in range(5))),
            "anomaly_primary_n": int(sum(counts.get(k, 0) for k in range(1, 5))),
        })
    seg_df = pd.DataFrame(segment_rows)
    seg_df.to_csv(output / "p08a_segment_summary.csv", index=False)

    reset_rows = s.loc[s.timestamp_reset, [
        "sample_nr", "first_timestamp", "label", "label_name", "segment_id"
    ]].copy()
    reset_rows.to_csv(output / "p08a_timestamp_resets.csv", index=False)

    label_by_segment = (
        s.groupby(["segment_id", "label", "label_name"], sort=True)
        .size().rename("samples").reset_index()
    )
    label_by_segment.to_csv(output / "p08a_label_by_segment.csv", index=False)

    healthy_segments = seg_df.loc[seg_df.normal_n > 0, [
        "segment_id", "sample_nr_min", "sample_nr_max", "samples", "normal_n",
        "anomaly_primary_n", "loosening_picking_n"
    ]].copy()
    healthy_segments.to_csv(output / "p08a_healthy_segment_candidates.csv", index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(dataset),
        "total_samples": int(len(s)),
        "inferred_segments": int(seg_df.segment_id.nunique()),
        "timestamp_resets": int(s.timestamp_reset.sum()),
        "label_counts": {str(int(k)): int(v) for k, v in s.label.value_counts().sort_index().items()},
        "normal_samples": int((s.label == 0).sum()),
        "primary_anomaly_samples": int(s.label.isin([1, 2, 3, 4]).sum()),
        "supplementary_label5_samples": int((s.label == 5).sum()),
        "segmentation_rule": "new segment when first_timestamp decreases versus previous sample in sample_nr order",
        "important_note": (
            "sample_nr is not treated as a global physical clock. The audit only uses timestamp resets "
            "to identify candidate acquisition blocks before any source/target split is frozen."
        ),
    }
    (output / "p08a_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print("\nSegment summary:")
    print(seg_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
