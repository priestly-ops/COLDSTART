from __future__ import annotations

import argparse

import experiments.run_e10_calibration_robustness as e10


def _csv_ints(value: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not vals or len(vals) != len(set(vals)):
        raise argparse.ArgumentTypeError("Expected unique comma-separated integers")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a subset of the frozen E10 design for smoke testing."
    )
    parser.add_argument("--n-values", type=_csv_ints, default=(100,))
    parser.add_argument("--seeds", type=_csv_ints, default=(0,))
    parser.add_argument("--m-values", type=_csv_ints, default=(50, 99, 100))
    args = parser.parse_args()

    if any(v not in e10.COMMISSIONING_GRID for v in args.n_values):
        raise ValueError(f"n-values must be subset of {e10.COMMISSIONING_GRID}")
    if any(v not in e10.SEEDS for v in args.seeds):
        raise ValueError("seeds must be subset of frozen 0..19")
    if any(v not in e10.CALIBRATION_PREFIX_SIZES for v in args.m_values):
        raise ValueError(
            f"m-values must be subset of {e10.CALIBRATION_PREFIX_SIZES}"
        )

    e10.COMMISSIONING_GRID = tuple(args.n_values)
    e10.SEEDS = tuple(args.seeds)
    e10.CALIBRATION_PREFIX_SIZES = tuple(args.m_values)
    e10.main()


if __name__ == "__main__":
    main()
