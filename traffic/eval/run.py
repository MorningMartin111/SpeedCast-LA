"""Score one validation or test prediction file against a data bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from traffic.data.bundle import HORIZON
from traffic.data.dataset import load_bundle

from .io import SCORED_SPLITS, atomic_output, read_predictions, read_run_metadata
from .metrics import masked_scores

METRIC_FIELDS = ("model", "seed", "split", "horizon_step", "n", "mae_mph", "rmse_mph", "bias_mph")


def score_run(data: str | Path, run_dir: str | Path, split: str) -> Path:
    """Write one metrics CSV with 15-, 30-, 60-minute and pooled scores."""
    if split not in SCORED_SPLITS:
        raise ValueError(f"Expected one of {SCORED_SPLITS}, got {split!r}")
    bundle = load_bundle(data)
    metadata = read_run_metadata(run_dir, bundle)
    checksums = metadata.get("prediction_sha256", {})
    if not isinstance(checksums, dict):
        raise ValueError("run.json prediction_sha256 must map splits to checksums")
    predictions = read_predictions(run_dir, split, bundle, checksums.get(split))

    target_indices = bundle.origins[split][:, None] + np.arange(1, HORIZON + 1)
    truth = bundle.speed_mph[target_indices]
    observed = bundle.observed[target_indices]
    common_fields = {"model": metadata["model"], "seed": metadata["seed"], "split": split}
    rows = []
    for horizon in (3, 6, 12):
        index = horizon - 1
        scores = masked_scores(truth[:, index], predictions[:, index], observed[:, index])
        rows.append({**common_fields, "horizon_step": horizon, **scores})
    rows.append({**common_fields, "horizon_step": "all12", **masked_scores(truth, predictions, observed)})

    output = Path(run_dir) / f"metrics_{split}.csv"
    with atomic_output(output) as temporary, temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=SCORED_SPLITS)
    args = parser.parse_args()
    try:
        output = score_run(args.data, args.run_dir, args.split)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Scoring failed: {exc}\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
