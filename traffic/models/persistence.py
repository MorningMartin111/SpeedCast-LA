"""Export a Persistence forecast by repeating the last available sensor speeds."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from contextlib import suppress
from pathlib import Path

import numpy as np

from traffic.data.bundle import HORIZON
from traffic.data.dataset import load_bundle
from traffic.eval.io import SCORED_SPLITS, atomic_output, read_run_metadata, write_predictions


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export_persistence(data: str | Path, run_dir: str | Path, split: str) -> Path:
    """Export one split using the bundle's causally filled speeds at each origin."""
    if split not in SCORED_SPLITS:
        raise ValueError(f"Expected one of {SCORED_SPLITS}, got {split!r}")
    bundle = load_bundle(data)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "run.json"
    if not metadata_path.exists():
        status = _git("status", "--porcelain")
        metadata = {
            "model": "persistence",
            "seed": None,
            "protocol_version": bundle.manifest["protocol_version"],
            "data_id": bundle.data_id,
            "units": "mph",
            "code_commit": _git("rev-parse", "HEAD"),
            "code_dirty": None if status is None else bool(status),
            "implementation_source": "Protocol v1.0 Persistence baseline",
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "configuration": None,
            "best_epoch": None,
            "val_mae_mph": None,
            "checkpoint": None,
        }
        with suppress(FileExistsError), atomic_output(metadata_path, overwrite=False) as temporary:
            temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata = read_run_metadata(directory, bundle)
    if metadata["model"] != "persistence" or metadata["seed"] is not None:
        raise ValueError("Choose a run directory dedicated to Persistence")

    origins = bundle.origins[split]
    # filled_mph at the origin is the same causal speed returned as last_mph by the dataset.
    last_mph = bundle.filled_mph[origins]
    predictions = np.broadcast_to(last_mph[:, None, :], (len(origins), HORIZON, len(bundle.sensor_ids)))
    return write_predictions(directory, split, bundle, origins, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=SCORED_SPLITS)
    args = parser.parse_args()
    try:
        output = export_persistence(args.data, args.run_dir, args.split)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Persistence export failed: {exc}\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
