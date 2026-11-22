"""Generate the hand-checkable 120-row protocol v1.0 fixture."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .bundle import build_bundle


def fixture_inputs() -> dict:
    total = 120
    timestamps = [(datetime(2012, 3, 5) + timedelta(minutes=5 * row)).isoformat(sep=" ") for row in range(total)]
    raw = np.empty((total, 3), dtype=np.float32)
    raw[:, 0] = 40 + np.arange(total, dtype=np.float32) / 4
    raw[:, 1] = 60
    raw[[96, 99], 1] = 0
    raw[:, 2] = 50
    raw[89:92, 2] = 20
    raw[99:109, 2] = 20
    return {"timestamps": timestamps, "sensor_ids": ["S02", "S01", "S03"],
            "coordinates": np.asarray([[34.00, -118.30], [34.01, -118.29], [34.02, -118.28]]),
            "raw_speed": raw,
            "adj_dir": np.asarray([[0, 1, 0], [0, 0, 0.5], [0, 0, 0]], dtype=np.float32),
            "source": {"kind": "protocol-v1-fixture", "description": "Synthetic interface checks only; not METR-LA measurements"}}


def generate_fixture(out: str | Path) -> dict:
    return build_bundle(out, **fixture_inputs())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/fixture-v1"))
    args = parser.parse_args()
    manifest = generate_fixture(args.out)
    print(json.dumps({"out": str(args.out), "data_id": manifest["data_id"], "origin_counts": manifest["origin_counts"]}, indent=2))


if __name__ == "__main__":
    main()
