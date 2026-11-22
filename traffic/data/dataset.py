"""Public PyTorch Dataset and validated protocol v1.0 bundle reader."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .bundle import (HISTORY, HORIZON, PROTOCOL_VERSION, SEMANTICS, SPLIT_NAMES,
                     fingerprint, make_origins, parse_timestamps, validate_nodes)


@dataclass
class Bundle:
    path: Path
    manifest: dict
    sensor_ids: list[str]
    timestamps: list[str]
    coordinates: np.ndarray
    speed_mph: np.ndarray
    filled_mph: np.ndarray
    observed: np.ndarray
    ref_speed_mph: np.ndarray
    tod_slot: np.ndarray
    dow: np.ndarray
    adj_dir: np.ndarray
    origins: dict[str, np.ndarray]
    mean_mph: float
    std_mph: float
    data_id: str

    def dataset(self, split: str) -> "TrafficDataset":
        return TrafficDataset(self, split)


class TrafficDataset(Dataset):
    def __init__(self, bundle: Bundle, split: str):
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {split!r}; expected {SPLIT_NAMES}")
        self.bundle = bundle
        self.split = split
        self.origins = bundle.origins[split]

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        origin = int(self.origins[index])
        history = slice(origin - HISTORY + 1, origin + 1)
        targets = slice(origin + 1, origin + HORIZON + 1)
        bundle = self.bundle
        nodes = len(bundle.sensor_ids)
        x = np.empty((HISTORY, nodes, 3), dtype=np.float32)
        x[:, :, 0] = (bundle.filled_mph[history] - bundle.mean_mph) / bundle.std_mph
        x[:, :, 1] = bundle.tod_slot[history, None] / 288
        x[:, :, 2] = bundle.dow[history, None]
        return {
            "x": torch.from_numpy(x),
            "y_mph": torch.from_numpy(bundle.speed_mph[targets].copy()),
            "y_mask": torch.from_numpy(bundle.observed[targets].copy()),
            "last_mph": torch.from_numpy(bundle.filled_mph[origin].copy()),
            "origin_observed": torch.from_numpy(bundle.observed[origin].copy()),
            "origin_idx": torch.tensor(origin, dtype=torch.int64),
        }


def _arrays(path: Path, expected: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError(f"Unexpected fields in {path.name}: {archive.files}")
        return {name: archive[name].copy() for name in archive.files}


def load_bundle(path: str | Path, verify: bool = True) -> Bundle:
    """Load and validate a bundle; verify=False skips only the logical SHA-256 check."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    scaler = json.loads((path / "scaler.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION or manifest.get("processing") != SEMANTICS or manifest.get("units") != "mph":
        raise ValueError("Unsupported protocol, units or processing semantics")
    with (path / "timestamps.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["time_idx", "timestamp_local"]:
            raise ValueError("Invalid timestamps.csv header")
        rows = list(reader)
    if [row["time_idx"] for row in rows] != [str(index) for index in range(len(rows))]:
        raise ValueError("timestamps.csv must preserve contiguous integer row order")
    timestamps, elapsed, tod, dow, time_semantics = parse_timestamps([row["timestamp_local"] for row in rows])
    if [row["timestamp_local"] for row in rows] != timestamps:
        raise ValueError("Timestamp strings must use the bundle's canonical local ISO format")
    with (path / "sensors.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sensor_id", "latitude", "longitude"]:
            raise ValueError("Invalid sensors.csv header")
        sensors = list(reader)
    ids = [row["sensor_id"] for row in sensors]
    coords = np.asarray([[float(row["latitude"]), float(row["longitude"])] for row in sensors], dtype=np.float64)
    series = _arrays(path / "series.npz", {"speed_mph", "filled_mph", "observed", "ref_speed_mph", "tod_slot", "dow"})
    adj = _arrays(path / "graph.npz", {"adj_dir"})["adj_dir"]
    origins = _arrays(path / "splits.npz", set(SPLIT_NAMES))
    validate_nodes(ids, coords, adj)
    total, nodes = len(timestamps), len(ids)
    if manifest.get("T") != total or manifest.get("N") != nodes:
        raise ValueError("Manifest dimensions differ from the data files")
    specs = {"speed_mph": ((total, nodes), np.float32), "filled_mph": ((total, nodes), np.float32),
             "observed": ((total, nodes), np.bool_), "ref_speed_mph": ((nodes,), np.float32),
             "tod_slot": ((total,), np.int64), "dow": ((total,), np.int64)}
    for name, (shape, dtype) in specs.items():
        if series[name].shape != shape or series[name].dtype != dtype:
            raise ValueError(f"Invalid {name} shape/dtype; expected {shape}/{np.dtype(dtype)}")
    if adj.dtype != np.float32:
        raise ValueError("adj_dir must be float32")
    if not np.isfinite(series["speed_mph"]).all() or not np.isfinite(series["filled_mph"]).all():
        raise ValueError("Speed arrays must be finite")
    if np.any(series["speed_mph"][~series["observed"]] != 0) or np.any(series["speed_mph"][series["observed"]] <= 0):
        raise ValueError("Observed speed/missing placeholder convention is violated")
    if (series["filled_mph"] <= 0).any() or np.isinf(series["ref_speed_mph"]).any() or (series["ref_speed_mph"][np.isfinite(series["ref_speed_mph"])] <= 0).any():
        raise ValueError("Filled and finite reference speeds must be positive")
    if not np.array_equal(series["tod_slot"], tod) or not np.array_equal(series["dow"], dow):
        raise ValueError("Calendar arrays differ from local timestamp semantics")
    if manifest.get("time_semantics") != time_semantics:
        raise ValueError("Manifest time semantics differ from timestamps")
    a, b = 7 * total // 10, 8 * total // 10
    if manifest.get("split_boundaries") != {"train_end": a, "val_end": b}:
        raise ValueError("Invalid time split boundaries")
    expected_origins = make_origins(elapsed)
    for name in SPLIT_NAMES:
        if origins[name].dtype != np.int64 or origins[name].ndim != 1 or not np.array_equal(origins[name], expected_origins[name]):
            raise ValueError(f"Invalid {name} origins or ordering")
    if manifest.get("origin_counts") != {name: len(values) for name, values in origins.items()}:
        raise ValueError("Manifest origin counts differ from splits.npz")
    if set(scaler) != {"mean_mph", "std_mph"} or not np.isfinite(list(scaler.values())).all() or scaler["std_mph"] <= 0:
        raise ValueError("Invalid training scaler")
    data_id = fingerprint(timestamps, ids, coords, series, adj, origins, scaler, time_semantics)
    if verify and data_id != manifest.get("data_id"):
        raise ValueError("Bundle fingerprint mismatch; regenerate the bundle from its raw source")
    return Bundle(path=path, manifest=manifest, sensor_ids=ids, timestamps=timestamps, coordinates=coords,
                  adj_dir=adj, origins=origins, mean_mph=float(scaler["mean_mph"]), std_mph=float(scaler["std_mph"]),
                  data_id=manifest["data_id"], **series)
