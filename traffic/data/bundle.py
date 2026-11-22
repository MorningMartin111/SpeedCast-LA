"""Protocol v1.0 preprocessing and a container-independent bundle fingerprint."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROTOCOL_VERSION = "1.0"
HISTORY = HORIZON = 12
SPLIT_NAMES = ("train", "val", "test")
SEMANTICS = {
    "history": HISTORY,
    "horizon": HORIZON,
    "interval_seconds": 300,
    "origin": "last input row; targets origin+1 through origin+12",
    "units": "mph",
    "observed": "finite raw speed strictly greater than zero",
    "missing_speed_placeholder": 0,
    "fill": "causal per-sensor forward fill; reset at gaps; training scalar mean fallback",
    "scaler": "all observed training-period speeds; scalar population std (ddof=0)",
    "reference_speed": "per-sensor training observed linear quantile 0.85; NaN when unavailable",
    "adj_dir": "row i to column j; directed original weights",
    "x_channels": ["standardized filled speed", "tod_slot / 288", "integer dow as float32"],
    "tod_slot": "local hour * 12 + local minute // 5; 0..287",
    "dow": "local Monday=0 through Sunday=6",
    "split": "floor(0.70*T), floor(0.80*T); all targets inside split; earlier history allowed",
    "window": "all 24 rows separated by exactly 300 elapsed seconds",
}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def parse_timestamps(timestamps: Sequence[str]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict]:
    """Keep local clock text/calendar while comparing aware times by UTC elapsed time."""
    if len(timestamps) < 24:
        raise ValueError("At least 24 timestamps are required for a 12-in/12-out window")
    dates = [datetime.fromisoformat(str(value)) for value in timestamps]
    aware = [date.utcoffset() is not None for date in dates]
    if any(aware) and not all(aware):
        raise ValueError("Cannot mix timezone-aware and naive timestamps")
    if any(date.second or date.microsecond or date.minute % 5 for date in dates):
        raise ValueError("Timestamps must lie on five-minute local clock slots")
    if all(aware):
        elapsed = np.array([round(date.timestamp() * 1_000_000) for date in dates], dtype=np.int64)
        semantics = {"timezone": "offset-aware source timestamps", "gap_clock": "UTC elapsed", "calendar_clock": "source local"}
    else:
        first = dates[0]
        elapsed = np.array([round((date - first).total_seconds() * 1_000_000) for date in dates], dtype=np.int64)
        semantics = {"timezone": "naive source local clock; no timezone conversion", "gap_clock": "source local", "calendar_clock": "source local"}
    if np.any(np.diff(elapsed) <= 0):
        raise ValueError("Timestamps must be unique and strictly increasing in elapsed time")
    canonical = [date.isoformat(sep=" ", timespec="seconds") for date in dates]
    tod_slot = np.asarray([date.hour * 12 + date.minute // 5 for date in dates], dtype=np.int64)
    dow = np.asarray([date.weekday() for date in dates], dtype=np.int64)
    return canonical, elapsed, tod_slot, dow, semantics


def make_origins(elapsed_us: np.ndarray) -> dict[str, np.ndarray]:
    total = len(elapsed_us)
    a, b = 7 * total // 10, 8 * total // 10
    origins = {name: [] for name in SPLIT_NAMES}
    breaks = np.concatenate(([0], np.cumsum(np.diff(elapsed_us) != 300_000_000)))
    for origin in range(HISTORY - 1, total - HORIZON):
        if breaks[origin + HORIZON] != breaks[origin - HISTORY + 1]:
            continue
        first, last = origin + 1, origin + HORIZON
        for name, lower, upper in (("train", 0, a), ("val", a, b), ("test", b, total)):
            if first >= lower and last < upper:
                origins[name].append(origin)
                break
    return {name: np.asarray(values, dtype=np.int64) for name, values in origins.items()}


def validate_nodes(sensor_ids: Sequence[str], coordinates: np.ndarray, adj_dir: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not sensor_ids or any(not isinstance(value, str) or not value for value in sensor_ids):
        raise ValueError("sensor_ids must be nonempty strings; explicitly decode byte IDs first")
    ids = list(sensor_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sensor IDs")
    n = len(ids)
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.shape != (n, 2):
        raise ValueError(f"coordinates must have shape {(n, 2)}")
    if np.isinf(coords).any() or np.any(np.abs(coords[:, 0][np.isfinite(coords[:, 0])]) > 90) or np.any(np.abs(coords[:, 1][np.isfinite(coords[:, 1])]) > 180):
        raise ValueError("Coordinates must be valid latitude/longitude values or NaN")
    adj = np.asarray(adj_dir, dtype=np.float32)
    if adj.shape != (n, n) or not np.isfinite(adj).all() or (adj < 0).any():
        raise ValueError(f"adj_dir must be finite, nonnegative and have shape {(n, n)}")
    return ids, coords.copy(), adj.copy()


def fingerprint(timestamps: Sequence[str], sensor_ids: Sequence[str], coordinates: np.ndarray,
                series: dict[str, np.ndarray], adj_dir: np.ndarray, origins: dict[str, np.ndarray],
                scaler: dict, time_semantics: dict, processing: dict = SEMANTICS) -> str:
    """Hash logical values, names, shapes and canonical little-endian dtypes, never ZIP bytes."""
    digest = hashlib.sha256()

    def block(name: str, payload: bytes) -> None:
        label = name.encode("utf-8")
        digest.update(len(label).to_bytes(8, "little"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    metadata = {"protocol_version": PROTOCOL_VERSION, "timestamps": list(timestamps), "sensor_ids": list(sensor_ids),
                "scaler": scaler, "time_semantics": time_semantics, "processing": processing}
    block("metadata", _json(metadata).encode("utf-8"))
    arrays = {"coordinates": coordinates, "adj_dir": adj_dir, **series,
              **{f"origins/{name}": origins[name] for name in SPLIT_NAMES}}
    for name, value in sorted(arrays.items()):
        arr = np.asarray(value)
        dtype = arr.dtype.newbyteorder("<")
        arr = np.array(arr, dtype=dtype, order="C", copy=True)
        if arr.dtype.kind == "f":
            arr[np.isnan(arr)] = np.nan  # Canonical quiet NaN payload, including missing coordinates.
            arr[arr == 0] = 0  # Canonical positive zero.
        block(name + "/header", _json({"shape": list(arr.shape), "dtype": dtype.str}).encode("utf-8"))
        block(name + "/values", arr.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def window_summary(timestamps: Sequence[str], origins: dict[str, np.ndarray]) -> dict:
    result = {}
    for name, values in origins.items():
        detail = {"count": len(values)}
        for label, index in (("first", 0), ("last", -1)):
            if len(values):
                origin = int(values[index])
                detail[label] = {"origin_idx": origin, "origin_time": timestamps[origin],
                                 "history_first_time": timestamps[origin - 11],
                                 "target_first_time": timestamps[origin + 1],
                                 "target_last_time": timestamps[origin + 12]}
        result[name] = detail
    return result


def build_bundle(out: str | Path, timestamps: Sequence[str], sensor_ids: Sequence[str],
                 coordinates: np.ndarray, raw_speed: np.ndarray, adj_dir: np.ndarray,
                 source: dict | None = None) -> dict:
    """Write the seven protocol files and return their manifest. Inputs are already ID aligned."""
    times, elapsed, tod_slot, dow, time_semantics = parse_timestamps(timestamps)
    ids, coords, adj = validate_nodes(sensor_ids, coordinates, adj_dir)
    raw = np.asarray(raw_speed, dtype=np.float32)
    total, nodes = len(times), len(ids)
    if raw.shape != (total, nodes):
        raise ValueError(f"raw_speed must have shape {(total, nodes)}")
    observed = np.isfinite(raw) & (raw > 0)
    speed = np.where(observed, raw, 0).astype(np.float32)
    a, b = 7 * total // 10, 8 * total // 10
    train_values = speed[:a][observed[:a]].astype(np.float64)
    if not train_values.size:
        raise ValueError("Training period contains no valid observed speeds")
    mean, std = float(train_values.mean()), float(train_values.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training population standard deviation is zero or non-finite")
    scaler = {"mean_mph": mean, "std_mph": std}
    ref_speed = np.full(nodes, np.nan, dtype=np.float32)
    no_train = []
    for node, sensor_id in enumerate(ids):
        values = speed[:a, node][observed[:a, node]]
        if len(values):
            ref_speed[node] = np.quantile(values, 0.85, method="linear")
        else:
            no_train.append(sensor_id)
    filled = np.empty_like(speed)
    previous = np.full(nodes, mean, dtype=np.float32)
    has_history = np.zeros(nodes, dtype=bool)
    fallback_count = forward_count = 0
    for row in range(total):
        if row == 0 or elapsed[row] - elapsed[row - 1] != 300_000_000:
            previous.fill(mean)
            has_history.fill(False)
        missing = ~observed[row]
        fallback_count += int(np.count_nonzero(missing & ~has_history))
        forward_count += int(np.count_nonzero(missing & has_history))
        previous = np.where(observed[row], speed[row], previous)
        has_history |= observed[row]
        filled[row] = previous
    series = {"speed_mph": speed, "filled_mph": filled, "observed": observed,
              "ref_speed_mph": ref_speed, "tod_slot": tod_slot, "dow": dow}
    origins = make_origins(elapsed)
    intervals, interval_counts = np.unique(np.diff(elapsed) / 1_000_000, return_counts=True)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_id": fingerprint(times, ids, coords, series, adj, origins, scaler, time_semantics),
        "source": source or {}, "time_semantics": time_semantics, "T": total, "N": nodes,
        "start_time": times[0], "end_time": times[-1], "units": "mph",
        "split_boundaries": {"train_end": a, "val_end": b},
        "origin_counts": {name: len(values) for name, values in origins.items()},
        "processing": SEMANTICS,
        "quality": {"duplicate_times": 0, "interval_seconds_counts": {str(float(delta)): int(count) for delta, count in zip(intervals, interval_counts)},
                    "observed_fraction": float(observed.mean()), "missing_fraction": float((~observed).mean()),
                    "valid_fraction_by_sensor": {sensor_id: float(observed[:, node].mean()) for node, sensor_id in enumerate(ids)},
                    "forward_filled_fraction": forward_count / raw.size, "mean_fallback_fraction": fallback_count / raw.size,
                    "sensors_without_training_observations": no_train,
                    "sensors_missing_coordinates": [ids[node] for node in range(nodes) if np.isnan(coords[node]).any()]},
        "windows": window_summary(times, origins),
    }
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_idx", "timestamp_local"])
        writer.writerows(enumerate(times))
    with (path / "sensors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sensor_id", "latitude", "longitude"])
        writer.writerows((sensor_id, repr(float(coords[node, 0])), repr(float(coords[node, 1])))
                         for node, sensor_id in enumerate(ids))
    np.savez_compressed(path / "series.npz", **series)
    np.savez_compressed(path / "graph.npz", adj_dir=adj)
    np.savez_compressed(path / "splits.npz", **origins)
    (path / "scaler.json").write_text(_json(scaler) + "\n", encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
