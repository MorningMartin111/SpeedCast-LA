"""Align DCRNN author's METR-LA HDF5, directed graph, and sensor coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import warnings

import numpy as np

from traffic.data.download import SOURCES, UPSTREAM_REVISION, sha256_file


def sensor_id(value) -> str:
    """Decode Python 2 pickle bytes before comparing IDs with HDF5 columns."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_graph(path: Path) -> tuple[list[str], np.ndarray]:
    # Only load the matched graph from the documented, trusted author release.
    with path.open("rb") as stream:
        contents = pickle.load(stream, encoding="latin1")
    if not isinstance(contents, (tuple, list)) or len(contents) != 3:
        raise ValueError("Graph pickle must contain [sensor_ids, id_to_index, adjacency]")
    raw_ids, mapping, adjacency = contents
    ids = [sensor_id(value) for value in raw_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Graph sensor IDs must be unique and nonempty")
    decoded_mapping = {sensor_id(key): value for key, value in mapping.items()}
    if len(decoded_mapping) != len(mapping) or set(decoded_mapping) != set(ids):
        raise ValueError("Graph index mapping and graph sensor IDs disagree")
    if any(decoded_mapping[value] != index for index, value in enumerate(ids)):
        raise ValueError("Graph ID-to-index mapping disagrees with ordered sensor IDs")
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.shape != (len(ids), len(ids)):
        raise ValueError("Graph adjacency dimensions disagree with ordered sensor IDs")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("Graph adjacency must contain finite nonnegative weights")
    return ids, adjacency


def read_coordinates(path: Path, ids: list[str]) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(path, dtype={"sensor_id": str})
    required = {"sensor_id", "latitude", "longitude"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Coordinates must contain columns {sorted(required)}")
    frame["sensor_id"] = frame["sensor_id"].map(sensor_id)
    if frame["sensor_id"].duplicated().any():
        raise ValueError("Coordinate sensor IDs must be unique")
    frame = frame.set_index("sensor_id")
    extra = set(frame.index) - set(ids)
    if extra:
        raise ValueError(f"Coordinate IDs absent from the graph: {sorted(extra)[:5]}")
    aligned = frame.reindex(ids)[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    if np.isinf(aligned).any():
        raise ValueError("Coordinate values cannot be infinite")
    for axis, limit in enumerate((90, 180)):
        values = aligned[:, axis]
        if (np.abs(values[np.isfinite(values)]) > limit).any():
            raise ValueError("Coordinates are outside latitude/longitude bounds")
    return aligned


def read_speed_frame(path: Path):
    """Read modern pandas stores or the author's legacy single numeric block.

    pandas 3 rejects the release's byte-valued ``pandas_type`` metadata. The
    fallback reads only the documented fixed-frame numeric arrays; it never
    edits the original HDF5 or unpickles its old calendar-frequency metadata.
    """
    import pandas as pd

    try:
        return pd.read_hdf(path), "pandas.read_hdf"
    except TypeError as original_error:
        import tables

        with tables.open_file(path, mode="r") as handle:
            groups = [group for group in handle.root._v_groups.values()
                      if sensor_id(getattr(group._v_attrs, "pandas_type", "")) == "frame"]
            if len(groups) != 1:
                raise original_error
            group = groups[0]
            required = {"axis0", "axis1", "block0_items", "block0_values"}
            if not required.issubset(group._v_children) or int(group._v_attrs.nblocks) != 1:
                raise original_error
            kind = sensor_id(getattr(group.axis1._v_attrs, "kind", ""))
            supported_units = {"datetime64": "ns", "datetime64[ns]": "ns", "datetime64[us]": "us", "datetime64[ms]": "ms", "datetime64[s]": "s"}
            if kind not in supported_units:
                raise original_error
            columns = [sensor_id(value) for value in group.axis0[:]]
            block_columns = [sensor_id(value) for value in group.block0_items[:]]
            if columns != block_columns:
                raise ValueError("Legacy HDF5 axis and numeric block columns disagree")
            epoch_ns = group.axis1[:]
            values = group.block0_values[:]
            if not np.issubdtype(epoch_ns.dtype, np.integer) or not np.issubdtype(values.dtype, np.number):
                raise original_error
            if values.shape != (len(epoch_ns), len(columns)):
                raise ValueError("Legacy HDF5 numeric block has unexpected axis order")
            # This fallback handles the release's naive timestamps only.
            if getattr(group.axis1._v_attrs, "tz", None) is not None:
                raise original_error
            frame = pd.DataFrame(values, index=pd.to_datetime(epoch_ns, unit=supported_units[kind]), columns=columns)
            return frame, "legacy-fixed-numeric-tables (pandas byte-metadata compatibility)"


def prepare(raw_dir: Path, out: Path, *, speed_file="metr-la.h5", graph_file="adj_mx.pkl",
            locations_file="graph_sensor_locations.csv") -> dict:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Real-data preparation requires pandas and tables: python -m pip install -r requirements-data.txt") from exc
    from traffic.data.bundle import build_bundle
    from traffic.data.dataset import load_bundle

    paths = {"metr-la.h5": raw_dir / speed_file,
             "adj_mx.pkl": raw_dir / graph_file,
             "graph_sensor_locations.csv": raw_dir / locations_file}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run python -m traffic.data.download --out {raw_dir}")
    ids, adjacency = read_graph(paths["adj_mx.pkl"])
    coordinates = read_coordinates(paths["graph_sensor_locations.csv"], ids)
    frame, hdf_reader = read_speed_frame(paths["metr-la.h5"])
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Speed HDF5 must contain a DataFrame with a DatetimeIndex")
    original_shape = list(frame.shape)
    original_ids = [sensor_id(value) for value in frame.columns]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("HDF5 sensor columns are not unique after decoding")
    if set(original_ids) != set(ids):
        raise ValueError("HDF5 and graph sensor ID sets differ: "
                         f"missing={sorted(set(ids)-set(original_ids))[:5]}, "
                         f"extra={sorted(set(original_ids)-set(ids))[:5]}")
    if frame.index.hasnans:
        raise ValueError("Speed timestamps contain NaT")
    duplicate_count = int(frame.index.duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Speed table contains {duplicate_count} duplicate absolute timestamps")
    frame.columns = original_ids
    was_sorted = frame.index.is_monotonic_increasing
    frame = frame.sort_index().loc[:, ids]
    timestamps = [value.isoformat() for value in frame.index]
    # asi8 represents UTC elapsed nanoseconds for timezone-aware indices.
    intervals = np.diff(frame.index.as_unit("ns").asi8) / 1_000_000_000
    values, counts = np.unique(intervals, return_counts=True)
    raw_speed = frame.to_numpy(dtype=np.float32)
    valid = np.isfinite(raw_speed) & (raw_speed > 0)
    provenance_path = raw_dir / "sources.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    sources = {
        "dataset": "METR-LA",
        "repository": "https://github.com/liyaguang/DCRNN",
        "revision": provenance.get("revision", UPSTREAM_REVISION),
        "time_assumption": "Source timezone-naive local wall-clock timestamps retained without localization" if frame.index.tz is None else f"Source timezone {frame.index.tz}; elapsed intervals in UTC, calendar in local clock",
        "units": "mph",
        "files": {},
    }
    if provenance.get("download_date"):
        sources["download_date"] = provenance["download_date"]
    for canonical_name, path in paths.items():
        previous = provenance.get("files", {}).get(canonical_name, {})
        digest = sha256_file(path)
        if previous.get("sha256") and previous["sha256"] != digest:
            raise ValueError(f"Raw source checksum no longer matches sources.json: {path.name}")
        sources["files"][canonical_name] = {
            "actual_filename": path.name, "sha256": digest,
            "bytes": path.stat().st_size, "url": previous.get("url", SOURCES[canonical_name]),
            "url_status": "download provenance" if previous else "documented author source; local acquisition not verified",
        }
    build_bundle(out, timestamps, ids, coordinates, raw_speed, adjacency, source=sources)
    bundle = load_bundle(out)
    split_summary = {}
    target_sets = {}
    for split, origins in bundle.origins.items():
        target_sets[split] = {int(origin) + horizon for origin in origins for horizon in range(1, 13)}
        first_last = []
        for origin in ([int(origins[0]), int(origins[-1])] if len(origins) else []):
            first_last.append({"origin_idx": origin, "origin_time": timestamps[origin],
                               "history_start": timestamps[origin - 11],
                               "target_start": timestamps[origin + 1], "target_end": timestamps[origin + 12]})
        split_summary[split] = {"count": len(origins), "first_last_windows": first_last}
    if target_sets["train"] & target_sets["val"] or target_sets["val"] & target_sets["test"]:
        raise AssertionError("Future label timestamps overlap between adjacent splits")
    trace = []
    for graph_index in sorted(set([0, len(ids) // 2, len(ids) - 1])):
        trace.append({"graph_index": graph_index, "sensor_id": ids[graph_index],
                      "original_speed_column_index": original_ids.index(ids[graph_index]),
                      "first_raw_speed_mph": float(raw_speed[0, graph_index]) if np.isfinite(raw_speed[0, graph_index]) else None})
    train_end = int(np.floor(0.70 * len(frame)))
    no_training_reference = [ids[index] for index in range(len(ids)) if not valid[:train_end, index].any()]
    if no_training_reference:
        warnings.warn(f"No valid training observations/reference speed: {no_training_reference}")
    audit = {
        "data_id": bundle.data_id, "protocol_version": "1.0", "units": "mph",
        "original_shape": original_shape, "prepared_shape": list(raw_speed.shape),
        "date_range": [timestamps[0], timestamps[-1]], "duplicate_timestamp_count": duplicate_count,
        "original_timestamps_sorted": bool(was_sorted),
        "hdf_reader": hdf_reader,
        "interval_counts_seconds": {str(float(value)): int(count) for value, count in zip(values, counts)},
        "time_semantics": sources["time_assumption"],
        "valid_rate_by_sensor": {value: float(valid[:, index].mean()) for index, value in enumerate(ids)},
        "missing_rate": float(1 - valid.mean()),
        "input_fill_fraction": float(1 - valid.mean()),
        "forward_filled_fraction": bundle.manifest["quality"]["forward_filled_fraction"],
        "training_mean_fallback_fraction": bundle.manifest["quality"]["mean_fallback_fraction"],
        "missing_coordinates": [ids[index] for index in range(len(ids)) if not np.isfinite(coordinates[index]).all()],
        "no_training_reference": no_training_reference,
        "source_trace": trace, "splits": split_summary,
        "future_label_sets_disjoint": True,
        "validation_history_rule": "The first validation origin may read its 12 observed history rows before the train/val boundary; all 12 future labels lie inside validation. Causal fill may inherit earlier valid observations, and resets at an elapsed-time gap.",
        "calendar_ranges": {"tod_slot": [int(bundle.tod_slot.min()), int(bundle.tod_slot.max())],
                            "dow": [int(bundle.dow.min()), int(bundle.dow.max())]},
        "source": sources,
    }
    # Derive field shapes through the public Dataset, not a separate preparation path.
    audit["first_sample_shapes"] = {}
    for split in bundle.origins:
        if len(bundle.origins[split]):
            audit["first_sample_shapes"][split] = {key: list(value.shape) for key, value in bundle.dataset(split)[0].items()}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("data_id", "prepared_shape", "date_range", "interval_counts_seconds", "missing_rate", "splits", "source_trace", "first_sample_shapes", "calendar_ranges")}, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/metr-la-v1"))
    parser.add_argument("--speed-file", default="metr-la.h5")
    parser.add_argument("--graph-file", default="adj_mx.pkl")
    parser.add_argument("--locations-file", default="graph_sensor_locations.csv")
    args = parser.parse_args()
    prepare(args.raw_dir, args.out, speed_file=args.speed_file, graph_file=args.graph_file, locations_file=args.locations_file)


if __name__ == "__main__":
    main()
