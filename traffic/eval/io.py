"""Read and write protocol prediction files in bundle order."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, is_zipfile

import numpy as np
from numpy.typing import ArrayLike

from traffic.data.bundle import HORIZON
from traffic.data.dataset import Bundle

SCORED_SPLITS = ("val", "test")


@contextmanager
def atomic_output(path: Path, *, overwrite: bool = True):
    """Publish a complete file from a temporary sibling."""
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        if overwrite:
            temporary.replace(path)
        else:
            path.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _axes(bundle: Bundle, split: str) -> dict[str, np.ndarray]:
    if split not in bundle.origins:
        raise ValueError(f"Unknown split {split!r}; expected one of {tuple(bundle.origins)}")
    origins = bundle.origins[split]
    return {
        "origin_idx": origins,
        "origin_time": np.asarray([bundle.timestamps[i] for i in origins], dtype=str),
        "sensor_id": np.asarray(bundle.sensor_ids, dtype=str),
        "split": np.asarray(split),
    }


def _validate_speeds(predictions: np.ndarray, origins: np.ndarray, sensors: np.ndarray) -> None:
    shape = (origins.size, HORIZON, sensors.size)
    if not isinstance(predictions, np.ndarray) or predictions.shape != shape or predictions.dtype != np.float32:
        raise ValueError(f"pred_mph must have shape {shape} and dtype float32")
    if not np.isfinite(predictions).all():
        raise ValueError("pred_mph must contain only finite mph values")


def write_predictions(
    run_dir: str | Path, split: str, bundle: Bundle, origin_idx: ArrayLike, pred_mph: ArrayLike
) -> Path:
    """Export a complete, bundle-ordered split as float32 mph and Unicode metadata."""
    axes = _axes(bundle, split)
    origins = np.asarray(origin_idx)
    if origins.dtype.kind not in "iu" or not np.array_equal(origins, axes["origin_idx"]):
        raise ValueError(f"origin_idx must contain all {split} origins in bundle order")
    predictions = np.asarray(pred_mph)
    if predictions.dtype.kind not in "fiu":
        raise ValueError("pred_mph must contain real numbers")
    with np.errstate(over="ignore"):
        predictions = predictions.astype(np.float32, copy=False)
    _validate_speeds(predictions, axes["origin_idx"], axes["sensor_id"])

    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}.npz"
    with atomic_output(path) as temporary, temporary.open("wb") as handle:
        np.savez_compressed(handle, pred_mph=predictions, **axes)
    return path


def read_run_metadata(run_dir: str | Path, bundle: Bundle) -> dict:
    metadata = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("run.json must contain a JSON object")
    expected = {"data_id": bundle.data_id, "protocol_version": bundle.manifest["protocol_version"], "units": "mph"}
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"run.json {field} must match the bundle ({value!r})")
    if not isinstance(metadata.get("model"), str) or not metadata["model"].strip():
        raise ValueError("run.json must declare a model name")
    if "seed" not in metadata or (metadata["seed"] is not None and type(metadata["seed"]) is not int):
        raise ValueError("run.json seed must be an integer or null")
    return metadata


def read_predictions(
    run_dir: str | Path, split: str, bundle: Bundle, expected_sha256: str | None = None
) -> np.ndarray:
    """Load mph predictions and verify their shape, dtype and axis metadata."""
    axes = _axes(bundle, split)
    path = Path(run_dir) / f"{split}.npz"
    with path.open("rb") as handle:
        if expected_sha256 is not None and hashlib.file_digest(handle, "sha256").hexdigest() != expected_sha256:
            raise ValueError(f"{path.name} SHA-256 differs from run.json")
        handle.seek(0)
        if not is_zipfile(handle):
            raise ValueError(f"{path.name} must be a NumPy .npz archive")
        handle.seek(0)
        try:
            with np.load(handle, allow_pickle=False) as archive:
                if set(archive.files) != {*axes, "pred_mph"}:
                    raise ValueError(f"Unexpected prediction fields in {path.name}: {archive.files}")
                for field, expected in axes.items():
                    values = archive[field]
                    if not isinstance(values, np.ndarray):
                        raise ValueError(f"{field} in {path.name} must be a NumPy array")
                    dtype_matches = values.dtype == np.int64 if field == "origin_idx" else values.dtype.kind == "U"
                    if not dtype_matches or not np.array_equal(values, expected):
                        raise ValueError(f"{field} in {path.name} does not match the {split} bundle order or format")
                predictions = archive["pred_mph"]
        except BadZipFile as exc:
            raise ValueError(f"Unreadable prediction archive {path.name}") from exc
    _validate_speeds(predictions, axes["origin_idx"], axes["sensor_id"])
    return predictions
