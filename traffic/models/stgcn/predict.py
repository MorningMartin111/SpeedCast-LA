"""Run inference in bundle order, exporting through B's shared writer when ready."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from traffic.data.dataset import load_bundle
from .model import STGCN
from .train import sha256_file, write_json


@torch.no_grad()
def predict(args: argparse.Namespace) -> dict:
    bundle = load_bundle(args.data)
    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata["data_id"] != bundle.data_id or metadata["units"] != "mph":
        raise ValueError("Run data_id or units differ from the loaded bundle")
    checkpoint_path = run_dir / "best.pt"
    if sha256_file(checkpoint_path) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from run.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["data_id"] != bundle.data_id or checkpoint["sensor_ids"] != list(bundle.sensor_ids):
        raise ValueError("Checkpoint data_id or sensor order differs from the loaded bundle")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config != checkpoint["config"] or config != metadata["configuration"]:
        raise ValueError("Run and checkpoint configurations disagree")
    torch.set_num_threads(config["num_threads"])
    device = torch.device(args.device)
    model = STGCN(bundle.adj_dir, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = bundle.dataset(args.split)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    origin_batches, prediction_batches = [], []
    for batch in loader:
        # Labels, masks, and all future values remain outside the model call.
        prediction = model(batch["x"].to(device)) * bundle.std_mph + bundle.mean_mph
        prediction_batches.append(prediction.cpu().numpy().astype(np.float32))
        origin_batches.append(batch["origin_idx"].numpy().astype(np.int64))
    origins = np.concatenate(origin_batches) if origin_batches else np.empty(0, dtype=np.int64)
    predictions = np.concatenate(prediction_batches) if prediction_batches else np.empty((0, 12, len(bundle.sensor_ids)), dtype=np.float32)
    if not np.array_equal(origins, bundle.origins[args.split]):
        raise ValueError("Inference origin order differs from bundle")
    if predictions.shape != (len(origins), 12, len(bundle.sensor_ids)) or not np.isfinite(predictions).all():
        raise ValueError("Prediction shape or finite-value check failed")
    summary = {"data_id": bundle.data_id, "model": config["model"], "split": args.split,
               "shape": list(predictions.shape), "units": "mph", "finite": True,
               "origin_order_verified": True, "sensor_order_verified": True,
               "first_origin": int(origins[0]) if len(origins) else None,
               "last_origin": int(origins[-1]) if len(origins) else None,
               "checkpoint_sha256": metadata["checkpoint_sha256"], "exported": False}
    if args.check_only:
        print(json.dumps(summary), flush=True)
        return summary
    try:
        writer = importlib.import_module("traffic.eval.io").write_predictions
    except ModuleNotFoundError as exc:
        if exc.name not in {"traffic.eval", "traffic.eval.io"}:
            raise
        raise ValueError("B's traffic.eval.io.write_predictions is not available. Use --check-only for in-memory verification, then rerun this command after B's PR is merged.") from exc
    writer(run_dir, args.split, bundle, origins, predictions)
    prediction_path = run_dir / f"{args.split}.npz"
    if not prediction_path.is_file():
        raise ValueError("B's writer did not create the expected split.npz")
    metadata.setdefault("prediction_sha256", {})[args.split] = sha256_file(prediction_path)
    write_json(run_dir / "run.json", metadata)
    summary["exported"] = True
    summary["prediction_path"] = str(prediction_path)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-only", action="store_true", help="Verify in-memory forecasts without requiring B's writer")
    args = parser.parse_args()
    try:
        predict(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")


if __name__ == "__main__":
    main()
