"""Train STGCN using masked MAE in mph and validation all12 early stopping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from traffic.data.dataset import load_bundle
from .model import SOURCE, STGCN


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
    status = command("status", "--porcelain")
    return {"code_commit": command("rev-parse", "HEAD"), "code_dirty": None if status is None else bool(status)}


def seed_everything(seed: int, threads: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def read_config(path: Path, max_epochs: int | None = None,
                max_train_batches: int | None = None, max_val_batches: int | None = None) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"model", "hidden_channels", "blocks", "temporal_kernel", "cheb_order", "dropout",
                "learning_rate", "weight_decay", "batch_size", "max_epochs", "patience", "num_workers", "num_threads"}
    optional = {"max_train_batches", "max_val_batches", "seed", "device"}
    if required - set(config) or set(config) - required - optional:
        raise ValueError(f"Config fields differ: missing={sorted(required - set(config))}, unknown={sorted(set(config) - required - optional)}")
    if max_epochs is not None:
        config["max_epochs"] = max_epochs
    for name, override in (("max_train_batches", max_train_batches), ("max_val_batches", max_val_batches)):
        config[name] = override if override is not None else config.get(name)
        if config[name] is not None and (not isinstance(config[name], int) or config[name] < 1):
            raise ValueError(f"{name} must be a positive integer when specified")
    for name in ("batch_size", "max_epochs", "patience", "num_threads"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(config["num_workers"], int) or config["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if not np.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    return config


def common_metric() -> object | None:
    """B's implementation is optional until the evaluation PR is merged."""
    try:
        module = importlib.import_module("traffic.eval.metrics")
    except ModuleNotFoundError as exc:
        if exc.name not in {"traffic.eval", "traffic.eval.metrics"}:
            raise
        return None
    return module.masked_scores


@torch.no_grad()
def validation_mae(model: STGCN, loader: DataLoader, mean: float, std: float,
                   device: torch.device, metric: object | None = None,
                   max_batches: int | None = None) -> tuple[float, int]:
    model.eval()
    error_sum, count = 0.0, 0
    truths, predictions, masks = [], [], []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        prediction = model(batch["x"].to(device)) * std + mean
        truth, mask = batch["y_mph"].to(device), batch["y_mask"].to(device)
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite validation prediction")
        selected = (prediction - truth)[mask].double()
        error_sum += selected.abs().sum().item()
        count += selected.numel()
        if metric is not None:
            truths.append(truth.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            masks.append(mask.cpu().numpy())
    if not count:
        raise ValueError("Validation set has no valid target scalars")
    mae = error_sum / count
    if metric is not None:
        scores = metric(np.concatenate(truths), np.concatenate(predictions), np.concatenate(masks))
        common_mae = scores.get("mae_mph", scores.get("mae"))
        if common_mae is None or not np.isclose(mae, common_mae, rtol=1e-6, atol=1e-6):
            raise ValueError(f"B's all12 MAE disagrees with training aggregation: local={mae}, common={scores}")
    return mae, count


def train(args: argparse.Namespace) -> dict:
    config = read_config(Path(args.config), args.max_epochs, args.max_train_batches, args.max_val_batches)
    seed_everything(args.seed, config["num_threads"])
    bundle = load_bundle(args.data)
    train_dataset, val_dataset = bundle.dataset("train"), bundle.dataset("val")
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Training and validation splits must both contain legal windows")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    config.update({"seed": args.seed, "device": str(device)})
    model = STGCN(bundle.adj_dir, config).to(device)
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Run directory is nonempty; choose a new --run-dir: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              generator=generator, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    metric = common_metric()
    metadata = {
        "model": config["model"], "seed": args.seed,
        "protocol_version": bundle.manifest["protocol_version"],
        "data_id": bundle.data_id, "units": "mph", **git_metadata(),
        "configuration": config, "implementation_source": SOURCE,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                        "cuda_available": torch.cuda.is_available(), "device": str(device), "platform": platform.platform(),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "train_windows": len(train_dataset), "val_windows": len(val_dataset),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "temporal_lengths": model.temporal_lengths,
        "input_channels": ["standardized_filled_speed", "tod_slot/288", "integer_dow_as_float"],
        "prediction_axes": ["sample", "horizon_step_1_to_12", "bundle_sensor_order"],
        "graph_processing": "max(A,A.T); diagonal removed; symmetric normalized Laplacian; lambda_max=2; T0..T(K-1)",
        "cross_node_mixing": "only graph convolution" if config["model"] == "stgcn" else "none",
        "selection_metric": ("val prefix all12 MAE in mph; smoke-only partial validation" if config["max_val_batches"]
                             else "val all12 MAE in mph; valid-scalar weighted"),
        "metric_backend": "traffic.eval.metrics.masked_scores cross-check" if metric else "local protocol-equivalent MAE; B cross-check pending",
        "budget_note": "CLI max-epochs override" if args.max_epochs is not None else "default config budget",
        "smoke_only": bool(config["max_train_batches"] or config["max_val_batches"]),
        "hyperparameter_trials": 1, "checkpoint": "best.pt", "status": "running",
        "source_sha256": {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
    }
    write_json(run_dir / "run.json", metadata)
    start = time.perf_counter()
    best_mae, best_epoch, stale = float("inf"), None, 0
    total_skipped = 0
    history_fields = ["epoch", "train_n", "train_mae_mph", "val_n", "val_mae_mph", "skipped_all_missing_batches", "elapsed_seconds"]
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            train_error, train_count, skipped = 0.0, 0, 0
            for batch_index, batch in enumerate(train_loader):
                if config["max_train_batches"] is not None and batch_index >= config["max_train_batches"]:
                    break
                mask = batch["y_mask"].to(device)
                if not mask.any():
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["x"].to(device)) * bundle.std_mph + bundle.mean_mph
                selected = (prediction - batch["y_mph"].to(device))[mask]
                if not torch.isfinite(selected).all():
                    raise ValueError("Nonfinite training error")
                loss = selected.abs().mean()
                loss.backward()
                optimizer.step()
                train_error += selected.detach().double().abs().sum().item()
                train_count += selected.numel()
            if not train_count:
                raise ValueError("Every training batch has missing targets; no parameter updates were possible")
            val_mae, val_count = validation_mae(model, val_loader, bundle.mean_mph, bundle.std_mph, device, metric,
                                               config["max_val_batches"])
            total_skipped += skipped
            row = {"epoch": epoch, "train_n": train_count, "train_mae_mph": train_error / train_count,
                   "val_n": val_count, "val_mae_mph": val_mae, "skipped_all_missing_batches": skipped,
                   "elapsed_seconds": time.perf_counter() - start}
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if val_mae < best_mae:
                best_mae, best_epoch, stale = val_mae, epoch, 0
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "config": config, "data_id": bundle.data_id,
                            "sensor_ids": [str(value) for value in bundle.sensor_ids],
                            "epoch": epoch, "val_mae_mph": val_mae}, run_dir / "best.pt")
            else:
                stale += 1
            if stale >= config["patience"]:
                break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    restored_mae, _ = validation_mae(model, val_loader, bundle.mean_mph, bundle.std_mph, device, metric,
                                    config["max_val_batches"])
    if not np.isclose(restored_mae, best_mae, rtol=1e-7, atol=1e-7):
        raise RuntimeError("Restored checkpoint does not reproduce selected validation MAE")
    metadata.update({"status": "complete", "best_epoch": best_epoch, "val_mae_mph": best_mae,
                     "epochs_completed": epoch, "training_seconds": time.perf_counter() - start,
                     "skipped_all_missing_batches": total_skipped,
                     "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
                     "history_sha256": sha256_file(run_dir / "history.csv")})
    write_json(run_dir / "run.json", metadata)
    print(json.dumps({"run_dir": str(run_dir), "best_epoch": best_epoch, "val_mae_mph": best_mae,
                      "parameters": metadata["parameter_count"], "training_seconds": metadata["training_seconds"]}), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int, help="Smoke runs only: cap training batches per epoch")
    parser.add_argument("--max-val-batches", type=int, help="Smoke runs only: use a validation prefix, not the full selection metric")
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
