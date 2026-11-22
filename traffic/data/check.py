"""Inspect a bundle and optionally verify the protocol's exact hand-worked fixture."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from torch.utils.data import DataLoader

from .dataset import load_bundle


def check_fixture(bundle) -> dict:
    if bundle.sensor_ids != ["S02", "S01", "S03"] or bundle.manifest["origin_counts"] != {"train": 61, "val": 1, "test": 13}:
        raise ValueError("Fixture station order or split counts differ from protocol")
    if not bundle.timestamps[96].endswith("08:00:00") or not bundle.timestamps[99].endswith("08:15:00"):
        raise ValueError("Fixture origin/target clock alignment is incorrect")
    sample = bundle.dataset("test")[int(np.flatnonzero(bundle.origins["test"] == 96)[0])]
    np.testing.assert_array_equal(sample["last_mph"].numpy(), [64, 60, 50])
    np.testing.assert_array_equal(sample["y_mph"][2].numpy(), [64.75, 0, 20])
    np.testing.assert_array_equal(sample["y_mask"][2].numpy(), [True, False, True])
    np.testing.assert_array_equal(sample["origin_observed"].numpy(), [True, False, True])
    errors = (sample["last_mph"].numpy() - sample["y_mph"][2].numpy())[sample["y_mask"][2].numpy()].astype(np.float64)
    scores = {"n": int(errors.size), "mae_mph": float(np.abs(errors).mean()), "rmse_mph": float(np.sqrt(np.mean(errors ** 2))), "bias_mph": float(errors.mean())}
    np.testing.assert_allclose([scores["mae_mph"], scores["rmse_mph"], scores["bias_mph"]], [15.375, math.sqrt(450.28125), 14.625], rtol=0, atol=1e-12)
    if bundle.ref_speed_mph[2] != 50:
        raise ValueError("Fixture S03 training reference speed must be 50 mph")
    return {"origin_idx": 96, "origin_time": bundle.timestamps[96], "target_time_step3": bundle.timestamps[99],
            "persistence_mph": sample["last_mph"].tolist(), "true_mph_step3": [64.75, None, 20], **scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--fixture", action="store_true", help="Assert exact synthetic fixture values")
    args = parser.parse_args()
    bundle = load_bundle(args.data)
    result = {"data_id": bundle.data_id, "T": len(bundle.timestamps), "N": len(bundle.sensor_ids),
              "sensor_ids_first3": bundle.sensor_ids[:3], "mean_mph": bundle.mean_mph, "std_mph": bundle.std_mph,
              "origin_counts": {name: len(values) for name, values in bundle.origins.items()},
              "windows": bundle.manifest["windows"], "quality": bundle.manifest["quality"]}
    if len(bundle.dataset("train")):
        batch = next(iter(DataLoader(bundle.dataset("train"), batch_size=2, shuffle=False)))
        result["first_batch_shapes"] = {name: list(value.shape) for name, value in batch.items()}
    if args.fixture:
        result["hand_check"] = check_fixture(bundle)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
