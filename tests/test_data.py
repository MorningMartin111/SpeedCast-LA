"""Data invariants guarding leakage, node axes, missing labels and stable identities."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from traffic.data.bundle import build_bundle
from traffic.data.check import check_fixture
from traffic.data.dataset import load_bundle
from traffic.data.fixture import fixture_inputs, generate_fixture


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_fixture_and_torch_batch(self):
        generate_fixture(self.root / "fixture")
        bundle = load_bundle(self.root / "fixture")
        check_fixture(bundle)
        self.assertEqual(bundle.sensor_ids, ["S02", "S01", "S03"])
        np.testing.assert_array_equal(bundle.adj_dir, [[0, 1, 0], [0, 0, .5], [0, 0, 0]])
        np.testing.assert_array_equal(bundle.origins["train"], np.arange(11, 72))
        np.testing.assert_array_equal(bundle.origins["val"], [83])
        np.testing.assert_array_equal(bundle.origins["test"], np.arange(95, 108))
        batch = next(iter(DataLoader(bundle.dataset("test"), batch_size=2)))
        self.assertEqual(tuple(batch["x"].shape), (2, 12, 3, 3))
        self.assertEqual(batch["x"].dtype, torch.float32)
        self.assertEqual(batch["y_mask"].dtype, torch.bool)
        self.assertEqual(batch["origin_idx"].dtype, torch.int64)
        np.testing.assert_allclose(batch["x"][1, -1, :, 0] * bundle.std_mph + bundle.mean_mph, [64, 60, 50], atol=5e-6)
        np.testing.assert_allclose(batch["x"][1, -1, :, 1], 96 / 288)
        np.testing.assert_array_equal(batch["x"][1, :, :, 2], 0)
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            bundle.dataset("validation")

    def test_future_changes_do_not_change_preprocessing_or_earlier_inputs(self):
        first = fixture_inputs()
        changed = fixture_inputs()
        changed["raw_speed"][105:] = [[5, 10, 15]]
        build_bundle(self.root / "first", **first)
        build_bundle(self.root / "changed", **changed)
        a, b = load_bundle(self.root / "first"), load_bundle(self.root / "changed")
        self.assertNotEqual(a.data_id, b.data_id)
        self.assertEqual((a.mean_mph, a.std_mph), (b.mean_mph, b.std_mph))
        np.testing.assert_array_equal(a.ref_speed_mph, b.ref_speed_mph)
        np.testing.assert_array_equal(a.speed_mph[:105], b.speed_mph[:105])
        np.testing.assert_array_equal(a.filled_mph[:105], b.filled_mph[:105])
        for i in range(len(a.dataset("train"))):
            for field in a.dataset("train")[i]:
                torch.testing.assert_close(a.dataset("train")[i][field], b.dataset("train")[i][field], rtol=0, atol=0)
        for i, origin in enumerate(a.origins["test"]):
            if origin < 105:
                torch.testing.assert_close(a.dataset("test")[i]["x"], b.dataset("test")[i]["x"], rtol=0, atol=0)

    def test_missing_labels_forward_fill_and_gap_reset(self):
        values = fixture_inputs()
        values["raw_speed"][[0, 1, 30, 31, 50], 0] = 0
        # Removing five minutes creates a discontinuity between rows 49 and 50.
        values["timestamps"][50:] = [(datetime.fromisoformat(time) + timedelta(minutes=5)).isoformat(sep=" ") for time in values["timestamps"][50:]]
        build_bundle(self.root / "gaps", **values)
        bundle = load_bundle(self.root / "gaps")
        np.testing.assert_allclose(bundle.filled_mph[:2, 0], bundle.mean_mph, atol=5e-6)
        np.testing.assert_array_equal(bundle.filled_mph[30:32, 0], [47.25, 47.25])
        self.assertAlmostEqual(float(bundle.filled_mph[50, 0]), bundle.mean_mph, places=5)
        self.assertFalse(bundle.observed[30, 0])
        self.assertEqual(bundle.speed_mph[30, 0], 0)
        for origins in bundle.origins.values():
            self.assertTrue(all(not (origin - 11 < 50 <= origin + 12) for origin in origins))
        # The validation window can use the training period's already observed history.
        self.assertEqual(int(bundle.origins["val"][0]), 83)
        np.testing.assert_array_equal(bundle.dataset("val")[0]["last_mph"], values["raw_speed"][83])

    def test_identity_survives_copy_recompression_and_provenance_changes(self):
        generate_fixture(self.root / "original")
        expected = load_bundle(self.root / "original").data_id
        shutil.copytree(self.root / "original", self.root / "copy")
        for filename in ("series.npz", "graph.npz", "splits.npz"):
            path = self.root / "copy" / filename
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in reversed(archive.files)}
            np.savez(path, **arrays)
        manifest_path = self.root / "copy" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source"] = {"downloaded_at": "2099-01-01", "local_path": "another/computer"}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(load_bundle(self.root / "copy").data_id, expected)
        path = self.root / "copy" / "series.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["speed_mph"][110, 0] += 1
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            load_bundle(self.root / "copy")

    def test_identity_covers_node_order_graph_coordinates_scaler_and_semantics(self):
        generate_fixture(self.root / "original")
        expected = load_bundle(self.root / "original").data_id
        graph_change = fixture_inputs()
        graph_change["adj_dir"][0, 1] = 0.75
        build_bundle(self.root / "graph", **graph_change)
        self.assertNotEqual(load_bundle(self.root / "graph").data_id, expected)
        coords_change = fixture_inputs()
        coords_change["coordinates"][0, 0] += .1
        build_bundle(self.root / "coords", **coords_change)
        self.assertNotEqual(load_bundle(self.root / "coords").data_id, expected)
        permuted = fixture_inputs()
        permuted["sensor_ids"] = ["S01", "S02", "S03"]
        build_bundle(self.root / "nodes", **permuted)
        self.assertNotEqual(load_bundle(self.root / "nodes").data_id, expected)
        scaler_path = self.root / "original" / "scaler.json"
        scaler = json.loads(scaler_path.read_text())
        scaler["mean_mph"] += .25
        scaler_path.write_text(json.dumps(scaler))
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            load_bundle(self.root / "original")

    def test_missing_training_sensor_reference_and_input_fallback(self):
        values = fixture_inputs()
        values["raw_speed"][:84, 1] = 0
        values["coordinates"][1] = np.nan
        build_bundle(self.root / "missing_sensor", **values)
        bundle = load_bundle(self.root / "missing_sensor")
        self.assertTrue(np.isnan(bundle.ref_speed_mph[1]))
        np.testing.assert_allclose(bundle.filled_mph[:84, 1], bundle.mean_mph, atol=5e-6)
        self.assertEqual(bundle.manifest["quality"]["sensors_without_training_observations"], ["S01"])
        self.assertEqual(bundle.manifest["quality"]["sensors_missing_coordinates"], ["S01"])
        self.assertTrue(bundle.observed[84, 1])

    def test_timezone_offset_transition_uses_elapsed_time_and_local_calendar(self):
        values = fixture_inputs()
        # Spring-forward: local 01:55 -> 03:00 is still five elapsed minutes.
        utc_start = datetime(2024, 3, 10, 6, 0, tzinfo=timezone.utc)
        utc_dates = [utc_start + timedelta(minutes=5 * index) for index in range(120)]
        values["timestamps"] = [date.astimezone(timezone(timedelta(hours=-5 if date.hour < 7 else -4))).isoformat(sep=" ") for date in utc_dates]
        build_bundle(self.root / "dst", **values)
        bundle = load_bundle(self.root / "dst")
        self.assertEqual(bundle.manifest["origin_counts"], {"train": 61, "val": 1, "test": 13})
        self.assertEqual(int(bundle.tod_slot[11]), 23)
        self.assertEqual(int(bundle.tod_slot[12]), 36)
        self.assertEqual(int(bundle.dow[12]), 6)
        self.assertEqual(bundle.manifest["time_semantics"]["gap_clock"], "UTC elapsed")

    def test_bad_raw_inputs_fail_with_actionable_errors(self):
        for label, change, regex in (
            ("no_data", lambda v: v["raw_speed"].fill(0), "no valid observed"),
            ("zero_std", lambda v: v["raw_speed"].fill(50), "standard deviation is zero"),
            ("duplicate_ids", lambda v: v["sensor_ids"].__setitem__(1, "S02"), "Duplicate sensor"),
            ("negative_graph", lambda v: v["adj_dir"].__setitem__((0, 0), -1), "nonnegative"),
            ("bad_coords", lambda v: v["coordinates"].__setitem__((0, 0), 100), "latitude/longitude"),
            ("duplicate_time", lambda v: v["timestamps"].__setitem__(1, v["timestamps"][0]), "unique and strictly increasing"),
        ):
            with self.subTest(label=label):
                values = fixture_inputs()
                change(values)
                with self.assertRaisesRegex(ValueError, regex):
                    build_bundle(self.root / label, **values)

    def test_split_labels_are_disjoint_and_time_index_tampering_rejected(self):
        generate_fixture(self.root / "fixture")
        bundle = load_bundle(self.root / "fixture")
        targets = {name: {int(origin) + step for origin in origins for step in range(1, 13)} for name, origins in bundle.origins.items()}
        self.assertFalse(targets["train"] & targets["val"])
        self.assertFalse(targets["val"] & targets["test"])
        path = self.root / "fixture" / "timestamps.csv"
        text = path.read_text(encoding="utf-8").replace("\n1,", "\n2,", 1)
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contiguous integer row order"):
            load_bundle(self.root / "fixture")


if __name__ == "__main__":
    unittest.main()
