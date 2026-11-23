"""Prediction exchange and scoring contract tests on the shared fixture."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from traffic.data.dataset import load_bundle
from traffic.data.fixture import generate_fixture
from traffic.eval.io import read_predictions, write_predictions
from traffic.eval.run import score_run
from traffic.models.persistence import export_persistence


class EvaluationWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data = Path(directory.name) / "fixture"
        self.run_dir = Path(directory.name) / "persistence"
        generate_fixture(self.data)
        self.bundle = load_bundle(self.data)

    def test_persistence_exports_both_splits_and_scores_complete_test_split(self):
        for split in ("val", "test"):
            self.assertEqual(export_persistence(self.data, self.run_dir, split), self.run_dir / f"{split}.npz")
        predictions = read_predictions(self.run_dir, "test", self.bundle)
        np.testing.assert_array_equal(predictions[1], np.tile([64, 60, 50], (12, 1)))

        output = score_run(self.data, self.run_dir, "test")
        self.assertEqual(output.name, "metrics_test.csv")
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["horizon_step"] for row in rows], ["3", "6", "12", "all12"])
        self.assertEqual([int(row["n"]) for row in rows], [38, 39, 39, 463])
        self.assertTrue(all(row["model"] == "persistence" and row["seed"] == "" for row in rows))
        self.assertAlmostEqual(float(rows[0]["mae_mph"]), 159.75 / 38)

    def test_writer_rejects_reordered_origins_and_nonfinite_predictions(self):
        origins = self.bundle.origins["test"]
        predictions = np.zeros((len(origins), 12, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "origin_idx"):
            write_predictions(self.run_dir, "test", self.bundle, origins[::-1], predictions)
        predictions[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            write_predictions(self.run_dir, "test", self.bundle, origins, predictions)
        self.assertFalse((self.run_dir / "test.npz").exists())

    def test_scorer_rejects_incompatible_run_metadata(self):
        export_persistence(self.data, self.run_dir, "val")
        metadata_path = self.run_dir / "run.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for field, bad_value in (("data_id", "wrong-bundle"), ("units", "km/h"), ("protocol_version", "0")):
            with self.subTest(field=field):
                metadata_path.write_text(json.dumps({**metadata, field: bad_value}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, field):
                    score_run(self.data, self.run_dir, "val")
        self.assertFalse((self.run_dir / "metrics_val.csv").exists())

    def test_reader_rejects_mismatched_axes(self):
        path = export_persistence(self.data, self.run_dir, "test")
        with np.load(path, allow_pickle=False) as archive:
            fields = {field: archive[field] for field in archive.files}
        for field in ("origin_idx", "origin_time", "sensor_id"):
            with self.subTest(field=field):
                np.savez(path, **{**fields, field: fields[field][::-1]})
                with self.assertRaisesRegex(ValueError, field):
                    read_predictions(self.run_dir, "test", self.bundle)
        np.savez(path, **fields)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            read_predictions(self.run_dir, "test", self.bundle, expected_sha256="0" * 64)
