"""Regression checks for the shared evaluation metrics."""

import math
import tempfile
import unittest
from pathlib import Path

from traffic.data.dataset import load_bundle
from traffic.data.fixture import generate_fixture
from traffic.eval.metrics import masked_scores


class MaskedScoresTests(unittest.TestCase):
    def test_persistence_fixture_origin_96_step_3(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "fixture"
            generate_fixture(fixture_path)
            bundle = load_bundle(fixture_path)
            sample = next(row for row in bundle.dataset("test") if row["origin_idx"].item() == 96)
            scores = masked_scores(sample["y_mph"][2], sample["last_mph"], sample["y_mask"][2])

        self.assertEqual(scores["n"], 2)
        self.assertEqual(scores["mae_mph"], 15.375)
        self.assertAlmostEqual(scores["rmse_mph"], math.sqrt(450.28125))
        self.assertEqual(scores["bias_mph"], 14.625)

    def test_all_missing_returns_empty_metrics(self):
        scores = masked_scores([float("nan")], [60.0], [False])

        self.assertEqual(scores, {"n": 0, "mae_mph": None, "rmse_mph": None, "bias_mph": None})


if __name__ == "__main__":
    unittest.main()
