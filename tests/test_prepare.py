"""A1 source alignment and real preprocessing checks (synthetic inputs)."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

try:
    import pandas as pd
    import tables
except ImportError:
    pd = None

from traffic.data.dataset import load_bundle
from traffic.data.fixture import fixture_inputs
from traffic.data.prepare import prepare, read_graph


@unittest.skipIf(pd is None, "Install requirements-data.txt for HDF5 preparation tests")
class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.inputs = fixture_inputs()
        self.ids = self.inputs["sensor_ids"]
        graph = [[value.encode() for value in self.ids],
                 {value.encode(): index for index, value in enumerate(self.ids)},
                 self.inputs["adj_dir"]]
        with (self.raw / "adj_mx.pkl").open("wb") as stream:
            pickle.dump(graph, stream)
        coords = pd.DataFrame({"sensor_id": self.ids,
                               "latitude": self.inputs["coordinates"][:, 0],
                               "longitude": self.inputs["coordinates"][:, 1]})
        coords.iloc[[2, 0, 1]].to_csv(self.raw / "graph_sensor_locations.csv", index=False)
        self.frame = pd.DataFrame(self.inputs["raw_speed"],
                                  index=pd.to_datetime(self.inputs["timestamps"]), columns=self.ids)

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self, frame, name):
        frame.to_hdf(self.raw / "metr-la.h5", key="df", mode="w")
        with redirect_stdout(StringIO()):
            audit = prepare(self.raw, self.root / name)
        return load_bundle(self.root / name), audit

    def test_permuted_columns_and_byte_graph_ids_preserve_axis_and_data_id(self):
        original, audit = self.run_prepare(self.frame, "original")
        shuffled, _ = self.run_prepare(self.frame.iloc[:, [2, 0, 1]], "shuffled")
        self.assertEqual(original.sensor_ids, ["S02", "S01", "S03"])
        self.assertEqual(original.data_id, shuffled.data_id)
        np.testing.assert_array_equal(original.speed_mph, shuffled.speed_mph)
        np.testing.assert_array_equal(original.adj_dir, np.array([[0, 1, 0], [0, 0, .5], [0, 0, 0]]))
        self.assertEqual(audit["interval_counts_seconds"], {"300.0": 119})
        self.assertEqual([audit["splits"][key]["count"] for key in ("train", "val", "test")], [61, 1, 13])

    def test_future_mutation_leaves_training_statistics_and_past_inputs_equal(self):
        original, _ = self.run_prepare(self.frame, "original")
        changed = self.frame.copy()
        changed.iloc[110:, :] = 1000
        future, _ = self.run_prepare(changed, "future")
        self.assertNotEqual(original.data_id, future.data_id)
        self.assertEqual(original.mean_mph, future.mean_mph)
        self.assertEqual(original.std_mph, future.std_mph)
        np.testing.assert_array_equal(original.ref_speed_mph, future.ref_speed_mph)
        np.testing.assert_array_equal(original.filled_mph[:110], future.filled_mph[:110])
        for index in range(len(original.dataset("train"))):
            for key in ("x", "y_mph", "y_mask"):
                np.testing.assert_array_equal(original.dataset("train")[index][key], future.dataset("train")[index][key])

    def test_legacy_byte_metadata_fallback_preserves_numeric_source(self):
        self.frame.to_hdf(self.raw / "metr-la.h5", key="df", mode="w")
        with tables.open_file(self.raw / "metr-la.h5", mode="a") as handle:
            handle.root.df._v_attrs.pandas_type = np.bytes_(b"frame")
        with redirect_stdout(StringIO()):
            audit = prepare(self.raw, self.root / "legacy")
        self.assertIn("legacy-fixed", audit["hdf_reader"])
        bundle = load_bundle(self.root / "legacy")
        np.testing.assert_array_equal(bundle.speed_mph, self.inputs["raw_speed"])

    def test_missing_coordinate_retains_sensor_and_duplicate_time_rejected(self):
        coords = pd.read_csv(self.raw / "graph_sensor_locations.csv")
        coords.iloc[:2].to_csv(self.raw / "graph_sensor_locations.csv", index=False)
        bundle, audit = self.run_prepare(self.frame, "missing-coordinate")
        self.assertEqual(len(bundle.sensor_ids), 3)
        self.assertEqual(len(audit["missing_coordinates"]), 1)
        duplicate = self.frame.copy()
        duplicate.index = list(duplicate.index[:-1]) + [duplicate.index[-2]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_prepare(duplicate, "bad")

    def test_graph_map_mismatch_rejected(self):
        with (self.raw / "adj_mx.pkl").open("rb") as stream:
            graph = pickle.load(stream)
        graph[1][b"S02"] = 2
        with (self.raw / "adj_mx.pkl").open("wb") as stream:
            pickle.dump(graph, stream)
        with self.assertRaisesRegex(ValueError, "mapping"):
            read_graph(self.raw / "adj_mx.pkl")


if __name__ == "__main__":
    unittest.main()
