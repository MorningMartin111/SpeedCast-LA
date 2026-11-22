"""Numerical model checks, independent of B's evaluation implementation."""

import io
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from traffic.models.stgcn.model import STGCN, chebyshev_supports
from traffic.models.stgcn.train import train, validation_mae


def configuration(model="stgcn"):
    return {"model": model, "hidden_channels": 8, "blocks": 2,
            "temporal_kernel": 3, "cheb_order": 3, "dropout": 0.0}


class STGCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.adjacency = np.array([[0, 1, 0], [0, 0, 0.5], [0, 0, 0]], dtype=np.float32)
        self.input = torch.randn(2, 12, 3, 3)

    def test_graph_scaling_direction_symmetrization_and_isolates(self):
        adjacency = np.array([[5, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        original = adjacency.copy()
        support = chebyshev_supports(adjacency, 3)
        expected = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        torch.testing.assert_close(support[0], torch.eye(3))
        torch.testing.assert_close(support[1], expected)
        torch.testing.assert_close(support[2], 2 * expected @ expected - torch.eye(3))
        np.testing.assert_array_equal(adjacency, original)
        self.assertTrue(torch.isfinite(support).all())
        torch.testing.assert_close(chebyshev_supports(adjacency.T, 3), support)

    def test_forward_backward_updates_parameters_for_both_models(self):
        for name in ("stgcn", "stgcn_temporal"):
            with self.subTest(model=name):
                model = STGCN(self.adjacency, configuration(name))
                self.assertEqual(model.temporal_lengths, [12, 10, 8, 6, 4])
                previous = model.head.weight.detach().clone()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                prediction = model(self.input)
                self.assertEqual(tuple(prediction.shape), (2, 12, 3))
                self.assertTrue(torch.isfinite(prediction).all())
                prediction.square().mean().backward()
                optimizer.step()
                self.assertFalse(torch.equal(previous, model.head.weight))

    def test_tiny_sample_loss_decreases(self):
        model = STGCN(self.adjacency, configuration())
        target = torch.linspace(-0.5, 0.5, 12)[None, :, None].expand(2, 12, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = (model(self.input) - target).abs().mean().item()
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(self.input) - target).abs().mean()
            loss.backward()
            optimizer.step()
        final = (model(self.input) - target).abs().mean().item()
        self.assertLess(final, 0.3 * initial, (initial, final))

    def test_checkpoint_reload_reproduces_eval(self):
        config = {**configuration(), "dropout": 0.2}
        model = STGCN(self.adjacency, config).eval()
        with torch.no_grad():
            expected = model(self.input)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = STGCN(self.adjacency, config).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            actual = restored(self.input)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_temporal_model_has_no_cross_node_dependency(self):
        changed = self.input.clone()
        changed[:, :, 0, :] += 7
        observed = {}
        for name in ("stgcn", "stgcn_temporal"):
            torch.manual_seed(42)
            model = STGCN(self.adjacency, configuration(name)).eval()
            with torch.no_grad():
                before, after = model(self.input), model(changed)
            observed[name] = (before[:, :, 1] - after[:, :, 1]).abs().max().item()
        self.assertLessEqual(observed["stgcn_temporal"], 1e-7)
        self.assertGreater(observed["stgcn"], 1e-5)

    def test_invalid_graph_and_temporal_extent_rejected(self):
        with self.assertRaisesRegex(ValueError, "exhaust"):
            STGCN(self.adjacency, {**configuration(), "blocks": 3})
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            chebyshev_supports(np.array([[0, -1], [0, 0]]), 3)
        model = STGCN(self.adjacency, configuration())
        with self.assertRaisesRegex(ValueError, "Expected"):
            model(torch.zeros(1, 13, 3, 3))

    def test_validation_mae_weights_valid_scalars_not_batches(self):
        class ZeroModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 12, 3)
        mask = torch.zeros(1, 12, 3, dtype=torch.bool)
        mask[0, 0, 0] = True
        batches = [{"x": self.input[:1], "y_mph": torch.full((1, 12, 3), 4.0), "y_mask": mask},
                   {"x": self.input[:1], "y_mph": torch.ones(1, 12, 3), "y_mask": torch.ones(1, 12, 3, dtype=torch.bool)}]
        mae, count = validation_mae(ZeroModel(), batches, 0.0, 1.0, torch.device("cpu"))
        self.assertEqual(count, 37)
        self.assertAlmostEqual(mae, 40 / 37)

    def test_training_skips_all_missing_batch_and_records_smoke_controls(self):
        valid = {"x": self.input[0], "y_mph": torch.ones(12, 3), "y_mask": torch.ones(12, 3, dtype=torch.bool)}
        missing = {**valid, "y_mask": torch.zeros(12, 3, dtype=torch.bool)}
        bundle = SimpleNamespace(adj_dir=self.adjacency, data_id="synthetic-test", sensor_ids=["a", "b", "c"],
                                 mean_mph=0.0, std_mph=1.0, manifest={"protocol_version": "1.0"},
                                 dataset=lambda split: [missing, valid] if split == "train" else [valid])
        config = {**configuration(), "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 1,
                  "max_epochs": 1, "patience": 10, "num_workers": 0, "num_threads": 2}
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args = argparse.Namespace(config=str(config_path), data="unused", seed=42, device="cpu",
                                      run_dir=str(Path(directory) / "run"), max_epochs=1,
                                      max_train_batches=None, max_val_batches=1)
            with patch("traffic.models.stgcn.train.load_bundle", return_value=bundle), \
                 patch("traffic.models.stgcn.train.common_metric", return_value=None), \
                 patch("traffic.models.stgcn.train.git_metadata", return_value={"code_commit": None, "code_dirty": None}):
                result = train(args)
            self.assertEqual(result["skipped_all_missing_batches"], 1)
            self.assertTrue(result["smoke_only"])
            self.assertEqual(result["configuration"]["max_val_batches"], 1)
            self.assertEqual(result["status"], "complete")
            self.assertTrue((Path(args.run_dir) / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
