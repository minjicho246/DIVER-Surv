"""Dependency-free tests for the public command-line interface."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from survival_diffusion.cli import build_parser, configure_environment
from survival_diffusion.settings import RuntimeConfig, load_runtime_config


class CliAndConfigTest(unittest.TestCase):
    def test_default_configuration(self):
        args = build_parser().parse_args([])
        expected = {
            "lr1": 1e-5,
            "lr2": 3e-6,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "epochs": 100,
            "dropout_rate": 0.2,
            "lambda_diff": 0.2,
            "consi_loss": 0.5,
            "survival_t_max": 100,
            "diffusion_steps": 1000,
            "diffusion_beta_start": 1e-4,
            "diffusion_beta_end": 0.02,
            "diffusion_embed_dim": 256,
            "survival_hidden_dim": 256,
            "fusion_num_layers": 2,
            "fusion_num_heads": 8,
        }
        for name, value in expected.items():
            self.assertEqual(getattr(args, name), value, name)
        for removed in [
            "rank_loss_weight",
            "rank_margin",
            "hard_pair_fraction",
            "diffusion_min_snr_gamma",
            "survival_t_mode",
        ]:
            self.assertFalse(hasattr(args, removed), removed)

    def test_partial_config_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"output_root": "results", "wandb_entity": None}))
            config = load_runtime_config(path)
        self.assertEqual(config.output_root, "results")
        self.assertIsNone(config.wandb_entity)
        self.assertEqual(config.ucsf_csv_path, RuntimeConfig().ucsf_csv_path)

    def test_invalid_config_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"unknown": "value"}))
            with self.assertRaises(ValueError):
                load_runtime_config(path)

    def test_cuda_visibility(self):
        parser = build_parser()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            configure_environment(parser.parse_args([]))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
            configure_environment(parser.parse_args(["--cuda_device", "1"]))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")

    def test_public_tree_contains_no_private_defaults(self):
        files = [
            *ROOT.joinpath("survival_diffusion").rglob("*.py"),
            *ROOT.joinpath("docs").rglob("*.md"),
            ROOT / "README.md",
            ROOT / "configs/example.json",
        ]
        content = "\n".join(path.read_text(errors="ignore") for path in files)
        self.assertNotIn("/workspace/", content)
        self.assertNotIn("mjcho246", content)
        self.assertFalse((ROOT / "README.ko.md").exists())
        self.assertFalse((ROOT / "survival_diffusion/data/masking.py").exists())

    def test_help_does_not_import_ml_stack(self):
        code = (
            "import sys; import survival_diffusion.cli; "
            "assert not any(x in sys.modules for x in "
            "('torch', 'wandb', 'monai', 'sentence_transformers'))"
        )
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("--weight_decay", result.stdout)
        self.assertIn("--fusion_num_heads", result.stdout)
        self.assertNotIn("--rank_loss_weight", result.stdout)
        self.assertNotIn("--diffusion_min_snr_gamma", result.stdout)


if __name__ == "__main__":
    unittest.main()
