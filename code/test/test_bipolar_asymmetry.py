import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from CPVM.code.bipolar_asymmetry import (
    analyze_bipolar_asymmetry,
    bootstrap_transformer_layers,
    build_result_tables,
)


CONFIG = {
    "model_type": "synthetic",
    "hidden_size": 2,
    "num_hidden_layers": 2,
    "vocab_size": 2,
    "tie_word_embeddings": True,
}


def _write_checkpoint(
    path: Path,
    tensors,
    *,
    duplicate_lm_head=False,
    lm_head_offset=0.0,
):
    path.mkdir()
    weights = {name: value.clone() for name, value in tensors.items()}
    if duplicate_lm_head:
        weights["lm_head.weight"] = (
            weights["model.embed_tokens.weight"].clone() + lm_head_offset
        )
    save_file(weights, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")


def _synthetic_weights(low_direction: float):
    base = {
        "model.embed_tokens.weight": torch.tensor([[0.2, -0.1], [0.3, 0.4]]),
        "model.layers.0.self_attn.q_proj.weight": torch.tensor(
            [[0.1, 0.2], [-0.3, 0.4]]
        ),
        "model.layers.1.mlp.up_proj.weight": torch.tensor(
            [[-0.2, 0.5], [0.7, -0.4]]
        ),
        "model.norm.weight": torch.tensor([1.0, 1.0]),
    }
    update = {
        "model.embed_tokens.weight": torch.tensor([[0.1, -0.2], [0.3, 0.1]]),
        "model.layers.0.self_attn.q_proj.weight": torch.tensor(
            [[0.2, 0.1], [-0.1, 0.4]]
        ),
        "model.layers.1.mlp.up_proj.weight": torch.tensor(
            [[-0.3, 0.2], [0.1, -0.2]]
        ),
        # Exercise per-tensor zero-update handling in build_result_tables.
        "model.norm.weight": torch.zeros(2),
    }
    high = {name: base[name] + update[name] for name in base}
    low = {name: base[name] + low_direction * update[name] for name in base}
    return base, high, low


class BipolarAsymmetryTest(unittest.TestCase):
    def _run(self, low_direction):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            base, high, low = _synthetic_weights(low_direction)
            _write_checkpoint(root / "base", base)
            _write_checkpoint(root / "high", high, duplicate_lm_head=True)
            _write_checkpoint(root / "low", low, duplicate_lm_head=True)
            return analyze_bipolar_asymmetry(
                root / "base",
                root / "high",
                root / "low",
                chunk_numel=3,
                show_progress=False,
            )

    def test_exact_opposite_direction_with_unequal_magnitude(self):
        result = self._run(low_direction=-2.0)
        metrics = result["global_metrics"]

        self.assertAlmostEqual(metrics["cosine"], -1.0, places=6)
        self.assertLess(metrics["anti_angle_deg"], 0.02)
        self.assertAlmostEqual(metrics["optimal_anti_scale"], 0.5, places=6)
        self.assertLess(metrics["optimal_anti_residual"], 0.001)
        self.assertGreater(metrics["midpoint_error"], 0.0)
        self.assertEqual(result["ignored_keys"], ["lm_head.weight"])

        _, layer, _, tensor = build_result_tables(result)
        self.assertEqual(
            set(layer["group"]), {"embedding", "layer_00", "layer_01", "final_norm"}
        )
        self.assertTrue(
            tensor.loc[tensor["parameter"] == "model.norm.weight", "cosine"]
            .isna()
            .all()
        )
        ci = bootstrap_transformer_layers(
            result["layer_raw"], n_bootstrap=100, seed=7
        )
        angle = ci.set_index("metric").loc["anti_angle_deg"]
        self.assertAlmostEqual(
            angle["global_estimate"], metrics["anti_angle_deg"], places=10
        )
        self.assertLess(angle["global_estimate"], 0.02)

    def test_same_direction_is_detected_as_asymmetric(self):
        result = self._run(low_direction=1.0)
        metrics = result["global_metrics"]

        self.assertAlmostEqual(metrics["cosine"], 1.0, places=6)
        self.assertAlmostEqual(metrics["anti_angle_deg"], 180.0, places=5)
        self.assertAlmostEqual(metrics["optimal_anti_residual"], 1.0, places=6)
        self.assertAlmostEqual(metrics["opposite_sign_rate"], 0.0, places=6)
        self.assertAlmostEqual(metrics["positive_dot_mass_ratio"], 1.0, places=6)

    def test_inconsistent_tied_lm_head_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            base, high, low = _synthetic_weights(-1.0)
            _write_checkpoint(root / "base", base)
            _write_checkpoint(
                root / "high",
                high,
                duplicate_lm_head=True,
                lm_head_offset=1.0,
            )
            _write_checkpoint(root / "low", low, duplicate_lm_head=True)

            with self.assertRaisesRegex(ValueError, "lm_head.weight differs"):
                analyze_bipolar_asymmetry(
                    root / "base",
                    root / "high",
                    root / "low",
                    chunk_numel=3,
                    show_progress=False,
                )


if __name__ == "__main__":
    unittest.main()
