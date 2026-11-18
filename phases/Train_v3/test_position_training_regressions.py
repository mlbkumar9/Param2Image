from __future__ import annotations

import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def _synthetic_impact(cx: int, cy: int, size: int = 64) -> torch.Tensor:
    image = torch.full((1, 4, size, size), -1.0, dtype=torch.float32)
    image[:, 2, :, :] = 1.0  # blue undamaged plate
    image[:, 3, :, :] = 1.0
    image[:, :3, cy - 3:cy + 4, cx - 3:cx + 4] = 1.0  # bright impact zone
    return image


class PositionSupervisionTests(unittest.TestCase):
    def _module(self):
        return importlib.import_module("position_supervision")

    def test_position_loss_is_zero_for_aligned_impacts_and_larger_when_shifted(self):
        pos = self._module()
        real = _synthetic_impact(18, 24)
        aligned = real.clone()
        shifted = _synthetic_impact(48, 43)

        aligned_loss = pos.impact_location_loss(aligned, real, downsample_size=64)
        shifted_loss = pos.impact_location_loss(shifted, real, downsample_size=64)

        self.assertLess(float(aligned_loss), 1e-7)
        self.assertGreater(float(shifted_loss), float(aligned_loss) + 0.01)

    def test_position_loss_backpropagates_to_generated_image(self):
        pos = self._module()
        real = _synthetic_impact(16, 18)
        fake = _synthetic_impact(45, 44).requires_grad_(True)

        loss = pos.impact_location_loss(fake, real, downsample_size=64)
        loss.backward()

        self.assertIsNotNone(fake.grad)
        self.assertGreater(float(fake.grad.abs().sum()), 0.0)

    def test_pixel_error_reports_shift_magnitude(self):
        pos = self._module()
        real = _synthetic_impact(16, 18)
        shifted = _synthetic_impact(46, 18)

        error_px = pos.impact_position_error_pixels(shifted, real, downsample_size=64)

        self.assertAlmostEqual(float(error_px), 30.0, delta=1.0)


class PipelineWiringTests(unittest.TestCase):
    def test_local_training_defaults_to_train_v4_dataset(self):
        for name in ("train_p2i_pytorch.py", "train_p2i_xla.py"):
            source = (SCRIPT_DIR / name).read_text(encoding="utf-8")
            self.assertIn("DEFAULT_DATA_DIR = data_dir()", source, name)
            self.assertIn("from p2i_paths import", source, name)

    def test_evaluation_passes_noise_before_conditioning_params(self):
        for name in ("eval_p2i_colab.py",):
            source = (SCRIPT_DIR / name).read_text(encoding="utf-8")
            self.assertNotIn("G(params_n, z)", source, name)
            self.assertNotIn("G(p_n, z)", source, name)
            self.assertRegex(source, r"G\(z,\s*(params_n|p_n)\)", name)

    def test_colab_notebooks_enable_position_training_in_fresh_checkpoints(self):
        for name in ("notebook_train_p2i_colab.ipynb", "notebook_train_p2i_colab_tpu.ipynb"):
            notebook = json.loads((SCRIPT_DIR / name).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list)
                else cell.get("source", "")
                for cell in notebook["cells"]
            )
            self.assertIn("P2I_LAMBDA_POS", source, name)
            self.assertIn("training_checkpoints_position", source, name)

    def test_position_training_defaults_do_not_resume_legacy_checkpoints(self):
        expected_defaults = {
            "train_p2i_pytorch.py": "training_checkpoints_position",
            "train_p2i_xla.py": "training_checkpoints_position_xla",
            "generate_p2i_pytorch.py": "training_checkpoints_position",
        }
        for name, folder in expected_defaults.items():
            source = (SCRIPT_DIR / name).read_text(encoding="utf-8")
            self.assertIn(folder, source, name)

    def test_generation_prefers_position_selected_checkpoint(self):
        source = (SCRIPT_DIR / "generate_p2i_pytorch.py").read_text(encoding="utf-8")
        self.assertIn('DEFAULT_CKPT = CHECKPOINT_DIR / "ckpt_best_position.pt"', source)

    def test_training_scripts_persist_resume_artifacts_before_epoch_end(self):
        expected = {
            "train_p2i_pytorch.py": "save_checkpoint(LATEST_CKPT, state)",
            "train_p2i_xla.py": "save_checkpoint_xla(LATEST_CKPT, state)",
        }
        for name, save_call in expected.items():
            source = (SCRIPT_DIR / name).read_text(encoding="utf-8")
            self.assertIn("SAVE_EVERY_STEPS = _env_int(\"P2I_SAVE_EVERY_STEPS\"", source, name)
            self.assertIn("TRAIN_LOG_PATH = _env_str(\"P2I_TRAIN_LOG_PATH\"", source, name)
            self.assertIn("install_output_tee()", source, name)
            self.assertIn("mirror_to_drive(SCALER_PATH)", source, name)
            self.assertIn("mirror_to_drive(CONFIG_PATH)", source, name)
            self.assertIn("def make_checkpoint_state(epoch):", source, name)
            self.assertIn('"epoch_complete": False', source, name)
            self.assertIn('ck.get("epoch_complete", True)', source, name)
            self.assertIn('state["epoch_complete"] = True', source, name)
            self.assertIn("RESUME_CKPT = _env_str(\"P2I_RESUME_CKPT\"", source, name)
            self.assertIn("def resolve_resume_checkpoint():", source, name)
            self.assertIn("checkpoint_has_finite_weights", source, name)
            self.assertIn("next displayed epoch is {start_epoch + 1}", source, name)
            self.assertIn("if SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0:", source, name)
            self.assertIn(save_call, source, name)

    def test_colab_notebooks_write_logs_and_periodic_checkpoints_to_drive(self):
        expected_log_dirs = {
            "notebook_train_p2i_colab.ipynb": "logs_position",
            "notebook_train_p2i_colab_tpu.ipynb": "logs_position_xla",
        }
        for name, drive_log_dir in expected_log_dirs.items():
            notebook = json.loads((SCRIPT_DIR / name).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list)
                else cell.get("source", "")
                for cell in notebook["cells"]
            )
            self.assertIn("P2I_SAVE_EVERY_STEPS", source, name)
            self.assertIn("P2I_RESUME_CKPT", source, name)
            self.assertIn("os.environ['P2I_RESUME_CKPT']       = 'auto'", source, name)
            self.assertNotIn("os.environ['P2I_RESUME_CKPT']       = 'best_position'", source, name)
            self.assertIn("P2I_MAX_R1_PENALTY", source, name)
            self.assertIn("P2I_TRAIN_LOG_PATH", source, name)
            self.assertIn("DRIVE_LOG_DIR", source, name)
            self.assertIn(drive_log_dir, source, name)
            self.assertIn("os.environ['P2I_LOG_DIR']          = DRIVE_LOG_DIR", source, name)

    def test_colab_train_cells_run_from_local_project_directory(self):
        expected_scripts = {
            "notebook_train_p2i_colab.ipynb": "train_p2i_pytorch.py",
            "notebook_train_p2i_colab_tpu.ipynb": "train_p2i_xla.py",
        }
        for name, script in expected_scripts.items():
            notebook = json.loads((SCRIPT_DIR / name).read_text(encoding="utf-8"))
            train_cells = []
            for cell in notebook["cells"]:
                source = "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else cell.get("source", "")
                if script in source and "python -u" in source:
                    train_cells.append(source)
            self.assertTrue(train_cells, name)
            self.assertIn("os.chdir(LOCAL_TRAIN_PATH)", train_cells[-1], name)

    def test_cuda_training_module_imports_without_optional_tensorboard(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, r'{SCRIPT_DIR}'); import train_p2i_pytorch",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
