"""
Generate a post-impact image and numerical predictions from impact parameters.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torchvision.utils import save_image

from p2i.model import load_generator_checkpoint
from p2i.paths import default_generator_path, repo_root, results_dir
from p2i.preprocessing import ImpactTabularScaler, default_scaler_path_for_model

REPO_ROOT = repo_root(Path(__file__))
NUM_PARAMS = 4
NUM_NUMERICAL_OUTPUTS = 3
LATENT_DIM = 100
IMG_CHANNELS = 4


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate():
    parser = argparse.ArgumentParser(
        description="Generate an image and numerical data using a trained Param2Image cGAN."
    )
    parser.add_argument(
        "--model_path",
        default=str(default_generator_path(Path(__file__))),
        help="Path to the trained generator model.",
    )
    parser.add_argument("--velocity", type=float, required=True, help="Projectile velocity (m/s).")
    parser.add_argument("--radius", type=float, required=True, help="Projectile radius (mm).")
    parser.add_argument("--impact_x", type=float, required=True, help="Impact x-coordinate (mm).")
    parser.add_argument("--impact_y", type=float, required=True, help="Impact y-coordinate (mm).")
    parser.add_argument(
        "--scaler_path",
        default=None,
        help="Optional path to normalization_stats.json. Defaults to the model checkpoint folder.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(results_dir(Path(__file__)) / "predictions"),
        help="Directory for prediction run folders.",
    )
    args = parser.parse_args()

    main_output_dir = Path(args.output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = main_output_dir / f"sim_pred_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    image_path = run_dir / "generated_image.png"
    csv_path = run_dir / "predicted_outputs.csv"

    print(f"--- Output will be saved to: {run_dir} ---")

    device = get_device()
    print(f"--- Using device: {device} ---")

    print(f"--- Loading generator model from '{args.model_path}' ---")
    generator, head_mode = load_generator_checkpoint(
        args.model_path,
        device,
        num_params=NUM_PARAMS,
        latent_dim=LATENT_DIM,
        num_numerical_outputs=NUM_NUMERICAL_OUTPUTS,
        img_channels=IMG_CHANNELS,
        numerical_head_mode="auto",
    )
    print(f"--- Numerical head mode: {head_mode} ---")

    scaler_path = Path(args.scaler_path) if args.scaler_path else default_scaler_path_for_model(args.model_path)
    scaler = ImpactTabularScaler.load(scaler_path) if scaler_path.exists() else None
    if scaler is not None:
        print(f"--- Loaded tabular scaler from '{scaler_path}' ---")
    else:
        print("--- No tabular scaler found; using raw input/output scale ---")

    raw_params = [[args.velocity, args.radius, args.impact_x, args.impact_y]]
    model_params = scaler.transform_params_array(raw_params) if scaler is not None else raw_params
    input_params = torch.tensor(model_params, dtype=torch.float32).to(device)
    z = torch.randn(1, LATENT_DIM, device=device)

    with torch.no_grad():
        print("--- Generating multi-modal output ---")
        generated_image, predicted_numerical = generator(z, input_params)

    predicted_values = predicted_numerical.cpu().numpy()
    if scaler is not None:
        predicted_values = scaler.inverse_transform_outputs_array(predicted_values)
    predicted_values = predicted_values.flatten()

    print("\n--- Predicted Numerical Outputs ---")
    print(f"  Max Mises Stress (MPa): {predicted_values[0]:.2f}")
    print(f"  Max PEEQ:               {predicted_values[1]:.4f}")
    print(f"  Max Deflection (mm):    {predicted_values[2]:.2f}")
    print("---------------------------------")

    print(f"\n--- Saving generated image to '{image_path}' ---")
    save_image(generated_image, str(image_path), normalize=True)

    print(f"--- Saving predicted outputs to '{csv_path}' ---")
    df = pd.DataFrame(
        {
            "input_velocity_ms": [args.velocity],
            "input_proj_radius_mm": [args.radius],
            "input_impact_x_mm": [args.impact_x],
            "input_impact_y_mm": [args.impact_y],
            "pred_max_mises_mpa": [predicted_values[0]],
            "pred_max_peeq": [predicted_values[1]],
            "pred_max_deflection_mm": [predicted_values[2]],
        }
    )
    df.to_csv(csv_path, index=False)
    print("\n--- Generation Complete. ---")


if __name__ == "__main__":
    generate()
