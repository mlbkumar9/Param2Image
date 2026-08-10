"""
Inference for Param2Image v3.1.

Loads the EMA generator + EMA physics head from a single checkpoint and
produces:
  * the post-impact damage image (saved as PNG)
  * the numerical predictions (max von Mises, PEEQ, deflection) with
    heteroscedastic uncertainty in physical units (mean +/- 1 sigma)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model_p2i_pytorch import Generator, PhysicsHead
from preprocessing_p2i import (
    OUTPUT_DISPLAY_NAMES,
    OUTPUT_SHORT_NAMES,
    OUTPUT_UNITS,
    ImpactTabularScaler,
    SCALER_FILENAME,
)


CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
from p2i_paths import checkpoint_dir
CHECKPOINT_DIR = Path(os.environ.get("P2I_CHECKPOINT_DIR", str(checkpoint_dir("training_checkpoints_position"))))
DEFAULT_CKPT = CHECKPOINT_DIR / "ckpt_best_position.pt"
LATEST_CKPT = CHECKPOINT_DIR / "ckpt_latest.pt"
CONFIG_PATH = CHECKPOINT_DIR / "config.json"
SCALER_PATH = CHECKPOINT_DIR / SCALER_FILENAME

OUT_DIR = Path(os.environ.get("P2I_OUT_DIR", str(CURRENT_DIR / "Prediction_OUT")))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--velocity", type=float, required=True, help="Impact velocity [m/s]")
    p.add_argument("--radius", type=float, required=True, help="Projectile radius [mm]")
    p.add_argument("--impact_x", type=float, required=True, help="Impact x position [mm]")
    p.add_argument("--impact_y", type=float, required=True, help="Impact y position [mm]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = pick_device()

    ckpt_path = Path(args.ckpt) if args.ckpt else (DEFAULT_CKPT if DEFAULT_CKPT.exists() else LATEST_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No config at {CONFIG_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    scaler = ImpactTabularScaler.load(SCALER_PATH)

    gen = Generator(
        num_params=config["num_params"],
        num_numerical_outputs=config["num_numerical_outputs"],
        latent_dim=config["latent_dim"],
        img_channels=config["img_channels"],
        img_size=config["img_size"],
        init_size=config["init_size"],
        init_channels=config["init_channels"],
        w_dim=config.get("w_dim", 512),
        use_grad_checkpoint=False,
    ).to(device)

    phys = PhysicsHead(
        num_params=config["num_params"],
        num_outputs=config["num_numerical_outputs"],
        hidden=config.get("phys_hidden", 256),
        depth=config.get("phys_depth", 6),
        num_freqs=config.get("phys_num_freqs", 16),
        sigma=config.get("phys_sigma", 5.0),
    ).to(device)

    ck = torch.load(ckpt_path, map_location=device)
    gen.load_state_dict(ck.get("ema_g") or ck["generator"], strict=True)
    phys.load_state_dict(ck.get("ema_p") or ck["physics_head"], strict=True)
    gen.eval(); phys.eval()

    raw_params = np.array(
        [[args.velocity, args.radius, args.impact_x, args.impact_y]], dtype=np.float32
    )
    norm_params = scaler.transform_params_array(raw_params)
    params_t = torch.from_numpy(norm_params).to(device)

    torch.manual_seed(args.seed)
    z = torch.randn(1, config["latent_dim"], device=device)

    with torch.no_grad():
        img = gen(z, params_t)
        mu, log_var = phys(params_t)
        # mu/log_var are in scaler-space (log1p + standardize). To get
        # mean +/- sigma in physical units we sample N points in scaler space
        # and invert per-sample, then summarize.
        std = torch.exp(0.5 * log_var)
        N = 256
        samples_norm = mu + std * torch.randn(N, mu.size(-1), device=device)
        # broadcast batched inversion expects (B, num_outputs); already is.
        samples_phys = scaler.inverse_transform_outputs_tensor(samples_norm).cpu().numpy()
        mean_phys = samples_phys.mean(axis=0)
        std_phys = samples_phys.std(axis=0)

    # Image -> PNG
    arr = ((img.clamp(-1, 1) + 1.0) * 127.5).cpu().numpy().astype(np.uint8)[0]
    arr = np.transpose(arr, (1, 2, 0))
    pil = Image.fromarray(arr, mode="RGBA" if config["img_channels"] == 4 else "RGB")

    if args.out:
        out_dir = Path(args.out)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        existing = sorted(OUT_DIR.glob("sim_pred_*"))
        next_id = (int(existing[-1].name.split("_")[-1]) + 1) if existing else 1
        out_dir = OUT_DIR / f"sim_pred_{next_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pil.save(out_dir / "generated_image.png")

    summary = {
        "checkpoint": str(ckpt_path),
        "model_version": config.get("model_version", "v3.x"),
        "inputs": {
            "velocity_ms": args.velocity,
            "proj_radius_mm": args.radius,
            "impact_x_mm": args.impact_x,
            "impact_y_mm": args.impact_y,
            "seed": args.seed,
        },
        "predictions": {
            name: {
                "mean": float(m),
                "std": float(s),
                "unit": unit,
                "display": display,
            }
            for name, m, s, unit, display in zip(
                OUTPUT_SHORT_NAMES, mean_phys, std_phys, OUTPUT_UNITS, OUTPUT_DISPLAY_NAMES
            )
        },
    }
    (out_dir / "prediction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved image      -> {out_dir/'generated_image.png'}")
    print(f"Saved prediction -> {out_dir/'prediction.json'}")
    for k, v in summary["predictions"].items():
        print(f"  {v['display']:25s}: {v['mean']:.3f} +/- {v['std']:.3f} {v['unit']}")


if __name__ == "__main__":
    main()
