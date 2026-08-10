"""
Train a multi-modal Param2Image cGAN (PyTorch).

Maps impact parameters to a post-impact RGBA image and three numerical FEA
responses (max Mises, max PEEQ, max deflection).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None

from p2i.data_loader import load_dataset_pytorch
from p2i.model import Discriminator, Generator
from p2i.paths import checkpoint_dir, default_data_dir, logs_dir, repo_root
from p2i.preprocessing import SCALER_FILENAME, ImpactTabularScaler

REPO_ROOT = repo_root(Path(__file__))
DATA_DIR = default_data_dir(Path(__file__))
CSV_PATH = DATA_DIR / "results_master.csv"

IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 4
NUM_PARAMS = 4
NUM_NUMERICAL_OUTPUTS = 3
LATENT_DIM = 100
BATCH_SIZE = 2
EPOCHS = 150
LAMBDA_L1 = 100
LAMBDA_NUMERICAL = 10.0
LEARNING_RATE = 2e-4
BETA1 = 0.5
USE_TABULAR_SCALING = True
NUMERICAL_HEAD_MODE = "physics"
GRAD_CLIP_NORM = 5.0

CHECKPOINT_DIR = checkpoint_dir(Path(__file__))
LOG_DIR = logs_dir(Path(__file__))
MODEL_SAVE_PATH = CHECKPOINT_DIR / "cGAN_generator.pth"
SCALER_SAVE_PATH = CHECKPOINT_DIR / SCALER_FILENAME
CONFIG_SAVE_PATH = CHECKPOINT_DIR / "cGAN_generator_config.json"


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif ipex is not None and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.version.hip:
        device = torch.device("hip")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"--- Using device: {device} ---")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    scaler = ImpactTabularScaler(output_transform="log1p") if USE_TABULAR_SCALING else None
    dataloader, scaler = load_dataset_pytorch(
        str(CSV_PATH),
        str(DATA_DIR),
        BATCH_SIZE,
        IMG_HEIGHT,
        IMG_WIDTH,
        img_channels=IMG_CHANNELS,
        scaler=scaler,
        fit_scaler=USE_TABULAR_SCALING,
        return_scaler=True,
    )
    if scaler is not None:
        scaler.save(SCALER_SAVE_PATH)
        print(f"--- Saved tabular normalization statistics to {SCALER_SAVE_PATH} ---")

    generator = Generator(
        num_params=NUM_PARAMS,
        latent_dim=LATENT_DIM,
        num_numerical_outputs=NUM_NUMERICAL_OUTPUTS,
        img_channels=IMG_CHANNELS,
        numerical_head_mode=NUMERICAL_HEAD_MODE,
    ).to(device)
    discriminator = Discriminator(
        num_params=NUM_PARAMS, img_size=IMG_HEIGHT, img_channels=IMG_CHANNELS
    ).to(device)

    adversarial_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    numerical_loss = nn.SmoothL1Loss(beta=0.5)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999)
    )

    writer = SummaryWriter(str(LOG_DIR))

    print("\n--- Starting Multi-modal Model Training ---")
    for epoch in range(EPOCHS):
        start_time = time.time()
        for i, (params, real_imgs, real_numerical) in enumerate(dataloader):
            params = params.to(device)
            real_imgs = real_imgs.to(device)
            real_numerical = real_numerical.to(device)

            valid = torch.ones(real_imgs.size(0), 1, device=device, requires_grad=False)
            fake = torch.zeros(real_imgs.size(0), 1, device=device, requires_grad=False)

            optimizer_g.zero_grad()
            z = torch.randn(real_imgs.size(0), LATENT_DIM, device=device)
            fake_imgs, predicted_numerical = generator(z, params)

            g_adv_loss = adversarial_loss(discriminator(fake_imgs, params), valid)
            g_l1_loss = l1_loss(fake_imgs, real_imgs)
            g_numerical_loss = numerical_loss(predicted_numerical, real_numerical)
            g_loss = g_adv_loss + (LAMBDA_L1 * g_l1_loss) + (LAMBDA_NUMERICAL * g_numerical_loss)

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_NORM)
            optimizer_g.step()

            optimizer_d.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, params), valid)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach(), params), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_NORM)
            optimizer_d.step()

            if i % 10 == 0:
                print(
                    f"  [Epoch {epoch + 1}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G total loss: {g_loss.item():.4f}] "
                    f"[G adv: {g_adv_loss.item():.4f}] [G L1: {g_l1_loss.item():.4f}] "
                    f"[G num: {g_numerical_loss.item():.4f}]"
                )

        writer.add_scalar("Loss/Discriminator", d_loss.item(), epoch)
        writer.add_scalar("Loss/Generator_Total", g_loss.item(), epoch)
        writer.add_scalar("Loss/Generator_Adversarial", g_adv_loss.item(), epoch)
        writer.add_scalar("Loss/Generator_L1", g_l1_loss.item(), epoch)
        writer.add_scalar("Loss/Generator_Numerical", g_numerical_loss.item(), epoch)

        print(f"--- Epoch {epoch + 1} finished in {time.time() - start_time:.2f}s ---")

        if (epoch + 1) % 20 == 0:
            torch.save(generator.state_dict(), MODEL_SAVE_PATH)
            print(f"--- Saved model checkpoint to {MODEL_SAVE_PATH} ---")

    writer.close()
    torch.save(generator.state_dict(), MODEL_SAVE_PATH)
    config = {
        "num_params": NUM_PARAMS,
        "num_numerical_outputs": NUM_NUMERICAL_OUTPUTS,
        "latent_dim": LATENT_DIM,
        "img_channels": IMG_CHANNELS,
        "img_size": IMG_HEIGHT,
        "numerical_head_mode": NUMERICAL_HEAD_MODE,
        "tabular_scaling": USE_TABULAR_SCALING,
        "scaler_path": str(SCALER_SAVE_PATH) if scaler is not None else None,
        "lambda_l1": LAMBDA_L1,
        "lambda_numerical": LAMBDA_NUMERICAL,
        "numerical_loss": "SmoothL1Loss(beta=0.5)",
        "learning_rate": LEARNING_RATE,
    }
    CONFIG_SAVE_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\n--- Training Complete. ---")
    print(f"Final model saved to {MODEL_SAVE_PATH}")
    print(f"Training configuration saved to {CONFIG_SAVE_PATH}")


if __name__ == "__main__":
    train()
