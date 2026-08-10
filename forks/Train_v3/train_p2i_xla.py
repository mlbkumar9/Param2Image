"""
Param2Image v3.1 training script — TPU / XLA backend.

Designed for Colab free-tier TPU v5e-1 (16 GB HBM, 2-hour session) and
any other single-chip TPU reachable via torch_xla.

The model code in model_p2i_pytorch.py is unchanged — every layer used
here (Conv2d, Linear, InstanceNorm2d, LeakyReLU, SiLU, GELU, LayerNorm,
spectral_norm, F.interpolate) is XLA-compatible and runs natively on TPU.

What changes versus train_p2i_pytorch.py (the CUDA version):
  * device = xm.xla_device() instead of CUDA
  * Native bf16 via XLA_USE_BF16=1 (no GradScaler — TPUs don't need loss scaling)
  * DataLoader wrapped in MpDeviceLoader for async device transfer
  * xm.optimizer_step() instead of optimizer.step() (handles mark_step)
  * xm.save() for checkpoints (correct host gating, even on single chip)
  * cudnn.benchmark removed (XLA has its own autotuner)
  * Gradient checkpointing optional but discouraged (XLA recompute interacts
    poorly with the compiled graph; usually slower than just storing activations
    on TPU's large HBM)

Expected per-epoch time on v5e-1 with batch=8 and bf16:
  ~60-90 s/epoch once XLA finishes its first-iteration compilation.
The first iteration is slow (3-10 min) because XLA compiles each
distinct computational graph it sees. Subsequent iterations reuse the
cached compiled graph.

R1 lazy regularization is KEPT (even though it causes XLA to compile two
distinct graphs — one with R1, one without). Both graphs are cached after
the first encounter, so the cost is amortized across all subsequent steps.
"""
from __future__ import annotations

# Set BEFORE importing torch_xla.
import os
os.environ.setdefault("XLA_USE_BF16", "1")          # native bf16 everywhere
os.environ.setdefault("PT_XLA_DEBUG", "0")          # set to 1 to debug recompiles
os.environ.setdefault("XLA_IR_DEBUG", "0")

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    # TPU Colab runtime ships without tensorboard. Fall back to a no-op writer
    # so training proceeds and metrics still go to stdout.
    class SummaryWriter:
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def flush(self): pass
        def close(self): pass

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.distributed.parallel_loader import MpDeviceLoader

from data_loader_p2i_pytorch import load_dataset_pytorch
from model_p2i_pytorch import (
    Discriminator,
    EMA,
    Generator,
    PhysicsHead,
    diff_augment,
)
from position_supervision import impact_location_loss, impact_position_error_pixels, pixels_to_plate_mm
from preprocessing_p2i import SCALER_FILENAME, ImpactTabularScaler


# ---------------------------------------------------------------------------
# Configuration (env-overridable — same names as CUDA script when possible)
# ---------------------------------------------------------------------------

def _env_int(name, default):
    v = os.environ.get(name);  return int(v) if v is not None else default

def _env_float(name, default):
    v = os.environ.get(name);  return float(v) if v is not None else default

def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None: return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_str(name, default):
    v = os.environ.get(name);  return v if v is not None else default


CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
from p2i_paths import data_dir, checkpoint_dir
DEFAULT_DATA_DIR = data_dir()

DATA_DIR = Path(_env_str("P2I_DATA_DIR", str(DEFAULT_DATA_DIR)))
CSV_PATH = Path(_env_str("P2I_CSV_PATH", str(DATA_DIR / "results_master.csv")))

IMG_SIZE = _env_int("P2I_IMG_SIZE", 1024)
IMG_CHANNELS = _env_int("P2I_IMG_CHANNELS", 4)
NUM_PARAMS = 4
NUM_NUMERICAL_OUTPUTS = 3
LATENT_DIM = _env_int("P2I_LATENT_DIM", 100)
INIT_SIZE = _env_int("P2I_INIT_SIZE", 16)
INIT_CHANNELS = _env_int("P2I_INIT_CHANNELS", 512)
W_DIM = _env_int("P2I_W_DIM", 512)

# TPU has 16 GB HBM and bf16 native -> much bigger batch works than T4.
BATCH_SIZE = _env_int("P2I_BATCH_SIZE", 8)
GRAD_ACCUM = _env_int("P2I_GRAD_ACCUM", 1)
EPOCHS = _env_int("P2I_EPOCHS", 200)
WARMUP_EPOCHS = _env_int("P2I_WARMUP_EPOCHS", 5)
VAL_SPLIT = _env_float("P2I_VAL_SPLIT", 0.10)

LR_G = _env_float("P2I_LR_G", 1e-4)
LR_D = _env_float("P2I_LR_D", 4e-5)
LR_PHYS = _env_float("P2I_LR_PHYS", 5e-4)
LR_MIN = _env_float("P2I_LR_MIN", 1e-6)

WD_G = _env_float("P2I_WD_G", 1e-4)
WD_D = _env_float("P2I_WD_D", 1e-4)
WD_PHYS = _env_float("P2I_WD_PHYS", 1e-3)
BETA1_GAN = _env_float("P2I_BETA1_GAN", 0.0)
BETA2_GAN = _env_float("P2I_BETA2_GAN", 0.99)
BETA1_PHYS = _env_float("P2I_BETA1_PHYS", 0.9)
BETA2_PHYS = _env_float("P2I_BETA2_PHYS", 0.999)

EMA_DECAY = _env_float("P2I_EMA_DECAY", 0.999)

LAMBDA_L1 = _env_float("P2I_LAMBDA_L1", 10.0)
LAMBDA_NUM = _env_float("P2I_LAMBDA_NUM", 1.0)
LAMBDA_POS = _env_float("P2I_LAMBDA_POS", 10.0)
R1_GAMMA = _env_float("P2I_R1_GAMMA", 10.0)
R1_INTERVAL = _env_int("P2I_R1_INTERVAL", 16)
POS_DOWNSAMPLE_SIZE = _env_int("P2I_POS_DOWNSAMPLE_SIZE", 128)
POS_TEMPERATURE = _env_float("P2I_POS_TEMPERATURE", 12.0)
POS_BORDER_FRACTION = _env_float("P2I_POS_BORDER_FRACTION", 0.04)
PLATE_WIDTH_MM = _env_float("P2I_PLATE_WIDTH_MM", 2500.0)

DIFFAUG_POLICY = _env_str("P2I_DIFFAUG_POLICY", "color,cutout")
MIXUP_ALPHA = _env_float("P2I_MIXUP_ALPHA", 0.2)

PHYS_NUM_FREQS = _env_int("P2I_PHYS_NUM_FREQS", 16)
PHYS_HIDDEN = _env_int("P2I_PHYS_HIDDEN", 256)
PHYS_DEPTH = _env_int("P2I_PHYS_DEPTH", 6)
PHYS_SIGMA = _env_float("P2I_PHYS_SIGMA", 5.0)

USE_GRAD_CHECKPOINT = _env_bool("P2I_USE_GRAD_CHECKPOINT", False)
GRAD_CLIP_NORM = _env_float("P2I_GRAD_CLIP_NORM", 5.0)
NUM_WORKERS = _env_int("P2I_NUM_WORKERS", 2)
SEED = _env_int("P2I_SEED", 42)

CHECKPOINT_DIR = Path(_env_str("P2I_CHECKPOINT_DIR", str(checkpoint_dir("training_checkpoints_position_xla"))))
DRIVE_BACKUP_DIR = _env_str("P2I_DRIVE_BACKUP_DIR", "")
LOG_DIR = Path(_env_str("P2I_LOG_DIR", str(CURRENT_DIR / "logs_position_xla")))
SAVE_EVERY_STEPS = _env_int("P2I_SAVE_EVERY_STEPS", 50)
TRAIN_LOG_PATH = _env_str("P2I_TRAIN_LOG_PATH", "")
RESUME_CKPT = _env_str("P2I_RESUME_CKPT", "auto")
MAX_R1_PENALTY = _env_float("P2I_MAX_R1_PENALTY", 100.0)

LATEST_CKPT = CHECKPOINT_DIR / "ckpt_latest.pt"
BEST_CKPT = CHECKPOINT_DIR / "ckpt_best.pt"
BEST_POS_CKPT = CHECKPOINT_DIR / "ckpt_best_position.pt"
CONFIG_PATH = CHECKPOINT_DIR / "config.json"
SCALER_PATH = CHECKPOINT_DIR / SCALER_FILENAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_warmup_lr(step, total_steps, warmup_steps, lr_max, lr_min):
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint_xla(path: Path, state: dict):
    """Write checkpoint via xm.save (host-gated, sync-safe). Single-chip-friendly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    xm.save(state, str(tmp))
    if xm.is_master_ordinal():
        os.replace(tmp, path)


def mirror_to_drive(local_path: Path):
    if not DRIVE_BACKUP_DIR:
        return
    dest_root = Path(DRIVE_BACKUP_DIR)
    if not dest_root.exists():
        return
    try:
        import shutil
        shutil.copy2(local_path, dest_root / local_path.name)
    except Exception as exc:
        print(f"[warn] Drive mirror failed: {exc}", flush=True)


class _TeeStream:
    def __init__(self, primary, log_file):
        self.primary = primary
        self.log_file = log_file
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, text):
        self.primary.write(text)
        self.log_file.write(text)
        if "\n" in text:
            self.flush()

    def flush(self):
        self.primary.flush()
        self.log_file.flush()

    def isatty(self):
        return self.primary.isatty()


def install_output_tee():
    if getattr(sys.stdout, "_p2i_tee_installed", False):
        return
    log_path = Path(TRAIN_LOG_PATH) if TRAIN_LOG_PATH else None
    if log_path is None and DRIVE_BACKUP_DIR:
        log_path = Path(DRIVE_BACKUP_DIR) / "train_console_xla.log"
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.stdout, log_file)
        sys.stderr = _TeeStream(sys.stderr, log_file)
        sys.stdout._p2i_tee_installed = True
        print(f"--- Appending console log to {log_path} ---", flush=True)
    except Exception as exc:
        print(f"[warn] Could not open training log {log_path}: {exc}", flush=True)


def _is_finite_tree(obj):
    if torch.is_tensor(obj):
        return bool(torch.isfinite(obj.detach().cpu()).all().item())
    if isinstance(obj, dict):
        return all(_is_finite_tree(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_is_finite_tree(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


def checkpoint_has_finite_weights(ck):
    keys = ("generator", "discriminator", "physics_head", "ema_g", "ema_p", "opt_g", "opt_d", "opt_p")
    return all(key not in ck or _is_finite_tree(ck[key]) for key in keys)


def assert_checkpoint_state_finite(state):
    if not checkpoint_has_finite_weights(state):
        raise FloatingPointError("Refusing to save checkpoint with non-finite weights or optimizer state.")


def ensure_finite_tensor(name, value):
    if not torch.isfinite(value.detach().cpu()).all():
        raise FloatingPointError(f"Non-finite {name}; aborting before checkpoint corruption.")


def ensure_finite_float(name, value):
    if value is not None and not math.isfinite(float(value)):
        raise FloatingPointError(f"Non-finite {name}; aborting before checkpoint corruption.")


def resolve_resume_checkpoint():
    choice = RESUME_CKPT.strip()
    choice_l = choice.lower()
    if choice_l in {"", "none", "off", "false", "0"}:
        return None
    if choice_l in {"auto", "latest"}:
        candidates = [LATEST_CKPT] if choice_l == "latest" else [LATEST_CKPT, BEST_POS_CKPT, BEST_CKPT]
    elif choice_l in {"best_position", "best_pos", "position"}:
        candidates = [BEST_POS_CKPT]
    elif choice_l == "best":
        candidates = [BEST_CKPT]
    else:
        candidates = [Path(choice)]

    for path in candidates:
        if not path.exists():
            continue
        try:
            ck = torch.load(path, map_location="cpu")
            if checkpoint_has_finite_weights(ck):
                return path
            print(f"[warn] Skipping non-finite checkpoint {path}", flush=True)
        except Exception as exc:
            print(f"[warn] Skipping unreadable checkpoint {path}: {exc}", flush=True)
    return None


# Non-saturating logistic GAN loss
def d_logistic_loss(real_logits, fake_logits):
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()

def g_nonsaturating_loss(fake_logits):
    return F.softplus(-fake_logits).mean()


def hetero_gaussian_nll(mu, log_var, y):
    return 0.5 * ((y - mu) ** 2 * torch.exp(-log_var) + log_var).mean()


def mixup_pairs(params, outputs, alpha):
    if alpha <= 0.0 or params.size(0) < 2:
        return params, outputs
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(params.size(0), device=params.device)
    return (
        lam * params + (1.0 - lam) * params[perm],
        lam * outputs + (1.0 - lam) * outputs[perm],
    )


@torch.no_grad()
def ssim_proxy(x, y):
    x = (x + 1.0) / 2.0 if x.min() < 0 else x
    y = (y + 1.0) / 2.0 if y.min() < 0 else y
    mu_x = x.mean(dim=(2, 3), keepdim=True)
    mu_y = y.mean(dim=(2, 3), keepdim=True)
    var_x = ((x - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    var_y = ((y - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    cov = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3), keepdim=True)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    return float(
        (((2 * mu_x * mu_y + c1) * (2 * cov + c2))
         / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))).mean().item()
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train():
    install_output_tee()
    seed_all(SEED)
    # Prefer the new torch_xla.device() API (torch_xla >= 2.4) and fall back to
    # the legacy xm.xla_device() for older installs.
    device = getattr(torch_xla, "device", xm.xla_device)()
    print(f"--- Device: {device} (XLA / TPU) ---", flush=True)
    print(f"--- IMG_SIZE={IMG_SIZE}, batch={BATCH_SIZE} x grad_accum={GRAD_ACCUM} ---", flush=True)
    print(f"--- LR_G={LR_G} | LR_D={LR_D} | LR_PHYS={LR_PHYS} ---", flush=True)
    print(f"--- AdamW GAN(beta1={BETA1_GAN}, beta2={BETA2_GAN}) | "
          f"AdamW Phys(beta1={BETA1_PHYS}, beta2={BETA2_PHYS}) | EMA={EMA_DECAY} ---", flush=True)
    print(f"--- DiffAug='{DIFFAUG_POLICY}' | Mixup alpha={MIXUP_ALPHA} | "
          f"lambda_pos={LAMBDA_POS} | bf16=native ---", flush=True)
    print(f"--- DATA_DIR={DATA_DIR} ---", flush=True)
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    if SCALER_PATH.exists():
        scaler = ImpactTabularScaler.load(SCALER_PATH)
        fit_scaler = False
        print(f"--- Loaded scaler from {SCALER_PATH} ---", flush=True)
    else:
        scaler = ImpactTabularScaler(output_transform="log1p")
        fit_scaler = True

    if VAL_SPLIT > 0:
        train_loader_raw, val_loader_raw = load_dataset_pytorch(
            str(CSV_PATH), str(DATA_DIR), BATCH_SIZE, IMG_SIZE, IMG_SIZE,
            num_workers=NUM_WORKERS, img_channels=IMG_CHANNELS, scaler=scaler,
            fit_scaler=fit_scaler, val_split=VAL_SPLIT, return_scaler=False,
        )
    else:
        train_loader_raw = load_dataset_pytorch(
            str(CSV_PATH), str(DATA_DIR), BATCH_SIZE, IMG_SIZE, IMG_SIZE,
            num_workers=NUM_WORKERS, img_channels=IMG_CHANNELS, scaler=scaler,
            fit_scaler=fit_scaler,
        )
        val_loader_raw = None

    if fit_scaler:
        scaler.save(SCALER_PATH)
        mirror_to_drive(SCALER_PATH)
        print(f"--- Saved scaler to {SCALER_PATH} ---", flush=True)

    # Wrap loaders for async device transfer.
    train_loader = MpDeviceLoader(train_loader_raw, device)
    val_loader = MpDeviceLoader(val_loader_raw, device) if val_loader_raw is not None else None

    # --- Models ---
    generator = Generator(
        num_params=NUM_PARAMS,
        num_numerical_outputs=NUM_NUMERICAL_OUTPUTS,
        latent_dim=LATENT_DIM,
        img_channels=IMG_CHANNELS,
        img_size=IMG_SIZE,
        init_size=INIT_SIZE,
        init_channels=INIT_CHANNELS,
        w_dim=W_DIM,
        use_grad_checkpoint=USE_GRAD_CHECKPOINT,
    ).to(device)

    discriminator = Discriminator(
        num_params=NUM_PARAMS,
        img_channels=IMG_CHANNELS,
        img_size=IMG_SIZE,
    ).to(device)

    physics_head = PhysicsHead(
        num_params=NUM_PARAMS,
        num_outputs=NUM_NUMERICAL_OUTPUTS,
        hidden=PHYS_HIDDEN,
        depth=PHYS_DEPTH,
        num_freqs=PHYS_NUM_FREQS,
        sigma=PHYS_SIGMA,
    ).to(device)

    ema_g = EMA(generator, decay=EMA_DECAY)
    ema_p = EMA(physics_head, decay=EMA_DECAY)

    n_g = sum(p.numel() for p in generator.parameters())
    n_d = sum(p.numel() for p in discriminator.parameters())
    n_p = sum(p.numel() for p in physics_head.parameters())
    print(f"--- Generator: {n_g/1e6:.2f} M | Discriminator: {n_d/1e6:.2f} M | "
          f"PhysicsHead: {n_p/1e6:.2f} M ---", flush=True)

    opt_g = torch.optim.AdamW(generator.parameters(),
                              lr=LR_G, betas=(BETA1_GAN, BETA2_GAN), weight_decay=WD_G)
    opt_d = torch.optim.AdamW(discriminator.parameters(),
                              lr=LR_D, betas=(BETA1_GAN, BETA2_GAN), weight_decay=WD_D)
    opt_p = torch.optim.AdamW(physics_head.parameters(),
                              lr=LR_PHYS, betas=(BETA1_PHYS, BETA2_PHYS), weight_decay=WD_PHYS)

    steps_per_epoch = max(1, len(train_loader_raw) // GRAD_ACCUM)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    print(f"--- Total optimizer steps: {total_steps} (warmup: {warmup_steps}) ---", flush=True)

    # --- Resume ---
    start_epoch = 0
    best_ssim = -1.0
    best_pos_px = float("inf")
    global_step = 0
    resume_path = resolve_resume_checkpoint()
    if resume_path is not None:
        print(f"--- Resuming from {resume_path} ---", flush=True)
        ck = torch.load(resume_path, map_location="cpu")
        if not checkpoint_has_finite_weights(ck):
            raise FloatingPointError(f"Checkpoint {resume_path} contains non-finite weights.")
        generator.load_state_dict(ck["generator"])
        discriminator.load_state_dict(ck["discriminator"])
        physics_head.load_state_dict(ck["physics_head"])
        ema_g.load_state_dict(ck["ema_g"])
        ema_p.load_state_dict(ck["ema_p"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        opt_p.load_state_dict(ck["opt_p"])
        epoch_complete = ck.get("epoch_complete", True)
        start_epoch = ck["epoch"] + (1 if epoch_complete else 0)
        global_step = ck.get("global_step", start_epoch * steps_per_epoch)
        best_ssim = ck.get("best_ssim", -1.0)
        best_pos_px = ck.get("best_pos_px", float("inf"))
        rng_state = ck.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state.cpu())
        print(f"--- Resumed at stored epoch index {start_epoch}; next displayed epoch is {start_epoch + 1}, "
              f"global_step {global_step}, best SSIM {best_ssim:.4f}, best position {best_pos_px:.2f}px ---", flush=True)

    writer = SummaryWriter(str(LOG_DIR))

    config = {
        "model_version": "v3.1-xla",
        "backend": "torch_xla TPU",
        "img_size": IMG_SIZE, "img_channels": IMG_CHANNELS,
        "num_params": NUM_PARAMS, "num_numerical_outputs": NUM_NUMERICAL_OUTPUTS,
        "latent_dim": LATENT_DIM, "init_size": INIT_SIZE,
        "init_channels": INIT_CHANNELS, "w_dim": W_DIM,
        "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
        "lr_g": LR_G, "lr_d": LR_D, "lr_phys": LR_PHYS, "lr_min": LR_MIN,
        "wd_g": WD_G, "wd_d": WD_D, "wd_phys": WD_PHYS,
        "beta1_gan": BETA1_GAN, "beta2_gan": BETA2_GAN,
        "beta1_phys": BETA1_PHYS, "beta2_phys": BETA2_PHYS,
        "ema_decay": EMA_DECAY,
        "lambda_l1": LAMBDA_L1, "lambda_num": LAMBDA_NUM, "lambda_pos": LAMBDA_POS,
        "r1_gamma": R1_GAMMA, "r1_interval": R1_INTERVAL,
        "max_r1_penalty": MAX_R1_PENALTY,
        "resume_ckpt": RESUME_CKPT,
        "pos_downsample_size": POS_DOWNSAMPLE_SIZE,
        "pos_temperature": POS_TEMPERATURE,
        "pos_border_fraction": POS_BORDER_FRACTION,
        "plate_width_mm": PLATE_WIDTH_MM,
        "diffaug_policy": DIFFAUG_POLICY, "mixup_alpha": MIXUP_ALPHA,
        "phys_hidden": PHYS_HIDDEN, "phys_depth": PHYS_DEPTH,
        "phys_num_freqs": PHYS_NUM_FREQS, "phys_sigma": PHYS_SIGMA,
        "optimizer": "AdamW (parameter groups)",
        "gan_loss": "non-saturating logistic + R1 lazy",
        "numerical_loss": "heteroscedastic Gaussian NLL",
        "precision": "bf16 native (XLA_USE_BF16=1)",
        "use_grad_checkpoint": USE_GRAD_CHECKPOINT,
        "scaler_path": str(SCALER_PATH),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    mirror_to_drive(CONFIG_PATH)

    def make_checkpoint_state(epoch):
        # Bring weights to CPU before pickling. xm.save handles host gating.
        return {
            "epoch": epoch,
            "epoch_complete": False,
            "global_step": global_step,
            "generator": {k: v.detach().cpu() for k, v in generator.state_dict().items()},
            "discriminator": {k: v.detach().cpu() for k, v in discriminator.state_dict().items()},
            "physics_head": {k: v.detach().cpu() for k, v in physics_head.state_dict().items()},
            "ema_g": {k: v.detach().cpu() for k, v in ema_g.state_dict().items()},
            "ema_p": {k: v.detach().cpu() for k, v in ema_p.state_dict().items()},
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "opt_p": opt_p.state_dict(),
            "best_ssim": best_ssim,
            "best_pos_px": best_pos_px,
            "rng_state": torch.get_rng_state(),
            "config": config,
        }

    print("\n--- Starting training ---\n", flush=True)
    print("NOTE: first iteration compiles the XLA graph (3-10 minutes). "
          "Subsequent iterations are fast.\n", flush=True)

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        generator.train(); discriminator.train(); physics_head.train()
        opt_g.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)
        opt_p.zero_grad(set_to_none=True)
        running = {k: 0.0 for k in ("d", "g", "adv", "l1", "pos", "r1", "phys_nll")}
        running["n"] = 0
        accum = 0

        for it, (params, real_imgs, real_num) in enumerate(train_loader):
            bsz = params.size(0)
            z = torch.randn(bsz, LATENT_DIM, device=device)

            # ---- LR schedule (on optimizer-step boundaries) ----
            if accum == 0:
                lr_g_now = cosine_warmup_lr(global_step, total_steps, warmup_steps, LR_G, LR_MIN)
                lr_d_now = cosine_warmup_lr(global_step, total_steps, warmup_steps, LR_D, LR_MIN)
                lr_p_now = cosine_warmup_lr(global_step, total_steps, warmup_steps, LR_PHYS, LR_MIN)
                for pg in opt_g.param_groups: pg["lr"] = lr_g_now
                for pg in opt_d.param_groups: pg["lr"] = lr_d_now
                for pg in opt_p.param_groups: pg["lr"] = lr_p_now

            # ============================================================
            # Single G forward, reused for D and G updates.
            # ============================================================
            fake_imgs = generator(z, params)

            # ============================================================
            # Discriminator update (non-saturating logistic + DiffAug + R1)
            # ============================================================
            for p in generator.parameters():
                p.requires_grad_(False)
            d_real_logits = discriminator(diff_augment(real_imgs, DIFFAUG_POLICY), params)
            d_fake_logits = discriminator(diff_augment(fake_imgs.detach(), DIFFAUG_POLICY), params)
            d_loss = d_logistic_loss(d_real_logits, d_fake_logits)

            r1_pen = torch.zeros((), device=device)
            if R1_GAMMA > 0 and (global_step % R1_INTERVAL == 0):
                real_for_r1 = real_imgs.detach().requires_grad_(True)
                d_real_for_r1 = discriminator(real_for_r1, params)
                grad_real = torch.autograd.grad(
                    outputs=d_real_for_r1.sum(),
                    inputs=real_for_r1,
                    create_graph=True,
                    retain_graph=True,
            )[0]
            r1_pen = grad_real.pow(2).flatten(1).sum(dim=1).mean() * 0.5 * R1_GAMMA
            if MAX_R1_PENALTY > 0 and torch.isfinite(r1_pen.detach().cpu()):
                r1_pen = r1_pen.clamp(max=MAX_R1_PENALTY)

            d_total = (d_loss + r1_pen) / GRAD_ACCUM
            ensure_finite_tensor("d_loss", d_loss)
            ensure_finite_tensor("r1_pen", r1_pen)
            ensure_finite_tensor("d_total", d_total)
            d_total.backward()
            for p in generator.parameters():
                p.requires_grad_(True)

            # ============================================================
            # Generator update (adversarial + reconstruction + location losses)
            # ============================================================
            for p in discriminator.parameters():
                p.requires_grad_(False)
            g_fake_logits = discriminator(diff_augment(fake_imgs, DIFFAUG_POLICY), params)
            g_adv = g_nonsaturating_loss(g_fake_logits)
            g_l1 = F.l1_loss(fake_imgs, real_imgs)
            g_pos = impact_location_loss(
                fake_imgs,
                real_imgs,
                downsample_size=POS_DOWNSAMPLE_SIZE,
                temperature=POS_TEMPERATURE,
                border_fraction=POS_BORDER_FRACTION,
            )
            g_loss = g_adv + LAMBDA_L1 * g_l1 + LAMBDA_POS * g_pos
            ensure_finite_tensor("g_loss", g_loss)
            (g_loss / GRAD_ACCUM).backward()
            for p in discriminator.parameters():
                p.requires_grad_(True)

            # ============================================================
            # PhysicsHead update (dedicated, with optional mixup)
            # ============================================================
            pm, py = mixup_pairs(params, real_num, MIXUP_ALPHA)
            mu_m, log_var_m = physics_head(pm)
            phys_nll = hetero_gaussian_nll(mu_m, log_var_m, py)
            ensure_finite_tensor("phys_nll", phys_nll)
            (LAMBDA_NUM * phys_nll / GRAD_ACCUM).backward()

            running["d"] += float(d_loss.detach())
            running["g"] += float(g_loss.detach())
            running["adv"] += float(g_adv.detach())
            running["l1"] += float(g_l1.detach())
            running["pos"] += float(g_pos.detach())
            running["phys_nll"] += float(phys_nll.detach())
            running["r1"] += float(r1_pen.detach())
            running["n"] += 1

            accum += 1
            if accum >= GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(physics_head.parameters(), GRAD_CLIP_NORM)

                # xm.optimizer_step handles mark_step + cross-replica reduce.
                # On a single chip there's no reduce, just the mark_step.
                xm.optimizer_step(opt_d, barrier=True)
                xm.optimizer_step(opt_g, barrier=True)
                xm.optimizer_step(opt_p, barrier=True)

                opt_d.zero_grad(set_to_none=True)
                opt_g.zero_grad(set_to_none=True)
                opt_p.zero_grad(set_to_none=True)
                ema_g.update(generator)
                ema_p.update(physics_head)
                global_step += 1
                if SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0:
                    state = make_checkpoint_state(epoch)
                    assert_checkpoint_state_finite(state)
                    save_checkpoint_xla(LATEST_CKPT, state)
                    mirror_to_drive(LATEST_CKPT)
                    writer.flush()
                    print(f"--- Saved step checkpoint global_step={global_step} -> {LATEST_CKPT} ---", flush=True)
                accum = 0

            if it % 100 == 0:
                print(
                    f"  [Ep {epoch+1}/{EPOCHS}] [Step {it}/{len(train_loader_raw)}] "
                    f"D={float(d_loss.detach()):.4f} G={float(g_loss.detach()):.4f} "
                    f"adv={float(g_adv.detach()):.4f} L1={float(g_l1.detach()):.4f} "
                    f"pos={float(g_pos.detach()):.4f} physNLL={float(phys_nll.detach()):.4f} "
                    f"r1={float(r1_pen.detach()):.4f} lr_g={opt_g.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        n = max(1, running["n"])
        for k in ("d", "g", "adv", "l1", "pos", "r1", "phys_nll"):
            writer.add_scalar(f"Loss/{k}", running[k] / n, epoch)
        writer.add_scalar("LR/G", opt_g.param_groups[0]["lr"], epoch)
        writer.add_scalar("LR/D", opt_d.param_groups[0]["lr"], epoch)
        writer.add_scalar("LR/PhysicsHead", opt_p.param_groups[0]["lr"], epoch)

        # ---- Validation with EMA copies (constructed on CPU then moved to TPU) ----
        ssim_val = None
        pos_error_px = None
        pos_error_mm = None
        if val_loader is not None and len(val_loader_raw) > 0:
            eval_gen = Generator(
                num_params=NUM_PARAMS, num_numerical_outputs=NUM_NUMERICAL_OUTPUTS,
                latent_dim=LATENT_DIM, img_channels=IMG_CHANNELS, img_size=IMG_SIZE,
                init_size=INIT_SIZE, init_channels=INIT_CHANNELS, w_dim=W_DIM,
                use_grad_checkpoint=False,
            )
            ema_g.copy_to(eval_gen); eval_gen.eval(); eval_gen = eval_gen.to(device)

            eval_phys = PhysicsHead(
                num_params=NUM_PARAMS, num_outputs=NUM_NUMERICAL_OUTPUTS,
                hidden=PHYS_HIDDEN, depth=PHYS_DEPTH,
                num_freqs=PHYS_NUM_FREQS, sigma=PHYS_SIGMA,
            )
            ema_p.copy_to(eval_phys); eval_phys.eval(); eval_phys = eval_phys.to(device)

            ssims, mu_phys_all, num_phys_all = [], [], []
            position_error_sum = 0.0
            position_sample_count = 0
            with torch.no_grad():
                for params_v, real_v, num_v in val_loader:
                    z_v = torch.randn(params_v.size(0), LATENT_DIM, device=device)
                    fake_v = eval_gen(z_v, params_v)
                    ssims.append(ssim_proxy(fake_v.float(), real_v.float()))
                    batch_pos_error = impact_position_error_pixels(
                        fake_v.float(),
                        real_v.float(),
                        downsample_size=POS_DOWNSAMPLE_SIZE,
                        temperature=POS_TEMPERATURE,
                        border_fraction=POS_BORDER_FRACTION,
                    )
                    position_error_sum += float(batch_pos_error) * params_v.size(0)
                    position_sample_count += params_v.size(0)
                    mu_v, _ = eval_phys(params_v)
                    mu_v_clipped = mu_v.float().clamp(-5.0, 5.0)
                    mu_phys = scaler.inverse_transform_outputs_tensor(mu_v_clipped.cpu())
                    num_phys = scaler.inverse_transform_outputs_tensor(num_v.float().cpu())
                    mu_phys_all.append(mu_phys.numpy())
                    num_phys_all.append(num_phys.numpy())
            del eval_gen, eval_phys

            ssim_val = float(np.mean(ssims)) if ssims else 0.0
            ensure_finite_float("ssim_val", ssim_val)
            if position_sample_count:
                pos_error_px = position_error_sum / position_sample_count
                ensure_finite_float("pos_error_px", pos_error_px)
                pos_error_mm = pixels_to_plate_mm(
                    pos_error_px,
                    IMG_SIZE,
                    plate_width_mm=PLATE_WIDTH_MM,
                    border_fraction=POS_BORDER_FRACTION,
                )
            mu_phys_all = np.concatenate(mu_phys_all, axis=0) if mu_phys_all else np.zeros((0, NUM_NUMERICAL_OUTPUTS))
            num_phys_all = np.concatenate(num_phys_all, axis=0) if num_phys_all else np.zeros((0, NUM_NUMERICAL_OUTPUTS))
            denom = np.abs(num_phys_all).clip(min=1e-6)
            mape_per_output = np.abs(mu_phys_all - num_phys_all) / denom
            writer.add_scalar("Val/SSIM_proxy", ssim_val, epoch)
            if pos_error_px is not None:
                writer.add_scalar("Val/Impact_Position_Error_px", pos_error_px, epoch)
                writer.add_scalar("Val/Impact_Position_Error_mm_approx", pos_error_mm, epoch)
            for i, short in enumerate(("mises", "peeq", "deflection")):
                if i < mape_per_output.shape[1] and mape_per_output.size:
                    writer.add_scalar(f"Val/MAPE_{short}", float(mape_per_output[:, i].mean()), epoch)
            print(
                f"--- Epoch {epoch+1} | Val SSIM (proxy): {ssim_val:.4f} | "
                f"impact pos={pos_error_px:.2f}px (~{pos_error_mm:.2f}mm) | "
                f"Val MAPE: mises={mape_per_output[:,0].mean():.1%} "
                f"peeq={mape_per_output[:,1].mean():.1%} "
                f"defl={mape_per_output[:,2].mean():.1%} ---"
                if mape_per_output.size else
                f"--- Epoch {epoch+1} | Val SSIM (proxy): {ssim_val:.4f} ---",
                flush=True,
            )

        # ---- Checkpoint ----
        state = make_checkpoint_state(epoch)
        state["epoch_complete"] = True
        assert_checkpoint_state_finite(state)
        if ssim_val is not None and ssim_val > best_ssim:
            best_ssim = ssim_val
            state["best_ssim"] = best_ssim
            save_checkpoint_xla(BEST_CKPT, state)
            mirror_to_drive(BEST_CKPT)
            print(f"--- New best SSIM {best_ssim:.4f} -> saved to {BEST_CKPT} ---", flush=True)

        if pos_error_px is not None and pos_error_px < best_pos_px:
            best_pos_px = pos_error_px
            state["best_pos_px"] = best_pos_px
            save_checkpoint_xla(BEST_POS_CKPT, state)
            mirror_to_drive(BEST_POS_CKPT)
            print(f"--- New best impact position {best_pos_px:.2f}px -> saved to {BEST_POS_CKPT} ---", flush=True)

        # Save latest after metric updates so resume state records current bests.
        state["best_ssim"] = best_ssim
        state["best_pos_px"] = best_pos_px
        save_checkpoint_xla(LATEST_CKPT, state)
        mirror_to_drive(LATEST_CKPT)

        print(f"--- Epoch {epoch+1} done in {time.time()-t0:.1f}s ---\n", flush=True)

    writer.close()
    print("--- Training complete ---", flush=True)


if __name__ == "__main__":
    train()
