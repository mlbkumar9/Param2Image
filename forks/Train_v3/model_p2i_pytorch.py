"""
Param2Image v3.1 model: StyleGAN-lite generator + SN-PatchGAN discriminator
                       + dedicated Fourier-featurized physics head.

Designed to fit 1024x1024 RGBA output on a 16 GB GPU (Colab T4) and to
produce both photorealistic damage images and well-calibrated numerical
predictions (peak von Mises, PEEQ, deflection) from impact parameters.

Key SOTA pieces:
  * StyleMapping  : params + noise -> latent style vector w
  * StyleResBlock : ResBlock with AdaIN-style channel-wise modulation
                     driven by w. Replaces plain BN+conv.
  * StyleGenerator: progressive upsampling stack of StyleResBlocks
  * StyleDiscriminator: SN-PatchGAN with minibatch-stddev layer
  * PhysicsHead   : deterministic Fourier-MLP regressor, heteroscedastic
                     outputs (mean + log-variance) for the numerical heads.
  * DiffAug       : differentiable augmentation for D (color + translation
                     + cutout), shared real/fake.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint as ckpt


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _load_state_dict(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("ema_state_dict", "generator_state_dict", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


# ---------------------------------------------------------------------------
# Style mapping (params + noise -> w)
# ---------------------------------------------------------------------------

class StyleMapping(nn.Module):
    """Map (params, noise) -> style vector w used at every G stage.
    StyleGAN-style 4-layer MLP. w has the same dim at every level."""

    def __init__(self, num_params: int, latent_dim: int, w_dim: int = 512, depth: int = 4):
        super().__init__()
        in_dim = num_params + latent_dim
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
        self.w_dim = w_dim

    def forward(self, params, noise):
        x = torch.cat([params, noise], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Style-modulated residual block (AdaIN)
# ---------------------------------------------------------------------------

class StyleResBlock(nn.Module):
    """ResBlock that injects the style vector w via channel-wise AdaIN.

    For each Conv2d output (in_channels -> out_channels), we predict
    a per-channel (scale, bias) from w and apply:
        h_norm = (h - mean) / std        (instance norm)
        h_styled = h_norm * (1 + scale) + bias
    """

    def __init__(self, in_channels: int, out_channels: int, w_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.style1 = nn.Linear(w_dim, out_channels * 2)
        self.style2 = nn.Linear(w_dim, out_channels * 2)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=False)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip = nn.Identity()

    def _ada_in(self, h, style, norm):
        scale, bias = style.chunk(2, dim=1)        # (B, C)
        scale = scale.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return norm(h) * (1.0 + scale) + bias

    def forward(self, x, w):
        identity = self.skip(x)
        h = self.conv1(x)
        h = self._ada_in(h, self.style1(w), self.in1)
        h = self.act(h)
        h = self.conv2(h)
        h = self._ada_in(h, self.style2(w), self.in2)
        h = self.act(h)
        return h + identity


class StyleUpStage(nn.Module):
    """Bilinear x2 upsample + StyleResBlock."""

    def __init__(self, in_channels: int, out_channels: int, w_dim: int):
        super().__init__()
        self.block = StyleResBlock(in_channels, out_channels, w_dim)

    def forward(self, x, w):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x, w)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Style-modulated residual generator with adjustable resolution.

    Defaults tuned for 1024x1024 on a 16 GB GPU:
      init_size=16, init_channels=512, w_dim=512,
      schedule = [512, 512, 384, 256, 128, 64, 32] (size 7, num_stages=6).
    """

    def __init__(
        self,
        num_params: int = 4,
        num_numerical_outputs: int = 3,
        latent_dim: int = 100,
        img_channels: int = 4,
        img_size: int = 1024,
        init_size: int = 16,
        init_channels: int = 512,
        w_dim: int = 512,
        channel_schedule=None,
        use_grad_checkpoint: bool = True,
    ):
        super().__init__()
        if img_size % init_size != 0:
            raise ValueError(f"img_size ({img_size}) must be a multiple of init_size ({init_size}).")

        num_stages = 0
        s = init_size
        while s < img_size:
            s *= 2
            num_stages += 1
        if init_size * (2 ** num_stages) != img_size:
            raise ValueError("img_size must be init_size * 2^k for some k.")

        if channel_schedule is None:
            # Halve channels per stage, clamped to 32 minimum, kept wider at the
            # high-resolution end (vs naive halving) to retain expressiveness.
            schedule = [
                max(init_channels // (2 ** k), 32) for k in range(num_stages + 1)
            ]
            schedule[0] = init_channels  # keep seed wide
            for i in range(1, len(schedule)):
                schedule[i] = max(schedule[i], 32)
        else:
            if len(channel_schedule) != num_stages + 1:
                raise ValueError(f"channel_schedule length must be {num_stages + 1}")
            schedule = list(channel_schedule)

        self.num_params = num_params
        self.num_numerical_outputs = num_numerical_outputs
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = init_size
        self.init_channels = schedule[0]
        self.w_dim = w_dim
        self.num_stages = num_stages
        self.use_grad_checkpoint = use_grad_checkpoint

        # Style mapping + learned constant seed (StyleGAN trick).
        self.mapping = StyleMapping(num_params=num_params, latent_dim=latent_dim, w_dim=w_dim)
        self.const = nn.Parameter(torch.randn(1, schedule[0], init_size, init_size) * 0.02)
        self.seed_block = StyleResBlock(schedule[0], schedule[0], w_dim)

        self.up_stages = nn.ModuleList(
            [StyleUpStage(schedule[k], schedule[k + 1], w_dim) for k in range(num_stages)]
        )

        self.to_image = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(schedule[-1], img_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, params):
        w = self.mapping(params, noise)
        b = noise.size(0)
        h = self.const.expand(b, -1, -1, -1)
        if self.use_grad_checkpoint and self.training:
            h = ckpt(self.seed_block, h, w, use_reentrant=False)
            for stage in self.up_stages:
                h = ckpt(stage, h, w, use_reentrant=False)
        else:
            h = self.seed_block(h, w)
            for stage in self.up_stages:
                h = stage(h, w)
        img = self.to_image(h)
        return img


# ---------------------------------------------------------------------------
# Physics head (deterministic, heteroscedastic regressor)
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """Random Gaussian Fourier features. Maps R^n -> R^(2*num_freqs).

    These features dramatically improve the MLP's ability to fit non-linear
    physical responses (Tancik et al., NeurIPS 2020).
    """

    def __init__(self, num_inputs: int, num_freqs: int = 16, sigma: float = 5.0):
        super().__init__()
        B = torch.randn(num_inputs, num_freqs) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2.0 * math.pi * x @ self.B   # (B, num_freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class _ResMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        return x + self.lin2(self.drop(self.act(self.lin1(self.norm(x)))))


class PhysicsHead(nn.Module):
    """Deterministic params -> (mu, log_var) heteroscedastic regressor.

    Inputs are first passed through Fourier features, then a deep ResMLP.
    Outputs two vectors per sample of length num_outputs: predicted mean
    and predicted log-variance per output. The training script consumes
    both via a Gaussian NLL loss.
    """

    def __init__(
        self,
        num_params: int = 4,
        num_outputs: int = 3,
        hidden: int = 256,
        depth: int = 6,
        num_freqs: int = 16,
        sigma: float = 5.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.ff = FourierFeatures(num_params, num_freqs=num_freqs, sigma=sigma)
        feat_dim = 2 * num_freqs + num_params  # concat raw inputs to keep linear path
        self.input_proj = nn.Linear(feat_dim, hidden)
        self.blocks = nn.ModuleList([_ResMLPBlock(hidden, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.out_mu = nn.Linear(hidden, num_outputs)
        self.out_log_var = nn.Linear(hidden, num_outputs)
        # Clamp log-variance to a sane range to prevent collapse during training.
        self.log_var_min = -10.0
        self.log_var_max = 4.0

    def forward(self, params):
        feat = torch.cat([self.ff(params), params], dim=-1)
        h = self.input_proj(feat)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        mu = self.out_mu(h)
        log_var = self.out_log_var(h).clamp(self.log_var_min, self.log_var_max)
        return mu, log_var


# ---------------------------------------------------------------------------
# Discriminator (SN-PatchGAN + minibatch stddev)
# ---------------------------------------------------------------------------

class MinibatchStdDev(nn.Module):
    """Append a per-batch std statistic as an extra feature map.
    Standard StyleGAN/ProGAN trick to discourage mode collapse."""

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        b, c, h, w = x.shape
        gs = min(self.group_size, b)
        if gs <= 1:
            extra = torch.zeros(b, 1, h, w, device=x.device, dtype=x.dtype)
        else:
            # Pad to a multiple of gs if needed
            pad = (gs - b % gs) % gs
            if pad:
                x_pad = torch.cat([x, x[:pad]], dim=0)
                b_pad = b + pad
            else:
                x_pad = x
                b_pad = b
            y = x_pad.view(gs, b_pad // gs, c, h, w)
            y = y - y.mean(dim=0, keepdim=True)
            y = (y ** 2).mean(dim=0)
            y = (y + 1e-8).sqrt()
            y = y.mean(dim=[1, 2, 3], keepdim=True)
            y = y.repeat(gs, 1, h, w)[:b]
            extra = y.unsqueeze(1).squeeze(2)
            if extra.dim() == 5:
                extra = extra.squeeze(2)
        return torch.cat([x, extra], dim=1)


class Discriminator(nn.Module):
    """SN-PatchGAN with parameter conditioning broadcast as extra channels,
    and a minibatch-stddev layer near the output."""

    def __init__(
        self,
        num_params: int = 4,
        img_channels: int = 4,
        img_size: int = 1024,
        base_channels: int = 64,
        num_downsamples: int = 5,
        param_embed_channels: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_params = num_params
        self.param_embed_channels = param_embed_channels

        self.param_embed = nn.Sequential(
            nn.Linear(num_params, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, param_embed_channels * 8 * 8),
        )

        def sn_conv(in_c, out_c, kernel=4, stride=2, padding=1):
            return spectral_norm(nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding))

        layers = []
        in_c = img_channels + param_embed_channels
        c = base_channels
        for _ in range(num_downsamples):
            layers.append(sn_conv(in_c, c, stride=2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = c
            c = min(c * 2, 512)

        self.body = nn.Sequential(*layers)
        self.mb_std = MinibatchStdDev(group_size=4)
        self.tail = nn.Sequential(
            sn_conv(in_c + 1, in_c, kernel=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(in_c, 1, kernel=4, stride=1, padding=1),
        )

    def forward(self, image, params):
        b = image.size(0)
        emb = self.param_embed(params).view(b, self.param_embed_channels, 8, 8)
        emb = F.interpolate(emb, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = torch.cat([image, emb], dim=1)
        h = self.body(x)
        h = self.mb_std(h)
        return self.tail(h)


# ---------------------------------------------------------------------------
# Differentiable augmentation (DiffAug) for small-data GANs
# ---------------------------------------------------------------------------

def _rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)

def _rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2.0) + mean

def _rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + mean

def _rand_translation(x, ratio=0.125):
    b, c, h, w = x.shape
    shift_x = int(h * ratio + 0.5)
    shift_y = int(w * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, (b, 1, 1), device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, (b, 1, 1), device=x.device)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    grid_y = grid_y.unsqueeze(0) + ty
    grid_x = grid_x.unsqueeze(0) + tx
    grid_y = grid_y.clamp(0, h - 1)
    grid_x = grid_x.clamp(0, w - 1)
    out = x.new_zeros(x.shape)
    for i in range(b):
        out[i] = x[i, :, grid_y[i].squeeze(0), grid_x[i].squeeze(0)]
    return out

def _rand_cutout(x, ratio=0.25):
    b, c, h, w = x.shape
    cut = int(h * ratio + 0.5)
    cy = torch.randint(0, h, (b,), device=x.device)
    cx = torch.randint(0, w, (b,), device=x.device)
    mask = torch.ones_like(x)
    for i in range(b):
        y1 = (cy[i] - cut // 2).clamp(0, h)
        y2 = (cy[i] + cut // 2).clamp(0, h)
        x1 = (cx[i] - cut // 2).clamp(0, w)
        x2 = (cx[i] + cut // 2).clamp(0, w)
        mask[i, :, y1:y2, x1:x2] = 0.0
    return x * mask


def diff_augment(x, policy="color,translation,cutout"):
    """Differentiable Augmentation (Zhao et al. NeurIPS 2020) for GAN training."""
    if not policy:
        return x
    if "color" in policy:
        x = _rand_brightness(x)
        x = _rand_saturation(x)
        x = _rand_contrast(x)
    if "translation" in policy:
        x = _rand_translation(x)
    if "cutout" in policy:
        x = _rand_cutout(x)
    return x


# ---------------------------------------------------------------------------
# EMA helper (re-used for G and PhysicsHead)
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_generator_checkpoint(model_path, device, config: dict):
    state_dict = _load_state_dict(model_path, device)
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
    gen.load_state_dict(state_dict, strict=True)
    gen.eval()
    return gen


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = Generator(img_size=1024, init_size=16, init_channels=512, use_grad_checkpoint=False).to(device)
    disc = Discriminator(img_size=1024).to(device)
    head = PhysicsHead(num_params=4, num_outputs=3).to(device)

    n_g = sum(p.numel() for p in gen.parameters())
    n_d = sum(p.numel() for p in disc.parameters())
    n_p = sum(p.numel() for p in head.parameters())
    print(f"Generator parameters:    {n_g/1e6:.2f} M")
    print(f"Discriminator parameters:{n_d/1e6:.2f} M")
    print(f"PhysicsHead parameters:  {n_p/1e6:.2f} M")

    z = torch.randn(1, 100, device=device)
    p = torch.randn(1, 4, device=device)
    img = gen(z, p)
    mu, log_var = head(p)
    d_out = disc(img, p)
    print("Image shape:                ", tuple(img.shape))
    print("PhysicsHead mu shape:       ", tuple(mu.shape))
    print("PhysicsHead log_var shape:  ", tuple(log_var.shape))
    print("Discriminator patch shape:  ", tuple(d_out.shape))


if __name__ == "__main__":
    main()
