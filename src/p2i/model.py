"""
Param2Image model: StyleGAN-lite generator + SN-PatchGAN discriminator
                       + dedicated Fourier-featurized physics head.

Produces the four-channel physical field maps and numerical predictions
(peak von Mises, PEEQ, deflection) from impact parameters.

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


# ---------------------------------------------------------------------------
# Spatial conditioning
# ---------------------------------------------------------------------------
# AdaIN is spatially uniform per channel, so a style-modulated stack has to
# encode "the impact is at (x, y)" in channel statistics and decode it back into
# a position. That is the expensive part of this task: the response pattern is
# roughly the same shape wherever it lands, and only its location changes.
# Handing the decoder a distance-to-impact field turns translation from
# something learned into something given.

def impact_grid(xy_cells, radius_cells, size, device, dtype):
    """(B,4,size,size): offset in x, offset in y, distance, projectile footprint.

    xy_cells is the impact position in CELL units measured from the grid centre,
    at full resolution; it is rescaled here to whatever stage is asking.
    """
    b = xy_cells.shape[0]
    lin = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    xx = xx.view(1, 1, size, size).expand(b, 1, size, size)
    yy = yy.view(1, 1, size, size).expand(b, 1, size, size)
    # impact position in the same [-1, 1] frame
    px = xy_cells[:, 0].view(b, 1, 1, 1)
    py = xy_cells[:, 1].view(b, 1, 1, 1)
    dx, dy = xx - px, yy - py
    r = torch.sqrt(dx * dx + dy * dy + 1e-8)
    rad = radius_cells.view(b, 1, 1, 1).clamp(min=1e-3)
    disk = torch.exp(-(r / rad) ** 2)
    return torch.cat([dx, dy, r, disk], dim=1)


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
        spatial_cond: bool = False,
        param_mean=None,
        param_std=None,
    ):
        super().__init__()
        self.spatial_cond = bool(spatial_cond)
        # the de-standardisation constants travel with the weights, so a
        # checkpoint reproduces its own conditioning without the scaler
        pm = torch.zeros(4) if param_mean is None else torch.as_tensor(
            param_mean, dtype=torch.float32)
        ps = torch.ones(4) if param_std is None else torch.as_tensor(
            param_std, dtype=torch.float32)
        self.register_buffer("p_mean", pm)
        self.register_buffer("p_std", ps.clamp(min=1e-6))
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

        # One 1x1 projection of the impact grid per stage, added to the
        # features. Additive rather than concatenated so channel counts, and
        # therefore every other shape in the network, are untouched. Zero
        # initialised, so an untrained model is identical to the old one and
        # the conditioning is something the optimiser switches on if it helps.
        if self.spatial_cond:
            self.cond_proj = nn.ModuleList(
                [nn.Conv2d(4, c, 1) for c in schedule])
            for m in self.cond_proj:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        self.to_image = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(schedule[-1], img_channels, 3, padding=1),
            nn.Tanh(),
        )

    def _cond(self, params):
        """Impact position in [-1,1] image coords and radius in the same units.

        params arrive standardised; de-standardise with the stored buffers.
        Column order is velocity, radius, x, y, and the plate is 600 mm across,
        so half-width 300 mm maps to 1.0. The locator convention is (row, col) =
        (y, x), which is why x indexes the last axis below.
        """
        raw = params[:, :4] * self.p_std + self.p_mean
        xy = raw[:, 2:4] / 300.0
        rad = (raw[:, 1:2] / 300.0).clamp(min=1e-3)
        return xy, rad

    def forward(self, noise, params):
        w = self.mapping(params, noise)
        b = noise.size(0)
        h = self.const.expand(b, -1, -1, -1)
        xy = rad = None
        if self.spatial_cond:
            xy, rad = self._cond(params)
            h = h + self.cond_proj[0](
                impact_grid(xy, rad, h.shape[-1], h.device, h.dtype))
        if self.use_grad_checkpoint and self.training:
            h = ckpt(self.seed_block, h, w, use_reentrant=False)
            for i, stage in enumerate(self.up_stages):
                h = ckpt(stage, h, w, use_reentrant=False)
                if self.spatial_cond:
                    h = h + self.cond_proj[i + 1](
                        impact_grid(xy, rad, h.shape[-1], h.device, h.dtype))
        else:
            h = self.seed_block(h, w)
            for i, stage in enumerate(self.up_stages):
                h = stage(h, w)
                if self.spatial_cond:
                    h = h + self.cond_proj[i + 1](
                        impact_grid(xy, rad, h.shape[-1], h.device, h.dtype))
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

    WHY THE FOURIER FEATURES ARE OFF BY DEFAULT
    -------------------------------------------
    An earlier head projected its four standardised inputs through 16
    random Fourier features at sigma = 5.0. On standardised inputs that is a
    very short length scale, and it dominated the fit: the head was left
    badly underfit while looking converged. Measured on the validation split,
    three seeds, against a RandomForest reference of R2 0.812 / 0.784 / 0.952:

        shipped head (RFF sigma 5, mixup, NLL)   0.720 / 0.713 / 0.834
        same but with the RFF removed            0.813 / 0.787 / 0.949

    Deflection MAPE fell from 15.64 % to 8.83 % on that one change. Widening or
    deepening the network made it worse, which is the signature of a feature
    map that is too high-frequency rather than a capacity shortfall. Fourier
    features remain available (`num_freqs > 0`) but are no longer the default.

    PHYSICS FEATURES
    ----------------
    With `use_physics_features`, the four inputs are expanded to nine using
    kinetic energy and boundary distance (see data.physics_features). No
    information is added; the response simply becomes easier to express. Worth
    a further 0.935 -> 0.949 on deflection R2.

    The expansion needs RAW inputs while the caller supplies standardised ones,
    so the de-standardisation and re-standardisation constants live here as
    buffers. They travel in the state_dict, so a checkpoint is self-contained.
    """

    def __init__(
        self,
        num_params: int = 4,
        num_outputs: int = 3,
        hidden: int = 256,
        depth: int = 6,
        num_freqs: int = 0,
        sigma: float = 5.0,
        dropout: float = 0.05,
        use_physics_features: bool = True,
        detach_var_head: bool = True,
    ):
        super().__init__()
        self.detach_var_head = detach_var_head
        from .data import N_PHYS_FEATURES, physics_features as _pf
        self._pf = staticmethod(_pf).__func__
        self.use_physics_features = use_physics_features
        n_in = N_PHYS_FEATURES if use_physics_features else num_params
        # de-standardise the incoming params, then standardise the expansion
        self.register_buffer("param_mean", torch.zeros(num_params))
        self.register_buffer("param_std", torch.ones(num_params))
        self.register_buffer("feat_mean", torch.zeros(n_in))
        self.register_buffer("feat_std", torch.ones(n_in))

        self.num_freqs = num_freqs
        self.ff = FourierFeatures(n_in, num_freqs=num_freqs, sigma=sigma) if num_freqs else None
        feat_dim = (2 * num_freqs + n_in) if num_freqs else n_in
        self.input_proj = nn.Linear(feat_dim, hidden)
        self.blocks = nn.ModuleList([_ResMLPBlock(hidden, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.out_mu = nn.Linear(hidden, num_outputs)
        self.out_log_var = nn.Linear(hidden, num_outputs)
        # Clamp log-variance to a sane range to prevent collapse during training.
        self.log_var_min = -10.0
        self.log_var_max = 4.0

    @torch.no_grad()
    def set_feature_stats(self, params_raw):
        """Fit the expansion's standardisation on the TRAINING split only."""
        f = self._pf(params_raw) if self.use_physics_features else params_raw
        m, sd = f.mean(0), f.std(0)
        sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
        self.feat_mean.copy_(m.to(self.feat_mean.dtype))
        self.feat_std.copy_(sd.to(self.feat_std.dtype))

    def forward(self, params):
        if self.use_physics_features:
            raw = params * self.param_std + self.param_mean
            x = self._pf(raw)
        else:
            x = params
        x = (x - self.feat_mean) / self.feat_std
        feat = torch.cat([self.ff(x), x], dim=-1) if self.num_freqs else x
        h = self.input_proj(feat)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        mu = self.out_mu(h)
        # The variance head reads a DETACHED trunk. Without this, the NLL term
        # that fits log_var still back-propagates through the shared body and
        # perturbs the mean: measured at 4.0e-03 on the trunk gradients, which
        # is exactly the escape hatch that made a plain heteroscedastic NLL
        # underfit. Detaching leaves log_var free to calibrate while making it
        # provably unable to move the mean.
        hv = h.detach() if self.detach_var_head else h
        log_var = self.out_log_var(hv).clamp(self.log_var_min, self.log_var_max)
        return mu, log_var


def physics_head_loss(mu, log_var, target, kind="huber", beta=0.5):
    """Fit the mean robustly; learn the variance without disturbing the mean.

    A plain heteroscedastic NLL lets the model raise sigma on a hard sample
    instead of fitting it, and measured here it costs 0.07 of deflection R2
    against a Huber fit. But the log-variance is worth keeping: it is what the
    paper reports as predictive uncertainty. So the mean is fitted with Huber,
    and the variance by a Gaussian NLL on a DETACHED residual, with the head
    also reading a detached trunk (see PhysicsHead.forward). Both detachments
    are needed: without the second, the NLL still reaches the mean through the
    shared body, which was measured rather than assumed.

    Validation, three seeds: Huber 0.812 / 0.778 / 0.955 against NLL
    0.813 / 0.797 / 0.885, and Huber is better on all three MAPEs.
    """
    if kind == "nll":
        return (0.5 * torch.exp(-log_var) * (target - mu) ** 2 + 0.5 * log_var).mean()
    if kind == "mse":
        mean_loss = F.mse_loss(mu, target)
    else:
        mean_loss = F.smooth_l1_loss(mu, target, beta=beta)
    resid = (target - mu).detach() ** 2
    var_loss = (0.5 * torch.exp(-log_var) * resid + 0.5 * log_var).mean()
    return mean_loss + 0.1 * var_loss


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
