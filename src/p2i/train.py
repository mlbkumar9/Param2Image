"""Train the Param2Image v3 field-map surrogate.

DESIGN
------
* Targets are the 4-channel physical field maps on the FE grid, not rendered
  screenshots, so error is reported in MPa / strain / mm.
* THREE-way split. The test partition is never read here: not for model
  selection, not for early stopping, not for scaling statistics. Checkpoint
  selection uses the validation split only.
* Every resolved hyper-parameter is written into the checkpoint, so the
  configuration table in the paper is generated FROM the checkpoint instead of
  being transcribed by hand.
* The physics head has its own loss (L_phys) and its own checkpoint.
* Scalers are fitted on the training split only.

USAGE
    python -m p2i.train --csv results_master.csv --pack fields_pack.npy \
                        --out runs/v36

The defaults reproduce the reported run (add --cache-ram --num-workers 8 for
speed on a large machine).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import model as M
from .data import (CHANNEL_NAMES, FieldMapDataset, MASK_CHANNEL, PARAM_COLS,
                  TARGET_COLS,
                  U3_CHANNEL, hard_intact_peak_mm, intact_mask_from_norm,
                  measure_channel_weights, soft_peak,
                  TabularScaler, channel_spec_dict)
from .metrics import (channel_physical_error, image_scalar_consistency,
                     impact_position_error_mm, impact_position_loss,
                     ms_ssim, position_error_px, position_loss, px_to_mm,
                     ssim)


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pack", default="")
    ap.add_argument("--out", default="runs/v36")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=200)
    ap.add_argument("--init-size", type=int, default=25)
    ap.add_argument("--init-channels", type=int, default=512)
    ap.add_argument("--w-dim", type=int, default=512)
    ap.add_argument("--latent-dim", type=int, default=100)
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=4e-5)
    ap.add_argument("--lr-phys", type=float, default=5e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--lambda-l1", type=float, default=10.0)
    ap.add_argument("--lambda-pos", type=float, default=10.0,
                    help="weight on the impact-position term. Raising it to 100 did "
                         "not improve impact error (3.82 mm against 3.85 mm at 10) "
                         "and cost image fidelity (SSIM 0.9584 to 0.9468).")
    ap.add_argument("--lambda-phys", type=float, default=1.0)
    ap.add_argument("--lambda-peak", type=float, default=15.0,
                    help="weight on the peak terms. A peak is a single extremum that "
                         "a field-averaged L1 barely constrains, yet it is what the "
                         "paper reports; supervising it directly is what stopped the "
                         "generated deflection field being well-shaped but wrongly "
                         "scaled. The term averages the supervised channels, so 15 "
                         "keeps each channel's influence at the single-channel 5.0.")
    ap.add_argument("--peak-temperature", type=float, default=20.0,
                    help="sharpness of the differentiable max used by --lambda-peak")
    ap.add_argument("--peak-channels", default="mises,peeq,u3",
                    help="comma-separated channels given a direct peak loss")
    ap.add_argument("--pos-target", choices=["truth", "real-image"], default="truth",
                    help="'truth' scores the generated impact against the recorded "
                         "impact coordinates. 'real-image' matches one soft centroid "
                         "to another, which inherits that locator's 40.6 mm "
                         "ground-truth error.")
    ap.add_argument("--phys-loss", choices=["huber", "mse", "nll"], default="huber",
                    help="mean-fitting loss for the physics head. A plain "
                         "heteroscedastic NLL underfits: validation deflection R2 "
                         "0.885 against 0.955 for Huber.")
    ap.add_argument("--phys-num-freqs", type=int, default=0,
                    help="random Fourier features in the physics head. Was 16 at "
                         "sigma 5.0 in the earliest head and was the largest cause of "
                         "its underfitting; 0 disables them.")
    ap.add_argument("--phys-raw-params", action="store_true",
                    help="feed the head the four raw parameters instead of the "
                         "nine physics-informed features")
    ap.add_argument("--phys-epochs", type=int, default=100,
                    help="stop updating the physics head after this many epochs "
                         "(0 = train it for the full run; default 100). The head shares no weights "
                         "with the generator and converges far sooner: at batch 32 it "
                         "peaked near epoch 60-100 and then slid from mean R2 0.847 to "
                         "0.746 by epoch 320. ckpt_best_phys.pt is selected "
                         "independently of the generator.")
    ap.add_argument("--phys-wd", type=float, default=1e-3,
                    help="weight decay for the physics head, previously hardcoded at "
                         "1e-3. The head sees the same 90k updates as the generator "
                         "but converges in roughly 60-100 epochs and then degrades: "
                         "measured at batch 32, mean R2 peaks near 0.847 and falls to "
                         "0.746 by epoch 320. ckpt_best_phys.pt is what protects the "
                         "reported numbers from that, since the two branches share no "
                         "weights and are selected independently.")
    ap.add_argument("--phys-mixup", type=float, default=0.0,
                    help="mixup alpha for the physics head. Blending targets of a "
                         "deterministic nonlinear response is label corruption; "
                         "measured to cost ~0.06 R2, so it defaults to off.")
    ap.add_argument("--lambda-adv", type=float, default=1.0,
                    help="weight on the adversarial term. It was hardcoded at "
                         "1.0 against lambda_l1 = 10, so the generator is "
                         "mostly a regression model and the discriminator "
                         "winning is the expected steady state, not a fault. "
                         "0 removes the term entirely; the target is already "
                         "low-pass filtered, so the high-frequency realism a "
                         "discriminator supplies is largely the band that was "
                         "deliberately filtered out.")
    ap.add_argument("--d-exclude-mask", action="store_true",
                    help="hide the mask channel from the discriminator. Ground "
                         "truth masks are exactly +-1 in 100%% of cells while a "
                         "tanh generator cannot be, so this one channel "
                         "separates every real from every fake for free. The "
                         "mask stays supervised by L1 and the peak terms.")
    ap.add_argument("--spatial-cond", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="give the generator a coordinate and "
                         "distance-to-impact field at every stage. AdaIN is "
                         "spatially uniform per channel, so without this the "
                         "network has to synthesise the impact location out of "
                         "channel statistics. Zero initialised. On by default; "
                         "--no-spatial-cond disables it.")
    ap.add_argument("--lambda-msssim", type=float, default=0.0,
                    help="weight on (1 - MS-SSIM) over the channels named by "
                         "--msssim-channels. SSIM is the reported metric but "
                         "the objective was pure L1, which optimises each cell "
                         "independently and is indifferent to the local "
                         "covariance SSIM actually measures. 0 disables.")
    ap.add_argument("--msssim-channels", default="mises",
                    help="comma-separated channel names the MS-SSIM term "
                         "applies to. Default is mises alone: peeq, u3 and "
                         "mask sit within 0.005 of their reproducibility "
                         "ceiling already, so a term on them buys nothing and "
                         "competes for gradient.")
    ap.add_argument("--channel-weights", default="1.93,0.63,0.90,0.53",
                    help="four comma-separated L1 weights (mises,peeq,u3,mask); the "
                         "default is the set measured on the training split. 'auto' "
                         "re-measures inverse mean-absolute-deviation weights on "
                         "the training split; 'none' for a plain unweighted L1.")
    ap.add_argument("--r1-gamma", type=float, default=10.0)
    ap.add_argument("--r1-interval", type=int, default=16)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--diffaug", default="color,cutout")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--smooth", type=int, default=3,
                    help="box-blur the field targets by this many cells "
                         "(0 or 1 disables). Applied to the training target, the "
                         "validation reference and the evaluation "
                         "reference alike, so no metric compares a "
                         "smoothed prediction with a raw reference.")
    ap.add_argument("--smooth-mask", action="store_true",
                    help="also smooth the mask channel. Off by default: a "
                         "non-binary mask makes the perforation area "
                         "non-integral, and the mask ceiling is already "
                         "0.984 unsmoothed.")
    ap.add_argument("--cache-ram", action="store_true",
                    help="hold the packed field maps in RAM instead of reading the "
                         "memmap per sample (~1.8 GB for train+val). Removes the "
                         "input pipeline as the throughput limit; identical maths.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--patience", type=int, default=120,
                    help="stop when validation SSIM has not improved for N epochs "
                         "(0 disables). Makes --epochs a ceiling, not a guess.")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="improvement below this does not reset patience")
    ap.add_argument("--min-epochs", type=int, default=150,
                    help="never early-stop before this epoch")
    ap.add_argument("--lr-schedule", choices=["cosine", "plateau"], default="cosine",
                    help="cosine anneals over --epochs; plateau halves the LR when "
                         "validation SSIM stalls and needs no epoch guess")
    ap.add_argument("--lr-patience", type=int, default=10,
                    help="epochs of no improvement before plateau halves the LR")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="bf16 on Ampere+ (A100): fp32 exponent range, so no gradient "
                         "scaling and far fewer NaN blow-ups than fp16 in GAN training")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the generator and discriminator "
                         "(first epoch pays a compilation cost)")
    ap.add_argument("--channels-last", action="store_true",
                    help="NHWC memory format; faster convolutions on tensor cores")
    ap.add_argument("--limit-steps", type=int, default=0,
                    help="stop each epoch after N optimiser steps "
                         "(benchmark / smoke-test only)")
    ap.add_argument("--mirror", default="",
                    help="durable but slow dir (e.g. a network drive) to mirror into")
    ap.add_argument("--mirror-every", type=int, default=10,
                    help="mirror the full resume checkpoint every N epochs")
    return ap.parse_args()


def cosine_lr(base, epoch, total, warmup, lr_min):
    if epoch < warmup:
        return base * (epoch + 1) / max(warmup, 1)
    t = (epoch - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (base - lr_min) * (1 + np.cos(np.pi * t))


@torch.no_grad()
def evaluate(G, P, loader, device, scaler, latent_dim):
    G.eval()
    P.eval()
    agg = {}
    n = 0
    for params, real, y in loader:
        params = params.to(device)
        real = real.to(device)
        y = y.to(device)
        b = params.size(0)
        z = torch.randn(b, latent_dim, device=device)
        fake = G(z, params).clamp(-1, 1)
        mu, _ = P(params)

        s_all = ssim(fake, real)
        s_ch = ssim(fake, real, per_channel=True)
        pe = position_error_px(fake, real)
        # Impact position against the coordinates the run was given. The legacy
        # pos_err_px compares two soft centroids and never sees them.
        pm = torch.tensor(scaler.param_mean, dtype=params.dtype, device=device)
        ps = torch.tensor(scaler.param_std, dtype=params.dtype, device=device)
        txy = params[:, 2:4] * ps[2:4] + pm[2:4]
        ie = impact_position_error_mm(fake, txy)
        ie_real = impact_position_error_mm(real, txy)

        rec = {"ssim": s_all.sum().item(), "pos_err_px": pe.sum().item(),
               "impact_err_mm": ie.sum().item(),
               "impact_floor_mm": ie_real.sum().item()}
        for i, nm in enumerate(CHANNEL_NAMES):
            rec["ssim_" + nm] = s_ch[:, i].sum().item()
        for k, v in channel_physical_error(fake, real).items():
            rec[k] = v * b
        for k, v in image_scalar_consistency(fake, mu, scaler).items():
            rec[k] = v * b

        pred_phys = scaler.inv_targets(mu)
        true_phys = scaler.inv_targets(y)
        rel = (pred_phys - true_phys).abs() / true_phys.abs().clamp(min=1e-6)
        for i, nm in enumerate(["mises", "peeq", "defl"]):
            rec["mape_" + nm] = 100 * rel[:, i].sum().item()

        # Deflection read off the GENERATED image with the erosion rule that
        # defines the target, compared against the true scalar. This is the
        # number that exposed the old representation: the field can score SSIM
        # 0.98 while its peak is uncorrelated with the truth.
        pk = hard_intact_peak_mm(fake)
        good = torch.isfinite(pk) & (true_phys[:, 2].abs() > 0.1)
        rec["img_defl_mape"] = (100 * (pk[good] - true_phys[good, 2]).abs()
                                / true_phys[good, 2].abs()).sum().item()
        rec["img_defl_n"] = float(good.sum().item())

        for k, v in rec.items():
            agg[k] = agg.get(k, 0.0) + v
        n += b

    out = {k: v / max(n, 1) for k, v in agg.items()}
    nd = agg.get("img_defl_n", 0.0)
    out["img_defl_mape"] = agg.get("img_defl_mape", 0.0) / max(nd, 1.0)
    out["img_defl_n"] = nd
    out["pos_err_mm"] = px_to_mm(out["pos_err_px"])
    G.train()
    P.train()
    return out


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    df = pd.read_csv(args.csv)
    scaler = TabularScaler.fit(df[df.split == "train"])

    ds = {s: FieldMapDataset(args.csv, s, scaler, args.pack or None,
                             cache_ram=args.cache_ram,
                             smooth=args.smooth,
                             smooth_mask=args.smooth_mask)
          for s in ("train", "val")}
    if args.smooth > 1:
        print("field targets box-smoothed by %d cells (%.0f mm); mask %s"
              % (args.smooth, args.smooth * 3.0,
                 "smoothed" if args.smooth_mask else "left binary"))
    if args.cache_ram:
        gb = sum(d.cache.numel() * d.cache.element_size() for d in ds.values()
                 if d.cache is not None) / 1e9
        print("field maps cached in RAM: %.2f GB" % gb)
    n_test = int((df.split == "test").sum())
    print("train %d   val %d   (test held out: %d)" % (len(ds["train"]), len(ds["val"]), n_test))

    dl_tr = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                       drop_last=True, persistent_workers=args.num_workers > 0)
    dl_va = DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # ---- per-channel reconstruction weights --------------------------------
    # An unweighted L1 over the four channels is not neutral: the binary mask
    # carries ~90 % of the total signal and the two sparse physical channels
    # under 3 % between them, so the generator learns the mask and ignores the
    # deflection field. Weighting by inverse mean absolute deviation equalises
    # the contributions. Measured on the training split only, and stored in the
    # checkpoint so the number behind every result is recoverable.
    msssim_chans = [CHANNEL_NAMES.index(n.strip())
                    for n in args.msssim_channels.split(",") if n.strip()]
    if args.lambda_msssim > 0:
        print("MS-SSIM term on %s, weight %.3g"
              % (", ".join(CHANNEL_NAMES[i] for i in msssim_chans),
                 args.lambda_msssim))

    if args.channel_weights == "none":
        ch_w = [1.0] * len(CHANNEL_NAMES)
    elif args.channel_weights == "auto":
        ch_w = measure_channel_weights(dl_tr)
    else:
        ch_w = [float(v) for v in args.channel_weights.split(",")]
        if len(ch_w) != len(CHANNEL_NAMES):
            raise SystemExit("--channel-weights needs %d values" % len(CHANNEL_NAMES))
    print("channel L1 weights: " + "  ".join(
        "%s %.3f" % (n, w) for n, w in zip(CHANNEL_NAMES, ch_w)))
    ch_w_t = torch.tensor(ch_w, device=device).view(1, -1, 1, 1)

    # Channels given a direct peak loss, resolved once from their names.
    peak_chans = [CHANNEL_NAMES.index(c.strip()) for c in args.peak_channels.split(",")
                  if c.strip()]
    print("peak loss on: %s" % ", ".join(CHANNEL_NAMES[i] for i in peak_chans)
          if peak_chans else "peak loss: disabled")

    # Constants for recovering raw impact coordinates from standardised params,
    # so the position loss can be scored against the coordinates the simulation
    # was actually given rather than against another centroid.
    p_mean = torch.tensor(scaler.param_mean, dtype=torch.float32, device=device)
    p_std = torch.tensor(scaler.param_std, dtype=torch.float32, device=device)
    print("position target: %s" % args.pos_target)

    G = M.Generator(num_params=len(PARAM_COLS), num_numerical_outputs=len(TARGET_COLS),
                    latent_dim=args.latent_dim, img_channels=len(CHANNEL_NAMES),
                    img_size=args.img_size, init_size=args.init_size,
                    init_channels=args.init_channels, w_dim=args.w_dim,
                    use_grad_checkpoint=False,
                    spatial_cond=args.spatial_cond,
                    param_mean=scaler.param_mean,
                    param_std=scaler.param_std).to(device)
    if args.spatial_cond:
        print("spatial conditioning on: impact grid added at every stage")
    d_channels = [i for i in range(len(CHANNEL_NAMES))
                  if not (args.d_exclude_mask and i == MASK_CHANNEL)]
    if args.d_exclude_mask:
        print("discriminator sees %s (mask hidden: it is exactly +-1 in truth "
              "and cannot be in a tanh output)"
              % ", ".join(CHANNEL_NAMES[i] for i in d_channels))
    if args.lambda_adv != 1.0:
        print("adversarial weight %.3g" % args.lambda_adv)
    D = M.Discriminator(num_params=len(PARAM_COLS), img_channels=len(d_channels),
                        img_size=args.img_size).to(device)
    P = M.PhysicsHead(num_params=len(PARAM_COLS), num_outputs=len(TARGET_COLS),
                      hidden=256, depth=6, num_freqs=args.phys_num_freqs, sigma=5.0,
                      use_physics_features=not args.phys_raw_params).to(device)
    # The head expands its inputs using kinetic energy and boundary distance, which
    # needs RAW values; it receives standardised ones, so it carries the constants
    # to undo that. Fitted on the TRAINING split only and stored in the state_dict.
    P.param_mean.copy_(torch.tensor(scaler.param_mean, dtype=torch.float32, device=device))
    P.param_std.copy_(torch.tensor(scaler.param_std, dtype=torch.float32, device=device))
    P.set_feature_stats(torch.tensor(
        df[df.split == "train"][PARAM_COLS].to_numpy(np.float32), device=device))
    ema_g = M.EMA(G, args.ema_decay)
    ema_p = M.EMA(P, args.ema_decay)

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr_g, betas=(0.0, 0.99), weight_decay=args.wd)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.0, 0.99), weight_decay=args.wd)
    opt_p = torch.optim.AdamW(P.parameters(), lr=args.lr_phys, betas=(0.9, 0.999),
                              weight_decay=args.phys_wd)

    # ---- precision ---------------------------------------------------------
    # bf16 carries fp32's exponent range, so it needs no loss scaling and is far
    # less prone to the inf/NaN collapses fp16 causes in adversarial training.
    # fp16 keeps a GradScaler; fp32 disables autocast entirely.
    use_cuda = device.type == "cuda"
    prec = "fp32" if (args.no_amp or not use_cuda) else args.precision
    if prec == "bf16" and use_cuda and not torch.cuda.is_bf16_supported():
        print("bf16 unsupported on this GPU -> falling back to fp16")
        prec = "fp16"
    amp = prec in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if prec == "bf16" else torch.float16
    if use_cuda:
        # TF32 tensor cores for the fp32 matmul/conv paths (Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print("precision: %s%s" % (prec, "" if use_cuda else " (cpu)"))

    # A GradScaler is only meaningful for fp16; disabled it is a transparent
    # pass-through, so the call sites below stay identical for all three modes.
    try:
        scaler_amp = torch.amp.GradScaler("cuda", enabled=(prec == "fp16"))
    except TypeError:
        scaler_amp = torch.cuda.amp.GradScaler(enabled=(prec == "fp16"))

    if args.channels_last and use_cuda:
        G = G.to(memory_format=torch.channels_last)
        D = D.to(memory_format=torch.channels_last)

    if args.compile and use_cuda:
        try:
            G = torch.compile(G)
            D = torch.compile(D)
            print("torch.compile enabled (first epoch includes compilation)")
        except Exception as exc:
            print("torch.compile unavailable (%s); continuing eager" % exc)

    config = dict(vars(args))
    config.update(channels=channel_spec_dict(args.smooth, args.smooth_mask),
                  scaler=scaler.to_dict(),
                  channel_l1_weights=ch_w, peak_channels=peak_chans,
                  msssim_channels=msssim_chans,
                  phys_head=dict(hidden=256, depth=6, num_freqs=args.phys_num_freqs,
                                 sigma=5.0,
                                 use_physics_features=not args.phys_raw_params),
                  param_cols=PARAM_COLS, target_cols=TARGET_COLS,
                  n_train=len(ds["train"]), n_val=len(ds["val"]), n_test=n_test,
                  params_generator=sum(p.numel() for p in G.parameters()),
                  params_discriminator=sum(p.numel() for p in D.parameters()),
                  params_physics=sum(p.numel() for p in P.parameters()))
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    # Read the value back rather than trust the write: this field is what
    # evaluate.py uses to build its references, and a stale 0 here silently
    # scores a smoothed model against raw truth.
    _rec = config["channels"]["smooth_cells"]
    if int(_rec) != int(args.smooth or 0):
        raise SystemExit("checkpoint would record smooth_cells %s for a run "
                         "trained with --smooth %s" % (_rec, args.smooth))

    start_epoch = 0
    best = -1.0
    best_phys_resume, best_phys_epoch_resume = None, 0
    stale_resume, best_epoch_resume = 0, 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ck["G"])
        D.load_state_dict(ck["D"])
        P.load_state_dict(ck["P"])
        ema_g.load_state_dict(ck["ema_g"])
        ema_p.load_state_dict(ck["ema_p"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        opt_p.load_state_dict(ck["opt_p"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best_ssim", -1.0)
        # Carry the physics head's own best across as well. Without this it reset
        # to infinity on every reconnect, so the first validated epoch after a
        # disconnect always counted as "a new best" and overwrote
        # ckpt_best_phys.pt with whatever head happened to be current. The SSIM
        # best was already carried; this one was not, so a session that
        # drops once silently discarded the best head of the run.
        best_phys_resume = ck.get("best_phys", None)
        best_phys_epoch_resume = int(ck.get("best_phys_epoch", 0))
        # The early-stopping counter needs the same treatment. Left at zero on
        # resume it restarts the patience window, so a run that reconnects can
        # sail past the stopping point: the reported run recorded its last SSIM
        # improvement at epoch 347 with patience 120 and should have halted near
        # 467, but reconnected and continued to 587.
        stale_resume = int(ck.get("stale", 0))
        best_epoch_resume = int(ck.get("best_epoch", 0))
        print("resumed from %s at epoch %d (best SSIM %.4f%s)"
              % (args.resume, start_epoch, best,
                 "" if best_phys_resume is None else
                 ", best physics MAPE %.2f%% at epoch %d"
                 % (best_phys_resume, best_phys_epoch_resume)))

    # Restore the epoch history on resume. Without this every reconnect would
    # overwrite history.csv with only the epochs since the restart, and a run
    # that disconnects a few times would end with unusable training curves.
    hist = []
    stale = 0          # epochs since validation SSIM last improved
    lr_stale = 0       # same counter, used by the plateau LR schedule
    lr_factor = 1.0    # cumulative decay applied by the plateau schedule
    best_epoch = 0
    best_phys = float("inf")   # physics head has its own overfitting curve
    best_phys_epoch = 0
    if start_epoch > 0 and best_phys_resume is not None:
        best_phys, best_phys_epoch = best_phys_resume, best_phys_epoch_resume
    if start_epoch > 0:
        stale, best_epoch = stale_resume, best_epoch_resume
    hist_path = os.path.join(args.out, "history.csv")
    if start_epoch > 0 and os.path.exists(hist_path):
        try:
            prev = pd.read_csv(hist_path)
            prev = prev[prev.epoch <= start_epoch]
            hist = prev.to_dict("records")
            print("restored %d earlier epochs from history.csv" % len(hist))
        except Exception as exc:
            print("WARNING: could not restore history (%s); starting a new one" % exc)
    gstep = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        # Ramp the auxiliary weights over the LR warmup. At initialisation the
        # impact is ~180 mm out, which puts the position term at ~79% of the
        # generator loss and leaves the adversarial signal under 2%; a few epochs
        # of that distorts GAN dynamics before the term decays on its own. Full
        # strength by --warmup-epochs, constant thereafter.
        w_aux = min(1.0, (epoch + 1) / max(args.warmup_epochs, 1))

        for opt, lr in ((opt_g, args.lr_g), (opt_d, args.lr_d), (opt_p, args.lr_phys)):
            if args.lr_schedule == "cosine":
                v = cosine_lr(lr, epoch, args.epochs, args.warmup_epochs, args.lr_min)
            else:
                # linear warmup then halve-on-plateau: unlike cosine this does
                # not depend on guessing the right epoch budget up front
                warm = min(1.0, (epoch + 1) / max(args.warmup_epochs, 1))
                v = max(lr * warm * lr_factor, args.lr_min)
            for pg in opt.param_groups:
                pg["lr"] = v

        run = {}
        steps_done = 0
        t_steps = time.time()
        for params, real, y in dl_tr:
            params = params.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            b = params.size(0)

            # ---------------- discriminator ----------------
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
                z = torch.randn(b, args.latent_dim, device=device)
                fake = G(z, params).detach()
                d_real = D(M.diff_augment(real[:, d_channels], args.diffaug),
                           params)
                d_fake = D(M.diff_augment(fake[:, d_channels], args.diffaug),
                           params)
                loss_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
            scaler_amp.scale(loss_d).backward()
            scaler_amp.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            scaler_amp.step(opt_d)

            # lazy R1 on real samples
            if gstep % args.r1_interval == 0:
                opt_d.zero_grad(set_to_none=True)
                # R1 must penalise the gradient on exactly the tensor the
                # discriminator consumes, so slice before requires_grad_.
                r = real[:, d_channels].detach().requires_grad_(True)
                out = D(r, params).sum()
                grad = torch.autograd.grad(out, r, create_graph=True)[0]
                r1 = 0.5 * args.r1_gamma * grad.pow(2).sum(dim=(1, 2, 3)).mean() * args.r1_interval
                r1.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
                opt_d.step()
                run["r1"] = run.get("r1", 0.0) + r1.detach()

            # ---------------- generator ----------------
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
                z = torch.randn(b, args.latent_dim, device=device)
                fake = G(z, params)
                adv = F.softplus(
                    -D(M.diff_augment(fake[:, d_channels], args.diffaug),
                       params)).mean()
                l1 = ((fake - real).abs() * ch_w_t).mean()
                # position term is assigned below according to --pos-target
                # Peak of each supervised channel, read out exactly as its target
                # is defined: max_mises_mpa and max_peeq are plain field maxima,
                # while max_deflection_mm is the peak over material at least two
                # cells from a damage front. Masks come from the REAL sample and
                # carry no gradient, so this trains the generated magnitudes
                # without letting the model cheat through the mask channel.
                peak = fake.sum() * 0.0
                if args.lambda_peak > 0 and peak_chans:
                    with torch.no_grad():
                        im = intact_mask_from_norm(real)
                        ok = im.any(dim=(1, 2))
                        full = torch.ones_like(im)
                    terms = []
                    for ci in peak_chans:
                        m, sel = (im, ok) if ci == U3_CHANNEL else (full, None)
                        pf = soft_peak(fake[:, ci], m, args.peak_temperature)
                        pr = soft_peak(real[:, ci], m, args.peak_temperature)
                        if sel is not None:
                            if not bool(sel.any()):
                                continue
                            pf, pr = pf[sel], pr[sel]
                        terms.append(F.smooth_l1_loss(pf, pr.detach()))
                    if terms:
                        peak = torch.stack(terms).mean()

                # Impact position. The impact coordinates are an INPUT to the
                # generator and are recorded per run, so the exact answer is
                # available; the locator used here is wrong by 1.11 mm on real maps.
                if args.pos_target == "truth":
                    true_xy = params[:, 2:4] * p_std[2:4] + p_mean[2:4]
                    pos = impact_position_loss(fake, true_xy)
                else:
                    pos = position_loss(fake, real)
                # Multi-scale structural term. Restricted to a channel
                # subset because only von Mises has headroom left to win.
                msl = fake.sum() * 0.0
                if args.lambda_msssim > 0 and msssim_chans:
                    msl = (1.0 - ms_ssim(fake[:, msssim_chans],
                                         real[:, msssim_chans],
                                         per_channel=True)).mean()
                loss_g = (args.lambda_adv * adv + args.lambda_l1 * l1
                          + w_aux * args.lambda_pos * pos
                          + w_aux * args.lambda_peak * peak
                          + w_aux * args.lambda_msssim * msl)
            scaler_amp.scale(loss_g).backward()
            scaler_amp.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            scaler_amp.step(opt_g)

            # ------- physics head: heteroscedastic NLL (L_phys, NOT L_pos) -------
            opt_p.zero_grad(set_to_none=True)
            pin, tgt = params, y
            if args.phys_mixup > 0:
                lam = float(np.random.beta(args.phys_mixup, args.phys_mixup))
                perm = torch.randperm(b, device=device)
                pin = lam * params + (1 - lam) * params[perm]
                tgt = lam * y + (1 - lam) * y[perm]
            train_phys = not args.phys_epochs or epoch < args.phys_epochs
            mu, logvar = P(pin)
            nll = M.physics_head_loss(mu, logvar, tgt, args.phys_loss)
            if train_phys:
                (args.lambda_phys * nll).backward()
                torch.nn.utils.clip_grad_norm_(P.parameters(), 5.0)
                opt_p.step()

            scaler_amp.update()
            ema_g.update(G)
            if train_phys:
                ema_p.update(P)
            gstep += 1
            steps_done += 1
            if args.limit_steps and steps_done >= args.limit_steps:
                break
            # Accumulate losses ON DEVICE. Calling .item() here would force a
            # full CPU/GPU synchronisation six times per step, serialising the
            # pipeline; the values are only needed once, at the end of the
            # epoch. Purely a throughput change - the maths is identical.
            for k, v in (("g", loss_g), ("d", loss_d), ("l1", l1),
                         ("pos", pos), ("adv", adv), ("nll", nll), ("peak", peak)):
                run[k] = run.get(k, 0.0) + v.detach()

        # throughput, parsed by wrapper scripts to estimate total runtime
        sec_per_step = (time.time() - t_steps) / max(steps_done, 1)
        print("sec_per_step %.4f  (%d steps)" % (sec_per_step, steps_done), flush=True)
        if args.limit_steps:
            print("benchmark only (--limit-steps %d); stopping" % args.limit_steps)
            return

        nb = max(steps_done, 1)
        tr_log = {k: (v.item() if torch.is_tensor(v) else v) / nb
                  for k, v in run.items()}   # single sync per epoch

        # ---- validate with EMA weights ----
        gb = {k: v.detach().clone() for k, v in G.state_dict().items()}
        pb = {k: v.detach().clone() for k, v in P.state_dict().items()}
        ema_g.copy_to(G)
        ema_p.copy_to(P)
        va = evaluate(G, P, dl_va, device, scaler, args.latent_dim)
        G.load_state_dict(gb)
        P.load_state_dict(pb)

        # Computed here rather than beside its checkpoint below, because the
        # resume payload has to record it and that payload is written first.
        phys_score = (va.get("mape_mises", 0.0) + va.get("mape_peeq", 0.0)
                      + va.get("mape_defl", 0.0)) / 3.0

        rec = {"epoch": epoch + 1, "sec": time.time() - t0}
        rec.update({"tr_" + k: v for k, v in tr_log.items()})
        rec.update({"va_" + k: v for k, v in va.items()})
        hist.append(rec)
        pd.DataFrame(hist).to_csv(os.path.join(args.out, "history.csv"), index=False)
        print("[%3d/%d] %5.0fs  G %6.3f D %5.3f | val SSIM %.4f  impact %.1fmm "
              "(floor %.1f)  MAPE defl %.2f%%  imgDefl %.1f%%"
              % (epoch + 1, args.epochs, rec["sec"], tr_log.get("g", 0),
                 tr_log.get("d", 0), va["ssim"], va.get("impact_err_mm", float("nan")),
                 va.get("impact_floor_mm", float("nan")), va["mape_defl"],
                 va.get("img_defl_mape", float("nan"))), flush=True)

        # ---- checkpointing -------------------------------------------------
        # Two different objects with two different jobs:
        #   ckpt_last  full resume state (weights + EMAs + all three optimizer
        #              states) ~300 MB. Written to fast local disk every epoch.
        #   ckpt_best  EMA weights + config only, ~50 MB. That is all evaluation
        #              and inference need, so it is cheap to mirror immediately.
        # Writing the 300 MB resume file to Drive every epoch would add ~60 GB of
        # traffic over a 200-epoch run and dominate the wall clock, so the full
        # file is mirrored only every --mirror-every epochs (and at the end).
        full = dict(epoch=epoch, G=G.state_dict(), D=D.state_dict(), P=P.state_dict(),
                    ema_g=ema_g.state_dict(), ema_p=ema_p.state_dict(),
                    opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(),
                    opt_p=opt_p.state_dict(),
                    config=config, val=va, best_ssim=max(best, va["ssim"]),
                    best_phys=min(best_phys, phys_score),
                    best_phys_epoch=(epoch + 1 if phys_score < best_phys
                                     else best_phys_epoch),
                    stale=stale, best_epoch=best_epoch)
        torch.save(full, os.path.join(args.out, "ckpt_last.pt"))

        slim = dict(epoch=epoch, ema_g=ema_g.state_dict(), ema_p=ema_p.state_dict(),
                    config=config, val=va, best_ssim=max(best, va["ssim"]))

        # ---- early stopping / plateau bookkeeping ---------------------------
        # --epochs is a CEILING, not a target: training halts once validation
        # SSIM has genuinely stopped improving. Without this a run keeps going
        # long past its best checkpoint, and in adversarial training it can
        # actively degrade once the discriminator starts winning (D loss below
        # 2*log2 = 1.3863).
        # ---- physics head: selected INDEPENDENTLY -------------------------
        # It shares no weights and no gradients with G/D, so it has its own
        # overfitting curve. Selecting it on the generator SSIM takes it well
        # past its own optimum: in the 300-epoch run its deflection MAPE was
        # 15.78% at epoch 103 but 16.40% at epoch 293, where best-SSIM landed.
        if phys_score < best_phys:
            best_phys = phys_score
            best_phys_epoch = epoch + 1
            torch.save(dict(epoch=epoch, ema_p=ema_p.state_dict(), config=config,
                            val=va, phys_score=phys_score),
                       os.path.join(args.out, "ckpt_best_phys.pt"))
            print("    new best physics head (mean scalar MAPE %.2f%%, epoch %d)"
                  % (phys_score, epoch + 1))

        improved = va["ssim"] > best + args.min_delta
        if improved:
            stale = 0
            lr_stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
            lr_stale += 1

        if args.lr_schedule == "plateau" and lr_stale >= args.lr_patience:
            lr_factor *= 0.5
            lr_stale = 0
            print("    val SSIM stalled %d epochs -> learning rate x0.5 (factor %.4f)"
                  % (args.lr_patience, lr_factor))

        # Discriminator-collapse warning: at equilibrium D loss sits at
        # 2*log(2) = 1.3863. A sustained drop below that means D is beating G,
        # which is what precedes late-training degradation in this setup.
        d_now = tr_log.get("d", float("nan"))
        if d_now == d_now and d_now < 1.30:
            print("    NOTE: D loss %.3f is below equilibrium (1.386) - the "
                  "discriminator is winning. If val metrics slip, lower --lr-d "
                  "or raise --r1-gamma." % d_now)

        # `improved` is the EARLY-STOPPING test and carries a min-delta, so it is
        # deliberately insensitive to tiny gains. Whether the best checkpoint was
        # rewritten is a different question, and mirroring on `improved` meant a
        # sub-min-delta improvement updated the local file but not Drive: a
        # disconnect then left Drive holding an older checkpoint than the one the
        # run had actually selected. Mirror on the write itself.
        wrote_best = va["ssim"] > best
        if wrote_best:
            best = va["ssim"]
            if not improved:
                best_epoch = epoch + 1
            full["best_ssim"] = best
            slim["best_ssim"] = best
            torch.save(slim, os.path.join(args.out, "ckpt_best.pt"))
            print("    new best val SSIM %.4f (epoch %d)" % (best, epoch + 1))

        if args.mirror:
            try:
                os.makedirs(args.mirror, exist_ok=True)
                # history is tiny: mirror it every epoch so progress survives a crash
                pd.DataFrame(hist).to_csv(os.path.join(args.mirror, "history.csv"), index=False)
                if wrote_best:
                    torch.save(slim, os.path.join(args.mirror, "ckpt_best.pt"))
                pb = os.path.join(args.out, "ckpt_best_phys.pt")
                if os.path.exists(pb):
                    shutil.copy2(pb, os.path.join(args.mirror, "ckpt_best_phys.pt"))
                last_ep = (epoch + 1) == args.epochs
                if last_ep or (epoch + 1) % max(args.mirror_every, 1) == 0:
                    t_m = time.time()
                    torch.save(full, os.path.join(args.mirror, "ckpt_last.pt"))
                    print("    mirrored resume checkpoint (%.0fs)" % (time.time() - t_m))
            except Exception as exc:
                # a Drive hiccup must never kill a long run: local state is intact
                print("    WARNING: mirror failed (%s); local checkpoints are fine" % exc)

        if args.patience and (epoch + 1) >= args.min_epochs and stale >= args.patience:
            print("\nEARLY STOP: no val-SSIM improvement for %d epochs "
                  "(best %.4f at epoch %d)" % (stale, best, best_epoch))
            break

    print("\ndone. best val SSIM %.4f at epoch %d (ran %d epochs)"
          % (best, best_epoch, epoch + 1))
    print("best physics head: mean scalar MAPE %.2f%% at epoch %d"
          % (best_phys, best_phys_epoch))


if __name__ == "__main__":
    main()
