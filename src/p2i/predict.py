"""Predict the four physical field maps and the scalar peaks for one impact.

The generator is stochastic, so the reported estimate is the mean of several
latent draws (8 by default), taken in normalised space and then decoded to
physical units.

USAGE
    python -m p2i.predict --ckpt ckpt_best.pt --ckpt-phys ckpt_best_phys.pt \\
        --velocity-kmh 40 --radius-mm 20 --x-mm 0 --y-mm 0 \\
        --draws 8 --out pred.npz [--png]

pred.npz holds `fields` (4, 200, 200) in physical units (von Mises in MPa, PEEQ
dimensionless, |U3| in mm, surviving-material mask 0/1), `channel_names`,
`channel_units`, the physics head's `peak_mises_mpa`, `peak_peeq`,
`peak_deflection_mm`, and the same peaks read off the generated image.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from . import model as M
from .data import (CHANNEL_NAMES, CHANNEL_UNITS, PARAM_COLS, TARGET_COLS,
                   TabularScaler, norm_to_fields)
from .metrics import image_peaks

DRAWS = 8


def load_generator(ckpt_path, device="cpu"):
    """The EMA generator, built to the architecture recorded in its checkpoint."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    sc = TabularScaler.from_dict(cfg["scaler"])
    G = M.Generator(num_params=len(PARAM_COLS), num_numerical_outputs=len(TARGET_COLS),
                    latent_dim=cfg["latent_dim"], img_channels=len(CHANNEL_NAMES),
                    img_size=cfg["img_size"], init_size=cfg["init_size"],
                    init_channels=cfg["init_channels"], w_dim=cfg["w_dim"],
                    use_grad_checkpoint=False,
                    spatial_cond=bool(cfg.get("spatial_cond", False)),
                    param_mean=sc.param_mean, param_std=sc.param_std)
    em = M.EMA(G)
    em.load_state_dict(ck["ema_g"])
    em.copy_to(G)
    G.eval().to(device)
    return G, cfg, sc, ck


def load_physics_head(ckpt_path, device="cpu"):
    """The EMA physics head. `ckpt_path` may be ckpt_best_phys.pt or a full checkpoint."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    ph = cfg.get("phys_head", {})
    sc = TabularScaler.from_dict(cfg["scaler"])
    P = M.PhysicsHead(num_params=len(PARAM_COLS), num_outputs=len(TARGET_COLS),
                      hidden=ph.get("hidden", 256), depth=ph.get("depth", 6),
                      num_freqs=ph.get("num_freqs", 0),
                      sigma=ph.get("sigma", 5.0),
                      use_physics_features=ph.get("use_physics_features", True))
    merged = dict(P.state_dict())
    merged.update(ck["ema_p"])
    P.load_state_dict(merged, strict=True)
    P.eval().to(device)
    return P, cfg, sc, ck


@torch.no_grad()
def predict(G, params, latent_dim, draws=DRAWS):
    """Conditional-mean field for a (B,4) standardised parameter batch: mean of `draws` samples."""
    b = params.shape[0]
    acc = None
    for _ in range(draws):
        z = torch.randn(b, latent_dim, device=params.device)
        out = G(z, params).clamp(-1, 1)
        acc = out if acc is None else acc + out
    return (acc / draws).clamp(-1, 1)


def _save_pngs(fields, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for i, (name, unit) in enumerate(zip(CHANNEL_NAMES, CHANNEL_UNITS)):
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(fields[i], origin="lower", cmap="viridis")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label=unit)
        path = "%s_%s.png" % (stem, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="generator checkpoint (ckpt_best.pt)")
    ap.add_argument("--ckpt-phys", default="",
                    help="physics-head checkpoint (ckpt_best_phys.pt); default: "
                         "the head stored in --ckpt")
    ap.add_argument("--velocity-kmh", type=float, required=True)
    ap.add_argument("--radius-mm", type=float, required=True)
    ap.add_argument("--x-mm", type=float, required=True)
    ap.add_argument("--y-mm", type=float, required=True)
    ap.add_argument("--draws", type=int, default=DRAWS,
                    help="latent draws averaged per case")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="pred.npz")
    ap.add_argument("--png", action="store_true",
                    help="also write one PNG per channel next to --out")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    G, cfg, sc, _ = load_generator(args.ckpt, device)
    P, _, psc, _ = load_physics_head(args.ckpt_phys or args.ckpt, device)

    raw = [args.velocity_kmh, args.radius_mm, args.x_mm, args.y_mm]
    params = torch.from_numpy(sc.tx_params(raw)).unsqueeze(0).to(device)
    norm = predict(G, params, cfg["latent_dim"], args.draws)
    fields = norm_to_fields(norm)[0].cpu().numpy()
    img_peaks = image_peaks(norm)[0].cpu().numpy()
    with torch.no_grad():
        mu, _ = P(torch.from_numpy(psc.tx_params(raw)).unsqueeze(0).to(device))
    peaks = psc.inv_targets(mu)[0].cpu().numpy()

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, fields=fields,
             channel_names=np.array(CHANNEL_NAMES),
             channel_units=np.array(CHANNEL_UNITS),
             params=np.array(raw, dtype=np.float64),
             param_names=np.array(PARAM_COLS),
             peak_mises_mpa=float(peaks[0]), peak_peeq=float(peaks[1]),
             peak_deflection_mm=float(peaks[2]),
             image_peak_mises_mpa=float(img_peaks[0]),
             image_peak_peeq=float(img_peaks[1]),
             image_peak_deflection_mm=float(img_peaks[2]),
             draws=args.draws)
    print("wrote %s" % out)
    print("physics-head peaks: mises %.1f MPa, peeq %.3f, deflection %.2f mm"
          % tuple(peaks))
    if args.png:
        for p in _save_pngs(fields, os.path.splitext(out)[0]):
            print("wrote %s" % p)


if __name__ == "__main__":
    main()
