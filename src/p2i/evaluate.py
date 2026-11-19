"""Single source of truth for every number reported in the paper.

Reviewer 1 asked which checkpoint, subset, resolution, preprocessing, filter and
metric definition produced each result, and required that all figures, tables
and abstract values be regenerated from ONE evaluation script and ONE model.
This is that script. It:

  * loads exactly one checkpoint and uses its stored config (channel ranges,
    scaler, architecture) -- nothing is re-derived or re-typed;
  * evaluates on the HELD-OUT TEST split by default, which was never touched
    during training or checkpoint selection;
  * states every filter it applies, and reports the count each filter removed;
  * reports image metrics (SSIM overall and per channel) and physical metrics
    (MAPE, R2 in MPa / strain / mm) on the same samples;
  * breaks error down by impact regime, answering the "no ablation isolates the
    source of error" criticism directly;
  * cross-checks the image branch against the scalar branch, which are
    architecturally independent, by decoding peak values from the generated
    field through the invertible channel mapping;
  * reports KID as well as FID, because FID is strongly biased for the sample
    counts available here.

Everything lands in one JSON plus CSVs, so the manuscript can cite a file
rather than a remembered number.

USAGE
    python evaluate.py --ckpt runs/v3/ckpt_best.pt \
                       --csv data/Impact_Dataset_60/results_master.csv \
                       --pack data/Impact_Dataset_60/fields_pack.npy \
                       --split test --out results/eval_v3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import model as M
from .data import (CELL_MM, CHANNEL_NAMES, FieldMapDataset,
                  PARAM_COLS, TARGET_COLS, TARGET_UNITS, TabularScaler)
from .metrics import (image_peaks, impact_position_error_mm,
                     position_error_px, px_to_mm, ssim)

SCALAR_NAMES = ["mises", "peeq", "deflection"]


def r2_score(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mape(y_true, y_pred, floor):
    """MAPE with an explicitly stated magnitude floor (documented, not hidden)."""
    keep = np.abs(y_true) > floor
    if not keep.any():
        return float("nan"), 0, len(y_true)
    v = 100.0 * np.mean(np.abs((y_pred[keep] - y_true[keep]) / y_true[keep]))
    return float(v), int(keep.sum()), int((~keep).sum())


# --------------------------------------------------------------------- FID/KID
def _inception_features(batches, device):
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
    except Exception as exc:
        return None, "torchvision unavailable: %s" % exc
    try:
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    except Exception as exc:
        return None, "inception weights unavailable offline: %s" % exc
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    feats = []
    with torch.no_grad():
        for x in batches:
            # deterministic RGB rendering of the three physical channels
            rgb = x[:, :3].clamp(-1, 1)
            rgb = torch.nn.functional.interpolate(rgb, size=(299, 299),
                                                  mode="bilinear", align_corners=False)
            feats.append(net(rgb.to(device)).cpu().numpy())
    return np.concatenate(feats), None


def frechet_distance(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False)
    s2 = np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kernel_distance(a, b, n_subsets=100, subset_size=None, seed=0):
    """Unbiased KID (polynomial kernel MMD^2), appropriate at small n."""
    rng = np.random.RandomState(seed)
    n = min(len(a), len(b))
    subset_size = subset_size or min(100, n)
    d = a.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), subset_size, replace=False)]
        y = b[rng.choice(len(b), subset_size, replace=False)]
        kxx = (x.dot(x.T) / d + 1) ** 3
        kyy = (y.dot(y.T) / d + 1) ** 3
        kxy = (x.dot(y.T) / d + 1) ** 3
        m = subset_size
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


_SMOOTH_OVERRIDE = None


def _smooth_from_cfg(cfg):
    """Smoothing width to score against: the override, else the checkpoint's.

    Falls back to the top-level `smooth` argument the run was launched with
    when the channels block says 0, which is how checkpoints written before
    the provenance fix look. Without that fallback those checkpoints score
    against raw references while claiming to honour the recorded width.
    """
    ch = cfg.get("channels", {}) if isinstance(cfg, dict) else {}
    rec = int(ch.get("smooth_cells", 0) or 0)
    if not rec and isinstance(cfg, dict):
        rec = int(cfg.get("smooth", 0) or 0)
    if _SMOOTH_OVERRIDE is not None and _SMOOTH_OVERRIDE >= 0:
        rec = int(_SMOOTH_OVERRIDE)
    return rec, bool(ch.get("smooth_mask", False))


def _retrieval_baseline(args, cfg, scaler, meta, device):
    """Per-channel SSIM between each evaluated case and its nearest TRAINING case.

    Nearest in standardised 4-D parameter space. Both maps are real
    simulations, so this measures how much two neighbouring solutions differ,
    which is the ceiling a parameter-conditioned model is working against.
    """
    try:
        full = pd.read_csv(args.csv)
        tr = full[full.split == "train"].reset_index(drop=True)
        if not len(tr):
            return {"status": "not computed", "reason": "no training split in csv"}
        mu = np.asarray(scaler.param_mean, np.float64)
        sd = np.asarray(scaler.param_std, np.float64)
        Xtr = (tr[PARAM_COLS].to_numpy(np.float64) - mu) / sd
        Xte = (meta[PARAM_COLS].to_numpy(np.float64) - mu) / sd
        d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        nn = d.argmin(1)

        _sm, _smm = _smooth_from_cfg(cfg)
        ds_te = FieldMapDataset(args.csv, args.split if args.split != "all" else None,
                                scaler, args.pack or None,
                                smooth=_sm, smooth_mask=_smm)
        ds_tr = FieldMapDataset(args.csv, "train", scaler, args.pack or None,
                                smooth=_sm, smooth_mask=_smm)
        vals, per_ch = [], []
        for i in range(0, len(meta), 32):
            sl = range(i, min(i + 32, len(meta)))
            A = torch.stack([ds_te[j][1] for j in sl]).to(device)
            B = torch.stack([ds_tr[int(nn[j])][1] for j in sl]).to(device)
            with torch.no_grad():
                vals.append(ssim(A, B).cpu().numpy())
                per_ch.append(ssim(A, B, per_channel=True).cpu().numpy())
        v = np.concatenate(vals)
        c = np.concatenate(per_ch)
        gap = np.sqrt(d.min(1))
        return {"ssim_mean": float(v.mean()),
                "ssim_per_channel": {n: float(c[:, k].mean())
                                     for k, n in enumerate(CHANNEL_NAMES)},
                "param_gap_z_median": float(np.median(gap)),
                "median_dv_kmh": float(np.median(np.abs(
                    meta.velocity_kmh.values - tr.velocity_kmh.values[nn]))),
                "median_dpos_mm": float(np.median(np.hypot(
                    meta.impact_x_mm.values - tr.impact_x_mm.values[nn],
                    meta.impact_y_mm.values - tr.impact_y_mm.values[nn]))),
                "note": ("SSIM between each evaluated case and the nearest TRAINING "
                         "case, both real simulations. The model must beat this to "
                         "be doing more than retrieval.")}
    except Exception as exc:
        return {"status": "not computed", "reason": str(exc)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-phys", default="",
                    help="take the physics head from this checkpoint instead of "
                         "--ckpt, keeping the generator from --ckpt. The two "
                         "branches share no weights and no gradients, so their best "
                         "epochs differ; ckpt_best_phys.pt stores only the head and "
                         "cannot be passed as --ckpt on its own.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pack", default="")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--defl-floor", type=float, default=0.1,
                    help="MAPE magnitude floor for deflection (mm)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--draws", type=int, default=8,
                    help="average this many latent draws per case and score "
                         "the mean (1 = single sample). "
                         "The generator is stochastic, so a single draw is not "
                         "its best estimate; the mean is. Saturates near 8. "
                         "Report the value used alongside any score.")
    ap.add_argument("--smooth", type=int, default=-1,
                    help="override the smoothing width recorded in the "
                         "checkpoint; -1 (default) uses the recorded value. "
                         "Scoring a smoothed-target model against raw "
                         "references is a valid comparison but a different "
                         "one, so it has to be asked for explicitly.")
    args = ap.parse_args()

    global _SMOOTH_OVERRIDE
    _SMOOTH_OVERRIDE = args.smooth

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    scaler = TabularScaler.from_dict(cfg["scaler"])

    G = M.Generator(num_params=len(PARAM_COLS), num_numerical_outputs=len(TARGET_COLS),
                    latent_dim=cfg["latent_dim"], img_channels=len(CHANNEL_NAMES),
                    img_size=cfg["img_size"], init_size=cfg["init_size"],
                    init_channels=cfg["init_channels"], w_dim=cfg["w_dim"],
                    use_grad_checkpoint=False,
                    spatial_cond=bool(cfg.get("spatial_cond", False)),
                    param_mean=scaler.param_mean,
                    param_std=scaler.param_std).to(device)
    # The architecture has to match the checkpoint, not the current default.
    # Rebuilding an unconditioned generator for a conditioned checkpoint would
    # drop the projections and load with strict=False somewhere downstream,
    # scoring a crippled model without saying so.
    if bool(cfg.get("spatial_cond", False)):
        print("checkpoint uses spatial conditioning; generator built to match")
    # Take the head's shape from the checkpoint. Older runs stored no
    # phys_head block and used 16 Fourier features on the raw parameters, so
    # those defaults reproduce them and old checkpoints stay loadable.
    # The head's architecture must come from the checkpoint that supplies its
    # WEIGHTS, not from --ckpt. The two branches share nothing, so pairing a
    # generator from one run with a head from another is legitimate and useful,
    # and those runs may have different head designs.
    # Reading the shape from the wrong config makes that pairing fail to load.
    phys_ck = ck
    if args.ckpt_phys:
        phys_ck = torch.load(args.ckpt_phys, map_location=device, weights_only=False)
    ph = phys_ck.get("config", cfg).get("phys_head", {})
    P = M.PhysicsHead(num_params=len(PARAM_COLS), num_outputs=len(TARGET_COLS),
                      hidden=ph.get("hidden", 256), depth=ph.get("depth", 6),
                      num_freqs=ph.get("num_freqs", 16), sigma=ph.get("sigma", 5.0),
                      use_physics_features=ph.get("use_physics_features", False)
                      ).to(device)
    def _load_ema(module, shadow, what):
        """Copy EMA weights in, tolerating buffers added after the checkpoint.

        PhysicsHead gained four standardisation buffers. An older checkpoint
        has no entry for them, and a strict load fails outright. Their defaults
        (mean 0, std 1) reproduce the old behaviour exactly when physics
        features are off, which is how a legacy checkpoint is reconstructed, so
        filling them from the fresh module is exact rather than approximate.
        """
        cur = module.state_dict()
        unknown = [k for k in shadow if k not in cur]
        if unknown:
            raise KeyError("%s checkpoint has %d unexpected key(s): %s"
                           % (what, len(unknown), unknown[:4]))
        merged = dict(cur)
        merged.update(shadow)
        added = [k for k in cur if k not in shadow]
        if added:
            print("%s: checkpoint predates %d buffer(s) (%s); using defaults"
                  % (what, len(added), ", ".join(added[:4])))
        module.load_state_dict(merged, strict=True)

    # EMA weights are what the paper reports
    _load_ema(G, ck["ema_g"], "generator")
    phys_src = phys_ck
    if args.ckpt_phys:
        print("physics head from %s (epoch %d); generator from %s (epoch %d)"
              % (os.path.basename(args.ckpt_phys), phys_src["epoch"] + 1,
                 os.path.basename(args.ckpt), ck["epoch"] + 1))
    _load_ema(P, phys_src["ema_p"], "physics head")
    G.eval(); P.eval()

    _sm, _smm = _smooth_from_cfg(cfg)
    if int(getattr(args, "draws", 1)) > 1:
        print("averaging %d latent draws per case (conditional mean)"
              % args.draws)
    print("scoring against %s references"
          % ("raw (unsmoothed)" if _sm <= 1 else "%d-cell smoothed" % _sm))
    if _sm > 1:
        print("evaluating against %d-cell smoothed references (mask %s)"
              % (_sm, "smoothed" if _smm else "binary"))
    ds = FieldMapDataset(args.csv, None if args.split == "all" else args.split,
                         scaler, args.pack or None,
                         smooth=_sm, smooth_mask=_smm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    meta = ds.df.copy()

    ssims, ssims_ch, pos_px = [], [], []
    pred_scalars, true_scalars, img_peaks, real_peaks = [], [], [], []
    pos_true_fake, pos_true_real = [], []
    # DataLoader(shuffle=False) walks ds.df in order, and meta is a copy of it,
    # so a running offset lines each batch up with its recorded coordinates.
    # An index arithmetic shortcut would break on the final partial batch.
    true_xy_all = meta[["impact_x_mm", "impact_y_mm"]].to_numpy(np.float32)
    seen = 0
    real_batches, fake_batches = [], []

    with torch.no_grad():
        for params, real, y in dl:
            params, real = params.to(device), real.to(device)
            n_draw = max(1, int(getattr(args, "draws", 1)))
            if n_draw == 1:
                z = torch.randn(params.size(0), cfg["latent_dim"], device=device)
                fake = G(z, params).clamp(-1, 1)
            else:
                acc = None
                for _ in range(n_draw):
                    z = torch.randn(params.size(0), cfg["latent_dim"],
                                    device=device)
                    out = G(z, params).clamp(-1, 1)
                    acc = out if acc is None else acc + out
                fake = (acc / n_draw).clamp(-1, 1)
            mu, _ = P(params)

            # Position against the coordinates the simulation was GIVEN, which
            # is the only ground truth here. This locator is wrong by 1.11 mm on real maps.
            nb = params.size(0)
            txy = torch.as_tensor(true_xy_all[seen:seen + nb], device=device)
            seen += nb
            pos_true_fake.append(impact_position_error_mm(fake, txy).cpu().numpy())
            pos_true_real.append(impact_position_error_mm(real, txy).cpu().numpy())

            ssims.append(ssim(fake, real).cpu().numpy())
            ssims_ch.append(ssim(fake, real, per_channel=True).cpu().numpy())
            pos_px.append(position_error_px(fake, real).cpu().numpy())

            pred_scalars.append(scaler.inv_targets(mu).cpu().numpy())
            true_scalars.append(scaler.inv_targets(y).numpy())
            img_peaks.append(image_peaks(fake).cpu().numpy())
            real_peaks.append(image_peaks(real).cpu().numpy())
            real_batches.append(real.cpu())
            fake_batches.append(fake.cpu())

    ssims = np.concatenate(ssims)
    ssims_ch = np.concatenate(ssims_ch)
    pos_px = np.concatenate(pos_px)
    pred = np.concatenate(pred_scalars)
    true = np.concatenate(true_scalars)
    peaks = np.concatenate(img_peaks)
    rpeaks = np.concatenate(real_peaks)

    R = {"checkpoint": os.path.abspath(args.ckpt),
         "epoch_selected": int(ck["epoch"]) + 1,
         "selected_on": "validation SSIM (test split never used for selection)",
         "physics_head_checkpoint": (os.path.abspath(args.ckpt_phys)
                                     if args.ckpt_phys else "same as checkpoint"),
         "physics_head_epoch": int(phys_src["epoch"]) + 1,
         "split_evaluated": args.split,
         "n_samples": int(len(ssims)),
         "resolution": "%dx%d (one pixel per finite element)" % (cfg["img_size"], cfg["img_size"]),
         "cell_mm": CELL_MM,
         "weights": "EMA",
         "params_generator_M": cfg["params_generator"] / 1e6,
         "params_discriminator_M": cfg["params_discriminator"] / 1e6,
         "params_physics_M": cfg["params_physics"] / 1e6}

    R["ssim_mean"] = float(ssims.mean())
    R["ssim_median"] = float(np.median(ssims))
    R["ssim_std"] = float(ssims.std())
    R["ssim_min"] = float(ssims.min())
    R["ssim_max"] = float(ssims.max())
    R["ssim_per_channel"] = {n: float(ssims_ch[:, i].mean()) for i, n in enumerate(CHANNEL_NAMES)}

    ptf = np.concatenate(pos_true_fake)
    ptr = np.concatenate(pos_true_real)
    R["impact_position"] = {
        "error_mm_mean": float(ptf.mean()),
        "error_mm_median": float(np.median(ptf)),
        "error_mm_p90": float(np.percentile(ptf, 90)),
        "ground_truth_floor_mm_mean": float(ptr.mean()),
        "measured_against": "impact_x_mm / impact_y_mm recorded for each run",
        "locator": "centroid of (missing material + (u3/max)^8)",
        "note": ("ground_truth_floor is the same locator applied to REAL field maps; "
                 "it bounds what this metric can resolve. The legacy figure below "
                 "compares two soft centroids of the deflection channel and never "
                 "checks either against the recorded coordinates, which is why it "
                 "carries a 40.6 mm floor of its own.")}
    R["position_error_px_mean"] = float(pos_px.mean())
    R["position_error_mm_mean"] = float(px_to_mm(pos_px.mean()))
    R["position_error_mm_median"] = float(px_to_mm(np.median(pos_px)))
    R["position_error_note"] = "LEGACY metric (centroid vs centroid); prefer impact_position"
    R["position_conversion"] = "exact: 1 cell = %.3f mm (600 mm plate / 200 cells)" % CELL_MM

    floors = {"mises": 1.0, "peeq": 1e-3, "deflection": args.defl_floor}
    R["scalar"] = {}
    for i, nm in enumerate(SCALAR_NAMES):
        m, kept, drop = mape(true[:, i], pred[:, i], floors[nm])
        R["scalar"][nm] = {"unit": TARGET_UNITS[i], "mape_percent": m,
                           "r2": r2_score(true[:, i], pred[:, i]),
                           "mae": float(np.abs(pred[:, i] - true[:, i]).mean()),
                           "mape_floor": floors[nm],
                           "n_used_for_mape": kept, "n_excluded_by_floor": drop,
                           "true_range": [float(true[:, i].min()), float(true[:, i].max())]}

    # ---- image branch vs scalar branch (architecturally independent) ----
    # Three numbers per quantity, because one alone is not interpretable:
    #   floor        the same readout applied to GROUND-TRUTH maps against the
    #                stored scalars. It must be ~0. If it is not, the metric is
    #                measuring its own definition rather than the model, which
    #                is exactly what happened before the erosion rule was
    #                matched: the deflection floor was 197 %.
    #   vs_truth     the generated field's own physical accuracy, independent
    #                of the physics head.
    #   mape_percent the cross-branch agreement itself.
    R["image_scalar_consistency"] = {}
    for i, nm in enumerate(SCALAR_NAMES):
        ok = np.isfinite(peaks[:, i]) & np.isfinite(rpeaks[:, i])
        den_h = np.clip(np.abs(pred[:, i]), 1e-6, None)
        den_t = np.clip(np.abs(true[:, i]), 1e-6, None)
        R["image_scalar_consistency"][nm] = {
            "mape_percent": float(100 * np.mean(np.abs(peaks[ok, i] - pred[ok, i]) / den_h[ok])),
            "corr": float(np.corrcoef(peaks[ok, i], pred[ok, i])[0, 1]),
            "vs_truth_mape_percent": float(100 * np.mean(np.abs(peaks[ok, i] - true[ok, i]) / den_t[ok])),
            "vs_truth_corr": float(np.corrcoef(peaks[ok, i], true[ok, i])[0, 1]),
            "ground_truth_floor_mape_percent": float(
                100 * np.mean(np.abs(rpeaks[ok, i] - true[ok, i]) / den_t[ok])),
            "n": int(ok.sum())}
    R["image_scalar_consistency"]["_note"] = (
        "ground_truth_floor is the same readout applied to real field maps against "
        "their own stored scalars and must be near zero; a non-zero floor means the "
        "readout and the target are not the same quantity.")

    # ---- nearest-neighbour retrieval baseline ----
    # The honest question is not 'is SSIM high' but 'is it higher than simply
    # returning the closest simulation already in the training set'. This also
    # bounds what any parameter-conditioned model can reach: two genuinely
    # simulated cases at this parameter spacing do not agree perfectly either.
    R["retrieval_baseline"] = _retrieval_baseline(args, cfg, scaler, meta, device)

    # ---- per-regime breakdown ----
    meta = meta.reset_index(drop=True)
    meta["ssim"] = ssims
    for i, nm in enumerate(CHANNEL_NAMES):
        meta["ssim_" + nm] = ssims_ch[:, i]
    meta["pos_err_mm"] = px_to_mm(pos_px)
    meta["impact_err_mm"] = ptf
    meta["impact_err_floor_mm"] = ptr
    for i, nm in enumerate(SCALAR_NAMES):
        meta["imgpeak_" + nm] = peaks[:, i]
    for i, nm in enumerate(SCALAR_NAMES):
        meta["pred_" + nm] = pred[:, i]
        meta["true_" + nm] = true[:, i]
    meta["vband"] = pd.cut(meta.velocity_kmh, [0, 30, 100, 200, 400, 600, 10000],
                           labels=["0-30", "30-100", "100-200", "200-400", "400-600", "600+"])

    rows = []
    for key, grp in meta.groupby("vband", observed=True):
        if not len(grp):
            continue
        m, _, _ = mape(grp.true_deflection.values, grp.pred_deflection.values, args.defl_floor)
        rows.append({"regime": "velocity " + str(key), "n": len(grp),
                     "ssim": grp.ssim.mean(), "pos_err_mm": grp.pos_err_mm.mean(),
                     "mape_deflection": m})
    for key, grp in meta.groupby("perforated"):
        m, _, _ = mape(grp.true_deflection.values, grp.pred_deflection.values, args.defl_floor)
        rows.append({"regime": "perforated" if key else "non-perforating", "n": len(grp),
                     "ssim": grp.ssim.mean(), "pos_err_mm": grp.pos_err_mm.mean(),
                     "mape_deflection": m})
    regime = pd.DataFrame(rows)
    regime.to_csv(os.path.join(args.out, "per_regime.csv"), index=False)
    R["per_regime"] = rows

    # ---- FID / KID ----
    if not args.no_fid:
        fr, err = _inception_features(real_batches, device)
        if fr is None:
            R["fid_kid"] = {"status": "not computed", "reason": err}
        else:
            ff, _ = _inception_features(fake_batches, device)
            kid_m, kid_s = kernel_distance(fr, ff)
            R["fid_kid"] = {
                "fid": frechet_distance(fr, ff),
                "kid_mean": kid_m, "kid_std": kid_s,
                "n": int(len(fr)),
                "note": ("Computed on a deterministic RGB rendering of the mises/PEEQ/"
                         "deflection channels. FID is strongly biased below ~2000 samples, "
                         "so KID (unbiased for small n) is reported alongside and should be "
                         "preferred at this sample count.")}
    else:
        R["fid_kid"] = {"status": "skipped (--no-fid)"}

    meta.to_csv(os.path.join(args.out, "per_sample.csv"), index=False)
    def _jsonable(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=_jsonable)

    # ---- console summary ----
    print("=" * 74)
    print("Param2Image v3 - evaluation on the %s split (%d samples)" % (args.split, R["n_samples"]))
    print("checkpoint: %s   epoch %d   EMA weights" % (args.ckpt, R["epoch_selected"]))
    print("=" * 74)
    print("SSIM      mean %.4f  median %.4f  sd %.4f  range [%.3f, %.3f]"
          % (R["ssim_mean"], R["ssim_median"], R["ssim_std"], R["ssim_min"], R["ssim_max"]))
    print("  per channel: " + "  ".join("%s %.4f" % (n, v) for n, v in R["ssim_per_channel"].items()))
    ip = R["impact_position"]
    print("impact position  %.2f mm mean, %.2f mm median  (locator floor on real "
          "maps %.2f mm)" % (ip["error_mm_mean"], ip["error_mm_median"],
                            ip["ground_truth_floor_mm_mean"]))
    print("legacy centroid-vs-centroid  %.2f px = %.1f mm  (exact %.3f mm/cell)"
          % (R["position_error_px_mean"], R["position_error_mm_mean"], CELL_MM))
    print("\nscalar responses:")
    for nm, v in R["scalar"].items():
        print("  %-11s MAPE %7.2f%%  R2 %6.3f  MAE %10.4f %-4s  (floor %.3g, %d used / %d excluded)"
              % (nm, v["mape_percent"], v["r2"], v["mae"], v["unit"],
                 v["mape_floor"], v["n_used_for_mape"], v["n_excluded_by_floor"]))
    print("\nimage vs scalar branch agreement (independent branches)")
    print("  %-11s %10s %10s %10s %8s"
          % ("", "floor", "vs truth", "vs head", "corr"))
    for nm, v in R["image_scalar_consistency"].items():
        if nm.startswith("_"):
            continue
        print("  %-11s %9.3f%% %9.2f%% %9.2f%% %8.3f"
              % (nm, v["ground_truth_floor_mape_percent"], v["vs_truth_mape_percent"],
                 v["mape_percent"], v["vs_truth_corr"]))
    print("  floor = the same readout on GROUND-TRUTH maps against their own stored")
    print("  scalars. It must be near zero; anything else means the readout and the")
    print("  target are not the same quantity and the other columns are not usable.")

    rb = R.get("retrieval_baseline", {})
    if "ssim_mean" in rb:
        print("\nnearest-neighbour retrieval baseline (both maps are real simulations):")
        print("  overall SSIM %.4f   vs model %.4f   %s"
              % (rb["ssim_mean"], R["ssim_mean"],
                 "model ahead by %+.4f" % (R["ssim_mean"] - rb["ssim_mean"])
                 if R["ssim_mean"] > rb["ssim_mean"] else "MODEL DOES NOT BEAT RETRIEVAL"))
        print("  per channel: " + "  ".join(
            "%s %.4f" % (k, v) for k, v in rb["ssim_per_channel"].items()))
        print("  nearest training case differs by %.1f km/h and %.1f mm in position (median)"
              % (rb["median_dv_kmh"], rb["median_dpos_mm"]))
    elif rb:
        print("\nretrieval baseline: %s" % rb.get("reason", rb.get("status")))

    print("\nper regime:")
    print(regime.to_string(index=False, float_format=lambda x: "%.3f" % x))
    fk = R["fid_kid"]
    if "fid" in fk:
        print("\nFID %.2f   KID %.5f +/- %.5f  (n=%d)" % (fk["fid"], fk["kid_mean"], fk["kid_std"], fk["n"]))
    else:
        print("\nFID/KID: %s" % fk.get("reason", fk.get("status")))
    print("\nwritten to %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
