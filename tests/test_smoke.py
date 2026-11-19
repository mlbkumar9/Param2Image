"""End-to-end smoke test on a tiny synthetic dataset (CPU, no real data)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from p2i import __version__, evaluate, predict, train
from p2i import model as M
from p2i.data import GRID, PARAM_COLS, TARGET_COLS, fields_to_norm

SMALL = ["--init-channels", "16", "--w-dim", "16", "--latent-dim", "8"]


def _synthetic_dataset(root, n=24, seed=0):
    rng = np.random.default_rng(seed)
    rows, fields = [], []
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    for i in range(n):
        v = rng.uniform(50, 800)
        r = rng.uniform(25, 100)
        x, y = rng.uniform(-200, 200, 2)
        cx, cy = GRID / 2 + x / 3.0, GRID / 2 + y / 3.0
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        bump = np.exp(-d2 / (2 * 20.0 ** 2))
        phys = np.stack([1000 * bump * v / 800, 1.5 * bump * r / 100,
                         40 * bump * v / 800,
                         (d2 > 9.0).astype(np.float32)]).astype(np.float32)
        fields.append(fields_to_norm(phys).astype(np.float16))
        rows.append(dict(run_id="RUN_%03d" % (i + 1),
                         split="train" if i < 14 else ("val" if i < 19 else "test"),
                         velocity_kmh=v, proj_radius_mm=r, impact_x_mm=x,
                         impact_y_mm=y, max_mises_mpa=float(phys[0].max()),
                         max_peeq=float(phys[1].max()),
                         max_deflection_mm=float(phys[2].max()),
                         perforated=int(i % 2), field_npz="fields/RUN_%03d.npz" % (i + 1)))
    csv = os.path.join(root, "results_master.csv")
    pack = os.path.join(root, "fields_pack.npy")
    pd.DataFrame(rows).to_csv(csv, index=False)
    np.save(pack, np.stack(fields))
    pd.DataFrame({"run_id": [r["run_id"] for r in rows]}).to_csv(
        os.path.join(root, "fields_pack_index.csv"), index=False)
    return csv, pack


def _run(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", [module.__name__] + [str(a) for a in argv])
    module.main()


def test_version():
    assert __version__ == "3.6.0"


def test_model_forward_shapes():
    G = M.Generator(num_params=4, num_numerical_outputs=3, latent_dim=8,
                    img_channels=4, img_size=GRID, init_size=25,
                    init_channels=16, w_dim=16, use_grad_checkpoint=False,
                    spatial_cond=True, param_mean=[0.0] * 4, param_std=[1.0] * 4)
    D = M.Discriminator(num_params=4, img_channels=4, img_size=GRID)
    P = M.PhysicsHead(num_params=4, num_outputs=3)
    p = torch.randn(2, 4)
    img = G(torch.randn(2, 8), p)
    assert img.shape == (2, 4, GRID, GRID)
    mu, logvar = P(p)
    assert mu.shape == (2, 3) and logvar.shape == (2, 3)
    assert D(img, p).shape[0] == 2


def test_train_evaluate_predict(tmp_path, monkeypatch):
    csv, pack = _synthetic_dataset(str(tmp_path))
    run = str(tmp_path / "run")
    _run(monkeypatch, train, ["--csv", csv, "--pack", pack, "--out", run,
                              "--epochs", 2, "--batch-size", 4, "--num-workers", 0,
                              "--min-epochs", 1, "--phys-epochs", 1] + SMALL)
    for f in ("config.json", "history.csv", "ckpt_last.pt", "ckpt_best.pt",
              "ckpt_best_phys.pt"):
        assert os.path.exists(os.path.join(run, f)), f
    cfg = json.load(open(os.path.join(run, "config.json")))
    assert cfg["spatial_cond"] is True and cfg["smooth"] == 3

    ev = str(tmp_path / "eval")
    _run(monkeypatch, evaluate, ["--csv", csv, "--pack", pack,
                                 "--ckpt", os.path.join(run, "ckpt_best.pt"),
                                 "--ckpt-phys", os.path.join(run, "ckpt_best_phys.pt"),
                                 "--split", "test", "--out", ev, "--no-fid",
                                 "--draws", 2, "--batch-size", 4])
    assert os.path.exists(os.path.join(ev, "results.json"))

    out = str(tmp_path / "pred" / "pred.npz")
    _run(monkeypatch, predict, ["--ckpt", os.path.join(run, "ckpt_best.pt"),
                                "--ckpt-phys", os.path.join(run, "ckpt_best_phys.pt"),
                                "--velocity-kmh", 300, "--radius-mm", 50,
                                "--x-mm", 10, "--y-mm", -20, "--draws", 2,
                                "--out", out, "--png"])
    z = np.load(out)
    assert z["fields"].shape == (4, GRID, GRID)
    assert np.isfinite(z["fields"]).all()
    assert float(z["peak_mises_mpa"]) >= 0
    for ch in ("mises", "peeq", "u3", "mask"):
        assert os.path.exists(str(tmp_path / "pred" / ("pred_%s.png" % ch)))
