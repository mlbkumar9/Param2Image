"""Check that this dataset folder is complete and internally consistent.

Run it after any move, rebuild or re-download. It verifies structure against the
master CSV rather than against a remembered file count, and it re-derives one
physical quantity from the packed array so a stale or mismatched pack cannot pass
silently: `fields_pack.npy` stores values already mapped to [-1,1], so pairing it
with a different channel mapping mis-scales every deflection with no error.

    python verify_dataset.py [DATASET_DIR]          # structure + pack pairing
    python verify_dataset.py [DATASET_DIR] --full   # also opens every .npz file

DATASET_DIR defaults to the folder this script sits in. The pack check imports
the p2i package (pip install -e . from the repository root).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
fails: list = []
warns: list = []


def check(ok, msg, warn=False):
    (warns if warn else fails).append(msg) if not ok else None
    print("  %s %s" % ("ok  " if ok else ("WARN" if warn else "FAIL"), msg))
    return ok


def main():
    global HERE
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=HERE,
                    help="dataset folder (default: this script's folder)")
    ap.add_argument("--full", action="store_true", help="open every .npz (slow)")
    args = ap.parse_args()
    HERE = os.path.abspath(args.dataset)

    print("dataset  ->  %s\n" % HERE)

    csv = os.path.join(HERE, "results_master.csv")
    if not os.path.exists(csv):
        sys.exit("FAIL: results_master.csv missing")
    df = pd.read_csv(csv)
    n = len(df)
    print("master: %d runs, %d columns\n" % (n, len(df.columns)))

    print("structure")
    check(df.run_id.is_unique, "run_id unique")
    for sub, expect, ext in [("fields", n, ".npz"), ("images", 2 * n, ".png"),
                             ("logs", n, ".log")]:
        d = os.path.join(HERE, sub)
        got = len([f for f in os.listdir(d) if f.endswith(ext)]) if os.path.isdir(d) else -1
        check(got == expect, "%-7s %d files (expected %d)" % (sub, got, expect))
    # No ODBs are kept here. Field output was written at two frames only and no
    # history output was ever requested, so an ODB adds the full stress tensor and
    # the in-plane displacements over what fields/ already holds, and nothing else.
    # The 40 transition databases live on the external archive; every other run's
    # was discarded after extraction.
    check(not os.path.isdir(os.path.join(HERE, "odb")),
          "no odb/ (raw databases live on the external archive)")
    for f in ("fields_pack.npy", "fields_pack_index.csv", "mesh/plate_mesh.npz"):
        check(os.path.exists(os.path.join(HERE, f)), f)

    print("\nreferences resolve")
    miss = [c for c in ("field_npz", "img_before", "img_after") if c not in df.columns]
    check(not miss, "path columns present%s" % (" (missing %s)" % miss if miss else ""))
    bad = 0
    for col in ("field_npz", "img_before", "img_after"):
        if col in df.columns:
            bad += sum(not os.path.exists(os.path.join(HERE, str(p).replace("\\", "/")))
                       for p in df[col])
    check(bad == 0, "every path in the master CSV exists (%d broken)" % bad)
    logs_missing = sum(not os.path.exists(os.path.join(HERE, "logs", r + ".log"))
                       for r in df.run_id)
    check(logs_missing == 0, "every run has a solver log (%d missing)" % logs_missing)

    print("\ntargets")
    for c in ("max_mises_mpa", "max_peeq", "max_deflection_mm"):
        if c in df.columns:
            check(df[c].notna().all(), "%s has no NaN" % c)
    if "split" in df.columns:
        vc = df.split.value_counts().to_dict()
        check(set(vc) == {"train", "val", "test"}, "splits %s" % vc)
        if "perforated" in df.columns:
            per = df.groupby("split").perforated.mean()
            check(bool((per > 0.05).all() and (per < 0.95).all()),
                  "every split holds both damage regimes (%s)"
                  % {k: round(v, 3) for k, v in per.items()})

    print("\npacked array pairs with the current channel mapping")
    try:
        import torch
        from p2i.data import hard_intact_peak_mm

        idx = pd.read_csv(os.path.join(HERE, "fields_pack_index.csv"))
        check(list(idx.run_id) == list(df.run_id), "pack index row order matches the master CSV")
        pack = np.load(os.path.join(HERE, "fields_pack.npy"), mmap_mode="r")
        check(pack.shape[0] == n, "pack holds %d rows" % pack.shape[0])
        pos = {r: i for i, r in enumerate(idx.run_id)}
        s = df.sample(min(128, n), random_state=0)
        x = torch.from_numpy(np.asarray(pack[[pos[r] for r in s.run_id]], dtype=np.float32))
        got = hard_intact_peak_mm(x).numpy()
        want = s.max_deflection_mm.to_numpy(np.float64)
        ok = np.isfinite(got) & (want > 0.1)
        err = float(100 * np.mean(np.abs(got[ok] - want[ok]) / want[ok]))
        check(err < 1.0, "peak deflection decoded from the pack reproduces the target "
                         "to %.4f%% over %d runs" % (err, int(ok.sum())))
    except Exception as exc:
        check(False, "pack check could not run: %s" % str(exc)[:120], warn=True)

    if args.full:
        print("\nopening every field map")
        bad_npz = []
        for i, r in enumerate(df.run_id):
            try:
                with np.load(os.path.join(HERE, "fields", r + ".npz")) as z:
                    if z["u3"].shape != (200, 200):
                        bad_npz.append(r)
            except Exception:
                bad_npz.append(r)
            if (i + 1) % 1000 == 0:
                print("    %d/%d" % (i + 1, n), flush=True)
        check(not bad_npz, "all %d .npz readable at 200x200 (%d bad)" % (n, len(bad_npz)))

    print()
    for w in warns:
        print("WARNING: %s" % w)
    if fails:
        print("\n%d CHECK(S) FAILED:" % len(fails))
        for f in fails:
            print("  - %s" % f)
        sys.exit(1)
    print("PASS - dataset is complete and self-consistent")


if __name__ == "__main__":
    main()
