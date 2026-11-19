"""Pack all field maps into one float16 memmap for fast training / easy upload.

6000 individual .npz files are slow to read per-sample and awkward to move to
another machine. This writes a single (N, 4, 200, 200) float16 array of ALREADY-NORMALISED
values plus a run_id index, so training does one memory-mapped read per sample
and the whole dataset is a single ~1.8 GB file.

float16 on [-1, 1] carries ~5e-4 resolution, i.e. ~0.6 MPa on the stress
channel and ~0.03 mm on the deflection channel -- far below the FE solution's
own discretisation error.

USAGE
    python -m p2i.pack_dataset --csv data/Impact_Dataset_60/results_master.csv \
                           --out data/Impact_Dataset_60/fields_pack.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .data import GRID, CHANNEL_NAMES, fields_to_norm, load_field_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n = len(df)
    root = os.path.dirname(os.path.abspath(args.csv))
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16,
                                    shape=(n, len(CHANNEL_NAMES), GRID, GRID))
    bad = []
    for i, path in enumerate(df.field_npz):
        p = path if os.path.isabs(path) else os.path.join(root, path)
        try:
            out[i] = fields_to_norm(load_field_npz(p)).astype(np.float16)
        except Exception as exc:
            bad.append((df.run_id.iloc[i], str(exc)[:80]))
        if (i + 1) % 500 == 0:
            print(f"  packed {i+1}/{n}", flush=True)
    out.flush()
    df[["run_id"]].to_csv(os.path.splitext(args.out)[0] + "_index.csv", index=False)

    size_gb = os.path.getsize(args.out) / 1e9
    print(f"\npacked {n - len(bad)}/{n} runs -> {args.out}  ({size_gb:.2f} GB)")
    if bad:
        print(f"FAILED {len(bad)}:")
        for r, e in bad[:10]:
            print("   ", r, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
