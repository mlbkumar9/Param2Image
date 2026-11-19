# -*- coding: utf-8 -*-
"""
Single-pass ODB extraction: corrected scalars + exact 4-channel field maps.

WHY
---
1. The campaign script recorded ``max(|U|)`` over every node carrying U output.
   Field output is already restricted to the plate, so the stored
   ``max_deflection_mm`` is dominated by plate nodes orphaned by element
   deletion and numerically ejected at several hundred m/s. All 5000 stored values exceed
   the distance the projectile itself could travel in the 2 ms step.
   Verified on RUN_001:  stored 1071.160 mm  ->  corrected 52.484 mm.
   ``max_peeq`` is contaminated the same way (stored 0.61-54.6 -> real ~0.45-0.56),
   because elements at the instant of deletion run away numerically.

2. The plate is a perfectly structured, uniform grid:
       201 x 201 x 7 nodes = 282807      200 x 200 x 6 elements = 240000
       in-plane spacing exactly 3.000 mm, element labels 1..240000 contiguous
   so element (i,j) maps to pixel (i,j) with no interpolation. That makes an
   exact, invertible, 4-channel physical image:

       R = von Mises (MPa)     G = PEEQ (-)
       B = |U3| (mm)           A = surviving-material mask (0/1)

   at 3.000 mm/pixel, which is the true native resolution of the FE solution.
   A rendered viewport screenshot of the same run contains no more spatial
   information - it is these same cells drawn larger, with a colour map,
   lighting and mesh edges painted on top.

The mesh is identical in every run, so connectivity and the element->grid
index map are computed once and cached to .npz.

Runs under Abaqus Python (needs odbAccess), not the system Python. All paths are
command-line arguments; nothing is configured in the file.

USAGE
    abaqus python abaqus/extract_fields.py \
        --odb-root /path/to/Abaqus_OUT \
        --out-csv  /path/to/results.csv \
        --out-fields /path/to/fields \
        [--start 1] [--end 5000] [--limit N] [--no-fields]

Resumable: a run is skipped when its CSV row is OK and its field file exists.
The ODB tree is opened read-only and never modified.
"""
from __future__ import print_function

import argparse
import csv
import os
import sys
import time
import traceback

import numpy as np

from odbAccess import openOdb
from abaqusConstants import MISES


PLATE_KEYWORD = "PLATE"
NX = NY = 200          # in-plane elements
NZ = 6                 # layers through the 5 mm thickness
PLATE_L_MM = 600.0
PLATE_T_MM = 5.0
CELL_MM = PLATE_L_MM / NX          # 3.000 mm per pixel, exact

FIELDNAMES = [
    "run_id",
    "max_deflection_mm",        # max |U3| over surviving plate NODES  <- the fix
    "max_disp_mag_mm",
    "max_deflection_map_mm",    # max of the B channel (element-averaged)
    "max_mises_mpa",
    "max_peeq",
    "max_sdeg",
    "n_elem_total",
    "n_elem_deleted",
    "n_node_surviving",
    "perforated",
    "hole_area_mm2",
    "hole_frac",
    "legacy_max_deflection_mm",
    "extract_status",
]


# ---------------------------------------------------------------- mesh cache
def _plate(odb):
    ra = odb.rootAssembly
    keys = [k for k in ra.instances.keys() if PLATE_KEYWORD in k.upper()]
    if not keys:
        raise RuntimeError("no plate instance in %s" % list(ra.instances.keys()))
    return ra.instances[keys[0]]


def build_or_load_mesh(odb, cache_path):
    """Return dict with connectivity, sorted node labels and grid indices."""
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}

    plate = _plate(odb)
    t0 = time.time()

    node_lab = np.array([n.label for n in plate.nodes], dtype=np.int64)
    node_xyz = np.array([n.coordinates for n in plate.nodes], dtype=np.float64)
    o = np.argsort(node_lab)
    node_lab_s, node_xyz_s = node_lab[o], node_xyz[o]

    elems = plate.elements
    ne = len(elems)
    elem_lab = np.empty(ne, dtype=np.int64)
    conn = np.empty((ne, len(elems[0].connectivity)), dtype=np.int64)
    for i, e in enumerate(elems):
        elem_lab[i] = e.label
        conn[i] = e.connectivity

    conn_idx = np.searchsorted(node_lab_s, conn.ravel()).reshape(conn.shape)
    cent = node_xyz_s[conn_idx].mean(axis=1)

    gi = np.clip(((cent[:, 0] + PLATE_L_MM / 2.0) / CELL_MM).astype(np.int32), 0, NX - 1)
    gj = np.clip(((cent[:, 1] + PLATE_L_MM / 2.0) / CELL_MM).astype(np.int32), 0, NY - 1)
    gk = np.clip((cent[:, 2] / (PLATE_T_MM / NZ)).astype(np.int32), 0, NZ - 1)

    mesh = dict(node_lab_s=node_lab_s, elem_lab=elem_lab, conn_idx=conn_idx,
                gi=gi, gj=gj, gk=gk)
    sys.stderr.write("  cached mesh (%d elems) in %.1fs\n" % (ne, time.time() - t0))
    if cache_path:
        np.savez_compressed(cache_path, **mesh)
    return mesh


# ------------------------------------------------------------- bulk readers
def _blocks(field, inst_name):
    labs, dats = [], []
    for b in field.bulkDataBlocks:
        inst = getattr(b, "instance", None)
        if inst is not None and inst.name.upper() != inst_name.upper():
            continue
        lab = getattr(b, "nodeLabels", None)
        if lab is None:
            lab = getattr(b, "elementLabels", None)
        if lab is None:
            continue
        labs.append(np.asarray(lab))
        dats.append(np.asarray(b.data))
    if not labs:
        return None, None
    return np.concatenate(labs), np.concatenate(dats)


def _elem_vector(fo, name, mesh, plate_name, invariant=None):
    """Per-element scalar aligned to the cached element ordering."""
    if name not in fo.keys():
        return None
    f = fo[name]
    if invariant is not None:
        f = f.getScalarField(invariant=invariant)
    lab, dat = _blocks(f, plate_name)
    if lab is None:
        return None
    dat = np.asarray(dat).reshape(len(dat), -1)[:, 0]
    elem_lab = mesh["elem_lab"]
    if len(lab) == len(elem_lab) and np.array_equal(lab, elem_lab):
        return dat
    o = np.argsort(elem_lab)
    pos = o[np.searchsorted(elem_lab, lab, sorter=o)]
    out = np.zeros(len(elem_lab), dtype=np.float64)
    out[pos] = dat
    return out


def _reduce_to_grid(vals, alive, mesh, how):
    """Through-thickness reduction over surviving elements -> (NX, NY)."""
    acc = np.zeros((NX, NY, NZ), dtype=np.float64)
    cnt = np.zeros((NX, NY, NZ), dtype=np.float64)
    gi, gj, gk = mesh["gi"], mesh["gj"], mesh["gk"]
    v = np.where(alive, vals, 0.0)
    acc[gi, gj, gk] = v
    cnt[gi, gj, gk] = alive.astype(np.float64)
    if how == "max":
        acc = np.where(cnt > 0, acc, -np.inf)
        out = acc.max(axis=2)
        return np.where(np.isfinite(out), out, 0.0)
    if how == "absmax":
        acc = np.where(cnt > 0, np.abs(acc), -np.inf)
        out = acc.max(axis=2)
        return np.where(np.isfinite(out), out, 0.0)
    s = acc.sum(axis=2)
    c = cnt.sum(axis=2)
    return np.where(c > 0, s / np.maximum(c, 1.0), 0.0)


# ------------------------------------------------------------------ extract
def extract_one(odb_path, mesh_cache, field_path):
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        plate = _plate(odb)
        pname = plate.name
        mesh = build_or_load_mesh(odb, mesh_cache)

        frame = odb.steps[list(odb.steps.keys())[-1]].frames[-1]
        fo = frame.fieldOutputs

        status = _elem_vector(fo, "STATUS", mesh, pname)
        alive = status > 0.5
        n_total = int(len(alive))
        n_del = int(n_total - alive.sum())

        mises = _elem_vector(fo, "S", mesh, pname, MISES)
        peeq = _elem_vector(fo, "PEEQ", mesh, pname)
        sdeg = _elem_vector(fo, "SDEG", mesh, pname)

        # nodal displacement
        ulab, udat = _blocks(fo["U"], pname)
        node_lab_s = mesh["node_lab_s"]
        u3_node = np.zeros(len(node_lab_s))
        umag_node = np.zeros(len(node_lab_s))
        pos = np.searchsorted(node_lab_s, ulab)
        u3_node[pos] = udat[:, 2]
        umag_node[pos] = np.sqrt((udat ** 2).sum(axis=1))

        # legacy buggy number, over EVERY node in the model
        legacy = 0.0
        for b in fo["U"].bulkDataBlocks:
            d = np.asarray(b.data)
            if d.size:
                legacy = max(legacy, float(np.sqrt((d ** 2).sum(axis=1)).max()))

        conn_idx = mesh["conn_idx"]
        alive_nodes = np.unique(conn_idx[alive])
        max_u3 = float(np.abs(u3_node[alive_nodes]).max())
        max_umag = float(umag_node[alive_nodes].max())

        u3_elem = u3_node[conn_idx].mean(axis=1)

        # ---- field maps ----
        m_mises = _reduce_to_grid(mises, alive, mesh, "max")
        m_peeq = _reduce_to_grid(peeq, alive, mesh, "max")
        m_u3 = _reduce_to_grid(u3_elem, alive, mesh, "absmax")
        cnt = np.zeros((NX, NY, NZ))
        cnt[mesh["gi"], mesh["gj"], mesh["gk"]] = alive.astype(float)
        m_mask = (cnt.sum(axis=2) > 0).astype(np.float32)

        if field_path:
            np.savez_compressed(
                field_path,
                mises=m_mises.astype(np.float32),
                peeq=m_peeq.astype(np.float32),
                u3=m_u3.astype(np.float32),
                mask=m_mask,
            )

        hole_frac = float(1.0 - m_mask.mean())
        return {
            "max_deflection_mm": round(max_u3, 5),
            "max_disp_mag_mm": round(max_umag, 5),
            "max_deflection_map_mm": round(float(m_u3.max()), 5),
            "max_mises_mpa": round(float(mises[alive].max()), 4) if alive.any() else "",
            "max_peeq": round(float(peeq[alive].max()), 6) if peeq is not None and alive.any() else "",
            "max_sdeg": round(float(sdeg[alive].max()), 6) if sdeg is not None and alive.any() else "",
            "n_elem_total": n_total,
            "n_elem_deleted": n_del,
            "n_node_surviving": int(len(alive_nodes)),
            "perforated": int(n_del > 0),
            "hole_area_mm2": round(hole_frac * PLATE_L_MM ** 2, 3),
            "hole_frac": round(hole_frac, 6),
            "legacy_max_deflection_mm": round(legacy, 5),
            "extract_status": "OK",
        }
    finally:
        try:
            odb.close()
        except Exception:
            pass


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odb-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-fields", default="")
    ap.add_argument("--mesh-cache", default="")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-fields", action="store_true")
    args = ap.parse_args()

    if args.out_fields and not os.path.isdir(args.out_fields):
        os.makedirs(args.out_fields)

    done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv, "r") as fh:
            for row in csv.DictReader(fh):
                if row.get("extract_status") != "OK":
                    continue
                if args.no_fields or not args.out_fields:
                    done.add(row["run_id"])
                elif os.path.exists(os.path.join(args.out_fields, row["run_id"] + ".npz")):
                    done.add(row["run_id"])
        sys.stderr.write("resuming: %d complete\n" % len(done))

    new_file = not os.path.exists(args.out_csv)
    fh = open(args.out_csv, "a")
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
    if new_file:
        writer.writeheader()

    n_ok = n_err = 0
    t0 = time.time()
    for i in range(args.start, args.end + 1):
        run_id = "RUN_%03d" % i
        if run_id in done:
            continue
        odb_path = os.path.join(args.odb_root, run_id, "ImpactJob_%s.odb" % run_id)
        if not os.path.exists(odb_path):
            continue

        fpath = "" if (args.no_fields or not args.out_fields) \
            else os.path.join(args.out_fields, run_id + ".npz")
        row = {"run_id": run_id}
        t1 = time.time()
        try:
            row.update(extract_one(odb_path, args.mesh_cache, fpath))
            n_ok += 1
        except Exception as exc:
            row["extract_status"] = "ERROR: %s" % str(exc).replace("\n", " ")[:180]
            n_err += 1
            sys.stderr.write(traceback.format_exc())
        writer.writerow(row)
        fh.flush()

        n = n_ok + n_err
        if n % 25 == 0:
            rate = (time.time() - t0) / max(n, 1)
            sys.stderr.write("  %s ok=%d err=%d %.2fs/odb ETA %.2f h\n"
                             % (run_id, n_ok, n_err, rate, rate * (args.end - i) / 3600.0))
        if args.limit and n >= args.limit:
            break

    fh.close()
    sys.stderr.write("DONE ok=%d err=%d in %.1f min\n" % (n_ok, n_err, (time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
