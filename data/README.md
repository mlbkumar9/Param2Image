# Data format

The 6040-simulation dataset used in the paper is not distributed. This page describes the layout the code expects, so the pipeline can be run on other simulations.

## Simulation setup (paper)

600 x 600 x 5 mm Al 2024-T351 plate, clamped on all edges, struck normally by a rigid steel sphere. 240,000 C3D8R elements on a uniform 200 x 200 in-plane grid (3.000 mm per element, six layers through the thickness). Johnson-Cook plasticity and damage with element deletion, 2 ms step.

## Dataset folder

```text
<dataset>/
    results_master.csv      one row per run
    fields_pack.npy         (N, 4, 200, 200) float16, mapped to [-1, 1]
    fields_pack_index.csv   run_id per pack row, same order as the CSV
    fields/RUN_*.npz        per-run field maps in physical units (input to p2i-pack)
    images/RUN_*_{before,after}.png   ABAQUS viewport renders (two per run)
    logs/RUN_*.log          solver logs
    mesh/plate_mesh.npz     the shared plate mesh
```

Training and evaluation read only `results_master.csv` and `fields_pack.npy` (or the per-run `.npz` files). The other items are part of the full dataset record and are checked by `verify_dataset.py`.

`results_master.csv` needs at least: `run_id`, `split` (`train` / `val` / `test`), the inputs `velocity_kmh`, `proj_radius_mm`, `impact_x_mm`, `impact_y_mm`, the targets `max_mises_mpa`, `max_peeq`, `max_deflection_mm`, the flag `perforated`, and `field_npz` (path of the run's `.npz`, relative to the dataset folder). `verify_dataset.py` also expects `img_before` and `img_after` (render paths) and requires every split to contain both perforating and non-perforating runs.

## Field maps

Each `.npz` holds four 200 x 200 arrays on the element grid:

| key | field | unit |
|---|---|---|
| `mises` | von Mises stress | MPa |
| `peeq` | equivalent plastic strain | - |
| `u3` | out-of-plane deflection magnitude | mm |
| `mask` | surviving material | 0/1 |

`p2i-pack` maps them to [-1, 1] (linear on `mises` and `mask`, square root on `peeq` and `u3`; both invertible) and writes the pack. The pack is only valid under the mapping in `src/p2i/data.py`.

`max_deflection_mm` is the peak abs(U3) over intact material, at least two cells (6 mm) from any damage front, so a one-cell rim at a hole boundary does not set the target.

## Checking a dataset

```bash
python data/verify_dataset.py <dataset>          # structure, references, targets, pack pairing
python data/verify_dataset.py <dataset> --full   # also opens every .npz
```
