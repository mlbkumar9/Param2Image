# Dataset layout

Training and inference read from:

```text
data/impact_dataset/
  results_master.csv
  RUN_001/
    ...
  RUN_002/
    ...
```

## Required CSV columns

`results_master.csv` must include at least:

| Column | Role |
|--------|------|
| `run_id` | Sample identifier |
| `velocity_ms` | Projectile velocity (m/s) |
| `proj_radius_mm` | Projectile radius (mm) |
| `impact_x_mm` | Impact X (mm) |
| `impact_y_mm` | Impact Y (mm) |
| `max_mises_mpa` | Max Mises stress (MPa) |
| `max_peeq` | Max equivalent plastic strain |
| `max_deflection_mm` | Max deflection (mm) |
| `img_after` | Relative path to the post-impact RGBA/RGB image |

Override the default path with:

```bash
# PowerShell
$env:P2I_DATA_DIR = "X:\path\to\impact_dataset"

# bash
export P2I_DATA_DIR=/path/to/impact_dataset
```

`data/impact_dataset/` is shipped with this repository (CSV + images). Solver log files are ignored by git.
