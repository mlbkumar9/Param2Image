# Abaqus_PS (local dataset)

This tree holds the Parametric Study Abaqus workflow and run outputs. It is local-only and gitignored except for this README.

## Layout

- abaqus_autorun.py / abaqus_autorun - Copy.py: batch autorun helpers
- combine_csv.py: merges per-run CSVs
- run.txt: run notes / command hints
- Abaqus_OUT/: simulation outputs
  - results_master.csv (when present): combined master table
  - RUN_* folders: per-case Abaqus results

## How to place data

Copy or symlink your study tree under data/Abaqus_PS/ so that data/Abaqus_PS/Abaqus_OUT exists, or set P2I_DATA_DIR to the folder that contains the RUN_* outputs (often data/Abaqus_OUT or data/Abaqus_PS/Abaqus_OUT).

Do not commit the bulky Abaqus_OUT contents (~150+ MB).
