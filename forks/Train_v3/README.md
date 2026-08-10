# Train_v3 experimental fork

This folder is an experimental Param2Image v3 fork (StyleGAN-lite generator, position supervision, optional XLA/TPU training). It lives on the train-v3 branch under forks/Train_v3/.

The main package on master (src/p2i/ + scripts/) tracks the Train_v2-style cGAN workflow. Prefer master for day-to-day work unless you are iterating on v3 ideas.

## Paths (no machine-specific hardcodes)

Set env vars as needed:

- P2I_DATA_DIR - directory containing RUN_* outputs / images (defaults via p2i_paths.data_dir())
- P2I_CSV_PATH - optional override for results_master.csv
- P2I_CHECKPOINT_DIR - checkpoint folder override
- P2I_OUT_DIR - generation output folder
- P2I_FIGURES_DIR / P2I_REPORT_FIGURES_DIR - figure outputs

p2i_paths.py resolves data in this order when P2I_DATA_DIR is unset:

1. <repo>/data/Abaqus_OUT
2. <fork>/Abaqus_OUT (local, gitignored)
3. <repo>/data/Abaqus_PS/Abaqus_OUT

Weights (*.pt / *.pth) stay local. JSON configs under training_checkpoints_position/ may be committed as small placeholders.

## Quick start

    cd forks/Train_v3
    # optional: set P2I_DATA_DIR to your Abaqus_OUT
    python train_p2i_pytorch.py
    python generate_p2i_pytorch.py --velocity 120 --radius 5 --impact_x 0 --impact_y 0
    python -m unittest test_position_training_regressions.py

Colab: see notebook_train_p2i_colab_tpu.ipynb. The large notebook_train_p2i_colab.ipynb (outputs embedded) is kept locally but gitignored.

## Local-only siblings

Bulky leftovers may remain under Pres_v2/Train_v3 or Legacy/Train_v3/ on disk. Do not commit Abaqus dumps, zips, or weight files.
