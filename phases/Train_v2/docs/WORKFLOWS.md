# Workflows

End-to-end commands for the restructured Param2Image repository. All paths are relative to the repository root after `pip install -e .`.

## Prerequisites

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Place the dataset at `data/impact_dataset/` (see [data/README.md](../../../data/README.md)) or set `P2I_DATA_DIR`.

Trained weights are **not** in git. Train locally or point `--model_path` at your checkpoint.

---

## 1. Train

```bash
python phases/Train_v2/scripts/train.py
```

Writes under `training_checkpoints/`:

- `cGAN_generator.pth`
- `normalization_stats.json`
- `cGAN_generator_config.json`

TensorBoard:

```bash
tensorboard --logdir logs
```

---

## 2. Generate (single case)

```bash
python phases/Train_v2/scripts/generate.py ^
  --velocity 120 --radius 5 --impact_x 0 --impact_y 0
```

Unix:

```bash
python phases/Train_v2/scripts/generate.py \
  --velocity 120 --radius 5 --impact_x 0 --impact_y 0
```

Outputs: `results/predictions/sim_pred_<timestamp>/` (`generated_image.png`, `predicted_outputs.csv`).

Optional flags: `--model_path`, `--scaler_path`, `--output_dir`.

---

## 3. Evaluate

```bash
python phases/Train_v2/scripts/evaluate.py --num_samples 50
```

Custom data / weights:

```bash
python phases/Train_v2/scripts/evaluate.py --num_samples 50 ^
  --data_dir path\to\impact_dataset ^
  --model_path training_checkpoints\cGAN_generator.pth
```

Outputs: `results/evaluation/eval_<timestamp>/` (metrics CSVs, best/worst/random panels, dashboards).

---

## 4. Compare

```bash
python phases/Train_v2/scripts/compare.py --num_samples 6 --selection stratified
```

Other selections: `random`, `tail`. Outputs under `results/comparison/comparison_<timestamp>/`.

---

## 5. Analyze

Full pipeline (needs torch + data + weights):

```bash
python phases/Train_v2/scripts/analyze.py
```

From existing evaluation CSVs only (CPU-friendly):

```bash
python phases/Train_v2/scripts/analyze_cpu.py
```

Outputs under `results/analysis/`.

---

## 6. Notebook

`phases/Train_v2/notebooks/train_p2i.ipynb` is a Colab-oriented helper. For local work prefer `phases/Train_v2/scripts/train.py` with `./data/impact_dataset`.

---

## Environment overrides

| Variable | Effect |
|----------|--------|
| `P2I_DATA_DIR` | Root of `impact_dataset` (`results_master.csv` + `RUN_*` folders) |

---

## Suggested research loop

1. Train → inspect TensorBoard losses.
2. Generate a few hand-picked cases for sanity.
3. Evaluate on a held-out sample count.
4. Compare stratified panels for papers.
5. Analyze for extended figures / LaTeX tables.

Deep method details: [ALGORITHM.md](ALGORITHM.md).
