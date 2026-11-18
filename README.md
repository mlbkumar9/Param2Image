# Param2Image: Impact Simulation cGAN (PyTorch)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Param2Image** is a multi-modal conditional GAN (cGAN) that acts as a fast surrogate for Abaqus impact simulations. From four impact parameters it predicts a post-impact **RGBA 512x512 image** together with three engineering scalars (peak Mises stress, peak PEEQ, and peak deflection), so you can explore design space without re-running full FEA for every case.

Trained weights are **not** shipped. Train locally or point scripts at your own checkpoint.

## Development phases

The repository keeps each development phase side by side under `phases/`:

| Phase | Path | Status |
|-------|------|--------|
| Train_v3 | [`phases/Train_v3/`](phases/Train_v3/) | Current — StyleGAN-lite generator, position supervision, optional XLA/TPU training at 1024x1024 |
| Train_v2 | [`phases/Train_v2/`](phases/Train_v2/) | Earlier phase — original cGAN pipeline: importable `p2i` package + CLIs |

Shared assets (`data/`, `results/`, `logs/`, `training_checkpoints/`) live at the repository root.

## Features

- Conditional generator with convolutional image head + numerical head (`physics` or `legacy`)
- Parameter-conditioned discriminator (LSGAN / MSE)
- Tabular standardization with **log1p** response scaling (`ImpactTabularScaler`)
- Train / generate / evaluate / compare / analyze CLIs under `phases/Train_v2/scripts/`
- Importable package `p2i` via `pip install -e .`
- Results organized under `results/` (evaluation, comparison, analysis, predictions)

## Method overview

Full-order Abaqus impact runs are accurate but costly when many parameter combinations must be screened. Param2Image learns a conditional surrogate $G$ that maps impact settings to a post-impact field image and three scalar FEA responses. After offline training on a finite simulation set, new queries run in milliseconds.

The sections below describe the Train_v2 pipeline (see `phases/Train_v2/`); Train_v3 extends the same surrogate idea with a StyleGAN-lite generator and position supervision. They follow the full path: **data and preprocessing → training loop → optional evaluation → prediction/inference**. Layer-level detail lives in [phases/Train_v2/docs/ALGORITHM.md](phases/Train_v2/docs/ALGORITHM.md).

### Inputs and outputs

| Role | Quantity | Notes |
|------|----------|--------|
| Input | Projectile velocity $v$ | m/s |
| Input | Projectile radius $r$ | mm |
| Input | Impact location $(x,y)$ | mm |
| Output | After-impact image $\hat{\mathbf{I}}$ | RGBA, $512\times 512$, values in $[-1,1]$ |
| Output | Peak Mises $\hat{\sigma}_{\mathrm{Mises}}$ | MPa |
| Output | Peak PEEQ $\hat{\varepsilon}_{\mathrm{PEEQ}}$ | dimensionless |
| Output | Peak deflection $\hat{\delta}$ | mm |

With $\mathbf{p}=(v,r,x,y)^\top$ and latent noise $\mathbf{z}\sim\mathcal{N}(0,I_{100})$,

$$
\big(\hat{\mathbf{I}},\,\hat{\mathbf{y}}\big)=G(\mathbf{z},\,\tilde{\mathbf{p}}),\qquad
\hat{\mathbf{y}}=(\hat{\sigma}_{\mathrm{Mises}},\,\hat{\varepsilon}_{\mathrm{PEEQ}},\,\hat{\delta}).
$$

### End-to-end workflow

The diagram below is the product path from Abaqus exports through training to a saved prediction. Training and inference share the same scaler statistics and generator weights. Solid arrows are the main data path; dashed arrows mark optional evaluation and the handoff into inference.

```mermaid
flowchart TB
  dataLoad["Load results_master.csv + after-images<br/>from data/impact_dataset"]
  prep["Fit/apply ImpactTabularScaler<br/>params: z-score<br/>outputs: log1p then z-score<br/>images: 512x512 RGBA to -1..1"]
  dataLoad --> prep

  trainPath["Train path"]
  prep --> trainPath
  batchIn["Batch: scaled params, real image I, scaled y"]
  trainPath --> batchIn
  sampleZ["Sample latent z ~ N(0, I_100)"]
  batchIn --> sampleZ
  gFwd["G forward: fake image + predicted numerics"]
  sampleZ --> gFwd
  gStep["Update G: adv MSE + 100 image L1 + 10 SmoothL1<br/>Adam + grad-norm clip 5"]
  gFwd --> gStep
  dStep["Update D: MSE on real vs fake<br/>params-conditioned; Adam + grad clip"]
  gStep --> dStep
  saveArt["Save generator checkpoint<br/>normalization_stats.json + config<br/>every 20 epochs and at end"]
  dStep --> saveArt

  evalOpt["Optional: evaluate.py / compare.py<br/>SSIM, PSNR, R2 and side-by-side vs GT"]
  saveArt -.-> evalOpt

  inferPath["Infer path"]
  saveArt -.-> inferPath
  loadG["Load G + scaler"]
  inferPath --> loadG
  userP["User params p = v, r, x, y"]
  loadG --> userP
  normP["Normalize params with scaler"]
  userP --> normP
  zInf["Sample z"]
  loadG --> zInf
  genOut["G.eval: RGBA image + scaled numerics"]
  normP --> genOut
  zInf --> genOut
  savePred["Inverse-transform numerics to physical units<br/>save PNG + CSV under results/predictions/"]
  genOut --> savePred
```

**Training path.** For each batch the code samples $\mathbf{z}$, runs $G$, updates $G$ with adversarial MSE, image L1, and SmoothL1 on scaled numerics, then updates $D$ on real vs fake images (params-conditioned). Defaults: Adam $\mathrm{lr}=2\times10^{-4}$, $\beta_1=0.5$, batch size 2, 150 epochs, grad-norm clip 5. Device order: CUDA, XPU, HIP, MPS, CPU. Scaler stats are written when fitted; generator weights and config land in `training_checkpoints/` with TensorBoard logs in `logs/`.

**Inference path.** Load `cGAN_generator.pth` and `normalization_stats.json`, standardize the four user parameters, sample $\mathbf{z}$, run $G$ in eval mode, write the RGBA image under `results/predictions/`, and inverse-transform the three scalars to physical units in a CSV.

**Evaluation (optional).** After training, `phases/Train_v2/scripts/evaluate.py` and `phases/Train_v2/scripts/compare.py` score image metrics and numerical parity against Abaqus; they are not required for a single forward prediction.

### Generator and discriminator

The generator is multi-modal. Latent noise and scaled parameters are concatenated, $\mathbf{h}_0=[\mathbf{z};\,\tilde{\mathbf{p}}]\in\mathbb{R}^{104}$.

- **Image head.** A linear layer forms a $256\times 64\times 64$ feature volume, then three $\times 2$ upsamples with convolutions yield $512\times 512$ RGBA, followed by Tanh ($64\to128\to256\to512$).
- **Numerical head.** Two modes are supported:
  - `physics` (default for new training): MLP on $\tilde{\mathbf{p}}$ only (deterministic scalars for a given impact case).
  - `legacy`: MLP on $[\mathbf{z};\,\tilde{\mathbf{p}}]$ for older checkpoints. Loaders auto-detect the mode from weight shapes.

The discriminator $D(\mathbf{I},\tilde{\mathbf{p}})$ embeds the four parameters to a small spatial map, upsamples it to $512\times 512$, concatenates with the image, and applies five stride-2 conv blocks to a scalar validity logit (LSGAN-style; no sigmoid inside the network).

### Objectives

With real image $\mathbf{I}$, fake image $G_I$, scaled targets $\tilde{\mathbf{y}}$, and MSE adversarial loss:

$$
\mathcal{L}_D=\tfrac{1}{2}\Big(\mathrm{MSE}\big(D(\mathbf{I},\tilde{\mathbf{p}}),1\big)+\mathrm{MSE}\big(D(G_I,\tilde{\mathbf{p}}),0\big)\Big),
$$

$$
\mathcal{L}_G^{\mathrm{adv}}=\mathrm{MSE}\big(D(G_I,\tilde{\mathbf{p}}),1\big),\quad
\mathcal{L}_G^{L1}=\|G_I-\mathbf{I}\|_1,\quad
\mathcal{L}_G^{\mathrm{num}}=\mathrm{SmoothL1}_{\beta=0.5}(G_y,\tilde{\mathbf{y}}),
$$

$$
\mathcal{L}_G=\mathcal{L}_G^{\mathrm{adv}}+\lambda_{L1}\mathcal{L}_G^{L1}+\lambda_{\mathrm{num}}\mathcal{L}_G^{\mathrm{num}},
\qquad \lambda_{L1}=100,\;\lambda_{\mathrm{num}}=10.
$$

### Tabular normalization

Parameters use per-feature z-scores. Positive responses use $\log(1+y)$ then z-scores. Images use channel-wise mean/std $0.5$ so pixels lie in $[-1,1]$. Fitted statistics are saved as `normalization_stats.json` next to the generator weights and inverted at inference.

Layer equations, head diagrams, and design notes: [phases/Train_v2/docs/ALGORITHM.md](phases/Train_v2/docs/ALGORITHM.md). CLI steps: [phases/Train_v2/docs/WORKFLOWS.md](phases/Train_v2/docs/WORKFLOWS.md).

## Repository layout

```text
Param2Image/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml
├── requirements.txt
├── .github/workflows/
├── phases/
│   ├── Train_v3/             # current version (StyleGAN-lite, position supervision, XLA/TPU)
│   │   ├── train_p2i_pytorch.py
│   │   ├── train_p2i_xla.py
│   │   ├── model_p2i_pytorch.py
│   │   ├── position_supervision.py
│   │   ├── eval_p2i_colab.py
│   │   └── ...
│   └── Train_v2/             # earlier phase (original cGAN pipeline)
│       ├── src/p2i/          # importable package
│       ├── scripts/          # train | generate | evaluate | compare | analyze
│       ├── notebooks/        # train_p2i.ipynb
│       └── docs/             # ALGORITHM.md | WORKFLOWS.md
├── data/                     # impact_dataset (CSV + images)
├── training_checkpoints/     # weights written here (not in git)
└── results/                  # evaluation | comparison | analysis | predictions
```

## Dataset

```text
data/impact_dataset/
  results_master.csv
  RUN_001/
  RUN_002/
  ...
```

See [data/README.md](data/README.md). Override with `P2I_DATA_DIR` or `--data_dir` where supported.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install a GPU build of PyTorch from [pytorch.org](https://pytorch.org) when available.

## Quickstart

```bash
# Train (Train_v2 pipeline)
python phases/Train_v2/scripts/train.py

# Generate one case
python phases/Train_v2/scripts/generate.py --velocity 120 --radius 5 --impact_x 0 --impact_y 0

# Evaluate
python phases/Train_v2/scripts/evaluate.py --num_samples 50

# Compare
python phases/Train_v2/scripts/compare.py --num_samples 6 --selection stratified

# Analyze
python phases/Train_v2/scripts/analyze.py
# or CPU-only from existing evaluation CSVs:
python phases/Train_v2/scripts/analyze_cpu.py
```

For the current Train_v3 pipeline see [phases/Train_v3/README.md](phases/Train_v3/README.md).

Step-by-step detail: **[phases/Train_v2/docs/WORKFLOWS.md](phases/Train_v2/docs/WORKFLOWS.md)**.

### Output locations

| Script | Default output |
|--------|----------------|
| `train.py` | `training_checkpoints/`, `logs/` |
| `generate.py` | `results/predictions/` |
| `evaluate.py` | `results/evaluation/` |
| `compare.py` | `results/comparison/` |
| `analyze*.py` | `results/analysis/` |

## Weights policy

- Do **not** commit `*.pth` / `*.pt` or bulky result dumps.
- Train on your machine, or load via `--model_path`.
- See [training_checkpoints/README.md](training_checkpoints/README.md).

## Citation

If you use this surrogate in academic work, please cite the repository and describe the Param2Image / Train_v2 cGAN pipeline (parameters → image + numerical FEA responses).

## License

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security notes: [SECURITY.md](SECURITY.md).
