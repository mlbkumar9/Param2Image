# Param2Image

A conditional StyleGAN-lite surrogate for ABAQUS/Explicit projectile-impact simulations. From four impact parameters (velocity, projectile radius, strike position x and y) it predicts four physical fields on the plate's own finite-element grid, plus the three peak responses with their uncertainty.

This repository holds the model reported in the manuscript *Generative Deep Learning for Impact Simulations: A Conditional StyleGAN-Lite Framework for Parameter-to-Image Synthesis* (M. Behera, A.K. Sahu, V.K. Sutrakar, P.R. Budarapu; submitted to *Machine Learning with Applications*). It corresponds to internal version 3.6.

## What it predicts

| channel | field | unit |
|---|---|---|
| 0 | von Mises stress | MPa |
| 1 | equivalent plastic strain (PEEQ) | - |
| 2 | out-of-plane deflection, abs(U3) | mm |
| 3 | surviving material (perforation mask) | 0/1 |

Output is 200 x 200, one pixel per element at exactly 3.000 mm, so every pixel converts back to engineering units. A separate physics head predicts peak von Mises stress, peak PEEQ and peak deflection as a mean and a variance.

## Results (held-out test split, 604 cases, 8 latent draws averaged)

| metric | model | nearest-neighbour baseline | two simulations of the same impact |
|---|---|---|---|
| SSIM, all channels | **0.962** | 0.908 | 0.963 |
| SSIM, von Mises | **0.897** | 0.765 | 0.897 |
| impact position error, median | 0.99 mm | | 1.1 mm (locator floor, mean) |

| peak response | MAPE | R2 |
|---|---|---|
| von Mises stress | 2.3 % | 0.81 |
| PEEQ | 11.3 % | 0.81 |
| deflection | 9.45 % | 0.90 |

One forward pass takes about 0.6 s on four CPU cores, against a mean ABAQUS solve of 82.6 s on the same cores. Training took 4.85 h on one A100 (600-epoch budget, early stop at epoch 545, best validation SSIM at epoch 543).

## Method

- **Generator.** A mapping MLP turns the standardised parameters and a 100-d latent into a style vector w. A learned 25 x 25 constant is upsampled through style-modulated residual blocks (AdaIN) to 200 x 200 x 4. Spatial conditioning adds a rendered map of the strike position and projectile footprint at every stage.
- **Physics head.** A Fourier-feature residual MLP maps the parameters to the mean and log-variance of the three peak responses. The mean is fitted with a Huber loss and the variance by a Gaussian NLL on the detached residual, so the uncertainty does not pull the mean. It has its own optimiser, trains for the first 100 epochs and is checkpointed on its own validation loss (`ckpt_best_phys.pt`).
- **Discriminator.** Spectral-normalised PatchGAN with parameter conditioning and a minibatch standard-deviation layer, trained with DiffAugment.
- **Losses.** Non-saturating logistic GAN loss with lazy R1, channel-weighted L1 on the fields (weights 1.93, 0.63, 0.90, 0.53), a differentiable impact-position loss, and a soft-peak loss tying field maxima to the scalar targets. The training target is box-filtered over 3 cells; evaluation scores against the raw fields.
- **Weights.** EMA of the generator, cosine learning rate with 5 warm-up epochs.

## Install

```bash
pip install -e .            # add .[dev] for the tests
```

Install a CUDA build of PyTorch first if you have a GPU (https://pytorch.org).

## Data

The 6040-simulation dataset is part of an ongoing study and is not distributed. [data/README.md](data/README.md) gives the format the code expects (master CSV, packed field array, split) so the pipeline can run on your own simulations, and `data/verify_dataset.py` checks a dataset folder against it.

## Usage

Train with the reported configuration (all defaults):

```bash
p2i-train --csv DATA/results_master.csv --pack DATA/fields_pack.npy --out runs/v36 --cache-ram
```

Evaluate on the test split:

```bash
p2i-evaluate --csv DATA/results_master.csv --pack DATA/fields_pack.npy \
    --ckpt runs/v36/ckpt_best.pt --ckpt-phys runs/v36/ckpt_best_phys.pt \
    --split test --out results/eval_test
```

Predict one case:

```bash
p2i-predict --ckpt runs/v36/ckpt_best.pt --ckpt-phys runs/v36/ckpt_best_phys.pt \
    --velocity-kmh 300 --radius-mm 50 --x-mm 0 --y-mm 0 --out pred.npz --png
```

Each command also runs as `python -m p2i.<train|evaluate|predict|pack_dataset>`.

## Generating the dataset

`abaqus/run_campaign.py` builds, meshes and solves the impact cases under ABAQUS Python; `abaqus/extract_fields.py` reads each ODB into a per-run `.npz` field map. `p2i-pack` then packs the maps into the training array.

## Layout

```text
src/p2i/      model, data pipeline, metrics, train / evaluate / predict / pack CLIs
abaqus/       simulation campaign and field extraction (ABAQUS Python)
data/         results_master.csv, pack index, dataset verifier
tests/        CPU smoke tests
```

## License

MIT. See [LICENSE](LICENSE).
