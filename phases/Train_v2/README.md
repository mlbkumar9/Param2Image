# Train_v2 — earlier development phase

The original Param2Image cGAN pipeline: importable `p2i` package (`src/p2i/`) plus
train / generate / evaluate / compare / analyze CLIs (`scripts/`), the
`notebooks/train_p2i.ipynb` helper, and the method docs (`docs/`).

Kept as a development phase of the project. The current version is
[`../Train_v3/`](../Train_v3/).

Install the package from the repository root:

```bash
pip install -e .
```

Then run the CLIs from the repository root or this folder, for example:

```bash
python scripts/train.py
python scripts/generate.py --velocity 120 --radius 5 --impact_x 0 --impact_y 0
```
