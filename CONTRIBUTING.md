# Contributing

Thanks for contributing to **Param2Image** (impact-simulation cGAN).

## Scope

This repository contains the `p2i` package, training/inference CLIs, preprocessing, evaluation helpers, and documentation for a parameters-to-image surrogate of Abaqus impact runs.

## Guidelines

1. Keep PRs focused: package code, scripts, docs, tests, or lightweight CI, not trained weights or result dumps.
2. Do **not** commit `*.pth` / `*.pt`, or contents of `results/`, `logs/`, or `data/Abaqus_OUT/`.
3. Prefer repo-relative defaults (`./data/Abaqus_OUT`, `P2I_DATA_DIR`) over hard-coded machine paths.
4. Import from `p2i` after `pip install -e .`; avoid `sys.path` hacks.
5. Match existing Python style; avoid drive-by refactors.
6. Open an issue for large design changes before implementing them.

## Local checks

```bash
pip install -e .
python -m py_compile src/p2i/*.py scripts/*.py
python scripts/generate.py --help
python scripts/evaluate.py --help
python scripts/compare.py --help
```

GPU and full `Abaqus_OUT` data are not required for syntax checks.
