# Contributing to Param2Image

Thank you for your interest in improving Param2Image. Bug reports, documentation fixes and focused pull requests are all welcome.

## Reporting issues

Open an issue using one of the templates. For bugs, include the command you ran, the full error message, and your Python, PyTorch and operating-system versions. Please do not attach simulation data or checkpoints you are not permitted to share.

## Development setup

```bash
git clone https://github.com/mlbkumar9/Param2Image.git
cd Param2Image
pip install -e ".[dev]"
pytest -q
```

A GPU is not needed for development; the test suite runs on CPU.

## Pull requests

1. Open an issue first for any change to the model, the losses or the training defaults, since these define the reported results.
2. Keep each pull request to one change, and add or update a test when you change behaviour.
3. Run `pytest -q` before pushing. CI runs the same suite on every pull request.
4. Do not commit datasets, checkpoints (`*.pt`, `*.pth`), packed arrays (`*.npy`) or run outputs (`runs/`, `results/`).
5. Use paths relative to the repository or passed on the command line, never machine-specific absolute paths.
6. Follow the style of the surrounding code and keep imports inside the `p2i` package relative.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold it.
