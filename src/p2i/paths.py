"""Shared path helpers for Param2Image (portable repo-relative defaults)."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Resolve the repository root from a script or package file path."""
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "training_checkpoints").is_dir() and (candidate / "src").is_dir():
            return candidate
        if (candidate / "training_checkpoints").is_dir() and (candidate / "data").is_dir():
            return candidate
    # Fallback: scripts/ -> repo, package file -> repo
    if here.name == "scripts":
        return here.parent
    if here.name == "p2i":
        return here.parents[1]
    return here


def default_data_dir(start: Path | None = None) -> Path:
    """Abaqus dataset root: ``P2I_DATA_DIR`` or ``<repo>/data/Abaqus_OUT``."""
    env = os.environ.get("P2I_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root(start) / "data" / "Abaqus_OUT"


def checkpoint_dir(start: Path | None = None) -> Path:
    return repo_root(start) / "training_checkpoints"


def default_generator_path(start: Path | None = None) -> Path:
    return checkpoint_dir(start) / "cGAN_generator.pth"


def results_dir(start: Path | None = None) -> Path:
    return repo_root(start) / "results"


def logs_dir(start: Path | None = None) -> Path:
    return repo_root(start) / "logs"
