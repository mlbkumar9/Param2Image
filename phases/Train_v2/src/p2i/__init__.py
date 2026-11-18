"""Param2Image (p2i): conditional GAN surrogate for Abaqus impact simulations."""

from .model import Discriminator, Generator, load_generator_checkpoint
from .preprocessing import (
    OUTPUT_COLUMNS,
    PARAM_COLUMNS,
    ImpactTabularScaler,
    load_scaler_if_available,
)
from .paths import (
    checkpoint_dir,
    default_data_dir,
    default_generator_path,
    logs_dir,
    repo_root,
    results_dir,
)

__all__ = [
    "Generator",
    "Discriminator",
    "load_generator_checkpoint",
    "ImpactTabularScaler",
    "PARAM_COLUMNS",
    "OUTPUT_COLUMNS",
    "load_scaler_if_available",
    "repo_root",
    "default_data_dir",
    "checkpoint_dir",
    "default_generator_path",
    "results_dir",
    "logs_dir",
]

__version__ = "0.1.0"
