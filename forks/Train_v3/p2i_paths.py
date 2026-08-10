"""Portable path helpers for the Train_v3 fork."""
from pathlib import Path
import os

FORK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FORK_ROOT.parents[1]  # forks/Train_v3 -> repo


def data_dir() -> Path:
    env = os.environ.get("P2I_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    for cand in [
        REPO_ROOT / "data" / "Abaqus_OUT",
        FORK_ROOT / "Abaqus_OUT",
        REPO_ROOT / "data" / "Abaqus_PS" / "Abaqus_OUT",
    ]:
        if cand.is_dir():
            return cand
    return REPO_ROOT / "data" / "Abaqus_OUT"


def checkpoint_dir(default_name: str = "training_checkpoints") -> Path:
    env = os.environ.get("P2I_CHECKPOINT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return FORK_ROOT / default_name


def figures_dir() -> Path:
    env = os.environ.get("P2I_FIGURES_DIR")
    if env:
        return Path(env).expanduser().resolve()
    cand = REPO_ROOT / "Documentation" / "DML_Mahendra" / "figures"
    if cand.parent.is_dir():
        return cand
    return FORK_ROOT / "figures"


def report_figures_dir() -> Path:
    env = os.environ.get("P2I_REPORT_FIGURES_DIR")
    if env:
        return Path(env).expanduser().resolve()
    cand = REPO_ROOT / "Documentation" / "Param2Image_Report" / "figures"
    if cand.parent.is_dir():
        return cand
    return FORK_ROOT / "report_figures"
