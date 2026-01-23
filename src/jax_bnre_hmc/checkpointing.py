from __future__ import annotations

import json
from pathlib import Path

from flax.training import train_state
from orbax.checkpoint import PyTreeCheckpointer

from hydra.core.hydra_config import HydraConfig


def get_run_dir() -> Path:
    """Get Hydra's run output directory."""
    return Path(HydraConfig.get().run.dir).resolve()


def ensure_dirs(base_dir: Path, checkpoint_dirname: str) -> tuple[Path, Path]:
    """
    Create checkpoint directories if they don't exist.
    
    Returns:
        latest_dir: Path to latest checkpoint directory
        best_dir: Path to best checkpoint directory
    """
    checkpoint_base = base_dir / checkpoint_dirname
    latest_dir = checkpoint_base / "latest"
    best_dir = checkpoint_base / "best"
    
    latest_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    
    return latest_dir, best_dir


def write_meta(meta_path: Path, epoch: int, val_loss: float) -> None:
    """Write metadata JSON file."""
    meta = {
        "epoch": int(epoch),
        "val_loss": float(val_loss),
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def save_latest(
    state: train_state.TrainState,
    latest_dir: Path,
    meta_path: Path,
    epoch: int,
    val_loss: float,
) -> None:
    """Save latest TrainState and metadata."""
    checkpointer = PyTreeCheckpointer()
    checkpointer.save(latest_dir, state, force=True)
    write_meta(meta_path, epoch, val_loss)


def save_best(
    params: dict,
    best_dir: Path,
    meta_path: Path,
    epoch: int,
    val_loss: float,
) -> None:
    """Save best model parameters and metadata."""
    checkpointer = PyTreeCheckpointer()
    checkpointer.save(best_dir, params, force=True)
    write_meta(meta_path, epoch, val_loss)


def load_best_params(best_dir: Path) -> dict:
    """Load best model parameters from checkpoint directory."""
    checkpointer = PyTreeCheckpointer()
    return checkpointer.restore(best_dir)
