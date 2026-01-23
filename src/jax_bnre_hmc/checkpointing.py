from __future__ import annotations

import json
from pathlib import Path

from flax.training import train_state
from orbax.checkpoint import PyTreeCheckpointer

from hydra.core.hydra_config import HydraConfig


def get_run_dir() -> Path:
    """Get Hydra's run output directory.
    
    Returns:
        Absolute path to the current Hydra run output directory.
    """
    return Path(HydraConfig.get().run.dir).resolve()


def ensure_dirs(base_dir: Path, checkpoint_dirname: str) -> tuple[Path, Path]:
    """Create checkpoint directories if they don't exist.
    
    Creates the directory structure for storing checkpoints:
    - base_dir/checkpoint_dirname/latest/
    - base_dir/checkpoint_dirname/best/
    
    Args:
        base_dir: Base directory for checkpoints (typically Hydra run output directory).
        checkpoint_dirname: Name of the checkpoint subdirectory.
    
    Returns:
        A tuple containing:
            - latest_dir: Path to latest checkpoint directory.
            - best_dir: Path to best checkpoint directory.
    """
    checkpoint_base = base_dir / checkpoint_dirname
    latest_dir = checkpoint_base / "latest"
    best_dir = checkpoint_base / "best"
    
    latest_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    
    return latest_dir, best_dir


def write_meta(meta_path: Path, epoch: int, val_loss: float) -> None:
    """Write checkpoint metadata to a JSON file.
    
    Args:
        meta_path: Path where the metadata file should be written.
        epoch: Epoch number at which the checkpoint was saved.
        val_loss: Validation loss at the checkpoint.
    """
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
    """Save the latest training state and metadata.
    
    Saves the complete TrainState (including parameters, optimizer state, etc.)
    and writes metadata to a JSON file. Overwrites any existing checkpoint.
    
    Args:
        state: Current training state to save.
        latest_dir: Directory where the checkpoint should be saved.
        meta_path: Path where the metadata JSON file should be written.
        epoch: Current epoch number.
        val_loss: Current validation loss.
    """
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
    """Save the best model parameters and metadata.
    
    Saves only the model parameters (not the full TrainState) and writes metadata
    to a JSON file. Overwrites any existing checkpoint. This is used to save the
    model with the best validation loss.
    
    Args:
        params: Model parameters PyTree to save.
        best_dir: Directory where the checkpoint should be saved.
        meta_path: Path where the metadata JSON file should be written.
        epoch: Epoch number at which the best model was found.
        val_loss: Validation loss of the best model.
    """
    checkpointer = PyTreeCheckpointer()
    checkpointer.save(best_dir, params, force=True)
    write_meta(meta_path, epoch, val_loss)


def load_best_params(best_dir: Path) -> dict:
    """Load best model parameters from checkpoint directory.
    
    Args:
        best_dir: Directory containing the saved best model checkpoint.
    
    Returns:
        Model parameters PyTree that can be used with the model's apply function.
    """
    checkpointer = PyTreeCheckpointer()
    return checkpointer.restore(best_dir)
