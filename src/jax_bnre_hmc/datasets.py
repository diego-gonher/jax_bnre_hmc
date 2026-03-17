from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class DatasetSplits:
    """Train/validation/test splits for (theta, x) pairs.

    Expected dataset paths in the HDF5 file:
      - theta_train, x_train, x_train_mask (optional)
      - theta_val,  x_val, x_val_mask (optional)
      - theta_test, x_test, x_test_mask (optional)
    """

    theta_train: np.ndarray
    x_train: np.ndarray
    theta_val: np.ndarray
    x_val: np.ndarray
    theta_test: np.ndarray
    x_test: np.ndarray

    # Optional masks for x only (missing-data scenarios, e.g. masked 1D time series).
    # If present, they must have the same shape as the corresponding x_* arrays.
    # HDF5 keys: x_train_mask, x_val_mask, x_test_mask.
    mask_train: np.ndarray | None = None
    mask_val: np.ndarray | None = None
    mask_test: np.ndarray | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    """Optional metadata attached to a loaded dataset."""

    theta_names: tuple[str, ...] | None = None
    x_names: tuple[str, ...] | None = None
    description: str | None = None
    attrs: dict[str, Any] | None = None


@dataclass(frozen=True)
class LoadedDataset:
    """Container for loaded splits and associated metadata."""

    splits: DatasetSplits
    metadata: DatasetMetadata | None = None


@dataclass(frozen=True)
class DatasetValidationReport:
    """Result of validating an HDF5 dataset against the expected contract."""

    ok: bool
    errors: list[str]


class DatasetValidationError(ValueError):
    """Raised when an HDF5 dataset fails validation."""


_REQUIRED_PATHS: tuple[str, ...] = (
    "theta_train",
    "x_train",
    "theta_val",
    "x_val",
    "theta_test",
    "x_test",
)


def _is_numeric_array(arr: Any) -> bool:
    return isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number)


def validate_hdf5_dataset(path: str | Path) -> DatasetValidationReport:
    """Validate that an HDF5 file follows the (train/val/test) dataset contract.

    The required structure is:
      - theta_train, x_train
      - theta_val,   x_val
      - theta_test,  x_test

    Each array must be:
      - present
      - 2D
      - numeric and finite
    Within each split, theta and x must share the same number of rows.
    Across splits, theta feature dims must match; likewise for x feature dims.
    Optional x masks (x_train_mask, x_val_mask, x_test_mask) apply only to x;
    if any is present, all three must be present and match the shape of x_*.
    """
    path = Path(path)
    errors: list[str] = []

    if not path.exists():
        return DatasetValidationReport(ok=False, errors=[f"File does not exist: {path}"])

    try:
        with h5py.File(path, "r") as f:
            # 1. Check required datasets exist
            for ds_path in _REQUIRED_PATHS:
                if ds_path not in f:
                    errors.append(f"Missing required dataset '{ds_path}'.")

            if errors:
                return DatasetValidationReport(ok=False, errors=errors)

            # 2. Load arrays and check basic properties
            arrays: dict[str, np.ndarray] = {}
            for ds_path in _REQUIRED_PATHS:
                arr = np.asarray(f[ds_path])
                arrays[ds_path] = arr
                if arr.ndim != 2:
                    errors.append(
                        f"Dataset '{ds_path}' must be 2D, got shape {arr.shape} (ndim={arr.ndim})."
                    )
                if not _is_numeric_array(arr):
                    errors.append(
                        f"Dataset '{ds_path}' must be numeric; got dtype {arr.dtype}."
                    )
                if not np.all(np.isfinite(arr)):
                    errors.append(f"Dataset '{ds_path}' contains non-finite values.")

            if errors:
                return DatasetValidationReport(ok=False, errors=errors)

            # 3. Per-split row count consistency
            split_keys = {
                "train": ("theta_train", "x_train"),
                "val": ("theta_val", "x_val"),
                "test": ("theta_test", "x_test"),
            }
            for split, (theta_key, x_key) in split_keys.items():
                theta_arr = arrays[theta_key]
                x_arr = arrays[x_key]
                if theta_arr.shape[0] != x_arr.shape[0]:
                    errors.append(
                        f"Row count mismatch in split '{split}': "
                        f"{theta_key}.shape[0]={theta_arr.shape[0]} vs "
                        f"{x_key}.shape[0]={x_arr.shape[0]}."
                    )

            # 4. Feature dimension consistency across splits
            theta_dims = {
                "train": arrays["theta_train"].shape[1],
                "val": arrays["theta_val"].shape[1],
                "test": arrays["theta_test"].shape[1],
            }
            x_dims = {
                "train": arrays["x_train"].shape[1],
                "val": arrays["x_val"].shape[1],
                "test": arrays["x_test"].shape[1],
            }

            if len(set(theta_dims.values())) != 1:
                errors.append(
                    "Theta feature dimensions differ across splits: "
                    + ", ".join(f"{k}={v}" for k, v in theta_dims.items())
                )
            if len(set(x_dims.values())) != 1:
                errors.append(
                    "X feature dimensions differ across splits: "
                    + ", ".join(f"{k}={v}" for k, v in x_dims.items())
                )

            # 5. Optional x-masks: if any are present, require all three and
            # ensure shapes match the corresponding x_* arrays.
            mask_keys = {
                "train": "x_train_mask",
                "val": "x_val_mask",
                "test": "x_test_mask",
            }
            have_masks = {split: key in f for split, key in mask_keys.items()}
            if any(have_masks.values()):
                for split, present in have_masks.items():
                    if not present:
                        errors.append(
                            f"x-mask dataset for split '{split}' is missing "
                            f"(expected '{mask_keys[split]}')."
                        )

                # Only attempt further checks if all three are present.
                if all(have_masks.values()):
                    for split, (theta_key, x_key) in split_keys.items():
                        mask_key = mask_keys[split]
                        mask_arr = np.asarray(f[mask_key])

                        if mask_arr.ndim != 2:
                            errors.append(
                                f"Dataset '{mask_key}' must be 2D, got shape "
                                f"{mask_arr.shape} (ndim={mask_arr.ndim})."
                            )

                        if not _is_numeric_array(mask_arr):
                            errors.append(
                                f"Dataset '{mask_key}' must be numeric; got dtype "
                                f"{mask_arr.dtype}."
                            )

                        if not np.all(np.isfinite(mask_arr)):
                            errors.append(
                                f"Dataset '{mask_key}' contains non-finite values."
                            )

                        # Check that mask has same (N, T) as x_* for this split.
                        x_arr = arrays[x_key]
                        if mask_arr.shape != x_arr.shape:
                            errors.append(
                                f"Shape mismatch between '{mask_key}' and '{x_key}': "
                                f"{mask_arr.shape} vs {x_arr.shape}."
                            )

    except OSError as e:
        errors.append(f"Failed to open HDF5 file '{path}': {e}")

    return DatasetValidationReport(ok=not errors, errors=errors)


def load_hdf5_dataset(path: str | Path, validate: bool = True) -> LoadedDataset:
    """Load a pre-split HDF5 dataset following the standard contract.

    This function:
      - optionally validates the file structure and contents
      - loads the six 2D arrays into float32 numpy arrays
      - extracts simple metadata if present

    Note:
      - No scaling, shuffling, or reshaping is performed here.
      - Scaling should be fit on train only and applied to val/test downstream.
    """
    path = Path(path)

    if validate:
        report = validate_hdf5_dataset(path)
        if not report.ok:
            joined = "\n".join(f"- {msg}" for msg in report.errors)
            raise DatasetValidationError(f"Invalid dataset '{path}':\n{joined}")

    with h5py.File(path, "r") as f:
        def _load(ds_path: str) -> np.ndarray:
            return np.asarray(f[ds_path], dtype=np.float32)

        theta_train = _load("theta_train")
        x_train = _load("x_train")
        theta_val = _load("theta_val")
        x_val = _load("x_val")
        theta_test = _load("theta_test")
        x_test = _load("x_test")

        # Optional x-masks: HDF5 keys x_train_mask, x_val_mask, x_test_mask.
        mask_train = _load("x_train_mask") if "x_train_mask" in f else None
        mask_val = _load("x_val_mask") if "x_val_mask" in f else None
        mask_test = _load("x_test_mask") if "x_test_mask" in f else None

        splits = DatasetSplits(
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            theta_test=theta_test,
            x_test=x_test,
            mask_train=mask_train,
            mask_val=mask_val,
            mask_test=mask_test,
        )

        # Optional metadata
        theta_names = None
        x_names = None
        description = None
        attrs: dict[str, Any] = {}

        # Try file-level attributes first
        for key, value in f.attrs.items():
            attrs[key] = value
            if key == "theta_names":
                try:
                    theta_names = tuple(str(v) for v in value)
                except Exception:
                    theta_names = None
            elif key == "x_names":
                try:
                    x_names = tuple(str(v) for v in value)
                except Exception:
                    x_names = None
            elif key == "description":
                description = str(value)

        metadata = DatasetMetadata(
            theta_names=theta_names,
            x_names=x_names,
            description=description,
            attrs=attrs or None,
        )

    return LoadedDataset(splits=splits, metadata=metadata)

