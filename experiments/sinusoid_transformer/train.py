from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["ABSL_LOGGING_THRESHOLD"] = "2"  # 0=INFO,1=WARNING,2=ERROR,3=FATAL

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

import json
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler

from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.data import make_joint_and_marginal
from jax_bnre_hmc.datasets import load_hdf5_dataset
from jax_bnre_hmc.loss import nre_loss_bce_style_from_logits, nre_loss_from_logits
from jax_bnre_hmc.model import RatioEstimatorTransformer
from jax_bnre_hmc.train import TrainConfig, train


@hydra.main(config_path="../../configs/sinusoid_transformer", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset_file = cfg.data.get("dataset_file")
    if dataset_file is None:
        raise ValueError(
            "data.dataset_file must be set for sinusoid_transformer experiment "
            "(path to HDF5 with theta_train, x_train, x_train_mask; theta_val, x_val, x_val_mask; "
            "theta_test, x_test, x_test_mask)"
        )
    dataset_file = str(dataset_file)
    print(f"\nLoading dataset from {dataset_file}")

    loaded = load_hdf5_dataset(dataset_file, validate=True)
    splits = loaded.splits

    theta_train_raw = splits.theta_train
    x_train_raw = splits.x_train
    mask_train_raw = splits.mask_train
    theta_val_raw = splits.theta_val
    x_val_raw = splits.x_val
    mask_val_raw = splits.mask_val
    theta_test_raw = splits.theta_test
    x_test_raw = splits.x_test
    mask_test_raw = splits.mask_test

    if mask_train_raw is None or mask_val_raw is None or mask_test_raw is None:
        raise ValueError(
            "Sinusoid transformer dataset must provide x_train_mask, x_val_mask, x_test_mask "
            "in the HDF5 file, with shapes matching x_train, x_val, x_test."
        )

    print(f"\ntheta_train shape: {theta_train_raw.shape}")
    print(f"x_train shape:     {x_train_raw.shape}")
    print(f"mask_train shape:  {mask_train_raw.shape}")
    print(f"theta_val shape:   {theta_val_raw.shape}")
    print(f"x_val shape:       {x_val_raw.shape}")
    print(f"mask_val shape:    {mask_val_raw.shape}")
    print(f"theta_test shape:  {theta_test_raw.shape}")
    print(f"x_test shape:      {x_test_raw.shape}")
    print(f"mask_test shape:   {mask_test_raw.shape}")

    # -----------------------------
    # Scale theta and y only
    # -----------------------------

    # For train split: replace invalid values with mean across observation axis,
    # then fit scalers on theta_train_raw and filled y_train.
    mask_train = mask_train_raw.astype(np.float32)
    valid_train = mask_train > 0.5
    valid_counts_train = np.sum(valid_train, axis=0)
    valid_sums_train = np.sum(x_train_raw * valid_train, axis=0)
    col_means_train = valid_sums_train / np.maximum(valid_counts_train, 1)
    y_train_filled = np.where(valid_train, x_train_raw, col_means_train[None, :])

    theta_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    theta_train = theta_scaler.fit_transform(theta_train_raw).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(y_train_filled).astype(np.float32)
    y_train_scaled = y_train_scaled * mask_train
    x_train_tokens = np.stack([y_train_scaled, mask_train], axis=-1).astype(np.float32)

    print(f"\nscaled theta_train shape: {theta_train.shape}")
    print(f"scaled y_train shape:      {y_train_scaled.shape}")
    print(f"x_train_tokens shape:      {x_train_tokens.shape}")  # (n_train, T, 2)

    # Apply the same scalers to val and test splits
    def _transform_split(theta_raw: np.ndarray, x_raw: np.ndarray, mask_raw: np.ndarray):
        mask_split = mask_raw.astype(np.float32)
        valid_split = mask_split > 0.5
        valid_counts_split = np.sum(valid_split, axis=0)
        valid_sums_split = np.sum(x_raw * valid_split, axis=0)
        col_means_split = valid_sums_split / np.maximum(valid_counts_split, 1)
        y_split_filled = np.where(valid_split, x_raw, col_means_split[None, :])

        theta_scaled_split = theta_scaler.transform(theta_raw).astype(np.float32)
        y_scaled_split = y_scaler.transform(y_split_filled).astype(np.float32)
        y_scaled_split = y_scaled_split * mask_split
        x_tokens_split = np.stack([y_scaled_split, mask_split], axis=-1).astype(np.float32)
        return theta_scaled_split, x_tokens_split

    theta_val, x_val_tokens = _transform_split(theta_val_raw, x_val_raw, mask_val_raw)
    theta_test, x_test_tokens = _transform_split(theta_test_raw, x_test_raw, mask_test_raw)

    # -----------------------------
    # Train / validation split
    # -----------------------------
    # Note: dataset is already split into train/val/test in the HDF5 file.
    # We simply use theta_train/theta_val and x_train_tokens/x_val_tokens.
    theta_train_split = theta_train
    x_train_split = x_train_tokens
    theta_val_split = theta_val
    x_val_split = x_val_tokens

    # -----------------------------
    # Train config
    # -----------------------------
    train_cfg = TrainConfig(
        seed=int(cfg.seed),
        lr=float(cfg.train.lr),
        epochs=int(cfg.train.epochs),
        bnre_gamma=float(cfg.train.bnre_gamma),
        print_every=int(cfg.train.print_every),
        batch_size=int(cfg.train.batch_size),
        clip_max_norm=cfg.train.clip_max_norm,
        save_every=int(cfg.train.save_every),
        checkpoint_dirname=cfg.train.checkpoint_dirname,
        stop_after_epochs=cfg.train.stop_after_epochs,
    )
    print("\nTraining configuration created\nStarting training loop:")

    # -----------------------------
    # Model
    # -----------------------------
    model = RatioEstimatorTransformer(
        d_model=int(cfg.model.d_model),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        transformer_mlp_dim=int(cfg.model.transformer_mlp_dim),
        transformer_activation=str(cfg.model.transformer_activation),
        head_hidden_dims=tuple(cfg.model.head_hidden_dims),
        head_activation=str(cfg.model.head_activation),
        head_norm=str(cfg.model.head_norm),
    )

    start_time = time.time()
    train_output = train(
        theta_train=theta_train_split,
        x_train=x_train_split,
        theta_val=theta_val_split,
        x_val=x_val_split,
        model=model,
        cfg=train_cfg,
    )
    total_train_time = time.time() - start_time
    state, train_losses, train_bce_losses, val_losses, val_bce_losses = train_output

    # -----------------------------
    # Output directory
    # -----------------------------
    run_dir = Path(HydraConfig.get().run.dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # -----------------------------
    # Basic sanity prints
    # -----------------------------
    print("done. final train loss:", float(train_losses[-1]))
    print("done. final train bce :", float(train_bce_losses[-1]))
    print("done. final val loss  :", float(val_losses[-1]))
    print("done. final val bce   :", float(val_bce_losses[-1]))

    # -----------------------------
    # Joint vs marginal sanity check
    # Use scaled/tokenized data
    # -----------------------------
    key2 = jax.random.PRNGKey(int(cfg.seed) + 1)
    theta_all = jnp.asarray(
        np.concatenate([theta_train, theta_val, theta_test], axis=0), dtype=jnp.float32
    )
    x_all = jnp.asarray(
        np.concatenate([x_train_tokens, x_val_tokens, x_test_tokens], axis=0),
        dtype=jnp.float32,
    )

    joint, marginal = make_joint_and_marginal(key2, theta_all, x_all)
    lj = state.apply_fn(state.params, joint.theta, joint.x)
    lm = state.apply_fn(state.params, marginal.theta, marginal.x)

    print("mean(logit) joint   :", float(jnp.mean(lj)))
    print("mean(logit) marginal:", float(jnp.mean(lm)))

    pj = jax.nn.sigmoid(lj)
    pm = jax.nn.sigmoid(lm)

    print("mean(sigmoid) joint   :", float(jnp.mean(pj)))
    print("mean(sigmoid) marginal:", float(jnp.mean(pm)))

    # -----------------------------
    # Save metrics
    # -----------------------------
    epochs_run = len(train_losses)
    time_per_epoch = total_train_time / max(epochs_run, 1)
    (run_dir / "metrics.txt").write_text(
        f"final_train_loss: {float(train_losses[-1])}\n"
        f"final_val_loss: {float(val_losses[-1])}\n"
        f"final_train_bce_style_loss: {float(train_bce_losses[-1])}\n"
        f"final_val_bce_style_loss: {float(val_bce_losses[-1])}\n"
        f"mean_logit_joint: {float(jnp.mean(lj))}\n"
        f"mean_logit_marginal: {float(jnp.mean(lm))}\n"
        f"mean_sigmoid_joint: {float(jnp.mean(pj))}\n"
        f"mean_sigmoid_marginal: {float(jnp.mean(pm))}\n"
        f"training_time_seconds: {float(total_train_time)}\n"
        f"training_time_per_epoch_seconds: {float(time_per_epoch)}\n"
    )

    # -----------------------------
    # Plots
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.savefig(run_dir / "losses.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_bce_losses, label="train_bce_style_loss")
    plt.plot(val_bce_losses, label="val_bce_style_loss")
    plt.legend()
    plt.savefig(run_dir / "bce_style_losses.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot only a subset of sigmoid outputs, otherwise this can be visually messy
    n_plot = min(500, len(pj))
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(pj[:n_plot]), label="joint")
    plt.plot(np.array(pm[:n_plot]), label="marginal")
    plt.legend()
    plt.savefig(run_dir / "sigmoid_subset.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Load best params and verify validation loss
    # -----------------------------
    best_dir = run_dir / cfg.train.checkpoint_dirname / "best"
    best_meta_path = run_dir / cfg.train.checkpoint_dirname / "best_meta.json"

    if best_dir.exists() and best_meta_path.exists():
        best_params = load_best_params(best_dir)

        best_meta = json.loads(best_meta_path.read_text())
        expected_best_val_loss = best_meta["val_loss"]

        key_val = jax.random.PRNGKey(int(cfg.seed) + 117)
        theta_val_jnp = jnp.asarray(theta_val_split, dtype=jnp.float32)
        x_val_jnp = jnp.asarray(x_val_split, dtype=jnp.float32)

        joint_val, marginal_val = make_joint_and_marginal(key_val, theta_val_jnp, x_val_jnp)
        logits_joint_val = state.apply_fn(best_params, joint_val.theta, joint_val.x)
        logits_marg_val = state.apply_fn(best_params, marginal_val.theta, marginal_val.x)
        recomputed_val_loss = float(
            nre_loss_bce_style_from_logits(logits_joint_val, logits_marg_val)
        )

        print(f"\nBest model verification:")
        print(f"  Expected best val_loss (from metadata): {expected_best_val_loss:.6f}")
        print(f"  Recomputed val_loss (from loaded params): {recomputed_val_loss:.6f}")
        print(f"  Difference: {abs(recomputed_val_loss - expected_best_val_loss):.6e}")

        if abs(recomputed_val_loss - expected_best_val_loss) < 1e-5:
            print("  ✓ Validation loss matches!")
        else:
            print("  ⚠ Warning: Validation loss mismatch, likely due to random seed mismatch for shuffling!")


if __name__ == "__main__":
    main()
