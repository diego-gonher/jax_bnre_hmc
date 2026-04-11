from __future__ import annotations

import os
os.environ["ABSL_LOGGING_THRESHOLD"] = "2"

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

import json
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler


@hydra.main(config_path="../../configs/dw", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    device = cfg.get("device", "cpu")
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "single_gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
    else:
        raise ValueError(f"Unknown device={device!r}. Use 'cpu' or 'single_gpu'.")

    import jax
    import jax.numpy as jnp
    from jax_bnre_hmc.checkpointing import load_best_params
    from jax_bnre_hmc.data import make_joint_and_marginal
    from jax_bnre_hmc.datasets import load_hdf5_dataset
    from jax_bnre_hmc.loss import nre_loss_bce_style_from_logits
    from jax_bnre_hmc.model import RatioEstimatorMLP
    from jax_bnre_hmc.plotting import save_training_diagnostic_plots
    from jax_bnre_hmc.plot_style import apply_plot_style
    from jax_bnre_hmc.train import TrainConfig, train, write_train_summary

    print(f"JAX backend: {jax.default_backend()}")

    apply_plot_style()
    key = jax.random.PRNGKey(int(cfg.seed))

    try:
        dataset_file = cfg.data.get("dataset_file")
        if dataset_file is None:
            raise ValueError(
                "data.dataset_file must be set for dw experiment "
                "(path to HDF5 with 'theta_train', 'x_train', 'theta_val', 'x_val', 'theta_test', 'x_test')"
            )
        dataset_file = str(dataset_file)
        print(f"\nLoading dataset from {dataset_file}")

        loaded = load_hdf5_dataset(dataset_file, validate=True)
        splits = loaded.splits

        theta_train_raw = splits.theta_train
        x_train_raw = splits.x_train
        theta_val_raw = splits.theta_val
        x_val_raw = splits.x_val
        theta_test_raw = splits.theta_test
        x_test_raw = splits.x_test

        print("\nLoaded dataset splits:")
        print(f" - theta_train shape: {theta_train_raw.shape}")
        print(f" - x_train shape:     {x_train_raw.shape}")
        print(f" - theta_val shape:   {theta_val_raw.shape}")
        print(f" - x_val shape:       {x_val_raw.shape}")
        print(f" - theta_test shape:  {theta_test_raw.shape}")
        print(f" - x_test shape:      {x_test_raw.shape}")

        # MinMax scalers on both theta and x (fit on train only), same pattern as sinusoid/train.py
        theta_scaler = MinMaxScaler(feature_range=(-1, 1))
        x_scaler = MinMaxScaler(feature_range=(-1, 1))

        theta_train = theta_scaler.fit_transform(theta_train_raw)
        x_train = x_scaler.fit_transform(x_train_raw)

        theta_val = theta_scaler.transform(theta_val_raw)
        x_val = x_scaler.transform(x_val_raw)

        theta_test = theta_scaler.transform(theta_test_raw)
        x_test = x_scaler.transform(x_test_raw)

        print(f"\nscaled theta_train shape: {theta_train.shape}")
        print(f"scaled x_train shape:     {x_train.shape}")

        train_cfg = TrainConfig(
            seed=int(cfg.seed),
            lr=float(cfg.train.lr),
            epochs=int(cfg.train.epochs),
            bnre_lambda=float(cfg.train.bnre_lambda),
            print_every=int(cfg.train.print_every),
            batch_size=int(cfg.train.batch_size),
            clip_max_norm=cfg.train.clip_max_norm,
            save_every=int(cfg.train.save_every),
            checkpoint_dirname=cfg.train.checkpoint_dirname,
            stop_after_epochs=cfg.train.stop_after_epochs,
            model_type="mlp",
        )
        print("\nTraining configuration created\nStarting training loop:")

        model = RatioEstimatorMLP(
            hidden_dims=tuple(cfg.model.hidden_dims),
            activation=str(cfg.model.activation),
            norm=str(cfg.model.norm),
        )

        start_time = time.time()
        train_output = train(
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            model=model,
            cfg=train_cfg,
        )
    except Exception as e:
        run_dir = Path(HydraConfig.get().run.dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        msg = str(e) if str(e) else type(e).__name__
        write_train_summary(run_dir, status="error", message=msg)
        raise

    total_train_time = time.time() - start_time
    state, train_losses, train_bce_losses, val_losses, val_bce_losses = train_output

    run_dir = Path(HydraConfig.get().run.dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train.yaml").write_text(OmegaConf.to_yaml(cfg))

    print("done. final train loss:", float(train_losses[-1]))
    print("done. final train bce :", float(train_bce_losses[-1]))
    print("done. final val loss:", float(val_losses[-1]))
    print("done. final val bce :", float(val_bce_losses[-1]))

    key2 = jax.random.PRNGKey(int(cfg.seed) + 1)
    theta_all = jnp.concatenate(
        [
            jnp.asarray(theta_train_raw, dtype=jnp.float32),
            jnp.asarray(theta_val_raw, dtype=jnp.float32),
            jnp.asarray(theta_test_raw, dtype=jnp.float32),
        ],
        axis=0,
    )
    x_all = jnp.concatenate(
        [
            jnp.asarray(x_train_raw, dtype=jnp.float32),
            jnp.asarray(x_val_raw, dtype=jnp.float32),
            jnp.asarray(x_test_raw, dtype=jnp.float32),
        ],
        axis=0,
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

    save_training_diagnostic_plots(
        run_dir,
        train_losses,
        val_losses,
        train_bce_losses,
        val_bce_losses,
        pj,
        pm,
    )

    best_dir = run_dir / cfg.train.checkpoint_dirname / "best"
    best_meta_path = run_dir / cfg.train.checkpoint_dirname / "best_meta.json"

    if best_dir.exists() and best_meta_path.exists():
        best_params = load_best_params(best_dir)
        best_meta = json.loads(best_meta_path.read_text())
        expected_best_val_loss = best_meta["val_loss"]

        key_val = jax.random.PRNGKey(int(cfg.seed) + 117)
        joint_val, marginal_val = make_joint_and_marginal(key_val, theta_val, x_val)
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
            print("  Validation loss matches.")
        else:
            print("  Warning: Validation loss mismatch, likely due to random seed mismatch for shuffling.")


if __name__ == "__main__":
    main()
