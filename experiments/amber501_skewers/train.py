from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["ABSL_LOGGING_THRESHOLD"] = "2"  # 0=INFO,1=WARNING,2=ERROR,3=FATAL

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

import json
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import h5py

from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.data import make_joint_and_marginal
from jax_bnre_hmc.datasets import load_hdf5_dataset
from jax_bnre_hmc.loss import nre_loss_bce_style_from_logits, nre_loss_from_logits
from jax_bnre_hmc.model import RatioEstimatorMLP
from jax_bnre_hmc.train import TrainConfig, train


@hydra.main(config_path="../../configs/amber501_skewers", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Set the seed
    key = jax.random.PRNGKey(int(cfg.seed))

    # Load the dataset (pre-split) and preprocess it
    dataset_file = cfg.data.get("dataset_file")
    if dataset_file is None:
        raise ValueError(
            "data.dataset_file must be set for sinusoid experiment "
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

    # use min max scalers on theta
    # create min max scaler, symmetric around 0
    theta_scaler = MinMaxScaler(feature_range=(-1,1))
    theta_scaler.fit(theta_train_raw)

    # Transform all splits using train-fitted scalers
    theta_train = theta_scaler.transform(theta_train_raw)
    theta_val   = theta_scaler.transform(theta_val_raw)
    theta_test  = theta_scaler.transform(theta_test_raw)

    # plt.hist(theta_train.ravel())
    # plt.show()

    # use a combination of scalers to make x good
    x_scaler_1 = np.log1p
    x_scaler_2 = StandardScaler()
    # x_scaler_3 = MinMaxScaler(feature_range=(-5,5)) 

    # Transform x with first scaler
    x_train = x_scaler_1(x_train_raw)
    x_val   = x_scaler_1(x_val_raw)
    x_test  = x_scaler_1(x_test_raw)

    # Transform all splits using train-fitted with second scaler
    x_scaler_2.fit(x_train)
    x_train = x_scaler_2.transform(x_train)
    x_val   = x_scaler_2.transform(x_val)
    x_test  = x_scaler_2.transform(x_test)

    # # Transform all splits using train-fitted with third scaler
    # x_scaler_3.fit(x_train)
    # x_train = x_scaler_3.transform(x_train)
    # x_val   = x_scaler_3.transform(x_val)
    # x_test  = x_scaler_3.transform(x_test)

    # n, bins, patches = plt.hist(x_train.ravel())
    # plt.show()
    # print(n, bins, patches)


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
    print('\nTraining configuration created\nStarting training loop:')

    model = RatioEstimatorMLP(
        hidden_dims=tuple(cfg.model.hidden_dims),
        activation=str(cfg.model.activation),
        norm=str(cfg.model.norm),
    )

    train_output = train(
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        model=model,
        cfg=train_cfg,
    )

    state, train_losses, train_bce_losses, val_losses, val_bce_losses = train_output

    # Output directory
    run_dir = Path(HydraConfig.get().run.dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config in output directory
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # Basic sanity prints
    print("done. final train loss:", float(train_losses[-1]))
    print("done. final train bce :", float(train_bce_losses[-1]))
    print("done. final val loss:", float(val_losses[-1]))
    print("done. final val bce :", float(val_bce_losses[-1]))

    # Evaluate mean logit on joint vs marginal for a quick sanity check
    # (higher on joint is a good sign)
    key2 = jax.random.PRNGKey(int(cfg.seed) + 1)
    joint, marginal = make_joint_and_marginal(key2, theta_test, x_test)
    lj = state.apply_fn(state.params, joint.theta, joint.x)
    lm = state.apply_fn(state.params, marginal.theta, marginal.x)
    print("mean(logit) joint   :", float(jnp.mean(lj)))
    print("mean(logit) marginal:", float(jnp.mean(lm)))

    pj = jax.nn.sigmoid(lj)
    pm = jax.nn.sigmoid(lm)
    print("mean(sigmoid) joint   :", float(jnp.mean(pj)))
    print("mean(sigmoid) marginal:", float(jnp.mean(pm)))

    # Save the metrics in a txt file
    (run_dir / "metrics.txt").write_text(
        f"final_train_loss: {float(train_losses[-1])}\n"
        f"final_val_loss: {float(val_losses[-1])}\n"
        f"final_train_bce_style_loss: {float(train_bce_losses[-1])}\n"
        f"final_val_bce_style_loss: {float(val_bce_losses[-1])}\n"
        f"mean_logit_joint: {float(jnp.mean(lj))}\n"
        f"mean_logit_marginal: {float(jnp.mean(lm))}\n"
        f"mean_sigmoid_joint: {float(jnp.mean(pj))}\n"
        f"mean_sigmoid_marginal: {float(jnp.mean(pm))}\n"
    )

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

    plt.figure(figsize=(10, 5))
    plt.plot(pj, label="joint")
    plt.plot(pm, label="marginal")
    plt.legend()
    plt.savefig(run_dir / "sigmoid.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
