from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["ABSL_LOGGING_THRESHOLD"] = "2"  # 0=INFO,1=WARNING,2=ERROR,3=FATAL

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

import json
import time
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import h5py

from jax_bnre_hmc.model import RatioEstimatorMLP
from jax_bnre_hmc.train import TrainConfig, train
from jax_bnre_hmc.data import make_joint_and_marginal
from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.loss import nre_loss_bce_style_from_logits, nre_loss_from_logits


@hydra.main(config_path="../../configs/amber501_p1d", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Set the seed
    key = jax.random.PRNGKey(int(cfg.seed))

    # -------------------------------
    # Load dataset and preprocess it
    # -------------------------------
    dataset_file = str(cfg.data.dataset_file)
    print(f"\nLoading dataset from {dataset_file}")

    with h5py.File(dataset_file, "r") as f:
        # load the needed arrays
        params = f['params'][:, :5]                  # (4509, 5)
        mocks = f['subset_of_mock_datasets'][:]      # (4509, 1000, 19)

        # Split by parameter combination first
        theta_train, theta_test, x_train, x_test = train_test_split(
            params,
            mocks,
            test_size=float(cfg.data.test_fraction),
            random_state=152637,
        )

        theta_train, theta_val, x_train, x_val = train_test_split(
            theta_train,
            x_train,
            test_size=float(cfg.data.validation_fraction),
            random_state=152637,
        )

        def expand_theta_mock_pairs(theta, x):
            """
            Convert:
                theta: (n_theta, n_params)
                x:     (n_theta, n_mocks, x_dim)

            into:
                theta_pairs: (n_theta * n_mocks, n_params)
                x_pairs:     (n_theta * n_mocks, x_dim)
            """
            n_theta, n_mocks, x_dim = x.shape
            n_params = theta.shape[1]

            theta_pairs = np.repeat(theta, n_mocks, axis=0)
            x_pairs = x.reshape(n_theta * n_mocks, x_dim)

            return theta_pairs, x_pairs


        theta_train_pairs, x_train_pairs = expand_theta_mock_pairs(theta_train, x_train)
        theta_val_pairs, x_val_pairs     = expand_theta_mock_pairs(theta_val, x_val)
        theta_test_pairs, x_test_pairs   = expand_theta_mock_pairs(theta_test, x_test)
        

        _1, theta_train, _2, x_train = train_test_split(
            theta_train_pairs,
            x_train_pairs,
            test_size=int(cfg.data.n_train),
            random_state=47805,
        )

        _1, theta_val, _2, x_val = train_test_split(
            theta_val_pairs,
            x_val_pairs,
            test_size=int(cfg.data.n_val),
            random_state=940856,
        )

        _1, theta_test, _2, x_test = train_test_split(
            theta_test_pairs,
            x_test_pairs,
            test_size=int(cfg.data.n_test),
            random_state=496702,
        )

        print(f'\nFinal dataset splits for SBI:')
        print(f' - Train')
        print(f'    - theta shape: {theta_train.shape}')
        print(f'    - x shape: {x_train.shape}')
        print(f' - Validation')
        print(f'    - theta shape: {theta_val.shape}')
        print(f'    - x shape: {x_val.shape}')
        print(f' - Test')
        print(f'    - theta shape: {theta_test.shape}')
        print(f'    - x shape: {x_test.shape}\n')

        # use min max scalers on both
        # create min max scalers, symmetric around 0
        theta_scaler = MinMaxScaler(feature_range=(-1,1))
        x_scaler = MinMaxScaler(feature_range=(-1,1))

        theta_scaler.fit(theta_train)
        x_scaler.fit(x_train)

        # Transform all splits using train-fitted scalers
        theta_train = theta_scaler.transform(theta_train)
        theta_val   = theta_scaler.transform(theta_val)
        theta_test  = theta_scaler.transform(theta_test)

        x_train = x_scaler.transform(x_train)
        x_val   = x_scaler.transform(x_val)
        x_test  = x_scaler.transform(x_test)


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

    start_time = time.time()
    train_output = train(
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        model=model,
        cfg=train_cfg,
    )
    total_train_time = time.time() - start_time
    state, train_losses, train_bce_losses, val_losses, val_bce_losses = train_output

    # Output directory
    run_dir = Path(HydraConfig.get().run.dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config in output directory
    (run_dir / "train.yaml").write_text(OmegaConf.to_yaml(cfg))

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
