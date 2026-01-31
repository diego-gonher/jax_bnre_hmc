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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import h5py

from jax_bnre_hmc.train import TrainConfig, train
from jax_bnre_hmc.data import make_joint_and_marginal
from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.loss import nre_loss_bce_style_from_logits, nre_loss_from_logits


@hydra.main(config_path="../../configs/2param", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Set the seed
    key = jax.random.PRNGKey(int(cfg.seed))

    # Load the dataset and preprocess it
    dataset_file = f'/Users/diegogonzalez/Desktop/models/imported_models/twoparam_mocks_z5.50.hdf5'

    print(f'\nLoading dataset from {dataset_file}')

    with h5py.File(dataset_file, 'r') as f:
        # load the parameters
        theta = f['theta'][:]
        print(f'\ntheta shape: {theta.shape}')
        # load the mocks and the velocity bins
        x = f['x'][:]
        print(f'x shape: {x.shape}')
        f.close()

        # use min max scalers on both
        # create min max scalers
        theta_scaler = MinMaxScaler()
        x_scaler = MinMaxScaler()

        # fit and transform the parameters and mocks
        theta_scaled = theta_scaler.fit_transform(theta)
        x_scaled = x_scaler.fit_transform(x) 
        print(f'\nscaled theta shape: {theta_scaled.shape}')
        print(f'scaled x shape: {x_scaled.shape}')

        # reshape the dataset
        dataset_params_scaled_reshaped = theta_scaled.reshape(558, 1000, 2)
        dataset_mocks_scaled_reshaped = x_scaled.reshape(558, 1000, 22)

        # choose only 
        no_mocks_per_parameter_pair = 500

        # split the dataset into a training and test set
        theta_scaled, _1, x_scaled, _2 = train_test_split(dataset_params_scaled_reshaped, 
                                                          dataset_mocks_scaled_reshaped, 
                                                          test_size=int(558-no_mocks_per_parameter_pair), 
                                                          random_state=42)
        
        # only keep the first mocks_per_model mocks for each model, and reshape the dataset to 2D
        theta_scaled = theta_scaled[:, :no_mocks_per_parameter_pair, :].reshape(-1, 2)
        x_scaled = x_scaled[:, :no_mocks_per_parameter_pair, :].reshape(-1, 22)

        # print a quick message and the shapes of the training set
        print(f'\nPREPARING DATASET FOR TRAINING\n')
        print(f'Using all parameter pairs and {no_mocks_per_parameter_pair} mocks per model.')
        print(f' - Shape of the full dataset: {theta_scaled.shape}, {x_scaled.shape}')
        

    # Split the dataset into train and validation sets, it must be the scaled versions
    print("\nSplitting dataset into train and validation sets...")
    theta_train, theta_val, x_train, x_val = train_test_split(theta_scaled, x_scaled, 
                                                              test_size=float(cfg.data.validation_fraction), 
                                                              random_state=int(cfg.seed))

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

    train_output = train(
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        model_hidden_dims=tuple(cfg.model.hidden_dims),
        model_activation=str(cfg.model.activation),
        model_norm=str(cfg.model.norm),
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
    joint, marginal = make_joint_and_marginal(key2, theta, x)
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

    # Load best params and verify validation loss
    best_dir = run_dir / cfg.train.checkpoint_dirname / "best"
    best_meta_path = run_dir / cfg.train.checkpoint_dirname / "best_meta.json"
    
    if best_dir.exists() and best_meta_path.exists():
        # Load best params
        best_params = load_best_params(best_dir)
        
        # Read expected best validation loss from metadata
        best_meta = json.loads(best_meta_path.read_text())
        expected_best_val_loss = best_meta["val_loss"]
        
        # Recompute validation loss using best params
        key_val = jax.random.PRNGKey(int(cfg.seed) + 117)  # Use different key for verification
        joint_val, marginal_val = make_joint_and_marginal(key_val, theta_val, x_val)
        logits_joint_val = state.apply_fn(best_params, joint_val.theta, joint_val.x)
        logits_marg_val = state.apply_fn(best_params, marginal_val.theta, marginal_val.x)
        recomputed_val_loss = float(nre_loss_from_logits(logits_joint_val, logits_marg_val))
        
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
