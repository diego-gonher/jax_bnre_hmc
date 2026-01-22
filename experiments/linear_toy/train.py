from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from jax_bnre_hmc.train import TrainConfig, train
from jax_bnre_hmc.data import make_joint_and_marginal


def simulate_linear_dataset(
    key: jax.Array,
    n: int,
    n_points: int,
    sigma: float,
    m_low: float,
    m_high: float,
    b_low: float,
    b_high: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulator:
      y = m * x + b + Normal(0, sigma)
    where x_grid = [0, 1, ..., n_points-1]
    Observation x_o is the full y vector (length n_points)
    and theta = (m, b)

    Returns:
      theta: (n, 2)
      x:     (n, n_points)
    """
    key_m, key_b, key_noise = jax.random.split(key, 3)

    m = jax.random.uniform(key_m, (n,), minval=m_low, maxval=m_high)
    b = jax.random.uniform(key_b, (n,), minval=b_low, maxval=b_high)
    theta = jnp.stack([m, b], axis=-1)  # (n, 2)

    x_grid = jnp.arange(n_points, dtype=jnp.float32)  # (n_points,)
    y_clean = m[:, None] * x_grid[None, :] + b[:, None]  # (n, n_points)

    noise = sigma * jax.random.normal(key_noise, (n, n_points))
    y_noisy = y_clean + noise

    return theta.astype(jnp.float32), y_noisy.astype(jnp.float32)


@hydra.main(config_path="../../configs/linear_toy", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Set the seed
    key = jax.random.PRNGKey(int(cfg.seed))

    # Simulate the full dataset
    theta, x = simulate_linear_dataset(
        key=key,
        n=int(cfg.data.n_simulations),
        n_points=int(cfg.data.n_points),
        sigma=float(cfg.data.sigma),
        m_low=float(cfg.prior.m_low),
        m_high=float(cfg.prior.m_high),
        b_low=float(cfg.prior.b_low),
        b_high=float(cfg.prior.b_high),
    )

    # Split the dataset into train and validation sets
    theta_train, theta_val, x_train, x_val = train_test_split(theta, x, test_size=float(cfg.data.validation_fraction), random_state=int(cfg.seed))


    train_cfg = TrainConfig(
        seed=int(cfg.seed),
        lr=float(cfg.train.lr),
        epochs=int(cfg.train.epochs),
        print_every=int(cfg.train.print_every),
        batch_size=int(cfg.train.batch_size),
    )

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


if __name__ == "__main__":
    main()
