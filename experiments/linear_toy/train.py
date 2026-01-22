from __future__ import annotations

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from jax_bnre_hmc.train import TrainConfig, train


def simulate_linear_dataset(
    key: jax.Array,
    n: int,
    n_points: int,
    sigma: float,
    m_low: float,
    m_high: float,
    b_low: float,
    b_high: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    key = jax.random.PRNGKey(int(cfg.seed))

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

    train_cfg = TrainConfig(
        seed=int(cfg.seed),
        lr=float(cfg.train.lr),
        epochs=int(cfg.train.epochs),
        print_every=int(cfg.train.print_every),
    )

    state, losses = train(
        theta=theta,
        x=x,
        model_hidden_dims=tuple(cfg.model.hidden_dims),
        model_activation=str(cfg.model.activation),
        cfg=train_cfg,
    )

    # Basic sanity prints
    print("done. final loss:", float(losses[-1]))
    # Evaluate mean logit on joint vs marginal for a quick sanity check
    # (higher on joint is a good sign)
    from jax_bnre_hmc.data import make_joint_and_marginal

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

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="loss")
    plt.legend()
    plt.show()

    # Plot the sigmoid of the logits
    plt.figure(figsize=(10, 5))
    plt.plot(pj, label="joint")
    plt.plot(pm, label="marginal")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
