from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .data import make_joint_and_marginal
from .loss import nre_loss
from .model import RatioEstimatorMLP


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    lr: float = 1e-3
    epochs: int = 2000
    print_every: int = 200


def create_train_state(
    rng: jax.Array,
    theta_dim: int,
    x_dim: int,
    hidden_dims: Sequence[int],
    activation: str,
    lr: float,
) -> tuple[RatioEstimatorMLP, train_state.TrainState]:
    """
    Initialize model + TrainState (params + optimizer).
    Returns the model (to use its apply_fn) and the TrainState.
    """
    model = RatioEstimatorMLP(hidden_dims=tuple(hidden_dims), activation=activation)

    dummy_theta = jnp.zeros((1, theta_dim), dtype=jnp.float32)
    dummy_x = jnp.zeros((1, x_dim), dtype=jnp.float32)
    params = model.init(rng, dummy_theta, dummy_x)

    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


@jax.jit
def train_step(
    state: train_state.TrainState,
    theta: jnp.ndarray,
    x: jnp.ndarray,
    rng: jax.Array,
) -> tuple[train_state.TrainState, jnp.ndarray]:
    """
    One gradient step on the full (theta, x) dataset (v0: no minibatching).
    Deterministic given rng.
    """
    rng_shuffle, _ = jax.random.split(rng)
    joint, marginal = make_joint_and_marginal(rng_shuffle, theta, x)

    def loss_fn(params):
        return nre_loss(state.apply_fn, params, joint, marginal)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train(
    theta: jnp.ndarray,
    x: jnp.ndarray,
    model_hidden_dims: Sequence[int] = (50, 50, 50),
    model_activation: str = "tanh",
    cfg: TrainConfig = TrainConfig(),
) -> tuple[train_state.TrainState, jnp.ndarray]:
    """
    Fixed-epoch training loop (v0).
    Uses full-batch training: each epoch uses the entire dataset.
    """
    theta = jnp.asarray(theta, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.float32)

    rng = jax.random.PRNGKey(cfg.seed)
    rng_init, rng_loop = jax.random.split(rng)

    _, state = create_train_state(
        rng=rng_init,
        theta_dim=theta.shape[1],
        x_dim=x.shape[1],
        hidden_dims=model_hidden_dims,
        activation=model_activation,
        lr=cfg.lr,
    )

    losses = []
    for epoch in range(cfg.epochs):
        rng_loop, rng_step = jax.random.split(rng_loop)
        state, loss = train_step(state, theta, x, rng_step)
        losses.append(loss)

        if (epoch + 1) % cfg.print_every == 0:
            print(f"epoch {epoch+1:5d} | loss {float(loss):.6f}")

    return state, jnp.stack(losses)
