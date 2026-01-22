from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .data import make_joint_and_marginal
from .loss import nre_loss_bce_style_from_logits, nre_loss_from_logits
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
    norm: str,
    lr: float,
) -> tuple[RatioEstimatorMLP, train_state.TrainState]:
    """
    Initialize model + TrainState (params + optimizer).
    Returns the model (to use its apply_fn) and the TrainState.
    """
    hidden_dims = tuple(int(d) for d in hidden_dims)  # Ensure tuple of ints
    model = RatioEstimatorMLP(hidden_dims=hidden_dims, activation=activation, norm=norm)

    # Create dummy data to initialize parameters
    dummy_theta = jnp.zeros((1, theta_dim), dtype=jnp.float32)
    dummy_x = jnp.zeros((1, x_dim), dtype=jnp.float32)
    params = model.init(rng, dummy_theta, dummy_x)

    # Create optimizer
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


@jax.jit
def train_step(
    state: train_state.TrainState,
    theta: jnp.ndarray,
    x: jnp.ndarray,
    rng: jax.Array,
) -> tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
    """
    One gradient step on the full (theta, x) dataset (v0: no minibatching).
    Deterministic given rng.
    """
    rng_shuffle, _ = jax.random.split(rng)
    joint, marginal = make_joint_and_marginal(rng_shuffle, theta, x)

    def loss_and_metric(params):
        logits_joint = state.apply_fn(params, joint.theta, joint.x)
        logits_marg = state.apply_fn(params, marginal.theta, marginal.x)

        loss = nre_loss_from_logits(logits_joint, logits_marg)
        bce_loss = nre_loss_bce_style_from_logits(logits_joint, logits_marg)
        return loss, bce_loss

    (loss, bce_loss), grads = jax.value_and_grad(loss_and_metric, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, bce_loss


def train(
    theta: jnp.ndarray,
    x: jnp.ndarray,
    model_hidden_dims: Sequence[int],
    model_activation: str = "tanh",
    model_norm: str = "layernorm",
    cfg: TrainConfig = TrainConfig(),
) -> tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
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
        norm=model_norm,
        lr=cfg.lr,
    )

    losses = []
    bce_losses = []
    for epoch in range(cfg.epochs):
        rng_loop, rng_step = jax.random.split(rng_loop)
        state, loss, bce_loss = train_step(state, theta, x, rng_step)
        losses.append(loss)
        bce_losses.append(bce_loss)

        if (epoch + 1) % cfg.print_every == 0:
            print(f"epoch {epoch+1:5d} | loss {float(loss):.6f} | bce {float(bce_loss):.6f}")

    return state, jnp.stack(losses), jnp.stack(bce_losses)
