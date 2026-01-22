from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .data import make_batches, make_joint_and_marginal
from .loss import nre_loss_bce_style_from_logits, nre_loss_from_logits
from .model import RatioEstimatorMLP


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    lr: float = 1e-3
    epochs: int = 2000
    batch_size: int = 1024
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
    One gradient step on a batch of (theta, x) data.
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


@jax.jit
def validation_step(
    state: train_state.TrainState,
    theta: jnp.ndarray,
    x: jnp.ndarray,
    rng: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute validation losses without gradients.
    Deterministic given rng.
    """
    rng_shuffle, _ = jax.random.split(rng)
    joint, marginal = make_joint_and_marginal(rng_shuffle, theta, x)

    logits_joint = state.apply_fn(state.params, joint.theta, joint.x)
    logits_marg = state.apply_fn(state.params, marginal.theta, marginal.x)

    loss = nre_loss_from_logits(logits_joint, logits_marg)
    bce_loss = nre_loss_bce_style_from_logits(logits_joint, logits_marg)
    return loss, bce_loss


def train(
    theta_train: jnp.ndarray,
    x_train: jnp.ndarray,
    theta_val: jnp.ndarray,
    x_val: jnp.ndarray,
    model_hidden_dims: Sequence[int],
    model_activation: str = "tanh",
    model_norm: str = "layernorm",
    cfg: TrainConfig = TrainConfig(),
) -> tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fixed-epoch training loop with mini-batching.
    Each epoch processes the dataset in batches with shuffling.
    Drops remainder examples to keep batch shapes static.
    
    Returns:
        state: Final training state
        train_losses: Training losses per epoch (averaged over batches)
        train_bce_losses: Training BCE losses per epoch (averaged over batches)
        val_losses: Validation losses per epoch (averaged over batches)
        val_bce_losses: Validation BCE losses per epoch (averaged over batches)
    """
    theta_train = jnp.asarray(theta_train, dtype=jnp.float32)
    x_train = jnp.asarray(x_train, dtype=jnp.float32)
    theta_val = jnp.asarray(theta_val, dtype=jnp.float32)
    x_val = jnp.asarray(x_val, dtype=jnp.float32)

    rng = jax.random.PRNGKey(cfg.seed)
    rng_init, rng_train, rng_val = jax.random.split(rng, 3)

    _, state = create_train_state(
        rng=rng_init,
        theta_dim=theta_train.shape[1],
        x_dim=x_train.shape[1],
        hidden_dims=model_hidden_dims,
        activation=model_activation,
        norm=model_norm,
        lr=cfg.lr,
    )

    train_losses = []
    train_bce_losses = []
    val_losses = []
    val_bce_losses = []
    
    for epoch in range(cfg.epochs):
        # Training: iterate over batches
        rng_train, rng_epoch = jax.random.split(rng_train)
        epoch_train_losses = []
        epoch_train_bce_losses = []
        
        for theta_batch, x_batch in make_batches(rng_epoch, theta_train, x_train, cfg.batch_size):
            rng_train, rng_step = jax.random.split(rng_train)
            state, batch_loss, batch_bce_loss = train_step(state, theta_batch, x_batch, rng_step)
            epoch_train_losses.append(batch_loss)
            epoch_train_bce_losses.append(batch_bce_loss)
        
        # Average losses over batches for this epoch
        train_loss = jnp.mean(jnp.stack(epoch_train_losses))
        train_bce_loss = jnp.mean(jnp.stack(epoch_train_bce_losses))
        train_losses.append(train_loss)
        train_bce_losses.append(train_bce_loss)

        # Validation step (no mini-batching)
        rng_val, rng_val_step = jax.random.split(rng_val)
        val_loss, val_bce_loss = validation_step(state, theta_val, x_val, rng_val_step)
        val_losses.append(val_loss)
        val_bce_losses.append(val_bce_loss)

        if (epoch + 1) % cfg.print_every == 0:
            print(
                f"epoch {epoch+1:5d} | "
                f"train_loss {float(train_loss):.6f} | train_bce {float(train_bce_loss):.6f} | "
                f"val_loss {float(val_loss):.6f} | val_bce {float(val_bce_loss):.6f}"
            )

    return (
        state,
        jnp.stack(train_losses),
        jnp.stack(train_bce_losses),
        jnp.stack(val_losses),
        jnp.stack(val_bce_losses),
    )
