from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .checkpointing import ensure_dirs, get_run_dir, save_best, save_latest
from .data import make_batches, make_joint_and_marginal
from .loss import bnre_balance_from_logits, nre_loss_bce_style_from_logits, nre_loss_from_logits
from .model import RatioEstimatorMLP


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training a Neural Ratio Estimator.
    
    Attributes:
        seed: Random seed for reproducibility.
        lr: Learning rate for the optimizer.
        epochs: Number of training epochs.
        batch_size: Size of each training batch. Remainder examples are dropped.
        clip_max_norm: Maximum gradient norm for clipping. If None, no clipping is applied.
        print_every: Print training metrics every N epochs.
        save_every: Save latest checkpoint every N epochs. If 0, latest checkpoint saving is disabled.
        checkpoint_dirname: Directory name for storing checkpoints.
        bnre_lambda: Weight for the BNRE balance penalty. Set to 0.0 for standard NRE training.
    """
    seed: int = 0
    lr: float = 1e-3
    epochs: int = 2000
    batch_size: int = 1024
    clip_max_norm: float | None = 5.0 
    print_every: int = 200
    save_every: int = 200
    checkpoint_dirname: str = "checkpoints"
    bnre_lambda: float = 10.0


def create_train_state(
    rng: jax.Array,
    theta_dim: int,
    x_dim: int,
    hidden_dims: Sequence[int],
    activation: str,
    norm: str,
    lr: float,
    clip_max_norm: float | None,
) -> tuple[RatioEstimatorMLP, train_state.TrainState]:
    """Initialize model and training state with optimizer.
    
    Args:
        rng: Random key for parameter initialization.
        theta_dim: Dimensionality of the parameter space.
        x_dim: Dimensionality of the observation space.
        hidden_dims: Sequence of hidden layer dimensions for the MLP.
        activation: Activation function name (e.g., "tanh", "relu").
        norm: Normalization type ("layernorm" or "none").
        lr: Learning rate for the Adam optimizer.
        clip_max_norm: Maximum gradient norm for clipping. If None, no clipping.
    
    Returns:
        A tuple containing:
            - model: The initialized RatioEstimatorMLP model.
            - state: Flax TrainState with initialized parameters and optimizer.
    """
    hidden_dims = tuple(int(d) for d in hidden_dims)  # Ensure tuple of ints
    model = RatioEstimatorMLP(hidden_dims=hidden_dims, activation=activation, norm=norm)

    # Create dummy data to initialize parameters
    dummy_theta = jnp.zeros((1, theta_dim), dtype=jnp.float32)
    dummy_x = jnp.zeros((1, x_dim), dtype=jnp.float32)
    params = model.init(rng, dummy_theta, dummy_x)

    # Create optimizer
    if clip_max_norm is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_max_norm),
            optax.adam(lr),
        )
    else:
        tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


@jax.jit
def train_step(
    state: train_state.TrainState,
    theta: jnp.ndarray,
    x: jnp.ndarray,
    rng: jax.Array,
    bnre_lambda: float,
) -> tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform one gradient step on a batch of training data.
    
    Computes the loss (NRE + optional BNRE penalty), computes gradients, and updates
    the model parameters. The function is JIT-compiled for efficiency.
    
    Args:
        state: Current training state containing model parameters and optimizer.
        theta: Parameter batch of shape (batch_size, theta_dim).
        x: Observation batch of shape (batch_size, x_dim).
        rng: Random key for shuffling joint/marginal pairs.
        bnre_lambda: Weight for BNRE balance penalty. Set to 0.0 for standard NRE.
    
    Returns:
        A tuple containing:
            - state: Updated training state after gradient step.
            - total_loss: Total loss value (NRE + bnre_lambda * penalty).
            - bce_loss: BCE-style loss metric (for logging).
            - penalty: BNRE balance penalty value.
            - balance: BNRE balance value (mean(sigmoid(joint)) + mean(sigmoid(marginal))).
    """
    rng_shuffle, _ = jax.random.split(rng)
    joint, marginal = make_joint_and_marginal(rng_shuffle, theta, x)

    def loss_and_metric(params):
        logits_joint = state.apply_fn(params, joint.theta, joint.x)
        logits_marg = state.apply_fn(params, marginal.theta, marginal.x)

        nre_loss = nre_loss_from_logits(logits_joint, logits_marg)
        bce_loss = nre_loss_bce_style_from_logits(logits_joint, logits_marg)
        penalty, balance = bnre_balance_from_logits(logits_joint, logits_marg)
        
        # Always compute total_loss = nre_loss + bnre_lambda * penalty
        # When bnre_lambda == 0.0, this equals nre_loss exactly
        total_loss = nre_loss + bnre_lambda * penalty
        
        return total_loss, (bce_loss, penalty, balance)

    (total_loss, (bce_loss, penalty, balance)), grads = jax.value_and_grad(loss_and_metric, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, total_loss, bce_loss, penalty, balance


@jax.jit
def validation_step(
    state: train_state.TrainState,
    theta: jnp.ndarray,
    x: jnp.ndarray,
    rng: jax.Array,
    bnre_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute validation losses without computing gradients.
    
    Evaluates the model on validation data and computes all loss metrics.
    The function is JIT-compiled for efficiency.
    
    Args:
        state: Current training state containing model parameters.
        theta: Parameter batch of shape (batch_size, theta_dim).
        x: Observation batch of shape (batch_size, x_dim).
        rng: Random key for shuffling joint/marginal pairs.
        bnre_lambda: Weight for BNRE balance penalty. Set to 0.0 for standard NRE.
    
    Returns:
        A tuple containing:
            - total_loss: Total loss value (NRE + bnre_lambda * penalty).
            - bce_loss: BCE-style loss metric (for logging).
            - penalty: BNRE balance penalty value.
            - balance: BNRE balance value (mean(sigmoid(joint)) + mean(sigmoid(marginal))).
    """
    rng_shuffle, _ = jax.random.split(rng)
    joint, marginal = make_joint_and_marginal(rng_shuffle, theta, x)

    logits_joint = state.apply_fn(state.params, joint.theta, joint.x)
    logits_marg = state.apply_fn(state.params, marginal.theta, marginal.x)

    nre_loss = nre_loss_from_logits(logits_joint, logits_marg)
    bce_loss = nre_loss_bce_style_from_logits(logits_joint, logits_marg)
    penalty, balance = bnre_balance_from_logits(logits_joint, logits_marg)
    
    # Always compute total_loss = nre_loss + bnre_lambda * penalty
    # When bnre_lambda == 0.0, this equals nre_loss exactly
    total_loss = nre_loss + bnre_lambda * penalty
    
    return total_loss, bce_loss, penalty, balance


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
    """Train a Neural Ratio Estimator with mini-batching and checkpointing.
    
    Implements a fixed-epoch training loop with mini-batching. Each epoch processes
    the dataset in batches with shuffling. Remainder examples are dropped to keep
    batch shapes static for JIT compilation efficiency. Supports both NRE and BNRE
    training based on the bnre_lambda configuration.
    
    Args:
        theta_train: Training parameter samples of shape (n_train, theta_dim).
        x_train: Training observation samples of shape (n_train, x_dim).
        theta_val: Validation parameter samples of shape (n_val, theta_dim).
        x_val: Validation observation samples of shape (n_val, x_dim).
        model_hidden_dims: Sequence of hidden layer dimensions for the MLP.
        model_activation: Activation function name (default: "tanh").
        model_norm: Normalization type, "layernorm" or "none" (default: "layernorm").
        cfg: Training configuration (learning rate, epochs, batch size, etc.).
    
    Returns:
        A tuple containing:
            - state: Final training state with trained model parameters.
            - train_losses: Training losses per epoch (averaged over batches).
            - train_bce_losses: Training BCE-style losses per epoch (averaged over batches).
            - val_losses: Validation losses per epoch.
            - val_bce_losses: Validation BCE-style losses per epoch.
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
        clip_max_norm=cfg.clip_max_norm,
    )

    train_losses = []
    train_bce_losses = []
    val_losses = []
    val_bce_losses = []
    
    # Initialize checkpointing
    best_val_loss = float("inf")
    run_dir = get_run_dir()
    latest_dir, best_dir = ensure_dirs(run_dir, cfg.checkpoint_dirname)
    latest_meta_path = run_dir / cfg.checkpoint_dirname / "latest_meta.json"
    best_meta_path = run_dir / cfg.checkpoint_dirname / "best_meta.json"
    
    for epoch in range(cfg.epochs):
        # Training: iterate over batches
        rng_train, rng_epoch = jax.random.split(rng_train)
        epoch_train_losses = []
        epoch_train_bce_losses = []
        epoch_train_penalties = []
        epoch_train_balances = []
        
        for theta_batch, x_batch in make_batches(rng_epoch, theta_train, x_train, cfg.batch_size):
            rng_train, rng_step = jax.random.split(rng_train)
            state, batch_loss, batch_bce_loss, batch_penalty, batch_balance = train_step(
                state, theta_batch, x_batch, rng_step, cfg.bnre_lambda
            )
            epoch_train_losses.append(batch_loss)
            epoch_train_bce_losses.append(batch_bce_loss)
            epoch_train_penalties.append(batch_penalty)
            epoch_train_balances.append(batch_balance)
        
        # Average losses over batches for this epoch
        train_loss = jnp.mean(jnp.stack(epoch_train_losses))
        train_bce_loss = jnp.mean(jnp.stack(epoch_train_bce_losses))
        train_penalty = jnp.mean(jnp.stack(epoch_train_penalties))
        train_balance = jnp.mean(jnp.stack(epoch_train_balances))
        train_losses.append(train_loss)
        train_bce_losses.append(train_bce_loss)

        # Validation step (no mini-batching)
        rng_val, rng_val_step = jax.random.split(rng_val)
        val_loss, val_bce_loss, val_penalty, val_balance = validation_step(
            state, theta_val, x_val, rng_val_step, cfg.bnre_lambda
        )
        # val_key_fixed = jax.random.PRNGKey(cfg.seed + 117)  # this is to verify the best model
        # val_loss, val_bce_loss, val_penalty, val_balance = validation_step(state, theta_val, x_val, val_key_fixed, cfg.bnre_lambda)  # this is to verify the best model
        val_losses.append(val_loss)
        val_bce_losses.append(val_bce_loss)
        
        val_loss_float = float(val_loss)
        
        # Checkpointing: save latest
        if cfg.save_every and cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            save_latest(state, latest_dir, latest_meta_path, epoch + 1, val_loss_float)
        
        # Checkpointing: save best
        if val_loss_float < best_val_loss:
            best_val_loss = val_loss_float
            save_best(state.params, best_dir, best_meta_path, epoch + 1, val_loss_float)

        if (epoch + 1) % cfg.print_every == 0:
            if cfg.bnre_lambda > 0.0:
                print(
                    f"epoch {epoch+1:5d} | "
                    f"train_loss {float(train_loss):.6f} | train_bce {float(train_bce_loss):.6f} | "
                    f"val_loss {val_loss_float:.6f} | val_bce {float(val_bce_loss):.6f} | "
                    f"balance {float(train_balance):.6f} | penalty {float(train_penalty):.6e}"
                )
            else:
                print(
                    f"epoch {epoch+1:5d} | "
                    f"train_loss {float(train_loss):.6f} | train_bce {float(train_bce_loss):.6f} | "
                    f"val_loss {val_loss_float:.6f} | val_bce {float(val_bce_loss):.6f}"
                )

    return (
        state,
        jnp.stack(train_losses),
        jnp.stack(train_bce_losses),
        jnp.stack(val_losses),
        jnp.stack(val_bce_losses),
    )
