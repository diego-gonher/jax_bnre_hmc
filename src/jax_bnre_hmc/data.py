from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Batch:
    theta: jnp.ndarray  # (B, theta_dim)
    x: jnp.ndarray      # (B, x_dim)


def _derangement(key: jax.Array, n: int) -> jnp.ndarray:
    """
    Sample a random permutation of {0,...,n-1} with no fixed points (a derangement).

    This avoids accidentally creating "marginal" pairs that are actually joint pairs
    when perm[i] == i.
    """
    idx = jnp.arange(n)

    def cond(state):
        # Keep looping while there exists any fixed point
        _, perm = state
        return jnp.any(perm == idx)

    def body(state):
        key, _ = state
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, n)
        return (key, perm)

    # Initial proposal
    key, sub = jax.random.split(key)
    perm0 = jax.random.permutation(sub, n)

    # Resample until no fixed points
    _, perm = jax.lax.while_loop(cond, body, (key, perm0))
    return perm


def make_joint_and_marginal(
    key: jax.Array,
    theta: jnp.ndarray,
    x: jnp.ndarray,
) -> tuple[Batch, Batch]:
    """
    Given paired samples (theta, x) from the joint, create:
    - joint batch: (theta, x)
    - marginal batch: (theta, x_shuffled) where x_shuffled breaks pairing

    Uses a derangement permutation to avoid fixed points (perm[i] == i), which would
    otherwise create mislabeled negatives.
    """
    n = theta.shape[0]
    perm = _derangement(key, n)
    x_marg = x[perm]
    return Batch(theta=theta, x=x), Batch(theta=theta, x=x_marg)


def make_batches(
    rng: jax.Array,
    theta: jnp.ndarray,
    x: jnp.ndarray,
    batch_size: int,
):
    """
    Shuffles data and yields mini-batches.
    Drops remainder to keep shapes static.
    
    Args:
        rng: Random key for shuffling
        theta: Parameter array of shape (n, theta_dim)
        x: Observation array of shape (n, x_dim)
        batch_size: Size of each batch
        
    Yields:
        (theta_batch, x_batch): Batches of shape (batch_size, ...)
    """
    n = theta.shape[0]
    assert x.shape[0] == n

    perm = jax.random.permutation(rng, n)

    n_batches = n // batch_size
    for i in range(n_batches):
        idx = perm[i * batch_size : (i + 1) * batch_size]
        yield theta[idx], x[idx]
