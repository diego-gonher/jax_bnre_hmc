from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Batch:
    """A batch of parameter-observation pairs.
    
    Attributes:
        theta: Parameter batch of shape (B, theta_dim).
        x: Observation batch of shape (B, x_dim).
    """
    theta: jnp.ndarray  # (B, theta_dim)
    x: jnp.ndarray      # (B, x_dim)


def _derangement(key: jax.Array, n: int) -> jnp.ndarray:
    """Sample a random derangement (permutation with no fixed points).
    
    Generates a random permutation of {0, ..., n-1} such that perm[i] != i for all i.
    This is used to ensure that when creating marginal pairs by shuffling, we don't
    accidentally create pairs that are actually joint pairs (which would be mislabeled).
    
    Args:
        key: Random key for permutation.
        n: Size of the set to permute.
    
    Returns:
        Array of shape (n,) containing a derangement permutation.
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
    """Create joint and marginal batches from paired samples.
    
    Given paired samples (theta, x) from the joint distribution, creates:
    - A joint batch: (theta, x) - the original paired samples
    - A marginal batch: (theta, x_shuffled) - where x_shuffled breaks the pairing
    
    Uses a derangement permutation to ensure that no element is mapped to itself,
    avoiding the creation of mislabeled negative examples.
    
    Args:
        key: Random key for shuffling.
        theta: Parameter samples of shape (n, theta_dim).
        x: Observation samples of shape (n, x_dim).
    
    Returns:
        A tuple containing:
            - joint: Batch of joint (theta, x) pairs.
            - marginal: Batch of marginal (theta, x_shuffled) pairs.
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
    """Shuffle data and yield mini-batches.
    
    Randomly shuffles the data and yields batches of the specified size.
    Drops remainder examples to keep batch shapes static, which is important
    for JIT compilation efficiency.
    
    Args:
        rng: Random key for shuffling.
        theta: Parameter array of shape (n, theta_dim).
        x: Observation array of shape (n, x_dim).
        batch_size: Size of each batch.
    
    Yields:
        A tuple (theta_batch, x_batch) where each batch has shape (batch_size, ...).
        The number of batches yielded is floor(n / batch_size).
    """
    n = theta.shape[0]
    assert x.shape[0] == n

    perm = jax.random.permutation(rng, n)

    n_batches = n // batch_size
    for i in range(n_batches):
        idx = perm[i * batch_size : (i + 1) * batch_size]
        yield theta[idx], x[idx]
