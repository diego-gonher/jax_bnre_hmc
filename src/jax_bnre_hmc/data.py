from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Batch:
    theta: jnp.ndarray  # (B, theta_dim)
    x: jnp.ndarray      # (B, x_dim)


def make_joint_and_marginal(
    key: jax.Array,
    theta: jnp.ndarray,
    x: jnp.ndarray,
) -> tuple[Batch, Batch]:
    """
    Given paired samples (theta, x) from the joint, create:
    - joint batch: (theta, x)
    - marginal batch: (theta, x_shuffled) where x_shuffled breaks pairing
    """
    n = theta.shape[0]
    perm = jax.random.permutation(key, n)
    x_marg = x[perm]
    return Batch(theta=theta, x=x), Batch(theta=theta, x=x_marg)
