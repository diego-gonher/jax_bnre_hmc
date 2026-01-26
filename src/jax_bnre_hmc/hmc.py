# src/jax_bnre_hmc/hmc.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS


Array = jax.Array


@dataclass(frozen=True)
class BoxPrior:
    low: jnp.ndarray   # (D,)
    high: jnp.ndarray  # (D,)

    def __post_init__(self):
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have same shape")
        if not jnp.all(self.high > self.low):
            raise ValueError("All high must be > low")


def z_to_theta(z: Array, prior: BoxPrior) -> tuple[Array, Array]:
    """Unconstrained -> box via sigmoid."""
    u = jax.nn.sigmoid(z)                       # (D,) in (0,1)
    theta = prior.low + (prior.high - prior.low) * u
    return theta, u


def logabsdet_dtheta_dz(u: Array, prior: BoxPrior, eps: float = 1e-12) -> Array:
    """log |det dtheta/dz| for theta = low + (high-low)*sigmoid(z).

    Uses u = sigmoid(z). Adds eps for numerical safety.
    """
    # dtheta/dz = (high-low) * u * (1-u)
    return jnp.sum(
        jnp.log(prior.high - prior.low)
        + jnp.log(jnp.clip(u, eps, 1.0))
        + jnp.log(jnp.clip(1.0 - u, eps, 1.0))
    )


def make_log_ratio_fn(
    apply_fn: Callable,
    params,
    x_obs: Array,
) -> Callable[[Array], Array]:
    """Return a function log_ratio(theta) -> scalar.

    apply_fn(params, theta_batch, x_batch) should return logits (B,).
    """
    x_obs = jnp.asarray(x_obs, dtype=jnp.float32)

    def log_ratio(theta: Array) -> Array:
        theta = jnp.asarray(theta, dtype=jnp.float32)
        # shape to (1, D) and (1, X)
        logits = apply_fn(params, theta[None, :], x_obs[None, ...])
        return jnp.squeeze(logits, axis=0)  # scalar

    return log_ratio


def make_potential_fn(
    log_ratio_fn: Callable[[Array], Array],
    prior: BoxPrior,
    x_obs: Array | None = None,   # kept for signature symmetry; not used if baked into log_ratio_fn
) -> Callable[[Array], Array]:
    """Return a potential_fn(z) suitable for NUTS(potential_fn=...)."""

    def potential(z: Array) -> Array:
        theta, u = z_to_theta(z, prior)
        ladj = logabsdet_dtheta_dz(u, prior)
        lr = log_ratio_fn(theta)
        return -(lr + ladj)

    return potential


def run_nuts(
    potential_fn: Callable[[Array], Array],
    rng_key: Array,
    init_z: Array,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    **nuts_kwargs,
):
    """Run NUTS with a custom potential_fn."""
    kernel = NUTS(potential_fn=potential_fn, **nuts_kwargs)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key, init_params=init_z)
    return mcmc
